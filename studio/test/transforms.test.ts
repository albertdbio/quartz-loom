import { describe, expect, it } from "vitest"
import {
  GOALS,
  TRANSFORMS,
  personalLine,
  loadProfile,
  rankTransforms,
  saveProfile,
  starterTransformIndex,
  type Goal,
} from "../src/lib/transforms"

describe("transform deck", () => {
  it("has a unique name for every transform", () => {
    const names = TRANSFORMS.map((s) => s.name)
    expect(new Set(names).size).toBe(names.length)
  })
})

describe("rankTransforms", () => {
  it("keeps every transform exactly once for every goal (no drops, no dupes)", () => {
    for (const g of GOALS) {
      const ranked = rankTransforms(TRANSFORMS, g.key)
      expect(ranked).toHaveLength(TRANSFORMS.length)
      expect(new Set(ranked.map((s) => s.name)).size).toBe(TRANSFORMS.length)
    }
  })

  it("leads with the goal's signature transform", () => {
    expect(rankTransforms(TRANSFORMS, "viral")[0]!.name).toBe("8-bit")
    expect(rankTransforms(TRANSFORMS, "product")[0]!.name).toBe("Midas")
    expect(rankTransforms(TRANSFORMS, "art")[0]!.name).toBe("Sketch")
    expect(rankTransforms(TRANSFORMS, "teach")[0]!.name).toBe("Sketch")
  })

  it("leaves the deck untouched for the undecided", () => {
    expect(rankTransforms(TRANSFORMS, "explore").map((s) => s.name)).toEqual(TRANSFORMS.map((s) => s.name))
  })

  it("is pure — the source deck is never mutated", () => {
    const before = TRANSFORMS.map((s) => s.name)
    rankTransforms(TRANSFORMS, "viral")
    expect(TRANSFORMS.map((s) => s.name)).toEqual(before)
  })
})

describe("starterTransformIndex", () => {
  it("points at the lead transform's position in the RANKED deck (always 0)", () => {
    for (const g of GOALS) {
      const ranked = rankTransforms(TRANSFORMS, g.key)
      const i = starterTransformIndex(ranked, g.key)
      expect(i).toBe(0)
      expect(ranked[i]).toBeDefined()
    }
  })
})

describe("personalLine", () => {
  it("returns a non-empty line naming the starter transform for every combination", () => {
    const crafts = ["creator", "artist", "marketer", "educator", "fun"] as const
    for (const craft of crafts) {
      for (const g of GOALS) {
        const ranked = rankTransforms(TRANSFORMS, g.key)
        const line = personalLine(craft, g.key, ranked[0]!.name)
        expect(line.length).toBeGreaterThan(10)
        expect(line).toContain(ranked[0]!.name)
      }
    }
  })

  it("differs by goal so the answer visibly mattered", () => {
    const lines = GOALS.map((g) => personalLine("creator", g.key, rankTransforms(TRANSFORMS, g.key)[0]!.name))
    expect(new Set(lines).size).toBe(GOALS.length)
  })
})

describe("goal catalogue", () => {
  it("every goal has copy and a distinct key", () => {
    expect(GOALS.length).toBeGreaterThanOrEqual(4)
    for (const g of GOALS) {
      expect(g.label.length).toBeGreaterThan(2)
      expect(g.sub.length).toBeGreaterThan(2)
    }
    const keys: Goal[] = GOALS.map((g) => g.key)
    expect(new Set(keys).size).toBe(GOALS.length)
  })
})

describe("profile persistence across the Mochiverse rename", () => {
  // node has no localStorage; a plain in-memory stub is enough here.
  function useStorage(seed: Record<string, string> = {}) {
    const map = new Map(Object.entries(seed))
    ;(globalThis as { localStorage?: unknown }).localStorage = {
      getItem: (k: string) => map.get(k) ?? null,
      setItem: (k: string, v: string) => void map.set(k, v),
      removeItem: (k: string) => void map.delete(k),
    }
    return map
  }

  const profile = { craft: "creator" as const, goal: "viral" as const, at: 123 }

  it("carries a pre-rename profile over instead of re-onboarding", () => {
    const store = useStorage({ "wand.profile.v1": JSON.stringify(profile) })

    expect(loadProfile()).toEqual(profile)
    // moved to the new key, and the old one is gone
    expect(store.get("mochiverse.profile.v1")).toBe(JSON.stringify(profile))
    expect(store.has("wand.profile.v1")).toBe(false)
  })

  it("prefers the new key and never writes the legacy one", () => {
    const store = useStorage()
    saveProfile(profile)
    expect(store.has("wand.profile.v1")).toBe(false)
    expect(loadProfile()).toEqual(profile)
  })

  it("returns null when neither key is present", () => {
    useStorage()
    expect(loadProfile()).toBeNull()
  })
})
