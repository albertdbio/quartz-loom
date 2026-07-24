import { describe, expect, it } from "vitest"
import {
  GOALS,
  SPELLS,
  personalLine,
  rankSpells,
  starterSpellIndex,
  type Goal,
} from "../src/lib/spells"

describe("spell deck", () => {
  it("has a unique name for every spell", () => {
    const names = SPELLS.map((s) => s.name)
    expect(new Set(names).size).toBe(names.length)
  })
})

describe("rankSpells", () => {
  it("keeps every spell exactly once for every goal (no drops, no dupes)", () => {
    for (const g of GOALS) {
      const ranked = rankSpells(SPELLS, g.key)
      expect(ranked).toHaveLength(SPELLS.length)
      expect(new Set(ranked.map((s) => s.name)).size).toBe(SPELLS.length)
    }
  })

  it("leads with the goal's signature spell", () => {
    expect(rankSpells(SPELLS, "viral")[0]!.name).toBe("8-bit")
    expect(rankSpells(SPELLS, "product")[0]!.name).toBe("Midas")
    expect(rankSpells(SPELLS, "art")[0]!.name).toBe("Sketch")
    expect(rankSpells(SPELLS, "teach")[0]!.name).toBe("Sketch")
  })

  it("leaves the deck untouched for the undecided", () => {
    expect(rankSpells(SPELLS, "explore").map((s) => s.name)).toEqual(SPELLS.map((s) => s.name))
  })

  it("is pure — the source deck is never mutated", () => {
    const before = SPELLS.map((s) => s.name)
    rankSpells(SPELLS, "viral")
    expect(SPELLS.map((s) => s.name)).toEqual(before)
  })
})

describe("starterSpellIndex", () => {
  it("points at the lead spell's position in the RANKED deck (always 0)", () => {
    for (const g of GOALS) {
      const ranked = rankSpells(SPELLS, g.key)
      const i = starterSpellIndex(ranked, g.key)
      expect(i).toBe(0)
      expect(ranked[i]).toBeDefined()
    }
  })
})

describe("personalLine", () => {
  it("returns a non-empty line naming the starter spell for every combination", () => {
    const crafts = ["creator", "artist", "marketer", "educator", "fun"] as const
    for (const craft of crafts) {
      for (const g of GOALS) {
        const ranked = rankSpells(SPELLS, g.key)
        const line = personalLine(craft, g.key, ranked[0]!.name)
        expect(line.length).toBeGreaterThan(10)
        expect(line).toContain(ranked[0]!.name)
      }
    }
  })

  it("differs by goal so the answer visibly mattered", () => {
    const lines = GOALS.map((g) => personalLine("creator", g.key, rankSpells(SPELLS, g.key)[0]!.name))
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
