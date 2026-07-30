import { afterEach, describe, expect, it, vi } from "vitest"
import { generateCharacter } from "../src/server/character"
import { alphaBBox } from "../src/lib/sprite"

/**
 * Wire-format tests shaped from REAL fal.ai responses captured 2026-07-28:
 * flux/schnell returns `{ images: [{ url }] }`, imageutils/rembg returns
 * `{ image: { url } }` — note the singular/plural asymmetry, which is exactly
 * the kind of thing a refactor silently breaks.
 */

const PNG = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3])

function mockFetchSequence(responses: Array<{ ok?: boolean; status?: number; json?: unknown; bytes?: Uint8Array }>) {
  const calls: Array<{ url: string; body?: unknown }> = []
  let i = 0
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), body: init?.body ? JSON.parse(String(init.body)) : undefined })
    const r = responses[Math.min(i++, responses.length - 1)]!
    return {
      ok: r.ok !== false,
      status: r.status ?? 200,
      json: async () => r.json,
      arrayBuffer: async () => (r.bytes ?? PNG).buffer,
    } as Response
  }))
  return calls
}

afterEach(() => vi.unstubAllGlobals())

describe("generateCharacter", () => {
  it("chains schnell -> rembg -> bytes and returns a PNG data URL", async () => {
    const calls = mockFetchSequence([
      { json: { images: [{ url: "https://fal.media/drawn.png" }] } },
      { json: { image: { url: "https://fal.media/cut.png" } } },
      { bytes: PNG },
    ])
    const out = await generateCharacter("a tiny dragon", "test-key")
    expect(out.startsWith("data:image/png;base64,")).toBe(true)
    expect(calls[0]!.url).toContain("fal-ai/flux/schnell")
    expect(calls[1]!.url).toContain("fal-ai/imageutils/rembg")
    expect(calls[1]!.body).toEqual({ image_url: "https://fal.media/drawn.png" })
    expect(calls[2]!.url).toBe("https://fal.media/cut.png")
  })

  it("wraps the subject in the character template, forbidding a painted shadow", async () => {
    const calls = mockFetchSequence([
      { json: { images: [{ url: "https://fal.media/drawn.png" }] } },
      { json: { image: { url: "https://fal.media/cut.png" } } },
      { bytes: PNG },
    ])
    await generateCharacter("a knight made of cheese", "test-key")
    const prompt = (calls[0]!.body as { prompt: string }).prompt
    expect(prompt).toContain("a knight made of cheese")
    expect(prompt).toContain("no ground shadow")
    expect(prompt).toContain("single character")
  })

  it("fails loudly when generation returns no image", async () => {
    mockFetchSequence([{ json: { images: [] } }])
    await expect(generateCharacter("x y z", "test-key")).rejects.toThrow("no image")
  })

  it("fails loudly when background removal returns no image", async () => {
    mockFetchSequence([
      { json: { images: [{ url: "https://fal.media/drawn.png" }] } },
      { json: {} },
    ])
    await expect(generateCharacter("x y z", "test-key")).rejects.toThrow("background removal")
  })

  it("refuses an absurdly large sprite instead of shipping it to localStorage", async () => {
    mockFetchSequence([
      { json: { images: [{ url: "https://fal.media/drawn.png" }] } },
      { json: { image: { url: "https://fal.media/cut.png" } } },
      { bytes: new Uint8Array(5 * 1024 * 1024) },
    ])
    await expect(generateCharacter("x y z", "test-key")).rejects.toThrow("large")
  })

  it("surfaces a fal HTTP failure with the endpoint name", async () => {
    mockFetchSequence([{ ok: false, status: 429, json: {} }])
    await expect(generateCharacter("x y z", "test-key")).rejects.toThrow("flux/schnell failed (429)")
  })
})

describe("alphaBBox (sprite crop core)", () => {

  function canvasData(w: number, h: number, opaque: Array<[number, number]>) {
    const d = new Uint8ClampedArray(w * h * 4)
    for (const [x, y] of opaque) d[(y * w + x) * 4 + 3] = 255
    return d
  }

  it("finds the opaque bounds inside transparent padding", () => {
    const d = canvasData(10, 10, [[3, 2], [6, 8]])
    expect(alphaBBox(d, 10, 10)).toEqual({ left: 3, top: 2, right: 6, bottom: 8 })
  })

  it("returns null for a fully transparent image", () => {
    expect(alphaBBox(canvasData(4, 4, []), 4, 4)).toBeNull()
  })

  it("ignores near-invisible antialiasing haze below the threshold", () => {
    const d = canvasData(10, 10, [[5, 5]])
    d[(0 * 10 + 0) * 4 + 3] = 4 // faint corner speck
    expect(alphaBBox(d, 10, 10)).toEqual({ left: 5, top: 5, right: 5, bottom: 5 })
  })
})
