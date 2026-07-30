import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { nativeSpeechAvailable, startNativeSpeech } from "../src/lib/native"

/**
 * The native speech session is a protocol adapter: bridge messages in,
 * callbacks out. A fake window is enough to pin the whole contract —
 * including the invariant that a stopped session never speaks again.
 */

type Listener = (ev: { data: string }) => void

function fakeShell({ speech = true } = {}) {
  const listeners = new Set<Listener>()
  const sent: Array<{ type: string }> = []
  const win = {
    __mochiverseNative: { version: 2, speech },
    ReactNativeWebView: { postMessage: (m: string) => sent.push(JSON.parse(m)) },
    addEventListener: (_: string, fn: Listener) => listeners.add(fn),
    removeEventListener: (_: string, fn: Listener) => listeners.delete(fn),
  }
  vi.stubGlobal("window", win)
  const emit = (payload: unknown) => listeners.forEach((fn) => fn({ data: JSON.stringify(payload) }))
  return { sent, emit, listenerCount: () => listeners.size }
}

function collector() {
  return {
    interim: [] as string[],
    final: [] as string[],
    ends: [] as Array<string | null>,
    cb(this: void) {},
  }
}

afterEach(() => vi.unstubAllGlobals())

describe("startNativeSpeech", () => {
  it("starts the session and routes interim/final transcripts", () => {
    const shell = fakeShell()
    const got = collector()
    const session = startNativeSpeech({
      onInterim: (t) => got.interim.push(t),
      onFinal: (t) => got.final.push(t),
      onEnd: (r) => got.ends.push(r),
    })
    expect(session).not.toBeNull()
    expect(shell.sent).toEqual([{ v: 1, type: "speech:start" }])

    shell.emit({ type: "speech:interim", text: "make it" })
    shell.emit({ type: "speech:final", text: "make it snow" })
    expect(got.interim).toEqual(["make it"])
    expect(got.final).toEqual(["make it snow"])
    expect(got.ends).toEqual([])
  })

  it("surfaces a permission denial as an end with its reason", () => {
    const shell = fakeShell()
    const got = collector()
    startNativeSpeech({
      onInterim: () => {},
      onFinal: () => {},
      onEnd: (r) => got.ends.push(r),
    })
    shell.emit({ type: "speech:error", reason: "not-allowed" })
    expect(got.ends).toEqual(["not-allowed"])
    // the dead session must not leak its listener
    expect(shell.listenerCount()).toBe(0)
  })

  it("reports a natural end with null so the caller can choose to restart", () => {
    const shell = fakeShell()
    const got = collector()
    startNativeSpeech({ onInterim: () => {}, onFinal: () => {}, onEnd: (r) => got.ends.push(r) })
    shell.emit({ type: "speech:end" })
    expect(got.ends).toEqual([null])
  })

  it("goes silent after stop — no callbacks, and speech:stop is sent", () => {
    const shell = fakeShell()
    const got = collector()
    const session = startNativeSpeech({
      onInterim: (t) => got.interim.push(t),
      onFinal: () => {},
      onEnd: (r) => got.ends.push(r),
    })!
    session.stop()
    expect(shell.sent.map((m) => m.type)).toEqual(["speech:start", "speech:stop"])
    shell.emit({ type: "speech:interim", text: "ghost words" })
    shell.emit({ type: "speech:end" })
    expect(got.interim).toEqual([])
    expect(got.ends).toEqual([])
    expect(shell.listenerCount()).toBe(0)
  })

  it("returns null when the shell lacks the capability", () => {
    fakeShell({ speech: false })
    expect(nativeSpeechAvailable()).toBe(false)
    expect(startNativeSpeech({ onInterim: () => {}, onFinal: () => {}, onEnd: () => {} })).toBeNull()
  })
})
