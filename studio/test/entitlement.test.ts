import { describe, expect, it } from "vitest"
import {
  FREE_SECONDS,
  cookieHeader,
  meterFromRequest,
  parseCookies,
  signState,
  readState,
  subFromRequest,
} from "../src/server/entitlement"

const SECRET = "test-secret-please-ignore"

const requestWithCookie = (cookie: string): Request =>
  new Request("http://localhost/api/decart/token", { headers: { cookie } })

describe("signed state", () => {
  it("round-trips claims through sign + read", async () => {
    const jwt = await signState({ used: 60 }, SECRET)
    const claims = await readState<{ used: number }>(jwt, SECRET)
    expect(claims).not.toBeNull()
    expect(claims!.used).toBe(60)
  })

  it("rejects a tampered token", async () => {
    const jwt = await signState({ used: 0 }, SECRET)
    // flip a payload character — signature must no longer verify
    const [h, p, s] = jwt.split(".")
    const tampered = `${h}.${p!.slice(0, -2)}AA.${s}`
    expect(await readState(tampered, SECRET)).toBeNull()
  })

  it("rejects a token signed with a different secret", async () => {
    const jwt = await signState({ used: 0 }, "other-secret")
    expect(await readState(jwt, SECRET)).toBeNull()
  })
})

describe("cookie plumbing", () => {
  it("parses a cookie header", () => {
    const jar = parseCookies("a=1; studio_meter=abc.def.ghi; b=2")
    expect(jar["studio_meter"]).toBe("abc.def.ghi")
  })

  it("parses an absent header to an empty jar", () => {
    expect(parseCookies(null)).toEqual({})
  })

  it("serializes an HttpOnly cookie with Max-Age", () => {
    const h = cookieHeader("studio_meter", "tok", 3600, false)
    expect(h).toContain("studio_meter=tok")
    expect(h).toContain("HttpOnly")
    expect(h).toContain("Max-Age=3600")
    expect(h).toContain("SameSite=Lax")
    expect(h).not.toContain("Secure")
    expect(cookieHeader("x", "y", 1, true)).toContain("Secure")
  })
})

describe("meter", () => {
  it("defaults to zero usage with no cookie", async () => {
    const m = await meterFromRequest(requestWithCookie(""), SECRET)
    expect(m.used).toBe(0)
  })

  it("reads a signed meter and reports exhaustion at FREE_SECONDS", async () => {
    const jwt = await signState({ used: FREE_SECONDS }, SECRET)
    const m = await meterFromRequest(requestWithCookie(`studio_meter=${jwt}`), SECRET)
    expect(m.used).toBe(FREE_SECONDS)
    expect(m.used >= FREE_SECONDS).toBe(true)
  })

  it("treats a tampered meter as zero (cannot self-grant, cannot self-exhaust others)", async () => {
    const m = await meterFromRequest(requestWithCookie("studio_meter=not.a.jwt"), SECRET)
    expect(m.used).toBe(0)
  })
})

describe("server-enforced expiry (review: 35-day cap must not depend on the browser)", () => {
  it("embeds exp when signed with a max age", async () => {
    const jwt = await signState({ used: 0 }, SECRET, 3600)
    const claims = await readState<{ exp?: number }>(jwt, SECRET)
    const expected = Math.floor(Date.now() / 1000) + 3600
    expect(claims?.exp).toBeGreaterThan(expected - 5)
    expect(claims?.exp).toBeLessThan(expected + 5)
  })

  it("rejects an expired token even if replayed outside the browser", async () => {
    const { SignJWT } = await import("jose")
    const expired = await new SignJWT({ cus: "cus_1", sub: "sub_1", ver: 0 })
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt(Math.floor(Date.now() / 1000) - 7200)
      .setExpirationTime(Math.floor(Date.now() / 1000) - 3600)
      .sign(new TextEncoder().encode(SECRET))
    expect(await readState(expired, SECRET)).toBeNull()
  })
})

describe("subscription claim", () => {
  it("reads a valid subscription cookie", async () => {
    const jwt = await signState({ cus: "cus_123", sub: "sub_456", ver: 1_700_000_000 }, SECRET)
    const s = await subFromRequest(requestWithCookie(`studio_sub=${jwt}`), SECRET)
    expect(s).not.toBeNull()
    expect(s!.cus).toBe("cus_123")
    expect(s!.sub).toBe("sub_456")
  })

  it("returns null with no cookie or an invalid one", async () => {
    expect(await subFromRequest(requestWithCookie(""), SECRET)).toBeNull()
    expect(await subFromRequest(requestWithCookie("studio_sub=garbage"), SECRET)).toBeNull()
  })

  it("clamps a future `ver` so re-verification cannot be deferred (review finding)", async () => {
    const future = Math.floor(Date.now() / 1000) + 10 * 24 * 3600
    const jwt = await signState({ cus: "cus_1", sub: "sub_1", ver: future }, SECRET)
    const s = await subFromRequest(requestWithCookie(`studio_sub=${jwt}`), SECRET)
    expect(s!.ver).toBeLessThanOrEqual(Math.floor(Date.now() / 1000))
  })
})
