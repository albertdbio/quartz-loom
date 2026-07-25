import { describe, expect, it } from "vitest"
import {
  CODE_LENGTH,
  generateCode,
  normalizePhone,
  openChallenge,
  phoneId,
  sealChallenge,
} from "../src/server/otp"

const SECRET = "test-secret-please-ignore-0123456789"

describe("normalizePhone", () => {
  it("normalizes common US shapes to E.164", () => {
    for (const input of ["5551234567", "(555) 123-4567", "555-123-4567", "+1 555 123 4567", "1 555 123 4567"]) {
      expect(normalizePhone(input)).toBe("+15551234567")
    }
  })

  it("keeps an already-E.164 international number", () => {
    expect(normalizePhone("+442071838750")).toBe("+442071838750")
    expect(normalizePhone("+81 3-1234-5678")).toBe("+81312345678")
  })

  it("rejects anything that isn't a plausible phone number", () => {
    for (const bad of ["", "   ", "abc", "123", "+", "+0123456789", "1".repeat(20), "555-12"]) {
      expect(normalizePhone(bad)).toBeNull()
    }
  })
})

describe("generateCode", () => {
  it("is always CODE_LENGTH digits", () => {
    for (let i = 0; i < 200; i++) {
      const c = generateCode()
      expect(c).toMatch(new RegExp(`^\\d{${CODE_LENGTH}}$`))
    }
  })

  it("is not constant (uses real randomness)", () => {
    const seen = new Set(Array.from({ length: 100 }, () => generateCode()))
    expect(seen.size).toBeGreaterThan(50)
  })
})

describe("phoneId", () => {
  it("is stable for the same number and different across numbers", async () => {
    const a = await phoneId("+15551234567", SECRET)
    const b = await phoneId("+15551234567", SECRET)
    const c = await phoneId("+15559999999", SECRET)
    expect(a).toBe(b)
    expect(a).not.toBe(c)
  })

  it("does not contain the raw phone number", async () => {
    const id = await phoneId("+15551234567", SECRET)
    expect(id).not.toContain("5551234567")
  })
})

describe("challenge sealing", () => {
  it("round-trips the phone and code", async () => {
    const token = await sealChallenge({ phone: "+15551234567", code: "123456" }, SECRET, 600)
    const opened = await openChallenge(token, SECRET)
    expect(opened).not.toBeNull()
    expect(opened!.phone).toBe("+15551234567")
    expect(opened!.code).toBe("123456")
  })

  it("MUST NOT leak the code to whoever holds the cookie", async () => {
    // The challenge cookie lives in the browser of whoever STARTED the login —
    // which may be an attacker who typed someone else's number. If the payload
    // were merely signed, they could read (or offline-brute-force) the 6-digit
    // code that was texted to the victim and log in as them. It must be opaque.
    const token = await sealChallenge({ phone: "+15551234567", code: "123456" }, SECRET, 600)
    expect(token).not.toContain("123456")
    expect(token).not.toContain("15551234567")
    // and it must not be readable by simply base64-decoding the segments
    const decoded = token
      .split(".")
      .map((seg) => {
        try {
          return Buffer.from(seg, "base64url").toString("utf8")
        } catch {
          return ""
        }
      })
      .join("|")
    expect(decoded).not.toContain("123456")
    expect(decoded).not.toContain("15551234567")
  })

  it("rejects a token sealed with a different secret", async () => {
    const token = await sealChallenge({ phone: "+15551234567", code: "123456" }, "other-secret-entirely", 600)
    expect(await openChallenge(token, SECRET)).toBeNull()
  })

  it("rejects a tampered token", async () => {
    const token = await sealChallenge({ phone: "+15551234567", code: "123456" }, SECRET, 600)
    const parts = token.split(".")
    const tampered = [...parts.slice(0, 3), `${parts[3]!.slice(0, -2)}AA`, parts[4]].join(".")
    expect(await openChallenge(tampered, SECRET)).toBeNull()
  })

  it("rejects an expired challenge", async () => {
    const token = await sealChallenge({ phone: "+15551234567", code: "123456" }, SECRET, -1)
    expect(await openChallenge(token, SECRET)).toBeNull()
  })

  it("rejects garbage", async () => {
    expect(await openChallenge("not-a-token", SECRET)).toBeNull()
    expect(await openChallenge("", SECRET)).toBeNull()
  })
})
