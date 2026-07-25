import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  marketingAudience,
  recordSignIn,
  resetSubscribersDbForTests,
  revokeConsent,
  subscriberCount,
  subscribersDb,
} from "../src/server/subscribers"

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "wand-subs-"))
  resetSubscribersDbForTests()
  subscribersDb(join(dir, "test.db"))
})

afterEach(() => {
  resetSubscribersDbForTests()
  rmSync(dir, { recursive: true, force: true })
})

describe("subscriber store", () => {
  it("records a new sign-in once and updates on return", () => {
    expect(recordSignIn({ phoneId: "pid1", phone: "+15551234567", marketingConsent: true })).toBe(true)
    expect(recordSignIn({ phoneId: "pid1", phone: "+15551234567", marketingConsent: true })).toBe(false)
    expect(subscriberCount()).toBe(1)
  })

  it("stores the dialable number ONLY with consent", () => {
    recordSignIn({ phoneId: "no-consent", phone: "+15551110000", marketingConsent: false })
    recordSignIn({ phoneId: "consented", phone: "+15552220000", marketingConsent: true })

    const audience = marketingAudience()
    expect(audience.map((a) => a.phoneId)).toEqual(["consented"])
    expect(audience[0]!.phone).toBe("+15552220000")

    // the non-consenting user is still counted, just not dialable
    expect(subscriberCount()).toBe(2)
  })

  it("drops the number when consent is withdrawn on a later sign-in", () => {
    recordSignIn({ phoneId: "pid1", phone: "+15551234567", marketingConsent: true })
    expect(marketingAudience()).toHaveLength(1)
    recordSignIn({ phoneId: "pid1", phone: "+15551234567", marketingConsent: false })
    expect(marketingAudience()).toHaveLength(0)
  })

  it("revokeConsent removes the number but keeps the pseudonymous row", () => {
    recordSignIn({ phoneId: "pid1", phone: "+15551234567", marketingConsent: true })
    revokeConsent("pid1")
    expect(marketingAudience()).toHaveLength(0)
    expect(subscriberCount()).toBe(1)
  })

  it("tracks last_seen across visits", () => {
    recordSignIn({ phoneId: "pid1", phone: "+1555", marketingConsent: false, now: 1_000 })
    recordSignIn({ phoneId: "pid1", phone: "+1555", marketingConsent: false, now: 5_000 })
    const row = subscribersDb().prepare("SELECT created_at, last_seen_at FROM subscribers WHERE phone_id='pid1'").get() as {
      created_at: number
      last_seen_at: number
    }
    expect(row.created_at).toBe(1_000)
    expect(row.last_seen_at).toBe(5_000)
  })
})
