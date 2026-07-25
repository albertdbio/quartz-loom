import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { resetSubscribersDbForTests, subscribersDb } from "../src/server/subscribers"
import { resetUsageTableForTests } from "../src/server/usage"
import { currentPeriod, grantSession, secondsUsed } from "../src/server/usage"

const QUOTA = 300 // 5 minutes
const SESSION = 120 // 2 minutes per grant
let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "wand-usage-"))
  resetSubscribersDbForTests()
  resetUsageTableForTests()
  subscribersDb(join(dir, "test.db"))
})

afterEach(() => {
  resetSubscribersDbForTests()
  rmSync(dir, { recursive: true, force: true })
})

describe("currentPeriod", () => {
  it("is the calendar month in YYYY-MM", () => {
    expect(currentPeriod(Date.UTC(2026, 6, 22))).toBe("2026-07")
    expect(currentPeriod(Date.UTC(2026, 11, 31))).toBe("2026-12")
    expect(currentPeriod(Date.UTC(2027, 0, 1))).toBe("2027-01")
  })
})

describe("grantSession", () => {
  it("grants a full session to a fresh member and records the burn", () => {
    const g = grantSession("pid1", { quota: QUOTA, session: SESSION })
    expect(g.granted).toBe(SESSION)
    expect(g.remaining).toBe(QUOTA - SESSION)
    expect(secondsUsed("pid1")).toBe(SESSION)
  })

  it("burns at grant time, so an unreported session still costs the member", () => {
    grantSession("pid1", { quota: QUOTA, session: SESSION })
    // no "session ended" call happens — the mint IS the metering event
    expect(secondsUsed("pid1")).toBe(SESSION)
  })

  it("never grants beyond the monthly quota, and the last grant is partial", () => {
    grantSession("pid1", { quota: QUOTA, session: SESSION }) // 120
    grantSession("pid1", { quota: QUOTA, session: SESSION }) // 240
    const third = grantSession("pid1", { quota: QUOTA, session: SESSION })
    expect(third.granted).toBe(60) // only 60s left of 300
    expect(third.remaining).toBe(0)
    expect(secondsUsed("pid1")).toBe(QUOTA)
  })

  it("refuses once exhausted", () => {
    for (let i = 0; i < 3; i++) grantSession("pid1", { quota: QUOTA, session: SESSION })
    const denied = grantSession("pid1", { quota: QUOTA, session: SESSION })
    expect(denied.granted).toBe(0)
    expect(denied.remaining).toBe(0)
    expect(secondsUsed("pid1")).toBe(QUOTA) // total never exceeds quota
  })

  it("keeps members independent", () => {
    grantSession("pid1", { quota: QUOTA, session: SESSION })
    expect(secondsUsed("pid2")).toBe(0)
    expect(grantSession("pid2", { quota: QUOTA, session: SESSION }).granted).toBe(SESSION)
  })

  it("resets when the calendar month rolls over", () => {
    const july = Date.UTC(2026, 6, 20)
    const august = Date.UTC(2026, 7, 1)
    for (let i = 0; i < 3; i++) grantSession("pid1", { quota: QUOTA, session: SESSION, now: july })
    expect(grantSession("pid1", { quota: QUOTA, session: SESSION, now: july }).granted).toBe(0)

    const fresh = grantSession("pid1", { quota: QUOTA, session: SESSION, now: august })
    expect(fresh.granted).toBe(SESSION)
    expect(secondsUsed("pid1", august)).toBe(SESSION)
    // July's ledger is untouched
    expect(secondsUsed("pid1", july)).toBe(QUOTA)
  })
})
