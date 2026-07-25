import { subscribersDb } from "./subscribers"

/**
 * Monthly generation budget per phone-verified member.
 *
 * The hosted engine bills per second of active generation, so an uncapped
 * "member" tier is a blank cheque drawn on the operator's card — worse with
 * payments off, because there is no revenue on the other side. Every mint
 * therefore draws down a monthly allowance.
 *
 * Burn happens AT GRANT TIME, exactly like the anonymous free minute: the mint
 * is the metering event. Nothing depends on the client reporting back when a
 * session ended, so closing the tab, killing the app, or dropping the network
 * can't buy free seconds. The client is told how long it was granted and stops
 * itself; the server has already charged for it.
 */

/** 20 minutes a month, in seconds. */
export const MEMBER_MONTHLY_SECONDS = Number(process.env["MEMBER_MONTHLY_SECONDS"] ?? 1200)
/** How much one mint is allowed to hold, in seconds. */
export const SESSION_SECONDS = Number(process.env["SESSION_SECONDS"] ?? 120)

let ready = false

function table() {
  const db = subscribersDb()
  if (!ready) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS usage (
        phone_id TEXT NOT NULL,
        period   TEXT NOT NULL,
        seconds  INTEGER NOT NULL,
        PRIMARY KEY (phone_id, period)
      ) STRICT;
    `)
    ready = true
  }
  return db
}

/** Calendar month key, e.g. "2026-07". Quotas reset on the 1st. */
export function currentPeriod(now: number = Date.now()): string {
  const d = new Date(now)
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`
}

export function secondsUsed(phoneId: string, now: number = Date.now()): number {
  const row = table()
    .prepare("SELECT seconds FROM usage WHERE phone_id = ? AND period = ?")
    .get(phoneId, currentPeriod(now)) as { seconds?: number } | undefined
  return row?.seconds ?? 0
}

export interface Grant {
  /** Seconds this session may run. 0 means the monthly budget is spent. */
  readonly granted: number
  /** Seconds left after this grant. */
  readonly remaining: number
}

/**
 * Draw one session from the member's monthly budget. The final grant of a month
 * is partial rather than refused, so a member never loses the tail of what they
 * were promised.
 */
export function grantSession(
  phoneId: string,
  opts: { quota?: number; session?: number; now?: number } = {},
): Grant {
  const quota = opts.quota ?? MEMBER_MONTHLY_SECONDS
  const session = opts.session ?? SESSION_SECONDS
  const now = opts.now ?? Date.now()
  const period = currentPeriod(now)

  const db = table()
  const used = secondsUsed(phoneId, now)
  const left = Math.max(0, quota - used)
  if (left <= 0) return { granted: 0, remaining: 0 }

  const granted = Math.min(session, left)
  db.prepare(
    `INSERT INTO usage (phone_id, period, seconds) VALUES (?, ?, ?)
     ON CONFLICT(phone_id, period) DO UPDATE SET seconds = seconds + excluded.seconds`,
  ).run(phoneId, period, granted)

  return { granted, remaining: left - granted }
}

/** Seconds still available this month, without spending any. */
export function remainingSeconds(phoneId: string, now: number = Date.now()): number {
  return Math.max(0, MEMBER_MONTHLY_SECONDS - secondsUsed(phoneId, now))
}

/** Test seam — the usage table lives in the subscribers db. */
export function resetUsageTableForTests(): void {
  ready = false
}
