import { DatabaseSync } from "node:sqlite"
import { mkdirSync } from "node:fs"
import { dirname } from "node:path"

/**
 * The engagement list — the one piece of durable state this app keeps.
 *
 * Entitlement stays cookie-based (no session table), but re-engagement needs a
 * list that survives a cookie: who signed in, when we last saw them, and
 * whether they agreed to hear from us. `node:sqlite` is built into Node 22.5+,
 * so this costs one file on disk and zero services.
 *
 * Privacy/compliance shape:
 * - `phone_id` is the HMAC pseudonym, and is the PRIMARY KEY.
 * - `phone` (E.164) is stored ONLY when the user ticked the consent box — you
 *   cannot text someone you have no permission to text, so there is no reason
 *   to hold the number otherwise. Withdrawing consent nulls the number.
 * - `marketing_consent` is explicit and per-user (TCPA): sign-in confirmation
 *   texts are transactional, but anything promotional needs this flag AND its
 *   own A2P campaign.
 */

export interface Subscriber {
  readonly phoneId: string
  readonly marketingConsent: boolean
  readonly createdAt: number
  readonly lastSeenAt: number
}

let db: DatabaseSync | null = null

export function subscribersDb(path = process.env["SUBSCRIBERS_DB"] ?? ".data/subscribers.db"): DatabaseSync {
  if (db) return db
  mkdirSync(dirname(path), { recursive: true })
  const handle = new DatabaseSync(path)
  handle.exec(`
    CREATE TABLE IF NOT EXISTS subscribers (
      phone_id          TEXT PRIMARY KEY,
      phone             TEXT,
      marketing_consent INTEGER NOT NULL DEFAULT 0,
      created_at        INTEGER NOT NULL,
      last_seen_at      INTEGER NOT NULL
    ) STRICT;
  `)
  db = handle
  return handle
}

/** Upsert on sign-in. Returns true when this was a brand-new subscriber. */
export function recordSignIn(args: {
  phoneId: string
  phone: string
  marketingConsent: boolean
  now?: number
}): boolean {
  const handle = subscribersDb()
  const now = args.now ?? Date.now()
  const existing = handle
    .prepare("SELECT phone_id FROM subscribers WHERE phone_id = ?")
    .get(args.phoneId) as { phone_id?: string } | undefined

  // Only keep the dialable number while consent stands.
  const phone = args.marketingConsent ? args.phone : null

  if (existing?.phone_id) {
    handle
      .prepare("UPDATE subscribers SET marketing_consent = ?, phone = ?, last_seen_at = ? WHERE phone_id = ?")
      .run(args.marketingConsent ? 1 : 0, phone, now, args.phoneId)
    return false
  }
  handle
    .prepare(
      "INSERT INTO subscribers (phone_id, phone, marketing_consent, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
    )
    .run(args.phoneId, phone, args.marketingConsent ? 1 : 0, now, now)
  return true
}

/** Everyone who agreed to hear from us, for a future campaign script. */
export function marketingAudience(): ReadonlyArray<{ phoneId: string; phone: string }> {
  const rows = subscribersDb()
    .prepare("SELECT phone_id, phone FROM subscribers WHERE marketing_consent = 1 AND phone IS NOT NULL")
    .all() as Array<{ phone_id: string; phone: string }>
  return rows.map((r) => ({ phoneId: r.phone_id, phone: r.phone }))
}

/** Honor an opt-out: drop the number, keep the pseudonymous row. */
export function revokeConsent(phoneId: string): void {
  subscribersDb()
    .prepare("UPDATE subscribers SET marketing_consent = 0, phone = NULL WHERE phone_id = ?")
    .run(phoneId)
}

export function subscriberCount(): number {
  const row = subscribersDb().prepare("SELECT COUNT(*) AS n FROM subscribers").get() as { n: number }
  return row.n
}

/** Test seam: close + forget the handle so a fresh path can be opened. */
export function resetSubscribersDbForTests(): void {
  db?.close()
  db = null
}
