import { subscribersDb } from "./subscribers"

/**
 * Device push tokens, so a member can be told their minutes reset.
 *
 * Keyed by TOKEN rather than by user, because the relationship is many-to-one
 * in both directions over time: one person can install on several devices, and
 * one device can be handed to a different account. Re-registering the same
 * token simply re-points it at whoever is signed in now, which is what keeps a
 * shared or resold device from being notified for a stranger.
 *
 * `phone_id` is nullable on purpose — the shell registers as soon as the user
 * accepts the OS prompt, which can happen before they ever sign in. Those rows
 * are addressable for install-wide messages and get claimed on the next
 * registration after sign-in.
 */

let ready = false

function table() {
  const db = subscribersDb()
  if (!ready) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS push_tokens (
        token        TEXT PRIMARY KEY,
        phone_id     TEXT,
        platform     TEXT NOT NULL,
        created_at   INTEGER NOT NULL,
        last_seen_at INTEGER NOT NULL
      ) STRICT;
    `)
    ready = true
  }
  return db
}

export type PushPlatform = "ios" | "android"

/** Upsert a device token. Returns true when the token was not already known. */
export function recordPushToken(args: {
  token: string
  platform: PushPlatform
  phoneId?: string | null
  now?: number
}): boolean {
  const db = table()
  const now = args.now ?? Date.now()
  const phoneId = args.phoneId ?? null

  const existing = db
    .prepare("SELECT token FROM push_tokens WHERE token = ?")
    .get(args.token) as { token?: string } | undefined

  if (existing?.token) {
    // A re-registration re-points the device at the current signer. An
    // anonymous re-register must NOT blank an existing link, or signing in
    // then reopening the app would quietly orphan the row.
    if (phoneId === null) {
      db.prepare("UPDATE push_tokens SET platform = ?, last_seen_at = ? WHERE token = ?").run(
        args.platform,
        now,
        args.token,
      )
    } else {
      db.prepare(
        "UPDATE push_tokens SET phone_id = ?, platform = ?, last_seen_at = ? WHERE token = ?",
      ).run(phoneId, args.platform, now, args.token)
    }
    return false
  }

  db.prepare(
    "INSERT INTO push_tokens (token, phone_id, platform, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
  ).run(args.token, phoneId, args.platform, now, now)
  return true
}

/** Every token for a signed-in person — the address list for a member notice. */
export function tokensFor(phoneId: string): ReadonlyArray<{ token: string; platform: PushPlatform }> {
  const rows = table()
    .prepare("SELECT token, platform FROM push_tokens WHERE phone_id = ?")
    .all(phoneId) as Array<{ token: string; platform: string }>
  return rows.map((r) => ({ token: r.token, platform: r.platform as PushPlatform }))
}

/** Drop a token the push service has told us is dead. */
export function forgetPushToken(token: string): void {
  table().prepare("DELETE FROM push_tokens WHERE token = ?").run(token)
}

/** Test hook: the cached table check is per-process, so it must be cleared
 *  whenever the underlying database is swapped. */
export function resetPushTableForTests(): void {
  ready = false
}
