import { jwtVerify, SignJWT } from "jose"

/**
 * Entitlement state, carried entirely in HMAC-signed cookies (no database).
 *
 * Two cookies:
 * - `studio_meter` — {used: seconds} of the FREE allowance already consumed.
 *   The free minute is burned server-side when a free visitor mints a Decart
 *   token (the mint IS the metering event), so withholding client heartbeats
 *   can't extend it. Clearing cookies resets the free minute — an accepted v1
 *   trade for a zero-infra deploy; see README "Billing".
 * - `studio_sub` — {cus, sub, ver}: Stripe customer + subscription ids and the
 *   epoch-seconds when the subscription was last verified against Stripe. The
 *   token route re-verifies when `ver` is older than SUB_REVERIFY_SECONDS.
 *
 * Tampering with either cookie fails signature verification and reads as
 * "no cookie" (never as a grant).
 */

export const FREE_SECONDS = 60
export const METER_COOKIE = "studio_meter"
export const SUB_COOKIE = "studio_sub"
/** Re-check the subscription with Stripe at most this often. */
export const SUB_REVERIFY_SECONDS = 6 * 60 * 60
/** Cookie lifetimes: meter ~1 year, sub claim ~35 days (a billing cycle + slack). */
export const METER_MAX_AGE = 365 * 24 * 60 * 60
export const SUB_MAX_AGE = 35 * 24 * 60 * 60

const encoder = new TextEncoder()

export interface MeterClaims {
  readonly used: number
}

export interface SubClaims {
  readonly cus: string
  readonly sub: string
  readonly ver: number
}

export async function signState(
  claims: Record<string, unknown>,
  secret: string,
  maxAgeSeconds?: number,
): Promise<string> {
  const jwt = new SignJWT(claims).setProtectedHeader({ alg: "HS256" }).setIssuedAt()
  // Server-enforced lifetime: jose rejects an expired `exp` at verify time, so
  // a saved cookie can't outlive its browser Max-Age (review finding: the
  // 35-day sub cap must not depend on the client honoring cookie expiry).
  if (maxAgeSeconds !== undefined) jwt.setExpirationTime(`${maxAgeSeconds}s`)
  return jwt.sign(encoder.encode(secret))
}

/** Verify + decode; null on ANY failure (absent, malformed, tampered, wrong key). */
export async function readState<T>(jwt: string, secret: string): Promise<T | null> {
  try {
    const { payload } = await jwtVerify(jwt, encoder.encode(secret))
    return payload as T
  } catch {
    return null
  }
}

export function parseCookies(header: string | null): Record<string, string> {
  const jar: Record<string, string> = {}
  if (!header) return jar
  for (const part of header.split(";")) {
    const eq = part.indexOf("=")
    if (eq === -1) continue
    jar[part.slice(0, eq).trim()] = part.slice(eq + 1).trim()
  }
  return jar
}

export function cookieHeader(name: string, value: string, maxAgeSeconds: number, secure: boolean): string {
  return (
    `${name}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAgeSeconds}` +
    (secure ? "; Secure" : "")
  )
}

export async function meterFromRequest(request: Request, secret: string): Promise<MeterClaims> {
  const jar = parseCookies(request.headers.get("cookie"))
  const raw = jar[METER_COOKIE]
  if (!raw) return { used: 0 }
  const claims = await readState<Partial<MeterClaims>>(raw, secret)
  const used = typeof claims?.used === "number" && Number.isFinite(claims.used) ? claims.used : 0
  return { used: Math.max(0, used) }
}

export async function subFromRequest(request: Request, secret: string): Promise<SubClaims | null> {
  const jar = parseCookies(request.headers.get("cookie"))
  const raw = jar[SUB_COOKIE]
  if (!raw) return null
  const claims = await readState<Partial<SubClaims>>(raw, secret)
  if (!claims || typeof claims.cus !== "string" || typeof claims.sub !== "string") return null
  const now = Math.floor(Date.now() / 1000)
  // Clamp: a future `ver` must not be able to defer re-verification.
  const ver = typeof claims.ver === "number" ? Math.min(claims.ver, now) : 0
  return { cus: claims.cus, sub: claims.sub, ver }
}

/** True when the request arrived over HTTPS (directly or via a proxy header). */
export function requestIsSecure(request: Request): boolean {
  const proto = request.headers.get("x-forwarded-proto")
  if (proto) return proto.split(",")[0]?.trim() === "https"
  return new URL(request.url).protocol === "https:"
}
