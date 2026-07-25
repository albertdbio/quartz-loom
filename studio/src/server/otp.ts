import { randomInt, timingSafeEqual } from "node:crypto"
import { EncryptJWT, jwtDecrypt } from "jose"

/**
 * Phone sign-in primitives.
 *
 * The login challenge is carried in an ENCRYPTED (JWE) cookie rather than a
 * signed one, and that distinction is load-bearing: the challenge cookie lands
 * in the browser of whoever *started* the login, who may have typed someone
 * else's number. A signed-but-readable payload would hand them the 6-digit code
 * that was texted to the victim (or a hash they could brute-force offline in
 * milliseconds). Encrypted, the cookie is opaque — the only way to complete the
 * login is to hold the phone.
 *
 * Everything here is stateless: no session table, no user rows. Identity is a
 * signed cookie carrying `phoneId` (an HMAC of the number, never the number
 * itself), matching the entitlement model the rest of the app already uses.
 */

/** Cookie holding the encrypted login challenge. */
export const CHALLENGE_COOKIE = "studio_otp"

export const CODE_LENGTH = 6
export const CHALLENGE_TTL_SECONDS = 10 * 60

const encoder = new TextEncoder()

/** A256GCM direct encryption needs exactly 32 bytes of key material. */
async function encryptionKey(secret: string): Promise<Uint8Array> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(`otp:${secret}`))
  return new Uint8Array(digest)
}

/**
 * Best-effort E.164 normalization. A bare 10-digit number is assumed US (+1),
 * which matches where this ships first; anything else must arrive with its
 * country code. Returns null when the input can't be a phone number — callers
 * treat null as a 400 rather than texting garbage to Twilio.
 */
export function normalizePhone(input: string): string | null {
  const trimmed = input.trim()
  if (trimmed.length === 0) return null
  const hadPlus = trimmed.startsWith("+")
  const digits = trimmed.replace(/\D/g, "")
  if (digits.length === 0) return null

  let e164: string
  if (hadPlus) {
    e164 = digits
  } else if (digits.length === 10) {
    e164 = `1${digits}`
  } else if (digits.length === 11 && digits.startsWith("1")) {
    e164 = digits
  } else {
    e164 = digits
  }

  // E.164: 8–15 digits, country code can't start with 0.
  if (e164.length < 8 || e164.length > 15) return null
  if (e164.startsWith("0")) return null
  return `+${e164}`
}

/** Uniform 6-digit code from a CSPRNG (leading zeros preserved). */
export function generateCode(): string {
  return String(randomInt(0, 10 ** CODE_LENGTH)).padStart(CODE_LENGTH, "0")
}

/**
 * Stable pseudonymous id for a phone number. Identity cookies and the
 * subscriber store hold THIS, never the raw number, so a leaked cookie or
 * database row doesn't hand over someone's phone number.
 */
export async function phoneId(phoneE164: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(`pid:${secret}`),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  )
  const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(phoneE164))
  return Buffer.from(new Uint8Array(sig)).toString("base64url")
}

export interface Challenge {
  readonly phone: string
  readonly code: string
}

/** Encrypt a login challenge into an opaque cookie value. */
export async function sealChallenge(
  challenge: Challenge,
  secret: string,
  ttlSeconds: number = CHALLENGE_TTL_SECONDS,
): Promise<string> {
  const key = await encryptionKey(secret)
  return new EncryptJWT({ phone: challenge.phone, code: challenge.code })
    .setProtectedHeader({ alg: "dir", enc: "A256GCM" })
    .setIssuedAt()
    .setExpirationTime(`${ttlSeconds}s`)
    .encrypt(key)
}

/** Decrypt a challenge cookie; null on any failure (absent, tampered, expired). */
export async function openChallenge(token: string, secret: string): Promise<Challenge | null> {
  if (!token) return null
  try {
    const key = await encryptionKey(secret)
    const { payload } = await jwtDecrypt(token, key)
    const phone = payload["phone"]
    const code = payload["code"]
    if (typeof phone !== "string" || typeof code !== "string") return null
    return { phone, code }
  } catch {
    return null
  }
}

/** Constant-time code comparison (length mismatch = false, never throws). */
export function codeMatches(candidate: string, actual: string): boolean {
  const a = Buffer.from(candidate)
  const b = Buffer.from(actual)
  if (a.length === 0 || a.length !== b.length) return false
  return timingSafeEqual(a, b)
}
