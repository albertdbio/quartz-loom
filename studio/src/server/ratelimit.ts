/**
 * Tiny in-memory sliding-window rate limiter for the token endpoint.
 *
 * Not distributed — one process only, which is fine for the studio demo/single
 * deploy. Keyed by client IP. Each token mint enables a paid realtime session
 * ($0.04/s), so this caps how fast sessions can be started from one address.
 * Before a multi-instance deploy, swap for a shared store (Redis/Upstash).
 */
const hits = new Map<string, number[]>()

export function rateLimit(key: string, max: number, windowMs: number): boolean {
  const now = Date.now()
  const recent = (hits.get(key) ?? []).filter((t) => now - t < windowMs)
  if (recent.length >= max) {
    hits.set(key, recent)
    return false
  }
  recent.push(now)
  hits.set(key, recent)
  return true
}
