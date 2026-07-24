import { Effect, Redacted } from "effect"
import { Billing, SessionSecretConfig, StripeError } from "./billing"
import {
  cookieHeader,
  FREE_SECONDS,
  meterFromRequest,
  requestIsSecure,
  signState,
  SUB_COOKIE,
  SUB_MAX_AGE,
  SUB_REVERIFY_SECONDS,
  subFromRequest,
} from "./entitlement"

/**
 * Resolve the caller's entitlement for this request.
 *
 * Pro path: a signed `studio_sub` cookie. If its last Stripe verification is
 * older than SUB_REVERIFY_SECONDS we re-check the subscription; a lapsed
 * subscription downgrades to the free path (and clears the cookie). If the
 * Stripe API itself is unreachable during re-verification we fail OPEN for the
 * existing claim — a paying customer is never locked out by our outage; the
 * cookie's own 35-day expiry bounds the exposure.
 *
 * Free path: the signed `studio_meter` cookie; `remaining` counts down from
 * FREE_SECONDS and the token route burns it on mint.
 */
/**
 * Sentinel `cus`/`sub` value for the operator's own entitlement (minted by
 * /api/billing/owner with OWNER_KEY). Owner claims are honored WITHOUT any
 * Stripe verification — there is no subscription behind them — and are still
 * HMAC-signed + expiry-bound like every other claim.
 */
export const OWNER_SUB = "owner"

export type ResolvedPlan =
  | { readonly plan: "pro"; readonly cus: string; readonly sub: string; readonly setCookies: ReadonlyArray<string> }
  | { readonly plan: "free"; readonly used: number; readonly remaining: number; readonly setCookies: ReadonlyArray<string> }

export const resolvePlan = (request: Request): Effect.Effect<ResolvedPlan, StripeError, Billing> =>
  Effect.gen(function* () {
    const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
    const secure = requestIsSecure(request)
    const claim = yield* Effect.promise(() => subFromRequest(request, secret))

    if (claim) {
      if (claim.sub === OWNER_SUB) {
        return { plan: "pro", cus: claim.cus, sub: claim.sub, setCookies: [] } as const
      }
      const ageSeconds = Math.floor(Date.now() / 1000) - claim.ver
      if (ageSeconds < SUB_REVERIFY_SECONDS) {
        return { plan: "pro", cus: claim.cus, sub: claim.sub, setCookies: [] } as const
      }
      const billing = yield* Billing
      // null = Stripe unreachable. Fail OPEN for this request only: the claim
      // stays honored but `ver` and the cookie are NOT refreshed, so grace is
      // bounded by the JWT's own signed expiry — not renewed forever (review
      // finding: outage grace must not extend the entitlement window).
      const active = yield* billing.subscriptionActive(claim.sub).pipe(
        Effect.map((a): boolean | null => a),
        Effect.catchTag("StripeError", () => Effect.succeed(null)),
      )
      if (active === null) {
        return { plan: "pro", cus: claim.cus, sub: claim.sub, setCookies: [] } as const
      }
      if (active) {
        const refreshed = yield* Effect.promise(() =>
          signState({ cus: claim.cus, sub: claim.sub, ver: Math.floor(Date.now() / 1000) }, secret, SUB_MAX_AGE),
        )
        return {
          plan: "pro",
          cus: claim.cus,
          sub: claim.sub,
          setCookies: [cookieHeader(SUB_COOKIE, refreshed, SUB_MAX_AGE, secure)],
        } as const
      }
      const meter = yield* Effect.promise(() => meterFromRequest(request, secret))
      return {
        plan: "free",
        used: meter.used,
        remaining: Math.max(0, FREE_SECONDS - meter.used),
        setCookies: [cookieHeader(SUB_COOKIE, "", 0, secure)],
      } as const
    }

    const meter = yield* Effect.promise(() => meterFromRequest(request, secret))
    return {
      plan: "free",
      used: meter.used,
      remaining: Math.max(0, FREE_SECONDS - meter.used),
      setCookies: [],
    } as const
  })

/**
 * Public origin for Stripe redirect URLs. PUBLIC_BASE_URL is REQUIRED in
 * production; the dev fallback uses only the request's own URL — never
 * `x-forwarded-host`, which is attacker-influenced on misconfigured proxies
 * and would let a crafted checkout send Stripe's success redirect (carrying
 * the session_id) to a hostile origin (review finding, P0).
 */
export function originOf(request: Request): string {
  const configured = process.env["PUBLIC_BASE_URL"]?.trim()
  if (configured) return configured.replace(/\/$/, "")
  if (process.env["NODE_ENV"] === "production" || process.env["VERCEL"] === "1") {
    throw new Error("PUBLIC_BASE_URL must be set in production for Stripe redirect URLs")
  }
  return new URL(request.url).origin
}

/** JSON response that can carry Set-Cookie headers. */
export function jsonWithCookies(data: unknown, cookies: ReadonlyArray<string>, status = 200): Response {
  const headers = new Headers({ "Content-Type": "application/json" })
  for (const c of cookies) headers.append("Set-Cookie", c)
  return new Response(JSON.stringify(data), { status, headers })
}
