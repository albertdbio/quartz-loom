import type { APIEvent } from "@solidjs/start/server"
import { Effect } from "effect"
import { Billing } from "~/server/billing"
import { cookieHeader, requestIsSecure } from "~/server/entitlement"
import { errorJson } from "~/server/http"
import { jsonWithCookies, originOf } from "~/server/plan"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

export const CHECKOUT_NONCE_COOKIE = "studio_ckn"

/**
 * Start a $20/month Studio Pro subscription — returns the Stripe Checkout URL.
 * A random nonce rides in the checkout session's metadata AND a short-lived
 * cookie; /api/billing/confirm requires them to match, so a session_id leaked
 * through history/logs can't hand the subscription cookie to another browser.
 */
export async function POST(event: APIEvent): Promise<Response> {
  const ip = clientIp(event.request)
  if (!rateLimit(`checkout:${ip}`, 10, 60_000)) {
    return errorJson(429, "too many checkout attempts — try again in a minute")
  }
  const nonce = crypto.randomUUID()
  return runtime.runPromise(
    Effect.gen(function* () {
      const billing = yield* Billing
      const url = yield* billing.createCheckoutSession(originOf(event.request), nonce)
      return jsonWithCookies({ url }, [
        cookieHeader(CHECKOUT_NONCE_COOKIE, nonce, 30 * 60, requestIsSecure(event.request)),
      ])
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`checkout session failed: ${e.message}`).pipe(
          Effect.as(errorJson(502, "could not start checkout — billing may not be configured")),
        ),
      ),
    ),
  )
}
