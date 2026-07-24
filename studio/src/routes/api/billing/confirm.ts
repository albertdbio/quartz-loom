import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted } from "effect"
import { Billing, SessionSecretConfig } from "~/server/billing"
import {
  cookieHeader,
  parseCookies,
  requestIsSecure,
  signState,
  SUB_COOKIE,
  SUB_MAX_AGE,
} from "~/server/entitlement"
import { runtime } from "~/server/runtime"
import { CHECKOUT_NONCE_COOKIE } from "../billing/checkout"

const redirect = (to: string, cookies: ReadonlyArray<string> = []): Response => {
  const headers = new Headers({ Location: to })
  for (const c of cookies) headers.append("Set-Cookie", c)
  return new Response(null, { status: 302, headers })
}

/**
 * Stripe Checkout success redirect. Verifies the session SERVER-SIDE with the
 * secret key (the session_id in the URL proves nothing by itself), then sets
 * the signed subscription cookie and lands the user back in the studio.
 */
export async function GET(event: APIEvent): Promise<Response> {
  const sessionId = new URL(event.request.url).searchParams.get("session_id")
  if (!sessionId) return redirect("/studio?checkout=failed")
  return runtime.runPromise(
    Effect.gen(function* () {
      const billing = yield* Billing
      const v = yield* billing.verifyCheckout(sessionId)
      if (!v.active || !v.customerId || !v.subscriptionId) {
        return redirect("/studio?checkout=failed")
      }
      // Nonce binding: the confirm must come from the browser that started
      // checkout — a bare (leaked/replayed) session_id is not enough.
      const jar = parseCookies(event.request.headers.get("cookie"))
      if (!v.nonce || jar[CHECKOUT_NONCE_COOKIE] !== v.nonce) {
        return redirect("/studio?checkout=failed")
      }
      const secure = requestIsSecure(event.request)
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const jwt = yield* Effect.promise(() =>
        signState({ cus: v.customerId, sub: v.subscriptionId, ver: Math.floor(Date.now() / 1000) }, secret, SUB_MAX_AGE),
      )
      return redirect("/studio?pro=1", [
        cookieHeader(SUB_COOKIE, jwt, SUB_MAX_AGE, secure),
        cookieHeader(CHECKOUT_NONCE_COOKIE, "", 0, secure),
      ])
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`checkout confirm failed: ${e.message}`).pipe(
          Effect.as(redirect("/studio?checkout=failed")),
        ),
      ),
    ),
  )
}
