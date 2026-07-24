import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted } from "effect"
import { Billing, SessionSecretConfig } from "~/server/billing"
import { subFromRequest } from "~/server/entitlement"
import { errorJson, json } from "~/server/http"
import { originOf } from "~/server/plan"
import { runtime } from "~/server/runtime"

/** Stripe billing portal (cancel / update payment) for the signed-in subscriber. */
export async function POST(event: APIEvent): Promise<Response> {
  return runtime.runPromise(
    Effect.gen(function* () {
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const claim = yield* Effect.promise(() => subFromRequest(event.request, secret))
      if (!claim) return errorJson(401, "no active subscription on this browser")
      const billing = yield* Billing
      const url = yield* billing.createPortalSession(claim.cus, originOf(event.request))
      return json({ url })
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`portal session failed: ${e.message}`).pipe(
          Effect.as(errorJson(502, "could not open the billing portal")),
        ),
      ),
    ),
  )
}
