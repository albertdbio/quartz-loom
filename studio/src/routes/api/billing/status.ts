import type { APIEvent } from "@solidjs/start/server"
import { Effect } from "effect"
import { errorJson } from "~/server/http"
import { jsonWithCookies, paymentsEnabled, resolvePlan } from "~/server/plan"
import { runtime } from "~/server/runtime"
import { remainingSeconds } from "~/server/usage"

/** Entitlement snapshot for the studio UI: plan + remaining free seconds. */
export async function GET(event: APIEvent): Promise<Response> {
  return runtime.runPromise(
    Effect.gen(function* () {
      const plan = yield* resolvePlan(event.request)
      if (plan.plan === "pro") return jsonWithCookies({ plan: "pro" }, plan.setCookies)
      if (plan.plan === "member") {
        const remaining = yield* Effect.sync(() => remainingSeconds(plan.pid))
        return jsonWithCookies({ plan: "member", remainingSeconds: remaining }, plan.setCookies)
      }
      return jsonWithCookies(
        { plan: "free", remainingSeconds: plan.remaining, paymentsEnabled: paymentsEnabled() },
        plan.setCookies,
      )
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`billing status failed: ${e.message}`).pipe(
          Effect.as(errorJson(502, "billing status unavailable")),
        ),
      ),
    ),
  )
}
