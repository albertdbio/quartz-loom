import type { APIEvent } from "@solidjs/start/server"
import { Effect } from "effect"
import { errorJson } from "~/server/http"
import { jsonWithCookies, resolvePlan } from "~/server/plan"
import { runtime } from "~/server/runtime"

/** Entitlement snapshot for the studio UI: plan + remaining free seconds. */
export async function GET(event: APIEvent): Promise<Response> {
  return runtime.runPromise(
    Effect.gen(function* () {
      const plan = yield* resolvePlan(event.request)
      return plan.plan === "pro"
        ? jsonWithCookies({ plan: "pro" }, plan.setCookies)
        : jsonWithCookies({ plan: "free", remainingSeconds: plan.remaining }, plan.setCookies)
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`billing status failed: ${e.message}`).pipe(
          Effect.as(errorJson(502, "billing status unavailable")),
        ),
      ),
    ),
  )
}
