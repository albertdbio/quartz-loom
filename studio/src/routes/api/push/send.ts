import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted, Schema } from "effect"
import { apnsConfigFromEnv, sendPush } from "~/server/apns"
import { OwnerKeyConfig, ownerKeyMatches } from "~/server/billing"
import { decodeBody, errorJson, json } from "~/server/http"
import { forgetPushToken, tokensFor, allPushTokens } from "~/server/push"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Operator push console — sends real APNs alerts.
 *
 * Gated by the owner key in a header, same posture as /api/billing/owner:
 * wrong or missing key answers 404 so the endpoint is indistinguishable from
 * absent, and attempts are rate limited. This is an operator tool and a
 * diagnostic — the response carries APNs' own answer per token, so "is the
 * key right / is the token alive" is answerable with one curl.
 *
 * audience:
 * - "all"      — every registered device (the monthly-reset broadcast)
 * - a phoneId  — every device that person signed in on
 * - a raw hex  — one specific device token (smoke tests)
 */
const SendBody = Schema.Struct({
  audience: Schema.String.check(Schema.isMinLength(3), Schema.isMaxLength(200)),
  title: Schema.String.check(Schema.isMinLength(1), Schema.isMaxLength(80)),
  body: Schema.String.check(Schema.isMinLength(1), Schema.isMaxLength(200)),
})

const notFound = (): Response => new Response("not found", { status: 404 })

export async function POST(event: APIEvent): Promise<Response> {
  if (!rateLimit(`pushsend:${clientIp(event.request)}`, 5, 60_000)) return notFound()
  const candidate = event.request.headers.get("x-owner-key") ?? ""

  return runtime.runPromise(
    Effect.gen(function* () {
      const configured = yield* OwnerKeyConfig.pipe(
        Effect.map((k) => Redacted.value(k)),
        Effect.catch(() => Effect.succeed("")),
      )
      if (!ownerKeyMatches(candidate, configured)) return notFound()

      const cfg = apnsConfigFromEnv()
      if (!cfg) {
        return errorJson(503, "APNs is not configured — set APNS_KEY_PATH and APNS_KEY_ID")
      }

      const req = yield* decodeBody(event.request, SendBody)
      const targets =
        req.audience === "all"
          ? yield* Effect.sync(() => allPushTokens())
          : /^[0-9a-f]{64,200}$/i.test(req.audience)
            ? [{ token: req.audience, platform: "ios" as const }]
            : yield* Effect.sync(() => [...tokensFor(req.audience)])

      if (targets.length === 0) return json({ sent: 0, results: [], note: "no tokens for audience" })

      const results: Array<{ token: string; outcome: string; status: number; reason?: string }> = []
      for (const t of targets) {
        const r = yield* Effect.promise(() => sendPush(cfg, t.token, req.title, req.body))
        if (r.outcome === "forget-token") {
          yield* Effect.sync(() => forgetPushToken(t.token))
        }
        // token prefix only: enough to correlate, useless to replay
        results.push({
          token: `${t.token.slice(0, 8)}…`,
          outcome: r.outcome,
          status: r.status,
          ...(r.reason !== undefined && { reason: r.reason }),
        })
      }

      const delivered = results.filter((r) => r.outcome === "delivered").length
      yield* Effect.logInfo(`push send: ${delivered}/${results.length} delivered`)
      return json({ sent: delivered, results })
    }).pipe(
      Effect.catchTag("BadRequest", () =>
        Effect.succeed(errorJson(400, "audience, title (<=80) and body (<=200) required")),
      ),
    ),
  )
}
