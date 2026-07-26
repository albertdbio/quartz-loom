import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted, Schema } from "effect"
import { SessionSecretConfig } from "~/server/billing"
import { uidFromRequest } from "~/server/entitlement"
import { decodeBody, errorJson, json } from "~/server/http"
import { recordPushToken } from "~/server/push"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Device registration for push, called by the native shell once the user has
 * accepted the OS prompt.
 *
 * Identity comes from the signed cookie, never from the body — a device that
 * could name its own owner could subscribe itself to someone else's
 * notifications. An unauthenticated register is still accepted and stored
 * unlinked, because the prompt can legitimately be answered before sign-in.
 */
const RegisterBody = Schema.Struct({
  token: Schema.String.check(Schema.isMinLength(8), Schema.isMaxLength(512)),
  platform: Schema.Literals(["ios", "android"]),
})

export async function POST(event: APIEvent): Promise<Response> {
  if (!rateLimit(`push:${clientIp(event.request)}`, 10, 60_000)) {
    return errorJson(429, "too many registrations")
  }

  return runtime.runPromise(
    Effect.gen(function* () {
      const body = yield* decodeBody(event.request, RegisterBody)
      // Same accessor as the other cookie routes: a dev default locally, and
      // a hard failure in production if the real secret is missing.
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const uid = yield* Effect.promise(() => uidFromRequest(event.request, secret))

      yield* Effect.sync(() =>
        recordPushToken({
          token: body.token,
          platform: body.platform,
          phoneId: uid?.pid ?? null,
        }),
      )

      yield* Effect.logInfo(`push token registered platform=${body.platform} linked=${uid !== null}`)
      return json({ ok: true, linked: uid !== null })
    }).pipe(
      Effect.catchTag("BadRequest", () => Effect.succeed(errorJson(400, "unexpected registration"))),
    ),
  )
}
