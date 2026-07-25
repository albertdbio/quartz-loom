import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted, Schema } from "effect"
import { SessionSecretConfig } from "~/server/billing"
import { cookieHeader, requestIsSecure } from "~/server/entitlement"
import { decodeBody, errorJson } from "~/server/http"
import { CHALLENGE_TTL_SECONDS, generateCode, normalizePhone, sealChallenge } from "~/server/otp"
import { jsonWithCookies } from "~/server/plan"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"
import { otpMessage, Sms } from "~/server/sms"

export const CHALLENGE_COOKIE = "studio_otp"

const Body = Schema.Struct({ phone: Schema.String })

/**
 * Begin phone sign-in: text a code and hand back an opaque challenge cookie.
 *
 * Two rate limits, because they stop different attacks: per-IP caps someone
 * enumerating many numbers from one machine, per-number caps someone using
 * many machines to SMS-bomb one victim (every send costs real money, so this
 * is a spend control as much as an abuse control).
 */
export async function POST(event: APIEvent): Promise<Response> {
  const ip = clientIp(event.request)
  if (!rateLimit(`otp-ip:${ip}`, 5, 10 * 60_000)) {
    return errorJson(429, "too many codes requested — try again in a few minutes")
  }
  return runtime.runPromise(
    Effect.gen(function* () {
      const body = yield* decodeBody(event.request, Body)
      const phone = normalizePhone(body.phone)
      if (!phone) return errorJson(400, "that doesn't look like a phone number")
      if (!rateLimit(`otp-num:${phone}`, 3, 10 * 60_000)) {
        return errorJson(429, "too many codes sent to that number — try again shortly")
      }

      const code = generateCode()
      const sms = yield* Sms
      yield* sms.send(phone, otpMessage(code))

      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const challenge = yield* Effect.promise(() => sealChallenge({ phone, code }, secret))
      return jsonWithCookies({ sent: true }, [
        cookieHeader(CHALLENGE_COOKIE, challenge, CHALLENGE_TTL_SECONDS, requestIsSecure(event.request)),
      ])
    }).pipe(
      Effect.catchTag("BadRequest", () => Effect.succeed(errorJson(400, "expected a phone number"))),
      Effect.catchTag("SmsError", (e) =>
        Effect.logError(`otp send failed: ${String(e.cause)}`).pipe(
          Effect.as(errorJson(502, "could not send your code — try again shortly")),
        ),
      ),
    ),
  )
}
