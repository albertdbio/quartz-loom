import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted, Schema } from "effect"
import { SessionSecretConfig } from "~/server/billing"
import {
  cookieHeader,
  parseCookies,
  requestIsSecure,
  signState,
  UID_COOKIE,
  UID_MAX_AGE,
} from "~/server/entitlement"
import { decodeBody, errorJson } from "~/server/http"
import { codeMatches, openChallenge, phoneId } from "~/server/otp"
import { jsonWithCookies } from "~/server/plan"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"
import { recordSignIn } from "~/server/subscribers"
import { CHALLENGE_COOKIE } from "../sms/start"

const Body = Schema.Struct({
  code: Schema.String,
  marketingConsent: Schema.optional(Schema.Boolean),
})

/**
 * Complete phone sign-in.
 *
 * Attempts are capped per IP because the challenge cookie itself can't hold a
 * trustworthy counter — a client could simply replay an older copy to reset it.
 * A 6-digit code with a 10-minute life and 6 tries is the practical guard;
 * a shared store (Redis) is the upgrade when this runs on more than one box.
 */
export async function POST(event: APIEvent): Promise<Response> {
  const ip = clientIp(event.request)
  if (!rateLimit(`otp-verify:${ip}`, 6, 10 * 60_000)) {
    return errorJson(429, "too many attempts — request a new code")
  }
  return runtime.runPromise(
    Effect.gen(function* () {
      const body = yield* decodeBody(event.request, Body)
      const jar = parseCookies(event.request.headers.get("cookie"))
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))

      const challenge = yield* Effect.promise(() => openChallenge(jar[CHALLENGE_COOKIE] ?? "", secret))
      if (!challenge) return errorJson(400, "that code expired — request a new one")
      if (!codeMatches(body.code.trim(), challenge.code)) {
        return errorJson(401, "that code isn't right")
      }

      const pid = yield* Effect.promise(() => phoneId(challenge.phone, secret))
      const consent = body.marketingConsent === true
      yield* Effect.try({
        try: () => recordSignIn({ phoneId: pid, phone: challenge.phone, marketingConsent: consent }),
        catch: (cause) => cause,
      }).pipe(
        // The subscriber list is for engagement, not entitlement — never fail a
        // valid sign-in because the list write had a problem.
        Effect.catch((cause) => Effect.logError(`subscriber write failed: ${String(cause)}`)),
      )

      const uid = yield* Effect.promise(() => signState({ pid }, secret, UID_MAX_AGE))
      const secure = requestIsSecure(event.request)
      return jsonWithCookies({ signedIn: true }, [
        cookieHeader(UID_COOKIE, uid, UID_MAX_AGE, secure),
        cookieHeader(CHALLENGE_COOKIE, "", 0, secure),
      ])
    }).pipe(Effect.catchTag("BadRequest", () => Effect.succeed(errorJson(400, "expected a code")))),
  )
}
