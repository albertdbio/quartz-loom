import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted } from "effect"
import { SessionSecretConfig } from "~/server/billing"
import { DecartRealtime } from "~/server/decart"
import {
  cookieHeader,
  FREE_SECONDS,
  METER_COOKIE,
  METER_MAX_AGE,
  requestIsSecure,
  signState,
} from "~/server/entitlement"
import { errorJson } from "~/server/http"
import { jsonWithCookies, resolvePlan } from "~/server/plan"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Mint a short-lived Decart realtime token for one browser studio session.
 * The permanent DECART_API_KEY stays server-side; the client connects with the
 * returned `apiKey`.
 *
 * BILLING GATE — each mint enables a paid realtime session ($0.04/s):
 * - Pro (signed `studio_sub` cookie, re-verified against Stripe on a cadence):
 *   unlimited mints.
 * - Free: exactly one session; the mint itself burns the FREE_SECONDS meter
 *   (server-authoritative — no client heartbeat to game) and the client
 *   auto-stops at the minute. Further mints return 402 with `paywall: true`
 *   until checkout completes.
 */
export async function POST(event: APIEvent): Promise<Response> {
  const ip = clientIp(event.request)
  if (!rateLimit(`token:${ip}`, 12, 60_000)) {
    return errorJson(429, "too many studio sessions from your address — try again in a minute")
  }
  return runtime.runPromise(
    Effect.gen(function* () {
      const plan = yield* resolvePlan(event.request)
      const decart = yield* DecartRealtime

      if (plan.plan === "pro") {
        const apiKey = yield* decart.mintToken()
        return jsonWithCookies({ apiKey, plan: "pro" }, plan.setCookies)
      }

      if (plan.remaining <= 0) {
        return jsonWithCookies(
          {
            error: "your free minute is used up — Studio Pro is $20/month for unlimited sessions",
            paywall: true,
          },
          plan.setCookies,
          402,
        )
      }

      // Concurrency guard: N parallel first-mints share one unburned cookie
      // jar, so the burn below can't stop them alone (review finding, P0).
      // One free mint per IP per minute collapses that race; pro is unaffected.
      if (!rateLimit(`freemint:${ip}`, 1, 60_000)) {
        return errorJson(429, "one free session per minute — or go unlimited with Studio Pro")
      }

      // Burn the free minute at mint time, then hand out the token.
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const burned = yield* Effect.promise(() => signState({ used: FREE_SECONDS }, secret, METER_MAX_AGE))
      const apiKey = yield* decart.mintToken()
      return jsonWithCookies(
        { apiKey, plan: "free", freeSeconds: plan.remaining },
        [...plan.setCookies, cookieHeader(METER_COOKIE, burned, METER_MAX_AGE, requestIsSecure(event.request))],
      )
    }).pipe(
      Effect.catchTag("StripeError", (e) =>
        Effect.logError(`entitlement check failed: ${e.message}`).pipe(
          Effect.as(errorJson(502, "could not verify your plan — try again shortly")),
        ),
      ),
      Effect.catchTag("DecartError", (e) =>
        Effect.logError(`decart token mint failed: ${String(e.cause)}`).pipe(
          Effect.as(errorJson(502, "could not create studio session")),
        ),
      ),
    ),
  )
}
