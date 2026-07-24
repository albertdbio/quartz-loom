import type { APIEvent } from "@solidjs/start/server"
import { Effect, Redacted } from "effect"
import { OwnerKeyConfig, ownerKeyMatches, SessionSecretConfig } from "~/server/billing"
import { cookieHeader, requestIsSecure, signState, SUB_COOKIE } from "~/server/entitlement"
import { OWNER_SUB } from "~/server/plan"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/** Owner cookies last a year — re-visit the redeem URL to renew. */
const OWNER_MAX_AGE = 365 * 24 * 60 * 60

const notFound = (): Response => new Response("not found", { status: 404 })

/**
 * Operator self-service: GET /api/billing/owner?key=<OWNER_KEY> plants a
 * signed owner entitlement cookie (unlimited sessions, no Stripe) on this
 * browser, then lands in the wand. 404 on a wrong/missing key OR when
 * OWNER_KEY isn't configured — the endpoint is indistinguishable from absent.
 */
export async function GET(event: APIEvent): Promise<Response> {
  const ip = clientIp(event.request)
  if (!rateLimit(`owner:${ip}`, 5, 60_000)) return notFound()
  const candidate = new URL(event.request.url).searchParams.get("key") ?? ""
  return runtime.runPromise(
    Effect.gen(function* () {
      const configured = yield* OwnerKeyConfig.pipe(
        Effect.map((k) => Redacted.value(k)),
        Effect.catch(() => Effect.succeed("")),
      )
      if (!ownerKeyMatches(candidate, configured)) return notFound()
      const secret = Redacted.value(yield* SessionSecretConfig.pipe(Effect.orDie))
      const jwt = yield* Effect.promise(() =>
        signState(
          { cus: OWNER_SUB, sub: OWNER_SUB, ver: Math.floor(Date.now() / 1000) },
          secret,
          OWNER_MAX_AGE,
        ),
      )
      const headers = new Headers({ Location: "/wand?owner=1" })
      headers.append(
        "Set-Cookie",
        cookieHeader(SUB_COOKIE, jwt, OWNER_MAX_AGE, requestIsSecure(event.request)),
      )
      return new Response(null, { status: 302, headers })
    }),
  )
}
