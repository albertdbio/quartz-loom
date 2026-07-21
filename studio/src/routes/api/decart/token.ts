import type { APIEvent } from "@solidjs/start/server"
import { Effect } from "effect"
import { DecartRealtime } from "~/server/decart"
import { errorJson, json } from "~/server/http"
import { rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Mint a short-lived Decart realtime token for one browser studio session.
 * The permanent DECART_API_KEY stays server-side; the client connects with the
 * returned `apiKey`.
 *
 * Each mint enables a paid realtime session ($0.04/s), so we cap starts per IP.
 * For a public/multi-instance deploy, add real auth and swap the in-memory
 * limiter for a shared store (see server/ratelimit.ts).
 */
export async function POST(event: APIEvent): Promise<Response> {
  const ip = event.request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "local"
  if (!rateLimit(`token:${ip}`, 12, 60_000)) {
    return errorJson(429, "too many studio sessions from your address — try again in a minute")
  }
  return runtime.runPromise(
    Effect.gen(function* () {
      const decart = yield* DecartRealtime
      const apiKey = yield* decart.mintToken()
      return json({ apiKey })
    }).pipe(
      Effect.catch((e) =>
        Effect.logError(`decart token mint failed: ${String(e)}`).pipe(
          Effect.as(errorJson(502, "could not create studio session")),
        ),
      ),
    ),
  )
}
