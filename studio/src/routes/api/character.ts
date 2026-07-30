import type { APIEvent } from "@solidjs/start/server"
import { Effect } from "effect"
import { CharacterBody, generateCharacter } from "~/server/character"
import { decodeBody, errorJson, json } from "~/server/http"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Character generation route. The pipeline itself lives in server/character.ts
 * — SolidStart transforms route modules, so a non-handler export from a route
 * file is `undefined` at runtime (the same trap that broke CHALLENGE_COOKIE
 * before it moved to server/otp.ts). Routes here hold handlers and nothing
 * else.
 */
export async function POST(event: APIEvent): Promise<Response> {
  if (!rateLimit(`character:${clientIp(event.request)}`, 3, 60_000)) {
    console.warn("character: rate limited")
    return errorJson(429, "a new friend needs a moment — try again shortly")
  }

  const key = process.env["FAL_API_KEY"]
  if (!key) {
    console.error("character: FAL_API_KEY missing")
    return errorJson(503, "character creation is not available right now")
  }

  return runtime.runPromise(
    Effect.gen(function* () {
      const body = yield* decodeBody(event.request, CharacterBody)
      const sprite = yield* Effect.tryPromise({
        try: () => generateCharacter(body.prompt.trim(), key),
        catch: (e) => new Error(e instanceof Error ? e.message : String(e)),
      }).pipe(
        Effect.catch((e: Error) =>
          Effect.logWarning(`character generation failed: ${e.message}`).pipe(
            Effect.as<string | null>(null),
          ),
        ),
      )

      if (!sprite) return errorJson(502, "your friend got lost on the way — try again")
      yield* Effect.logInfo("character generated")
      return json({ sprite })
    }).pipe(
      Effect.catchTag("BadRequest", () =>
        Effect.logWarning("character: bad request body").pipe(
          Effect.as(errorJson(400, "describe your friend in a few words (3-120 characters)")),
        ),
      ),
    ),
  )
}
