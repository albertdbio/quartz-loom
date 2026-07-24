import type { APIEvent } from "@solidjs/start/server"
import { Effect, Schema } from "effect"
import { decodeBody, errorJson, json } from "~/server/http"
import { clientIp, rateLimit } from "~/server/ratelimit"
import { runtime } from "~/server/runtime"

/**
 * Onboarding answers, for product insight only.
 *
 * Deliberately minimal: two enum answers, nothing else. No identifiers, no
 * device info, no free text — so this stays "Product Interaction, not linked
 * to you" on the App Store privacy label, and there is nothing here worth
 * breaching. Answers are logged, not stored: on a serverless deploy the log
 * IS the durable sink (upgrade to a real store when the volume justifies it).
 */
const ProfileBody = Schema.Struct({
  craft: Schema.Literals(["creator", "artist", "marketer", "educator", "fun"]),
  goal: Schema.Literals(["viral", "product", "art", "teach", "explore"]),
})

export async function POST(event: APIEvent): Promise<Response> {
  if (!rateLimit(`profile:${clientIp(event.request)}`, 10, 60_000)) {
    return errorJson(429, "too many submissions")
  }
  return runtime.runPromise(
    Effect.gen(function* () {
      const body = yield* decodeBody(event.request, ProfileBody)
      yield* Effect.logInfo(`onboarding craft=${body.craft} goal=${body.goal}`)
      return json({ ok: true })
    }).pipe(
      Effect.catchTag("BadRequest", () => Effect.succeed(errorJson(400, "unexpected answers"))),
    ),
  )
}
