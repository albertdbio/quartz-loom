import { Context, Data, Effect, Layer, Redacted } from "effect"
import { createDecartClient } from "@decartai/sdk"
import { DecartConfig } from "./config"

export class DecartError extends Data.TaggedError("DecartError")<{
  readonly cause: unknown
}> {}

/**
 * Decart realtime-token minting for Lucy 2.5 (real-time video-to-video editing).
 *
 * The browser cannot hold the permanent `DECART_API_KEY`, so this server-only
 * service exchanges it for a short-lived token (`client.tokens.create()`); the
 * client then does `createDecartClient({ apiKey: token })` +
 * `client.realtime.connect(cameraStream, { onRemoteStream })`. Config is
 * resolved PER CALL (not at layer build) so a missing `DECART_API_KEY` only
 * fails the /api/decart/token route, not the whole runtime.
 */
export class DecartRealtime extends Context.Service<DecartRealtime>()("studio/DecartRealtime", {
  make: Effect.succeed({
    /** Mint a fresh short-lived realtime token for one browser session. */
    mintToken: (): Effect.Effect<string, DecartError> =>
      Effect.gen(function* () {
        const cfg = yield* DecartConfig.pipe(
          Effect.mapError((cause) => new DecartError({ cause })),
        )
        return yield* Effect.tryPromise({
          try: async () => {
            const client = createDecartClient({ apiKey: Redacted.value(cfg.apiKey) })
            const token = await client.tokens.create()
            // The SDK returns the minted key as `apiKey`; the client connects with it.
            return token.apiKey
          },
          catch: (cause) => new DecartError({ cause }),
        })
      }),
  } as const),
}) {
  static readonly layer = Layer.effect(this, this.make)
}
