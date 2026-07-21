import { Config, Redacted } from "effect"

/**
 * Configuration read from the environment via Effect's default ConfigProvider.
 * Secrets are `Config.redacted` so they can't be logged by accident.
 *
 * Decart platform credentials — used ONLY server-side to mint short-lived
 * realtime tokens for the browser (Lucy 2.5 real-time video-to-video editing).
 * The permanent key never reaches the client; the browser connects with the
 * minted `token.apiKey`.
 */
export const DecartConfig = Config.all({
  apiKey: Config.redacted("DECART_API_KEY"),
  model: Config.string("DECART_REALTIME_MODEL").pipe(Config.withDefault("lucy-2.5")),
})

export { Redacted }
