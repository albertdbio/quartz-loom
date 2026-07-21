import "dotenv/config"
import { Layer, ManagedRuntime } from "effect"
import { DecartRealtime } from "./decart"

/**
 * Server-only Effect runtime for studio. Only the Decart realtime-token service
 * is needed; a missing DECART_API_KEY surfaces as a ConfigError when the
 * /api/decart/token route is first used. `dotenv/config` loads `.env` in dev.
 */
const AppLayer = Layer.mergeAll(DecartRealtime.layer)

export const runtime = ManagedRuntime.make(AppLayer)

/** Graceful shutdown: dispose the runtime (drains finalizers) before exit. */
const shutdown = (signal: NodeJS.Signals) => {
  void runtime.dispose().finally(() => {
    process.kill(process.pid, signal)
  })
}
process.once("SIGTERM", shutdown)
process.once("SIGINT", shutdown)
