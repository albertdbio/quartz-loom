import "dotenv/config"
import { Layer, ManagedRuntime } from "effect"
import { Billing } from "./billing"
import { DecartRealtime } from "./decart"

/**
 * Server-only Effect runtime for studio: Decart realtime-token minting plus
 * Stripe billing. Both resolve their config PER CALL, so missing env
 * (DECART_API_KEY / STRIPE_SECRET_KEY / STRIPE_PRICE_ID) only fails the routes
 * that need it, not the whole runtime. `dotenv/config` loads `.env` in dev.
 */
const AppLayer = Layer.mergeAll(DecartRealtime.layer, Billing.layer)

export const runtime = ManagedRuntime.make(AppLayer)

/** Graceful shutdown: dispose the runtime (drains finalizers) before exit. */
const shutdown = (signal: NodeJS.Signals) => {
  void runtime.dispose().finally(() => {
    process.kill(process.pid, signal)
  })
}
process.once("SIGTERM", shutdown)
process.once("SIGINT", shutdown)
