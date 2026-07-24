import { timingSafeEqual } from "node:crypto"
import { Config, Context, Data, Effect, Layer, Redacted } from "effect"

/**
 * Stripe billing for studio — subscription checkout, verification, and the
 * billing portal, over Stripe's form-encoded REST API (no SDK dependency).
 *
 * Pricing model: first minute of hosted-API studio time is free (metered by
 * `entitlement.ts`), then a $20/month subscription (STRIPE_PRICE_ID) unlocks
 * unlimited use. Config resolves PER CALL (like DecartRealtime) so missing
 * Stripe env only fails the billing routes, not the whole runtime.
 */

export class StripeError extends Data.TaggedError("StripeError")<{
  readonly message: string
}> {}

export const BillingConfig = Config.all({
  secretKey: Config.redacted("STRIPE_SECRET_KEY"),
  priceId: Config.string("STRIPE_PRICE_ID"),
})

const DEV_SESSION_SECRET = "studio-dev-session-secret"

/** Are we in a deployed environment (as opposed to a laptop dev server)? */
export const isProductionEnv = (): boolean =>
  process.env["NODE_ENV"] === "production" || process.env["VERCEL"] === "1"

/**
 * Session-cookie HMAC secret. In production a missing SESSION_SECRET is FATAL:
 * the dev fallback is public knowledge (it's in this file), so running with it
 * would let anyone sign their own pro cookie (review finding, P0).
 */
export const SessionSecretConfig = Config.redacted("SESSION_SECRET").pipe(
  Config.withDefault(Redacted.make(DEV_SESSION_SECRET)),
  Config.map((secret) => {
    if (isProductionEnv() && Redacted.value(secret) === DEV_SESSION_SECRET) {
      throw new Error("SESSION_SECRET must be set in production — refusing to sign entitlement cookies with the public dev secret")
    }
    return secret
  }),
)

/**
 * Owner bypass: OWNER_KEY (no default — the endpoint 404s when unset) lets the
 * operator mint themselves an unlimited "owner" entitlement cookie via
 * GET /api/billing/owner?key=... — no Stripe subscription involved.
 */
export const OwnerKeyConfig = Config.redacted("OWNER_KEY")

/** Constant-time key comparison (length mismatch = false, never throws). */
export function ownerKeyMatches(candidate: string, actual: string): boolean {
  if (candidate.length === 0 || actual.length === 0) return false
  const a = Buffer.from(candidate)
  const b = Buffer.from(actual)
  if (a.length !== b.length) return false
  return timingSafeEqual(a, b)
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested directly)
// ---------------------------------------------------------------------------

export function checkoutSessionParams(priceId: string, origin: string, nonce?: string): URLSearchParams {
  const params = new URLSearchParams({
    mode: "subscription",
    "line_items[0][price]": priceId,
    "line_items[0][quantity]": "1",
    success_url: `${origin}/api/billing/confirm?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${origin}/studio?checkout=cancelled`,
    allow_promotion_codes: "true",
  })
  // Binds the eventual confirm redirect to the browser that started checkout:
  // confirm.ts compares this metadata to the studio_ckn cookie (review
  // finding: a leaked session_id must not grant someone else the sub cookie).
  if (nonce) params.set("metadata[confirm_nonce]", nonce)
  return params
}

export function isActiveSubscriptionStatus(status: string | undefined): boolean {
  return status === "active" || status === "trialing"
}

export interface CheckoutVerification {
  readonly customerId: string
  readonly subscriptionId: string
  readonly active: boolean
  readonly nonce?: string | undefined
}

/** Interpret a retrieved Checkout Session (subscription possibly expanded). */
export function parseCheckoutVerification(session: {
  payment_status?: string
  status?: string
  customer?: unknown
  subscription?: unknown
  metadata?: Record<string, string> | null
}): CheckoutVerification {
  const customerId = typeof session.customer === "string" ? session.customer : ""
  const sub = session.subscription
  const subscriptionId =
    typeof sub === "string" ? sub : typeof (sub as { id?: string })?.id === "string" ? (sub as { id: string }).id : ""
  const subStatus = typeof sub === "object" && sub !== null ? (sub as { status?: string }).status : undefined
  const paid = session.payment_status === "paid" && session.status === "complete"
  // Unexpanded string subscription on a paid session counts as active; the
  // token route re-verifies against /v1/subscriptions on its cadence anyway.
  const active = paid && (typeof sub === "string" ? true : isActiveSubscriptionStatus(subStatus))
  return { customerId, subscriptionId, active, nonce: session.metadata?.["confirm_nonce"] }
}

/** One form-encoded call to Stripe. Throws Error with Stripe's message on non-2xx. */
export async function stripeRequest(
  secretKey: string,
  method: "GET" | "POST",
  path: string,
  params?: URLSearchParams,
): Promise<unknown> {
  const url =
    method === "GET" && params ? `https://api.stripe.com/v1/${path}?${params}` : `https://api.stripe.com/v1/${path}`
  const res = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${secretKey}`,
      ...(method === "POST" ? { "Content-Type": "application/x-www-form-urlencoded" } : {}),
    },
    ...(method === "POST" ? { body: params?.toString() ?? "" } : {}),
  })
  const body = (await res.json().catch(() => ({}))) as { error?: { message?: string } }
  if (!res.ok) {
    throw new Error(body.error?.message ?? `stripe ${path} failed (${res.status})`)
  }
  return body
}

// ---------------------------------------------------------------------------
// Effect service
// ---------------------------------------------------------------------------

const withConfig = <A>(f: (cfg: { secretKey: string; priceId: string }) => Promise<A>) =>
  Effect.gen(function* () {
    const cfg = yield* BillingConfig.pipe(
      Effect.mapError(() => new StripeError({ message: "billing is not configured (set STRIPE_SECRET_KEY and STRIPE_PRICE_ID)" })),
    )
    return yield* Effect.tryPromise({
      try: () => f({ secretKey: Redacted.value(cfg.secretKey), priceId: cfg.priceId }),
      catch: (cause) => new StripeError({ message: cause instanceof Error ? cause.message : String(cause) }),
    })
  })

export class Billing extends Context.Service<Billing>()("studio/Billing", {
  make: Effect.succeed({
    /** Create a $20/mo subscription Checkout Session; returns the redirect URL. */
    createCheckoutSession: (origin: string, nonce?: string): Effect.Effect<string, StripeError> =>
      withConfig(async ({ secretKey, priceId }) => {
        const session = (await stripeRequest(
          secretKey,
          "POST",
          "checkout/sessions",
          checkoutSessionParams(priceId, origin, nonce),
        )) as { url?: string }
        if (!session.url) throw new Error("stripe returned no checkout url")
        return session.url
      }),

    /** Verify a completed Checkout Session (called from the success redirect). */
    verifyCheckout: (sessionId: string): Effect.Effect<CheckoutVerification, StripeError> =>
      withConfig(async ({ secretKey }) => {
        const session = await stripeRequest(
          secretKey,
          "GET",
          `checkout/sessions/${encodeURIComponent(sessionId)}`,
          new URLSearchParams({ "expand[]": "subscription" }),
        )
        return parseCheckoutVerification(session as Parameters<typeof parseCheckoutVerification>[0])
      }),

    /** Is this subscription currently entitled to the product? */
    subscriptionActive: (subscriptionId: string): Effect.Effect<boolean, StripeError> =>
      withConfig(async ({ secretKey }) => {
        const sub = (await stripeRequest(
          secretKey,
          "GET",
          `subscriptions/${encodeURIComponent(subscriptionId)}`,
        )) as { status?: string }
        return isActiveSubscriptionStatus(sub.status)
      }),

    /** Stripe-hosted billing portal for cancel/update; returns the redirect URL. */
    createPortalSession: (customerId: string, origin: string): Effect.Effect<string, StripeError> =>
      withConfig(async ({ secretKey }) => {
        const session = (await stripeRequest(
          secretKey,
          "POST",
          "billing_portal/sessions",
          new URLSearchParams({ customer: customerId, return_url: `${origin}/studio` }),
        )) as { url?: string }
        if (!session.url) throw new Error("stripe returned no portal url")
        return session.url
      }),
  } as const),
}) {
  static readonly layer = Layer.effect(this, this.make)
}
