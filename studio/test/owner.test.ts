import { describe, expect, it } from "vitest"
import { Effect, Layer } from "effect"
import { ownerKeyMatches, Billing } from "../src/server/billing"
import { OWNER_SUB, resolvePlan } from "../src/server/plan"
import { signState } from "../src/server/entitlement"

describe("owner key comparison", () => {
  it("matches only the exact key (constant-time)", () => {
    expect(ownerKeyMatches("abc123", "abc123")).toBe(true)
    expect(ownerKeyMatches("abc124", "abc123")).toBe(false)
    expect(ownerKeyMatches("abc12", "abc123")).toBe(false)
    expect(ownerKeyMatches("", "abc123")).toBe(false)
    expect(ownerKeyMatches("abc123", "")).toBe(false)
  })
})

describe("owner plan resolution", () => {
  it("treats an owner claim as pro WITHOUT consulting Stripe, even when stale", async () => {
    // ver=0 would force a Stripe re-verify for a normal sub; the poisoned
    // Billing layer proves the owner branch never touches it.
    const poisonedBilling = Layer.succeed(Billing, {
      createCheckoutSession: () => Effect.die("stripe must not be called"),
      verifyCheckout: () => Effect.die("stripe must not be called"),
      subscriptionActive: () => Effect.die("stripe must not be called"),
      createPortalSession: () => Effect.die("stripe must not be called"),
    } as unknown as Billing["Service"])

    const jwt = await signState({ cus: OWNER_SUB, sub: OWNER_SUB, ver: 0 }, "studio-dev-session-secret")
    const request = new Request("http://localhost/api/decart/token", {
      headers: { cookie: `studio_sub=${jwt}` },
    })
    const plan = await Effect.runPromise(
      resolvePlan(request).pipe(Effect.provide(poisonedBilling)),
    )
    expect(plan.plan).toBe("pro")
  })
})
