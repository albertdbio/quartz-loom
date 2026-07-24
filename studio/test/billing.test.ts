import { afterEach, describe, expect, it, vi } from "vitest"
import {
  checkoutSessionParams,
  isActiveSubscriptionStatus,
  parseCheckoutVerification,
  stripeRequest,
} from "../src/server/billing"

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("checkout session params", () => {
  it("builds a subscription-mode checkout with success/cancel URLs on the caller's origin", () => {
    const p = checkoutSessionParams("price_123", "https://studio.example.com")
    expect(p.get("mode")).toBe("subscription")
    expect(p.get("line_items[0][price]")).toBe("price_123")
    expect(p.get("line_items[0][quantity]")).toBe("1")
    expect(p.get("success_url")).toBe(
      "https://studio.example.com/api/billing/confirm?session_id={CHECKOUT_SESSION_ID}",
    )
    expect(p.get("cancel_url")).toBe("https://studio.example.com/studio?checkout=cancelled")
  })

  it("binds a confirm nonce into checkout metadata (review: session_id replay)", () => {
    const p = checkoutSessionParams("price_123", "https://x.test", "nonce-abc")
    expect(p.get("metadata[confirm_nonce]")).toBe("nonce-abc")
    expect(checkoutSessionParams("price_123", "https://x.test").get("metadata[confirm_nonce]")).toBeNull()
  })
})

describe("subscription status", () => {
  it("accepts active and trialing, rejects everything else", () => {
    expect(isActiveSubscriptionStatus("active")).toBe(true)
    expect(isActiveSubscriptionStatus("trialing")).toBe(true)
    expect(isActiveSubscriptionStatus("past_due")).toBe(false)
    expect(isActiveSubscriptionStatus("canceled")).toBe(false)
    expect(isActiveSubscriptionStatus("incomplete")).toBe(false)
    expect(isActiveSubscriptionStatus(undefined)).toBe(false)
  })
})

describe("checkout verification parsing", () => {
  it("extracts customer + subscription from a paid session", () => {
    const v = parseCheckoutVerification({
      payment_status: "paid",
      status: "complete",
      customer: "cus_9",
      subscription: { id: "sub_7", status: "active" },
    })
    expect(v).toEqual({ customerId: "cus_9", subscriptionId: "sub_7", active: true })
  })

  it("handles a string subscription id (unexpanded)", () => {
    const v = parseCheckoutVerification({
      payment_status: "paid",
      status: "complete",
      customer: "cus_9",
      subscription: "sub_7",
    })
    expect(v).toEqual({ customerId: "cus_9", subscriptionId: "sub_7", active: true })
  })

  it("rejects an unpaid session", () => {
    const v = parseCheckoutVerification({
      payment_status: "unpaid",
      status: "open",
      customer: "cus_9",
      subscription: "sub_7",
    })
    expect(v.active).toBe(false)
  })

  it("rejects a paid session whose subscription is not active", () => {
    const v = parseCheckoutVerification({
      payment_status: "paid",
      status: "complete",
      customer: "cus_9",
      subscription: { id: "sub_7", status: "canceled" },
    })
    expect(v.active).toBe(false)
  })

  it("surfaces the confirm nonce from session metadata", () => {
    const v = parseCheckoutVerification({
      payment_status: "paid",
      status: "complete",
      customer: "cus_9",
      subscription: "sub_7",
      metadata: { confirm_nonce: "nonce-abc" },
    })
    expect(v.nonce).toBe("nonce-abc")
  })
})

describe("client ip for rate limiting (review: leftmost XFF is spoofable)", () => {
  it("uses the rightmost x-forwarded-for entry", async () => {
    const { clientIp } = await import("../src/server/ratelimit")
    const req = new Request("http://x.test", {
      headers: { "x-forwarded-for": "6.6.6.6, 7.7.7.7, 10.0.0.9" },
    })
    expect(clientIp(req)).toBe("10.0.0.9")
    expect(clientIp(new Request("http://x.test"))).toBe("local")
  })
})

describe("stripe request wire format", () => {
  it("POSTs form-encoded params with bearer auth and surfaces the parsed body", async () => {
    const seen: { url?: string; init?: RequestInit } = {}
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      seen.url = url
      seen.init = init
      return new Response(JSON.stringify({ id: "cs_1", url: "https://checkout.stripe.com/x" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    })
    const body = await stripeRequest("sk_test_x", "POST", "checkout/sessions", new URLSearchParams({ mode: "subscription" }))
    expect(seen.url).toBe("https://api.stripe.com/v1/checkout/sessions")
    expect((seen.init!.headers as Record<string, string>)["Authorization"]).toBe("Bearer sk_test_x")
    expect((seen.init!.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/x-www-form-urlencoded",
    )
    expect(seen.init!.body).toBe("mode=subscription")
    expect((body as { id: string }).id).toBe("cs_1")
  })

  it("throws a readable error from Stripe's error envelope", async () => {
    vi.stubGlobal("fetch", async () =>
      new Response(JSON.stringify({ error: { message: "No such price: price_nope" } }), { status: 400 }),
    )
    await expect(
      stripeRequest("sk_test_x", "POST", "checkout/sessions", new URLSearchParams()),
    ).rejects.toThrow(/No such price/)
  })

  it("GETs without a body", async () => {
    const seen: { init?: RequestInit } = {}
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      seen.init = init
      return new Response(JSON.stringify({ status: "active" }), { status: 200 })
    })
    await stripeRequest("sk_test_x", "GET", "subscriptions/sub_1")
    expect(seen.init!.method).toBe("GET")
    expect(seen.init!.body).toBeUndefined()
  })
})
