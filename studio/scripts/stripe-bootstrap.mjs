#!/usr/bin/env node
/**
 * One-time Stripe setup for Studio Pro: creates the product and its $20/month
 * price, then prints the STRIPE_PRICE_ID line for .env.
 *
 * Usage: STRIPE_SECRET_KEY=sk_test_... node scripts/stripe-bootstrap.mjs
 * (or run from studio/ with STRIPE_SECRET_KEY already in .env)
 */
import { readFileSync } from "node:fs"

let key = process.env.STRIPE_SECRET_KEY
if (!key) {
  try {
    const env = readFileSync(new URL("../.env", import.meta.url), "utf8")
    key = env.match(/^STRIPE_SECRET_KEY=(.+)$/m)?.[1]?.trim()
  } catch {
    // no .env — fall through to the error below
  }
}
if (!key) {
  console.error("Set STRIPE_SECRET_KEY (env or studio/.env) first.")
  process.exit(1)
}

async function stripe(path, params) {
  const res = await fetch(`https://api.stripe.com/v1/${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: params.toString(),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(body.error?.message ?? `${path} failed (${res.status})`)
  return body
}

const product = await stripe(
  "products",
  new URLSearchParams({
    name: "Studio Pro",
    description: "Unlimited real-time video sessions in studio (hosted API engine).",
  }),
)
console.log(`product: ${product.id}`)

const price = await stripe(
  "prices",
  new URLSearchParams({
    product: product.id,
    currency: "usd",
    unit_amount: "2000",
    "recurring[interval]": "month",
    lookup_key: "studio_pro_monthly",
  }),
)
console.log(`price:   ${price.id}  ($20.00/month)`)
console.log("")
console.log("Add to studio/.env:")
console.log(`STRIPE_PRICE_ID=${price.id}`)
