# Studio

Studio is a real-time video editing app powered by Decart's Lucy 2.5 model. Point a camera at a scene, describe an edit, and watch the transformed video stream live.

It was cloned from opentxt's SolidStart + Effect skeleton and adapted for Decart's real-time video-to-video API.

## Run locally

Studio requires Node.js 22.5 or newer and pnpm.

```bash
pnpm install
```

Create `studio/.env` with a Decart API key:

```dotenv
DECART_API_KEY=<your Decart API key>
```

Start the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

## Architecture

The permanent Decart API key stays on the server:

1. The browser calls `POST /api/decart/token`.
2. The server-side Effect service exchanges `DECART_API_KEY` for a short-lived Decart token and returns it as `{ "apiKey": "..." }`.
3. The browser creates an `@decartai/sdk` client and sends the camera stream over WebRTC with `client.realtime.connect(cameraStream, { onRemoteStream })`.
4. `onRemoteStream` receives Lucy 2.5's live edited stream for playback in the browser.
5. Prompt changes are applied without reconnecting through `realtimeClient.set({ prompt })`.

## Billing — first minute free, then $20/month

The hosted-API path (Lucy 2.5 and story mode's `api` engine) is metered:

- **Free**: one session, capped at 60 seconds. The `/api/decart/token` mint
  burns the free minute server-side (signed `studio_meter` cookie), and the
  client counts down and auto-stops at 0.
- **Studio Pro — $20/month**: unlimited sessions via a Stripe subscription.
  Checkout (`POST /api/billing/checkout`) → Stripe-hosted payment →
  `GET /api/billing/confirm` verifies the session server-side and sets the
  signed `studio_sub` cookie. The token route re-verifies the subscription
  against Stripe every 6 hours (fail-open for existing claims if Stripe is
  unreachable; bounded by the cookie's 35-day expiry). `POST /api/billing/portal`
  opens the Stripe billing portal for cancel/update.
- The **self-hosted open-model mode stays free** — it's your GPU.

Setup:

```bash
# 1. keys in studio/.env (see .env.example): STRIPE_SECRET_KEY, SESSION_SECRET
# 2. create the product + $20/mo price:
node scripts/stripe-bootstrap.mjs   # prints STRIPE_PRICE_ID= for .env
```

## Phone sign-in (the MVP gate)

`PAYMENTS_ENABLED` decides what a spent free minute costs you:

| flag | anonymous | after the free minute |
|---|---|---|
| `false` (MVP default) | 60s free | **sign in with a phone number → unlimited** |
| `true` | 60s free | Stripe paywall, $20/month |

Entitlement ladder: **owner → pro (Stripe) → member (phone-verified) → anonymous**.
Flipping the flag needs no code change; the Stripe paths stay built and tested
either way.

### What a member gets

`MEMBER_MONTHLY_SECONDS` (default 1200 = 20 min) is drawn down `SESSION_SECONDS`
(default 120) at a time. The budget is spent **at grant time**, exactly like the
anonymous free minute — the mint is the metering event, so closing the tab or
killing the app can't buy free seconds back. The last grant of a month is
partial rather than refused, and quotas reset on the 1st (UTC).

Uncapped members were the single largest exposure with payments off: the engine
bills per second of active generation, so "unlimited" meant an unbounded bill
with no revenue against it.

Mochiverse runs **`lucy-restyle-2`**, not `lucy-2.5` — same vendor and quality
tier, literally the "realtime style transfer" model, at half the per-second
price. Verify any model swap against the pricing page: the realtime and batch
rates differ, and the batch number is the one that looks authoritative.

### How the login works

1. `POST /api/auth/sms/start` `{phone}` → normalizes to E.164, texts a 6-digit
   code, and sets `studio_otp` — an **encrypted (JWE/A256GCM)** cookie.
2. `POST /api/auth/sms/verify` `{code, marketingConsent}` → sets `studio_uid`
   (a signed cookie holding an HMAC pseudonym, never the number) and records
   the subscriber.
3. `POST /api/auth/logout` → drops `studio_uid`.

**Why the challenge cookie is encrypted rather than signed** — the one thing
not to "simplify" later: that cookie lands in the browser of whoever *started*
the login, who may have typed someone else's number. A signed-but-readable
payload would hand them the code that was texted to the victim (or a 6-digit
hash they could brute-force offline in milliseconds). `test/otp.test.ts`
asserts the code and phone never appear in the token or its base64 segments.

Verify attempts are capped per IP, because a counter inside the cookie can't be
trusted — a client can replay an older copy to reset it. A shared store (Redis)
is the upgrade when this runs on more than one box.

### Twilio setup — reuse mochi's brand AND campaign

The account's A2P registration already covers this app; **do not register a new
campaign**:

- Brand: `APPROVED` / `VETTED_VERIFIED`
- Campaign `QE2c6890da8086d771620e9b13fadeba0b` on messaging service
  `MGe0a8243a58c015290a6d6a308e2675b2`: **MIXED**, `VERIFIED`, described as
  "one-time passcodes, notifications, updates", with the registered sample
  **"Your one-time passcode is 123456."** — our OTP is exactly that content.

What this app needs is its **own sender number attached to that same campaign**.
Both existing numbers are already answering as mochi personas and their inbound
webhooks point at mochi's agent, so sharing one would route a Mochiverse user's
"STOP" into a conversation with a different product. This app is outbound-only
(the code is typed into the app, never replied to), so it needs no inbound
webhook at all — Twilio answers STOP/HELP on the number itself.

Set `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY_SID`, `TWILIO_API_KEY_SECRET` (same
account as mochi/core) and `TWILIO_FROM_NUMBER` (the new number).

### The engagement list

`server/subscribers.ts` keeps one `node:sqlite` file (`SUBSCRIBERS_DB`): a row
per signed-in pseudonym, and the **dialable number only while marketing consent
stands** — withdrawing consent nulls it, so the store can't outlive permission.
The consent box is unticked, labelled "not required to sign in", names the
sender, and states frequency plus HELP/STOP (CTIA).

Before any promotional send, update the campaign's description/samples to cover
marketing content — today they describe only passcodes/notifications/updates,
and carriers audit traffic against what's registered.

### Owner bypass (unlimited, no Stripe)

Set `OWNER_KEY` in `.env` (`openssl rand -hex 24`), then visit once per
browser/device:

```
https://<host>/api/billing/owner?key=<OWNER_KEY>
```

It plants a signed `owner` entitlement cookie (1 year, renewable by
re-visiting) and redirects into the app. Owner claims are honored **without
any Stripe call** — there's no subscription behind them — and are HMAC-signed
and expiry-bound like every other claim. A wrong key, or an unset `OWNER_KEY`,
returns a plain 404 so the endpoint is indistinguishable from absent; attempts
are IP rate-limited (5/min) and the comparison is constant-time.

For the native app, point that personal build's `EXPO_PUBLIC_APP_URL` at the
redeem URL — the 302 lands on `/` and seeds the app's own cookie jar.
Never ship a store build with the key baked in.

State is carried entirely in HMAC-signed cookies (`jose`, HS256) — no
database. Known v1 trade-offs, accepted deliberately: clearing cookies resets
the free minute (the mint-burns-the-minute design still bounds abuse to one
minute per cookie jar per mint, and the token route stays IP rate-limited),
and entitlement is per-browser rather than per-account (a subscriber's other
devices need their own checkout-confirm hop or a fresh checkout; Stripe
de-duplicates the customer by card/email). Upgrade path: real accounts +
Stripe webhooks + a shared store.

## Security and cost

The token endpoint is IP rate-limited AND billing-gated (above). Keep the
permanent Decart API key server-side and monitor session creation and duration.

**Cost of a session (Decart pricing, verified 2026-07):** the *realtime* API
bills per second of active generation — `lucy-2.5` (what we ship) at
**$0.02/sec = $1.20/min**, `lucy-restyle-2` at **$0.01/sec = $0.60/min**. The
$0.04/sec figure is the *batch video* rate and does NOT apply to this app;
an earlier version of this file quoted it and understated nothing — it
overstated cost by 2×, but the correction matters in both directions.

**The $20/month plan is therefore capacity-limited, not unlimited-safe:** at
$1.20/min it goes underwater past **~16 minutes/user/month**, and the free
minute costs **$1.20 per person who taps begin**. Set a monthly minute cap and
watch the free-tier burn before any launch that could go viral.
