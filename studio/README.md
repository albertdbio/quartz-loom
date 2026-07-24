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

### Owner bypass (unlimited, no Stripe)

Set `OWNER_KEY` in `.env` (`openssl rand -hex 24`), then visit once per
browser/device:

```
https://<host>/api/billing/owner?key=<OWNER_KEY>
```

It plants a signed `owner` entitlement cookie (1 year, renewable by
re-visiting) and redirects into the wand. Owner claims are honored **without
any Stripe call** — there's no subscription behind them — and are HMAC-signed
and expiry-bound like every other claim. A wrong key, or an unset `OWNER_KEY`,
returns a plain 404 so the endpoint is indistinguishable from absent; attempts
are IP rate-limited (5/min) and the comparison is constant-time.

For the native app, point that personal build's `EXPO_PUBLIC_WAND_URL` at the
redeem URL — the 302 lands on `/wand` and seeds the app's own cookie jar.
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
permanent Decart API key server-side and monitor session creation and duration
— each hosted realtime session is paid usage at **$0.04 per second**, so the
$20/month price assumes typical creative-session volumes, not 24/7 streaming;
add hard monthly caps before scaling.
