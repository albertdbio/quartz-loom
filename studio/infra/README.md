# Deploy

Production runs on a Hetzner **CPX11** (`wand-vm-01`, `hil`) behind Caddy for
automatic Let's Encrypt certs.

Caddy is already configured for `mochiverse.io` and `whatisthemochiverse.com`;
both issue certs automatically once their A records point at `5.78.89.215`.
Until then the box answers on `https://5-78-89-215.sslip.io`, which stays
configured so existing installs keep working through the switch.

## Why a box and not serverless

The app used to be stateless, which would have suited Vercel. It no longer is:
`server/subscribers.ts` and `server/usage.ts` keep the subscriber list and the
monthly usage ledger in a sqlite file. On a serverless platform that file is
wiped between invocations, taking every member's remaining balance with it —
so the budget cap would silently stop capping. A box with a disk is the correct
shape now.

It stays **separate from the mochi box** on purpose: mochi pays rent, and a
consumer camera toy has abuse-shaped traffic. A runaway day here should not be
able to take that down.

## Layout

- `wand.service` — systemd unit running the Nitro build as the `wand` user
- `/home/wand/app/.env` — production secrets, `chmod 600`
- `/home/wand/app/.data/subscribers.db` — the only precious state on the box
- Caddy listens on 80/443; the app binds loopback only; ufw allows 22/80/443

## Deploy a new version

```bash
ssh wand@5.78.89.215
cd ~/src && git pull --ff-only        # MUST fail loud, never warn-and-continue
cd studio && pnpm install && pnpm build
# REPLACE the tree — do not `cp -r` onto the old one. Asset filenames are
# content-hashed, so copying layers new bundles beside stale ones forever:
# the app still serves the right hash, but every grep-based artifact check
# (including the one below) reports the OLD code as present.
rm -rf ~/app/.output.new && cp -r ~/src/studio/.output ~/app/.output.new
rm -rf ~/app/.output.old && mv ~/app/.output ~/app/.output.old \
  && mv ~/app/.output.new ~/app/.output
sudo systemctl restart wand && rm -rf ~/app/.output.old
```

Then verify the ARTIFACT changed, not just that the command exited 0:

```bash
ssh wand@5.78.89.215 'cd ~/src && git rev-parse HEAD'   # == origin/main
curl -s https://5-78-89-215.sslip.io/api/billing/status  # answers
grep -rho "<a string only the new code emits>" ~/app/.output/ | wc -l
```

## Custom domain

Point an A record at `5.78.89.215`, add the name to `/etc/caddy/Caddyfile`
alongside the sslip.io host, and `systemctl reload caddy` — Caddy issues the
cert automatically. Update `PUBLIC_BASE_URL` in `.env` and restart, since
Stripe redirect URLs are built from it.
