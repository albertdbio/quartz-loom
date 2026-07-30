import { createPrivateKey, type KeyObject } from "node:crypto"
import { readFileSync } from "node:fs"
import http2 from "node:http2"
import { SignJWT } from "jose"

/**
 * Direct APNs, HTTP/2 with token auth — our box talks to Apple, no relay.
 *
 * Config (all server env):
 * - APNS_KEY_PATH — the .p8 auth key from the developer portal (chmod 600)
 * - APNS_KEY_ID   — the 10-char id in the key's filename
 * - APPLE_TEAM_ID — defaults to the team this app ships under
 * - APNS_TOPIC    — defaults to the bundle id
 *
 * Dev-signed installs (aps-environment: development) live on the SANDBOX
 * gateway and TestFlight/App Store installs on production, and the same
 * account will hold both kinds of token during development. Rather than
 * tracking which environment minted each token, a send tries production
 * first and retries the sandbox on BadDeviceToken — the standard trick for
 * a mixed fleet, at the cost of one extra round trip for dev devices.
 */

export interface ApnsConfig {
  readonly keyPath: string
  readonly keyId: string
  readonly teamId: string
  readonly topic: string
}

export function apnsConfigFromEnv(env: NodeJS.ProcessEnv = process.env): ApnsConfig | null {
  const keyPath = env["APNS_KEY_PATH"]
  const keyId = env["APNS_KEY_ID"]
  if (!keyPath || !keyId) return null
  return {
    keyPath,
    keyId,
    teamId: env["APPLE_TEAM_ID"] ?? "THRN2A465Z",
    topic: env["APNS_TOPIC"] ?? "io.mochiverse.app",
  }
}

/** The alert payload we send. Pure — pinned by tests. */
export function apnsPayload(title: string, body: string): string {
  return JSON.stringify({ aps: { alert: { title, body }, sound: "default" } })
}

export type ApnsOutcome = "delivered" | "retry-sandbox" | "forget-token" | "failed"

/**
 * What a gateway's answer means for the token. Pure — pinned by tests.
 *
 * 410 Unregistered is APNs saying "this device uninstalled you": keeping the
 * token would mean pushing to ghosts forever. 400 BadDeviceToken from
 * PRODUCTION usually means a sandbox token (dev build), so the caller should
 * retry there; the SAME answer from the sandbox means the token is garbage.
 */
export function classifyApnsAnswer(
  status: number,
  reason: string | undefined,
  gateway: "production" | "sandbox",
): ApnsOutcome {
  if (status === 200) return "delivered"
  if (status === 410) return "forget-token"
  if (status === 400 && reason === "BadDeviceToken") {
    return gateway === "production" ? "retry-sandbox" : "forget-token"
  }
  return "failed"
}

// -- the wire ---------------------------------------------------------------- //

let cachedJwt: { value: string; at: number; keyId: string } | null = null
let cachedKey: { object: KeyObject; path: string } | null = null

/** APNs rejects provider tokens older than an hour; refresh at 45 minutes. */
const JWT_LIFETIME_MS = 45 * 60 * 1000

async function providerJwt(cfg: ApnsConfig): Promise<string> {
  const now = Date.now()
  if (cachedJwt && cachedJwt.keyId === cfg.keyId && now - cachedJwt.at < JWT_LIFETIME_MS) {
    return cachedJwt.value
  }
  if (!cachedKey || cachedKey.path !== cfg.keyPath) {
    cachedKey = { object: createPrivateKey(readFileSync(cfg.keyPath, "utf8")), path: cfg.keyPath }
  }
  const value = await new SignJWT({})
    .setProtectedHeader({ alg: "ES256", kid: cfg.keyId })
    .setIssuedAt()
    .setIssuer(cfg.teamId)
    .sign(cachedKey.object)
  cachedJwt = { value, at: now, keyId: cfg.keyId }
  return value
}

function postToGateway(
  host: string,
  token: string,
  jwt: string,
  topic: string,
  payload: string,
): Promise<{ status: number; reason?: string | undefined }> {
  return new Promise((resolve) => {
    const client = http2.connect(`https://${host}`)
    const bail = (msg: string) => {
      client.close()
      resolve({ status: 0, reason: msg })
    }
    client.on("error", (e) => bail(`connect: ${e.message}`))
    const req = client.request({
      ":method": "POST",
      ":path": `/3/device/${token}`,
      authorization: `bearer ${jwt}`,
      "apns-topic": topic,
      "apns-push-type": "alert",
      "apns-priority": "10",
    })
    req.setTimeout(10_000, () => bail("timeout"))
    let status = 0
    let body = ""
    req.on("response", (h) => {
      status = Number(h[":status"] ?? 0)
    })
    req.on("data", (c: Buffer) => (body += c.toString()))
    req.on("error", (e) => bail(`stream: ${e.message}`))
    req.on("end", () => {
      client.close()
      let reason: string | undefined
      try {
        reason = body ? (JSON.parse(body) as { reason?: string }).reason : undefined
      } catch {
        reason = body || undefined
      }
      resolve({ status, reason })
    })
    req.end(payload)
  })
}

export interface SendResult {
  readonly outcome: ApnsOutcome
  readonly gateway: "production" | "sandbox"
  readonly status: number
  readonly reason?: string | undefined
}

/** Send one alert to one device token, production first, sandbox on demand. */
export async function sendPush(
  cfg: ApnsConfig,
  deviceToken: string,
  title: string,
  body: string,
): Promise<SendResult> {
  const jwt = await providerJwt(cfg)
  const payload = apnsPayload(title, body)

  const prod = await postToGateway("api.push.apple.com", deviceToken, jwt, cfg.topic, payload)
  const prodOutcome = classifyApnsAnswer(prod.status, prod.reason, "production")
  if (prodOutcome !== "retry-sandbox") {
    return { outcome: prodOutcome, gateway: "production", status: prod.status, reason: prod.reason }
  }

  const sand = await postToGateway("api.sandbox.push.apple.com", deviceToken, jwt, cfg.topic, payload)
  return {
    outcome: classifyApnsAnswer(sand.status, sand.reason, "sandbox"),
    gateway: "sandbox",
    status: sand.status,
    reason: sand.reason,
  }
}

/** Test hook: the jwt/key caches are process-wide. */
export function resetApnsCachesForTests(): void {
  cachedJwt = null
  cachedKey = null
}
