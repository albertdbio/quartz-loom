#!/usr/bin/env node
/**
 * One-time: buy a sender number for phone sign-in and attach it to the
 * account's already-VERIFIED A2P campaign.
 *
 * Why attach rather than register: the campaign is a MIXED use case whose
 * registered sample is "Your one-time passcode is 123456." — our OTP is
 * already the registered content, so the number inherits compliant sending
 * with no new vetting.
 *
 * Why a NEW number rather than reusing one: the existing numbers answer as
 * mochi personas. The service has `useInboundWebhookOnNumber = true` and no
 * service-level inbound URL, so each number keeps its own inbound routing —
 * leaving this one's webhook UNSET makes it outbound-only without touching
 * mochi. Twilio still answers STOP/HELP itself.
 *
 * Usage (creds come from the environment; never hard-code them):
 *   node scripts/provision-number.mjs --area 619
 *   node scripts/provision-number.mjs --area 619 --buy     # actually purchases
 */
import twilio from "twilio"

const MESSAGING_SERVICE_SID = "MGe0a8243a58c015290a6d6a308e2675b2"

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`)
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback
}
const areaCode = arg("area", "619")
const doBuy = process.argv.includes("--buy")

const { TWILIO_ACCOUNT_SID, TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET } = process.env
for (const [k, v] of Object.entries({ TWILIO_ACCOUNT_SID, TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET })) {
  if (!v) {
    console.error(`missing ${k} in the environment`)
    process.exit(1)
  }
}
const client = twilio(TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET, { accountSid: TWILIO_ACCOUNT_SID })

const available = await client
  .availablePhoneNumbers("US")
  .local.list({ areaCode: Number(areaCode), smsEnabled: true, limit: 5 })

if (available.length === 0) {
  console.error(`no SMS-capable numbers free in area code ${areaCode} — try another`)
  process.exit(1)
}
console.log(`available in ${areaCode}:`)
for (const n of available) console.log(`  ${n.phoneNumber}  ${n.locality ?? ""}`)

if (!doBuy) {
  console.log("\n(dry run — re-run with --buy to purchase the first one)")
  process.exit(0)
}

const pick = available[0]
console.log(`\nbuying ${pick.phoneNumber} …`)
const purchased = await client.incomingPhoneNumbers.create({
  phoneNumber: pick.phoneNumber,
  friendlyName: "Mochiverse — sign-in",
  // Deliberately no smsUrl: outbound-only. An inbound webhook here would be
  // the only way a reply could land in another product's agent.
})
console.log(`  sid=${purchased.sid} number=${purchased.phoneNumber}`)

console.log(`attaching to messaging service ${MESSAGING_SERVICE_SID} (verified A2P campaign) …`)
await client.messaging.v1.services(MESSAGING_SERVICE_SID).phoneNumbers.create({
  phoneNumberSid: purchased.sid,
})

const attached = await client.messaging.v1.services(MESSAGING_SERVICE_SID).phoneNumbers.list({ limit: 20 })
console.log(`  service now carries ${attached.length} numbers:`)
for (const n of attached) console.log(`    ${n.phoneNumber}`)

console.log(`\nDone. Put this in studio/.env:\n  TWILIO_FROM_NUMBER=${purchased.phoneNumber}`)
