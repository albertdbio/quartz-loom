import { Config, Context, Data, Effect, Layer, Redacted } from "effect"
import twilio from "twilio"

/**
 * Outbound SMS for phone sign-in — the same Twilio account and API-key pattern
 * mochi/core uses, with its own FROM number.
 *
 * A separate sender number (not mochi's) is deliberate: replies to mochi's
 * number are answered by mochi's agent, so sharing it would route a Mochiverse
 * user's "STOP" or "huh?" into a conversation with a different product. Twilio
 * handles STOP/HELP on the number automatically, which is why v1 needs no
 * inbound webhook at all — the code is typed into the app, never replied to.
 *
 * US A2P 10DLC: long-code application traffic must be registered or carriers
 * filter it. Register this number under a "2FA / OTP" campaign before relying
 * on delivery, and keep any future marketing sends on a SEPARATE campaign with
 * its own consent — mixing them is how numbers get blocked.
 */

export class SmsError extends Data.TaggedError("SmsError")<{
  readonly cause: unknown
}> {}

export const SmsConfig = Config.all({
  accountSid: Config.string("TWILIO_ACCOUNT_SID"),
  apiKeySid: Config.string("TWILIO_API_KEY_SID"),
  apiKeySecret: Config.redacted("TWILIO_API_KEY_SECRET"),
  fromNumber: Config.string("TWILIO_FROM_NUMBER"),
})

export class Sms extends Context.Service<Sms>()("studio/Sms", {
  make: Effect.succeed({
    /**
     * Send one message. Config resolves PER CALL (like DecartRealtime and
     * Billing) so missing Twilio env only fails the auth routes rather than
     * taking down the whole runtime.
     */
    send: (to: string, body: string): Effect.Effect<string, SmsError> =>
      Effect.gen(function* () {
        const cfg = yield* SmsConfig.pipe(
          Effect.mapError(
            () =>
              new SmsError({
                cause: "sms is not configured (set TWILIO_ACCOUNT_SID, TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET, TWILIO_FROM_NUMBER)",
              }),
          ),
        )
        return yield* Effect.tryPromise({
          try: async () => {
            const client = twilio(cfg.apiKeySid, Redacted.value(cfg.apiKeySecret), {
              accountSid: cfg.accountSid,
            })
            const message = await client.messages.create({ to, from: cfg.fromNumber, body })
            return message.sid
          },
          catch: (cause) => new SmsError({ cause }),
        })
      }),
  } as const),
}) {
  static readonly layer = Layer.effect(this, this.make)
}

/** The one message this app sends. Kept short — carriers dislike long OTPs. */
export const otpMessage = (code: string): string =>
  `${code} is your Mochiverse code. It expires in 10 minutes.`
