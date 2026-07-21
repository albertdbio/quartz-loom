import { Data, Effect, Schema } from "effect"

/**
 * Pragmatic email shape: liberal local part, sane domain (letters/digits/
 * dots/hyphens + a TLD). Caught red-handed accepting `x@;y.z` during sim QA.
 */
export const Email = Schema.String.check(
  Schema.isPattern(/^[^\s@]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}$/),
)

/** A request body that failed to parse or decode. Routes map this to a 400. */
export class BadRequest extends Data.TaggedError("BadRequest")<{
  readonly reason: string
}> {}

export const json = (data: unknown, status = 200): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  })

export const errorJson = (status: number, message: string): Response =>
  json({ error: message }, status)

/** 429 with a Retry-After header (seconds, rounded up). */
export const tooManyRequests = (retryAfterMs: number): Response =>
  new Response(JSON.stringify({ error: "too many requests" }), {
    status: 429,
    headers: {
      "Content-Type": "application/json",
      "Retry-After": String(Math.max(1, Math.ceil(retryAfterMs / 1000))),
    },
  })

/** Read + decode a JSON request body through a Schema (parse, don't validate). */
export const decodeBody = <S extends Schema.Top>(
  request: Request,
  schema: S,
): Effect.Effect<S["Type"], BadRequest, S["DecodingServices"]> =>
  Effect.tryPromise({
    try: () => request.json(),
    catch: () => new BadRequest({ reason: "expected a JSON body" }),
  }).pipe(
    Effect.flatMap((body) =>
      Schema.decodeUnknownEffect(schema)(body).pipe(
        Effect.mapError((e) => new BadRequest({ reason: String(e) })),
      ),
    ),
  )

/** One SSE frame carrying a JSON payload. */
export const sseFrame = (data: unknown): string => `data: ${JSON.stringify(data)}\n\n`

export const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
} as const

/**
 * Shared voice-assistant instructions. Used by BOTH voice modes so LiveKit
 * agent sessions and direct OpenAI Realtime sessions behave identically.
 * `history` is the server-serialized text-chat context (see
 * `serializeVoiceHistory`) — never client-supplied free text.
 */
export const voiceInstructions = (history: string): string =>
  "You are opentxt's voice assistant. Your interface with the user is voice: " +
  "keep responses short and conversational, and avoid unpronounceable punctuation." +
  (history.length > 0 ? ` Previous chat history with this user: ${history}` : "")

/**
 * Cap keeps the serialized history comfortably inside a LiveKit token
 * attribute (the token rides the connect URL — a bloated JWT can trip
 * URL/header size limits) and inside the Realtime instructions budget.
 */
const VOICE_HISTORY_MAX_CHARS = 2400

/**
 * Serialize chat messages for the voice context bridge. Runs SERVER-SIDE
 * from DB rows the caller already ownership-checked: the client only ever
 * sends a chatId, so it cannot inject fabricated "history" into the
 * assistant's instructions.
 */
export const serializeVoiceHistory = (
  messages: ReadonlyArray<{ readonly role: string; readonly content: string }>,
): string => {
  const compact = messages.map((m) => `${m.role}: ${m.content}`).join("\n")
  return compact.length > VOICE_HISTORY_MAX_CHARS ? compact.slice(-VOICE_HISTORY_MAX_CHARS) : compact
}
