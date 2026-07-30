import { Schema } from "effect"

/**
 * Character generation — "any friend you can describe".
 *
 * Two fal.ai hops: flux/schnell draws the character on a plain background,
 * rembg cuts it out. The result returns as a data URL so the client can keep
 * it in localStorage — the character should survive reloads without us
 * running a media store.
 *
 * The template does the heavy lifting: it forces a single centered full-body
 * character with a clean silhouette (what rembg needs to cut well) and
 * forbids a painted ground shadow, because the scene overlay draws its own
 * contact shadow and a doubled shadow reads as a compositing error.
 *
 * Costs real money per call (~1-2¢), so the rate limit is deliberately
 * tighter than the other routes.
 */
export const CharacterBody = Schema.Struct({
  prompt: Schema.String.check(Schema.isMinLength(3), Schema.isMaxLength(120)),
})

const TEMPLATE = (subject: string) =>
  `full body chibi cartoon character, ${subject}, single character, centered, ` +
  `standing, facing viewer, clean sharp silhouette, plain solid white background, ` +
  `no ground shadow, no floor, no text, no watermark`

interface FalImage {
  readonly url?: string
}

async function falRun(endpoint: string, key: string, body: unknown): Promise<unknown> {
  const res = await fetch(`https://fal.run/${endpoint}`, {
    method: "POST",
    headers: { Authorization: `Key ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60_000),
  })
  if (!res.ok) throw new Error(`fal ${endpoint} failed (${res.status})`)
  return res.json()
}

/** prompt -> transparent PNG data URL. */
export async function generateCharacter(subject: string, key: string): Promise<string> {
  const gen = (await falRun("fal-ai/flux/schnell", key, {
    prompt: TEMPLATE(subject),
    image_size: "square",
    num_inference_steps: 4,
  })) as { images?: ReadonlyArray<FalImage> }
  const drawn = gen.images?.[0]?.url
  if (!drawn) throw new Error("generation returned no image")

  const cut = (await falRun("fal-ai/imageutils/rembg", key, { image_url: drawn })) as {
    image?: FalImage
  }
  const sprite = cut.image?.url
  if (!sprite) throw new Error("background removal returned no image")

  const img = await fetch(sprite, { signal: AbortSignal.timeout(30_000) })
  if (!img.ok) throw new Error(`sprite fetch failed (${img.status})`)
  const bytes = Buffer.from(await img.arrayBuffer())
  // A sprite is a UI asset, not a poster: past ~4MB something has gone wrong
  // upstream and we should not ship it into someone's localStorage.
  if (bytes.byteLength > 4 * 1024 * 1024) throw new Error("sprite unexpectedly large")
  return `data:image/png;base64,${bytes.toString("base64")}`
}

