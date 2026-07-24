/**
 * The spell deck + onboarding personalization.
 *
 * Every spell is a prompt that instructs the realtime model to transform ONLY
 * what the hand touches, spreading from the contact point, leaving the rest of
 * the frame photoreal. Onboarding answers are not a survey: the chosen goal
 * re-ranks this deck and pre-selects a starter spell, so answering visibly
 * changes the app on the very next screen.
 */

export interface Spell {
  readonly emoji: string
  readonly name: string
  readonly prompt: string
}

const SPELL_BASE =
  "Magic touch effect, one continuous photoreal camera shot: any object the person's hand " +
  "or handheld wand touches instantly transforms, the transformation spreading outward from " +
  "exactly the point of contact. Everything not yet touched stays completely photorealistic " +
  "and unchanged, with consistent real-world lighting and contact shadows. "

export const SPELLS: ReadonlyArray<Spell> = [
  {
    emoji: "🏆",
    name: "Midas",
    prompt: SPELL_BASE +
      "Touched objects turn into solid gleaming gold with mirror-like reflections, tiny golden " +
      "sparkles bursting from the contact point.",
  },
  {
    emoji: "❄️",
    name: "Frost",
    prompt: SPELL_BASE +
      "Touched objects freeze into crystalline blue ice, frost crystals crawling outward from the " +
      "fingertip, a wisp of cold mist rising.",
  },
  {
    emoji: "🌸",
    name: "Bloom",
    prompt: SPELL_BASE +
      "Touched objects burst into blooming flowers, moss and lush green vines spreading from the " +
      "touch point, petals drifting off gently.",
  },
  {
    emoji: "🧸",
    name: "Toy",
    prompt: SPELL_BASE +
      "Touched objects become glossy plastic toy versions of themselves with bright saturated " +
      "colors, smooth simplified shapes, and molded seams.",
  },
  {
    emoji: "🕹️",
    name: "8-bit",
    prompt: SPELL_BASE +
      "Touched objects turn into chunky 8-bit voxel pixel art with a limited retro palette, " +
      "little pixel particles scattering from the contact point.",
  },
  {
    emoji: "👻",
    name: "Spectral",
    prompt: SPELL_BASE +
      "Touched objects become translucent glowing ghost versions of themselves, ethereal cyan " +
      "wisps curling away from the contact point.",
  },
  {
    emoji: "🍬",
    name: "Candy",
    prompt: SPELL_BASE +
      "Touched objects turn into glossy candy — striped sugar, gumdrop textures, dripping " +
      "frosting — with a sugary sparkle at the contact point.",
  },
  {
    emoji: "✏️",
    name: "Sketch",
    prompt: SPELL_BASE +
      "Touched objects become hand-drawn pencil sketches of themselves, cross-hatched shading on " +
      "white paper texture, graphite dust puffing from the contact point.",
  },
]

// -- onboarding profile ------------------------------------------------------ //

export type Craft = "creator" | "artist" | "marketer" | "educator" | "fun"
export type Goal = "viral" | "product" | "art" | "teach" | "explore"

export interface WandProfile {
  readonly craft: Craft
  readonly goal: Goal
  /** epoch ms — lets us re-ask if the profile ever goes stale. */
  readonly at: number
}

export const PROFILE_KEY = "wand.profile.v1"

export const CRAFTS: ReadonlyArray<{ key: Craft; emoji: string; label: string; sub: string }> = [
  { key: "creator", emoji: "📱", label: "Social content", sub: "Reels, TikToks, Shorts" },
  { key: "artist", emoji: "🎨", label: "Art & film", sub: "Visual work, experiments" },
  { key: "marketer", emoji: "📣", label: "Brand & marketing", sub: "Promos, product spots" },
  { key: "educator", emoji: "🧑‍🏫", label: "Teaching & demos", sub: "Explaining things visually" },
  { key: "fun", emoji: "✨", label: "Just for fun", sub: "Play with it, see what happens" },
]

export const GOALS: ReadonlyArray<{ key: Goal; emoji: string; label: string; sub: string }> = [
  { key: "viral", emoji: "🚀", label: "Something people share", sub: "Scroll-stopping clips" },
  { key: "product", emoji: "💎", label: "Make objects look premium", sub: "Products, spaces, things" },
  { key: "art", emoji: "🖌", label: "Make something beautiful", sub: "Painterly, dreamlike, strange" },
  { key: "teach", emoji: "💡", label: "Show an idea clearly", sub: "Illustrate, explain, demo" },
  { key: "explore", emoji: "🔮", label: "Not sure yet — surprise me", sub: "Show me what it does" },
]

/**
 * Signature spells per goal, best-first. Anything unlisted keeps its original
 * order behind them, so the deck is re-ranked and never truncated.
 */
const PRIORITY: Record<Goal, ReadonlyArray<string>> = {
  viral: ["8-bit", "Candy", "Midas"],
  product: ["Midas", "Frost", "Toy"],
  art: ["Sketch", "Bloom", "Spectral"],
  teach: ["Sketch", "Toy", "8-bit"],
  explore: [],
}

/** Re-rank the deck for a goal. Pure: returns a new array, never mutates. */
export function rankSpells(deck: ReadonlyArray<Spell>, goal: Goal): ReadonlyArray<Spell> {
  const priority = PRIORITY[goal] ?? []
  if (priority.length === 0) return [...deck]
  const rank = (s: Spell): number => {
    const i = priority.indexOf(s.name)
    return i === -1 ? priority.length + deck.indexOf(s) : i
  }
  return [...deck].sort((a, b) => rank(a) - rank(b))
}

/** Position of the starter spell inside an ALREADY-RANKED deck. */
export function starterSpellIndex(ranked: ReadonlyArray<Spell>, goal: Goal): number {
  const first = PRIORITY[goal]?.[0]
  if (!first) return 0
  const i = ranked.findIndex((s) => s.name === first)
  return i === -1 ? 0 : i
}

const GOAL_LINE: Record<Goal, (spell: string) => string> = {
  viral: (s) => `Built for the scroll. Start with ${s} — it reads instantly on a small screen.`,
  product: (s) => `Let's make ordinary things look expensive. ${s} first.`,
  art: (s) => `Then we'll go strange and beautiful. ${s} is your opening move.`,
  teach: (s) => `Clarity first. ${s} turns whatever you touch into something explainable.`,
  explore: (s) => `Perfect — no plan required. Touch anything with ${s} and see.`,
}

const CRAFT_PREFIX: Record<Craft, string> = {
  creator: "For your feed:",
  artist: "For your work:",
  marketer: "For the brand:",
  educator: "For your audience:",
  fun: "Alright then:",
}

/** One personalized line shown after onboarding — names the starter spell. */
export function personalLine(craft: Craft, goal: Goal, starterSpellName: string): string {
  return `${CRAFT_PREFIX[craft]} ${GOAL_LINE[goal](starterSpellName)}`
}

// -- persistence (browser only) ---------------------------------------------- //

export function loadProfile(): WandProfile | null {
  try {
    const raw = localStorage.getItem(PROFILE_KEY)
    if (!raw) return null
    const p = JSON.parse(raw) as Partial<WandProfile>
    if (typeof p.craft !== "string" || typeof p.goal !== "string") return null
    return { craft: p.craft as Craft, goal: p.goal as Goal, at: typeof p.at === "number" ? p.at : 0 }
  } catch {
    return null
  }
}

export function saveProfile(profile: WandProfile): void {
  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile))
  } catch {
    // private mode / storage disabled — onboarding just repeats next launch
  }
}
