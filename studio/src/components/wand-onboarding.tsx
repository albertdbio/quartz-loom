import { createSignal, For, Show } from "solid-js"
import {
  CRAFTS,
  GOALS,
  personalLine,
  rankSpells,
  SPELLS,
  type Craft,
  type Goal,
  type WandProfile,
} from "~/lib/spells"

/**
 * wand-onboarding — first-run flow for Magic Wand.
 *
 * Four beats: the promise → what you make → what you want to make → a
 * personalized starter spell. The two answers are load-bearing, not a survey:
 * they re-rank the spell deck and pre-select the opening spell, so the last
 * screen visibly reflects what the user just said.
 *
 * Full-screen (not a card) because the app itself is a full-screen stage and
 * this is a phone-first experience. Skippable at every step — App Store review
 * (and users) punish walls in front of the value.
 */
type Props = {
  onDone: (profile?: WandProfile) => void
}

export default function WandOnboarding(props: Props) {
  const [step, setStep] = createSignal(0)
  const [craft, setCraft] = createSignal<Craft | null>(null)
  const [goal, setGoal] = createSignal<Goal | null>(null)

  const ranked = () => (goal() ? rankSpells(SPELLS, goal()!) : SPELLS)
  const starter = () => ranked()[0]!

  function pickCraft(c: Craft) {
    setCraft(c)
    setStep(2)
  }

  function pickGoal(g: Goal) {
    setGoal(g)
    setStep(3)
  }

  function finish() {
    const c = craft()
    const g = goal()
    props.onDone(c && g ? { craft: c, goal: g, at: Date.now() } : undefined)
  }

  return (
    <div class="wonb" role="dialog" aria-modal="true" aria-label="welcome to magic wand">
      <style>{CSS}</style>

      <button class="wonb-skip" onClick={() => props.onDone()}>
        skip
      </button>

      <div class="wonb-body">
        {/* 0 — the promise */}
        <Show when={step() === 0}>
          <div class="wonb-step has-cta">
            <div class="wonb-art" aria-hidden="true">✨</div>
            <h1>Magic Wand</h1>
            <p class="wonb-sub">Unlock your creative potential.</p>
            <p class="wonb-lede">
              Point your camera at the real world, reach out, and <b>touch something</b>.
              Whatever you touch transforms — live, as you watch.
            </p>
            {/* the middle of a hero screen is prime real estate: show the range
                of spells so the product sells itself before we ask anything */}
            <div class="wonb-preview" aria-hidden="true">
              <For each={SPELLS}>
                {(s) => (
                  <span class="wonb-pchip">
                    <span class="wonb-pemoji">{s.emoji}</span>
                    {s.name}
                  </span>
                )}
              </For>
            </div>
            <button class="wonb-cta" onClick={() => setStep(1)}>
              Get started
            </button>
          </div>
        </Show>

        {/* 1 — what do you make */}
        <Show when={step() === 1}>
          <div class="wonb-step">
            <p class="wonb-eyebrow">Step 1 of 2</p>
            <h2>What do you make?</h2>
            <p class="wonb-hint">So the app opens on the right thing for you.</p>
            <div class="wonb-grid">
              <For each={CRAFTS}>
                {(c) => (
                  <button class="wonb-card" onClick={() => pickCraft(c.key)}>
                    <span class="wonb-emoji">{c.emoji}</span>
                    <span class="wonb-label">{c.label}</span>
                    <span class="wonb-desc">{c.sub}</span>
                  </button>
                )}
              </For>
            </div>
          </div>
        </Show>

        {/* 2 — what do you want out of it */}
        <Show when={step() === 2}>
          <div class="wonb-step">
            <p class="wonb-eyebrow">Step 2 of 2</p>
            <h2>What do you want to create?</h2>
            <p class="wonb-hint">We'll put the right spells first.</p>
            <div class="wonb-grid">
              <For each={GOALS}>
                {(g) => (
                  <button class="wonb-card" onClick={() => pickGoal(g.key)}>
                    <span class="wonb-emoji">{g.emoji}</span>
                    <span class="wonb-label">{g.label}</span>
                    <span class="wonb-desc">{g.sub}</span>
                  </button>
                )}
              </For>
            </div>
            <button class="wonb-back" onClick={() => setStep(1)}>← back</button>
          </div>
        </Show>

        {/* 3 — the payoff: their deck, their starter spell */}
        <Show when={step() === 3}>
          <div class="wonb-step has-cta">
            <div class="wonb-art big" aria-hidden="true">{starter().emoji}</div>
            <h2>Your starter spell: {starter().name}</h2>
            <p class="wonb-lede">
              {craft() && goal() ? personalLine(craft()!, goal()!, starter().name) : ""}
            </p>
            <div class="wonb-deck">
              <For each={ranked().slice(0, 4)}>
                {(s, i) => (
                  <span class={`wonb-chip ${i() === 0 ? "lead" : ""}`}>
                    {s.emoji} {s.name}
                  </span>
                )}
              </For>
            </div>
            <button class="wonb-cta" onClick={finish}>
              Start casting ✨
            </button>
            <p class="wonb-fine">Your first minute is free — no account needed.</p>
          </div>
        </Show>
      </div>

      <Show when={step() === 1 || step() === 2}>
        <div class="wonb-dots" aria-hidden="true">
          <For each={[1, 2]}>{(i) => <span class={`wonb-dot ${i === step() ? "on" : ""}`} />}</For>
        </div>
      </Show>
    </div>
  )
}

const CSS = `
  .wonb { position:fixed; inset:0; z-index:20; display:flex; flex-direction:column;
    background:radial-gradient(ellipse at 50% 0%, #1b1338 0%, #07070d 62%);
    color:#f0ecff; font-family:system-ui,-apple-system,sans-serif;
    padding:calc(14px + env(safe-area-inset-top)) calc(20px + env(safe-area-inset-right))
            calc(14px + env(safe-area-inset-bottom)) calc(20px + env(safe-area-inset-left)); }
  .wonb-skip { position:absolute; top:calc(8px + env(safe-area-inset-top));
    right:calc(10px + env(safe-area-inset-right)); background:rgba(24,20,42,.7); border:1px solid #2a2440; color:#b5adcf;
    font:inherit; font-size:14px; cursor:pointer; padding:8px 16px; min-height:40px;
    border-radius:999px; z-index:2; }
  .wonb-skip:hover { color:#c9a0ff; }
  .wonb-body { flex:1; display:flex; align-items:stretch; justify-content:center; min-height:0;
    padding-bottom:6px; }
  .wonb-step { width:min(460px,100%); display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; gap:12px; max-height:100%; overflow-y:auto; }
  /* hero steps: content sits high, action lands under the thumb */
  .wonb-step.has-cta { justify-content:flex-start; padding-top:5vh; }
  .wonb-step.has-cta .wonb-cta { margin-top:auto; }
  .wonb-art { font-size:56px; filter:drop-shadow(0 0 26px rgba(201,160,255,.65)); line-height:1; }
  .wonb-art.big { font-size:68px; }
  .wonb h1 { margin:0; font-size:40px; font-weight:800; letter-spacing:-.03em;
    background:linear-gradient(135deg,#fff,#c9a0ff); -webkit-background-clip:text;
    background-clip:text; color:transparent; }
  .wonb h2 { margin:0; font-size:25px; font-weight:750; letter-spacing:-.02em; line-height:1.2; }
  .wonb-sub { margin:0; color:#d8bcff; font-size:17px; font-weight:650; letter-spacing:.01em; }
  .wonb-lede { margin:2px 0 4px; color:#b5adcf; font-size:15px; line-height:1.6;
    max-width:360px; text-wrap:balance; }
  .wonb-lede b { color:#ffd76a; }
  .wonb-eyebrow { margin:0; color:#6f6791; font-size:12px; text-transform:uppercase;
    letter-spacing:.14em; font-weight:700; }
  .wonb-hint { margin:0; color:#a79fc4; font-size:14px; }
  .wonb-grid { display:flex; flex-direction:column; gap:9px; width:100%; margin-top:6px; }
  .wonb-card { display:grid; grid-template-columns:34px 1fr; grid-template-rows:auto auto;
    align-items:center; gap:0 12px; text-align:left; background:rgba(24,20,42,.8);
    border:1px solid #2a2440; border-radius:15px; padding:13px 16px; cursor:pointer;
    font:inherit; color:inherit; min-height:60px; }
  .wonb-card:hover { border-color:#c9a0ff; background:rgba(38,29,72,.9); }
  .wonb-card:active { transform:scale(.99); }
  .wonb-emoji { grid-row:1 / span 2; font-size:24px; line-height:1; }
  .wonb-label { font-size:15.5px; font-weight:650; }
  .wonb-desc { font-size:13px; color:#a79fc4; }
  /* auto margins on BOTH the chips and the CTA split the leftover height
     evenly instead of pooling it all above the button */
  .wonb-preview { display:flex; flex-wrap:wrap; gap:8px; justify-content:center;
    margin:auto 0 0; max-width:400px; }
  .wonb-pchip { display:inline-flex; align-items:center; gap:6px; font-size:13px; color:#b5adcf;
    border:1px solid #2a2440; background:rgba(24,20,42,.7); border-radius:999px; padding:7px 13px; }
  .wonb-pemoji { font-size:15px; }
  .wonb-deck { display:flex; flex-wrap:wrap; gap:7px; justify-content:center; margin:4px 0 2px; }
  .wonb-chip { font-size:13px; color:#a79fc4; border:1px solid #2a2440; border-radius:999px;
    padding:6px 13px; background:rgba(24,20,42,.7); }
  .wonb-chip.lead { color:#0d0620; background:linear-gradient(135deg,#c9a0ff,#7f6aff);
    border-color:transparent; font-weight:750; }
  .wonb-cta { margin-top:8px; background:linear-gradient(135deg,#c9a0ff,#7f6aff); color:#0d0620;
    border:none; border-radius:999px; padding:16px 40px; font:inherit; font-size:16.5px;
    font-weight:800; cursor:pointer; box-shadow:0 8px 34px rgba(127,106,255,.5); min-height:52px; }
  .wonb-cta:active { transform:scale(.98); }
  .wonb-fine { margin:0; color:#9b93bd; font-size:13px; }
  .wonb-back { background:none; border:none; color:#6f6791; font:inherit; font-size:14px;
    cursor:pointer; padding:8px; margin-top:2px; }
  .wonb-back:hover { color:#c9a0ff; }
  .wonb-dots { display:flex; gap:8px; justify-content:center; padding-top:4px; }
  .wonb-dot { width:7px; height:7px; border-radius:50%; background:#2a2440; }
  .wonb-dot.on { background:#c9a0ff; }
  @media (max-height:700px){
    .wonb-art { font-size:42px; }
    .wonb h1 { font-size:33px; }
    .wonb-lede { font-size:14px; }
    .wonb-card { min-height:54px; padding:11px 14px; }
  }
`
