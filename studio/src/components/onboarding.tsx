import { createSignal, For, Show } from "solid-js"

/**
 * onboarding — first-run overlay for the studio.
 *
 * Three quick cards (what this is → the modes → voice & camera) ending in a
 * mode picker. Shown once (localStorage flag, owned by the caller) and
 * reopenable from the header's "?" button. Pure overlay: no router involvement,
 * client-only by construction (the caller mounts it after checking localStorage).
 */

export type OnboardingChoice = "lucy" | "open" | "story"

type Props = {
  onDone: (pick?: OnboardingChoice) => void
}

const STEPS = [
  {
    icon: "◉",
    title: "Real-time video, steered by you",
    body:
      "Point a camera — or just a blank canvas — at a diffusion model and describe what you want. " +
      "The picture transforms live while you watch, and every new description bends it mid-stream.",
  },
  {
    icon: "⇶",
    title: "Three ways to run it",
    body: "",
    modes: [
      { key: "lucy" as const, name: "API", desc: "Hosted model — highest quality, camera restyle, per-second billing." },
      { key: "open" as const, name: "Self-hosted", desc: "Your own GPU server over one WebSocket. Open weights, flat cost." },
      { key: "story" as const, name: "Story canvas", desc: "No camera. Narrate out loud and the model dreams the scene, evolving as you speak." },
    ],
  },
  {
    icon: "🎙",
    title: "Talk to it",
    body:
      "Tap the mic and speak — each sentence you finish becomes the live prompt. " +
      "On a phone, the back camera is the default and the browser will ask for camera/mic " +
      "permission on first use. Recording saves the output as a video file.",
  },
]

export default function Onboarding(props: Props) {
  const [step, setStep] = createSignal(0)
  const last = STEPS.length - 1

  return (
    <div class="onb" role="dialog" aria-modal="true" aria-label="welcome">
      <style>{CSS}</style>
      <div class="onb-card">
        <button class="onb-skip" onClick={() => props.onDone()}>
          skip
        </button>
        <div class="onb-icon" aria-hidden="true">
          {STEPS[step()]?.icon}
        </div>
        <h2>{STEPS[step()]?.title}</h2>
        <Show when={STEPS[step()]?.body}>
          <p>{STEPS[step()]?.body}</p>
        </Show>
        <Show when={STEPS[step()]?.modes} keyed>
          {(modes) => (
            <div class="onb-modes">
              <For each={modes}>
                {(m) => (
                  <button class="onb-mode" onClick={() => props.onDone(m.key)}>
                    <span class="onb-mode-name">{m.name}</span>
                    <span class="onb-mode-desc">{m.desc}</span>
                  </button>
                )}
              </For>
            </div>
          )}
        </Show>
        <div class="onb-nav">
          <div class="onb-dots" aria-hidden="true">
            <For each={STEPS}>{(_, i) => <span class={`onb-dot ${i() === step() ? "on" : ""}`} />}</For>
          </div>
          <Show
            when={step() < last}
            fallback={
              <button class="onb-next" onClick={() => props.onDone()}>
                let's go
              </button>
            }
          >
            <button class="onb-next" onClick={() => setStep(step() + 1)}>
              next
            </button>
          </Show>
        </div>
        <Show when={step() === 1}>
          <p class="onb-hint">Tap a mode to start there, or keep reading.</p>
        </Show>
      </div>
    </div>
  )
}

const CSS = `
  .onb { position:fixed; inset:0; z-index:50; display:flex; align-items:center; justify-content:center;
    background:rgba(5,8,12,.82); backdrop-filter:blur(6px); padding:18px; }
  .onb-card { position:relative; width:min(440px, 100%); background:#12161d; border:1px solid #232b35;
    border-radius:20px; padding:34px 28px 24px; color:#e8edf2; box-shadow:0 24px 60px -24px rgba(0,0,0,.8);
    font-family: system-ui,-apple-system,sans-serif; }
  .onb-skip { position:absolute; top:12px; right:14px; background:none; border:none; color:#5c6675;
    font-size:13px; cursor:pointer; padding:6px 8px; }
  .onb-skip:hover { color:#8a96a3; }
  .onb-icon { font-size:34px; margin-bottom:10px; color:#6ea8fe; }
  .onb-card h2 { margin:0 0 10px; font-size:22px; line-height:1.2; letter-spacing:-.01em; }
  .onb-card p { margin:0 0 14px; color:#8a96a3; font-size:15px; line-height:1.55; }
  .onb-modes { display:flex; flex-direction:column; gap:10px; margin:6px 0 14px; }
  .onb-mode { display:flex; flex-direction:column; gap:3px; text-align:left; background:#171c25;
    border:1px solid #232b35; border-radius:12px; padding:12px 14px; cursor:pointer; font:inherit; color:inherit; }
  .onb-mode:hover { border-color:#6ea8fe; }
  .onb-mode-name { font-weight:650; font-size:15px; color:#e8edf2; }
  .onb-mode-desc { font-size:13px; color:#8a96a3; line-height:1.45; }
  .onb-nav { display:flex; align-items:center; justify-content:space-between; margin-top:6px; }
  .onb-dots { display:flex; gap:7px; }
  .onb-dot { width:7px; height:7px; border-radius:50%; background:#232b35; }
  .onb-dot.on { background:#6ea8fe; }
  .onb-next { background:#6ea8fe; color:#08131f; border:none; border-radius:10px; padding:11px 22px;
    font:inherit; font-weight:650; font-size:15px; cursor:pointer; min-height:44px; }
  .onb-next:hover { filter:brightness(1.08); }
  .onb-hint { font-size:12px; color:#5c6675; margin:0; text-align:center; }
`
