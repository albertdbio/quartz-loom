import { Show } from "solid-js"

/**
 * Mochi — the character, drawn procedurally so she needs no asset pipeline.
 *
 * Proportions and palette come from the canonical Three.js character
 * (`costumes.html`'s MochiPlayer): a sphere body, capsule wings at ±0.45 of the
 * body radius, black eyes with white highlights, and #ffa7a7 cheeks at 60%
 * opacity. Rendering as SVG rather than WebGL is deliberate — this app already
 * runs a realtime WebRTC video pipeline, and a second GL context on a phone
 * competes with the thing the product is actually selling. SVG costs no GPU
 * and stays crisp at any size.
 *
 * `mood` drives the face: she reacts when a transform lands, which is what makes
 * her read as present in the scene rather than as a sticker.
 */
export type MochiMood = "idle" | "happy" | "wow"

export default function Mochi(props: {
  size?: number
  mood?: MochiMood
  /** Soft contact shadow — the cue that sells "standing in the scene". */
  shadow?: boolean
}) {
  const size = () => props.size ?? 96
  const mood = () => props.mood ?? "idle"

  return (
    <div class={`mochi mochi--${mood()}`} style={{ width: `${size()}px`, height: `${size()}px` }}>
      <style>{CSS}</style>
      <svg viewBox="0 0 100 100" width={size()} height={size()} aria-label="Mochi">
        <defs>
          <radialGradient id="mochiBody" cx="38%" cy="30%" r="78%">
            <stop offset="0%" stop-color="#8fd4f5" />
            <stop offset="55%" stop-color="#4a90e2" />
            <stop offset="100%" stop-color="#2f6fbf" />
          </radialGradient>
          <radialGradient id="mochiGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#8fd4f5" stop-opacity="0.5" />
            <stop offset="100%" stop-color="#8fd4f5" stop-opacity="0" />
          </radialGradient>
          {/* Contact shadow: a dark core that falls off fast. A flat-opacity
              ellipse reads as a drop shadow behind her; the falloff is what
              reads as her actually touching the ground. */}
          <radialGradient id="mochiShadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#000" stop-opacity="0.62" />
            <stop offset="45%" stop-color="#000" stop-opacity="0.30" />
            <stop offset="100%" stop-color="#000" stop-opacity="0" />
          </radialGradient>
        </defs>

        <circle cx="50" cy="52" r="46" fill="url(#mochiGlow)" />

        <Show when={props.shadow !== false}>
          {/* One broad pool reads as a halo around her lower body. Weight comes
              from a tight core under each foot, with the pool only filling in
              the ambient occlusion between and around them. */}
          <g class="mochi-shadow">
            <ellipse cx="50" cy="89.5" rx="24" ry="5" fill="url(#mochiShadow)" />
            <ellipse cx="40" cy="88.8" rx="7.5" ry="2.1" fill="#000" opacity="0.5" />
            <ellipse cx="60" cy="88.8" rx="7.5" ry="2.1" fill="#000" opacity="0.5" />
          </g>
        </Show>

        <g class="mochi-body">
          {/* wings — capsules at the body's equator, mirroring the 3D rig */}
          <rect x="4" y="46" width="16" height="8" rx="4" fill="#4a90e2" transform="rotate(-14 12 50)" />
          <rect x="80" y="46" width="16" height="8" rx="4" fill="#4a90e2" transform="rotate(14 88 50)" />

          {/* body */}
          <circle cx="50" cy="52" r="34" fill="url(#mochiBody)" />

          {/* feet */}
          <ellipse cx="40" cy="84" rx="7" ry="4.5" fill="#3f7fc9" />
          <ellipse cx="60" cy="84" rx="7" ry="4.5" fill="#3f7fc9" />

          {/* cheeks */}
          <ellipse class="mochi-cheek" cx="33" cy="58" rx="6" ry="4" fill="#ffa7a7" opacity="0.6" />
          <ellipse class="mochi-cheek" cx="67" cy="58" rx="6" ry="4" fill="#ffa7a7" opacity="0.6" />

          {/* eyes */}
          <g class="mochi-eyes">
            <ellipse cx="39" cy="47" rx="6.5" ry="8" fill="#111" />
            <ellipse cx="61" cy="47" rx="6.5" ry="8" fill="#111" />
            <circle cx="41.5" cy="44" r="2.4" fill="#fff" />
            <circle cx="63.5" cy="44" r="2.4" fill="#fff" />
          </g>

          {/* mouth */}
          <path class="mochi-mouth" d="M45 62 Q50 66 55 62" stroke="#111" stroke-width="2"
            fill="none" stroke-linecap="round" />
        </g>
      </svg>
    </div>
  )
}

const CSS = `
  /* A large soft drop-shadow halos the whole character and reads as "pasted on
     top". Keep it tight so the contact shadow below her does the grounding. */
  .mochi { pointer-events:none; user-select:none; -webkit-user-select:none;
    filter: drop-shadow(0 2px 5px rgba(0,0,0,.38)); }
  .mochi svg { overflow:visible; display:block; }
  /* idle bob — the whole body, so the shadow can breathe against it */
  .mochi-body { transform-origin:50% 62%; animation: mochi-bob 2.6s ease-in-out infinite; }
  .mochi-shadow { transform-origin:50% 92%; animation: mochi-shadow 2.6s ease-in-out infinite; }
  @keyframes mochi-bob {
    0%,100% { transform: translateY(0) scale(1,1); }
    50%     { transform: translateY(-4px) scale(0.99,1.01); }
  }
  @keyframes mochi-shadow {
    0%,100% { transform: scale(1); opacity:.28; }
    50%     { transform: scale(.86); opacity:.18; }
  }
  /* a transform just landed */
  .mochi--happy .mochi-body { animation: mochi-hop .55s cubic-bezier(.34,1.56,.64,1) 1, mochi-bob 2.6s ease-in-out infinite .55s; }
  @keyframes mochi-hop {
    0%   { transform: translateY(0) scale(1,1); }
    30%  { transform: translateY(-16px) scale(.94,1.08); }
    70%  { transform: translateY(0) scale(1.08,.92); }
    100% { transform: translateY(0) scale(1,1); }
  }
  .mochi--wow .mochi-eyes { animation: mochi-wow .5s ease-out; transform-origin:50% 47%; }
  @keyframes mochi-wow { 0%,100% { transform: scale(1); } 45% { transform: scale(1.22); } }
  @media (prefers-reduced-motion: reduce) {
    .mochi-body, .mochi-shadow, .mochi--happy .mochi-body { animation:none; }
  }
`
