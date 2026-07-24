import { createSignal, For, onCleanup, onMount, Show } from "solid-js"

/**
 * magic wand ✨ — touch anything, transform it.
 *
 * A product skin over the realtime restyle engine (Decart Lucy 2.5): the
 * camera streams full-screen, and a "spell" prompt instructs the model to
 * transform WHATEVER THE HAND (or a held wand) TOUCHES, spreading from the
 * contact point while everything untouched stays photoreal. The interaction
 * is physical — you reach out and touch the object on camera; the model does
 * the segmentation-by-description. Spells switch live via rt.set().
 *
 * Billing rides the same gated /api/decart/token mint as the studio: first
 * minute free (server-burned), then Studio Pro. See server/plan.ts.
 */
type Status = "idle" | "connecting" | "live" | "error"

interface Spell {
  readonly emoji: string
  readonly name: string
  readonly prompt: string
}

const SPELL_BASE =
  "Magic touch effect, one continuous photoreal camera shot: any object the person's hand " +
  "or handheld wand touches instantly transforms, the transformation spreading outward from " +
  "exactly the point of contact. Everything not yet touched stays completely photorealistic " +
  "and unchanged, with consistent real-world lighting and contact shadows. "

const SPELLS: ReadonlyArray<Spell> = [
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

export default function Wand() {
  let outputVideo: HTMLVideoElement | undefined
  let pipVideo: HTMLVideoElement | undefined
  const [status, setStatus] = createSignal<Status>("idle")
  const [err, setErr] = createSignal("")
  const [spell, setSpell] = createSignal(0)
  const [recording, setRecording] = createSignal(false)
  const [plan, setPlan] = createSignal<"loading" | "free" | "pro">("loading")
  const [freeLeft, setFreeLeft] = createSignal<number | null>(null)
  const [showPaywall, setShowPaywall] = createSignal(false)
  // Raw-camera PiP is OFF by default: on iOS WKWebView a <video> playing a
  // local capture stream composites ABOVE other video layers regardless of
  // z-index, burying the generated stream. The magic reads better without it
  // anyway — you can see reality with your own eyes.
  const [showPip, setShowPip] = createSignal(false)

  let rt: any = null
  let stream: MediaStream | null = null
  let editedStream: MediaStream | null = null
  let recorder: MediaRecorder | null = null
  let chunks: Blob[] = []
  let freeTimer: ReturnType<typeof setInterval> | null = null

  onMount(() => {
    void fetch("/api/billing/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((s: { plan: "free" | "pro" } | null) => s && setPlan(s.plan))
      .catch(() => {})
  })

  function stopCountdown() {
    if (freeTimer) clearInterval(freeTimer)
    freeTimer = null
    setFreeLeft(null)
  }

  function startCountdown(seconds: number) {
    setFreeLeft(seconds)
    freeTimer = setInterval(() => {
      const left = (freeLeft() ?? 0) - 1
      setFreeLeft(left)
      if (left <= 0) {
        stop()
        setShowPaywall(true)
      }
    }, 1000)
  }

  async function subscribe() {
    try {
      const res = await fetch("/api/billing/checkout", { method: "POST" })
      const body = (await res.json()) as { url?: string; error?: string }
      if (res.ok && body.url) window.location.href = body.url
      else setErr(body.error ?? "could not start checkout")
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  async function start() {
    setErr("")
    setStatus("connecting")
    try {
      const { createDecartClient, models } = await import("@decartai/sdk")
      const model = models.realtime("lucy-2.5")
      const res = await fetch("/api/decart/token", { method: "POST" })
      if (res.status === 402) {
        setShowPaywall(true)
        setStatus("idle")
        return
      }
      if (!res.ok) throw new Error(`could not start (${res.status})`)
      const minted = (await res.json()) as { apiKey: string; plan?: string; freeSeconds?: number }

      // Rear camera by default — magic wand is a walk-around-and-touch toy.
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          frameRate: model.fps,
          width: model.width,
          height: model.height,
          facingMode: { ideal: "environment" },
        },
        audio: false,
      })
      if (pipVideo && showPip()) {
        pipVideo.srcObject = stream
        void pipVideo.play().catch(() => {})
      }
      const client = createDecartClient({ apiKey: minted.apiKey })
      rt = await client.realtime.connect(stream, {
        model,
        mirror: false,
        onRemoteStream: (edited: MediaStream) => {
          editedStream = edited
          if (outputVideo) {
            outputVideo.srcObject = edited
            void outputVideo.play().catch(() => {})
          }
        },
        initialState: { prompt: { text: SPELLS[spell()]!.prompt, enhance: true } },
      })
      setStatus("live")
      if (minted.plan === "free") {
        setPlan("free")
        startCountdown(minted.freeSeconds ?? 60)
      } else if (minted.plan === "pro") {
        setPlan("pro")
      }
    } catch (e) {
      const name = e instanceof Error ? e.name : ""
      setErr(
        name === "NotAllowedError" || name === "SecurityError"
          ? "Camera access was blocked — allow the camera for this site, then try again."
          : e instanceof Error
            ? e.message
            : String(e),
      )
      setStatus("error")
      stop()
    }
  }

  function togglePip() {
    const next = !showPip()
    setShowPip(next)
    if (pipVideo) {
      if (next && stream) {
        pipVideo.srcObject = stream
        void pipVideo.play().catch(() => {})
      } else {
        pipVideo.srcObject = null
      }
    }
  }

  async function castSpell(i: number) {
    setSpell(i)
    if (!rt) return
    try {
      await rt.set({ prompt: SPELLS[i]!.prompt })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  function toggleRecord() {
    if (recording()) {
      recorder?.stop()
      return
    }
    if (!editedStream) return
    chunks = []
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm"
    recorder = new MediaRecorder(editedStream, { mimeType: mime })
    recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data)
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `magic-wand-${Date.now()}.webm`
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 4000)
      setRecording(false)
    }
    recorder.start()
    setRecording(true)
  }

  function stop() {
    try {
      recorder?.state === "recording" && recorder.stop()
      rt?.close?.()
      rt?.disconnect?.()
    } catch {
      // best-effort teardown
    }
    rt = null
    stopCountdown()
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    editedStream = null
    if (status() === "live") setStatus("idle")
  }

  onCleanup(stop)

  return (
    <main class="wand">
      <style>{CSS}</style>

      {/* the stage: generated stream, edge to edge */}
      <video ref={outputVideo} class="stage" autoplay playsinline muted />
      <video ref={pipVideo} class="pip" classList={{ hidden: !showPip() }} autoplay playsinline muted />

      {/* top chrome — floats over the stage, inset out of the notch */}
      <div class="top">
        <h1>
          magic wand <span class="sparkle">✨</span>
          <Show when={plan() === "pro"}>
            <span class="pro">PRO</span>
          </Show>
        </h1>
        <div class="top-right">
          <Show when={freeLeft() !== null}>
            <span class={`chip countdown ${(freeLeft() ?? 0) <= 10 ? "low" : ""}`}>
              {Math.floor((freeLeft() ?? 0) / 60)}:{String((freeLeft() ?? 0) % 60).padStart(2, "0")}
            </span>
          </Show>
          <Show when={status() === "live"}>
            <button class={`chip rec ${recording() ? "on" : ""}`} onClick={toggleRecord}>
              {recording() ? "■ save" : "● rec"}
            </button>
            <button class="chip" onClick={togglePip} title="raw camera preview">
              {showPip() ? "🙈" : "👁"}
            </button>
          </Show>
        </div>
      </div>

      {/* idle / connecting takeover */}
      <Show when={status() !== "live"}>
        <div class="overlay">
          <Show when={status() !== "connecting"} fallback={<p class="conn">summoning…</p>}>
            <h2 class="pitch">Touch anything.<br />Transform it.</h2>
            <p class="howto">
              Point the camera at the world, hold out your hand — or grab a pencil as your wand —
              and <b>touch something</b>.
            </p>
            <button class="cta" onClick={start}>begin ✨</button>
            <Show when={plan() === "free"}>
              <p class="free-hint">first minute free · then $20/mo</p>
            </Show>
            <Show when={err()}>
              <p class="err">{err()}</p>
            </Show>
          </Show>
        </div>
      </Show>

      {/* bottom chrome — spells + stop, inset off the home indicator */}
      <div class="bottom">
        <div class="spells">
          <For each={SPELLS}>
            {(s, i) => (
              <button
                class={`spell ${spell() === i() ? "on" : ""}`}
                onClick={() => void castSpell(i())}
                title={s.name}
              >
                <span class="emoji">{s.emoji}</span>
                <span class="name">{s.name}</span>
              </button>
            )}
          </For>
        </div>
        <Show when={status() === "live"}>
          <div class="live-row">
            <button class="stopbtn" onClick={stop}>stop</button>
            <Show when={err()}>
              <span class="err inline">{err()}</span>
            </Show>
          </div>
        </Show>
      </div>

      <Show when={showPaywall()}>
        <div class="pw-backdrop" onClick={(e) => e.target === e.currentTarget && setShowPaywall(false)}>
          <div class="pw" role="dialog" aria-modal="true" aria-label="Studio Pro">
            <h2>Your free minute of magic is up ✨</h2>
            <p>Keep casting as long as you like with Studio Pro.</p>
            <div class="price"><span class="amount">$20</span><span class="per">/month</span></div>
            <button class="cta" onClick={() => void subscribe()}>Subscribe — $20/month</button>
            <p class="fine">Unlimited wand + studio sessions · cancel anytime · payments by Stripe</p>
            <button class="dismiss" onClick={() => setShowPaywall(false)}>not now</button>
          </div>
        </div>
      </Show>
    </main>
  )
}

const CSS = `
  :root { --wand-bg:#07070d; }
  html, body { margin:0; padding:0; background:#07070d; overscroll-behavior:none; }
  .wand { --text:#f0ecff; --dim:#9a92b8; --accent:#c9a0ff; --gold:#ffd76a;
    --surface:#14121f; --border:#2a2440; --err:#ff6b8a;
    position:fixed; inset:0; overflow:hidden; background:#000; color:var(--text);
    font-family:system-ui,-apple-system,sans-serif; -webkit-user-select:none; user-select:none; }

  /* the stage fills everything; the generated stream is the interface */
  .wand .stage { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
    display:block; background:#000; }
  .wand .pip { position:absolute; right:14px; width:26%; min-width:104px; max-width:180px;
    aspect-ratio:3/4; object-fit:cover; border-radius:14px; border:1px solid rgba(255,255,255,.18);
    bottom:calc(190px + env(safe-area-inset-bottom)); opacity:.92; z-index:3;
    box-shadow:0 10px 30px rgba(0,0,0,.5); }
  .wand .pip.hidden { display:none; }

  /* floating chrome */
  .wand .top { position:absolute; top:0; left:0; right:0; z-index:6;
    display:flex; align-items:flex-start; justify-content:space-between; gap:10px;
    padding:calc(10px + env(safe-area-inset-top)) calc(14px + env(safe-area-inset-right))
            14px calc(14px + env(safe-area-inset-left));
    background:linear-gradient(to bottom, rgba(5,4,12,.72), rgba(5,4,12,0)); pointer-events:none; }
  .wand .top > * { pointer-events:auto; }
  .wand .top h1 { margin:0; font-size:19px; font-weight:700; letter-spacing:-.01em;
    display:flex; align-items:center; gap:8px; text-shadow:0 2px 12px rgba(0,0,0,.6); }
  .wand .sparkle { filter:drop-shadow(0 0 10px rgba(201,160,255,.85)); }
  .wand .top-right { display:flex; align-items:center; gap:8px; }
  .wand .chip { background:rgba(10,8,20,.62); color:var(--text); border:1px solid rgba(255,255,255,.16);
    border-radius:999px; padding:6px 12px; font:inherit; font-size:12px; cursor:pointer;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .wand .chip.countdown { color:var(--accent); font-variant-numeric:tabular-nums; font-weight:600; }
  .wand .chip.countdown.low { color:var(--err); border-color:rgba(255,107,138,.5); }
  .wand .chip.rec { color:#ff8aa5; }
  .wand .chip.rec.on { background:rgba(90,18,36,.8); color:#ff6b8a; border-color:#ff6b8a; }
  .wand .pro { font-size:10px; color:var(--gold); border:1px solid rgba(255,215,106,.45);
    background:rgba(60,45,10,.6); border-radius:999px; padding:2px 8px; font-weight:800; }

  /* idle takeover */
  .wand .overlay { position:absolute; inset:0; z-index:5; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:14px; text-align:center;
    padding:calc(70px + env(safe-area-inset-top)) 28px calc(200px + env(safe-area-inset-bottom));
    background:radial-gradient(ellipse at center, rgba(22,15,45,.9), rgba(5,4,12,.97)); }
  .wand .pitch { margin:0; font-size:34px; line-height:1.15; font-weight:800; letter-spacing:-.02em;
    background:linear-gradient(135deg,#fff,#c9a0ff); -webkit-background-clip:text;
    background-clip:text; color:transparent; }
  .wand .howto { color:var(--dim); max-width:340px; line-height:1.55; font-size:14px; margin:0; }
  .wand .howto b { color:var(--gold); }
  .wand .conn { color:var(--accent); font-size:16px; animation:wand-pulse 1.2s ease-in-out infinite; }
  .wand .cta { background:linear-gradient(135deg,#c9a0ff,#7f6aff); color:#0d0620; border:none;
    border-radius:999px; padding:15px 38px; font-size:16px; font-weight:800; cursor:pointer;
    box-shadow:0 8px 34px rgba(127,106,255,.5); }
  .wand .cta:active { transform:scale(.98); }
  .wand .free-hint { color:var(--dim); font-size:12px; margin:0; }

  /* bottom chrome */
  .wand .bottom { position:absolute; left:0; right:0; bottom:0; z-index:6;
    padding:20px calc(10px + env(safe-area-inset-right)) calc(12px + env(safe-area-inset-bottom))
            calc(10px + env(safe-area-inset-left));
    background:linear-gradient(to top, rgba(5,4,12,.85), rgba(5,4,12,0)); }
  .wand .spells { display:flex; gap:8px; overflow-x:auto; padding:2px 4px 8px;
    scrollbar-width:none; -webkit-overflow-scrolling:touch; }
  .wand .spells::-webkit-scrollbar { display:none; }
  .wand .spell { flex:0 0 auto; display:flex; flex-direction:column; align-items:center; gap:3px;
    background:rgba(20,18,31,.72); border:1px solid rgba(255,255,255,.14); border-radius:16px;
    padding:9px 14px; color:var(--dim); cursor:pointer; font:inherit; min-width:72px;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .wand .spell.on { border-color:var(--accent); color:var(--text); background:rgba(41,31,80,.85);
    box-shadow:0 0 20px rgba(201,160,255,.3); }
  .wand .spell .emoji { font-size:22px; line-height:1; }
  .wand .spell .name { font-size:11px; font-weight:600; }
  .wand .live-row { display:flex; align-items:center; gap:12px; padding:4px 6px 0; }
  .wand .stopbtn { background:rgba(20,18,31,.72); color:var(--text); border:1px solid rgba(255,255,255,.16);
    border-radius:999px; padding:9px 22px; font:inherit; font-size:13px; cursor:pointer;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .wand .err { color:var(--err); font-size:13px; margin:0; max-width:340px; }
  .wand .err.inline { font-size:12px; }

  /* paywall */
  .wand .pw-backdrop { position:fixed; inset:0; z-index:10; background:rgba(4,3,10,.8);
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    display:flex; align-items:center; justify-content:center; padding:22px; }
  .wand .pw { background:#141020; border:1px solid var(--border); border-radius:22px; padding:28px 28px 22px;
    max-width:380px; width:100%; text-align:center; box-shadow:0 24px 80px rgba(0,0,0,.65); }
  .wand .pw h2 { margin:0 0 8px; font-size:20px; }
  .wand .pw p { color:var(--dim); font-size:14px; margin:0 0 10px; }
  .wand .price { display:flex; align-items:baseline; justify-content:center; gap:4px; margin:8px 0 16px; }
  .wand .amount { font-size:44px; font-weight:800; color:var(--accent); }
  .wand .per { color:var(--dim); }
  .wand .pw .cta { width:100%; }
  .wand .fine { font-size:11px !important; margin-top:12px !important; }
  .wand .dismiss { background:transparent; border:1px solid var(--border); border-radius:10px;
    color:var(--dim); font-size:13px; cursor:pointer; padding:8px 18px; margin-top:8px; }

  @keyframes wand-pulse { 0%,100% { opacity:.5; } 50% { opacity:1; } }
  @media (min-width:900px){
    .wand .pitch { font-size:44px; }
    .wand .top h1 { font-size:22px; }
    .wand .spells { justify-content:center; }
  }
`
