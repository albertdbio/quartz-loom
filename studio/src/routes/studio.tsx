import { createSignal, For, onCleanup, onMount, Show } from "solid-js"
import type { OpenStats } from "~/lib/open-realtime"
import { startVoicePrompt, voiceSupported, type VoiceSession } from "~/lib/voice-prompt"
import Onboarding, { type OnboardingChoice } from "~/components/onboarding"

/**
 * studio — real-time video editing with Decart's Lucy 2.5.
 *
 * Camera → Lucy 2.5 (video-to-video, WebRTC) → live edited stream, driven by a
 * text prompt you can change on the fly. The permanent DECART_API_KEY never
 * touches the browser: we fetch a short-lived token from /api/decart/token and
 * connect with that. All WebRTC/SDK work is client-only (dynamic import), so
 * this route SSRs to a static shell and wires up on interaction.
 *
 * A second, additive mode ("Open model") swaps Lucy for our self-hosted
 * openstudio-server (StreamDiffusion sd-turbo on a rented GPU): camera frames
 * go out as JPEG over one WebSocket — reachable only through an SSH tunnel to
 * localhost — and restyled frames paint onto a canvas whose captureStream
 * feeds the same output <video>/recorder. See ~/lib/open-realtime.ts.
 */
type Status = "idle" | "connecting" | "live" | "error"
type Mode = "lucy" | "open" | "story"
type Facing = "user" | "environment"

const WS_URL_KEY = "openstudio.wsUrl"
const STORY_WS_URL_KEY = "openstudio.storyWsUrl"
const DEFAULT_WS_URL = "ws://localhost:8765"
const STORY_DEFAULT_WS_URL = "ws://localhost:8766"

const PRESETS: ReadonlyArray<{ label: string; prompt: string }> = [
  { label: "Cyberpunk", prompt: "Restyle the scene as a neon cyberpunk city at night, cinematic rim lighting, rain-slick reflections." },
  { label: "Anime", prompt: "Convert to hand-drawn anime style, cel shading, expressive linework, vivid saturated colors." },
  { label: "Oil painting", prompt: "Transform into a textured oil painting, visible brushstrokes, warm classical palette." },
  { label: "Beach", prompt: "Replace the background with a bright tropical beach at golden hour, keeping the person sharp and consistent." },
  { label: "Snow", prompt: "Make it a heavy snowfall winter scene, cold blue tones, soft falling snow, breath fog." },
  { label: "Claymation", prompt: "Render everything as stop-motion claymation with fingerprinted clay textures and soft studio lighting." },
  { label: "Sunglasses", prompt: "Add stylish black sunglasses to the person; keep everything else natural and unchanged." },
  { label: "Underwater", prompt: "Submerge the scene underwater with caustic light rays, drifting bubbles, and a teal ocean tint." },
]

export default function Studio() {
  let inputVideo: HTMLVideoElement | undefined
  let outputVideo: HTMLVideoElement | undefined
  const [prompt, setPrompt] = createSignal(PRESETS[0]?.prompt ?? "")
  const [status, setStatus] = createSignal<Status>("idle")
  const [err, setErr] = createSignal("")
  const [mirror, setMirror] = createSignal(true)
  const [recording, setRecording] = createSignal(false)
  const [mode, setMode] = createSignal<Mode>("lucy")
  const [wsUrl, setWsUrl] = createSignal(DEFAULT_WS_URL)
  const [storyWsUrl, setStoryWsUrl] = createSignal(STORY_DEFAULT_WS_URL)
  const [storyEngine, setStoryEngine] = createSignal<"api" | "open">("api")
  const activeWsUrl = () => (mode() === "story" ? storyWsUrl() : wsUrl())
  const [openStats, setOpenStats] = createSignal<OpenStats | null>(null)
  const [facing, setFacing] = createSignal<Facing>("user")
  const [voiceOn, setVoiceOn] = createSignal(false)
  const [voiceHeard, setVoiceHeard] = createSignal("")
  const [hasVoice, setHasVoice] = createSignal(false)
  const [showOnboarding, setShowOnboarding] = createSignal(false)
  // Mode/URL are only switchable between sessions.
  const modeLocked = () => status() !== "idle" && status() !== "error"

  onMount(() => {
    // Route SSRs — localStorage exists only client-side.
    const saved = localStorage.getItem(WS_URL_KEY)
    if (saved) setWsUrl(saved)
    const savedStory = localStorage.getItem(STORY_WS_URL_KEY)
    if (savedStory) setStoryWsUrl(savedStory)
    setHasVoice(voiceSupported())
    if (localStorage.getItem("studio.onboarded") !== "1") setShowOnboarding(true)
    // On phones the natural subject is the world, not the selfie — default to
    // the back camera (and no mirror; mirroring a back camera reads backwards).
    if (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) {
      setFacing("environment")
      setMirror(false)
    }
  })

  function finishOnboarding(pick?: OnboardingChoice) {
    localStorage.setItem("studio.onboarded", "1")
    setShowOnboarding(false)
    if (pick && !modeLocked()) setMode(pick)
  }

  function switchFacing(next: Facing) {
    setFacing(next)
    // Sensible mirror default per lens; the checkbox still overrides after.
    setMirror(next === "user")
  }

  // Held across the session; `any` because the SDK types load only in the browser.
  let rt: any = null
  let stream: MediaStream | null = null
  let editedStream: MediaStream | null = null
  let recorder: MediaRecorder | null = null
  let chunks: Blob[] = []

  // -- story mode: the model dreams on its own output ------------------------- //
  // A source canvas (not a camera) is the input stream. Every animation frame we
  // paint the latest MODEL OUTPUT back onto it plus a whisper of noise — the
  // feedback keeps the image alive and evolving instead of converging to a fixed
  // point, and each spoken prompt bends the loop toward the new scene. A louder
  // noise "kick" on prompt changes helps the image escape the previous attractor.
  const STORY_SIZE = 512
  const STORY_FEEDBACK_NOISE = 0.055
  const STORY_KICK_NOISE = 0.3
  let storyCanvas: HTMLCanvasElement | null = null
  let storyCtx: CanvasRenderingContext2D | null = null
  let noiseCanvas: HTMLCanvasElement | null = null
  let storyRaf = 0
  let storyKickPending = false

  function makeNoiseCanvas(): HTMLCanvasElement {
    const c = document.createElement("canvas")
    c.width = 128
    c.height = 128
    const ctx = c.getContext("2d")!
    const img = ctx.createImageData(128, 128)
    for (let i = 0; i < img.data.length; i += 4) {
      const v = (Math.random() * 255) | 0
      img.data[i] = v
      img.data[i + 1] = (Math.random() * 255) | 0
      img.data[i + 2] = (Math.random() * 255) | 0
      img.data[i + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
    return c
  }

  function paintNoise(alpha: number) {
    if (!storyCtx || !noiseCanvas) return
    storyCtx.save()
    storyCtx.globalAlpha = alpha
    // Random offset each pass so the noise field is temporally fresh.
    const ox = -(Math.random() * 128) | 0
    const oy = -(Math.random() * 128) | 0
    for (let x = ox; x < STORY_SIZE; x += 128)
      for (let y = oy; y < STORY_SIZE; y += 128) storyCtx.drawImage(noiseCanvas, x, y)
    storyCtx.restore()
  }

  function startStoryFeedback() {
    const step = () => {
      if (!storyCtx) return
      // Feed the model's own output back in, then keep a little entropy alive.
      if (outputVideo && outputVideo.readyState >= 2) {
        // Slow zoom + micro-rotation: perpetual motion keeps the dream from
        // locking into a static texture attractor, and reads as camera drift.
        const c = STORY_SIZE / 2
        storyCtx.save()
        storyCtx.translate(c, c)
        storyCtx.scale(1.008, 1.008)
        storyCtx.rotate(0.0015)
        storyCtx.translate(-c, -c)
        storyCtx.drawImage(outputVideo, 0, 0, STORY_SIZE, STORY_SIZE)
        storyCtx.restore()
      }
      paintNoise(storyKickPending ? STORY_KICK_NOISE : STORY_FEEDBACK_NOISE)
      storyKickPending = false
      storyRaf = requestAnimationFrame(step)
    }
    storyRaf = requestAnimationFrame(step)
  }

  function stopStoryFeedback() {
    // onCleanup also runs during SSR disposal, where rAF does not exist.
    if (typeof cancelAnimationFrame !== "undefined" && storyRaf !== 0) cancelAnimationFrame(storyRaf)
    storyRaf = 0
    storyCanvas = null
    storyCtx = null
  }

  async function connectApi(input: MediaStream, mirrorMode: "auto" | false) {
    const { createDecartClient, models } = await import("@decartai/sdk")
    const model = models.realtime("lucy-2.5")
    const res = await fetch("/api/decart/token", { method: "POST" })
    if (!res.ok) {
      const body = await res.text().catch(() => "")
      throw new Error(`token mint failed (${res.status}) ${body}`)
    }
    const { apiKey } = (await res.json()) as { apiKey: string }
    const client = createDecartClient({ apiKey })
    return client.realtime.connect(input, {
      model,
      mirror: mirrorMode,
      onRemoteStream: (edited: MediaStream) => {
        editedStream = edited
        if (outputVideo) {
          outputVideo.srcObject = edited
          void outputVideo.play().catch(() => {})
        }
        if (mode() === "story") startStoryFeedback()
      },
      initialState: { prompt: { text: prompt(), enhance: true } },
    })
  }

  async function start() {
    setErr("")
    setStatus("connecting")
    try {
      if (mode() === "story") {
        // Voice-driven dream canvas: no camera at all. The input "camera" is a
        // canvas seeded with soft noise; the feedback loop takes over once the
        // first output arrives. Speaking is the steering wheel.
        setOpenStats(null)
        const { connectOpenRealtime } = await import("~/lib/open-realtime")
        storyCanvas = document.createElement("canvas")
        storyCanvas.width = STORY_SIZE
        storyCanvas.height = STORY_SIZE
        storyCtx = storyCanvas.getContext("2d")
        if (!storyCtx) throw new Error("could not create the story canvas")
        noiseCanvas = makeNoiseCanvas()
        // Seed: a dim gradient + a strong first noise pass gives the model
        // something to hallucinate from on the very first prompt.
        const g = storyCtx.createLinearGradient(0, 0, STORY_SIZE, STORY_SIZE)
        g.addColorStop(0, "#26313f")
        g.addColorStop(1, "#0b0e12")
        storyCtx.fillStyle = g
        storyCtx.fillRect(0, 0, STORY_SIZE, STORY_SIZE)
        paintNoise(0.5)
        stream = storyCanvas.captureStream(15)
        if (inputVideo) {
          inputVideo.srcObject = stream
          void inputVideo.play().catch(() => {})
        }
        if (storyEngine() === "api") {
          rt = await connectApi(stream, false)
        } else {
          rt = await connectOpenRealtime(stream, {
            url: activeWsUrl(),
            prompt: prompt(),
            mirror: false,
            onRemoteStream: (edited: MediaStream) => {
              editedStream = edited
              if (outputVideo) {
                outputVideo.srcObject = edited
                void outputVideo.play().catch(() => {})
              }
              startStoryFeedback()
            },
            onFatal: (msg: string) => {
              setErr(msg)
              setStatus("error")
              stop()
            },
            onStats: (s) => setOpenStats(s),
          })
        }
        setStatus("live")
        // Voice is the whole point of story mode — arm it with this same tap.
        if (hasVoice() && !voiceOn()) toggleVoice()
        return
      }
      if (mode() === "open") {
        // Self-hosted path: no Decart SDK, no token mint — one WebSocket to the
        // tunneled pod. Camera constraints are ideals; the client cover-crops
        // to the server's square anyway.
        setOpenStats(null)
        const { connectOpenRealtime } = await import("~/lib/open-realtime")
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
            facingMode: { ideal: facing() },
          },
          audio: false,
        })
        if (inputVideo) {
          inputVideo.srcObject = stream
          void inputVideo.play().catch(() => {})
        }
        rt = await connectOpenRealtime(stream, {
          url: wsUrl(),
          prompt: prompt(),
          mirror: mirror(),
          onRemoteStream: (edited: MediaStream) => {
            editedStream = edited
            if (outputVideo) {
              outputVideo.srcObject = edited
              void outputVideo.play().catch(() => {})
            }
          },
          onFatal: (msg: string) => {
            setErr(msg)
            setStatus("error")
            stop()
          },
          onStats: (s) => setOpenStats(s),
        })
        setStatus("live")
        return
      }
      const { models } = await import("@decartai/sdk")
      const model = models.realtime("lucy-2.5")

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          frameRate: model.fps,
          width: model.width,
          height: model.height,
          facingMode: { ideal: facing() },
        },
        audio: false,
      })
      if (inputVideo) {
        inputVideo.srcObject = stream
        void inputVideo.play().catch(() => {})
      }
      rt = await connectApi(stream, mirror() ? "auto" : false)
      setStatus("live")
    } catch (e) {
      const name = e instanceof Error ? e.name : ""
      const msg =
        name === "NotAllowedError" || name === "SecurityError"
          ? "Camera access was blocked. Allow the camera for this site (address-bar camera icon), then click start again."
          : name === "NotFoundError" || name === "OverconstrainedError"
            ? "No usable camera found. Connect a webcam and try again."
            : e instanceof Error
              ? e.message
              : String(e)
      setErr(msg)
      setStatus("error")
      stop()
    }
  }

  async function applyPrompt(text?: string) {
    const p = text ?? prompt()
    if (text !== undefined) setPrompt(text)
    if (mode() === "story") storyKickPending = true // help the loop escape the old scene
    if (!rt) return
    try {
      await rt.set({ prompt: p })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  // -- voice: each spoken utterance becomes the live prompt ------------------- //
  let voice: VoiceSession | null = null

  function stopVoice() {
    voice?.stop()
    voice = null
    setVoiceOn(false)
    setVoiceHeard("")
  }

  function toggleVoice() {
    if (voiceOn()) {
      stopVoice()
      return
    }
    setErr("")
    voice = startVoicePrompt({
      onInterim: (t) => setVoiceHeard(t),
      onFinal: (t) => {
        setVoiceHeard(t)
        void applyPrompt(t)
      },
      onUnavailable: (reason) => {
        setErr(reason)
        setVoiceOn(false)
        voice = null
      },
    })
    if (voice) setVoiceOn(true)
  }

  function toggleRecord() {
    if (recording()) {
      recorder?.stop()
      return
    }
    if (!editedStream) return
    chunks = []
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm"
    recorder = new MediaRecorder(editedStream, { mimeType: mime })
    recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data)
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `studio-${Date.now()}.webm`
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
    stopStoryFeedback()
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    editedStream = null
    if (status() === "live") setStatus("idle")
  }

  onCleanup(() => {
    stopVoice()
    stop()
  })

  return (
    <main class={`studio ${mode() !== "lucy" ? "open" : ""}`}>
      <style>{CSS}</style>
      <Show when={showOnboarding()}>
        <Onboarding onDone={finishOnboarding} />
      </Show>
      <header>
        <h1>
          studio <span class="tag">real-time</span>
          <button class="help" title="how this works" onClick={() => setShowOnboarding(true)}>?</button>
        </h1>
        <p>
          Point your camera, describe the edit, watch it transform live.
          <br />
          <span class="hint">Click <b>start</b> and allow the camera, then pick a preset or type a prompt (⌘/Ctrl+Enter to apply).</span>
        </p>
      </header>

      <div class="mode-row">
        <div class="modes" role="group" aria-label="model">
          <button class={`seg ${mode() === "lucy" ? "on" : ""}`} disabled={modeLocked()} onClick={() => setMode("lucy")}>
            API
          </button>
          <button class={`seg ${mode() === "open" ? "on" : ""}`} disabled={modeLocked()} onClick={() => setMode("open")}>
            Open model (self-hosted)
          </button>
          <button class={`seg ${mode() === "story" ? "on" : ""}`} disabled={modeLocked()} onClick={() => setMode("story")}>
            Story (voice canvas)
          </button>
        </div>
        <Show when={mode() === "story"}>
          <div class="modes" role="group" aria-label="story engine">
            <button class={`seg ${storyEngine() === "api" ? "on" : ""}`} disabled={modeLocked()} onClick={() => setStoryEngine("api")}>
              api
            </button>
            <button class={`seg ${storyEngine() === "open" ? "on" : ""}`} disabled={modeLocked()} onClick={() => setStoryEngine("open")}>
              self-hosted
            </button>
          </div>
        </Show>
        <Show when={mode() === "open" || (mode() === "story" && storyEngine() === "open")}>
          <input
            class="wsurl"
            type="text"
            value={activeWsUrl()}
            disabled={modeLocked()}
            spellcheck={false}
            title="openstudio-server WebSocket (story mode uses the story-tuned server instance)"
            placeholder={mode() === "story" ? STORY_DEFAULT_WS_URL : DEFAULT_WS_URL}
            onInput={(e) => {
              if (mode() === "story") {
                setStoryWsUrl(e.currentTarget.value)
                localStorage.setItem(STORY_WS_URL_KEY, e.currentTarget.value)
              } else {
                setWsUrl(e.currentTarget.value)
                localStorage.setItem(WS_URL_KEY, e.currentTarget.value)
              }
            }}
          />
        </Show>
      </div>

      <div class="stage">
        <figure>
          <video ref={inputVideo} autoplay playsinline muted />
          <figcaption>{mode() === "story" ? "canvas · feedback + noise" : "camera"}</figcaption>
        </figure>
        <figure>
          <video ref={outputVideo} autoplay playsinline muted />
          <figcaption>
            <span>{mode() === "story" ? (storyEngine() === "api" ? "dreamed · api" : "dreamed · self-hosted") : mode() === "open" ? "edited · self-hosted" : "edited · api"}</span>
            <Show when={mode() !== "lucy" && status() === "live" && openStats()} keyed>
              {(s) => (
                <span class={`hud ${s.stalled ? "stalled" : ""}`}>
                  {s.stalled ? "stalled · " : ""}
                  {s.fpsOut.toFixed(1)} fps · gpu {Math.round(s.inferMs)} ms · e2e {Math.round(s.e2eMs)} ms
                </span>
              )}
            </Show>
            <Show when={status() === "live"}>
              <button
                class={`rec ${recording() ? "on" : ""}`}
                onClick={toggleRecord}
                title={recording() ? "stop & download" : "record the edited stream"}
              >
                {recording() ? "■ stop & save" : "● record"}
              </button>
            </Show>
          </figcaption>
        </figure>
      </div>

      <div class="presets">
        <For each={PRESETS}>
          {(p) => (
            <button class="chip" onClick={() => applyPrompt(p.prompt)}>
              {p.label}
            </button>
          )}
        </For>
      </div>

      <div class="controls">
        <textarea
          rows={2}
          value={prompt()}
          onInput={(e) => setPrompt(e.currentTarget.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void applyPrompt()
          }}
          placeholder="Describe the edit — swap the background, restyle, add/remove objects…  (⌘/Ctrl+Enter to apply)"
        />
        <div class="row">
          <Show
            when={status() === "live"}
            fallback={
              <button class="primary" onClick={start} disabled={status() === "connecting"}>
                {status() === "connecting" ? "connecting…" : "start"}
              </button>
            }
          >
            <button class="primary" onClick={() => applyPrompt()}>
              apply prompt
            </button>
            <button onClick={stop}>stop</button>
          </Show>
          <Show when={hasVoice()}>
            <button
              class={`mic ${voiceOn() ? "on" : ""}`}
              onClick={toggleVoice}
              title={voiceOn() ? "stop voice control" : "speak edits — each sentence becomes the live prompt"}
            >
              {voiceOn() ? "🎙 listening" : "🎙 voice"}
            </button>
          </Show>
          <Show when={mode() !== "story"}>
          <div class="cams" role="group" aria-label="camera">
            <button
              class={`seg ${facing() === "user" ? "on" : ""}`}
              disabled={status() === "live" || status() === "connecting"}
              onClick={() => switchFacing("user")}
            >
              front
            </button>
            <button
              class={`seg ${facing() === "environment" ? "on" : ""}`}
              disabled={status() === "live" || status() === "connecting"}
              onClick={() => switchFacing("environment")}
            >
              back
            </button>
          </div>
          <label class="mirror">
            <input type="checkbox" checked={mirror()} onChange={(e) => setMirror(e.currentTarget.checked)} disabled={status() === "live"} />
            mirror
          </label>
          </Show>
          <span class={`status ${status()}`}>{status()}</span>
        </div>
        <Show when={voiceOn()}>
          <p class="heard">
            <span class="pulse" aria-hidden="true"></span>
            {voiceHeard() ? `“${voiceHeard()}”` : "listening — say an edit, e.g. “make it look like a snowstorm”"}
          </p>
        </Show>
        <Show when={err()}>
          <p class="err">{err()}</p>
        </Show>
      </div>
    </main>
  )
}

const CSS = `
  .studio { --bg:#0b0e12; --text:#e8edf2; --dim:#8a96a3; --accent:#6ea8fe; --surface:#151a21; --border:#232b35; --err:#ff6b6b;
    max-width: 1100px; margin: 0 auto; padding: 28px 20px 48px; color: var(--text);
    font-family: system-ui,-apple-system,sans-serif; }
  .studio header h1 { font-size: 34px; margin: 0 0 4px; display:flex; align-items:center; gap:12px; }
  .studio .help { width:26px; height:26px; border-radius:50%; padding:0; font-size:14px; color:#8a96a3;
    background:transparent; border:1px solid #232b35; line-height:1; }
  .studio .help:hover { color:#6ea8fe; border-color:#6ea8fe; }
  .studio .tag { font-size:12px; color:var(--accent); border:1px solid var(--border); border-radius:999px; padding:3px 10px; font-weight:500; }
  .studio header p { color: var(--dim); margin: 0 0 20px; line-height:1.5; }
  .studio .hint { font-size:13px; opacity:.75; }
  .studio .hint b { color: var(--text); font-weight:600; }
  .studio .stage { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width: 760px){ .studio .stage { grid-template-columns:1fr; } }
  .studio figure { margin:0; background:var(--surface); border:1px solid var(--border); border-radius:16px; overflow:hidden; }
  .studio video { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#000; }
  .studio figcaption { color:var(--dim); font-size:12px; padding:8px 12px; display:flex; align-items:center; justify-content:space-between; }
  .studio .rec { background:transparent; color:#ff8a8a; border:1px solid var(--border); border-radius:8px; padding:3px 10px; font-size:11px; cursor:pointer; }
  .studio .rec.on { background:#3a1414; color:#ff6b6b; border-color:#ff6b6b; }
  .studio .presets { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
  .studio .chip { background:var(--surface); color:var(--text); border:1px solid var(--border); border-radius:999px; padding:6px 14px; font:inherit; font-size:13px; cursor:pointer; }
  .studio .chip:hover { border-color:var(--accent); color:var(--accent); }
  .studio .controls { margin-top:14px; display:flex; flex-direction:column; gap:10px; }
  .studio textarea { width:100%; background:var(--surface); color:var(--text); border:1px solid var(--border);
    border-radius:12px; padding:12px 14px; font:inherit; resize:vertical; }
  .studio textarea:focus { outline:none; border-color:var(--accent); }
  .studio .row { display:flex; align-items:center; gap:14px; }
  .studio button { background:var(--surface); color:var(--text); border:1px solid var(--border); border-radius:10px;
    padding:10px 18px; font:inherit; cursor:pointer; }
  .studio button:hover:not(:disabled) { border-color:var(--accent); }
  .studio button.primary { background:var(--accent); color:#08131f; border-color:var(--accent); font-weight:600; }
  .studio button:disabled { opacity:.6; cursor:default; }
  .studio .mirror { color:var(--dim); font-size:13px; display:flex; align-items:center; gap:6px; cursor:pointer; }
  .studio .status { margin-left:auto; font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }
  .studio .status.live { color:#5ee6a8; }
  .studio .status.connecting { color:var(--accent); }
  .studio .status.error { color:var(--err); }
  .studio .err { color:var(--err); font-size:13px; margin:2px 0 0; }
  .studio .mode-row { display:flex; align-items:center; gap:10px; margin:0 0 14px; flex-wrap:wrap; }
  .studio .modes { display:inline-flex; border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .studio .seg { border:none; border-radius:0; background:transparent; color:var(--dim); padding:8px 14px; font-size:13px; }
  .studio .seg:hover:not(:disabled) { color:var(--accent); }
  .studio .seg.on { background:var(--accent); color:#08131f; font-weight:600; }
  .studio .wsurl { flex:1; min-width:260px; background:var(--surface); color:var(--text); border:1px solid var(--border);
    border-radius:10px; padding:8px 12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  .studio .wsurl:focus { outline:none; border-color:var(--accent); }
  .studio .hud { color:var(--dim); font-size:11px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .studio .hud.stalled { color:#ffb266; }
  .studio.open video { aspect-ratio:1/1; }
  .studio .cams { display:inline-flex; border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .studio .mic { border-radius:999px; font-size:13px; }
  .studio .mic.on { background:#3a1414; color:#ff8a8a; border-color:#ff6b6b; }
  .studio .heard { color:var(--accent); font-size:13px; margin:2px 0 0; display:flex; align-items:center; gap:8px; min-height:20px; }
  .studio .pulse { width:8px; height:8px; border-radius:50%; background:#ff6b6b; flex:0 0 auto;
    animation: studio-pulse 1.2s ease-in-out infinite; }
  @keyframes studio-pulse { 0%,100% { opacity:.35; transform:scale(.85);} 50% { opacity:1; transform:scale(1.15);} }
  @media (max-width: 760px){
    .studio { padding: 16px 12px 40px; }
    .studio header h1 { font-size: 26px; }
    .studio button { padding:12px 18px; min-height:44px; }
    .studio .seg { padding:10px 14px; }
    .studio .chip { padding:9px 14px; font-size:14px; }
    .studio .row { flex-wrap:wrap; row-gap:10px; }
    .studio .status { margin-left:0; flex-basis:100%; }
  }
`
