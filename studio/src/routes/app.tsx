import { createSignal, For, onCleanup, onMount, Show } from "solid-js"
import AppOnboarding from "~/components/app-onboarding"
import SmsSignIn from "~/components/sms-signin"
import Mochi, { type MochiMood } from "~/components/mochi"
import { startVoicePrompt, voiceSupported, type VoiceSession } from "~/lib/voice-prompt"
import { cropSpriteToContent } from "~/lib/sprite"
import { alreadyAsked, isNative, markAsked, notificationStatus, requestNotifications } from "~/lib/native"
import {
  loadProfile,
  rankTransforms,
  saveProfile,
  TRANSFORMS,
  personalLine,
  starterTransformIndex,
  type UserProfile,
} from "~/lib/transforms"

/**
 * Mochiverse ✨ — touch anything, transform it.
 *
 * A product skin over the realtime restyle engine (Decart Lucy 2.5): the
 * camera streams full-screen, and a "transform" prompt instructs the model to
 * transform WHATEVER THE HAND (or a held object) TOUCHES, spreading from the
 * contact point while everything untouched stays photoreal. The interaction
 * is physical — you reach out and touch the object on camera; the model does
 * the segmentation-by-description. Transforms switch live via rt.set().
 *
 * Billing rides the same gated /api/decart/token mint as the studio: first
 * minute free (server-burned), then Studio Pro. See server/plan.ts.
 */
/**
 * Mochiverse is a restyle product, so it runs Decart's restyle model: same
 * vendor and quality tier as lucy-2.5 at half the per-second price, which is
 * the difference between a viable free tier and a money incinerator.
 */
const REALTIME_MODEL = "lucy-restyle-2"

type Status = "idle" | "connecting" | "live" | "error"

export default function App() {
  let outputVideo: HTMLVideoElement | undefined
  let pipVideo: HTMLVideoElement | undefined
  const [status, setStatus] = createSignal<Status>("idle")
  const [err, setErr] = createSignal("")
  const [transform, setTransform] = createSignal(0)
  const [recording, setRecording] = createSignal(false)
  const [plan, setPlan] = createSignal<"loading" | "free" | "pro" | "member">("loading")
  const [payments, setPayments] = createSignal(false)
  const [freeLeft, setFreeLeft] = createSignal<number | null>(null)
  const [showPaywall, setShowPaywall] = createSignal(false)
  const [paywallReason, setPaywallReason] = createSignal<"exhausted" | "time-up">("exhausted")
  const [memberLeft, setMemberLeft] = createSignal<number | null>(null)
  const [quotaSpent, setQuotaSpent] = createSignal(false)
  // The notification pitch rides on the out-of-minutes sheet rather than a
  // modal of its own: iOS allows one system prompt per install, so it is worth
  // spending only at the moment "tell me when they reset" is actually useful.
  const [canNotify, setCanNotify] = createSignal(false)
  const [notifyOn, setNotifyOn] = createSignal(false)

  async function maybeOfferNotify() {
    if (!isNative() || alreadyAsked()) return
    setCanNotify((await notificationStatus()) === "undetermined")
  }

  async function optInNotify() {
    markAsked()
    setCanNotify(false)
    setNotifyOn((await requestNotifications()) === "granted")
  }
  // Mochi stands in the scene. Position is a percentage of the stage so she
  // stays put across rotations and resizes; tapping re-places her, which is
  // what makes her feel present rather than pasted on.
  const [mochiOn, setMochiOn] = createSignal(false)
  // A generated friend replaces Mochi inside the same scaffold. The sprite is
  // a data URL in localStorage: characters should survive reloads without a
  // media store, and 300KB of PNG is cheap against a 5MB quota.
  const CHARACTER_KEY = "mochiverse.character.v1"
  const [sprite, setSprite] = createSignal<string | null>(null)
  const [friendOpen, setFriendOpen] = createSignal(false)
  const [friendPrompt, setFriendPrompt] = createSignal("")
  const [friendBusy, setFriendBusy] = createSignal(false)
  const [friendErr, setFriendErr] = createSignal("")
  // Voice: spoken words become the live prompt. The insertion experiment
  // showed the model re-hallucinates entities per frame, so voice is framed as
  // "changing the weather", not "commanding characters" — the overlay owns
  // the character.
  const [voiceOn, setVoiceOn] = createSignal(false)
  // True while the friend is riding INSIDE the outbound stream (the model
  // re-renders them as scene content). The DOM overlay hides for the
  // duration — otherwise the friend appears twice.
  const [fusionActive, setFusionActive] = createSignal(false)
  const [voiceHeard, setVoiceHeard] = createSignal("")
  let voiceSession: VoiceSession | null = null

  function loadCharacter() {
    try {
      const raw = localStorage.getItem(CHARACTER_KEY)
      if (!raw) return
      const c = JSON.parse(raw) as { sprite?: string }
      if (typeof c.sprite === "string" && c.sprite.startsWith("data:image/")) {
        setSprite(c.sprite)
        setMochiOn(true)
      }
    } catch {
      // corrupted entry — the friend can be re-made
    }
  }

  async function makeFriend() {
    const desc = friendPrompt().trim()
    if (desc.length < 3 || friendBusy()) return
    setFriendBusy(true)
    setFriendErr("")
    try {
      const res = await fetch("/api/character", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: desc }),
      })
      const body = (await res.json()) as { sprite?: string; error?: string }
      if (!res.ok || !body.sprite) throw new Error(body.error ?? "could not create your friend")
      // crop to opaque content so the scaffold renders honest pixels
      const cropped = await cropSpriteToContent(body.sprite).catch(() => body.sprite!)
      setSprite(cropped)
      setMochiOn(true)
      setFriendOpen(false)
      try {
        localStorage.setItem(CHARACTER_KEY, JSON.stringify({ sprite: cropped, prompt: desc, at: Date.now() }))
      } catch {
        // quota/private mode — the friend lives for this session only
      }
      reactMochi("wow")
    } catch (e) {
      setFriendErr(e instanceof Error ? e.message : String(e))
    } finally {
      setFriendBusy(false)
    }
  }

  function toggleVoice() {
    if (voiceOn()) {
      voiceSession?.stop()
      voiceSession = null
      setVoiceOn(false)
      setVoiceHeard("")
      return
    }
    voiceSession = startVoicePrompt({
      onInterim: (t) => setVoiceHeard(t),
      onFinal: (t) => {
        setVoiceHeard(t)
        void speakTransform(t)
      },
      onUnavailable: (reason) => {
        setErr(reason)
        setVoiceOn(false)
      },
    })
    if (voiceSession) setVoiceOn(true)
  }

  /** A spoken phrase becomes the live prompt, styled like the deck's prompts. */
  async function speakTransform(text: string) {
    if (!rt) return
    reactMochi("happy")
    try {
      await rt.set({ prompt: { text, enhance: true } })
    } catch {
      // a dropped set is invisible; the next utterance retries naturally
    }
  }
  const [mochiPos, setMochiPos] = createSignal({ x: 78, y: 62 })
  const [mochiMood, setMochiMood] = createSignal<MochiMood>("idle")
  let moodTimer: ReturnType<typeof setTimeout> | null = null

  function reactMochi(mood: MochiMood) {
    if (moodTimer) clearTimeout(moodTimer)
    setMochiMood(mood)
    moodTimer = setTimeout(() => setMochiMood("idle"), 900)
  }

  function placeMochi(e: MouseEvent | TouchEvent) {
    const host = e.currentTarget as HTMLElement
    const r = host.getBoundingClientRect()
    const pt = "touches" in e ? e.touches[0] : (e as MouseEvent)
    if (!pt) return
    setMochiPos({
      x: Math.min(92, Math.max(8, ((pt.clientX - r.left) / r.width) * 100)),
      y: Math.min(88, Math.max(20, ((pt.clientY - r.top) / r.height) * 100)),
    })
    reactMochi("wow")
  }
  // Raw-camera PiP is OFF by default: on iOS WKWebView a <video> playing a
  // local capture stream composites ABOVE other video layers regardless of
  // z-index, burying the generated stream. The effect reads better without it
  // anyway — you can see reality with your own eyes.
  const [showPip, setShowPip] = createSignal(false)
  // Onboarding answers re-rank the deck and choose the opening transform, so the
  // questions visibly pay off instead of being a survey.
  const [profile, setProfile] = createSignal<UserProfile | null>(null)
  const [showOnboarding, setShowOnboarding] = createSignal(false)
  const deck = () => {
    const p = profile()
    return p ? rankTransforms(TRANSFORMS, p.goal) : TRANSFORMS
  }
  const intro = () => {
    const p = profile()
    return p ? personalLine(p.craft, p.goal, deck()[0]?.name ?? "") : ""
  }

  let rt: any = null
  let stream: MediaStream | null = null
  let editedStream: MediaStream | null = null
  let recorder: MediaRecorder | null = null
  let chunks: Blob[] = []
  let freeTimer: ReturnType<typeof setInterval> | null = null

  async function refreshPlan() {
    try {
      const res = await fetch("/api/billing/status")
      if (!res.ok) return
      const st = (await res.json()) as {
        plan: "free" | "pro" | "member"
        paymentsEnabled?: boolean
        remainingSeconds?: number
      }
      setPlan(st.plan)
      setPayments(st.paymentsEnabled === true)
      if (st.plan === "member") setMemberLeft(st.remainingSeconds ?? null)
    } catch {
      // status is cosmetic — the token route is the real gate
    }
  }

  onMount(() => {
    void fetch("/api/billing/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((s: { plan: "free" | "pro" | "member"; paymentsEnabled?: boolean } | null) => {
        if (!s) return
        setPlan(s.plan)
        setPayments(s.paymentsEnabled === true)
      })
      .catch(() => {})
    const saved = loadProfile()
    if (saved) setProfile(saved)
    else setShowOnboarding(true)
    loadCharacter()
  })

  function finishOnboarding(p?: UserProfile) {
    setShowOnboarding(false)
    if (!p) return
    setProfile(p)
    saveProfile(p)
    setTransform(starterTransformIndex(deck(), p.goal))
    // Fire-and-forget: two enum answers, no identifiers, so the founder can
    // see what people come here to make. Never blocks the experience.
    void fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ craft: p.craft, goal: p.goal }),
    }).catch(() => {})
  }

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
        if (plan() === "member") {
          // a session ended; whether they can start another is the balance's call
          if ((memberLeft() ?? 0) <= 0) {
            setQuotaSpent(true)
            void maybeOfferNotify()
          }
        } else {
          setPaywallReason("time-up")
          setShowPaywall(true)
        }
      }
    }, 1000)
  }

  async function signOut() {
    try {
      await fetch("/api/auth/logout", { method: "POST" })
    } catch {
      // best effort — the cookies are cleared server-side or not at all
    }
    setPlan("free")
    await refreshPlan()
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
      const model = models.realtime(REALTIME_MODEL)
      const res = await fetch("/api/decart/token", { method: "POST" })
      if (res.status === 402) {
        const denial = (await res.json().catch(() => ({}))) as { quota?: boolean }
        if (denial.quota) {
          setQuotaSpent(true)
          setMemberLeft(0)
        } else {
          setPaywallReason("exhausted")
          setShowPaywall(true)
        }
        setStatus("idle")
        return
      }
      if (!res.ok) throw new Error(`could not start (${res.status})`)
      const minted = (await res.json()) as {
        apiKey: string
        plan?: "free" | "pro" | "member"
        freeSeconds?: number
        remainingSeconds?: number
      }

      // Rear camera by default — this is a walk-around-and-touch toy.
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          frameRate: model.fps,
          width: model.width,
          height: model.height,
          facingMode: { ideal: "environment" },
        },
        audio: false,
      })

      // FUSION (dev flag ?fusion=1): draw the friend INTO the outbound frames
      // so the model re-renders character and scene together — the experiment
      // that decides whether "in the world" can beat "on the glass". Kept off
      // the shipped path until the evidence says otherwise.
      let sendStream = stream
      // Fusion is the default whenever a friend exists — the experiment showed
      // the model preserves a composited character, restyles it WITH the scene
      // (8-bit made him pixel art; Midas turned him gold), and the per-frame
      // re-compositing is itself the persistence mechanism. `?nofusion` keeps
      // an escape hatch for debugging.
      if (sprite() && !new URLSearchParams(window.location.search).has("nofusion")) {
        const spriteImg = new Image()
        spriteImg.src = sprite()!
        const raw = document.createElement("video")
        raw.srcObject = stream
        raw.muted = true
        await raw.play().catch(() => {})
        const canvas = document.createElement("canvas")
        canvas.width = model.width
        canvas.height = model.height
        const ctx = canvas.getContext("2d")!
        const draw = () => {
          if (!stream) return
          ctx.drawImage(raw, 0, 0, canvas.width, canvas.height)
          if (spriteImg.complete && spriteImg.naturalWidth > 0) {
            // mirror the on-screen anchor: position as % of frame, ~28% tall,
            // with a slow bob so the model renders someone alive, not a decal
            const h = canvas.height * 0.28
            const w = h * (spriteImg.naturalWidth / spriteImg.naturalHeight)
            const bob = Math.sin(performance.now() / 650) * canvas.height * 0.008
            ctx.drawImage(
              spriteImg,
              (mochiPos().x / 100) * canvas.width - w / 2,
              (mochiPos().y / 100) * canvas.height - h * 0.9 + bob,
              w,
              h,
            )
          }
          requestAnimationFrame(draw)
        }
        draw()
        sendStream = canvas.captureStream(Number(model.fps) || 25)
        setFusionActive(true)
      }
      if (pipVideo && showPip()) {
        pipVideo.srcObject = stream
        void pipVideo.play().catch(() => {})
      }
      const client = createDecartClient({ apiKey: minted.apiKey })
      rt = await client.realtime.connect(sendStream, {
        model,
        mirror: false,
        onRemoteStream: (edited: MediaStream) => {
          editedStream = edited
          if (outputVideo) {
            outputVideo.srcObject = edited
            void outputVideo.play().catch(() => {})
          }
        },
        initialState: { prompt: { text: deck()[transform()]!.prompt, enhance: true } },
      })
      setStatus("live")
      if (minted.plan === "free") {
        setPlan("free")
        startCountdown(minted.freeSeconds ?? 60)
      } else if (minted.plan === "member") {
        setPlan("member")
        setMemberLeft(minted.remainingSeconds ?? null)
        startCountdown(minted.freeSeconds ?? 120)
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

  async function applyTransform(i: number) {
    setTransform(i)
    reactMochi("happy")
    if (!rt) return
    try {
      await rt.set({ prompt: deck()[i]!.prompt })
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
      a.download = `mochiverse-${Date.now()}.webm`
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
    setFusionActive(false)
    stopCountdown()
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    editedStream = null
    if (status() === "live") setStatus("idle")
  }

  onCleanup(() => {
    voiceSession?.stop()
    stop()
  })

  return (
    <main class="mochiverse">
      <style>{CSS}</style>

      <Show when={showOnboarding()}>
        <AppOnboarding onDone={finishOnboarding} />
      </Show>

      {/* the stage: generated stream, edge to edge */}
      <video
        ref={outputVideo}
        class="stage"
        autoplay
        playsinline
        muted
        onClick={placeMochi}
      />
      <video ref={pipVideo} class="pip" classList={{ hidden: !showPip() }} autoplay playsinline muted />

      {/* Mochi stands in the scene, anchored in stage space */}
      <Show when={mochiOn() && !(status() === "live" && fusionActive())}>
        <div class="mochi-anchor" style={{ left: `${mochiPos().x}%`, top: `${mochiPos().y}%` }}>
          <Mochi size={sprite() ? 172 : 116} mood={mochiMood()} sprite={sprite() ?? undefined} />
        </div>
      </Show>

      {/* top chrome — floats over the stage, inset out of the notch */}
      <div class="top">
        <h1>
          Mochiverse <span class="sparkle">✨</span>
          <Show when={plan() === "pro" || plan() === "member"}>
            <span class="pro">
              {plan() === "pro"
                ? "PRO"
                : memberLeft() === null
                  ? "✓"
                  : `${Math.ceil((memberLeft() ?? 0) / 60)}m left`}
            </span>
            <button class="signout" onClick={() => void signOut()} title="sign out">
              sign out
            </button>
          </Show>
        </h1>
        <div class="top-right">
          <button
            class={`chip ${mochiOn() ? "on" : ""}`}
            onClick={() => setMochiOn(!mochiOn())}
            title={mochiOn() ? "hide your friend" : "show your friend"}
          >
            {mochiOn() ? "🫧" : "🫥"}
          </button>
          <button class="chip" onClick={() => setFriendOpen(true)} title="create a friend">
            ✨
          </button>
          <Show when={status() === "live" && voiceSupported()}>
            <button
              class={`chip ${voiceOn() ? "on" : ""}`}
              onClick={toggleVoice}
              title={voiceOn() ? "stop voice control" : "speak to change the scene"}
            >
              🎙
            </button>
          </Show>
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
          <Show when={status() !== "connecting"} fallback={<p class="conn">warming up…</p>}>
            <h2 class="pitch">Touch anything.<br />Transform it.</h2>
            <p class="howto">
              Point the camera at the world, hold out your hand — or grab a pencil —
              and <b>touch something</b>.
            </p>
            <Show when={intro()}>
              <p class="intro">{intro()}</p>
            </Show>
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

      {/* bottom chrome — transforms + stop, inset off the home indicator */}
      <div class="bottom">
        <div class="transforms">
          <For each={deck()}>
            {(s, i) => (
              <button
                class={`transform ${transform() === i() ? "on" : ""}`}
                onClick={() => void applyTransform(i())}
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

      <Show when={voiceOn() && voiceHeard()}>
        <p class="voice-heard">“{voiceHeard()}”</p>
      </Show>

      <Show when={friendOpen()}>
        <div class="pw-backdrop" onClick={(e) => e.target === e.currentTarget && setFriendOpen(false)}>
          <div class="pw" role="dialog" aria-modal="true" aria-label="create a friend">
            <h2>Dream up a friend ✨</h2>
            <p>Describe anyone — they'll appear in your world.</p>
            <input
              class="friend-input"
              placeholder="a tiny dragon in pajamas"
              maxlength="120"
              value={friendPrompt()}
              onInput={(e) => setFriendPrompt(e.currentTarget.value)}
              onKeyDown={(e) => e.key === "Enter" && void makeFriend()}
            />
            <Show when={friendErr()}>
              <p class="err inline">{friendErr()}</p>
            </Show>
            <button class="cta" disabled={friendBusy()} onClick={() => void makeFriend()}>
              {friendBusy() ? "dreaming…" : "bring them to life"}
            </button>
            <button class="dismiss" onClick={() => setFriendOpen(false)}>not now</button>
          </div>
        </div>
      </Show>

      <Show when={quotaSpent()}>
        <div class="pw-backdrop" onClick={(e) => e.target === e.currentTarget && setQuotaSpent(false)}>
          <div class="pw" role="dialog" aria-modal="true" aria-label="monthly limit">
            <h2>That's this month's transforms ✨</h2>
            <p>
              You've used your minutes for the month. They reset on the 1st — and
              we're working on more.
            </p>
            <Show when={canNotify()}>
              <button class="cta" onClick={() => void optInNotify()}>
                Notify me when they reset
              </button>
            </Show>
            <Show when={notifyOn()}>
              <p class="fine">We'll let you know the moment your minutes are back.</p>
            </Show>
            <button class="dismiss" onClick={() => setQuotaSpent(false)}>ok</button>
          </div>
        </div>
      </Show>
      <Show when={showPaywall() && !payments()}>
        <SmsSignIn
          reason={paywallReason()}
          onDismiss={() => setShowPaywall(false)}
          onDone={() => {
            setShowPaywall(false)
            setPlan("member")
            void fetch("/api/billing/status")
          }}
        />
      </Show>
      <Show when={showPaywall() && payments()}>
        <div class="pw-backdrop" onClick={(e) => e.target === e.currentTarget && setShowPaywall(false)}>
          <div class="pw" role="dialog" aria-modal="true" aria-label="Studio Pro">
            <h2>Your free minute is up ✨</h2>
            <p>Keep transforming as long as you like with Studio Pro.</p>
            <div class="price"><span class="amount">$20</span><span class="per">/month</span></div>
            <button class="cta" onClick={() => void subscribe()}>Subscribe — $20/month</button>
            <p class="fine">Unlimited mochiverse + studio sessions · cancel anytime · payments by Stripe</p>
            <button class="dismiss" onClick={() => setShowPaywall(false)}>not now</button>
          </div>
        </div>
      </Show>
    </main>
  )
}

const CSS = `
  :root { --stage-bg:#07070d; }
  html, body { margin:0; padding:0; background:#07070d; overscroll-behavior:none; }
  .mochiverse { --text:#f0ecff; --dim:#9a92b8; --accent:#c9a0ff; --gold:#ffd76a;
    --surface:#14121f; --border:#2a2440; --err:#ff6b8a;
    position:fixed; inset:0; overflow:hidden; background:#000; color:var(--text);
    font-family:system-ui,-apple-system,sans-serif; -webkit-user-select:none; user-select:none; }

  /* the stage fills everything; the generated stream is the interface */
  .mochiverse .stage { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
    display:block; background:#000; }
  .mochiverse .pip { position:absolute; right:14px; width:26%; min-width:104px; max-width:180px;
    aspect-ratio:3/4; object-fit:cover; border-radius:14px; border:1px solid rgba(255,255,255,.18);
    bottom:calc(190px + env(safe-area-inset-bottom)); opacity:.92; z-index:3;
    box-shadow:0 10px 30px rgba(0,0,0,.5); }
  .mochiverse .pip.hidden { display:none; }
  /* Mochi is anchored in stage space and never eats taps meant for the video. */
  .mochiverse .mochi-anchor { position:absolute; z-index:3; transform:translate(-50%,-50%);
    pointer-events:none; transition:left .45s cubic-bezier(.34,1.3,.64,1),
    top .45s cubic-bezier(.34,1.3,.64,1); }
  .mochiverse .chip.on { border-color:var(--accent); color:var(--accent); }
  .mochiverse .voice-heard { position:absolute; left:50%; bottom:118px; transform:translateX(-50%);
    z-index:6; margin:0; max-width:80%; color:var(--accent); font-size:13px; text-align:center;
    background:rgba(10,8,20,.6); border-radius:999px; padding:6px 14px;
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .mochiverse .friend-input { width:100%; box-sizing:border-box; margin:6px 0 12px;
    background:rgba(10,8,20,.6); color:var(--text); border:1px solid rgba(255,255,255,.2);
    border-radius:12px; padding:12px 14px; font:inherit; font-size:15px; }
  .mochiverse .friend-input:focus { outline:none; border-color:var(--accent); }

  /* floating chrome */
  .mochiverse .top { position:absolute; top:0; left:0; right:0; z-index:6;
    display:flex; align-items:flex-start; justify-content:space-between; gap:10px;
    padding:calc(10px + env(safe-area-inset-top)) calc(14px + env(safe-area-inset-right))
            14px calc(14px + env(safe-area-inset-left));
    background:linear-gradient(to bottom, rgba(5,4,12,.72), rgba(5,4,12,0)); pointer-events:none; }
  .mochiverse .top > * { pointer-events:auto; }
  .mochiverse .top h1 { margin:0; font-size:19px; font-weight:700; letter-spacing:-.01em;
    display:flex; align-items:center; gap:8px; text-shadow:0 2px 12px rgba(0,0,0,.6); }
  .mochiverse .sparkle { filter:drop-shadow(0 0 10px rgba(201,160,255,.85)); }
  .mochiverse .top-right { display:flex; align-items:center; gap:8px; }
  .mochiverse .chip { background:rgba(10,8,20,.62); color:var(--text); border:1px solid rgba(255,255,255,.16);
    border-radius:999px; padding:6px 12px; font:inherit; font-size:12px; cursor:pointer;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .mochiverse .chip.countdown { color:var(--accent); font-variant-numeric:tabular-nums; font-weight:600; }
  .mochiverse .chip.countdown.low { color:var(--err); border-color:rgba(255,107,138,.5); }
  .mochiverse .chip.rec { color:#ff8aa5; }
  .mochiverse .chip.rec.on { background:rgba(90,18,36,.8); color:#ff6b8a; border-color:#ff6b8a; }
  .mochiverse .signout { font-size:10px; color:var(--dim); background:rgba(10,8,20,.55);
    border:1px solid rgba(255,255,255,.14); border-radius:999px; padding:4px 9px; cursor:pointer;
    font-family:inherit; }
  .mochiverse .signout:hover { color:var(--text); }
  .mochiverse .pro { font-size:10px; color:var(--gold); border:1px solid rgba(255,215,106,.45);
    background:rgba(60,45,10,.6); border-radius:999px; padding:2px 8px; font-weight:800; }

  /* idle takeover */
  .mochiverse .overlay { position:absolute; inset:0; z-index:5; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:14px; text-align:center;
    padding:calc(70px + env(safe-area-inset-top)) 28px calc(200px + env(safe-area-inset-bottom));
    background:radial-gradient(ellipse at center, rgba(22,15,45,.9), rgba(5,4,12,.97)); }
  .mochiverse .pitch { margin:0; font-size:34px; line-height:1.15; font-weight:800; letter-spacing:-.02em;
    background:linear-gradient(135deg,#fff,#c9a0ff); -webkit-background-clip:text;
    background-clip:text; color:transparent; }
  .mochiverse .howto { color:var(--dim); max-width:340px; line-height:1.55; font-size:14px; margin:0; }
  .mochiverse .howto b { color:var(--gold); }
  .mochiverse .conn { color:var(--accent); font-size:16px; animation:stage-pulse 1.2s ease-in-out infinite; }
  .mochiverse .cta { background:linear-gradient(135deg,#c9a0ff,#7f6aff); color:#0d0620; border:none;
    border-radius:999px; padding:15px 38px; font-size:16px; font-weight:800; cursor:pointer;
    box-shadow:0 8px 34px rgba(127,106,255,.5); }
  .mochiverse .cta:active { transform:scale(.98); }
  .mochiverse .free-hint { color:var(--dim); font-size:12px; margin:0; }
  .mochiverse .intro { margin:0; color:var(--gold); font-size:13.5px; line-height:1.5; max-width:340px; }

  /* bottom chrome */
  .mochiverse .bottom { position:absolute; left:0; right:0; bottom:0; z-index:6;
    padding:20px calc(10px + env(safe-area-inset-right)) calc(12px + env(safe-area-inset-bottom))
            calc(10px + env(safe-area-inset-left));
    background:linear-gradient(to top, rgba(5,4,12,.85), rgba(5,4,12,0)); }
  .mochiverse .transforms { display:flex; gap:8px; overflow-x:auto; padding:2px 4px 8px;
    scrollbar-width:none; -webkit-overflow-scrolling:touch; }
  .mochiverse .transforms::-webkit-scrollbar { display:none; }
  .mochiverse .transform { flex:0 0 auto; display:flex; flex-direction:column; align-items:center; gap:3px;
    background:rgba(20,18,31,.72); border:1px solid rgba(255,255,255,.14); border-radius:16px;
    padding:9px 14px; color:var(--dim); cursor:pointer; font:inherit; min-width:72px;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .mochiverse .transform.on { border-color:var(--accent); color:var(--text); background:rgba(41,31,80,.85);
    box-shadow:0 0 20px rgba(201,160,255,.3); }
  .mochiverse .transform .emoji { font-size:22px; line-height:1; }
  .mochiverse .transform .name { font-size:11px; font-weight:600; }
  .mochiverse .live-row { display:flex; align-items:center; gap:12px; padding:4px 6px 0; }
  .mochiverse .stopbtn { background:rgba(20,18,31,.72); color:var(--text); border:1px solid rgba(255,255,255,.16);
    border-radius:999px; padding:9px 22px; font:inherit; font-size:13px; cursor:pointer;
    backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .mochiverse .err { color:var(--err); font-size:13px; margin:0; max-width:340px; }
  .mochiverse .err.inline { font-size:12px; }

  /* paywall */
  .mochiverse .pw-backdrop { position:fixed; inset:0; z-index:10; background:rgba(4,3,10,.8);
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    display:flex; align-items:center; justify-content:center; padding:22px; }
  .mochiverse .pw { background:#141020; border:1px solid var(--border); border-radius:22px; padding:28px 28px 22px;
    max-width:380px; width:100%; text-align:center; box-shadow:0 24px 80px rgba(0,0,0,.65); }
  .mochiverse .pw h2 { margin:0 0 8px; font-size:20px; }
  .mochiverse .pw p { color:var(--dim); font-size:14px; margin:0 0 10px; }
  .mochiverse .price { display:flex; align-items:baseline; justify-content:center; gap:4px; margin:8px 0 16px; }
  .mochiverse .amount { font-size:44px; font-weight:800; color:var(--accent); }
  .mochiverse .per { color:var(--dim); }
  .mochiverse .pw .cta { width:100%; }
  .mochiverse .fine { font-size:11px !important; margin-top:12px !important; }
  .mochiverse .dismiss { background:transparent; border:1px solid var(--border); border-radius:10px;
    color:var(--dim); font-size:13px; cursor:pointer; padding:8px 18px; margin-top:8px; }

  @keyframes stage-pulse { 0%,100% { opacity:.5; } 50% { opacity:1; } }
  @media (min-width:900px){
    .mochiverse .pitch { font-size:44px; }
    .mochiverse .top h1 { font-size:22px; }
    .mochiverse .transforms { justify-content:center; }
  }
`
