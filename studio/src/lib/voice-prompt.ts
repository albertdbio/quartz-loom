/**
 * voice-prompt — continuous speech-to-prompt for the studio.
 *
 * Wraps the Web Speech API (SpeechRecognition / webkitSpeechRecognition) into a
 * start/stop session that survives the engines' habit of ending themselves:
 * mobile Safari and Chrome both terminate recognition after silence or ~60 s,
 * so an enabled session auto-restarts on `onend` until the caller stops it.
 * Requires a secure context (https or localhost) and a user gesture to start —
 * both are the caller's responsibility.
 *
 * Each FINAL utterance is delivered via onFinal (the caller applies it as the
 * live prompt); interim fragments stream through onInterim for UI feedback.
 * Fatal states (permission denied, no engine) surface through onUnavailable
 * with a human-readable reason, after which the session is dead.
 */

type SpeechRecognitionLike = {
  lang: string
  continuous: boolean
  interimResults: boolean
  start(): void
  stop(): void
  abort(): void
  onresult: ((ev: any) => void) | null
  onend: (() => void) | null
  onerror: ((ev: any) => void) | null
}

export type VoiceSession = {
  stop(): void
}

export type VoiceOptions = {
  onInterim: (text: string) => void
  onFinal: (text: string) => void
  onUnavailable: (reason: string) => void
  /** BCP-47 tag; defaults to the browser language. */
  lang?: string
}

export function voiceSupported(): boolean {
  if (typeof window === "undefined") return false
  const w = window as any
  return Boolean(w.SpeechRecognition || w.webkitSpeechRecognition)
}

/** Minimum characters for a final utterance to count as a prompt edit. */
const MIN_FINAL_CHARS = 3
/** Backoff between auto-restarts so a hard-failing engine can't spin. */
const RESTART_DELAY_MS = 250

export function startVoicePrompt(opts: VoiceOptions): VoiceSession | null {
  const w = window as any
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition
  if (!Ctor) {
    opts.onUnavailable("voice input is not supported in this browser")
    return null
  }

  let stopped = false
  let restartTimer: number | undefined
  let rec: SpeechRecognitionLike | null = null

  function build(): SpeechRecognitionLike {
    const r: SpeechRecognitionLike = new Ctor()
    r.lang = opts.lang ?? navigator.language ?? "en-US"
    r.continuous = true
    r.interimResults = true

    r.onresult = (ev: any) => {
      let interim = ""
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i]
        const text = String(res[0]?.transcript ?? "").trim()
        if (!text) continue
        if (res.isFinal) {
          if (text.length >= MIN_FINAL_CHARS) opts.onFinal(text)
          opts.onInterim("")
        } else {
          interim = interim ? `${interim} ${text}` : text
        }
      }
      if (interim) opts.onInterim(interim)
    }

    r.onerror = (ev: any) => {
      const code = String(ev?.error ?? "")
      // Terminal errors: the engine will not recover without user action.
      if (code === "not-allowed" || code === "service-not-allowed") {
        stopped = true
        opts.onUnavailable(
          "microphone access was blocked — allow the mic for this site (browser address bar), then tap the mic again",
        )
      }
      // "no-speech" / "aborted" / "network" fall through to onend, which restarts.
    }

    r.onend = () => {
      if (stopped) return
      // The engine self-terminated (silence timeout, utterance boundary on iOS,
      // transient network error). Keep listening until the user stops us.
      restartTimer = window.setTimeout(() => {
        if (stopped) return
        try {
          rec = build()
          rec.start()
        } catch {
          // start() can throw if called while another instance is winding down;
          // one more onend will follow and re-enter here.
        }
      }, RESTART_DELAY_MS)
    }

    return r
  }

  try {
    rec = build()
    rec.start()
  } catch (e) {
    opts.onUnavailable(`could not start voice input: ${e instanceof Error ? e.message : String(e)}`)
    return null
  }

  return {
    stop() {
      stopped = true
      window.clearTimeout(restartTimer)
      try {
        if (rec) {
          rec.onend = null
          rec.abort()
        }
      } catch {
        // best-effort teardown
      }
      rec = null
    },
  }
}
