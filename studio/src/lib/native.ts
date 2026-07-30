/**
 * Talking to the native shell.
 *
 * The experience is a web page either way; the shell only adds things a page
 * cannot do for itself. Notification permission is the first of those: iOS
 * gives an app exactly ONE system prompt, so the page owns the timing (and
 * shows its own explanation first) while the shell owns the asking.
 *
 * Every call degrades to "unsupported" in a plain browser, so nothing here
 * needs guarding at the call site.
 */

export type NotificationStatus = "granted" | "denied" | "undetermined" | "unsupported"

interface NativeBridge {
  readonly version: number
  /** Shell can transcribe speech natively (WKWebView has no Web Speech API). */
  readonly speech?: boolean
}

interface NativeWindow extends Window {
  __mochiverseNative?: NativeBridge
  ReactNativeWebView?: { postMessage: (msg: string) => void }
}

function nativeWindow(): NativeWindow | null {
  return typeof window === "undefined" ? null : (window as NativeWindow)
}

/** True when running inside the app shell rather than a browser tab. */
export function isNative(): boolean {
  const w = nativeWindow()
  return !!w?.__mochiverseNative && !!w.ReactNativeWebView
}

/**
 * Round-trips one request to the shell. Resolves "unsupported" rather than
 * rejecting on timeout: a permission opt-in must never be able to wedge the
 * UI that called it.
 */
function ask(type: string, resultType: string, timeoutMs = 30_000): Promise<NotificationStatus> {
  const w = nativeWindow()
  if (!isNative() || !w) return Promise.resolve("unsupported")

  return new Promise((resolve) => {
    let done = false
    const finish = (status: NotificationStatus) => {
      if (done) return
      done = true
      w.removeEventListener("mochiverse:native", onReply as EventListener)
      clearTimeout(timer)
      resolve(status)
    }

    const onReply = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as { type?: string; status?: NotificationStatus }
        if (msg.type === resultType && msg.status) finish(msg.status)
      } catch {
        // a malformed reply is not worth breaking the page over
      }
    }

    const timer = setTimeout(() => finish("unsupported"), timeoutMs)
    w.addEventListener("mochiverse:native", onReply as EventListener)
    w.ReactNativeWebView?.postMessage(JSON.stringify({ v: 1, type }))
  })
}

/** Current permission state, without raising the OS prompt. */
export function notificationStatus(): Promise<NotificationStatus> {
  return ask("notifications:status", "notifications:result", 5_000)
}

/** Raises the OS prompt. Only call this after the user has opted in on our side. */
export function requestNotifications(): Promise<NotificationStatus> {
  return ask("notifications:request", "notifications:result")
}

/**
 * The OS-level microphone prompt, raised natively. The WebView's own
 * getUserMedia grant is separate — callers still do that — but without the
 * OS grant first, iOS fails the in-page request with no prompt at all.
 */
export function requestNativeMic(): Promise<NotificationStatus> {
  return ask("mic:request", "mic:result")
}

/** True when the shell can transcribe speech natively. */
export function nativeSpeechAvailable(): boolean {
  const w = nativeWindow()
  return isNative() && w?.__mochiverseNative?.speech === true
}

export type NativeSpeechCallbacks = {
  onInterim: (text: string) => void
  onFinal: (text: string) => void
  /** Session died (permission, engine, or iOS ending it); the caller decides on restarts. */
  onEnd: (reason: string | null) => void
}

export type NativeSpeechSession = { stop(): void }

/**
 * Speech through the shell: SFSpeechRecognizer on the other side of the
 * bridge, surfaced with the same shape as the Web Speech session so the
 * caller cannot tell which engine is listening.
 */
export function startNativeSpeech(cb: NativeSpeechCallbacks): NativeSpeechSession | null {
  const w = nativeWindow()
  if (!nativeSpeechAvailable() || !w) return null

  let stopped = false
  const onEvent = (event: MessageEvent<string>) => {
    if (stopped) return
    try {
      const msg = JSON.parse(event.data) as { type?: string; text?: string; reason?: string }
      if (msg.type === "speech:interim" && msg.text) cb.onInterim(msg.text)
      else if (msg.type === "speech:final" && msg.text) cb.onFinal(msg.text)
      else if (msg.type === "speech:error") finish(msg.reason ?? "error")
      else if (msg.type === "speech:end") finish(null)
    } catch {
      // not ours
    }
  }
  const finish = (reason: string | null) => {
    if (stopped) return
    stopped = true
    w.removeEventListener("mochiverse:native", onEvent as EventListener)
    cb.onEnd(reason)
  }
  w.addEventListener("mochiverse:native", onEvent as EventListener)
  w.ReactNativeWebView?.postMessage(JSON.stringify({ v: 1, type: "speech:start" }))
  return {
    stop() {
      if (stopped) return
      stopped = true
      w.removeEventListener("mochiverse:native", onEvent as EventListener)
      w.ReactNativeWebView?.postMessage(JSON.stringify({ v: 1, type: "speech:stop" }))
    },
  }
}

/** Remembers that we already made our pitch, so it is never shown twice. */
export const NOTIFY_ASKED_KEY = "mochiverse.notify.asked.v1"

export function alreadyAsked(): boolean {
  try {
    return localStorage.getItem(NOTIFY_ASKED_KEY) === "1"
  } catch {
    return false
  }
}

export function markAsked(): void {
  try {
    localStorage.setItem(NOTIFY_ASKED_KEY, "1")
  } catch {
    // private mode — worst case the pitch appears once more
  }
}
