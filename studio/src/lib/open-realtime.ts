/**
 * open-realtime — browser client for openstudio-server (wire contract v1).
 *
 * Self-hosted counterpart to the Decart SDK: one WebSocket to the pod (via the
 * SSH tunnel), JPEG frames out, restyled JPEG frames in, JSON control beside
 * them. The exported session is deliberately the exact duck-type studio.tsx
 * already uses on the Decart object (`rt.set({ prompt })` / `rt.close()`), so
 * the route's applyPrompt/stop/record flows work unchanged.
 *
 * Contract essentials implemented here (proto 1 — see openstudio-server/server.py):
 * - Handshake: first server message is `hello` (verify proto === 1, size all
 *   canvases from hello.width/height) or `busy` + close 1013. Then send one
 *   `prompt` with the UI's current prompt and start the ticker.
 * - Binary framing, all little-endian: client header 13 bytes <BId>
 *   (magic 0x01, seq, capture_ts_ms) + JPEG; server header 17 bytes <BIdf>
 *   (magic 0x02, echoed seq, echoed capture_ts_ms, infer_ms) + JPEG.
 * - Backpressure is newest-frame-wins on BOTH sides: pacing is TIME-based
 *   (ticker at 1000/targetFps), never ack-based — there is NO per-frame
 *   response guarantee, so in-flight accounting would wedge. A tick that can't
 *   send (socket not open, tab hidden, bufferedAmount over the 256 KiB
 *   high-water mark, encode still running, camera not ready) is DROPPED, never
 *   deferred; capture-at-send makes the live video element the client's 1-slot
 *   mailbox.
 * - Paint path: discard the first 2 outputs after connect (they drain the
 *   server's pipelined denoising buffer), ignore stale seq, paint immediately.
 * - Liveness: server emits `stats` ~1/s; 5 s of total silence => stalled
 *   indicator (socket kept). Ping every 10 s keeps the tunnel warm.
 * - Mirror is client-only, baked into the sent pixels at capture time.
 */

export type OpenStats = {
  e2eMs: number
  inferMs: number
  fpsOut: number
  stalled: boolean
}

export type OpenSession = {
  set(state: { prompt: string }): Promise<void>
  close(): void
}

export type OpenRealtimeOptions = {
  url: string
  prompt: string
  mirror: boolean
  targetFps?: number
  onRemoteStream: (s: MediaStream) => void
  onFatal: (msg: string) => void
  onStats?: (s: OpenStats) => void
}

// ---- wire constants (contract v1 defaults table) --------------------------- //
export const PROTO = 1
export const HDR_IN_SIZE = 13
export const HDR_OUT_SIZE = 17
const MAGIC_IN = 0x01
const MAGIC_OUT = 0x02
const CLIENT_JPEG_QUALITY = 0.75
const DEFAULT_TARGET_FPS = 15
const BUFFERED_HIGH_WATER = 262_144 // 256 KiB — the tunnel-congestion gate
const WARMUP_DISCARD = 2
const PING_PERIOD_MS = 10_000
const STALL_AFTER_MS = 5_000
const HELLO_TIMEOUT_MS = 10_000
const PROMPT_ACK_TIMEOUT_MS = 3_000

const TUNNEL_HINT =
  "start the SSH tunnel first: ssh -N -L 8765:127.0.0.1:8765 -p <ssh-port> root@<pod-ip> " +
  "(RunPod direct-TCP SSH — the ssh.runpod.io proxy cannot -L). On the https deploy, Safari " +
  "cannot open ws://localhost — use Chrome, Edge, or Firefox (dev at http://localhost:3000 is unconstrained)."

// ---- header codecs (pure; unit-tested against contract_test.py's struct vectors) //

/** 13-byte client→server frame header: <BId> magic 0x01, seq (u32), capture_ts_ms (f64). */
export function encodeFrameHeader(seq: number, captureTsMs: number): ArrayBuffer {
  const buf = new ArrayBuffer(HDR_IN_SIZE)
  const dv = new DataView(buf)
  dv.setUint8(0, MAGIC_IN)
  dv.setUint32(1, seq >>> 0, true)
  dv.setFloat64(5, captureTsMs, true)
  return buf
}

export type OutputHeader = { seq: number; captureTsMs: number; inferMs: number }

/** Parse the 17-byte server→client header <BIdf>; null when malformed/empty payload. */
export function decodeOutputHeader(buf: ArrayBuffer): OutputHeader | null {
  if (buf.byteLength <= HDR_OUT_SIZE) return null
  const dv = new DataView(buf)
  if (dv.getUint8(0) !== MAGIC_OUT) return null
  return {
    seq: dv.getUint32(1, true),
    captureTsMs: dv.getFloat64(5, true),
    inferMs: dv.getFloat32(13, true),
  }
}

// ---- capture geometry ------------------------------------------------------ //

/** Cover-fit the current video frame into (w, h): scale to cover + center-crop,
 * never letterbox; mirror is baked into the pixels here (the server never mirrors). */
function drawCover(
  ctx: OffscreenCanvasRenderingContext2D,
  video: HTMLVideoElement,
  w: number,
  h: number,
  mirror: boolean,
): boolean {
  const vw = video.videoWidth
  const vh = video.videoHeight
  if (vw === 0 || vh === 0) return false
  const scale = Math.max(w / vw, h / vh)
  const sw = w / scale
  const sh = h / scale
  const sx = (vw - sw) / 2
  const sy = (vh - sh) / 2
  ctx.save()
  if (mirror) {
    ctx.translate(w, 0)
    ctx.scale(-1, 1)
  }
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h)
  ctx.restore()
  return true
}

// ---- session --------------------------------------------------------------- //

export function connectOpenRealtime(camera: MediaStream, opts: OpenRealtimeOptions): Promise<OpenSession> {
  return new Promise<OpenSession>((resolve, reject) => {
    let ws: WebSocket
    try {
      ws = new WebSocket(opts.url)
    } catch (e) {
      reject(new Error(`invalid WebSocket URL "${opts.url}": ${e instanceof Error ? e.message : String(e)}`))
      return
    }
    ws.binaryType = "arraybuffer"

    const targetFps = opts.targetFps ?? DEFAULT_TARGET_FPS

    // -- session state ------------------------------------------------------ //
    let helloWidth = 0
    let helloHeight = 0
    let gotHello = false
    let settled = false // connect promise resolved/rejected
    let done = false // torn down (client close, fatal, or failed connect)
    let closedByClient = false
    let seq = 0 // next seq to SEND; +1 per sent frame, wrap mod 2^32
    let encodeInFlight = false
    let discardRemaining = WARMUP_DISCARD
    let lastPaintedSeq = -1
    let lastMsgAt = 0
    let e2eEma = 0
    let inferMs = 0
    let fpsOut = 0
    let stalled = false
    let tickerId: number | undefined
    let pingId: number | undefined
    let livenessId: number | undefined
    let pendingPromptAcks: Array<() => void> = []

    // Hidden, self-contained camera playback — decoupled from whatever video
    // element the page mounts; this element IS the 1-slot newest-frame mailbox.
    const video = document.createElement("video")
    video.muted = true
    video.setAttribute("playsinline", "")
    video.srcObject = camera
    void video.play().catch(() => {})

    // Output surface: real <canvas> (captureStream needs one), sized from hello.
    const outCanvas = document.createElement("canvas")
    let outCtx: CanvasRenderingContext2D | null = null
    let outStream: MediaStream | null = null

    // Capture scratch, sized from hello.
    let scratch: OffscreenCanvas | null = null
    let scratchCtx: OffscreenCanvasRenderingContext2D | null = null

    function emitStats() {
      opts.onStats?.({ e2eMs: e2eEma, inferMs, fpsOut, stalled })
    }

    function sendJson(obj: Record<string, unknown>) {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
    }

    function resolvePromptAcks() {
      const acks = pendingPromptAcks
      pendingPromptAcks = []
      for (const ack of acks) ack()
    }

    function onPagehide() {
      // Never leak the server's single session slot on navigation/tab close.
      closedByClient = true
      shutdown()
    }

    function shutdown() {
      if (done) return
      done = true
      window.clearTimeout(helloTimer)
      if (tickerId !== undefined) window.clearInterval(tickerId)
      if (pingId !== undefined) window.clearInterval(pingId)
      if (livenessId !== undefined) window.clearInterval(livenessId)
      window.removeEventListener("pagehide", onPagehide)
      resolvePromptAcks()
      try {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close(1000)
      } catch {
        // best-effort teardown
      }
      outStream?.getTracks().forEach((t) => t.stop())
      outStream = null
      video.pause()
      video.srcObject = null
    }

    function failConnect(msg: string) {
      if (settled) return
      settled = true
      shutdown()
      reject(new Error(msg))
    }

    function fatal(msg: string) {
      shutdown()
      opts.onFatal(msg)
    }

    const helloTimer = window.setTimeout(() => {
      failConnect(`no hello from ${opts.url} within 10 s — is openstudio-server running? ${TUNNEL_HINT}`)
    }, HELLO_TIMEOUT_MS)

    window.addEventListener("pagehide", onPagehide)

    // -- capture ticker: TIME-based pacing, newest frame wins ---------------- //
    function tick() {
      if (done || !scratch || !scratchCtx) return
      // The five skip-gates; a skipped tick's frame simply never exists.
      if (ws.readyState !== WebSocket.OPEN) return
      if (document.hidden) return
      if (ws.bufferedAmount > BUFFERED_HIGH_WATER) return
      if (encodeInFlight) return
      if (video.readyState < 2) return

      const captureTs = performance.now() // at capture, echoed verbatim by the server
      if (!drawCover(scratchCtx, video, helloWidth, helloHeight, opts.mirror)) return
      encodeInFlight = true
      scratch
        .convertToBlob({ type: "image/jpeg", quality: CLIENT_JPEG_QUALITY })
        .then((blob) => blob.arrayBuffer())
        .then((jpeg) => {
          if (done || ws.readyState !== WebSocket.OPEN) return
          const payload = new Uint8Array(HDR_IN_SIZE + jpeg.byteLength)
          payload.set(new Uint8Array(encodeFrameHeader(seq, captureTs)), 0)
          payload.set(new Uint8Array(jpeg), HDR_IN_SIZE)
          ws.send(payload)
          seq = (seq + 1) >>> 0 // +1 per SENT frame
        })
        .catch(() => {
          // encode failure = dropped frame; the next tick captures fresh
        })
        .finally(() => {
          encodeInFlight = false
        })
    }

    function startLoops() {
      lastMsgAt = performance.now()
      tickerId = window.setInterval(tick, 1000 / targetFps)
      pingId = window.setInterval(() => sendJson({ type: "ping", t: performance.now() }), PING_PERIOD_MS)
      livenessId = window.setInterval(() => {
        const nowStalled = performance.now() - lastMsgAt > STALL_AFTER_MS
        if (nowStalled !== stalled) {
          stalled = nowStalled
          emitStats()
        } else if (stalled) {
          emitStats() // keep the indicator fresh while frames are paused
        }
      }, 1000)
    }

    // -- inbound: binary frames --------------------------------------------- //
    function handleBinary(buf: ArrayBuffer) {
      const hdr = decodeOutputHeader(buf)
      if (hdr === null) return
      if (discardRemaining > 0) {
        // Warmup drain: the first 2 outputs after (re)connect carry pre-session
        // content from the server's pipelined denoising buffer.
        discardRemaining -= 1
        return
      }
      if (hdr.seq <= lastPaintedSeq) return // belt-and-suspenders vs reordering
      const jpeg = new Uint8Array(buf, HDR_OUT_SIZE)
      createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }))
        .then((bmp) => {
          if (done || outCtx === null || hdr.seq <= lastPaintedSeq) {
            bmp.close()
            return
          }
          lastPaintedSeq = hdr.seq
          outCtx.drawImage(bmp, 0, 0)
          bmp.close()
          const e2e = performance.now() - hdr.captureTsMs // same clock, no sync needed
          e2eEma = e2eEma === 0 ? e2e : 0.8 * e2eEma + 0.2 * e2e
        })
        .catch(() => {
          // undecodable output frame — drop it, keep the session
        })
    }

    // -- inbound: JSON control ---------------------------------------------- //
    function handleHello(msg: Record<string, unknown>) {
      if (gotHello) return
      window.clearTimeout(helloTimer)
      if (msg["proto"] !== PROTO) {
        failConnect(`server speaks proto ${String(msg["proto"])}, this client speaks ${PROTO} — update the older side`)
        return
      }
      const w = msg["width"]
      const h = msg["height"]
      if (typeof w !== "number" || typeof h !== "number" || w <= 0 || h <= 0) {
        failConnect("malformed hello: missing width/height")
        return
      }
      try {
        gotHello = true
        helloWidth = w
        helloHeight = h
        // Size EVERYTHING from hello — never hardcode 512.
        outCanvas.width = w
        outCanvas.height = h
        outCtx = outCanvas.getContext("2d")
        scratch = new OffscreenCanvas(w, h)
        scratchCtx = scratch.getContext("2d")
        if (outCtx === null || scratchCtx === null) {
          failConnect("could not create 2d canvas contexts")
          return
        }
        outCtx.fillStyle = "#000"
        outCtx.fillRect(0, 0, w, h) // first captureStream frame has real dimensions
        outStream = outCanvas.captureStream()
        opts.onRemoteStream(outStream)
        // The UI's prompt is authoritative — send it even if it equals hello.prompt.
        sendJson({ type: "prompt", text: opts.prompt })
        startLoops()
        settled = true
        resolve(session)
      } catch (e) {
        failConnect(e instanceof Error ? e.message : String(e))
      }
    }

    function handleJson(raw: string) {
      let parsed: unknown
      try {
        parsed = JSON.parse(raw)
      } catch {
        return
      }
      if (typeof parsed !== "object" || parsed === null) return
      const msg = parsed as Record<string, unknown>
      const type = msg["type"]
      if (type === "hello") {
        handleHello(msg)
      } else if (type === "busy") {
        // Server closes 1013 right after; reject now with the clearer message.
        failConnect("another studio session is live on the server — one at a time; stop it (or wait) and retry")
      } else if (type === "prompt_applied") {
        resolvePromptAcks() // coalescing: one ack can cover several set()s
      } else if (type === "stats") {
        const fo = msg["fps_out"]
        const ema = msg["infer_ms_ema"]
        if (typeof fo === "number") fpsOut = fo
        if (typeof ema === "number") inferMs = ema
        emitStats()
      } else if (type === "error") {
        // NON-FATAL by contract; the only fatal signal is WS close.
        console.warn("openstudio-server:", msg["message"])
      }
      // pong: nothing to do — any message already reset the liveness clock
    }

    ws.onmessage = (ev: MessageEvent) => {
      lastMsgAt = performance.now()
      if (typeof ev.data === "string") handleJson(ev.data)
      else if (ev.data instanceof ArrayBuffer) handleBinary(ev.data)
    }

    ws.onclose = (ev: CloseEvent) => {
      if (closedByClient || done) {
        shutdown()
        return
      }
      if (!settled) {
        failConnect(
          ev.code === 1013
            ? "another studio session is live on the server — one at a time; stop it (or wait) and retry"
            : `could not connect to ${opts.url} (close ${ev.code}${ev.reason ? `: ${ev.reason}` : ""}) — ${TUNNEL_HINT}`,
        )
        return
      }
      fatal(
        ev.code === 1013
          ? "the server dropped this session: another session is live"
          : `connection lost (close ${ev.code}${ev.reason ? `: ${ev.reason}` : ""}) — check openstudio-server and the SSH tunnel`,
      )
    }

    const session: OpenSession = {
      set(state: { prompt: string }): Promise<void> {
        if (done || ws.readyState !== WebSocket.OPEN) return Promise.resolve()
        sendJson({ type: "prompt", text: state.prompt })
        return new Promise<void>((resolveAck) => {
          const entry = () => {
            window.clearTimeout(timer)
            resolveAck()
          }
          // Fire-and-forget UX (matches Lucy): resolve on ack or after 3 s, never reject.
          const timer = window.setTimeout(() => {
            pendingPromptAcks = pendingPromptAcks.filter((a) => a !== entry)
            resolveAck()
          }, PROMPT_ACK_TIMEOUT_MS)
          pendingPromptAcks.push(entry)
        })
      },
      close() {
        closedByClient = true
        shutdown()
      },
    }
  })
}
