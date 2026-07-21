#!/usr/bin/env python3
"""openstudio-server — real-time webcam restyle over one binary WebSocket.

Open-model counterpart to studio's Lucy 2.5 mode: the browser sends camera
frames as JPEG over a WebSocket, this server runs StreamDiffusion (SD-Turbo,
512x512, 2-step img2img) per frame on the pod GPU, and sends restyled JPEG
frames back. Prompt is hot-swappable without pipeline re-init.

DEPLOYMENT SHAPE (v1, tunnel-only — this is why there is no auth):
  pod$    ./run.sh                                   # binds 127.0.0.1:8765 ONLY
  laptop$ ssh -N -L 8765:127.0.0.1:8765 -p <ssh-port> root@<pod-ip> -i ~/.ssh/id_ed25519
  browser ws://localhost:8765                        # from the studio dev server page

  The server never listens on a routable interface, the pod template exposes no
  TCP port for 8765, and the only route in is the key-authenticated SSH tunnel.
  NOTE: RunPod's proxy SSH (…@ssh.runpod.io) does NOT support -L forwarding —
  use the pod's *direct TCP* SSH (public IP + mapped port). Add real auth
  before ever binding anything other than 127.0.0.1.

WIRE CONTRACT v1 (all multi-byte integers/floats little-endian)
===============================================================
Binary client -> server (a camera frame):
    offset 0  u8   magic = 0x01
    offset 1  u32  seq           client's monotonically increasing frame counter
    offset 5  f64  capture_ts_ms client clock at capture (echoed back verbatim;
                                 the server never interprets it — latency HUD
                                 subtracts it from the client's own clock)
    offset 13 ...  JPEG bytes    ideally already center-cropped square + scaled
                                 to 512x512 by the client (saves ~3.5x upstream
                                 bandwidth vs 720p); the server defensively
                                 cover-fits whatever arrives.

Binary server -> client (a restyled frame):
    offset 0  u8   magic = 0x02
    offset 1  u32  seq           seq of the consumed input frame (see NOTE)
    offset 5  f64  capture_ts_ms echoed from that input frame
    offset 13 f32  infer_ms      GPU pipeline time for this frame (server-side)
    offset 17 ...  JPEG bytes    512x512
    NOTE: StreamDiffusion's batched-denoising pipelining (t_index_list of len 2)
    means the *pixels* of output tagged seq N are a blend dominated by input
    N-1. The echoed timestamp still bounds true glass-to-glass latency.
    Clients should discard the first 2 output frames after connect (they drain
    the warmup buffer).

JSON (text) client -> server:
    {"type":"prompt","text":"..."}          hot-swap prompt (cheap, ~5-20 ms:
                                            one CLIP text-encoder forward via
                                            StreamDiffusion.update_prompt)
    {"type":"config","jpeg_quality":30-95}  runtime-safe knobs only. Resolution
                                            / t_index / model are BOOT flags:
                                            latent buffers are sized in
                                            prepare() and cannot change live.
    {"type":"ping","t":<number>}            app-level RTT probe

JSON (text) server -> client:
    {"type":"hello","server":"openstudio-server","proto":1,"pipeline":"...",
     "model":"...","width":512,"height":512,"t_index":[35,45],
     "prompt":"...","jpeg_quality":80,
     "vae":"taesd"|"full","cfg_type":"none"|"self","guidance_scale":1.0,
     "delta":1.0,"noise_mode":"add"|"deterministic",
     "similar_filter":null|<threshold>,"similar_max_skip":10,
     "lcm_lora":false,"seed":2}             first message after connect;
                                            reports every boot-time quality
                                            lever (additive since proto 1)
    {"type":"prompt_applied","text":"...","ms":<float>}
    {"type":"config_applied","jpeg_quality":<int>}
    {"type":"pong","t":<number>}
    {"type":"stats","fps_in":..,"fps_out":..,"infer_ms_ema":..,
     "dropped_stale":..,"dropped_outbox":..}   every ~1 s
    {"type":"busy","message":"..."}         second concurrent client; then
                                            close(code=1013). v1 is one
                                            session at a time — one GPU, one
                                            latent state.
    {"type":"error","message":"..."}

BACKPRESSURE (the core design decision):
  Ingress:  a 1-slot "latest frame wins" mailbox between the asyncio receiver
            and the GPU worker thread. A new frame overwrites an unprocessed
            one (counted as dropped_stale). The GPU never works on stale input
            and nothing ever queues.
  Egress:   at most 1 restyled frame pending per connection; a newer result
            replaces it (dropped_outbox). Control JSON is never dropped.
  Client:   MAY firehose at camera fps (the mailbox absorbs it), but the
            recommended client policy is to keep <=2 frames in flight
            (send only while sent_seq - last_received_seq < 2).

PROMPT HOT-SWAP — why cfg_type MUST stay "none" (or "self"):
  StreamDiffusion.update_prompt() re-encodes with do_classifier_free_guidance
  =False and REPLACES prompt_embeds wholesale (pipeline.py L255-262). Under
  cfg_type "initialize"/"full" with guidance>1, prepare() built prompt_embeds
  as cat[uncond, cond] (2x batch, pipeline.py L168-180) — a hot-swap would
  shape-mismatch the UNet batch. The official realtime webcam demo
  (demo/realtime-img2img/img2img.py) ships sd-turbo + t_index=[35,45] +
  cfg_type="none"; that is the default. --cfg-type self (RCFG) is the one safe
  upgrade: prepare() under "self" builds cond-only embeds exactly like "none",
  so update_prompt stays shape-safe, and the stock-noise residual adds virtual
  negative guidance for ~zero extra UNet cost — but ONLY when --guidance-scale
  > 1.0 (at 1.0 the residual term cancels; the server warns). "initialize"/
  "full" stay excluded. Consequence: negative prompts are a NO-OP in v1 under
  both modes (no uncond branch exists) — deliberately absent from the wire
  contract.

QUALITY LEVERS (all BOOT-time flags — latent buffers are sized in prepare()):
  --model + --lcm-lora       SD1.5-family checkpoint; finetunes (e.g.
                             Lykon/dreamshaper-8) need --lcm-lora, *turbo not
  --t-index i0,i1[,i2]       restyle strength ~ (50-i0)/50: [35,45]=0.30
                             subtle, [32,45]=0.36, [25,40]=0.50 strong — same
                             cost for len 2; len 3 costs +50% UNet
  --cfg-type self --guidance-scale 1.4 [--delta 1.0]  RCFG virtual negatives
  --noise-mode deterministic do_add_noise=False: the authors' vid2vid
                             anti-flicker latent path (softer, stabler)
  --similar-filter 0.98 [--similar-max-skip 10]  skip near-identical inputs
                             and replay the previous output (kills
                             static-scene shimmer; needs warmup >= 1)
  --full-vae                 full SD VAE instead of TAESD (+18-25 ms, crisper)
  --seed N                   init-noise seed (the fixed noise reused per frame)
bench_quality.py sweeps combinations of these offline, scores flicker + CLIP
prompt adherence + fps, and emits per-config `server_flags` strings that boot
this server in exactly the benched config (see README).

Run modes:
  python server.py                          # serve (GPU, StreamDiffusion)
  python server.py --pipeline fake         # serve with a CPU colormap stand-in
                                            # (transport dev on a laptop, no
                                            # torch/model needed)
  python server.py --selfcheck             # no client, no socket: synthesize
                                            # frames, run the exact frame path,
                                            # swap prompt midway, print timings,
                                            # write in/out JPEGs, exit 0/1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

log = logging.getLogger("openstudio")

PROTO_VERSION = 1
MAGIC_IN = 0x01
MAGIC_OUT = 0x02
HDR_IN = struct.Struct("<BId")  # magic, seq, capture_ts_ms          (13 bytes)
HDR_OUT = struct.Struct("<BIdf")  # magic, seq, capture_ts_ms, infer_ms (17 bytes)
MAX_WS_MESSAGE = 8 * 1024 * 1024


# --------------------------------------------------------------------------- #
# image geometry
# --------------------------------------------------------------------------- #

def cover_fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale so the image covers (w, h), then center-crop — no letterboxing.

    Letterboxing was rejected: black bars burn ~28% of the UNet's 512x512
    pixel budget and the diffusion model hallucinates content into them.
    Center-crop matches the Lucy UI, which displays with object-fit: cover.
    """
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * scale)), max(h, round(ih * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def synth_frame(i: int, w: int = 1280, h: int = 720) -> np.ndarray:
    """Synthetic webcam stand-in (BGR): gradient + orbiting circle + index."""
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.stack(
        [
            (xx * 255 // max(w - 1, 1)).astype(np.uint8),
            (yy * 255 // max(h - 1, 1)).astype(np.uint8),
            np.full((h, w), 96, np.uint8),
        ],
        axis=-1,
    )
    ang = i * 0.35
    cx = int(w / 2 + np.cos(ang) * w / 4)
    cy = int(h / 2 + np.sin(ang) * h / 4)
    cv2.circle(frame, (cx, cy), h // 6, (40, 40, 230), -1)
    cv2.putText(frame, f"frame {i}", (32, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return frame


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #

class FakePipeline:
    """CPU stand-in so transport/client work needs no GPU, torch, or model.

    Prompt-dependent on purpose: the colormap is chosen by hashing the prompt,
    so a prompt swap visibly changes the output during client dev.
    """

    name = "fake"
    _MAPS = [cv2.COLORMAP_OCEAN, cv2.COLORMAP_INFERNO, cv2.COLORMAP_SPRING,
             cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_AUTUMN]

    def __init__(self, prompt: str, delay_ms: float = 0.0) -> None:
        self.prompt = prompt
        self.delay_ms = delay_ms  # simulate GPU latency for local pacing tests

    def process(self, rgb: np.ndarray) -> np.ndarray:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cmap = self._MAPS[hash(self.prompt) % len(self._MAPS)]
        bgr = cv2.applyColorMap(gray, cmap)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def set_prompt(self, text: str) -> float:
        t0 = time.perf_counter()
        self.prompt = text
        return (time.perf_counter() - t0) * 1000.0

    def warmup(self, n: int) -> None:
        for i in range(n):
            self.process(cv2.cvtColor(synth_frame(i, 512, 512), cv2.COLOR_BGR2RGB))


class StreamPipeline:
    """StreamDiffusion sd-turbo img2img, driven directly (no repo `utils/`).

    NOTE: StreamDiffusionWrapper lives in the repo's UN-packaged `utils/` dir —
    `pip install streamdiffusion` does not ship it. We therefore build the
    pipeline exactly the way the README + realtime-img2img demo do, from the
    packaged `streamdiffusion` module only.
    """

    name = "streamdiffusion"

    def __init__(
        self,
        model_id: str,
        prompt: str,
        width: int,
        height: int,
        t_index_list: list[int],
        use_tiny_vae: bool,
        use_xformers: bool,
        cfg_type: str = "none",
        guidance_scale: float = 1.0,
        delta: float = 1.0,
        do_add_noise: bool = True,
        similar_filter: Optional[float] = None,
        similar_max_skip: int = 10,
        use_lcm_lora: bool = False,
        seed: int = 2,
    ) -> None:
        import torch  # lazy: --pipeline fake / local dev never imports torch
        from diffusers import AutoencoderTiny, StableDiffusionPipeline
        from PIL import Image
        from streamdiffusion import StreamDiffusion
        from streamdiffusion.image_utils import postprocess_image

        self._torch = torch
        self._Image = Image
        self._postprocess = postprocess_image
        self.width, self.height = width, height

        # cfg_type MUST be none/self for cheap update_prompt — see module docstring.
        if cfg_type not in ("none", "self"):
            raise ValueError(
                f"cfg_type must be 'none' or 'self' (got {cfg_type!r}) — "
                "'initialize'/'full' would shape-mismatch update_prompt"
            )
        if cfg_type == "self" and guidance_scale <= 1.0:
            log.warning(
                "cfg_type=self with guidance_scale=%.2f is a no-op "
                "(model_pred collapses to the text branch at gs<=1.0) — raise --guidance-scale",
                guidance_scale,
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available — StreamPipeline needs the pod GPU")

        log.info("loading %s (fp16) ...", model_id)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")

        self.stream = StreamDiffusion(
            pipe,
            t_index_list=t_index_list,
            torch_dtype=torch.float16,
            width=width,
            height=height,
            do_add_noise=do_add_noise,
            frame_buffer_size=1,
            use_denoising_batch=True,
            cfg_type=cfg_type,
        )
        # sd-turbo needs no LCM-LoRA (the official wrapper skips it for *turbo*);
        # plain SD1.5 finetunes (dreamshaper-8, kohaku-v2.1) DO need it fused.
        if use_lcm_lora:
            if "turbo" in model_id.lower():
                log.warning("--lcm-lora on a *turbo* model is usually wrong (already few-step); proceeding")
            log.info("loading + fusing latent-consistency/lcm-lora-sdv1-5 ...")
            self.stream.load_lcm_lora()
            self.stream.fuse_lora()
        if use_tiny_vae:
            self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                device=pipe.device, dtype=pipe.dtype
            )
        if use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        # Default attention is torch SDPA (AttnProcessor2_0) — fine on 4090.

        if similar_filter is not None:
            # Skips inference when input cosine-similarity > threshold and
            # replays prev_image_result (upstream pipeline.py __call__); the
            # warmup below guarantees a prev result exists before any skip.
            self.stream.enable_similar_image_filter(
                threshold=similar_filter, max_skip_frame=similar_max_skip
            )

        self.stream.prepare(
            prompt,
            "",
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            delta=delta,
            seed=seed,
        )
        log.info(
            "pipeline prepared: t_index=%s %dx%d tiny_vae=%s cfg=%s gs=%.2f delta=%.2f "
            "add_noise=%s similar_filter=%s lcm_lora=%s seed=%d",
            t_index_list, width, height, use_tiny_vae, cfg_type, guidance_scale, delta,
            do_add_noise, similar_filter, use_lcm_lora, seed,
        )

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 (H, W, 3) at model size -> restyled rgb uint8 512x512."""
        out = self.stream(self._Image.fromarray(rgb))
        np_img = self._postprocess(out.cpu(), output_type="np")[0]  # float32 [0,1] HWC
        return (np_img * 255.0).round().astype(np.uint8)

    def set_prompt(self, text: str) -> float:
        """Hot-swap: ONE CLIP text-encoder forward (pipeline.py update_prompt)."""
        t0 = time.perf_counter()
        self.stream.update_prompt(text)
        return (time.perf_counter() - t0) * 1000.0

    def warmup(self, n: int) -> None:
        """Fill cudnn/SDPA autotune caches AND the pipelined denoising buffer.

        First real frames would otherwise pay one-off compile costs and drain
        garbage from the empty latent buffer.
        """
        for i in range(n):
            self.process(cv2.cvtColor(synth_frame(i, self.width, self.height), cv2.COLOR_BGR2RGB))
        self._torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# frame processor — the ONE code path both the server and --selfcheck run
# --------------------------------------------------------------------------- #

@dataclass
class FrameTimings:
    decode_ms: float
    infer_ms: float
    encode_ms: float


class FrameProcessor:
    """decode JPEG -> cover-fit -> pipeline -> encode JPEG. Worker-thread-owned."""

    def __init__(self, pipeline: Any, width: int, height: int, jpeg_quality: int) -> None:
        self.pipeline = pipeline
        self.width, self.height = width, height
        self.jpeg_quality = jpeg_quality

    def process_jpeg(self, jpeg: bytes) -> tuple[Optional[bytes], Optional[FrameTimings]]:
        t0 = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None
        rgb = cv2.cvtColor(cover_fit(bgr, self.width, self.height), cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        out_rgb = self.pipeline.process(rgb)
        t2 = time.perf_counter()
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        t3 = time.perf_counter()
        if not ok:
            return None, None
        return bytes(buf), FrameTimings(
            decode_ms=(t1 - t0) * 1000.0,
            infer_ms=(t2 - t1) * 1000.0,
            encode_ms=(t3 - t2) * 1000.0,
        )


# --------------------------------------------------------------------------- #
# ingress mailbox (thread-safe, latest frame wins) + egress outbox (loop-side)
# --------------------------------------------------------------------------- #

class FrameMailbox:
    """1-slot mailbox: putting over an unconsumed frame replaces it (drop-stale)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Optional[tuple[int, float, bytes]] = None
        self.dropped_stale = 0

    def put(self, seq: int, ts: float, jpeg: bytes) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped_stale += 1
            self._item = (seq, ts, jpeg)
            self._cond.notify()

    def take(self, timeout: float) -> Optional[tuple[int, float, bytes]]:
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        with self._cond:
            self._item = None


class OutBox:
    """Loop-side egress buffer: control JSON is FIFO and never dropped; at most
    ONE binary frame is pending — a newer frame replaces it (latest wins)."""

    def __init__(self) -> None:
        self._ctrl: deque[str] = deque()
        self._frame: Optional[bytes] = None
        self._ev = asyncio.Event()
        self.dropped_outbox = 0

    def push_json(self, obj: dict[str, Any]) -> None:
        self._ctrl.append(json.dumps(obj, separators=(",", ":")))
        self._ev.set()

    def push_frame(self, payload: bytes) -> None:
        if self._frame is not None:
            self.dropped_outbox += 1
        self._frame = payload
        self._ev.set()

    async def pop(self) -> str | bytes:
        while True:
            if self._ctrl:
                return self._ctrl.popleft()
            if self._frame is not None:
                out, self._frame = self._frame, None
                return out
            self._ev.clear()
            await self._ev.wait()


# --------------------------------------------------------------------------- #
# GPU worker thread
# --------------------------------------------------------------------------- #

class Worker:
    """Owns the pipeline exclusively. One instance per process, started warm at
    boot; websocket sessions attach/detach without touching the pipeline.

    Threading model: torch CUDA ops and cv2 release the GIL for their heavy
    parts, so one worker thread + the asyncio loop is enough for v1 (the
    official demo uses multiprocessing only because it is multi-user).
    """

    def __init__(self, processor: FrameProcessor) -> None:
        self.processor = processor
        self.mailbox = FrameMailbox()
        self._control: deque[tuple[str, Any]] = deque()  # thread-safe appends
        self._emit_lock = threading.Lock()
        self._emit: Optional[Callable[[str, Any], None]] = None  # kind, payload
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        # counters (read via snapshot())
        self._lock = threading.Lock()
        self.frames_in = 0
        self.frames_out = 0
        self.decode_failures = 0
        self.infer_ms_ema = 0.0

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # -- session attach (loop thread) --------------------------------------- #
    def attach(self, emit: Callable[[str, Any], None]) -> None:
        with self._emit_lock:
            self._emit = emit
        self.mailbox.clear()

    def detach(self) -> None:
        with self._emit_lock:
            self._emit = None
        self.mailbox.clear()

    # -- inputs (loop thread) ----------------------------------------------- #
    def submit_frame(self, seq: int, ts: float, jpeg: bytes) -> None:
        with self._lock:
            self.frames_in += 1
        self.mailbox.put(seq, ts, jpeg)

    def submit_control(self, kind: str, payload: Any) -> None:
        # No wake-up needed: the worker's mailbox.take(timeout=0.1) bounds
        # control latency to <=100 ms even when no frames are flowing.
        self._control.append((kind, payload))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frames_in": self.frames_in,
                "frames_out": self.frames_out,
                "decode_failures": self.decode_failures,
                "infer_ms_ema": round(self.infer_ms_ema, 2),
                "dropped_stale": self.mailbox.dropped_stale,
            }

    # -- worker internals (worker thread) ------------------------------------ #
    def _post(self, kind: str, payload: Any) -> None:
        with self._emit_lock:
            emit = self._emit
        if emit is not None:
            emit(kind, payload)

    def _drain_control(self) -> None:
        ops: list[tuple[str, Any]] = []
        while True:
            try:
                ops.append(self._control.popleft())
            except IndexError:
                break
        # coalesce prompt ops: only the newest matters
        prompt_ops = [op for op in ops if op[0] == "prompt"]
        last_prompt = prompt_ops[-1] if prompt_ops else None
        for op in ops:
            kind, payload = op
            if kind == "prompt" and op is not last_prompt:
                continue
            try:
                if kind == "prompt":
                    ms = self.processor.pipeline.set_prompt(payload)
                    self._post("json", {"type": "prompt_applied", "text": payload, "ms": round(ms, 2)})
                    log.info("prompt applied in %.1f ms: %r", ms, payload[:80])
                elif kind == "config":
                    if "jpeg_quality" in payload:
                        q = int(payload["jpeg_quality"])
                        self.processor.jpeg_quality = min(95, max(30, q))
                    self._post("json", {"type": "config_applied", "jpeg_quality": self.processor.jpeg_quality})
            except Exception as e:  # noqa: BLE001 — worker must never die on a control op
                log.exception("control op %s failed", kind)
                self._post("json", {"type": "error", "message": f"{kind} failed: {e}"})

    def _run(self) -> None:
        log.info("gpu worker started")
        while not self._stop.is_set():
            self._drain_control()
            item = self.mailbox.take(timeout=0.1)
            if item is None:
                continue
            seq, ts, jpeg = item
            try:
                out_jpeg, t = self.processor.process_jpeg(jpeg)
            except Exception as e:  # noqa: BLE001 — a bad frame must not kill the loop
                log.exception("frame %d failed", seq)
                self._post("json", {"type": "error", "message": f"frame {seq} failed: {e}"})
                continue
            if out_jpeg is None:
                with self._lock:
                    self.decode_failures += 1
                continue
            header = HDR_OUT.pack(MAGIC_OUT, seq & 0xFFFFFFFF, ts, t.infer_ms)
            self._post("frame", header + out_jpeg)
            with self._lock:
                self.frames_out += 1
                self.infer_ms_ema = (
                    t.infer_ms if self.infer_ms_ema == 0.0 else 0.9 * self.infer_ms_ema + 0.1 * t.infer_ms
                )
        log.info("gpu worker stopped")


# --------------------------------------------------------------------------- #
# websocket server
# --------------------------------------------------------------------------- #

@dataclass
class ServerState:
    worker: Worker
    args: argparse.Namespace
    busy: bool = field(default=False)


async def _sender(ws: Any, outbox: OutBox) -> None:
    while True:
        await ws.send(await outbox.pop())


async def _stats_loop(ws: Any, outbox: OutBox, worker: Worker, interval: float) -> None:
    prev = worker.snapshot()
    prev_t = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        now = worker.snapshot()
        dt = max(time.monotonic() - prev_t, 1e-6)
        outbox.push_json(
            {
                "type": "stats",
                "fps_in": round((now["frames_in"] - prev["frames_in"]) / dt, 1),
                "fps_out": round((now["frames_out"] - prev["frames_out"]) / dt, 1),
                "infer_ms_ema": now["infer_ms_ema"],
                "dropped_stale": now["dropped_stale"],
                "dropped_outbox": outbox.dropped_outbox,
                "decode_failures": now["decode_failures"],
            }
        )
        prev, prev_t = now, time.monotonic()


def _make_handler(state: ServerState) -> Callable[[Any], Any]:
    async def handler(ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        if state.busy:
            await ws.send(json.dumps({"type": "busy", "message": "another studio session is live"}))
            await ws.close(code=1013, reason="busy")
            return
        state.busy = True
        loop = asyncio.get_running_loop()
        outbox = OutBox()

        def emit(kind: str, payload: Any) -> None:  # called from worker thread
            if kind == "json":
                loop.call_soon_threadsafe(outbox.push_json, payload)
            else:
                loop.call_soon_threadsafe(outbox.push_frame, payload)

        state.worker.attach(emit)
        a = state.args
        outbox.push_json(
            {
                "type": "hello",
                "server": "openstudio-server",
                "proto": PROTO_VERSION,
                "pipeline": state.worker.processor.pipeline.name,
                "model": a.model,
                "width": a.width,
                "height": a.height,
                "t_index": a.t_index,
                "prompt": getattr(state.worker.processor.pipeline, "prompt", a.prompt),
                "jpeg_quality": state.worker.processor.jpeg_quality,
                # boot-time quality levers (additive since proto 1)
                "vae": "full" if a.full_vae else "taesd",
                "cfg_type": a.cfg_type,
                "guidance_scale": a.guidance_scale,
                "delta": a.delta,
                "noise_mode": a.noise_mode,
                "similar_filter": a.similar_filter,
                "similar_max_skip": a.similar_max_skip,
                "lcm_lora": a.lcm_lora,
                "seed": a.seed,
            }
        )
        tasks = [
            asyncio.create_task(_sender(ws, outbox)),
            asyncio.create_task(_stats_loop(ws, outbox, state.worker, 1.0)),
        ]
        log.info("client connected: %s", getattr(ws, "remote_address", "?"))
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    if len(msg) <= HDR_IN.size or msg[0] != MAGIC_IN:
                        outbox.push_json({"type": "error", "message": "bad binary frame header"})
                        continue
                    _, seq, ts = HDR_IN.unpack_from(msg)
                    state.worker.submit_frame(seq, ts, bytes(msg[HDR_IN.size :]))
                else:
                    try:
                        obj = json.loads(msg)
                        mtype = obj.get("type")
                        if mtype == "ping":
                            outbox.push_json({"type": "pong", "t": obj.get("t")})
                        elif mtype == "prompt":
                            state.worker.submit_control("prompt", str(obj["text"]))
                        elif mtype == "config":
                            state.worker.submit_control("config", obj)
                        else:
                            outbox.push_json({"type": "error", "message": f"unknown type: {mtype!r}"})
                    except (json.JSONDecodeError, KeyError) as e:
                        outbox.push_json({"type": "error", "message": f"bad control message: {e}"})
        except ConnectionClosed:
            pass
        finally:
            state.worker.detach()
            for t in tasks:
                t.cancel()
            state.busy = False
            log.info("client disconnected")

    return handler


async def serve_forever(state: ServerState) -> None:
    from websockets.asyncio.server import serve

    a = state.args
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with serve(
        _make_handler(state),
        host=a.host,
        port=a.port,
        max_size=MAX_WS_MESSAGE,
        compression=None,  # JPEG doesn't deflate; permessage-deflate only adds CPU + latency
    ):
        log.info("listening on ws://%s:%d  (pipeline=%s)", a.host, a.port, state.worker.processor.pipeline.name)
        log.info("tunnel: ssh -N -L %d:127.0.0.1:%d -p <ssh-port> root@<pod-ip>", a.port, a.port)
        await stop.wait()
    log.info("server shut down")


# --------------------------------------------------------------------------- #
# selfcheck — smoke the whole frame path with zero clients
# --------------------------------------------------------------------------- #

def run_selfcheck(processor: FrameProcessor, args: argparse.Namespace) -> int:
    """Synthesize webcam frames, push them through the EXACT serve-path
    FrameProcessor, hot-swap the prompt midway, and demand sane output."""
    import pathlib

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.selfcheck_frames
    swap_at = (2 * n) // 3
    timings: list[FrameTimings] = []
    prompt_swap_ms = None
    failures: list[str] = []
    saved: dict[int, bytes] = {}

    for i in range(n):
        frame = synth_frame(i)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        assert ok
        if i == swap_at:
            prompt_swap_ms = processor.pipeline.set_prompt(args.selfcheck_swap_prompt)
        out_jpeg, t = processor.process_jpeg(bytes(jpg))
        if out_jpeg is None or t is None:
            failures.append(f"frame {i}: processing returned None")
            continue
        timings.append(t)
        if i in (0, swap_at, n - 1):
            saved[i] = out_jpeg
            (out_dir / f"in_{i:03d}.jpg").write_bytes(bytes(jpg))
            (out_dir / f"out_{i:03d}.jpg").write_bytes(out_jpeg)

    # -- assertions: look at what actually came out -------------------------- #
    for i, payload in saved.items():
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            failures.append(f"out frame {i}: not a decodable JPEG")
        elif img.shape[:2] != (args.height, args.width):
            failures.append(f"out frame {i}: shape {img.shape[:2]} != ({args.height}, {args.width})")
        elif float(img.std()) < 2.0:
            failures.append(f"out frame {i}: near-constant output (std={img.std():.2f}) — pipeline collapsed")
    if prompt_swap_ms is None:
        failures.append("prompt swap never ran")
    elif prompt_swap_ms > 1000.0:
        failures.append(f"prompt swap took {prompt_swap_ms:.0f} ms (>1000) — update_prompt path is wrong")
    if not timings:
        failures.append("no frames processed")

    def pctl(vals: list[float], p: float) -> float:
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * len(s)))]

    if timings:
        infer = [t.infer_ms for t in timings]
        total = [t.decode_ms + t.infer_ms + t.encode_ms for t in timings]
        summary = {
            "frames": len(timings),
            "decode_ms_p50": round(pctl([t.decode_ms for t in timings], 0.5), 2),
            "infer_ms_p50": round(pctl(infer, 0.5), 2),
            "infer_ms_p95": round(pctl(infer, 0.95), 2),
            "encode_ms_p50": round(pctl([t.encode_ms for t in timings], 0.5), 2),
            "total_ms_p50": round(pctl(total, 0.5), 2),
            "est_fps": round(1000.0 / max(pctl(total, 0.5), 1e-3), 1),
            "prompt_swap_ms": round(prompt_swap_ms or -1, 2),
            "outputs": str(out_dir),
        }
        print("SELFCHECK " + ("PASS " if not failures else "FAIL ") + json.dumps(summary))
    for f in failures:
        print(f"SELFCHECK FAILURE: {f}", file=sys.stderr)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_pipeline(args: argparse.Namespace) -> Any:
    if args.pipeline == "fake":
        p = FakePipeline(args.prompt, delay_ms=args.fake_delay_ms)
    else:
        p = StreamPipeline(
            model_id=args.model,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            t_index_list=args.t_index,
            use_tiny_vae=not args.full_vae,
            use_xformers=args.xformers,
            cfg_type=args.cfg_type,
            guidance_scale=args.guidance_scale,
            delta=args.delta,
            do_add_noise=(args.noise_mode == "add"),
            similar_filter=args.similar_filter,
            similar_max_skip=args.similar_max_skip,
            use_lcm_lora=args.lcm_lora,
            seed=args.seed,
        )
    t0 = time.perf_counter()
    p.warmup(args.warmup_frames)
    log.info("warmup (%d frames) took %.1fs", args.warmup_frames, time.perf_counter() - t0)
    return p


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="NEVER bind 0.0.0.0 without adding auth first")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pipeline", choices=["stream", "fake"], default="stream")
    ap.add_argument("--fake-delay-ms", type=float, default=0.0,
                    help="fake pipeline only: simulated GPU latency, for local pacing/drop tests")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--prompt", default="a cyberpunk city at night, neon lights, cinematic")
    ap.add_argument("--width", type=int, default=512, help="model res; BOOT-time only (latent buffers)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--t-index", type=lambda s: [int(x) for x in s.split(",")], default=[35, 45],
                    help="StreamDiffusion t_index_list, csv (official webcam demo: 35,45); "
                         "restyle strength ~ (50 - first_index)/50")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--warmup-frames", type=int, default=10)
    ap.add_argument("--full-vae", action="store_true",
                    help="use the full SD VAE instead of TAESD (+18-25 ms/frame, crisper + truer color)")
    ap.add_argument("--xformers", action="store_true", help="only if xformers was installed --no-deps")
    # --- quality levers (BOOT-time; see module docstring + bench_quality.py) ---
    ap.add_argument("--cfg-type", choices=["none", "self"], default="none",
                    help="'self' = RCFG virtual negative guidance at ~0 extra UNet cost; "
                         "initialize/full are excluded (they break prompt hot-swap)")
    ap.add_argument("--guidance-scale", type=float, default=1.0,
                    help="must be > 1.0 for --cfg-type self to have any effect (try 1.2-1.6)")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="RCFG residual moderation for --cfg-type self (0.5-1.0)")
    ap.add_argument("--noise-mode", choices=["add", "deterministic"], default="add",
                    help="add = re-noise each frame with the fixed init noise (default); "
                         "deterministic = do_add_noise=False, the authors' vid2vid anti-flicker path")
    ap.add_argument("--similar-filter", type=float, default=None, metavar="THRESH",
                    help="enable the similar-image filter: probabilistically skip inference when "
                         "input cosine-similarity > THRESH and replay the previous output "
                         "(try 0.98; kills static-scene shimmer; needs --warmup-frames >= 1)")
    ap.add_argument("--similar-max-skip", type=int, default=10,
                    help="max consecutive skipped frames before a forced inference")
    ap.add_argument("--lcm-lora", action="store_true",
                    help="load + fuse latent-consistency/lcm-lora-sdv1-5 — required for plain "
                         "SD1.5 finetunes (e.g. Lykon/dreamshaper-8); NOT for *turbo models")
    ap.add_argument("--seed", type=int, default=2,
                    help="init-noise seed (StreamDiffusion default 2); fixed noise is reused every frame")
    ap.add_argument("--selfcheck", action="store_true", help="no server: smoke the frame path, exit 0/1")
    ap.add_argument("--selfcheck-frames", type=int, default=24)
    ap.add_argument("--selfcheck-swap-prompt", default="an oil painting, thick brushstrokes")
    ap.add_argument("--out-dir", default="/tmp/openstudio-selfcheck")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    pipeline = build_pipeline(args)
    processor = FrameProcessor(pipeline, args.width, args.height, args.jpeg_quality)

    if args.selfcheck:
        return run_selfcheck(processor, args)

    worker = Worker(processor)
    worker.start()
    try:
        asyncio.run(serve_forever(ServerState(worker=worker, args=args)))
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
