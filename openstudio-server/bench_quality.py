#!/usr/bin/env python3
"""bench_quality.py — offline quality x speed bench for openstudio-server lever combos.

Runs EVERY config in CONFIGS (model x t_index x vae x cfg/RCFG x noise-mode x
similar-filter — seeded from the scout reports' top-6 ranking) over short
simulated webcam sequences: a slow pan/zoom + per-frame sensor noise over 4
photo-like test scenes (downloaded CC0 photos, or synthetic photo-like
stand-ins with --no-download / on any fetch failure). Per config it writes:

  sheet_<config>.jpg   contact sheet — input rows vs restyled rows per scene/prompt
  fps / infer_ms       p50/p95 wall time around pipeline.process (the same span
                       server.py reports as infer_ms)
  flicker              mean |out_t − out_{t−1}| (gray) restricted to pixels where
                       the INPUT pair is static — split into hold-segment-only
                       and motion-only, plus dup_ratio (bit-identical output
                       pairs: the similar-filter's freeze signature)
  clip / clip_gain     CLIP text-image similarity of outputs vs the prompt
                       (open_clip if importable, else transformers CLIP, else
                       null), and the gain over scoring the raw inputs

then ranks configs into results.json + results.md (+ compare.jpg strip).
Sequences hold still at both ends with fresh sensor noise each frame — a
bit-exact repeat would measure zero flicker for every config; noise-amplified
shimmer while "sitting still" is exactly the user-visible artifact.

Usage (pod, same venv as server.py):
    python bench_quality.py                       # full sweep (downloads 4 photos once)
    python bench_quality.py --configs baseline,rcfg-self
    python bench_quality.py --no-download         # synthetic scenes only
    python bench_quality.py --list                # show configs + boot flags, exit

Local CPU proof (no torch/GPU — exercises everything up to the pipeline boundary):
    python bench_quality.py --pipeline fake --no-download --no-clip --frames 12

The stream path constructs server.StreamPipeline — the SAME class `python
server.py` serves with — so a winning row's `server_flags` string boots the
production server in exactly the benched config.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402  — cover_fit / FakePipeline / StreamPipeline (torch stays lazy)

log = logging.getLogger("bench")

SRC_SIZE = 896          # working size of each test scene (pan/zoom source)
OUT_SIZE = 512          # model input/output size (matches server default)
SENSOR_NOISE = 2.0      # per-frame gaussian sensor noise sigma, in 8-bit levels
STATIC_THRESH = 2.0     # input-pair gray diff (after blur) below this = "static"
SKIP_FIRST = 2          # output pairs skipped at sequence start (buffer settle)
FLICKER_WEIGHT = 0.6    # quality_score = clip_norm − FLICKER_WEIGHT · flicker_norm
RANDOM_SEED = 20260721  # stdlib random — SimilarImageFilter's skip sampling

# --------------------------------------------------------------------------- #
# configs — seeded from the scout reports' ranking (C0..C5). Edit freely; every
# key not present falls back to DEFAULTS (= server.py boot defaults).
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = dict(
    model="stabilityai/sd-turbo",
    t_index=[35, 45],
    vae="taesd",              # "taesd" | "full"
    cfg_type="none",          # "none" | "self"   (initialize/full break hot-swap)
    guidance_scale=1.0,       # must be > 1.0 for cfg_type=self to matter
    delta=1.0,
    noise_mode="add",         # "add" | "deterministic" (do_add_noise False)
    similar_filter=None,      # None | threshold e.g. 0.98
    similar_max_skip=10,
    lcm_lora=False,           # fuse latent-consistency/lcm-lora-sdv1-5 (SD1.5 finetunes)
    seed=2,                   # StreamDiffusion init-noise seed (upstream default)
)

CONFIGS: list[dict[str, Any]] = [
    dict(name="baseline",
         desc="prod default — sd-turbo [35,45], the weak-restyle/flicker complaint"),
    dict(name="rcfg-self",
         desc="C1: RCFG self — zero-cost adherence punch on sd-turbo",
         t_index=[32, 45], cfg_type="self", guidance_scale=1.4, delta=1.0),
    dict(name="dreamshaper-rcfg",
         desc="C2: SD1.5 finetune + fused LCM-LoRA + RCFG (community quality recipe)",
         model="Lykon/dreamshaper-8", lcm_lora=True,
         t_index=[32, 45], cfg_type="self", guidance_scale=1.4, delta=1.0),
    dict(name="rcfg-temporal",
         desc="C3 on C1: deterministic noise + similar-filter anti-flicker bundle",
         t_index=[32, 45], cfg_type="self", guidance_scale=1.4, delta=1.0,
         noise_mode="deterministic", similar_filter=0.98),
    dict(name="dreamshaper-temporal",
         desc="C3 on C2: quality recipe + anti-flicker bundle",
         model="Lykon/dreamshaper-8", lcm_lora=True,
         t_index=[32, 45], cfg_type="self", guidance_scale=1.4, delta=1.0,
         noise_mode="deterministic", similar_filter=0.98),
    dict(name="strong-restyle",
         desc="C4: strength-knee probe — t_index [25,40] ≈ 0.50 strength, same cost",
         t_index=[25, 40], cfg_type="self", guidance_scale=1.4, delta=1.0,
         noise_mode="deterministic", similar_filter=0.98),
    dict(name="full-vae",
         desc="C5: buy back TAESD softness/color-shift with the full VAE (+18-25 ms)",
         t_index=[32, 45], cfg_type="self", guidance_scale=1.4, delta=1.0,
         noise_mode="deterministic", similar_filter=0.98, vae="full"),
]

PROMPTS = [
    "a cyberpunk city at night, neon lights, cinematic",  # server.py default prompt
    "an oil painting, thick brushstrokes",                # server.py selfcheck swap prompt
]

LEVER_KEYS = ("model", "t_index", "vae", "cfg_type", "guidance_scale", "delta",
              "noise_mode", "similar_filter", "similar_max_skip", "lcm_lora", "seed")


def merged(cfg: dict[str, Any]) -> dict[str, Any]:
    m = dict(DEFAULTS)
    m.update(cfg)
    return m


def pipeline_kwargs(cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
    """EXACTLY mirrors server.build_pipeline's stream branch: the bench must
    measure the same construction `python server.py <server_flags>` boots."""
    return dict(
        model_id=cfg["model"],
        prompt=prompt,
        width=OUT_SIZE,
        height=OUT_SIZE,
        t_index_list=list(cfg["t_index"]),
        use_tiny_vae=(cfg["vae"] != "full"),
        use_xformers=False,
        cfg_type=cfg["cfg_type"],
        guidance_scale=cfg["guidance_scale"],
        delta=cfg["delta"],
        do_add_noise=(cfg["noise_mode"] == "add"),
        similar_filter=cfg["similar_filter"],
        similar_max_skip=cfg["similar_max_skip"],
        use_lcm_lora=cfg["lcm_lora"],
        seed=cfg["seed"],
    )


def server_flags(cfg: dict[str, Any]) -> str:
    """The `python server.py` flag string that boots this exact config."""
    parts = [f"--model {cfg['model']}"]
    if cfg["lcm_lora"]:
        parts.append("--lcm-lora")
    parts.append("--t-index " + ",".join(str(i) for i in cfg["t_index"]))
    if cfg["vae"] == "full":
        parts.append("--full-vae")
    if cfg["cfg_type"] != "none":
        parts.append(f"--cfg-type {cfg['cfg_type']} "
                     f"--guidance-scale {cfg['guidance_scale']} --delta {cfg['delta']}")
    if cfg["noise_mode"] != "add":
        parts.append("--noise-mode deterministic")
    if cfg["similar_filter"] is not None:
        parts.append(f"--similar-filter {cfg['similar_filter']} "
                     f"--similar-max-skip {cfg['similar_max_skip']}")
    if cfg["seed"] != DEFAULTS["seed"]:
        parts.append(f"--seed {cfg['seed']}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# test scenes: 4 CC0 photos (picsum stable ids) with synthetic photo-like
# fallbacks — a portrait, a desk/workspace, an interior, a street.
# --------------------------------------------------------------------------- #

SCENES = [
    ("portrait", f"https://picsum.photos/id/64/{SRC_SIZE}/{SRC_SIZE}"),
    ("desk",     f"https://picsum.photos/id/0/{SRC_SIZE}/{SRC_SIZE}"),
    ("room",     f"https://picsum.photos/id/1040/{SRC_SIZE}/{SRC_SIZE}"),
    ("street",   f"https://picsum.photos/id/1011/{SRC_SIZE}/{SRC_SIZE}"),
]


def _photolike(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Camera-ify a painted scene: soft optics, sensor noise, vignette, JPEG pass."""
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = np.clip(img.astype(np.float32) + rng.normal(0.0, 3.0, img.shape), 0, 255)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    img *= (1.0 - 0.22 * r2)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR) if ok else img


def _scene_portrait(rng: np.random.Generator) -> np.ndarray:
    S = SRC_SIZE
    xx = np.mgrid[0:S, 0:S][1].astype(np.float32) / S
    img = np.clip(np.stack([120 + 45 * (1 - xx), 118 + 40 * (1 - xx), 112 + 30 * (1 - xx)],
                           axis=-1), 0, 255).astype(np.uint8)
    cv2.circle(img, (int(S * .80), int(S * .18)), int(S * .13), (150, 190, 235), -1)  # lamp
    img = cv2.GaussianBlur(img, (0, 0), S * 0.03)
    cv2.ellipse(img, (int(S * .5), int(S * 1.02)), (int(S * .36), int(S * .34)),
                0, 180, 360, (76, 60, 118), -1)                                        # sweater
    cv2.rectangle(img, (int(S * .445), int(S * .60)), (int(S * .555), int(S * .76)),
                  (118, 148, 196), -1)                                                 # neck
    cv2.ellipse(img, (int(S * .5), int(S * .44)), (int(S * .155), int(S * .205)),
                0, 0, 360, (132, 162, 212), -1)                                        # head
    for ex in (.345, .655):
        cv2.circle(img, (int(S * ex), int(S * .45)), int(S * .022), (126, 154, 202), -1)
    cv2.ellipse(img, (int(S * .5), int(S * .335)), (int(S * .165), int(S * .145)),
                0, 180, 360, (42, 46, 58), -1)                                         # hair
    for ex in (.443, .557):
        cv2.ellipse(img, (int(S * ex), int(S * .435)), (int(S * .022), int(S * .012)),
                    0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (int(S * ex), int(S * .437)), int(S * .009), (60, 46, 40), -1)
        cv2.line(img, (int(S * (ex - .03)), int(S * .405)), (int(S * (ex + .03)), int(S * .400)),
                 (52, 56, 70), 4)
    cv2.line(img, (int(S * .5), int(S * .45)), (int(S * .49), int(S * .485)), (112, 138, 184), 3)
    cv2.ellipse(img, (int(S * .5), int(S * .525)), (int(S * .030), int(S * .011)),
                0, 0, 180, (86, 96, 160), -1)                                          # mouth
    for ex in (.42, .58):
        cv2.circle(img, (int(S * ex), int(S * .48)), int(S * .02), (120, 140, 205), -1)
    return _photolike(img, rng)


def _scene_desk(rng: np.random.Generator) -> np.ndarray:
    S = SRC_SIZE
    img = np.full((S, S, 3), (70, 76, 84), np.uint8)
    for k in range(6):                                                                 # blinds light
        y = int(S * (0.06 + 0.055 * k))
        cv2.rectangle(img, (int(S * .55), y), (int(S * .96), y + int(S * .02)), (120, 138, 158), -1)
    desk_y = int(S * 0.58)
    img[desk_y:] = (96, 126, 160)                                                      # desk wood
    cv2.line(img, (0, desk_y), (S, desk_y), (70, 96, 128), 4)
    cv2.rectangle(img, (int(S * .30), int(S * .30)), (int(S * .66), desk_y), (52, 50, 48), -1)
    cv2.rectangle(img, (int(S * .32), int(S * .325)), (int(S * .64), int(S * .55)), (180, 150, 90), -1)
    cv2.rectangle(img, (int(S * .33), int(S * .35)), (int(S * .50), int(S * .38)), (220, 200, 160), -1)
    cv2.rectangle(img, (int(S * .33), int(S * .40)), (int(S * .63), int(S * .53)), (150, 120, 70), -1)
    cv2.rectangle(img, (int(S * .26), desk_y), (int(S * .70), int(S * .66)), (58, 56, 54), -1)
    cv2.rectangle(img, (int(S * .76), int(S * .50)), (int(S * .84), desk_y), (60, 60, 170), -1)  # mug
    cv2.ellipse(img, (int(S * .84), int(S * .53)), (int(S * .025), int(S * .035)),
                0, -90, 90, (60, 60, 170), 6)
    cv2.rectangle(img, (int(S * .08), int(S * .66)), (int(S * .26), int(S * .80)), (200, 205, 210), -1)
    cv2.line(img, (int(S * .10), int(S * .70)), (int(S * .24), int(S * .70)), (140, 150, 160), 2)
    cv2.line(img, (int(S * .10), int(S * .74)), (int(S * .24), int(S * .74)), (140, 150, 160), 2)
    cv2.line(img, (int(S * .30), int(S * .72)), (int(S * .42), int(S * .68)), (40, 90, 200), 5)   # pen
    return _photolike(img, rng)


def _scene_room(rng: np.random.Generator) -> np.ndarray:
    S = SRC_SIZE
    img = np.full((S, S, 3), (105, 112, 122), np.uint8)
    floor_y = int(S * 0.62)
    img[floor_y:] = (78, 103, 132)
    for k in range(1, 9):
        y = floor_y + int((S - floor_y) * k / 9)
        cv2.line(img, (0, y), (S, y), (64, 86, 112), 2)
    cv2.rectangle(img, (int(S * .07), int(S * .09)), (int(S * .40), int(S * .52)), (70, 74, 80), -1)
    cv2.rectangle(img, (int(S * .09), int(S * .11)), (int(S * .38), int(S * .50)), (235, 196, 148), -1)
    cv2.line(img, (int(S * .235), int(S * .11)), (int(S * .235), int(S * .50)), (70, 74, 80), 7)
    cv2.line(img, (int(S * .09), int(S * .305)), (int(S * .38), int(S * .305)), (70, 74, 80), 7)
    cv2.rectangle(img, (int(S * .50), int(S * .46)), (int(S * .94), int(S * .56)), (96, 84, 74), -1)
    cv2.rectangle(img, (int(S * .50), int(S * .55)), (int(S * .94), int(S * .70)), (110, 96, 84), -1)
    cv2.rectangle(img, (int(S * .53), int(S * .49)), (int(S * .69), int(S * .58)), (128, 112, 98), -1)
    cv2.rectangle(img, (int(S * .72), int(S * .49)), (int(S * .90), int(S * .58)), (86, 120, 150), -1)
    cv2.rectangle(img, (int(S * .42), int(S * .55)), (int(S * .475), int(S * .63)), (60, 92, 140), -1)
    for k in range(7):                                                                 # plant leaves
        ang = -90 + (k - 3) * 22
        cv2.ellipse(img, (int(S * .4475), int(S * .55)), (int(S * .012), int(S * .06)),
                    ang, 0, 360, (70, 140, 80), -1)
    ov = img.copy()                                                                    # window light
    pts = np.array([[int(S * .12), floor_y], [int(S * .42), floor_y],
                    [int(S * .55), int(S * .9)], [int(S * .05), int(S * .9)]])
    cv2.fillPoly(ov, [pts], (200, 210, 220))
    img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
    return _photolike(img, rng)


def _scene_street(rng: np.random.Generator) -> np.ndarray:
    S = SRC_SIZE
    yy = np.mgrid[0:S, 0:S][0].astype(np.float32) / S
    img = np.clip(np.stack([210 - 80 * yy, 180 - 60 * yy, 150 - 20 * yy], axis=-1),
                  0, 255).astype(np.uint8)
    cv2.circle(img, (int(S * .72), int(S * .30)), int(S * .06), (140, 200, 250), -1)   # low sun
    img = cv2.GaussianBlur(img, (0, 0), S * 0.01)
    horizon = int(S * 0.55)
    bx = 0
    for k, hfrac in enumerate([.18, .30, .22, .36, .26, .20]):
        bw = int(S * (0.12 + 0.05 * (k % 3)))
        top = horizon - int(S * hfrac)
        col = 58 + 12 * (k % 3)
        cv2.rectangle(img, (bx, top), (bx + bw, horizon), (col, col, col + 6), -1)
        wrng = np.random.default_rng(1000 + k)                                         # lit windows
        for wy in range(top + 14, horizon - 10, 26):
            for wx in range(bx + 8, bx + bw - 12, 22):
                if wrng.random() < 0.55:
                    cv2.rectangle(img, (wx, wy), (wx + 10, wy + 14), (120, 190, 235), -1)
        bx += bw + 2
    img[horizon:] = (60, 62, 66)
    pts = np.array([[int(S * .28), S], [int(S * .72), S],
                    [int(S * .55), horizon], [int(S * .45), horizon]])
    cv2.fillPoly(img, [pts], (74, 76, 82))
    for k in range(6):                                                                 # lane dashes
        t0, t1 = k / 6, (k + 0.45) / 6
        y0 = int(horizon + (S - horizon) * t0)
        y1 = int(horizon + (S - horizon) * t1)
        x0 = int(S * .5 - 2 - 6 * t0)
        cv2.rectangle(img, (x0, y0), (x0 + max(4, int(10 * t0) + 2), y1), (160, 200, 210), -1)
    cv2.rectangle(img, (int(S * .56), int(S * .72)), (int(S * .72), int(S * .82)), (150, 90, 60), -1)
    cv2.rectangle(img, (int(S * .58), int(S * .68)), (int(S * .70), int(S * .73)), (170, 120, 90), -1)
    for cx in (.585, .695):
        cv2.circle(img, (int(S * cx), int(S * .82)), int(S * .018), (30, 30, 34), -1)
    return _photolike(img, rng)


SYNTH = {"portrait": _scene_portrait, "desk": _scene_desk,
         "room": _scene_room, "street": _scene_street}


def fetch_scene(name: str, url: str, cache_dir: Path, no_download: bool,
                rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """896x896 BGR scene + its origin tag (cache:/download:/synthetic)."""
    cache = cache_dir / f"{name}.jpg"
    if cache.exists():
        img = cv2.imread(str(cache))
        if img is not None:
            return server.cover_fit(img, SRC_SIZE, SRC_SIZE), f"cache:{cache.name}"
    if not no_download:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "openstudio-bench/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None and min(img.shape[:2]) >= OUT_SIZE:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
                return server.cover_fit(img, SRC_SIZE, SRC_SIZE), f"download:{url}"
            log.warning("scene %s: bad image from %s — synthetic fallback", name, url)
        except Exception as e:  # noqa: BLE001 — any network failure falls back
            log.warning("scene %s: download failed (%s) — synthetic fallback", name, e)
    return SYNTH[name](rng), "synthetic"


# --------------------------------------------------------------------------- #
# simulated webcam sequence: hold — slow pan/zoom — hold, fresh sensor noise
# --------------------------------------------------------------------------- #

def _sensor_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32)
                   + rng.normal(0.0, SENSOR_NOISE, img.shape), 0, 255).astype(np.uint8)


def make_sequence(src_bgr: np.ndarray, n: int, hold: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """n RGB OUT_SIZE frames: `hold` static frames, slow pan+zoom, `hold` static.
    Every frame gets fresh sensor noise — the input jitter per-frame diffusion
    amplifies into flicker (a bit-exact repeat would measure zero for everyone)."""
    S = src_bgr.shape[0]
    w0, w1 = int(S * 0.750), int(S * 0.695)                    # ~7% zoom-in
    xa, ya = int(S * 0.0625), int(S * 0.0800)
    xb, yb = xa + int(S * 0.0535), ya + int(S * 0.0400)        # ~4 px/frame pan
    motion = max(n - 2 * hold, 1)
    frames = []
    for i in range(n):
        t = 0.0 if i < hold else 1.0 if i >= n - hold else (i - hold + 1) / (motion + 1)
        w = round(w0 + (w1 - w0) * t)
        x = min(max(round(xa + (xb - xa) * t), 0), S - w)
        y = min(max(round(ya + (yb - ya) * t), 0), S - w)
        crop = cv2.resize(src_bgr[y:y + w, x:x + w], (OUT_SIZE, OUT_SIZE),
                          interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(_sensor_noise(crop, rng), cv2.COLOR_BGR2RGB))
    return frames


def make_drain(f0: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """4 frames that flush the pipelined latent buffer into the CURRENT scene.
    The flips differ grossly so a similar-image filter cannot skip them (a skip
    would leave the previous scene in the buffer); the last two are f0 with
    fresh sensor noise so the buffer ends holding true scene content."""
    return [
        np.ascontiguousarray(f0[::-1, :, :]),
        np.ascontiguousarray(f0[:, ::-1, :]),
        _sensor_noise(f0, rng),
        _sensor_noise(f0, rng),
    ]


def sheet_frame_indices(n: int, hold: int) -> list[int]:
    cand = {1, hold + max(1, (n - 2 * hold) // 3), n - hold - 1, n - 2}
    return sorted(i for i in cand if 0 <= i < n)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _mean(vals: list[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def _rnd(v: Optional[float], nd: int = 3) -> Optional[float]:
    return None if v is None else round(v, nd)


def pctl(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def temporal_metrics(inputs: list[np.ndarray], outputs: list[np.ndarray],
                     hold: int) -> dict[str, Any]:
    """Flicker = mean |out_t − out_{t−1}| (gray) on pixels where the blurred
    INPUT pair moved < STATIC_THRESH — i.e. output change not explained by
    input change. Split into hold-only / motion-only; dup_ratio counts
    bit-identical output pairs (similar-filter freezes)."""
    n = len(inputs)
    gi = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 1.0).astype(np.float32)
          for f in inputs]
    go = [cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32) for o in outputs]
    masked, hold_v, motion_v = [], [], []
    dups = pairs = 0
    for t in range(max(1, SKIP_FIRST), n):
        mask = np.abs(gi[t] - gi[t - 1]) < STATIC_THRESH
        if float(mask.mean()) < 0.05:
            continue  # nothing static in this pair — no measurement
        v = float(np.abs(go[t] - go[t - 1])[mask].mean())
        masked.append(v)
        if t <= hold - 1 or t - 1 >= n - hold:   # pair fully inside a hold segment
            hold_v.append(v)
        else:
            motion_v.append(v)
        if np.array_equal(outputs[t], outputs[t - 1]):
            dups += 1
        pairs += 1
    return {
        "flicker": _rnd(_mean(masked)),
        "flicker_hold": _rnd(_mean(hold_v)),
        "flicker_motion": _rnd(_mean(motion_v)),
        "dup_ratio": round(dups / pairs, 3) if pairs else None,
        "pairs": pairs,
    }


class ClipScorer:
    """CLIP text-image similarity. Tries open_clip, then transformers CLIP
    (already in server deps), else scores are null — never a hard failure."""

    def __init__(self, enabled: bool) -> None:
        self.backend: Optional[str] = None
        self.reason = "disabled (--no-clip)"
        if not enabled:
            return
        try:
            import torch
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:  # noqa: BLE001
            self.reason = f"torch unavailable: {e}"
            return
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self._device)
            model.eval()
            self._model, self._preprocess = model, preprocess
            self._tokenize = open_clip.get_tokenizer("ViT-B-32")
            self.backend = "open_clip/ViT-B-32"
            return
        except Exception as e1:  # noqa: BLE001
            err1 = e1
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(self._device).eval()
            self._proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.backend = "transformers/clip-vit-base-patch32"
        except Exception as e2:  # noqa: BLE001
            self.reason = f"open_clip failed ({err1}); transformers CLIP failed ({e2})"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def score(self, frames_rgb: list[np.ndarray], text: str) -> Optional[float]:
        if not self.available or not frames_rgb:
            return None
        try:
            from PIL import Image
            pils = [Image.fromarray(f) for f in frames_rgb]
            with self._torch.no_grad():
                if self.backend.startswith("open_clip"):
                    ims = self._torch.stack([self._preprocess(p) for p in pils]).to(self._device)
                    img = self._model.encode_image(ims)
                    txt = self._model.encode_text(self._tokenize([text]).to(self._device))
                else:
                    inp = self._proc(text=[text], images=pils,
                                     return_tensors="pt", padding=True).to(self._device)
                    img = self._model.get_image_features(pixel_values=inp["pixel_values"])
                    txt = self._model.get_text_features(input_ids=inp["input_ids"],
                                                        attention_mask=inp["attention_mask"])
                img = img / img.norm(dim=-1, keepdim=True)
                txt = txt / txt.norm(dim=-1, keepdim=True)
                return float((img @ txt.T).mean().item())
        except Exception as e:  # noqa: BLE001 — scoring must never kill the bench
            log.warning("CLIP scoring failed: %s", e)
            return None


# --------------------------------------------------------------------------- #
# GPU bookkeeping (all guarded — fake mode never imports torch)
# --------------------------------------------------------------------------- #

def reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def gpu_peak_mb() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    return None


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def device_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} · torch {torch.__version__}"
        return f"cpu · torch {torch.__version__}"
    except Exception:  # noqa: BLE001
        return "cpu (no torch)"


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #

def _annot(img: np.ndarray, text: str, org: tuple[int, int] = (6, 20),
           scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _tile(rgb: np.ndarray, size: int, label: str = "") -> np.ndarray:
    t = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (size, size),
                   interpolation=cv2.INTER_AREA)
    if label:
        _annot(t, label, (6, 18), 0.45)
    return t


def _header_bar(width: int, lines: list[str]) -> np.ndarray:
    bar = np.full((14 + 22 * len(lines), width, 3), 24, np.uint8)
    for i, line in enumerate(lines):
        _annot(bar, line[:int(width / 9)], (8, 22 + 22 * i), 0.48)
    return bar


def build_contact_sheet(path: Path, header_lines: list[str],
                        rows: list[tuple[str, list[np.ndarray]]], thumb: int = 224) -> None:
    tiles = []
    for label, imgs in rows:
        tiles.append(np.hstack([_tile(im, thumb, label if j == 0 else "")
                                for j, im in enumerate(imgs)]))
    body = np.vstack(tiles)
    sheet = np.vstack([_header_bar(body.shape[1], header_lines), body])
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def build_compare(out_dir: Path, rows: list[dict[str, Any]], prompts: list[str]) -> None:
    entries = [r for r in rows if not r.get("error") and r.get("_compare", {}).get("in") is not None]
    if not entries:
        return
    thumb = 256
    grid = []
    for r in entries:
        c = r["_compare"]
        tiles = [_tile(c["in"], thumb, f"#{r.get('rank', '?')} {r['name']}")]
        tiles += [_tile(o, thumb, f"p{oi}") for oi, o in enumerate(c["outs"])]
        grid.append(np.hstack(tiles))
    body = np.vstack(grid)
    lines = ["compare — col0 input (scene0, settled frame), then one col per prompt; rows ranked"]
    lines += [f"p{i}: {p}" for i, p in enumerate(prompts)]
    cv2.imwrite(str(out_dir / "compare.jpg"),
                np.vstack([_header_bar(body.shape[1], lines), body]),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# --------------------------------------------------------------------------- #
# per-config run
# --------------------------------------------------------------------------- #

def build_bench_pipeline(cfg: dict[str, Any], args: argparse.Namespace, prompt: str) -> Any:
    if args.pipeline == "fake":
        return server.FakePipeline(prompt)
    return server.StreamPipeline(**pipeline_kwargs(cfg, prompt))


def run_config(cfg: dict[str, Any], scenes: list[dict[str, Any]], prompts: list[str],
               args: argparse.Namespace, clip: ClipScorer,
               clip_in_cache: dict[tuple[str, str], Optional[float]],
               out_dir: Path, hold: int) -> dict[str, Any]:
    name = cfg["name"]
    log.info("=== config %s — %s", name, cfg.get("desc", ""))
    random.seed(RANDOM_SEED)  # SimilarImageFilter's probabilistic skip sampling
    reset_gpu_peak()
    t0 = time.perf_counter()
    pipeline = build_bench_pipeline(cfg, args, prompts[0])
    pipeline.warmup(args.warmup)
    load_s = time.perf_counter() - t0

    frame_ms: list[float] = []
    per_runs: list[dict[str, Any]] = []
    sheet_cells: dict[tuple[int, str], list[np.ndarray]] = {}
    compare: dict[str, Any] = {"in": None, "outs": []}
    idxs = sheet_frame_indices(args.frames, hold)

    for pi, prompt in enumerate(prompts):
        if pi > 0:
            pipeline.set_prompt(prompt)
        for si, sc in enumerate(scenes):
            frames = sc["frames"]
            drain_rng = np.random.default_rng(RANDOM_SEED + si)
            for d in make_drain(frames[0], drain_rng):
                pipeline.process(d)
            outs: list[np.ndarray] = []
            for f in frames:
                t1 = time.perf_counter()
                o = pipeline.process(f)
                frame_ms.append((time.perf_counter() - t1) * 1000.0)
                o = np.asarray(o)
                if o.dtype != np.uint8:
                    o = np.clip(o, 0, 255).astype(np.uint8)
                if o.shape != f.shape:
                    raise RuntimeError(f"pipeline returned {o.shape}, expected {f.shape}")
                outs.append(o)
            tm = temporal_metrics(frames, outs, hold)
            cs = clip.score(outs[2::3], prompt)
            key = (sc["name"], prompt)
            if key not in clip_in_cache:
                clip_in_cache[key] = clip.score(frames[2::3], prompt)
            ci = clip_in_cache[key]
            per_runs.append({"prompt": prompt, "scene": sc["name"], **tm,
                             "clip": _rnd(cs, 4),
                             "clip_gain": None if (cs is None or ci is None) else round(cs - ci, 4)})
            if pi == 0:
                sheet_cells[(si, "in")] = [frames[i] for i in idxs]
            sheet_cells[(si, f"p{pi}")] = [outs[i] for i in idxs]
            if si == 0:
                if compare["in"] is None:
                    compare["in"] = frames[args.frames - 2]
                compare["outs"].append(outs[args.frames - 2])
            log.info("  p%d %-9s flicker=%s hold=%s dup=%s clip=%s",
                     pi, sc["name"], tm["flicker"], tm["flicker_hold"],
                     tm["dup_ratio"], _rnd(cs, 4))

    vram = gpu_peak_mb()
    del pipeline
    free_gpu()

    p50, p95 = pctl(frame_ms, 0.5), pctl(frame_ms, 0.95)
    row: dict[str, Any] = {
        "name": name,
        "desc": cfg.get("desc", ""),
        "params": {k: cfg[k] for k in LEVER_KEYS},
        "server_flags": server_flags(cfg),
        "pipeline": args.pipeline,
        "load_s": round(load_s, 1),
        "infer_ms_p50": round(p50, 2),
        "infer_ms_p95": round(p95, 2),
        "fps": round(1000.0 / max(p50, 1e-3), 1),
        "flicker": _rnd(_mean([r["flicker"] for r in per_runs])),
        "flicker_hold": _rnd(_mean([r["flicker_hold"] for r in per_runs])),
        "flicker_motion": _rnd(_mean([r["flicker_motion"] for r in per_runs])),
        "dup_ratio": _rnd(_mean([r["dup_ratio"] for r in per_runs])),
        "clip": _rnd(_mean([r["clip"] for r in per_runs]), 4),
        "clip_gain": _rnd(_mean([r["clip_gain"] for r in per_runs]), 4),
        "vram_mb": vram,
        "sheet": f"sheet_{name}.jpg",
        "runs": per_runs,
        "_compare": compare,
    }

    header = [
        f"{name} — {cfg.get('desc', '')}",
        "server.py " + row["server_flags"],
        f"p50 {row['infer_ms_p50']} ms · fps {row['fps']} · flicker {row['flicker']} "
        f"(hold {row['flicker_hold']} / motion {row['flicker_motion']}) · dup {row['dup_ratio']} "
        f"· clip {row['clip']}",
        f"cols: frames {idxs} of {args.frames} (hold {hold} each end) · "
        "rows per scene: input, then one row per prompt",
    ]
    sheet_rows = []
    for si, sc in enumerate(scenes):
        sheet_rows.append((f"{sc['name']} in", sheet_cells[(si, "in")]))
        for pi in range(len(prompts)):
            sheet_rows.append((f"{sc['name']} p{pi}", sheet_cells[(si, f"p{pi}")]))
    build_contact_sheet(out_dir / row["sheet"], header, sheet_rows)
    return row


# --------------------------------------------------------------------------- #
# ranking + reports
# --------------------------------------------------------------------------- #

def rank_rows(rows: list[dict[str, Any]], fps_floor: float) -> list[dict[str, Any]]:
    ok = [r for r in rows if not r.get("error")]
    err = [r for r in rows if r.get("error")]
    if ok:
        fls = [r["flicker"] for r in ok if r["flicker"] is not None]
        clips = [r["clip"] for r in ok]
        use_clip = all(c is not None for c in clips)

        def norm(v: float, vals: list[float]) -> float:
            lo, hi = min(vals), max(vals)
            return 0.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo)

        for r in ok:
            fn = norm(r["flicker"], fls) if fls and r["flicker"] is not None else 0.0
            cn = norm(r["clip"], clips) if use_clip else 0.0
            r["quality_score"] = round(cn - FLICKER_WEIGHT * fn, 4)
            r["meets_fps_floor"] = bool(r["fps"] >= fps_floor)
        ok.sort(key=lambda r: (not r["meets_fps_floor"], -r["quality_score"]))
        for i, r in enumerate(ok):
            r["rank"] = i + 1
    for r in err:
        r["rank"] = None
    return ok + err


def _fmt(v: Any, nd: Optional[int] = None) -> str:
    if v is None:
        return "—"
    if nd is not None and isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def write_results(out_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any],
                  prompts: list[str]) -> None:
    json_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": json_rows}, indent=2) + "\n")

    md: list[str] = ["# bench_quality results", ""]
    md.append(f"- generated {meta['generated_at']} · pipeline `{meta['pipeline']}` "
              f"· {meta['device']}")
    md.append(f"- scenes: " + ", ".join(f"{n} ({o.split(':')[0]})" for n, o in meta["scenes"])
              + f" · {meta['frames']} frames/scene (hold {meta['hold']} each end) "
              f"· {len(prompts)} prompts")
    md.append(f"- clip backend: {meta['clip_backend'] or 'unavailable — ranked on flicker only'} "
              f"· fps floor {meta['fps_floor']} "
              f"· quality_score = clip_norm − {FLICKER_WEIGHT}·flicker_norm")
    md.append("")
    ok = [r for r in rows if not r.get("error")]
    if ok:
        md.append("| # | config | model | t_index | cfg | noise | filter | vae "
                  "| p50 ms | fps | flicker | hold | motion | dup | clip | Δclip | ≥floor |")
        md.append("|--:|---|---|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|")
        for r in ok:
            p = r["params"]
            cfg_s = "none" if p["cfg_type"] == "none" else \
                f"self {p['guidance_scale']}/{p['delta']}"
            filt = "—" if p["similar_filter"] is None else \
                f"{p['similar_filter']}/{p['similar_max_skip']}"
            model = p["model"].split("/")[-1] + (" +lcm" if p["lcm_lora"] else "")
            md.append("| " + " | ".join([
                str(r["rank"]), r["name"], model, str(p["t_index"]), cfg_s,
                p["noise_mode"][:3], filt, p["vae"],
                _fmt(r["infer_ms_p50"], 1), _fmt(r["fps"], 1),
                _fmt(r["flicker"], 2), _fmt(r["flicker_hold"], 2),
                _fmt(r["flicker_motion"], 2), _fmt(r["dup_ratio"], 2),
                _fmt(r["clip"], 4), _fmt(r["clip_gain"], 4),
                "yes" if r["meets_fps_floor"] else "NO",
            ]) + " |")
        md.append("")
        md.append("## Boot the winner")
        md.append("")
        md.append("```bash")
        md.append(f"python server.py {ok[0]['server_flags']}")
        md.append("```")
        md.append("")
        md.append("## All configs")
        md.append("")
        for r in ok:
            md.append(f"- **{r['name']}** (score {_fmt(r.get('quality_score'))}, "
                      f"vram {_fmt(r['vram_mb'])} MB, load {_fmt(r['load_s'])} s) — {r['desc']}  ")
            md.append(f"  `server.py {r['server_flags']}`")
    failed = [r for r in rows if r.get("error")]
    if failed:
        md.append("")
        md.append("## Failed configs")
        md.append("")
        for r in failed:
            md.append(f"- **{r['name']}** — `{r['error']}`")
    md += [
        "",
        "## Metric notes",
        "",
        "- **flicker**: mean |out_t − out_{t−1}| (grayscale) on pixels where the blurred",
        f"  input pair changed < {STATIC_THRESH} levels — output churn not explained by input",
        "  motion. Lower is better. **hold** = static-camera segments only (pure",
        "  sitting-still shimmer); **motion** = during the slow pan.",
        "- **dup**: fraction of bit-identical consecutive outputs — the similar-image",
        "  filter's replay/freeze signature. High dup zeroes flicker by construction:",
        "  judge filter configs by motion flicker + the contact sheet, not hold flicker.",
        "- **clip / Δclip**: CLIP cosine(prompt, output frames), and the gain over the raw",
        "  inputs (Δclip > 0 = the restyle moved the image toward the prompt).",
        "- Scores are cheap proxies. Before shipping a winner, eyeball its",
        "  `sheet_*.jpg` and `compare.jpg`, then A/B live via the printed server flags.",
    ]
    (out_dir / "results.md").write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["stream", "fake"], default="stream",
                    help="fake = CPU colormap stand-in (local plumbing proof, no torch)")
    ap.add_argument("--out-dir", default="bench_out")
    ap.add_argument("--frames", type=int, default=24, help="frames per scene sequence")
    ap.add_argument("--hold", type=int, default=None,
                    help="static frames at each end of a sequence (default frames//4)")
    ap.add_argument("--warmup", type=int, default=6, help="pipeline warmup frames per config")
    ap.add_argument("--configs", default=None, help="csv of config names (default: all)")
    ap.add_argument("--list", action="store_true", help="list configs + server flags, exit")
    ap.add_argument("--no-download", action="store_true",
                    help="never hit the network; use synthetic photo-like scenes")
    ap.add_argument("--no-clip", action="store_true", help="skip CLIP prompt-adherence scoring")
    ap.add_argument("--fps-floor", type=float, default=8.0,
                    help="configs under this fps rank below all configs above it")
    ap.add_argument("--seed", type=int, default=42, help="scene/sequence rng seed")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    configs = [merged(c) for c in CONFIGS]
    if args.list:
        for c in configs:
            print(f"{c['name']:22s} {c['desc']}")
            print(f"{'':22s} server.py {server_flags(c)}")
        return 0
    if args.configs:
        want = [w.strip() for w in args.configs.split(",") if w.strip()]
        by = {c["name"]: c for c in configs}
        unknown = [w for w in want if w not in by]
        if unknown:
            print(f"unknown config(s) {unknown}; available: {sorted(by)}", file=sys.stderr)
            return 2
        configs = [by[w] for w in want]
    hold = args.hold if args.hold is not None else max(2, args.frames // 4)
    if args.frames < 2 * hold + 4:
        print(f"--frames {args.frames} too small for hold {hold} "
              f"(need >= {2 * hold + 4})", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = []
    for si, (name, url) in enumerate(SCENES):
        src, origin = fetch_scene(name, url, out_dir / "scenes", args.no_download,
                                  np.random.default_rng(args.seed + 100 + si))
        frames = make_sequence(src, args.frames, hold, np.random.default_rng(args.seed + si))
        scenes.append({"name": name, "origin": origin, "frames": frames})
        log.info("scene %-9s %s", name, origin)

    clip = ClipScorer(enabled=not args.no_clip)
    log.info("CLIP scorer: %s", clip.backend or f"unavailable ({clip.reason})")

    rows: list[dict[str, Any]] = []
    clip_in_cache: dict[tuple[str, str], Optional[float]] = {}
    for cfg in configs:
        try:
            rows.append(run_config(cfg, scenes, PROMPTS, args, clip,
                                   clip_in_cache, out_dir, hold))
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 — one bad config must not kill the sweep
            log.exception("config %s failed", cfg["name"])
            rows.append({"name": cfg["name"], "desc": cfg.get("desc", ""),
                         "params": {k: cfg[k] for k in LEVER_KEYS},
                         "server_flags": server_flags(cfg),
                         "error": f"{type(e).__name__}: {e}"})
            free_gpu()

    ranked = rank_rows(rows, args.fps_floor)
    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": args.pipeline,
        "device": device_info(),
        "frames": args.frames,
        "hold": hold,
        "warmup": args.warmup,
        "fps_floor": args.fps_floor,
        "flicker_weight": FLICKER_WEIGHT,
        "sensor_noise": SENSOR_NOISE,
        "prompts": PROMPTS,
        "scenes": [(s["name"], s["origin"]) for s in scenes],
        "clip_backend": clip.backend,
        "clip_unavailable_reason": None if clip.available else clip.reason,
        "seed": args.seed,
    }
    write_results(out_dir, ranked, meta, PROMPTS)
    build_compare(out_dir, ranked, PROMPTS)

    print()
    print(f"{'#':>2} {'config':22} {'fps':>6} {'p50ms':>8} {'flicker':>8} "
          f"{'dup':>5} {'clip':>7}  floor")
    for r in ranked:
        if r.get("error"):
            print(f"{'—':>2} {r['name']:22} ERROR: {r['error']}")
        else:
            print(f"{r['rank']:>2} {r['name']:22} {_fmt(r['fps']):>6} "
                  f"{_fmt(r['infer_ms_p50']):>8} {_fmt(r['flicker']):>8} "
                  f"{_fmt(r['dup_ratio']):>5} {_fmt(r['clip']):>7}  "
                  f"{'yes' if r.get('meets_fps_floor') else 'NO'}")
    print(f"\nwrote {out_dir}/results.md, results.json, sheet_*.jpg, compare.jpg")
    return 0 if any(not r.get("error") for r in ranked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
