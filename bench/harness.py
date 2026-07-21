#!/usr/bin/env python3
"""Phase-1 measurement harness: hook-based timing recorder for streaming video
diffusion inference, implementing PLAN.md's measurement contract.

Usage: import into a patched inference script on the GPU box, call the marks
around the existing code, get a JSON result comparable against roofline.py
predictions.

    from harness import BenchRecorder
    rec = BenchRecorder(system="self-forcing", model="Wan2.1-1.3B", hardware="H100 SXM",
                        config={"steps": 4, "chunk_latent_frames": 3, "ctx_latent_frames": 12,
                                "precision": "bf16", "vae": "wan", "vae_overlap": False,
                                "height": 480, "width": 832})
    rec.start()
    ... text encoding ...
    rec.mark_text_encoded()
    for chunk in generation_loop:
        rec.chunk_start()
        ... denoise steps ...
        rec.mark_chunk_denoised()
        ... vae decode -> N frames ...
        rec.mark_chunk_decoded(frames=12)
    rec.finish("results/self_forcing_h100_stock.json")

Timing contract:
- Every mark synchronizes CUDA first (when torch+cuda available) so wall-clock
  brackets device work. WARNING (gate-consensus, gpt): these device-wide syncs
  are correct for SERIAL pipelines only -- inserting them into an overlapped
  VAE/DiT pipeline SERIALIZES the overlap and destroys what you are measuring.
  For overlapped configs (SDV2 etc.) instrument with stream-local
  torch.cuda.Event pairs per stream instead; only chunk_start/chunk_decoded
  (chunk walls) remain valid as-is.
- Cold TTFF = first decoded frames available - start (includes text encode).
- Warm TTFF = first decoded frames available - text_encoded (prompt cache hit).
- Sustained fps = frames after the FIRST chunk / wall time after the first
  chunk (steady state; excludes warmup + TTFF).
- P50/P95 over per-chunk wall times, first chunk excluded.

stdlib-only; torch is optional (VRAM + GPU name + sync when present).
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

SCHEMA_VERSION = 1

try:  # torch optional: harness must also run in dry tests off-GPU
    import torch  # type: ignore
    _CUDA = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    _CUDA = False


def _sync() -> None:
    if _CUDA:
        torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


class BenchRecorder:
    def __init__(self, system: str, model: str, hardware: str, config: dict) -> None:
        self.meta = {
            "schema_version": SCHEMA_VERSION,
            "system": system,
            "model": model,
            "hardware": hardware,
            "config": config,
            "host": platform.node(),
            "gpu": (torch.cuda.get_device_name(0) if _CUDA else "none"),
            "torch": (torch.__version__ if torch else "none"),
            "unix_time": time.time(),
        }
        self.t_start: float | None = None
        self.t_text: float | None = None
        self.chunks: list[dict] = []
        self._chunk_t0: float | None = None
        self._chunk_denoised: float | None = None

    def start(self) -> None:
        if _CUDA:
            torch.cuda.reset_peak_memory_stats()
        self.t_start = _now()

    def mark_text_encoded(self) -> None:
        self.t_text = _now()

    def chunk_start(self) -> None:
        self._chunk_t0 = _now()
        self._chunk_denoised = None

    def mark_chunk_denoised(self) -> None:
        self._chunk_denoised = _now()

    def mark_chunk_decoded(self, frames: int) -> None:
        t = _now()
        assert self._chunk_t0 is not None, "chunk_start() not called"
        denoise_s = (self._chunk_denoised - self._chunk_t0) if self._chunk_denoised else None
        self.chunks.append({
            "t_end": t,
            "wall_s": t - self._chunk_t0,
            "denoise_s": denoise_s,
            "decode_s": (t - self._chunk_denoised) if self._chunk_denoised else None,
            "frames": frames,
        })
        self._chunk_t0 = None

    def summary(self) -> dict:
        assert self.t_start is not None and self.chunks, "start() + >=1 chunk required"
        first, rest = self.chunks[0], self.chunks[1:]
        walls = [c["wall_s"] for c in rest]
        steady_frames = sum(c["frames"] for c in rest)
        steady_time = self.chunks[-1]["t_end"] - first["t_end"]
        out = {
            **self.meta,
            "n_chunks": len(self.chunks),
            "total_frames": sum(c["frames"] for c in self.chunks),
            "ttff_cold_s": first["t_end"] - self.t_start,
            "ttff_warm_s": (first["t_end"] - self.t_text) if self.t_text else None,
            "text_encode_s": (self.t_text - self.t_start) if self.t_text else None,
            "sustained_fps": (steady_frames / steady_time) if rest and steady_time > 0 else None,
            "chunk_wall_p50_s": statistics.median(walls) if walls else None,
            "chunk_wall_p95_s": (statistics.quantiles(walls, n=20)[18] if len(walls) >= 5
                                 else (max(walls) if walls else None)),
            "denoise_mean_s": _mean([c["denoise_s"] for c in rest]),
            "decode_mean_s": _mean([c["decode_s"] for c in rest]),
            "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9 if _CUDA else None),
            "chunks": self.chunks,
        }
        return out

    def finish(self, path: str | Path) -> dict:
        s = self.summary()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=2))
        fps = s["sustained_fps"]
        fps_txt = f"sustained {fps:.1f} fps" if fps else "single-chunk run"
        print(f"[harness] {self.meta['system']} on {self.meta['gpu']}: {fps_txt}, " +
              f"cold TTFF {s['ttff_cold_s'] * 1e3:.0f} ms -> {p}")
        return s


def _mean(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    # Dry self-test (no GPU): simulate a 7-chunk run and verify the summary math.
    rec = BenchRecorder("selftest", "none", "cpu", {"steps": 0})
    rec.start()
    time.sleep(0.02)
    rec.mark_text_encoded()
    for i in range(7):
        rec.chunk_start()
        time.sleep(0.03)
        rec.mark_chunk_denoised()
        time.sleep(0.01)
        rec.mark_chunk_decoded(frames=12)
    s = rec.finish("/tmp/harness_selftest.json")
    assert s["n_chunks"] == 7 and s["total_frames"] == 84
    assert 200 < s["sustained_fps"] < 400, s["sustained_fps"]  # 12 frames / ~0.04s
    print("[harness] self-test OK")
