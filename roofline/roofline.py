#!/usr/bin/env python3
"""Roofline model for streaming causal video diffusion.

Predicts sustained fps, TTFF, and KV footprint for chunk-wise autoregressive
video DiTs (Wan-class) across hardware, from first principles:

  T_math  = FLOPs / (MFU * peak_FLOPs/s)          (DiT is compute-bound: chunk
                                                    of ~1.5-4.7k tokens >> B_crit ~300)
  T_vae   = ms/frame prior, scaled by pixels and  (VAE decode is treated as
            HBM bandwidth relative to H100         bandwidth-bound)
  fps     = frames_per_chunk / chunk_time          (serial or DiT/VAE overlapped)

The full Wan VAE and text-encoder constants are calibrated from Phase-1 H100
measurements. TAEHV remains an analytical per-frame prior here because rolling
per-block decode has material scheduling overhead; the measured gate is printed
separately rather than forced into the simple serial/overlap model.

Anchors (published, mid-2026): Self-Forcing ~17 fps H100 / ~10 fps 4090;
LongLive 20.7 fps H100 (24.8 fp8); Krea-Realtime-14B 11 fps B200;
CausVid 9.4 fps; MotionStream 29 fps H100; StreamDiffusionV2 ~16 fps/GPU
(64 fps on 4xH100, 1.3B). Our measured CF++ gate is also reported.

stdlib-only. Run: python3 roofline.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# ---------------------------------------------------------------- hardware

@dataclass(frozen=True)
class Hardware:
    name: str
    bf16_tflops: float          # dense, no sparsity
    fp8_tflops: float | None    # None = no usable fp8 tensor path
    hbm_tbs: float              # TB/s
    mem_gb: float
    note: str = ""


H100 = Hardware("H100 SXM", 989, 1979, 3.35, 80)
B200 = Hardware("B200", 2250, 4500, 8.0, 192, "nominal dense specs")
RTX4090 = Hardware("RTX 4090", 165, 330, 1.008, 24)
RTX5090 = Hardware("RTX 5090", 210, 419, 1.792, 32, "nominal")
M4MAX = Hardware("M4 Max 48GB", 34, None, 0.546, 48, "fp16 nominal, no tensor cores; MPS/MLX kernel quality dominates")
M4PRO = Hardware("M4 Pro 48GB", 17, None, 0.273, 48, "fp16 nominal")
TITANXP = Hardware("Titan Xp", 12.15, None, 0.548, 12, "fp32 only: Pascal fp16 is 1/64 rate; no FlashAttention")

# ------------------------------------------------------------------ models

@dataclass(frozen=True)
class Model:
    name: str
    params: float               # DiT params (excl. text encoder)
    layers: int
    d_model: int
    spatial_downsample: int = 16   # VAE 8x8 * 2x2 patchify (Wan2.2-5B: VAE 16x16, patchify 1 -> same 16)
    temporal_downsample: int = 4   # video frames per latent frame
    text_tokens: int = 512


# Constants verified against HF config.json 2026-07-19:
#   Wan2.1-T2V-1.3B: dim 1536, layers 30, ffn 8960, heads 12, in_dim 16
#   Wan2.1-T2V-14B:  dim 5120, layers 40, ffn 13824, heads 40, in_dim 16
#   Wan2.2-TI2V-5B:  dim 3072, layers 30, ffn 14336, heads 24, in_dim 48
#     -> in_dim 48 = Wan2.2 VAE (16x16 spatial, 48ch) + 2x2 patchify = 32x
#        spatial downsample: 390 tok/latent frame at 480x832 (4x FEWER than
#        1.3B!) -- the deeper VAE is how a 5B fits on a phone (MobileWan).
WAN_1_3B = Model("Wan2.1-1.3B", 1.3e9, 30, 1536)
WAN_5B = Model("Wan2.2-5B", 5.0e9, 30, 3072, spatial_downsample=32)
WAN_14B = Model("Wan2.1-14B", 14.0e9, 40, 5120)

# --------------------------------------------------------------- workloads

@dataclass(frozen=True)
class Workload:
    """One streaming-generation configuration.

    ctx_latent_frames: EFFECTIVE kv-context in latent frames (including the
    current chunk) that self-attention sees per step. Full-context models over
    a 21-frame window average ~12; short-window models (LongLive) ~6-8;
    chunk-local = chunk_latent_frames. Linear-state past-context ~= chunk-local.
    """
    height: int = 480
    width: int = 832
    chunk_latent_frames: int = 3
    steps: int = 4
    ctx_latent_frames: int = 12
    precision: str = "bf16"       # 'bf16' | 'fp8'
    vae: str = "wan"              # 'wan' | 'taehv'
    vae_overlap: bool = False     # decode chunk t while denoising chunk t+1
    per_chunk_overhead_ms: float = 0.0   # scheduler/launch overhead prior
    # MEASURED discovery (Phase-1 day 1): Self-Forcing runs ONE EXTRA forward
    # per chunk after denoising -- a timestep-0 pass on the clean chunk to
    # write its KV into cache (causal_inference.py "Step 3.3"). So SF-lineage
    # pipelines do steps+1 network evals per chunk. Confirmed for SF; assumed
    # for CausVid/Krea (same lineage); unknown for LongLive/APT2 (set 0).
    extra_forwards_per_chunk: int = 0


# VAE decode: ms per output frame at 480x832 on H100, treated as
# bandwidth-bound (scaled by pixel count and by H100_BW / hw_BW).
# 'wan' MEASURED 2026-07-19 (Phase-1 day 1, H100 SXM, Self-Forcing @33593df,
# 5 trials, <0.1% var): 1853 ms / 81 frames = 22.9 ms/f (batch decode).
# Old APT1-derived prior was 27.4. 'taehv' remains a prior.
VAE_MS_PER_FRAME_480P_H100 = {"wan": 22.9, "taehv": 3.0}
# Text encoder (umT5-xxl, batch 1) MEASURED same session: 270 ms cold (first
# call, includes CUDA warmup), 35 ms warm. Old prior 0.28s was the COLD number.
TEXT_ENCODER_COLD_S_H100 = 0.27
TEXT_ENCODER_WARM_S_H100 = 0.035
BASE_PIXELS = 480 * 832


# ------------------------------------------------------------- core math

def tokens_per_latent_frame(m: Model, w: Workload) -> int:
    return (w.height // m.spatial_downsample) * (w.width // m.spatial_downsample)


def dit_flops_per_step(m: Model, w: Workload) -> dict[str, float]:
    """FLOPs for ONE denoising forward pass of one chunk."""
    t_chunk = tokens_per_latent_frame(m, w) * w.chunk_latent_frames
    s_ctx = tokens_per_latent_frame(m, w) * w.ctx_latent_frames
    linear = 2.0 * m.params * t_chunk
    self_attn = 4.0 * t_chunk * s_ctx * m.d_model * m.layers
    cross_attn = 4.0 * t_chunk * m.text_tokens * m.d_model * m.layers
    return {"linear": linear, "self_attn": self_attn, "cross_attn": cross_attn,
            "total": linear + self_attn + cross_attn}


def peak_tflops(hw: Hardware, precision: str) -> float:
    if precision == "fp8" and hw.fp8_tflops:
        return hw.fp8_tflops
    return hw.bf16_tflops


def dit_chunk_seconds(m: Model, hw: Hardware, w: Workload, mfu: float) -> float:
    per_step = dit_flops_per_step(m, w)["total"]
    forwards = w.steps + w.extra_forwards_per_chunk
    return (per_step * forwards) / (mfu * peak_tflops(hw, w.precision) * 1e12)


def vae_chunk_seconds(hw: Hardware, w: Workload, m: Model) -> float:
    ms = VAE_MS_PER_FRAME_480P_H100[w.vae]
    ms *= (w.height * w.width) / BASE_PIXELS
    ms *= H100.hbm_tbs / hw.hbm_tbs            # bandwidth-bound scaling
    frames = w.chunk_latent_frames * m.temporal_downsample
    return ms * frames / 1e3


def chunk_seconds(m: Model, hw: Hardware, w: Workload, mfu: float) -> float:
    dit = dit_chunk_seconds(m, hw, w, mfu)
    vae = vae_chunk_seconds(hw, w, m)
    body = max(dit, vae) if w.vae_overlap else dit + vae
    return body + w.per_chunk_overhead_ms / 1e3


def sustained_fps(m: Model, hw: Hardware, w: Workload, mfu: float) -> float:
    frames = w.chunk_latent_frames * m.temporal_downsample
    return frames / chunk_seconds(m, hw, w, mfu)


def ttff_seconds(m: Model, hw: Hardware, w: Workload, mfu: float) -> dict[str, float]:
    """Cold-start time to first frames: text encode + first chunk (short context) + its decode.
    Overlap never helps TTFF (first decode waits on first denoise)."""
    first = replace(w, ctx_latent_frames=w.chunk_latent_frames)
    scale = H100.bf16_tflops / hw.bf16_tflops
    text_cold = TEXT_ENCODER_COLD_S_H100 * scale
    text_warm = TEXT_ENCODER_WARM_S_H100 * scale
    dit = dit_chunk_seconds(m, hw, first, mfu)
    vae = vae_chunk_seconds(hw, first, m)
    return {"text": text_cold, "text_warm": text_warm, "dit": dit, "vae": vae,
            "total": text_cold + dit + vae, "total_warm": text_warm + dit + vae}


def kv_gb_per_latent_frame(m: Model, w: Workload, kv_bytes: int = 2) -> float:
    return tokens_per_latent_frame(m, w) * m.layers * 2 * m.d_model * kv_bytes / 1e9


# --------------------------------------------------------------- reporting

# Grid revised after Phase-1 day 1: back-solved effective MFU on stock
# Self-Forcing/H100 is ~27.6% (5-forward accounting; 22.1% under 4-forward).
# 40% represents the kernel-work upside (compile/fused attention), not today.
MFU_GRID = (0.25, 0.30, 0.40)


def fmt_fps(m: Model, hw: Hardware, w: Workload) -> str:
    return "/".join(f"{sustained_fps(m, hw, w, mfu):5.1f}" for mfu in MFU_GRID)


def line(ch: str = "-", n: int = 100) -> None:
    print(ch * n)


def header(title: str) -> None:
    print()
    line("=")
    print(title)
    line("=")


def report_flops_breakdown() -> None:
    header("T0. FLOPs per denoise step, Wan2.1-1.3B @ 480x832, chunk = 3 latent frames (4,680 tokens)")
    print(f"{'context policy':<38}{'linear':>10}{'self-attn':>12}{'cross':>8}{'total/step':>12}{'x4 steps':>10}")
    for label, ctx in [("chunk-local (3 lf)", 3), ("short window (7 lf, LongLive-ish)", 7),
                       ("full 5s avg (12 lf, Self-Forcing)", 12), ("full 5s worst (21 lf)", 21)]:
        w = Workload(ctx_latent_frames=ctx)
        f = dit_flops_per_step(WAN_1_3B, w)
        print(f"{label:<38}{f['linear']/1e12:>9.1f}T{f['self_attn']/1e12:>11.1f}T{f['cross_attn']/1e12:>7.1f}T" +
              f"{f['total']/1e12:>11.1f}T{f['total']*4/1e12:>9.0f}T")
    tok = tokens_per_latent_frame(WAN_1_3B, Workload())
    print(f"\n  tokens/latent-frame = {tok}   B_crit(H100 bf16) ~= {989/3.35/1e0:.0f} tokens  ->  compute-bound by >10x")


def report_main_scenarios() -> None:
    header("T1. Sustained fps @ MFU 25/30/40% -- Wan2.1-1.3B, 480x832, 4-step, chunk 3, full-ctx avg 12 lf")
    print(f"{'hardware':<14}{'prec':<6}{'vae':<7}{'overlap':<9}{'fps @25/30/40% MFU':>22}   note")
    rows = [
        (H100, "bf16", "wan", False, "~= today's Self-Forcing setup (measured 17)"),
        (H100, "bf16", "wan", True, "overlap only"),
        (H100, "fp8", "taehv", True, "optimized target"),
        (RTX4090, "bf16", "wan", False, "naive port"),
        (RTX4090, "bf16", "taehv", True, "SF '~10 fps w/ optimizations' regime (short ctx helps too)"),
        (RTX4090, "fp8", "taehv", True, "4-step: does NOT reach 24"),
        (RTX5090, "fp8", "taehv", True, "4-step"),
    ]
    for hw, prec, vae, ov, note in rows:
        w = Workload(precision=prec, vae=vae, vae_overlap=ov)
        print(f"{hw.name:<14}{prec:<6}{vae:<7}{str(ov):<9}{fmt_fps(WAN_1_3B, hw, w):>22}   {note}")
    print("\n  HISTORICAL 4090 ANALYTICAL PATH (REJECTED BY PHASE-1 MEASUREMENT):")
    for steps in (4, 3, 2, 1):
        w = Workload(precision="fp8", vae="taehv", vae_overlap=True, ctx_latent_frames=7, steps=steps)
        tag = "  <-- >=24 fps closes here" if sustained_fps(WAN_1_3B, RTX4090, w, 0.40) >= 24 else ""
        print(f"{'RTX 4090':<14}{'fp8':<6}{steps}-step{'':<11}{fmt_fps(WAN_1_3B, RTX4090, w):>22}{tag}")
    print("\n  MEASURED CORRECTION: 4090 fp8 was 0.90x bf16 and CF++ students retain a cache-refresh")
    print("  forward. The 4090 >=24 headline failed; it is now an interactive/low-latency secondary.")
    print("  The primary measured path is CF++ 1-step + rolling TAEHV on one H100 (see T1b).")


def report_measured_h100_gate() -> None:
    header("T1b. Phase-1 measured H100 gate -- 480x832, 81 decoded frames")
    print(f"{'configuration':<43}{'mode':<19}{'warm e2e fps':>14}{'first GPU RGB':>16}{'P95 interval':>15}")
    rows = [
        ("CF++ 1-step + full Wan VAE", "batch", 19.7065, "--", "--"),
        ("CF++ 1-step + TAEHV", "batch, not stream", 34.3743, "~0.168s est", "--"),
        ("CF++ 1-step + rolling TAEHV", "streaming", 31.0394, "0.242s", "37.62ms"),
        ("CF++ 2-step + full Wan VAE", "batch", 15.7579, "--", "--"),
        ("CF++ 2-step + TAEHV", "batch, not stream", 24.0503, "~0.168s est", "--"),
        ("CF++ 2-step + rolling TAEHV", "streaming", 22.1599, "0.250s", "54.40ms"),
    ]
    for config, mode, fps, first_rgb, p95 in rows:
        print(f"{config:<43}{mode:<19}{fps:>14.2f}{first_rgb:>16}{p95:>15}")
    print("\n  Cold/warmup n=1, 1-step streaming: 24.64 fps / 0.867s first GPU RGB event.")
    print("  Historical latency ends at the post-decode CUDA event: not CPU-ready or browser-visible.")
    print("  Separate-stream overlap adds only 0.38% over serial rolling decode (31.04 vs 30.92 fps).")
    print("  Automated blind quality: CF1 5.67 > SF4 4.67 > CF2 4.33; absolute 7/10 gate FAILED.")
    print("  Consensus verdict B: performance milestone achieved; quality-qualified headline remains open.")


def report_validation() -> None:
    """Two anchor types:
    - 'band': measured must fall within 1.5x of some MFU-grid prediction
      (systems that run close to their roofline). NOTE this is a BAND check,
      not a strict bracket: SF-H100 (17.0 vs floor 17.4 -> eff. MFU ~29%) and
      LongLive-fp8 (24.8 vs floor 26.8 -> fp8 ~7.5% short of MFU-parity even
      on H100) sit just BELOW the floor. The fp8 shortfall is empirical
      evidence against the fp8-parity assumption and is a first-class Phase-1
      measurement (gate-consensus finding, kimi-k3/gpt/glm).
    - 'upper': model is an UPPER BOUND; measured sits below because the released
      implementation carries overheads the model deliberately excludes (pipeline
      bubbles, v2v encode, Python dispatch, pre-optimization code). The gap IS
      the engineering-margin story, quantified."""
    header("T2. Validation vs published anchors (predicted fps @25/30/40% MFU vs measured)")
    print(f"{'anchor':<34}{'assumed config':<40}{'pred @25/30/40':>18}{'meas':>7}")
    anchors = [
        ("Self-Forcing 1.3B H100 (paper)", WAN_1_3B, H100,
         Workload(ctx_latent_frames=12, vae="wan", extra_forwards_per_chunk=1),
         "4+1fwd chunk3 ctx12 wanVAE serial", 17.0, "band"),
        ("Self-Forcing H100 (OUR DAY-1 MEAS.)", WAN_1_3B, H100,
         Workload(ctx_latent_frames=12, vae="wan", extra_forwards_per_chunk=1),
         "same; measured batch e2e 81f/5.55s", 14.6, "band"),
        ("Self-Forcing 1.3B 4090 (optim.)", WAN_1_3B, RTX4090,
         Workload(ctx_latent_frames=7, vae="taehv", vae_overlap=True, extra_forwards_per_chunk=1),
         "4+1fwd chunk3 ctx7 taehv overlap", 10.0, "band"),
        ("LongLive 1.3B H100", WAN_1_3B, H100,
         Workload(chunk_latent_frames=1, ctx_latent_frames=7, vae="wan", vae_overlap=False),
         "4stp chunk1 sink+window~7 serial", 20.7, "band"),
        ("LongLive 1.3B H100 fp8", WAN_1_3B, H100,
         Workload(chunk_latent_frames=1, ctx_latent_frames=7, precision="fp8", vae="wan", vae_overlap=False),
         "as above, fp8", 24.8, "band"),
        ("Krea-Realtime 14B B200", WAN_14B, B200,
         Workload(ctx_latent_frames=12, vae="wan", extra_forwards_per_chunk=1),
         "4+1fwd chunk3 ctx12 serial", 11.0, "band"),
        ("MotionStream 1.3B H100 (FA3)", WAN_1_3B, H100,
         Workload(ctx_latent_frames=7, vae="wan", vae_overlap=True, extra_forwards_per_chunk=1),
         "4+1fwd chunk3 sink1+win1 overlap", 29.0, "band"),
        ("CausVid 1.3B (2025 impl.)", WAN_1_3B, H100,
         Workload(ctx_latent_frames=12, vae="wan", extra_forwards_per_chunk=1),
         "4+1fwd chunk3 ctx12 serial (2025 code)", 9.4, "upper"),
        ("StreamDiffusionV2 1.3B 4xH100/4", WAN_1_3B, H100,
         Workload(steps=2, ctx_latent_frames=7, vae="taehv", vae_overlap=True),
         "2stp ctx7 streamVAE (per-GPU share)", 16.0, "upper"),
    ]
    n_ok = 0
    for name, m, hw, w, cfg, meas, kind in anchors:
        preds = [sustained_fps(m, hw, w, mfu) for mfu in MFU_GRID]
        if kind == "band":
            ok = any(meas / 1.5 <= p <= meas * 1.5 for p in preds)
        else:  # upper bound: all predictions must sit at or above measured
            ok = all(p >= meas for p in preds)
        n_ok += ok
        flag = f"  ok ({kind})" if ok else f"  MISS ({kind})"
        print(f"{name:<34}{cfg:<40}{'/'.join(f'{p:5.1f}' for p in preds):>18}{meas:>7.1f}{flag}")
    print(f"\n  {n_ok}/{len(anchors)} anchors consistent. 'upper' anchors: model excludes pipeline bubbles,")
    print("  v2v VAE-encode, dispatch overhead -- measured below prediction quantifies that overhead")
    print("  (SDV2's per-GPU share also divides a 4-stage pipeline by 4, which double-counts bubbles).")


def report_14b() -> None:
    header("T3. 14B reality check (chunk 3, ctx 12, 4-step)")
    for hw, prec, vae, ov, note in [
        (H100, "bf16", "wan", False, "why 14B-on-H100 does NOT close"),
        (H100, "fp8", "taehv", True, "still short of 24"),
        (B200, "bf16", "wan", False, "~Krea today"),
        (B200, "fp8", "taehv", True, "14B real-time is a B200+fp8 story"),
    ]:
        w = Workload(precision=prec, vae=vae, vae_overlap=ov)
        print(f"{hw.name:<10}{prec:<6}{vae:<7}ovl={str(ov):<7}{fmt_fps(WAN_14B, hw, w):>22}   {note}")
    print("\n  1-2 step students change this: fps scales ~linearly in 1/steps (quality cost must be reported).")


def report_kv_and_ttff() -> None:
    header("T4. KV cache (bf16) and TTFF")
    for m in (WAN_1_3B, WAN_5B, WAN_14B):
        per = kv_gb_per_latent_frame(m, Workload())
        print(f"{m.name:<14} {per*1e3:6.0f} MB/latent-frame   21-frame window: {per*21:5.1f} GB   (fp8 KV: {per*21/2:5.1f} GB)")
    print()
    for hw in (H100, RTX4090):
        for prec in ("bf16", "fp8"):
            w = Workload(precision=prec, vae="taehv")
            t = ttff_seconds(WAN_1_3B, hw, w, 0.40)
            warm = t["total_warm"]
            print(f"TTFF {hw.name:<10} {prec:<5} @40% MFU: text {t['text']*1e3:5.0f} + dit {t['dit']*1e3:5.0f} + vae {t['vae']*1e3:4.0f}" +
                  f" = cold {t['total']*1e3:6.0f} ms | warm (text cached) {warm*1e3:5.0f} ms")
    print("\n  <1s TTFF on consumer GPUs requires caching/overlapping the umT5 text encode (cold-start is text-dominated).")


def report_local_hardware() -> None:
    header("T5. Local-hardware reality check (1.3B, chunk 3, ctx 7, taehv, overlap, 4-step)")
    w = Workload(ctx_latent_frames=7, vae="taehv", vae_overlap=True)
    for hw, note in [(TITANXP, "fp32 only; also 12GB won't hold weights+text-enc+KV comfortably"),
                     (M4PRO, "MPS/MLX kernel quality will land well below these MFUs"),
                     (M4MAX, "dev box; not a demo target")]:
        print(f"{hw.name:<14}{fmt_fps(WAN_1_3B, hw, w):>22} fps   {note}")
    print("\n  -> consensus decision confirmed by the math: Titan Xp = case-study row; M4 = dev machine; MLX port cut.")


if __name__ == "__main__":
    print("Roofline model for streaming causal video diffusion -- Phase 1 calibrated")
    report_flops_breakdown()
    report_main_scenarios()
    report_measured_h100_gate()
    report_validation()
    report_14b()
    report_kv_and_ttff()
    report_local_hardware()
    print()
    line("=")
    print("Caveats: single-MFU-per-config is a simplification (attn kernels < dense matmul MFU); full Wan")
    print("VAE/text are H100 measurements but TAEHV is still a simple prior here; fp8-MFU parity was")
    print("empirically false on 4090; max(DiT,VAE) overlap ignores scheduling/HBM contention. Use T1b")
    print("for the actual corrected rolling-stream result, and report its open quality gate beside fps.")
    line("=")
