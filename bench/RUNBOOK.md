# Phase-1 Runbook — Day 1: Self-Forcing on a rented H100

Goal of day 1: replace roofline.py's three priors (VAE ms/frame, text-encode s, effective MFU) with **measured** values, and produce the first `bench/results/*.json`. Budget: **≤ $30** (one ~8h spot session, kill early if done). Log actuals in PLAN.md ledger.

## 0. Rental spec (before paying, check ALL)

- **H100 SXM 80GB (HBM3, 3.35 TB/s)** — must be SXM. PCIe H100 is 2.0 TB/s and breaks anchor comparability (skews the BW-scaled VAE prior too). Verify IMMEDIATELY: `nvidia-smi -q | grep -iE "product name|max.*bandwidth"`; wrong SKU → kill instance.
- Spot/interruptible fine (inference only). $2-3/hr on RunPod/Vast/Lambda. **Fallback: if SXM spot is unavailable, pay on-demand for 2-3h** — at a $30 day budget, availability beats price (gate-consensus note, glm).
- ≥ 200 GB disk (Wan2.1-T2V-1.3B ~17 GB, umT5-xxl ~11 GB, SF checkpoint ~3 GB, VAE, prompts, headroom). **Pre-stage downloads first thing** — don't burn spot hours on 30-60 GB of weights (kimi-k3).
- Image: CUDA 12.4+, python 3.10/3.11, torch 2.4+cu124. RunPod `pytorch:2.4.0-py3.11-cuda12.4.1` class image is fine.
- Copy `harness.py` up; copy `results/*.json` back BEFORE killing the box.
- **Pin and record with every run**: Self-Forcing commit hash, model revision, seed, driver + CUDA + torch versions, and a `nvidia-smi -q -d CLOCK,POWER` snapshot — a thermally/power-capped instance silently corrupts the MFU back-solve (kimi-k3, gpt).

## 1. Stand up Self-Forcing (stock, unmodified)

```bash
git clone https://github.com/guandeh17/Self-Forcing && cd Self-Forcing
pip install -r requirements.txt && python setup.py develop
# weights (repo README is authoritative if paths moved):
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
# smoke: stock chunk-wise autoregressive inference, 1 prompt
python minimal_inference/autoregressive_inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --output_folder outputs/smoke --prompt_file_path prompts/smoke.txt
```

Gate: produces a coherent 81-frame 480×832 video. Eyeball it (quality next to every number — grab 2-3 sample MP4s for the record).

## 2. Instrument (patch points)

Copy `harness.py` next to the inference script. In the repo's inference pipeline (`pipeline/` / `minimal_inference/`), add marks at exactly four places:

1. after text encoding → `rec.mark_text_encoded()`
2. top of each AR chunk loop iteration → `rec.chunk_start()`
3. after the chunk's last denoise step (before VAE) → `rec.mark_chunk_denoised()`
4. after VAE decode of the chunk → `rec.mark_chunk_decoded(frames=<decoded video frames>)`

Marks synchronize CUDA themselves. Keep the patch to ≤ ~15 lines; save as a git stash/diff for reuse on LongLive/SDV2 later.

## 3. Day-1 measurement matrix (all on the SAME box, ≥ 3 prompts × ≥ 60s-equivalent each)

Warm up (≥1 full unmeasured generation) before every measured run; ≥3 measured trials per run id — report median + spread, never a single trial (gpt).

| run id | what | purpose |
|---|---|---|
| `sf_stock` | unmodified defaults (4-step, chunk 3, full ctx, wan VAE, serial) | anchor reproduction: expect ~17 fps ballpark (tolerance ±20%, not exact) |
| `sf_long` | extend rollout (sliding window / more chunks) | steady-state jitter P50/P95 over ≥ 60s |
| `sf_profile` | 1 chunk under `torch.profiler` (separate run — profiler skews timing) | where milliseconds go: per-op MFU, attn vs linear split; also empirically confirms the steps-per-chunk accounting (glm) |
| `sf_res` | 480×832 vs lower res (e.g. 320×576) | fps-vs-tokens scaling check against model |

Harness caveat: the CUDA-synced marks are correct for SERIAL configs (all of day 1). They would serialize an overlapped VAE/DiT pipeline — overlapped configs (SDV2, later) need stream-local CUDA events instead; do not reuse the day-1 patch there blindly (gpt).

## 4. Calibrate the roofline (same day, on the box or after)

From `sf_stock` + `sf_profile` (calibration is **regime-specific** — an H100/bf16/1.3B MFU does not transfer to 4090/fp8 or B200/14B; label every calibrated value with its {hardware, precision, model, ctx} regime — gpt):

1. **VAE prior**: `decode_mean_s / frames_per_chunk` → replaces `VAE_MS_PER_FRAME_480P_H100["wan"]` (prior: 27.4 ms, from APT1 63.3 ms/frame @720p × 0.4336 pixel ratio).
2. **Text-encode prior**: `text_encode_s` → replaces `TEXT_ENCODER_S_H100` (prior: 0.28 s).
3. **Effective MFU**: back-solve `MFU = (FLOPs_per_chunk from roofline.dit_flops_per_step × steps) / (denoise_mean_s × 989e12)`. Expect 0.28-0.45; record actual. **If back-solved MFU < 15%, STOP: the model has a systematic DiT-FLOPs accounting error (glm's flag) — re-derive before spending further.** If MFU varies > 5% across configs, report MFU per-config, not as one scalar (kimi-k3).
4. Re-run `roofline.py` validation with measured priors; update the T2 table and note deltas in the blog draft. Calibration anchors (day-1 SF runs) and validation anchors (LongLive/SDV2/4090, later days) must stay separate — re-checking the anchor you calibrated on is a consistency check, not validation (gpt).

**First-class Phase-1 question (gate consensus, all 4 voters): does fp8 on RTX 4090 (Ada) achieve MFU-parity with bf16 for this DiT?** The LongLive-fp8 anchor already shows ~7.5% parity shortfall on H100. The 4090 headline's 2-step row survives parity shortfall down to ~30% effective MFU (27 fps); the 3-step row does not (needs ≥40%). Measure fp8 vs bf16 denoise time explicitly in the 4090 session; if realized fp8 speedup < 1.3×, escalate to consensus (headline may need a 1-step student or an H100 demo target).

## 5. Teardown checklist

- [ ] `results/*.json` + sample MP4s + profiler trace copied off the box
- [ ] instance killed (verify billing stopped)
- [ ] spend logged in PLAN.md ledger
- [ ] progress log updated (measured priors, MFU, surprises)

## Day 2+ (separate sessions, same pattern)

- LongLive (frame-level AR + fp8 reference) → `ll_stock`, `ll_fp8`
- StreamDiffusionV2 single-GPU (`run_v2v.sh single`, 2-step) → `sdv2_single` — also scouts the codebase for Phase-2 PR targets (their TODO: FP8, TensorRT, dynamic scheduler)
- 4090 session (RunPod ~$0.4-0.7/hr): `sf_stock_4090` — calibrates the consumer-GPU rooflines the headline depends on
