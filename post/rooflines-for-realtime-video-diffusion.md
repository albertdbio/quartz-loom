# Rooflines for Real-Time Video Diffusion

**STATUS: DRAFT v0.3 (Phase 1, performance measured / quality gate open).** H100 + 4090 measurements use Self-Forcing @33593df and Causal-Forcing++ @8db419e on rented RunPod GPUs (2026-07-19). [`roofline.py`](../roofline/roofline.py) remains consistent with 9/9 published anchors. The H100 performance milestone now passes at 31.04 fps warm e2e; publication waits on the pre-registered 7/10 quality gate and a ≥60s sustained run (see §5).

> One-line thesis: **streaming video generation is a sequence of small prefills, not decodes — it lives on the compute roofline, not the bandwidth roofline — and that flips the entire optimization playbook relative to LLM inference.**

---

## Outline

1. Why real-time video generation is suddenly close (Self-Forcing/CausVid/APT2/LongLive in 12 months)
2. The workload, in tokens (Wan-class anatomy)
3. The regime flip: why this is NOT LLM decode
4. Where the milliseconds go (predicted vs measured, 7 anchors)
5. The finding: the 4090 doesn't close; H100 throughput does — but quality becomes the gate
6. KV is a capacity problem, not a bandwidth problem
7. TTFF anatomy
8. Why your gaming PC from 2017 and your MacBook can't (local-hardware rooflines)
9. Measurement contract + caveats (what fps/TTFF include; MFU sensitivity)

---

## 2. The workload, in tokens

480×832 Wan-class T2V: the VAE compresses 8×8 spatially and 4× temporally; the DiT patchifies 2×2. One **latent frame** = (480/16)×(832/16) = **1,560 tokens** and decodes to 4 video frames. A 5s/16fps clip is 21 latent frames ≈ 33k tokens. Chunk-wise AR models (Self-Forcing lineage) denoise a chunk of **3 latent frames = 4,680 tokens** with K denoising steps, then emit its **12 video frames** and move to the next chunk (units matter: 3 *latent* frames = 12 *video* frames — misreading this creates a phantom 4× error in either direction). Real-time means one chunk per 750 ms — all-in: K DiT forwards + VAE decode + overhead.

(Config-verified aside: **Wan2.2-TI2V-5B is different** — `in_dim: 48` means the deeper Wan2.2 VAE (16×16 spatial, 48ch), so with patchify it runs **390 tokens/latent frame**, 4× fewer than Wan2.1-1.3B, and carries **half** the 1.3B's KV bytes per frame (144 vs 288 MB). KV-memory-wise, the deeper VAE — not parameter count — is what made MobileWan's 5B-on-a-phone possible.)

FLOPs per denoise step (Wan2.1-1.3B, 30 layers, d=1536, chunk = 4,680 tokens) — **attention cost is a policy choice**, so state it per policy:

| context policy | linear | self-attn | cross | total/step | ×4 steps |
|---|---|---|---|---|---|
| chunk-local (3 lf) | 12.2T | 4.0T | 0.4T | 16.6T | 67T |
| short window (7 lf, LongLive-ish) | 12.2T | 9.4T | 0.4T | 22.0T | 88T |
| full 5s avg (12 lf, Self-Forcing) | 12.2T | 16.1T | 0.4T | 28.8T | 115T |
| full 5s worst (21 lf) | 12.2T | 28.3T | 0.4T | 40.9T | 163T |

(Linear-attention-state models — MobileWan — collapse past-context attention to ~0 plus a small bandwidth-bound state update; they behave like chunk-local.)

## 3. The regime flip

An LLM decode step processes **1 token**: arithmetic intensity ≈ 1, hopelessly memory-bandwidth-bound; the playbook is KV-cache bytes and weight bytes. A video "decode step" processes **4,680 tokens** — far above the ~300-token critical batch of an H100 (989 bf16 TFLOP/s ÷ 3.35 TB/s ≈ 295). Weight-loading for the whole 1.3B model is ~0.8 ms against 200-360 ms of compute; even reading a full 6 GB KV window is ~2 ms.

**Caveat that keeps this honest**: this holds *only in the bounded-context regime* (chunk-local / windowed / compressed-state attention). Unbounded full-KV causal attention grows O(N²) with video length and eventually leaves the regime — which is exactly why every long-video system (LongLive, MAGI-1, MobileWan) bounds or compresses context. And it describes the DiT; the VAE decoder is a bandwidth-bound convolutional workload, the text encoder is a one-shot cost, and small-chunk regimes leak launch overhead. "Compute-bound" is the headline for the dominant term, not the whole pipeline.

Playbook implied, corrected by measurement: **denoising steps (linear multiplier) > decoder class (full VAE → tiny streaming VAE) > attention structure > kernel/MFU work > precision > stream overlap**. TAEHV changed 1-step e2e from 19.71 to 31.04 fps; putting that already-tiny decoder on a second stream added only 0.38%. KV bytes still matter primarily for capacity and long-horizon quality, while a bad cache implementation can separately waste time on copies and graph breaks.

## 4. Where the milliseconds go — published anchors + measured runs

Model predictions at MFU ∈ {30, 40, 45}% vs published numbers. Two anchor types: *bracket* (well-optimized systems the model should predict within 1.5×) and *upper* (model deliberately excludes overheads the released implementation carries; the gap quantifies them).

| anchor | assumed config | pred @30/40/45% | measured | verdict |
|---|---|---|---|---|
| Self-Forcing 1.3B, H100 | 4-step, chunk 3, ctx 12, full VAE, serial | 16.7 / 19.4 / 20.4 | 17.0 | ok (band; brackets) |
| Self-Forcing 1.3B, 4090 "optimized" | 4-step, ctx 7, tiny VAE, overlap | 6.7 / 9.0 / 10.1 | ~10 | ok (band; brackets) |
| LongLive 1.3B, H100 | 4-step, frame-level, window ~7 | 19.2 / 21.8 / 22.8 | 20.7 | ok (band; brackets) |
| LongLive 1.3B, H100, fp8 | as above | 25.1 / 27.3 / 28.1 | 24.8 | ok (band; 1.2% below floor) |
| Krea-Realtime 14B, B200 | 4-step, chunk 3, ctx 12, serial | 8.9 / 11.5 / 12.7 | 11.0 | ok (band; brackets) |
| CausVid 1.3B (2025 code) | same workload as Self-Forcing | 16.7 / 19.4 / 20.4 | 9.4 | ok (upper) |
| StreamDiffusionV2 1.3B (4×H100 ÷ 4) | 2-step, ctx 7, stream-VAE | 80.8 / 107.7 / 121.2 | ~16/GPU | ok (upper) |

Readings: (a) today's best single-GPU systems run at ~29-40% effective MFU — the "free" engineering margin is ~1.3-1.7×, not the 2×+ a naive FLOP count suggests; (b) CausVid→Self-Forcing (9.4→17 fps, same math) is what one year of implementation work is worth; (c) the SDV2 gap prices pipeline bubbles + v2v encode + scheduling, the things a roofline deliberately ignores; (d) **the LongLive-fp8 anchor lands ~7.5-9% below strict fp8-MFU-parity predictions even on H100** — fp8 parity is an optimistic assumption, and it is exactly the assumption the consumer-GPU headline leans on (measured directly in Phase 1).

## 4b. Measured: where one generation's 5.28 CUDA-seconds actually go (H100, day-2 profiler)

| bucket | time | share | note |
|---|---|---|---|
| flash-attn (self+cross) | 1.75 s | 33.1% | 2,100 calls = 30 layers × 35 forwards × 2 |
| GEMMs (QKV/O/MLP) | ~1.05 s | ~20% | addmm + nvjet/xmma kernels |
| VAE decode | ~1.14 s | ~22% | cudnn convs 924 ms + NCHW↔NHWC transposes 212 ms |
| **KV-cache data movement** | **~1.0 s** | **~19%** | `torch.cat` + `copy_` + DtoD — the cache is rebuilt by concatenation |
| **launch-queue stalls** | **742 ms** | **14%** | "Command Buffer Full" — thousands of small kernels |
| elementwise (adaLN, norms) | ~0.8 s | ~15% | fusable |

(Buckets overlap kernel/op accounting; shares are indicative.) Two experiments against this profile, same day:
- **fp8 (torchao dynamic float8, no compile): 1.36× SLOWER** (5,008 vs 3,692 ms diffusion). Per-op quantize/rescale overhead swamps tensor-core gains at d=1536 GEMM sizes without fused epilogues. fp8 is a kernel-stack project, not a checkbox — plan accordingly.
- **torch.compile**: inductor-only works out of the box → **12.5% faster** (3,230 ms, MFU 27.6→31.5%) despite graph breaks; CUDA-graphs mode **crashes on the KV-cache mutate pattern** — the same pattern behind the 19% copy bucket.

Which yields the punchline for systems work: **a preallocated static KV ring buffer is the keystone systems artifact** — it removes the copy bucket, legalizes CUDA graphs (the stall bucket), and clears the graph breaks. But the day-5 gate changes its sequencing: don't accelerate a student whose quality path is not established. Quality repair/audit comes first; the ring buffer follows on the winning student.

## 5. The finding: the 4090 doesn't close; H100 throughput does — quality is next

The naive hope — "17 fps on H100, so fp8 + a tiny VAE + overlap gets a 4090 to 24" — is wrong, and the model shows it:

| RTX 4090, fp8, tiny VAE, overlap, short ctx | fps @30/40/45% MFU |
|---|---|
| 4-step | 13.5 / 18.0 / 20.2 |
| 3-step | 18.0 / 24.0 / 27.0 |
| **2-step** | **27.0 / 36.0 / 40.4** |
| 1-step | 53.9 / 71.9 / 80.9 |

**Measured update (day 3, rented 4090):** the fp8 branch of this story is dead on consumer Ada, and the reason is instructive. The 4090 runs this workload at **~68% effective MFU** (vs the H100's 27.6%) — a small GPU is nearly saturated by 4,680-token chunks, so there is no overhead pool for quantization to feast on. Measured: torch.compile buys only 5.3% (vs 12.5% on H100); torchao dynamic fp8, properly version-pinned and inductor-fused, is **0.90× — slower than bf16**. (LongLive's published 1.2× fp8 win is real but lives on H100-class headroom with a production quant stack.)

What survives is arithmetic nobody can quantize away: **forwards per emitted frame**. Stock Self-Forcing spends 5 forwards per 3-latent-frame chunk (4 denoise + 1 KV-repopulation). The day-3 hope was that a frame-wise 1-2-step student would push total forwards low enough to clear 24.

**Day-4 measured the actual frame-wise students (Causal-Forcing++), and it's a more interesting — and humbling — result.** The KV-repopulation forward *does not go away* in frame-wise students: framewise-2step runs **65 forwards/prompt** (3 per latent frame: 2 denoise + 1 cache), framewise-1step **45** (2 per frame). Frame-wise doesn't win by *fewer* forwards — it wins because each forward processes 1 latent frame (1,560 tokens) instead of a 3-frame chunk. Measured on the 4090, bf16, diffusion-only: 2-step **10.8 fps**, 1-step **15.2**, 1-step+compile **16.3** (0.21s latency).

Then the bottleneck moved. That reported FPS is **diffusion-only** — the VAE decode is excluded (the timer stops before it). Fold in the measured wan-VAE (22.9 ms/frame batch, ~4.6 s for 81 frames) and end-to-end collapses: 1-step+compile is **~8.5 fps e2e**, with the VAE now **~50% of the wall clock** — exactly APT1's "VAE > DiT" finding, reproduced. Even a TAEHV-class tiny decoder with perfect overlap tops out ~**15-16 fps e2e**. And the only config fast enough to flirt with the target (1-step) is the one with visible **temporal-drift smearing** (5/10 at frame 60 vs 7/10 for 2-step) — few-step students fail by error accumulation, and you see it deep in the rollout, not at frame 1.

**So: 24 fps at 480p does not close on a 4090 with a student swap, compile, and a VAE swap alone — it needs the systems work, and even then it's uncertain.** The honest headline moved to the H100 (where the SF lineage's own MotionStream already hits 29 fps with *no* systems optimization — so the interesting target becomes *beating* it with the ring buffer they skipped), with the 4090 reframed around its genuine win: **sub-second latency, interactive, on consumer hardware** — never labeled "real-time" below the 24 fps playback threshold. Real-time video generation is a **step-count + VAE + systems problem**; precision is secondary, and the VAE is no longer something you get to ignore.

### The H100 gate: 31 fps warm e2e, but not yet a quality-qualified headline

The next session ran the exact gate the consensus requested: the released CF++ 1/2-step students, full Wan VAE and TAEHV controls, a corrected rolling decoder with stream-local CUDA timing, and complete-video quality A/Bs. All rows are 81 decoded frames at 480×832 on one H100 80GB; steady numbers are means of three post-warmup trials.

| model + decoder | mode | forwards | warm e2e fps | warm first post-decode GPU-RGB event | P95 derived effective-frame interval | peak allocated / reserved |
|---|---|---:|---:|---:|---:|---:|
| CF++ 1-step + full Wan VAE | batch | 45 | 19.71 | — | — | 23.7 / 26.1 GB |
| CF++ 1-step + TAEHV | batch (not streaming) | 45 | 34.37 | estimated 0.168s | — | 36.2 / 47.1 GB |
| **CF++ 1-step + rolling TAEHV** | **streaming, separate CUDA stream** | **45** | **31.04** | **0.242s** | **37.62ms** | **23.1 / 27.3 GB** |
| CF++ 2-step + full Wan VAE | batch | 65 | 15.76 | — | — | 23.7 / 26.1 GB |
| CF++ 2-step + TAEHV | batch (not streaming) | 65 | 24.05 | estimated 0.168s | — | 36.2 / 47.1 GB |
| CF++ 2-step + rolling TAEHV | streaming, separate CUDA stream | 65 | 22.16 | 0.250s | 54.40ms | 23.1 / 27.3 GB |

The one cold/warmup 1-step trial was **24.64 fps with a 0.867s first post-decode GPU-RGB event**. So the supportable statement is narrow: **a three-trial warm mean of 31.04 fps e2e, a 0.242s warm post-decode GPU-RGB CUDA event, 480×832, 81 frames, one H100, CF++ 1-step + rolling TAEHV.** The historical runner copied pixels to CPU only after the complete rollout and never timed transport or browser presentation, so this is not a first-visible metric. The throughput is numerically 7.03% above MotionStream's published 29 fps, but not an apples-to-apples controlled superiority claim: cold is below 29, the derived P95 cadence corresponds to ~26.6 fps, and MotionStream's protocol is not ours. The 34.37-fps batch row is explicitly not streaming.

Two surprises matter more than the headline number:

1. **TAEHV is the win; overlap is not.** Separate-stream decode adds only 0.38% over serial rolling decode (31.04 vs 30.92 fps). The tiny decoder removes ~1.85 seconds of full-VAE work; after that, there is little decoder work left to hide.
2. **The released demo drops frames.** Its fixed `pixels[:, 12:]` trim is valid only after the three-latent history is full. During startup it emits 73 frames, not 81. The harness trims `prior_context_latents × 4` frames while filling (then 12) and asserts an 81-frame output. Every number above uses that correction.

Throughput, however, is only half the pre-registered gate. A blind full-video audit used three seed-matched prompts (animal anatomy, human face/hands, moving mechanics), SF-4-step and both CF++ students, and two independent passes from Gemini 3.1 Pro. Mean scores across six ratings were **CF++1 = 5.67, SF4 = 4.67, CF++2 = 4.33**. CF++1 beat SF4 on two of three prompts in both passes, but no system reached the absolute **7/10** bar and one judge family is not human validation. Decoder-only A/Bs (same latents) measured 31.88/30.80dB PSNR and were judged visually identical, so the failure belongs primarily to the student, not TAEHV.

The phase-gate consensus was therefore unanimous: **performance milestone achieved; quality-qualified headline open.** The fastest path is also the best aggregate path in this small sample, so optimizing the slower 2-step student is premature. Freeze the 1-step streaming stack, repair and audit quality on a broader prompt set, and require ≥7/10 while retaining ≥29 fps before calling the headline complete.

## 6. KV: capacity, not bandwidth

Per latent frame (bf16): 1.3B → 288 MB; 5B → **144 MB** (the deeper Wan2.2 VAE: 4× fewer tokens beats 2× wider d_model); 14B → **1.28 GB** (21-frame windows: 6.0 / 3.0 / 26.8 GB — the 14B number matching Krea's reported ~25 GB). Reading even the full 1.3B window costs ~2 ms/step against ~300 ms of compute. So KV work buys you **fit** (14B on one GPU: 27→13.4 GB with fp8 KV) and **long-horizon consistency** (what to keep: sinks, windows, recurrent state) — not step time. This is the sharpest single divergence from LLM-inference intuition.

## 7. TTFF anatomy (roofline prediction, followed by measurement)

| | text | first-chunk DiT | first-chunk VAE | cold | warm (text cached) |
|---|---|---|---|---|---|
| H100 bf16 | 280 | 168 | 36 | 484 ms | 204 ms |
| H100 fp8 | 280 | 84 | 36 | 400 ms | 120 ms |
| 4090 bf16 | 1,678 | 1,009 | 120 | 2,807 ms | 1,129 ms |
| 4090 fp8 | 1,678 | 504 | 120 | 2,302 ms | 624 ms |

Cold-start on consumer GPUs is **text-encoder-dominated** (umT5-xxl is 4× the DiT's parameter count). Sub-second TTFF on a 4090 requires caching or overlapping the prompt encode — a scheduling fix, not a model fix. Overlap never helps TTFF (the first decode waits on the first denoise).

Measured H100 decoding corrects the prior: CF++ 1-step's first post-decode GPU-RGB CUDA event is **0.242s warm mean / 0.867s cold n=1**; 2-step is **0.250s warm / 0.803s cold n=1**. The cold gap is mostly one-time text/kernel warmup. These are post-TAEHV GPU event timestamps, not the earlier pre-VAE “first chunk” timer—but they still precede D2H materialization, encoding, transport, and browser presentation.

## 8. Local-hardware rooflines (why the answer is "rent an H100")

| | fps (30/40/45% MFU) | why |
|---|---|---|
| Titan Xp (2017, 12GB) | 0.5 / 0.7 / 0.7 | Pascal: fp16 at 1/64 rate → fp32 only, no FlashAttention, and 12 GB won't hold weights + text encoder + KV |
| M4 Pro 48GB | 0.7 / 0.9 / 1.0 | ~17 nominal TFLOPs; MPS/MLX kernel quality lands below these MFUs |
| M4 Max 48GB | 1.4 / 1.9 / 2.1 | great dev box, 20× short of real-time |

The 48 GB of unified memory fits 14B fp8 weights with room to spare — memory was never the Mac's problem; compute is. (This is also why MobileWan's phone deployment is remarkable: they bought back the gap with an NPU, recurrence, pruning, and 3-step distillation simultaneously.)

## 9. Measurement contract (for Phase 1)

Every fps/TTFF number this project publishes states: hardware + precision; steps + chunk size + context policy; VAE variant; whether text encode is included (cold vs warm); decoded-frames-per-second at the output (not latent); P50/P95 per-chunk jitter over ≥60s sustained; and a quality report (VBench subset + side-by-sides vs the bf16 4-step baseline) next to every throughput claim. MFU-sensitivity (30/40/45%) accompanies every prediction.

Known model simplifications to fix with data: single MFU across ops (attention kernels < dense matmul), fp8-MFU-parity assumption (optimistic), VAE/DiT overlap contention (VAE is bandwidth-bound and fights the DiT for HBM/L2), first-latent-frame causal VAE off-by-one, CFG excluded (distilled students don't use it; teachers do).

---

*TODO before publish: run ≥60s sustained streaming; complete ≥10-prompt multi-family + human quality audit and repair to ≥7/10; add roofline log-log figure; add per-op MFU measurements; cite MobileWan / scaling book / Self-Forcing / Causal Forcing++ / APT / LongLive / MotionStream / SDV2 / Krea properly.*
