# Reverse-engineering Decart real-time video: Oasis, Mirage, Lucy 2.x, and the surrounding lineage

Research date: 2026-07-21. Open-Oasis code is pinned to commit [`f59deef2c019c212bd0c5a3a5b986a51f3701847`](https://github.com/etched-ai/open-oasis/tree/f59deef2c019c212bd0c5a3a5b986a51f3701847). This report distinguishes released code, statements in first-party technical pages, recoverable Git history, paper results, and my own inferences. That distinction matters: Decart's production inference stack and every Lucy model remain closed.

## Bottom line

Decart's real-time playbook is not one trick. It is a stack:

1. **Make time causal and emit a frame or tiny chunk immediately.** Oasis uses causal temporal attention and generates one latent frame at a time. Mirage discloses a three-output-frame recurrence. Lucy says it remains framewise and causal.
2. **Train for imperfect temporal context.** Oasis starts from Diffusion Forcing; Mirage adds corrupted-history fine-tuning; Lucy 2 names a self-output-exposure and drift-penalty variant, Smart History Augmentation; Lucy 2.5 adds a separately conditioned self-anchor.
3. **Compress diffusion work.** Mirage names shortcut distillation and pruning. The Lucy 2/2.5 pages disclose no Lucy-specific distillation family, NFE, or recipe. The closest open lineage now combines teacher-forcing consistency/ODE initialization with self-forced DMD, and One-Forcing adds a real-data latent GAN for the one-step limit.
4. **Keep context bounded.** Oasis-500M was trained on 32 frames and drops older frames from the DiT conditioning window. More recent open systems roll a fixed KV cache and often retain a first-frame sink. Decart has not disclosed Lucy's cache policy.
5. **Design around the hardware.** Axial attention, model dimensions aligned to tensor-core shapes, pruning/sparsity, low precision, fused mega-kernels, CUDA graphs/execution graphs, communication overlap, and streaming transport all matter. These optimizations buy budget; they do not themselves teach displacement.
6. **Use specialized silicon only as an economic multiplier.** Oasis already ran on an undisclosed system of NVIDIA H100s through Decart's GPU engine. Sohu was a projected transformer-only path to much larger models/resolution/concurrency, not the cause of Oasis's algorithmic breakthrough. As of this report, Etched says A0 N4P silicon for its current, unnamed inference product is in customer rack validation and production, with first racks scheduled to ship in summer 2026; the old Sohu performance projections remain unverified.

For CF++1's motion failure, the most important finding is **not** a Decart marketing technique. [One-Forcing](https://arxiv.org/html/2605.23458) tested framewise, one-step Wan2.1-1.3B and reported Dynamic Degree **52.76 with causal-ODE initialization versus 23.61 with causal-consistency-distillation initialization**, under its otherwise matched DMD+GAN recipe. The authors attribute the gap to richer motion in the multi-step ODE trajectories. They also found that a deterministic endpoint-MSE regularizer almost erased motion (Dynamic Degree 1.30). This directly shows that initialization choice can strongly suppress one-step motion even when the final objective is held fixed. Interpreting the causal-CD result as averaging uncertain displacement is plausible, but that mechanism was not measured directly.

## Evidence map

| Claim | Public evidence | Confidence / caveat |
|---|---|---|
| Oasis architecture and sampler | Released source and weights metadata | High for the 500M release; not necessarily the larger demo model |
| Oasis uses an inference KV cache | Official architecture figure labels history as “KV Cache Frames” | Production/concept claim only; released source has no attention KV cache |
| Oasis “dynamic noising” | Project prose and historical commit `abe07fa` | Exact historical implementation recovered; removed six minutes later and absent at HEAD |
| Mirage recurrence/history augmentation/shortcut distillation | First-party technical report | Mechanisms named, but losses, ratios, step counts, and cache internals absent |
| Lucy Smart History Augmentation and systems stack | First-party Lucy 2 page | High that Decart uses the named ideas; low on recipe details and isolated gains |
| Lucy 2.5 self-anchor, data recipe, quantization, sparse attention | First-party Lucy 2.5 page and current API docs | Mechanisms explicit; model architecture, losses, hardware, and isolated benchmarks absent |
| Lucy 2.5 1080p30 | Launch prose | Marketing claim; current SDK/docs configure public service at 1280x720 and 30 fps |
| Sohu speedups | Etched announcement / projections | Etched reports A0 silicon for its current, unnamed inference product and summer-2026 first-rack plans, but no transparent third-party Oasis/Lucy/Sohu benchmark or completed customer shipment was found |

## 1. Open-Oasis: the concrete goldmine

### 1.1 What was actually released

The repository is an inference-only snapshot: no training loop, optimizer, data loader, loss implementation, or authoritative config file is present. The Hugging Face repository is gated and contains duplicate `.pt` and `.safetensors` weights. Metadata reports `oasis500m.safetensors` at 2,431,808,520 bytes and `vit-l-20.safetensors` at 917,006,304 bytes; the total repository storage is about 6.73 GB because both formats are included. The constructor-derived parameter counts and artifact sizes agree with FP32 storage. The gate returned an authorization error in this environment, so I did not download 3.35 GB of duplicate-format tensors; source and metadata were sufficient for exact architecture recovery. See the [Hugging Face model](https://huggingface.co/Etched/oasis-500m/tree/main) and [API metadata](https://huggingface.co/api/models/Etched/oasis-500m?blobs=true).

The public code has several boundaries worth stating explicitly:

- fixed 360x640 input/output resolution;
- prerecorded prompt image/video plus prerecorded VPT-style actions, not a live server;
- output decoded and written only after the whole rollout;
- no `torch.compile`, custom CUDA kernels, quantization, incremental VAE decode, WebRTC, or production serving code;
- `--fps 20` sets MP4 playback metadata; it does not measure generation throughput;
- a known input-video path bug treats torchvision's THWC output as TCHW ([issue 37](https://github.com/etched-ai/open-oasis/issues/37)).

### 1.2 Exact latent and DiT geometry

The correct factory is [`dit.py:DiT_S_2`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/dit.py#L323-L332), not the defaults on the `DiT` class.

| Component | Exact released setting |
|---|---|
| RGB frame | 360x640x3 |
| Autoencoder patches | 20x20 non-overlapping patches -> 18x32 = 576 tokens/frame |
| AE latent | 16 continuous channels/token -> tensor `16x18x32` |
| DiT patching | 2x2 latent cells -> 9x16 = 144 DiT spatial tokens/frame |
| DiT width | 1024 |
| DiT depth | 16 spatiotemporal blocks |
| Attention | 16 heads, head dimension 64 |
| MLP | 4x expansion, tanh-approximate GELU |
| Context limit | 32 frames |
| Action condition | 25 dimensions, linearly projected to 1024 |
| Output | 16-channel velocity prediction |

[`dit.py:DiT.forward`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/dit.py#L292-L320) receives `x[B,T,16,18,32]` and a timestep tensor `t[B,T]`. Each temporal frame's timestep and action embedding is broadcast across that frame's 144 spatial tokens. There is no text encoder or classifier-free guidance path in this checkpoint; controls are actions only.

Each [`dit.py:SpatioTemporalDiTBlock`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/dit.py#L131-L192) performs:

1. global spatial self-attention over 144 tokens independently for every frame;
2. a spatial MLP;
3. causal temporal self-attention over up to 32 frames independently at each of the 144 spatial sites;
4. a temporal MLP.

Spatial and temporal sub-blocks have separate AdaLN-Zero modulation. Norms have no learned affine. The spatial path uses 2D RoPE and the temporal path uses 1D RoPE. Exact rearrangements and causal flags are visible in [`attention.py:SpatialAxialAttention.forward` and `TemporalAxialAttention.forward`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/attention.py#L14-L101). This axial factorization changes attention cost from full `(T*HW)^2` attention to per-frame spatial plus per-location temporal attention, while preserving global spatial interaction within each frame.

The learned DiT parameter count is **607,943,744**, plus 48 registered frozen RoPE-frequency elements. The major term is 16 blocks x 37,773,312 parameters. “500M” is therefore a nominal product label, not the literal released backbone size.

### 1.3 The transformer autoencoder, exactly

[`vae.py:AutoencoderKL`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/vae.py#L149-L225) and the [`ViT_L_20_Shallow_Encoder` factory](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/vae.py#L337-L354) instantiate a **frame-independent spatial ViT Gaussian autoencoder**, not a convolutional VAE and not a temporal VAE:

- patch embed: `Conv2d(3,1024,kernel=20,stride=20)`, producing 576 tokens;
- encoder: 6 pre-LN, global-attention ViT blocks, 16 heads, 4x MLP, 2D axial RoPE;
- posterior head: linear 1024 -> 32, interpreted as a 16-D mean and 16-D log variance for every patch token;
- inference: uses posterior mean, not a sample;
- decoder input: linear 16 -> 1024;
- decoder: 12 analogous global-attention ViT blocks;
- pixel head: linear 1024 -> 1200 = `3 * 20 * 20`, then deterministic unpatchify;
- no convolutional pyramid, ResNet blocks, skips, VQ codebook, overlapping patches, temporal attention, or temporal compression.

The AE has **229,246,160 learned parameters**. RGB-to-latent scalar compression is `360*640*3 / (18*32*16) = 75x`; temporal compression is 1x. The public stack is therefore about **837.19M learned parameters**, not 500M end to end.

[`vae.py:AutoencoderKL.forward`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/vae.py#L313-L334) returns reconstruction, posterior, and latent, but implements no KL, reconstruction, perceptual, or adversarial loss. `labels` and `split` are ignored. An external VQGAN-like harness may have existed, but the exact training loss is unpublished and cannot be inferred from `get_last_layer()` alone.

Relative to a normal convolutional VAE or CF++1's rolling TAEHV, the meaningful design choice is **not “transformers are magic.”** Oasis makes every decoded frame independent in the temporal dimension and assigns all motion modeling to the DiT. This eliminates one possible temporal-decoder smoothing pathway, at the cost of a very large global-spatial decoder; no Oasis ablation proves that such smoothing causes CF++1's morphing. Replacing TAEHV wholesale would require re-encoding data and retraining the latent generator, so the immediate value is the proposed decoder-bottleneck audit.

### 1.4 Diffusion Forcing: what is proved and what is not

The [Oasis project report](https://oasis-model.github.io/#architecture) says Diffusion Forcing independently noises each token. In this implementation, the finest exposed noise-level granularity is **one timestep per temporal latent frame**: `t[B,T]`, shared by all 144 spatial DiT tokens in a frame. The code does not support a separate timestep for each spatial patch without modification.

The public sampler proves that the checkpoint uses:

- a 1,000-step sigmoid beta schedule from -3 to 3 with `tau=1`, implemented by [`utils.py:sigmoid_beta_schedule`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/utils.py#L15-L28);
- velocity (`v`) prediction;
- independently assignable frame timesteps;
- causal temporal attention.

It does **not** reveal the Oasis training timestep distribution, loss weighting, optimizer, context-corruption mixture, or data curriculum. No official training code was released; the only training-code reply on Hugging Face points to an explicitly unofficial fork ([discussion](https://huggingface.co/Etched/oasis-500m/discussions/3)).

There is a useful but non-authoritative proxy. Open-Oasis directly derives code from the `buoyancy99/diffusion-forcing` repository. At the latest upstream commit dated before Oasis's release—not a pinned Oasis dependency—[`DiffusionForcingBase._generate_noise_levels`](https://github.com/buoyancy99/diffusion-forcing/blob/e2c4da10d3fe35105b24edbb3eaba7ba099361d7/algorithms/diffusion_forcing/df_base.py#L172-L186) samples iid integer timesteps uniformly in `[0,999]` with shape `[frames,batch]`. The upstream [video config](https://github.com/buoyancy99/diffusion-forcing/blob/e2c4da10d3fe35105b24edbb3eaba7ba099361d7/configurations/algorithm/df_video.yaml#L13-L34) uses sigmoid noise, `pred_v`, fused-SNR weighting, cumulative-SNR decay 0.96, training-noise clipping at 6, 100 sampling steps, `eta=0`, and stabilization level 15; the [diffusion implementation](https://github.com/buoyancy99/diffusion-forcing/blob/e2c4da10d3fe35105b24edbb3eaba7ba099361d7/algorithms/diffusion_forcing/models/diffusion.py#L218-L300) broadcasts independent noise per frame and trains a weighted v-MSE. Oasis shares sigmoid/v/stabilization-15 choices, so this is a rational experiment recipe, **not evidence that Oasis used every upstream hyperparameter**.

In that upstream recipe, fused-SNR combines the current token's signal-to-noise ratio with mean historical SNR as `S'_t = 1 - (1-S_t)(1-mean_history_SNR)`. This is a loss-weighting lead, not an inference noise schedule, and remains unconfirmed for Oasis.

The conceptual limitation is important for CF++1: Diffusion Forcing broadens the support of synthetic noisy contexts, but it still corrupts real frames rather than conditioning on the student's characteristic morphing errors. Self-forced rollout/history augmentation is the more direct train-inference match.

### 1.5 Exact autoregressive sampler

[`generate.py:main`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/generate.py#L23-L194) does the following for each new frame:

1. Encode the prompt frame with the AE posterior mean and multiply by scale `0.07843137255`.
2. Append a fresh Gaussian latent clipped to `[-20,20]`.
3. Condition the DiT on at most the latest 32 stored latents using `start=max(0,i+1-32)`. The full rollout remains in `x` for final decoding; old frames leave only the conditioning window.
4. With default 10 evaluations, construct timestep indices `[-1,99,199,...,999]` and traverse from 999 down to 99.
5. On every pass, run the **whole context window** through the DiT. History is presented clean but labeled with timestep 14; the current frame receives the active high-to-low timestep.
6. Convert predicted velocity to clean estimate

   `x0 = sqrt(alpha_bar_t) * x_t - sqrt(1-alpha_bar_t) * v`

   recover epsilon, and make a deterministic DDIM-like update

   `x_next = sqrt(alpha_bar_next) * x0 + sqrt(1-alpha_bar_next) * eps`.

7. Force `alpha_next=1` on the last current-frame pass, persist **only the newest frame**, and discard temporary history predictions.

The context timestep 14 corresponds to approximately `alpha_bar=0.99532` and noise standard deviation 0.0684 under this schedule, but HEAD does not q-sample that noise or even attenuate the stored self-generated context. The history is at its generated endpoint with no extra q-noise; the model only receives timestep label 14. The all-zero prepended action fixes an action/frame offset so generated frame `i` conditions on original action `i-1`; see [`utils.py:load_actions`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/utils.py#L109-L121) and the [maintainer explanation](https://github.com/etched-ai/open-oasis/issues/20#issuecomment-2465897145).

### 1.6 The KV-cache contradiction

The released model has **no attention KV cache**. [`TemporalAxialAttention.forward`](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/attention.py#L14-L55) projects Q, K, and V for all frames on every call and invokes causal SDPA over the full temporal window. The generator therefore recomputes all spatial blocks and all temporal keys/values for up to 32 frames on each of 10 evaluations per output frame. The only cached object in the repository is a RoPE frequency table, not activation K/V.

The [official dynamic-noising figure](https://oasis-model.github.io/dyno.png), however, labels prior columns “KV Cache Frames.” The most defensible conclusion is:

- the concept/production stack used or contemplated K/V caching;
- the downscaled public sampler does not release it;
- cache layout, invalidation, and interaction with changing history noise levels are unknown.

This distinction also explains why the public Python path should not be treated as the path that achieved the headline speed.

### 1.7 “Dynamic noising” was real, then removed

The [project prose](https://oasis-model.github.io/#architecture) says Oasis adds noise to history during early diffusion passes and gradually removes it in later passes: low-frequency structure can still be recovered early, while un-noised generated history restores detail late. That exact mechanism exists in Git history at commit [`abe07fa`](https://github.com/etched-ai/open-oasis/blob/abe07fa3a4f25429aaf9ab264fcce1c362f9e584/generate.py#L47-L137):

- `ctx_max_noise_idx = (N // 10) * 3`;
- for reverse pass index `j`, `ctx_idx = min(j, ctx_max_noise_idx)`;
- every context frame receives timestep `noise_range[ctx_idx]`;
- a **fresh clipped Gaussian** is drawn for every history latent on every pass;
- history is actually forward-noised:

  `x_ctx = sqrt(alpha_bar[t_ctx]) * x_clean + sqrt(1-alpha_bar[t_ctx]) * epsilon`;

- only the current frame is persisted.

At that revision `N=50`, early passes cap context near training timestep 299 and late passes reduce it toward timestep 19. This is an inference-time schedule across denoising passes, not an age-based schedule across history frames.

Six minutes later, commit [`3b3f6e9`](https://github.com/etched-ai/open-oasis/commit/3b3f6e9e8384e02e1921d35e833250dc9e3aa590) removed physical q-noising and replaced it with constant `t_ctx=14`; HEAD retains that version. Oasis author Julian Quevedo subsequently said the 500M checkpoint was trained on 32-frame sequences, discards anything older than 32, and that after schedule tests a constant context noise level worked best for fully denoising one new frame at a time ([issue 19](https://github.com/etched-ai/open-oasis/issues/19#issuecomment-2465885874)).

Therefore:

- **advertised/production idea:** noisy history early, un-noised generated history late;
- **recoverable experimental implementation:** exact capped q-noise schedule above;
- **released 500M HEAD:** stored self-generated history with no added q-noise, but with a constant small-noise timestep tag;
- **CF++1 implication:** a one-step model has no early/late pass axis, so literal dynamic noising cannot be ported without a two-pass diagnostic. A single-pass context-noise sweep is an analogy, not the Oasis algorithm.

### 1.8 What likely delivered Oasis's 20 fps

The [Decart Oasis report](https://decart.ai/publications/oasis-interactive-ai-video-game-model), [project report](https://oasis-model.github.io/), and archived [Etched write-up](https://web.archive.org/web/20241104055320/https://www.etched.com/blog-posts/oasis) name the full recipe: a downscaled DiT, axial causal attention, transformer AE, Diffusion Forcing, dynamic context noising, and heavily optimized GPU kernels/communication. Decart reports roughly 47 ms/frame and 150 ms/training iteration in its stack and describes GPU-primitive/kernel acceleration plus NVLink, PCIe 5, and NUMA-aware communication; neither timing has a complete public workload specification. The result used an undisclosed number/topology of NVIDIA H100s plus the proprietary GPU engine; no Sohu was required.

The public repo's `--fps=20` is not proof that the plain script runs at 20 generated frames/s. The live demo was also described as a larger model than the local 500M release. Treat “~20 fps at 360p on H100s” as a first-party system result with undisclosed accelerator count/topology, not a single-H100 benchmark or a reproducible result from `generate.py`.

## 2. From Oasis to Mirage to Lucy

### 2.1 Technique timeline

| Date | System | New disclosed ingredient |
|---|---|---|
| 2024-10-31 | Oasis | framewise DiT + spatial ViT AE, Diffusion Forcing, dynamic history noise, proprietary inference stack |
| 2025-07-17 | MirageLSD | open-domain live editing, explicit 3-frame generated-history recurrence, corrupted-history augmentation, shortcut distillation, pruning, Hopper mega-kernels |
| 2026-01-26 | Lucy 2 | pure-diffusion autoregressive framewise streaming, Smart History Augmentation, cycle-level architecture selection, mega-kernels, custom WebRTC |
| 2026-07-16 | Lucy 2.5 | self-anchoring, large propagated-edit corpus, coarse-to-fine prompts, MXFP8/NVFP4, dynamic sparse attention, deeper fusion |

### 2.2 MirageLSD: the bridge with the clearest recurrence

The first-party [MirageLSD report](https://decart.ai/publications/mirage) is more concrete than Lucy. At time `i+1`, the model consumes generated outputs `(F_{i-2},F_{i-1},F_i)`, the current incoming source frame `I_{i+1}`, and prompt `P`, emits `F_{i+1}`, then immediately feeds it back. This is an explicit finite recurrence, but not evidence of a KV-ring implementation.

Its three algorithmic disclosures are:

1. **Diffusion Forcing:** independently noise frames during pretraining.
2. **History augmentation:** fine-tune on teacher-forced history frames corrupted with artifacts that resemble model mistakes, and indicate that history may be inaccurate. The page does not disclose corruption distributions, the indicator, clean/corrupt mixing ratio, or recovery loss.
3. **Shortcut distillation:** train a smaller student to match a larger teacher's denoising trajectory with fewer evaluations. The page cites Shortcut Models but does not disclose NFE, teacher/student sizes, target parameterization, or an isolated ablation. It is not justified to relabel this DMD2.

For real-time execution, Mirage names Hopper-oriented mega-kernels, GPU communication hidden under compute, architecture-aware parameter sizing, pruning, and hardware sparse support. Decart reports both <40 ms response and 24 fps, although 24 fps is a 41.7 ms frame interval; treat response and cadence as separately defined or rounded first-party measures rather than deriving one from the other. The report explicitly discusses the difficulty of batch-1 latency. Remaining limitations include a short past window and weak precise object, spatial, and motion control—highly relevant evidence that history stability alone does not solve displacement.

### 2.3 Lucy 2: what Smart History Augmentation means

The [Lucy 2 report](https://decart.ai/publications/lucy-2-introducing-sota-video-generation-in-realtime) describes a pure diffusion system generating causally, frame by frame. Its central temporal claim is **Smart History Augmentation (SHA)**: expose the model during training to its own imperfect outputs and explicitly penalize quality drift so it learns to recover, reportedly maintaining streams for hours.

SHA is not a cache eviction rule. It changes the training distribution and objective. A recent-clean KV ring can ensure the model sees fresh context, but it does not teach the model what to do when that context contains its own morphs. No public Lucy source discloses:

- whether rollouts are online, replayed, or synthetically corrupted;
- rollout horizon or clean/self-history mixture;
- drift penalty or confidence indicator;
- model dimensions, temporal mask, cache representation, NFE, or parameter count;
- an isolated SHA ablation.

Lucy 2's systems disclosures are unusually consistent with the earlier stack: latency-focused autoregressive framewise execution; mega-kernels that keep activations close to tensor cores and reduce HBM traffic; model dimensions chosen with cycle-level microbenchmarks; and a custom WebRTC transport. The page's 1080p30 and near-zero-latency language is a first-party end-to-end claim, not a single-H100 benchmark. A public January 2026 changelog instead recorded an early Lucy 2 preview at 720p20 ([Decart changelog](https://docs.platform.decart.ai/changelog)).

### 2.4 Lucy 2.5: self-anchor and training-data scale

The [Lucy 2.5 report](https://decart.ai/publications/lucy-2-5-raising-the-bar-for-live-ai) adds two model/data ideas:

**Synthetic edit propagation.** Decart starts with high-quality image-to-image transformations, uses specialized internal models to propagate those edits across complete video sequences, aligns prompts to general scene descriptions, and supplies a coarse-to-fine prompt hierarchy (simple edit, then position/style/color/size). This manufactures temporally consistent edit supervision. Decart attributes replacement, contact, shadow, VFX, restyling, removal, and revealed-background capabilities to the resulting system, but does not disclose known object trajectories or displacement labels. The corpus size, filtering, propagation model, losses, and motion distribution are also undisclosed.

**Self-Anchoring.** Shortly into a stream, Lucy snapshots its own edited output and adopts it as an explicit new reference; training uses the same self-anchored condition. This anchor is therefore a separate conditioning signal, not merely a retained KV. Current [Lucy 2.5 API docs](https://docs.platform.decart.ai/models/realtime/lucy-2.5) make the mechanism operationally clear: recent generated frames are fed back as a reference image, the feature is enabled by default, and disabling it requires reconnecting. The docs warn that a stale anchor is harmful after a camera cut, different person, or large scene change. Self-anchor should therefore have a cut/reset policy and must be tested for motion freezing.

Lucy 2.5 also names:

- custom MXFP8 and NVFP4 inference, claiming up to 4x on compute-bound operations;
- dynamic sparse attention that identifies redundant attention work per call and prunes it;
- deeper kernel fusion and improved mega-kernels.

There are no public sparsity rules, accuracy curves, kernel sources, target GPUs, or per-optimization ablations.

### 2.5 Resolution and speed claims need calibration

The Lucy 2.5 launch says 1080p30. Current public documentation says 720p, and immutable SDK source at commit [`05f45d3`](https://github.com/DecartAI/sdk/blob/05f45d39b8b6263aecb25ed81245b475a26a0e72/packages/sdk/src/shared/model.ts#L337-L355) configures `lucy-2.1` at 1088x624/30 and `lucy-2.5` at **1280x720/30**. The same commit identifies package version 0.1.14 ([manifest](https://github.com/DecartAI/sdk/blob/05f45d39b8b6263aecb25ed81245b475a26a0e72/packages/sdk/package.json#L1-L5)). SDK capture resolution does not prove the internal latent or render resolution, but it is the strongest public evidence for service I/O.

Decart's current Lucy landing page also places <40 ms, 100 fps, 30x, 4x, and a FAQ saying 22 fps/under 200 ms on the same page, without a common resolution, device, model version, or baseline ([Lucy page](https://decart.ai/lucy)). These numbers should not be combined. There is no public demonstration that a closed Lucy 2.x checkpoint produces 1080p30 on one H100.

## 3. Adjacent open literature: what actually differs from CF++1

### 3.1 Diffusion Forcing and Self Forcing

[Diffusion Forcing](https://arxiv.org/abs/2407.01392) assigns independent noise levels along a sequence and trains a causal denoiser across arbitrary observation/generation patterns. Oasis uses its framewise video specialization. For CF++1, the new part would be a wider distribution of history noise/corruption; it still does not reproduce student artifacts. Vanilla Diffusion Forcing is **not** a justified replacement for CF++ Stage 1: [Causal Forcing](https://arxiv.org/html/2602.02214) reports collapse when training sees heavily noised prefixes while inference conditions on clean generated prefixes, with pathological motion also inflating Dynamic Degree. A small history-noise test should therefore be trained and inferred symmetrically and treated as a diagnostic.

[Self Forcing](https://arxiv.org/html/2506.08009) closes that remaining gap: during training it performs the same autoregressive rollout and KV-cache updates used at inference, conditions each new frame/chunk on self-generated history, detaches prior cached states, backpropagates through a stochastically selected final denoising step, and applies a holistic video-level DMD, SiD, or GAN objective. Its rolling KV cache evicts oldest frames without recomputation. The authors additionally train with local attention that hides the special first image latent when predicting the final training chunk, so eviction at inference is not out of distribution. On one H100, the paper reports 17 fps/0.69 s for 3-latent-frame, 4-step chunks and 8.9 fps/0.45 s for framewise generation—not the target 29 fps, but the on-policy recipe is directly relevant.

CF++ already contains the broad `teacher-forcing causal initialization -> self-forced asymmetric DMD` lineage. Simply “use Self Forcing” is not a new experiment. The meaningful deltas are the type of initialization, the critic/loss, the history distribution, and how the exact cache is simulated during training.

### 3.2 Causal Forcing and CF++

[Causal Forcing](https://arxiv.org/html/2602.02214) shows why ODE initialization from a bidirectional teacher is structurally wrong for a causal student: future-dependent teacher trajectories violate frame-level injectivity. It first trains a causal AR teacher, distills that teacher's causal ODE trajectories, then self-forces with DMD. On a 100-prompt motion-heavy set, its chunkwise result reports Dynamic Degree 68 versus 57 for Self Forcing under the same nominal throughput.

[Causal Forcing++](https://arxiv.org/html/2605.15141) replaces offline full causal-ODE trajectories with teacher-forcing causal consistency distillation: one online AR-teacher ODE step between adjacent timesteps, followed by the same self-forced DMD stage. This is much cheaper and was strongest in its reported 1-2-step initialization comparisons. It is the current CF++1 baseline, not a new lead.

The important newer caveat is that **best two-step initialization does not guarantee best one-step dynamics**.

### 3.3 One-Forcing: the closest match and strongest motion evidence

[One-Forcing](https://arxiv.org/html/2605.23458) uses the exact family of interest: Wan2.1-1.3B, 832x480, one latent frame per AR block, one NFE for later blocks, and a four-step first-block warm-up. It argues that Wan video ODE trajectories have a sharp high-noise curvature concentration. A one-step consistency model must jump across that bend, producing weak dynamics; DMD avoids path following but tends to blur.

Its fix is a real-data **noised-latent GAN added to DMD**:

- reuse the trainable fake-score Wan DiT as a joint denoiser/discriminator;
- add learned register queries and lightweight attention heads to selected transformer layers;
- classify independently noised real video latents versus current one-step self-rollouts;
- train a non-saturating logistic GAN branch, no decoded-frame discriminator;
- one fake-score/critic update every iteration, one generator update every five iterations;
- ground “real” in the dataset, not a multi-step student's generated output.

The [released code](https://github.com/Aurora-edu/One-Forcing) uses layers 21 and 29, two registers, feature dimension 1536, FFN dimension 2048, 12 heads, `lambda_G=lambda_D=0.03`, no R1/R2, generator and critic learning rate `1e-5`, EMA 0.99, and converges the reported framewise variant in 200 iterations on 8 H100s. Weights/data are on [Hugging Face](https://huggingface.co/JiaqiFeng/OneForcing).

Reported results:

- VBench total 83.76, quality 85.22, Dynamic Degree 52.76;
- one-step Self Forcing total 77.18;
- 88.4% human preference over one-step Self Forcing;
- ODE causal initialization vs CF++ causal-CD initialization under the same One-Forcing objective: total 83.76 vs 82.36, **Dynamic Degree 52.76 vs 23.61**;
- adding an endpoint-MSE “forward KL” surrogate with weight 1 drops total to 74.83 and Dynamic Degree to **1.30**.

[Adversarial Self-Distillation (ASD)](https://arxiv.org/abs/2511.01419) is a useful negative comparison: it classifies an `n`-step student against its own `n+1`-step output rather than against real data. One-Forcing reports that ASD's real/fake logit gap stays near zero, whereas its actual-real-video discriminator maintains a large, changing gap. This supports preferring the real-data critic for a usable adversarial signal; it does **not** isolate how much of One-Forcing's motion gain comes from the GAN rather than its ODE initialization.

This does not prove every CF++ checkpoint will behave identically, and the ODE initialization is more expensive. It is nevertheless the most direct published evidence that CF++-style causal-CD initialization can suppress one-step motion. Do not add naive clean-endpoint MSE as a motion fix.

### 3.4 Causal-rCM

[Causal-rCM](https://arxiv.org/html/2606.25473) is a newer open recipe that frames teacher-forcing consistency as mode-covering/offline initialization and self-forcing DMD as mode-seeking/on-policy refinement. It adds RF-native continuous-time sCM/MeanFlow under causal masks using a custom FlashAttention-2 JVP kernel, claiming similar quality in 1-2k iterations versus roughly 10k for discrete CM. Its staged Wan2.1-1.3B recipe reports VBench-T2V 84.63 with one denoising step; clean cache population makes that effectively two transformer evaluations per frame.

Two narrower leads matter here:

- **Noisy context:** reuse the KV from the last denoising pass instead of running an extra clean-context encoding pass. Residual noise acts as a low-pass filter and reduces an `N`-step model's effective cache cost from `N+1` to `N` forwards. However, its ablation says framewise one-step clean context beats framewise two-step noisy context; the latter helps three-frame chunks more.
- **Custom first-chunk schedule:** spend extra steps to establish the initial scene, then use one step for streaming. One-Forcing independently uses four steps only for the first AR block.

Causal-rCM is primarily a training-efficiency/quality upgrade, not direct proof of more displacement. Its own report says longer framewise SF-DMD training can induce directional camera drift.

### 3.5 Score Gradient Matching and phased DMD

[Score Gradient Matching Distillation (SGMD)](https://arxiv.org/html/2605.30116) is not a Decart technique, but it directly targets DMD's motion suppression. It replaces the usual score-difference generator update with a Fisher-divergence/score-gradient-matching game between generator and fake score. On a non-causal, four-step Wan2.1-14B setting, the paper reports optical-flow magnitude 9.29 versus 4.51 for DMD2 and Dynamic Degree 93.06 versus 80.56, with a modest quality/semantic tradeoff; `lambda=0.1` is its reported balance and 0.2 produces more motion with mild blur. Transfer to a framewise, one-step 1.3B student is unproven, and the paper's claimed LightX2V implementation was not discoverable in the current repository, so this is a higher-risk but unusually motion-specific Stage-3 objective test.

[Phased DMD](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Phased_DMD_Few-step_Distribution_Matching_Distillation_via_Score_Matching_within_CVPR_2026_paper.html) argues that stochastic gradient truncation can make **multi-step** DMD degenerate toward a one-step solution, reducing diversity and motion. It partitions the noise trajectory into SNR phases and matches distributions inside each phase; its reported optical-flow magnitude is 9.30 versus 3.23 for DMD2. It does not apply to a strict one-step model, but it is relevant if systems headroom permits a two-step tier. Using it to motivate an intermediate-checkpoint sweep for strict one-step Stage 3 is a low-cost inference, not a result demonstrated by this paper.

### 3.6 MotionStream

[MotionStream](https://arxiv.org/html/2511.01266) supplies three concrete deltas:

1. **Explicit motion condition.** A lightweight sinusoidal track embedding is concatenated channelwise instead of using a full ControlNet. Joint text-motion guidance is distilled into the student. This is control, not merely a training loss, but it demonstrates that a low-cost trajectory signal can prevent appearance-only solutions.
2. **Train the actual cache policy.** Self-rollout uses the same rolling KV, local window, RoPE assignment, and fixed initial-input-image attention sink at train and test. Cached rolling keys are stored pre-RoPE and re-indexed by cache position. Previous KVs are detached.
3. **Tiny streaming VAE.** A smaller decoder regresses the original VAE latent space with adversarial and LPIPS losses. On Wan2.1 it changes reported throughput from 16.7 to 29.5 fps and latency from 0.69 to 0.39 s on one H100, with LPIPS 0.360 -> 0.365.

The 29.5-fps configuration is not one-step framewise: the primary model uses **three latent frames per chunk and three denoising steps**. Its sparse-attention ablation found one sink chunk plus one recent-window chunk best; larger historical windows degraded results because old errors remained in context. The sink also makes complete scene changes harder. This supports testing a sink, not permanently hard-coding one.

### 3.7 StreamAvatar

[StreamAvatar](https://arxiv.org/html/2512.22065) adds:

- **Reference Sink:** never evict the reference-frame KV; retaining the first generated chunk too further improves identity.
- **Reference-Anchored Positional Re-encoding (RAPR):** cache keys before RoPE, cap the current frame's distance to the reference at a training-supported maximum, shift other cached-key positions coherently, then apply RoPE. This prevents unseen large indices and long-distance decay toward the reference.
- **Consistency-Aware Discriminator:** teacher-backbone features plus per-frame Q-Former queries; a local realism branch emits per-frame logits and a global branch cross-attends reference features to all future-frame features. Training uses real video, relativistic adversarial loss, and R1/R2.

The reported configuration uses a four-frame reference sink, six-frame rolling cache, and RAPR cap `D=9`. It also omits the extra clean-KV update and conditions the next chunk on the last denoising pass's slightly noisy KV, saving one forward. Its system is a 3-step avatar specialist at 25 fps using **two H800 GPUs**, with DiT first-frame delay 0.33 s, VAE delay 0.39 s, and about 1.20 s end-to-end latency after input buffering. It is not evidence for single-H100 general video. RAPR and the discriminator architecture are relevant only if long-horizon identity/position encoding, not immediate displacement, is the failure.

### 3.8 LiveEdit

[LiveEdit](https://arxiv.org/html/2606.26740) transfers a bidirectional Wan2.1-1.3B editor through 9k steps of foundation editing fine-tuning, 20k steps of chunkwise causal teacher forcing, then 10k steps of four-step DMD on eight A100s. It emits three latent frames per chunk. Its **AR-oriented mask cache** derives a source-vs-edit latent mask and prunes/reuses self-attention features for 70% of spatial tokens judged unchanged; caching FFN features fails badly in its ablation. This is specific to video editing because a source frame defines “unchanged,” and even self-attention caching slightly lowers its reported Dynamic Degree and imaging quality. The 12.66-fps system is therefore not a drop-in T2V motion fix despite loose “frame-by-frame/real-time” wording. The useful conceptual lead is spatial compute gating only after a reliable motion/edit mask exists.

### 3.9 DMD and DMD2

[Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828) minimizes an approximate reverse KL using the difference between a frozen real score and a trainable fake score on noised generated samples. Original DMD additionally used paired teacher-ODE LPIPS regression for stability and mode coverage. [DMD2](https://arxiv.org/html/2405.14867) removes that regression by updating the fake score five times per generator update, adds a real-data noised GAN branch to the fake-score bottleneck, and simulates the student's actual multi-step inputs during training. The direct video descendants are Self Forcing and One-Forcing. For CF++1, the key distinction is distribution-level supervision versus deterministic endpoint regression: One-Forcing directly shows that the latter can erase motion in the causal one-step setting.

### 3.10 Training-free and head-aware cache selection

[Future Forcing](https://arxiv.org/html/2605.30083) observes that pre-RoPE query distributions remain approximately stable across an AR rollout. It estimates a future-query proxy from historical queries, scores cache tokens against that proxy, and merges redundant token pairs in the proxy-induced affine subspace. The paper reports up to +1.49 VBench-Long subject consistency at a fixed cache budget. It also notes that Dynamic Degree can fall when the baseline's larger score came from abrupt, undesirable scene changes, reinforcing the need to score coherent foreground displacement rather than metric magnitude alone. This is a training-free alternative to “keep only the newest,” but its evidence is long-horizon identity rather than local motion.

[Forcing-KV](https://arxiv.org/html/2605.09681) profiles attention heads once and separates **static heads**, which depend on the current chunk and latest transition-anchor frame for intra-frame fidelity and chunk continuity, from **dynamic heads**, which use longer inter-frame correspondences for motion and subject consistency. It keeps the latest transition frame for static heads and prunes dynamic-head history by segment-wise adjacent-frame similarity. Its ablations show that deleting or misclassifying dynamic-head cache lowers Dynamic Degree, while removing the transition anchor sharply increases chunk discontinuity. The policy is training-free and directly challenges a uniform recent-ring policy. Its >29-fps and up-to-1.5x speed claims are DiT-only measurements on one H200, not end-to-end one-H100 evidence.

[Pyramid Forcing](https://arxiv.org/html/2605.13111) goes further: a 32-prompt, 15-second offline calibration uses sign-rate statistics and FFT periodicity to classify Anchor, Wave, and Veil heads, then assigns strided long-range retention, period-aligned sampling, or a compact local merge. Shared sink/recent frames and ragged-cache attention keep the result efficient. On Causal Forcing, it reports 60-second Dynamic Degree 57.03 -> 86.39 and total VBench-Long 79.14 -> 79.92; on Self Forcing, total rises 77.87 -> 81.21. These are long-horizon H200 results and Dynamic Degree remains artifact-sensitive, so they motivate a same-budget cache diagnostic below the early-motion objective work rather than a primary morphing cure.

## 4. Hardware and systems: adoptable versus non-adoptable

### 4.1 What is algorithmic and portable to one H100

| Technique | Portability | Motion relevance |
|---|---|---|
| causal frame/chunk generation | Already in CF++1 | prerequisite |
| teacher/self-forcing and corrupted-history training | yes | high: teaches recovery from its own morphs |
| causal ODE/CD initialization choice | yes, training-cost tradeoff | very high based on One-Forcing |
| DMD + real-data latent GAN | yes | high for realism/sharpness; GAN-only displacement gain is unisolated |
| bounded/head-aware KV + sink/RoPE re-indexing | yes | medium; stability/identity more than early displacement |
| first-block multi-step warm-up | yes, negligible steady-state cost | medium; better initial cache/scene |
| frame-independent AE diagnostic / compact causal decoder | yes, requires training/integration | separate motion-state diagnostic / high speed relevance |
| explicit track/flow conditioning or supervision | yes | high but changes product/training data |
| FP8 | H100 supports FP8 | indirect: frees budget for better sampler/context |
| fused attention/MLP, CUDA graphs, overlap | yes, engineering-heavy | indirect |
| dynamic sparsity | conceptually yes; kernel/accuracy work required | indirect and can harm moving tokens |

### 4.2 GPU stack versus specialized hardware

The proprietary GPU stack is a real part of Decart's result. Oasis mentions optimized primitives/kernels and topology-aware communication. Mirage names Hopper mega-kernels, pruning/sparsity, and communication fused into compute. Lucy says dimensions are cycle-profiled, HBM transactions and launches are fused away, and WebRTC is customized. A GTC 2026 Lucy Restyle session describes an optimization path from <1 fps through TensorRT execution graphs/kernel collapse, FP32 -> FP16 -> FP8 -> FP4, and a three-stage multi-GPU pipeline, ultimately demonstrating 1280x704 at 24 fps on NVIDIA B40/GeForce NOW—not Lucy 2 on one H100 ([NVIDIA session](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-s81596/)).

H100 has native Transformer Engine FP8 support ([H100](https://www.nvidia.com/en-us/data-center/h100/), [Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)). Lucy 2.5's native MXFP8/NVFP4 path is Blackwell-generation functionality and cannot be copied literally to H100; H100 experimentation should stop at FP8 or an emulated lower-precision scheme whose real speedup is measured. Kernel fusion and CUDA/TensorRT graphs are portable in principle but must be implemented for the actual Wan attention/cache shapes.

### 4.3 What Sohu buys—and what it does not

Etched's archived [Sohu announcement](https://web.archive.org/web/20240625230953/https://www.etched.com/announcing-etched) describes a transformer-only 4 nm ASIC with a fixed transformer data path and projected >90% FLOP utilization. It projected eight Sohu chips at >500k Llama-3-70B tokens/s and one server replacing 160 H100s. The benchmark caption specifies FP8, no sparsity, 8-way model parallelism, 2,048-token input and 128-token output; H100 was calculated with TensorRT-LLM 0.10.0.8, while the 8xGB200 figures were estimated. These were vendor projections rather than video measurements, and the throughput argument based on continuous batching does not prove batch-1 video latency. Later [Oasis project](https://oasis-model.github.io/) and archived [Etched Oasis](https://web.archive.org/web/20241104055320/https://www.etched.com/blog-posts/oasis) writeups separately projected Sohu at 100B+, 4K, and >10x user concurrency. The old Sohu announcement explicitly excludes CNNs, RNNs/LSTMs, DLRMs, AlphaFold2, and Stable Diffusion 2. Oasis's transformer DiT **and transformer AE** are unusually compatible with that constraint.

The current [Etched site](https://www.etched.com/) says its A0 silicon returned from TSMC N4P in early 2026, customer rack validation is under way, production has begun, and first racks are scheduled to ship in summer 2026. The current page does not name this product Sohu. As of 2026-07-21, I found no first-party evidence that customer shipment had occurred or racks were generally available, and no transparent third-party Sohu benchmark for Oasis or Lucy. [Chipstrat's Oasis analysis](https://www.chipstrat.com/p/etcheds-oasis-creating-a-market-for) correctly frames H100 as the demonstrated feasibility path and Sohu as the economic scaling thesis.

If Etched's projections validate, specialized hardware would buy Decart lower transformer inference cost, higher throughput/concurrency, and headroom for a larger model/resolution. It does **not** explain the causal factorization, Diffusion Forcing, history augmentation, one-step/few-step distillation, or temporal recovery. Those are the parts CF++1 can adopt.

AWS separately quotes Decart using Trainium for up to 4x frame throughput, 2x cost efficiency, and 40 -> 10 ms in an unspecified comparison ([AWS customer page](https://aws.amazon.com/ai/machine-learning/trainium/customers/)). With no model, baseline, or configuration, this is not an H100-portable technical result.

## 5. What the evidence says about “morph instead of displace”

The likely failure chain, ordered by evidence strength, is:

1. **One-step trajectory compression can suppress motion.** One-Forcing's matched one-step ablation directly shows a large Dynamic Degree gap between causal-ODE and causal-CD initialization. Its high-noise curvature analysis offers a mechanism: a single consistency jump crosses the most nonlinear part of the Wan video path. Deterministic endpoint MSE makes this worse. The paper does not directly establish that causal CD averages futures.
2. **The training context does not contain the student's characteristic errors.** A recent clean ring changes which KVs survive, but not the distribution of errors seen during training. Mirage history augmentation, Lucy SHA, Self Forcing, MotionStream, and Causal-rCM all converge on training under corrupted or self-generated context.
3. **A temporal decoder may be damping spatial transport.** Oasis delegates all time to the DiT; MotionStream reports the VAE as the main speed bottleneck and trains a dedicated streaming decoder. A zero-training decode audit can determine whether rolling TAEHV itself reduces measured displacement.
4. **The model lacks an explicit distribution-level realism signal.** One-Forcing's real-data noised-latent critic and StreamAvatar's reference-specific local/global discriminator may reject morph artifacts, but neither is trained with a dedicated “morph” label. VBench Dynamic Degree alone can reward flicker, so evaluation must also check track survival and shape consistency.
5. **History selection is a secondary stabilizer.** The current recent-clean-block KV-ring A/B is sensible and resembles released Oasis's bounded, self-generated endpoint history with no extra q-noise. It may reduce drift, but a ring cannot manufacture a missing motion prior. Attention sinks and self-anchors can even freeze motion if over-weighted.

## 6. Unknowns that should remain unknown

No public evidence establishes any of the following:

- Lucy 2/2.5 parameter count, layer count, hidden size, attention layout, latent codec, diffusion step count, cache layout, or loss equations;
- that Lucy uses the Open-Oasis architecture or its 32-frame window;
- that Smart History Augmentation is a KV ring;
- that Lucy 2 uses any particular distillation family or NFE—the official Lucy pages name none;
- that Oasis-500M training used every hyperparameter in the upstream Diffusion Forcing repository;
- that the larger Oasis demo used the released weights;
- that Open-Oasis HEAD implements physical dynamic noising or a KV cache;
- that Lucy 2.5's 1080p30 claim is a single-H100 result;
- that Etched's unnamed 2026 A0 rack product is unchanged Sohu, that customer shipment has occurred, or that it met the projected Oasis/LLM performance.

These gaps are not incidental. They prevent a faithful Lucy reproduction. The productive strategy is to test each public mechanism independently against CF++1's measured failure, as laid out in `EXPERIMENTS.md`.
