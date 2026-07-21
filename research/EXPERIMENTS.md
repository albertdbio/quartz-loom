# CF++1 experiment queue: from morphing to displacement at >=29 fps

This is a prioritized research plan, not an instruction to touch the running model. Priority is based on expected motion-quality impact multiplied by ease/low cost. “KV relation” explicitly says whether the current **recent-clean-block KV-ring** A/B already covers the idea.

## Executive recommendation

Run the no-training root-cause audit first, then port **One-Forcing's real-data noised-latent critic onto the current CF++ causal-CD checkpoint**. In parallel, prepare a controlled causal-ODE-init branch. The closest published one-step/Wan-1.3B evidence says causal-CD initialization can preserve overall VBench while cutting Dynamic Degree by more than half. The combined One-Forcing DMD+GAN recipe closes much of the one-step quality gap, but the paper does not isolate the critic's displacement gain from the initialization effect. Treat the critic as the closest low-cost realism port and the KV ring as a drift/context experiment, not a proven primary motion cure.

Do **not** add a deterministic clean-endpoint MSE to the one-step generator as a first response. One-Forcing's direct ablation drove Dynamic Degree from 52.76 to 1.30. That experiment did not test an optical-flow-map or correspondence loss. If adding motion supervision later, supervise correspondence or a distribution/critic—not a single averaged future endpoint.

## Shared evaluation protocol

A fast model can increase RAFT flow by flickering, melting, or moving the camera. Every A/B therefore needs both motion magnitude and object integrity.

### Fixed paired suite

- **Smoke:** 12 prompts x 2 seeds x 4 seconds; run after every 25-50 training updates.
- **Main:** 64 prompts x 4 seeds x 8 seconds = 256 paired clips. Use 16 rigid-object translation, 16 articulated locomotion, 16 camera/parallax, 8 occlusion/re-entry, and 8 multi-object crossing prompts.
- **Long:** the hardest 24 prompts x 2 seeds x 60 seconds. Use only for cache/history claims.
- Freeze prompts, seeds, first-frame inputs, sampler settings, decoder settings, and evaluation code before looking at results. Use the same H100 clocks, power limit, compilation warm-up, and batch size for throughput.

### Report these together

1. **Blind human motion score (0-10):** “Does the object preserve its identity/shape while changing position according to the prompt?” Keep the existing >=7/10 target. Also record pairwise preference.
2. **CoTracker or equivalent:** median track survival, foreground net displacement, trajectory smoothness, and fraction of points that move coherently with the object.
3. **Camera-compensated optical flow:** subtract a background homography/global camera flow before measuring foreground displacement.
4. **Morph guardrail:** flow-warped mask IoU, object area/aspect-ratio variation, RAFT warp residual, and DINO/feature identity across tracked crops.
5. **Standard quality:** VBench Dynamic Degree **plus** temporal smoothness, subject consistency, quality, semantic score, and a small human artifact review. Never accept Dynamic Degree alone.
6. **Systems:** steady-state model+decoder fps, p50/p95 frame time, first-frame delay, VRAM, and cache memory. The deployment gate remains >=29 fps at 480x832 on one H100; record first-frame warm-up separately.

A useful promotion gate is: paired human motion improves at least 0.7/10 (or the 95% paired interval is clearly positive), tracked displacement rises without worse track survival/warp residual, VBench quality drops <1 point, and the steady-state speed target is retained. This is a decision rule, not a literature claim.

## Priority table

Impact and ease are 1-5; priority is `impact x ease`. Conditional items are promoted only if their diagnostic gate fires.

| Rank | Experiment | Motion impact | Ease | Priority | KV relation |
|---:|---|---:|---:|---:|---|
| P0 | Stage x decoder x checkpoint collapse audit | diagnostic 5 | 5 | must run | Independent; also verifies ring parity |
| 1 | One-Forcing real-data noised-latent GAN | 3-4; displacement effect unisolated | 3 | 9-12 | **Genuinely new** |
| 2 | Train Stage 3 with the exact deployed ring + SHA-inspired failure replay | 4 | 3 | 12 | **Partly covered**: current A/B is inference selection only unless train-time policy already matches |
| 3 | Causal-ODE initialization vs current causal-CD initialization | 5 | 2 | 10 | **Genuinely new** |
| 4 | SGMD Stage-3 updater pilot | 4 | 2 | 8 | **Genuinely new** |
| 5 | Four-step first-block enhancement, if absent | 3 | 5 | 15, but likely already baseline | Independent; standard CF++/One-Forcing recipe |
| 6 | First-block sink + recent ring, trained symmetrically | 2.5 | 4 | 10 | **New extension** to the covered recent-ring family |
| 7 | Decoder intervention, only if P0 implicates rolling TAEHV | 4 | 2-3 | conditional | Independent |
| 8 | Oasis constant context tag / small symmetric context-noise sweep | 2 | 4 | 8 | **Genuinely new**, but weaker evidence |
| 9 | Motion-rich propagated data or explicit tracks/correspondence | 4 | 1-2 | 4-8 | Independent |
| 10 | Buy a second NFE with H100 FP8/fusion | 4 | 1 | 4 | Independent |
| 11 | TF-sCM/Causal-rCM initialization | 2.5 | 1 | 2.5 | Independent |
| 12 | RAPR, Future/Forcing-KV/Pyramid Forcing, spatial mask caching | 1-2 | 2-4 | long-horizon/speed only | Adjacent to ring; not first-line motion work |

The numerical order deliberately places a zero-cost baseline check before nominally high-scoring changes. Experiment 5 also scores highly on ease, but it is probably already enabled in a faithful CF++1 configuration and should be a verification, not a new training program.

## P0. Locate where displacement disappears

### Technique/source

This audit is motivated by three independent observations: [One-Forcing](https://arxiv.org/html/2605.23458) sees a large motion drop with causal-CD initialization; Oasis's AE is frame-independent; and [MotionStream](https://arxiv.org/html/2511.01266) finds the VAE dominates streaming runtime and trains a replacement.

### Exact factorial

Generate the fixed smoke/main prompts from every already-available checkpoint:

- `G0`: many-step Wan teacher or causal AR teacher;
- `G1`: post-Stage-2 causal-CD checkpoint, before asymmetric DMD;
- `G2...Gn`: stored Stage-3 checkpoints, ideally every 50 generator updates;
- `Gfinal`: current CF++1.

Decode the same generated latent sequences through:

- `D0`: full Wan VAE offline reference;
- `D1`: rolling TAEHV exactly as deployed;
- `D2`: TAEHV state reset before every latent block, or at fixed block intervals, plus a stateless/per-frame decode diagnostic if feasible. Merely resetting once per clip is normally identical to deployed rolling decode and does not isolate within-clip temporal-state damping.

Also verify the current implementation against the published CF++1/One-Forcing baseline:

- first latent block uses four denoising steps; later blocks use one;
- fake-score:generator update ratio is 5:1;
- flow timestep shift and CFG settings match the intended config;
- context noise is zero for the clean-context baseline;
- clean-cache update/re-encoding behavior is understood and counted in NFE;
- Stage-3 self-rollout uses the same cache length, eviction order, RoPE handling, block cleanliness, and recent-ring policy as deployment.

### Interpretation

| Result | Likely blocker | Next action |
|---|---|---|
| `G1` loses motion under every decoder | causal-CD one-step initialization | Experiment 3, then Experiment 1 |
| motion survives `G1` but decays across `G2...Gn` | DMD objective/overtraining | Experiment 1 or 4; select an earlier checkpoint |
| full Wan VAE moves but rolling TAEHV morphs | decoder temporal state | Experiment 7 |
| early frames move, long rollouts morph | exposure bias/history policy | Experiment 2, then 6 |
| ring changes drift but not early-frame motion | expected: ring is not the motion prior | keep best ring, proceed to 1/3/4 |

### Cost and gate

Very low: inference/evaluation only, no new checkpoint. This audit can prevent an expensive training run on the wrong component. **KV relation:** the ring A/B covers only one cell of the cache-policy check.

## 1. Port One-Forcing's real-data noised-latent critic

### Technique/source

[One-Forcing paper](https://arxiv.org/html/2605.23458), [released code](https://github.com/Aurora-edu/One-Forcing), and [checkpoint/data](https://huggingface.co/JiaqiFeng/OneForcing). This is the closest available match: Wan2.1-1.3B, 832x480, one latent frame per block, and one later-block denoising step. Count any separate clean-cache population forward in deployed NFE.

### Hypothesis

CF++1's self-forced DMD supplies a local score-difference gradient but no explicit real/fake rejection of a globally plausible-looking morph. A critic grounded in actual real video latents should improve blurry/melted one-step rollouts while preserving distributional supervision. Whether it increases coherent displacement, rather than only sharpness and realism, is the question this A/B must answer.

### Exact A/B

Hold the current causal-CD initialization, ring policy, self-rollout horizon, prompts/data, and number of **generator** updates fixed.

- `A`: current asymmetric DMD Stage 3.
- `B`: current DMD + One-Forcing GAN, with the current causal-CD initialization unchanged so this isolates the critic.
- Optional only after `B` is healthy: `B-low/B-high` with both `lambda_G` and `lambda_D` set to 0.01/0.06 around the paper default 0.03.

Paper-based starting recipe:

- fake-score Wan DiT doubles as critic backbone;
- generator and fake score are Wan2.1-1.3B; the frozen real score is Wan2.1-14B;
- register-attention heads on transformer layers 21 and 29;
- 2 learned registers, feature dim 1536, FFN dim 2048, 12 heads;
- independently noise real dataset latents and current one-step self-rollouts; use timestep shift 5 for rollout, DMD, and GAN sampling;
- non-saturating logistic GAN; `lambda_G=lambda_D=0.03`; no R1/R2 initially;
- one critic/fake-score update per iteration and one generator update every five;
- generator and frozen real-score CFG 5, fake-score CFG 0;
- AdamW with betas `(0,0.999)` and weight decay 0.01; generator and critic learning rate `1e-5`; EMA 0.99 beginning after 50 iterations;
- first block four steps, later blocks one;
- the paper's framewise model converged in 200 iterations on 8 H100s, batch 1/GPU. Evaluating every 25 updates is this plan's local protocol, not a paper setting.

The critic head is training-only, so successful `B` adds no deployed NFE. Monitor a nonzero, changing real/fake logit gap. One-Forcing specifically reports that [ASD's](https://arxiv.org/abs/2511.01419) model-output-vs-model-output discriminator stays near zero because its `n`-step and `n+1`-step classes are both imperfect model outputs.

If Experiment 3 becomes feasible, complete the attribution grid `CD+DMD`, `CD+DMD+GAN`, `ODE+DMD`, and `ODE+DMD+GAN`. The first pair isolates the critic on today's checkpoint; the second axis tests the paper's strongly motion-linked initialization effect.

### Expected result, cost, stop rule

Expected: strong one-step sharpness/realism potential with no inference-speed loss; coherent-displacement gain is unisolated in the paper. Medium engineering/training cost because code is public and backbone/config match closely. Stop if the critic separates trivially while generator quality collapses, if track displacement rises only through warp residual/flicker, or if no positive quality or motion signal appears by 200-300 local updates. The paper's exact convergence report is 200 updates.

**KV relation:** genuinely new. Keep whichever ring arm wins the current A/B constant in both training arms.

## 2. Train on the exact deployed cache policy; add SHA-inspired failure replay

### Technique/source

[Self Forcing](https://arxiv.org/html/2506.08009) and [MotionStream](https://arxiv.org/html/2511.01266) run the inference KV policy during training. [Mirage](https://decart.ai/publications/mirage) corrupts teacher history; Lucy 2's Smart History Augmentation trains on its own imperfect outputs and a drift objective. Lucy's recipe is closed, so the implementation below is a controlled adaptation, not a claimed reproduction.

### Hypothesis

If Stage 3 trains with full history, different eviction, different RoPE positions, or freshly encoded context while deployment uses a recent-clean ring, the current inference A/B cannot close exposure bias. Training through ring eviction teaches the model to condition on exactly the context that will survive at deployment. Replay of CF++1's own morph failures then tests recovery rather than merely discarding old frames.

### Exact staged A/B/C

Let `K` be the winning recent-clean block budget from the current A/B. Make rollout length at least `K+4` blocks so training actually crosses an eviction.

- `A`: current Stage-3 training cache policy.
- `B`: self-rollout with the exact deployed `K`-block clean ring, same eviction and RoPE/cache-update logic at train and inference; detach previous KVs as usual.
- `C1`: `B` plus a 25% stream of frozen-current-student histories selected for morphing/ghosting; make no architecture change.
- `C2`: separately test `B` plus a 25% declared synthetic-corruption stream on clean history: small spatial warp, double edge/ghost, local blur, codec artifact, latent noise, and partial mask deformation. Do not mix `C1` and `C2` in the first attribution pass.
- `C3`: only after `C1` or `C2` helps, add Mirage-inspired binary or scalar history confidence. At inference, either flag all self-generated blocks consistently or use a strictly causal confidence estimator available at deployment; never train with an oracle confidence unavailable online.

Keep all targets distributional: holistic DMD and preferably the Experiment-1 real-data GAN, not a deterministic next-frame endpoint MSE.

If either `C1` or `C2` helps, sweep its probability `{0.1,0.25,0.5}` and sample both short and long error horizons. If neither helps, distinguish “corruptions are unrealistic” from “history is not the early-motion blocker” using P0 results.

### Expected result, cost, stop rule

Expected: modest early motion gain but larger long-roll preservation and recovery after a bad frame. Medium cost; sequential rollouts increase per-update time, but detaching old KVs and using failure replay keeps memory bounded. Stop if early clean-context quality drops, or if it merely hides motion to become robust.

**KV relation:** partially covered. The current A/B tests inference retention/selection. `B` is already covered only if the exact same ring was present throughout Stage-3 self-rollout; the `C` arms are genuinely new.

## 3. Controlled causal-ODE initialization branch

### Technique/source

[Causal Forcing](https://arxiv.org/html/2602.02214) causal-ODE initialization and [One-Forcing's matched ablation](https://arxiv.org/html/2605.23458). One-Forcing reports Dynamic Degree 52.76 from causal-ODE init versus 23.61 from CF++ causal-CD init under the same one-step DMD+GAN objective.

### Hypothesis

Local adjacent-step consistency is efficient and strong in aggregate metrics, but in the extreme one-step setting it may underrepresent Wan's sharp high-noise bend. Causal ODE trajectories expose the student to a richer multi-step motion path and may seed a stronger displacement prior. The published evidence shows motion suppression, not that causal CD literally learns a conditional mean.

### Exact A/B

Two levels of evidence:

1. **Cheap compatibility screen:** evaluate the released One-Forcing ODE-initialized Wan1.3B checkpoint with the same prompt/decoder metrics, without treating it as a controlled result.
2. **Controlled run:** start from the same teacher-forced causal AR teacher and data.
   - `A`: current CF++ causal-CD Stage 2.
   - `B`: generate full PF-ODE trajectories with the causal teacher and distill their intermediate-to-clean pairs into the Causal Forcing **few-step** causal ODE initialization. Condition each pair on its matched clean ground-truth prefix; do not describe this Stage-2 model as directly regressed to one step.
   - Run identical Stage 3 on both, preferably the Experiment-1 DMD+GAN, with equal generator updates and the same ring.

The original Causal Forcing scale used roughly 3,000 causal trajectories and 1,000 initialization steps; begin with a smaller prompt-balanced pilot and promote only if P0 shows Stage-2 motion loss. Inspect trajectory temporal differences so the ODE set actually contains displacement-rich examples.

### Expected result, cost, stop rule

Expected: potentially the largest motion gain, at high preprocessing/training/storage cost. Evaluate the Stage-2 ODE initialization at its valid few-step schedule, then make the decisive comparison after identical one-step Stage 3; do not reject it merely because an unsupported pre-Stage-3 one-step sample is weak. Stop if the valid-schedule initialization contains no richer tracked motion or if any gain vanishes after identical Stage 3. Do **not** approximate this experiment by adding clean-endpoint MSE to the current generator; the published one-step ablation nearly eliminates dynamics.

**KV relation:** genuinely new and orthogonal.

## 4. Replace the Stage-3 score update with SGMD

### Technique/source

[Score Gradient Matching Distillation](https://arxiv.org/html/2605.30116). Its four-step Wan2.1-14B experiment reports optical flow 9.29 vs 4.51 and Dynamic Degree 93.06 vs 80.56 for DMD2, specifically targeting over-smoothed low-motion solutions.

### Hypothesis

The reverse-KL DMD score-difference update is mode-seeking and may choose a low-motion mode. SGMD's Fisher/score-gradient game can preserve broader dynamic modes without a paired endpoint target.

### Exact pilot

- `A`: current self-forced DMD updater.
- `B05/B10/B20`: preserve the generator, causal self-rollout, ring, fake-score model, data, and one-step sampler, but replace only the generator/fake-score objective with SGMD using `lambda={0.05,0.1,0.2}`; start with the paper's balanced 0.1.
- Use one fake-score update per generator update for the paper-exact pilot rather than silently retaining DMD2's 5:1 schedule.
- Evaluate after 50 and 100 generator updates before committing a full run.

Use the paper's objective/pseudocode exactly; a decoded RAFT loss is not SGMD. In the generator update, detach the teacher real-score path but preserve the gradient through the fake-score input/Jacobian. In the residual-contraction fake-score update, detach the generator. This is a two-backward algorithm, not a scalar-loss substitution. Code availability is the main risk: the claimed LightX2V path was not discoverable in the current repository, and the published evidence is 14B, non-causal, four-step, batch 32 on 32 H100s for 300 iterations.

### Expected result, cost, stop rule

Medium-to-high implementation risk, moderate training cost once implemented. Promote only if coherent tracked displacement rises while warp residual and identity remain stable; the paper notes higher lambda can add blur. **KV relation:** genuinely new.

## 5. Verify or enable four-step first-block enhancement

### Technique/source

CF++1, One-Forcing, and Causal-rCM all spend extra **denoising steps** on the first latent block, then stream with one denoising step. The first block establishes layout, appearance, and initial KVs. A separate clean-cache population forward, if used, must be counted independently.

### Exact A/B

- `A`: one denoising step on the first block and all later blocks.
- `B`: four denoising steps on the first block using the intended shifted Wan schedule; one denoising step on all later blocks.

Evaluate not only the first frame but motion over the next 64 frames and after the first occlusion. Measure first-frame delay separately; steady-state fps is unchanged.

### Cost and relation

Negligible engineering and +3 denoising evaluations once per stream relative to the same cache-update policy. Causal-rCM, for example, reports its clean-context one-step framewise path as NFE 2 because clean KV population costs another transformer evaluation. Likely already present in faithful CF++1; if so, record and close it rather than spending a run. **KV relation:** independent, not a ring alternative.

## 6. Add one immutable sink to the winning recent ring

### Technique/source

[MotionStream](https://arxiv.org/html/2511.01266) found one initial-input sink chunk plus one recent-window chunk better than no sink or a longer window; [StreamAvatar](https://arxiv.org/html/2512.22065) permanently retains reference KVs. Both train with the deployed policy. A text-to-video system without an input reference must adapt this to its first clean generated block rather than claim a literal port.

### Hypothesis

A recent-only ring keeps local motion but eventually loses the original object's identity/layout. A first-clean-generated-block sink may preserve identity while recent KVs carry displacement, but this is an adaptation of input/reference sinks. Too much sink/history can freeze motion or preserve stale geometry.

### Exact A/B/C

After the current ring test chooses `K`:

- `A`: recent-clean ring `K`, no sink—the current candidate.
- `B`: for `K>=2`, keep the same total KV token budget, reserve one block for the first clean generated block and `K-1` for recent blocks. If `K=1`, this arm is undefined.
- `C`: one sink plus one recent block, matching MotionStream's best reported `c3s1w1` pattern. If this differs from `A`'s total token budget, report it as a separate published-shape screen with its memory/latency delta, not as the equal-budget comparison.

First do an inference-only screen. Any promising arm must then receive a short Stage-3 fine-tune with the exact policy; MotionStream attributes stability to train/inference parity. Include scene-cut and large-camera-motion prompts because fixed sinks fail there.

### Expected result, cost, stop rule

Low implementation/memory cost, low-to-medium fine-tune cost. Expect identity/drift improvement more than immediate displacement. Reject if camera-compensated motion falls or objects snap back toward the sink. **KV relation:** the current A/B covers recent eviction, not a permanent sink; this is a new extension.

### Lucy self-anchor is a different follow-up

If the model has a reference-conditioning path, separately test `anchor refresh interval={8,16,32}` frames, trained and inferred identically, with reset on cuts/person changes. This adapts Lucy 2.5's explicit recent-output reference. Decart does not disclose anchor selection, refresh cadence, or whether the launch-page snapshot is one-time or repeated; these intervals are our controlled adaptation. Do not call a sink or ring “self-anchoring”; a Lucy-style anchor is a distinct conditioning channel and is medium effort. Watch especially for motion freezing.

## 7. If P0 implicates TAEHV, separate decoder-state and decoder-speed tests

### Technique/source

Oasis uses a frame-independent ViT AE with no temporal compression. [MotionStream](https://arxiv.org/html/2511.01266) instead trains a 9.84M compact decoder in the original Wan2.1 `8x8x4` temporally compressed codec using reconstruction, adversarial, and LPIPS losses; it moves the same streaming model from 16.7 to 29.5 fps on one H100 with little reported LPIPS change. MotionStream does **not** claim that this Tiny VAE is stateless or frame-independent.

### Hypothesis

Rolling TAEHV state may temporally average a translated edge into a deformation. A compact causal decoder tests the speed bottleneck; a block-stateless decoder tests temporal-state error propagation. Those are separate hypotheses.

### Exact staged test

1. Use P0's full-Wan-VAE vs deployed rolling TAEHV vs per-block/reset-interval TAEHV results as the gate.
2. For speed, train `B`, a MotionStream-style compact **causal** decoder against the full Wan VAE in the same Wan latent space, starting with its reconstruction + LPIPS + adversarial recipe.
3. Only if P0 implicates cross-block decoder state, train `C`, an explicitly original block-stateless decoder that consumes the same temporally compressed Wan latent blocks but carries no state between blocks.
4. Compare at equal generated latents: `A` rolling TAEHV, `B` compact causal Tiny VAE, and `C` block-stateless decoder. An optional `C+flow` correspondence loss is another original arm and must be judged for frozen textures.

### Cost and relation

Medium decoder-training/integration cost and zero DiT retraining for `B` or `C` if they preserve the Wan latent interface. `B` has direct speed evidence; `C` is a motion diagnostic with no MotionStream evidence. A literal Oasis ViT AE swap is high-cost and low priority because it changes temporal compression and latent distribution and would require generator retraining. **KV relation:** independent.

## 8. Oasis context stabilization sweep—carefully translated to one step

### Technique/source

Released Oasis-500M uses stored self-generated endpoint history with no added q-noise but a constant history timestep tag `t=14`; historical commit [`abe07fa`](https://github.com/etched-ai/open-oasis/blob/abe07fa3a4f25429aaf9ab264fcce1c362f9e584/generate.py#L47-L137) physically q-noises history near `t=299` early and reduces it toward roughly `t=19` late. The latter was removed from HEAD. [Causal-rCM](https://arxiv.org/html/2606.25473) finds residual noisy context helps chunkwise efficiency/robustness but hurts framewise fine detail.

### Hypothesis

A small history-noise label can tell the model not to copy malformed details literally while retaining coarse displacement. Physical noise may suppress high-frequency error accumulation, but it may also erase motion detail. Evidence for framewise one-step quality is weak.

### Exact sequence

**Phase 1, cheap constant-tag test:** with clean stored history values and the newest clean block protected, compare history conditioning levels matched by `alpha_bar`/SNR to Oasis `t={0,5,14,30}`. Do not pass those raw DDPM indices into Wan's rectified-flow schedule. Fine-tune and infer with the same matched level; an inference-only unseen tag is not a fair test.

**Phase 2, only if Phase 1 helps:** as an explicitly non-Oasis adaptation, q-noise only older history with `sigma={0.02,0.05,0.10}` while the newest block remains clean, using the same corruption during self-rollout training and inference. Compare fixed epsilon per cached block versus fresh epsilon only as a secondary ablation. Re-encoding changing noisy history can destroy the speed benefit, so count every cache update/NFE.

**Two-pass Oasis-inspired diagnostic:** temporarily run a first pass with physically noised history matched to historical Oasis `t≈299`, then a second pass with un-noised/small-tag generated history. This compresses the advertised early/noisy-to-late/clean axis but is not the literal implementation. The historical code used 50 current-frame reverse passes, `ctx_idx=min(j,15)`, fresh Gaussian context noise on every pass, and the same pass-dependent noise level for all history frames. The two-pass test will likely miss 29 fps and is only evidence for whether such a teacher should be distilled.

### Cost and relation

Low for tag wiring/small fine-tune; medium or speed-expensive for physical noising and re-cache. Expect drift robustness, not the main early-frame motion fix. **KV relation:** genuinely new; it changes context state/distribution, not which blocks survive.

## 9. Motion-targeted data and correspondence signals

### 9A. Lucy-2.5-inspired motion-pair corpus

**Source.** [Lucy 2.5](https://decart.ai/publications/lucy-2-5-raising-the-bar-for-live-ai) builds video-edit pairs by propagating high-quality single-image edits through full clips. Decart describes edits propagated through existing video, not synthetic object-motion trajectories or known displacement ground truth; the motion-controlled corpus below is our extension.

**Hypothesis.** The training mixture underrepresents rigid transport, occlusion, and reappearance, so a one-step model can minimize loss through appearance morphing. Known object translations make displacement identifiable.

**A/B.** Build a small, auditable corpus with masks/tracks and known transforms: horizontal/vertical/depth-like scale translations, articulation, object crossings, partial/full occlusion, shadow/contact movement, and camera parallax. Start with `{0,5,10,20}%` of Stage-3 batches, balanced by displacement magnitude. Use distributional DMD/GAN on full clips; do not regress the stochastic generator to one exact future. Report results by motion category to catch synthetic overfit.

**Cost.** High data/validation effort but high upside if P0 shows the teacher has motion and the student lacks coverage. Decart discloses no corpus details, so this is inspired by—not identical to—Lucy.

### 9B. MotionStream-style track conditioning

**Source.** [MotionStream](https://arxiv.org/html/2511.01266) concatenates lightweight sinusoidal sparse-track features and distills joint text-motion guidance.

**A/B.** If explicit user motion control is acceptable, add the paper-specified track condition and test text-only vs text+tracks with identical generation. No public MotionStream implementation was found in this audit. If CF++1 must remain text-only, an auxiliary RAFT/CoTracker head or correspondence-aware critic trained from teacher tracks is a speculative adaptation; keep it training-only and ablate it separately.

**Cost/relation.** High training/data/interface cost. Directly relevant to displacement, but not a cheap port. **KV relation:** independent.

## 10. Use H100 optimization to afford a second denoising step

### Technique/source

One-Forcing shows Wan video trajectories bend sharply at high noise; a second anchor can represent that bend. Portable leads from Decart and the NVIDIA Lucy Restyle session are H100 FP8, mega-kernels/deeper fusion, hardware-shaped dimensions, and TensorRT execution graphs. CUDA graphs, explicit attention/MLP fusion, communication removal on one GPU, and a fast decoder are adjacent engineering choices rather than disclosed Lucy-specific techniques. MXFP8/NVFP4 and Sohu are not H100 options.

### Exact quality-at-fixed-latency sequence

- `A`: current BF16/FP16 one-step CF++1 at measured speed.
- `B1`: change only the DiT precision to H100 Transformer Engine FP8; measure quality and end-to-end headroom.
- `B2`: add compile/execution graphs and attention/MLP fusion to the accepted `B1`, with the winning decoder/cache. Separating `B1` and `B2` preserves causal attribution.
- `C`: spend only verified `B2` headroom on a two-step framewise sampler and require the same >=29-fps steady-state gate. If compatible with the current RF parameterization, start from Causal-rCM's exact schedule: first block `1 -> 15/16 -> 5/6 -> 5/8 -> 0`, later blocks `1 -> 5/6 -> 0`; otherwise declare and train the CF++-native two-step times used.

If a full second denoising evaluation cannot fit, adaptive extra work on a low-rate keyframe followed by re-distillation is a speculative fallback, not a published Decart or One-Forcing technique. Do not claim a hardware win until model+decoder end-to-end time improves; FP8 speed can be limited by memory/cache overhead, and every separate clean-cache update still counts.

### Expected result and cost

Potentially high motion gain but highest systems effort. A second full DiT call starts near 2x compute, so H100 FP8/fusion may not be enough. Dynamic sparse attention is a later option; protect high-flow/moving tokens because indiscriminate pruning can manufacture stillness. **KV relation:** independent.

## 11. Causal-rCM TF-sCM initialization

### Technique/source

[Causal-rCM](https://arxiv.org/html/2606.25473) and [open code](https://github.com/NVlabs/rcm) use RF-native, teacher-forced continuous-time consistency via a custom-mask JVP, then self-forced DMD. It reaches a strong initialization in 1-2k rather than about 10k discrete-CM iterations.

### Exact A/B

- `A`: current CF++ discrete causal-CD init -> identical Stage 3.
- `B`: RF-native TF-sCM/JVP init -> identical Stage 3.

Keep the same causal teacher, data, step schedule, ring, and Stage-3 updates. Measure pre- and post-Stage-3 motion. Causal-rCM's own final framewise ablation gives TF-dCM 84.29 and TF-sCM 83.84 after SF-DMD, so the main demonstrated win is convergence efficiency, not motion.

### Cost/relation

High kernel/training integration cost and lower direct motion evidence than Experiments 1-4. **KV relation:** independent; still train with the deployed cache policy.

## 12. Long-horizon/cache/speed ideas to defer

These are valid leads but should not displace motion work:

### 12A. RAPR

[StreamAvatar](https://arxiv.org/html/2512.22065) stores pre-RoPE keys, caps reference distance, shifts all cached positions coherently, and reapplies RoPE. RAPR is defined around a permanently retained reference sink; without that sink, simple position capping is not a StreamAvatar port. Test only if quality falls when indices exceed the training horizon: compare reference-sink rolling RoPE against the same sink plus capped distance `{training_max/2, training_max}`, with identical KVs and symmetric training/inference. It addresses positional drift and identity, not early morphing.

### 12B. Future-aware token retention

[Future Forcing](https://arxiv.org/html/2605.30083) constructs a future-query proxy from historical pre-RoPE queries, scores cache tokens against it, and merges redundant pairs. After the recent-ring A/B, compare fixed recent eviction against future-aware selection at the **same token count, memory, and latency**. Promote only if long subject consistency improves without reducing moving-foreground attention. The paper cautions that Dynamic Degree can fall when a baseline's higher score comes from unstable scene changes, so use the shared displacement/warp guardrails. This is adjacent to but not covered by a block-level recent ring.

### 12C. Same-budget head-aware cache

[Forcing-KV](https://arxiv.org/html/2605.09681) finds that static heads need the latest transition-anchor frame for chunk continuity, whereas dynamic heads use longer inter-frame correspondences for motion. [Pyramid Forcing](https://arxiv.org/html/2605.13111) classifies Anchor, Wave, and Veil heads and assigns long-strided, periodic, or compact-local histories. Both are training-free cache-policy changes and remain below early-motion objective work because their strongest evidence is long-horizon and H200-based.

Use the winning uniform ring as `A`. For `B`, profile heads once on the fixed suite, retain the latest transition block for static heads, and allocate the same average KV-token budget to longer segment-similarity-selected history for dynamic heads, following Forcing-KV. For `C`, use Pyramid Forcing's offline sign-rate/FFT classification and head-specific retention at the same average token budget. First implement `B/C` with dense masks as a quality-only screen; build ragged/custom kernels only if one wins. Evaluate the long suite and reject any gain that comes from scene cuts or distortion. This is a genuinely new head-level extension to the current block-level ring.

### 12D. LiveEdit spatial feature cache

[LiveEdit](https://arxiv.org/html/2606.26740) reuses self-attention features only for source-defined inactive editing regions. Its own ablation slightly reduces Dynamic Degree and imaging quality, and T2V has no source frame defining “unchanged.” Defer until motion is solved; if used, cache only background/inactive self-attention tokens, never FFN or moving foreground. It is unrelated to temporal KV retention.

## Suggested run order and decision gates

1. Complete P0 and the already-running recent-ring A/B. Select the best existing checkpoint/decoder/ring without retraining.
2. Verify first-block enhancement and the 5:1 fake-score:generator update ratio. Fix configuration mismatches immediately.
3. Run One-Forcing `A/B` on the current CD init. It is the closest low-integration realism port and adds no inference work, but its displacement effect is not isolated in the paper.
4. If P0 shows Stage-2 motion loss, start causal-ODE init in parallel; otherwise prioritize Stage-3 objective/checkpoint changes.
5. Make the winning ring exact during training and add a small failure-replay arm. This is a defensible public-mechanism adaptation inspired by Mirage history augmentation and Lucy SHA, not a reproduction of the closed SHA recipe.
6. Run the SGMD pilot only if One-Forcing still under-delivers motion or its critic adds sharpness without displacement.
7. Promote decoder work only if the full-VAE/reset audit proves the rolling decoder is damping motion.
8. Use sink/self-anchor/context-noise and same-budget head-aware cache tests for long-horizon stability after early displacement improves.
9. Undertake motion-data and two-NFE systems programs only after the low/medium-cost objectives are exhausted.

The central separation to preserve is:

- **recent-clean KV ring:** which recent clean context survives;
- **attention sink/RAPR/Future Forcing/Forcing-KV/Pyramid Forcing:** which older/reference information survives, per block or per head, and how it is positioned;
- **dynamic/noisy context:** what numerical state/history noise the model consumes;
- **Smart History Augmentation/self-forcing:** which error distribution it trains on;
- **One-Forcing/SGMD/ODE initialization:** the realism objective and initialization choices most directly implicated in one-step motion.

Only the first item is covered by the current A/B.
