# SGMD Stage-3 pilot verdict: motion is preserved, stability is not (yet) — the objective is the knob

Date: 2026-07-21. Pod `sgmd-pilot-2xh100` (2×H100, ~1.9h total, ≈$12, stopped). Training: SGMD (arxiv 2605.30116) replacing asymmetric DMD as Stage-3, from `stage2_causal_ode.pt` (ODE init), λ=0.1, 1:1 update ratio, lr 1e-6, 100 generator updates, frozen Wan-14B real score CPU-offloaded, `prompts/demos.txt` (100 motion-heavy prompts), seed 42. Eval: checkpoints 25/50/75/100 decoded on the 4 P0 displacement prompts (exact fork-grid wiring, seed 0), scored with the calibrated displacement metric.

## Result matrix (vehicle prompt — the only prompt where the harness could score every step)

| checkpoint | displacement_score | visual |
|---|---|---|
| step 0 = ODE init @ 1-step | 0.007 | incoherent blur (init is 4-step-native) |
| **SGMD step 25** | **2.031** | **car enters, crosses, exits the frame — sharp for ~2/3 of the clip**, background degrades after |
| SGMD step 50 | 0.352 | large motion, degrading coherence |
| SGMD step 75 | 0.061 | motion re-collapsed |
| SGMD step 100 | 0.227 | near-static + stylization/saturation drift |
| DMD final (fork-grid baseline) | 0.287 | pristine near-still |

Training health: all 100 steps clean — finite losses, grad norms 0.04–1.6 (never at the 10.0 clip), critic learning, no NaN.

## The other three prompts (honest limitation)
Ball/walker/barrel at step 25: genuine motion onset (ball starts falling, barrel rolls, walker strides) but the scene **dissolves into pointillist noise within 1–2 s**. The displacement harness correctly refused most of these cells ("too few persistent background tracks for camera compensation") — blank cells are honest nulls, not missing data. step-25 coherence is prompt-dependent; the vehicle scene held, the others did not.

## Conclusions
1. **The Stage-3 objective is confirmed as the motion control knob.** Same init, same data, same steps: DMD → parked car at 0.29; SGMD@25 → car crosses the frame at 2.03 (≈ the 4-step init's 2.13). First checkpoint in the project to execute commanded object transport at 1 step.
2. **SGMD at this config trades long-horizon stability for motion** — the mirror image of DMD's trade. Continued training re-collapses motion (progressive dose effect, now measured at both extremes).
3. **No usable checkpoint exists yet** — step 25 is a proof of mechanism, not a model.

## Next experiments (ranked)
1. **Fisher-weight normalization** — reviewers flagged c(t)=α²/σ⁴ spans ~1e10 unnormalized; the low-σ explosion plausibly drives the pointillist dissolution. Normalize (DMD-style mean-abs or σ-clamp) and rerun. Cheapest, most-implicated fix.
2. **SGMD→DMD anneal / hybrid**: SGMD for the first ~25 updates (lock the motion prior into the 1-step student), then DMD or a blended objective to stabilize. Both endpoints are now measured; the anneal interpolates them.
3. **λ sweep** ({0.05, 0.2}) + EMA on + eval every 5 updates in the 10–40 window.
4. Harness gap for the measurement researcher: a screen-space-only fallback score when camera compensation is impossible, so degraded clips get a number + a coherence flag instead of a refusal.

## Caveats
Single seed, single λ, 4 prompts, 100-prompt training set, eyeball + single-metric scoring; the vehicle result is n=1 prompt at n=1 seed. The claim is mechanism (objective controls the motion↔stability trade), not a performance claim.
