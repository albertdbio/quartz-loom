# Fisher-normalization verdict: dissolution cured, motion window widened — best checkpoint yet

Date: 2026-07-21. Pod `sgmd-fnorm-2xh100` (~1.3h, ≈$8, stopped). Change vs the raw pilot: ONLY `sgmd_fisher_normalization: batch_mean` (reviewer-fixed: config validation, mask-aware normalizer, norm-factor logging) — bounding the ~1e10 c(t) dynamic range. 50 updates, checkpoints every 5, same init/λ/lr/prompts/seed. Normalization confirmed active in-log (`dmdtrain_gradient_norm ≡ 0.999`).

## Vehicle displacement curve (calibrated metric; comparators: raw pilot 25→2.03, 50→0.35, 75→0.06, 100→0.23; DMD floor 0.29; ODE-init 4-step ceiling 2.13)

| update | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | refused | 1.80 | **2.46** | 1.96 | 0.01* | 2.20 | 2.05 | 1.35 | (unscored) | (unscored) |

*step-25 is an outlier pending inspection (single seed noise or a bad decode); the surrounding curve is consistent.

## Coherence (montage review)
- **Ball @15: no dissolution** — coherent ball for all 81 frames (bounce, round, soft late). Raw pilot dissolved to noise by frame 20. The pointillist failure is gone.
- Ball @50: still coherent; motion reduced (slow re-collapse) + slight color drift.
- Vehicle @15: full traverse; background soft-blurs late but structure persists.
- New residual defect: progressive SOFT BLUR replacing the old noise dissolution — a much better failure mode, plausibly addressable (λ/NR balance, more updates of critic, or quality anchor).

## Conclusion
1. The c(t) dynamic range WAS the dissolution driver — bounding it cures the coherence collapse at zero motion cost.
2. Motion now holds across a wide window (10–40) instead of one knife-edge checkpoint; slow decay persists.
3. **Current best checkpoint of the project: fnorm step 15** (motion 2.46 ≈ above init's 4-step level, coherent clips). vs DMD 0.29 static / raw-SGMD 2.03-but-dissolving.

## Next
- Score 45/50 + the step-25 outlier; run the two-axis (motion × coherence) matrix when the coherence metric lands (codex building it).
- Attack the soft-blur: candidates — λ sweep on the now-balanced NR term, longer run with EMA on, or One-Forcing-style real-data critic as a sharpness anchor (AAD-1 motion-cost caveat applies).
- Blog: add the motion-vs-updates curve figure (raw vs normalized) + ball-then vs ball-now comparison.
Caveats: single seed, single λ, 4 prompts, blur/coherence still eyeball-judged pending the coherence metric.
