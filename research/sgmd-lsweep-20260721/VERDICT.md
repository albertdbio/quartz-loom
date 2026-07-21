# λ-sweep + two-axis verdict: fnorm λ=0.2 is the recipe — best motion×coherence operating points of the project

Date: 2026-07-21. Scored by the calibrated two-axis harness (motion = camera-compensated displacement, coherence 0-10; codex/agent0 batch run, all 11 vehicle cross-checks byte-identical to independent spot scores). Full data: `realtime-video/bench/results/sgmd-training-trajectories-20260721/{report.md,trajectory.csv}`.

## The four-run trajectory (mean displacement / mean coherence, compensated cells)

| run | best operating points |
|---|---|
| DMD final (fork baseline) | 0.29 motion / ~10 coherence — perfect stills |
| raw SGMD λ0.1 | 2.03 / **1.9** @25 (dissolving) → motion collapses by 50 |
| fnorm λ0.1 | wobbly: 1.8-2.5 motion @10-15 but coherence dips; stabilizes ~1.0-1.3 / 7.2-7.4 @35-50 |
| fnorm λ0.05 | 1.7 / 7.35 @40 best; weaker elsewhere |
| **fnorm λ0.2** | **2.65 / 6.99 @20 · 2.01 / 7.32 @40 · 1.69 / 7.40 @50** — sustained motion AND coherence |

## Conclusions
1. **Recipe freeze: normalized Fisher + λ=0.2, checkpoints in the 20-40 window.** The reviewer's advisory (normalization inflates the NR term's relative weight) proved *beneficial* at higher λ: λ0.2 delivers the most motion at good coherence, sustained across the window rather than at one knife-edge.
2. **The trade is now quantified end-to-end**: DMD (0.3, 10) → SGMD variants populate the frontier up to (2.65, 7.0). The operating point moved from "perfect stills" to "real transport at watchable coherence" via objective + normalization + λ — zero architecture changes, zero data changes.
3. **Step-25 anomaly resolved** (codex): fnorm-λ0.1 @25 vehicle = motion 0.009 at coherence 8.7 — a *coherent but motionless* checkpoint, i.e. genuine transient motion dropout (training wobble), not a bad decode. No re-decode needed; adjacent checkpoints carry the signal.
4. Honest residuals: ball/barrel prompts still under-perform (commanded arcs are harder than traverses; many cells degraded/static — the mean rows are diluted by them); single seed; coherence ~7 ≈ soft/stylized, not crisp. Next lever candidates: longer λ0.2 run with EMA, quality anchor (real-data critic with AAD-1's motion-cost caveat), 2-step sampling at deploy.

## Next experiment (when resumed)
Long run: fnorm λ0.2, 200 updates, EMA on, eval every 10 — does the 20-40 sweet spot extend or drift, and does EMA lift coherence toward 8+ at motion ≥2?
