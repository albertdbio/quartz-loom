# SGMD Stage-3 objective for Causal-Forcing (vendored delta)

Our modification of [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)
(Apache-2.0; upstream commit in `UPSTREAM_COMMIT`) replacing the Stage-3 asymmetric-DMD
objective with **Score Gradient Matching Distillation** (SGMD,
[arXiv 2605.30116](https://arxiv.org/abs/2605.30116)) plus an optional Fisher-weight
normalization (`sgmd_fisher_normalization: batch_mean`) that bounds the ~1e10 dynamic
range of the exact c(t)=α²/σ⁴ metric — the change that cured the high-frequency
dissolution we observed at raw weighting.

Files:
- `sgmd.py` → drop into `model/` (route `distribution_loss: sgmd`; see the patch)
- `test_sgmd_cpu.py` → `tests/` — CPU-only gradient-structure proofs (two-backward
  ordering, detach semantics), no GPU needed
- `causal_forcing_sgmd_framewise_1step.yaml` / `..._fnorm_...yaml` → `configs/`
- `upstream-changes.patch` — the wiring diff (trainer routing, `model/__init__`,
  config-driven CPU offload of the frozen 14B real score)

Result summary (see `../sgmd-*/VERDICT.md` and
`../../bench/results/sgmd-training-trajectories-20260721/`): DMD sits at
(0.29 motion, ~10 coherence) — perfect stills; normalized-Fisher SGMD at λ=0.2,
updates 20–40, reaches (2.65, 7.0) and sustains (1.7–2.0, 7.3–7.4) — real object
transport at watchable coherence, with no architecture or data changes.
