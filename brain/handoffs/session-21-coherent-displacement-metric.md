---
type: handoff
status: active
session: 21
date: 2026-07-21
description: "Shipped a local CoTracker3/DINOv2 coherent-displacement metric with red-first anti-flicker/camera tests and an honest P0 calibration."
branch: main
key_commits: []
prior_handoff: "session-20-editable-optional-prompt-enhancement"
---

# Session 21 Handoff — coherent displacement metric

- Shipped `bench/displacement_metrics.py`: MP4/PNG CLI+library, CoTracker3 primary-object tracks, two-space camera compensation, adjacent-flow/DINO guardrails, 0–10 score, and conservative boolean; **11/11 tests pass** and the official learned translating square scores 9.809/10.
- P0 calibration is directional, not definitive: 11 labels / 10 independent contents give deduplicated Spearman **ρ=0.608, p=0.062**; D1 rolling versus D0 Wan is equal within tolerance (Δscore 0.111, Δspan/W 0.00137), and both blind-worse ring pairs remain below OFF.
- The dedicated H100 is EXITED and untouched; next research should consume this harness as the measurement screen, keep blind review for semantics/articulation, and leave latent-generator experiment selection to the orchestrating Model/Inference researcher.
