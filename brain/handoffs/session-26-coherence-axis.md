---
type: handoff
status: active
session: 26
date: 2026-07-21
description: "Added and calibrated an independent coherence axis, then integrated it into deterministic motion-by-coherence batch comparison without changing legacy displacement output."
branch: main
key_commits: []
prior_handoff: "session-25-openstudio-4090-open-model-server"
---

# Session 26 Handoff — coherence axis

- Shipped `bench/coherence_metrics.py`: all-pairs DINOv2 plus RAFT temporal coherence, early-reference patch integrity, and causal early/middle/late trajectories; red-first review regressions now distinguish coherent object entry from abrupt texture replacement and prevent late failures from rewriting clean history.
- The read-only 26-clip calibration reproduces finals 9.796132 > four-step 5.673292 > one-step 0.969217; SGMD update-25 ball is 0.000000 and vehicle is 7.553927 with 10.000000 → 8.763047 → 6.935005 decay. `--with-coherence` produced the real 4×4 two-axis SGMD matrix while all 16 displacement reports remained byte-identical to the accepted baseline.
- Validation covers 25/25 fast batch, 14/14 displacement, exact 11-clip learned P0, exact 26-clip learned coherence, and real DINOv2+RAFT motion at 10/10/10; final independent review found no blocker, the CLI panel remained advisory after Kimi/Grok balance failures, and no pod, API, source mutation, model/inference change, or commit occurred.
