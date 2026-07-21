---
type: handoff
status: active
session: 22
date: 2026-07-21
description: "Shipped deterministic condition-by-prompt batch comparison around the accepted displacement scorer, with explicit holes and exact P0 reproduction."
branch: main
key_commits: []
prior_handoff: "session-21-coherent-displacement-metric"
---

# Session 22 Handoff — displacement batch comparison

- Shipped `bench/displacement_batch.py` without changing scorer logic: full per-clip JSON, deterministic JSON/Markdown matrices, explicit holes, and `--compare A B` as B-minus-A with per-prompt verdicts and a comparable-only mean.
- Final validation is green: 10/10 fast batch tests, 11/11 displacement tests, and the real CoTracker3+DINOv2 11-clip P0 regression exactly reproduces session-21 score/span/decision values; the example has 11 clips, nine holes, and ring ON-minus-OFF mean **-1.018855**.
- Handoff to the experiment owner: point the CLI at the decode-grid root plus ordered prompt file and retain blind review for semantics; no model/inference work, pod use, P0-source mutation, scorer change, or git commit occurred.
