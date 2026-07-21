---
type: handoff
status: active
session: 24
date: 2026-07-21
description: "Added explicit degraded displacement evidence and scored the four-update SGMD pilot without mixing screen-space fallback values into compensated comparisons."
branch: main
key_commits: []
prior_handoff: "session-23-fork-grid-displacement-baseline"
---

# Session 24 Handoff — SGMD degraded displacement

- Shipped the red-first error contract and narrow camera-compensation fallback: deterministic metric-error rows, fail-loud unexpected exceptions, fallback-only provenance, starred degraded cells, and default comparison exclusion with explicit opt-in.
- The read-only SGMD 4×4 pilot yielded 16 results / 0 ERROR / 0 holes / 8 degraded; compensated vehicle is exactly 2.030809 / 0.352059 / 0.061183 / 0.227143, supporting relative update-25 preservation followed by re-collapse while every cell remains below the absolute displaced gate.
- Validation is 20/20 batch, 14/14 scorer, and exact 11-clip learned P0; Opus and a local actual-file reviewer found no issue (Kimi/Grok balance failures kept consensus partial), with no pod, video API, source-clip mutation, model/inference change, or commit.
