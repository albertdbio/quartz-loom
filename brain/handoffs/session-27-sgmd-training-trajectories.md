---
type: handoff
status: active
session: 27
date: 2026-07-21
description: "Scored three SGMD distillation runs on the shared motion-by-coherence plane and recorded the higher-lambda result with denominator and measurement-confidence caveats."
branch: main
key_commits: []
prior_handoff: "session-26-coherence-axis"
---

# Session 27 Handoff — SGMD training trajectories

- Scored 96 read-only clips across 24 raw/normalized/lambda-sweep checkpoints into three sibling `batch_scores_2axis/` trees plus deterministic combined JSON, Markdown, and CSV; compensated-motion denominators, degraded counts, errors, and all prompt cells remain explicit.
- Normalized lambda=0.2 wins matched-prompt motion at four of five checkpoints and owns the full-coverage step-50 point at 1.693540 motion / 7.403891 coherence versus lambda=0.05 at 0.735094 / 6.834893, but the gain is prompt-specific and requires frame/blind fidelity review before promotion.
- Fnorm vehicle step 25 is a valid coherent decode with shortened/reversing transport plus tracker/identity-confidence failure, not a clean zero-motion measurement; 58 fast tests, the independent final audit, and the final 3/3 CLI panel are green after bidirectional homography-warp bounds, with no pod/API/input mutation/model work/commit.
