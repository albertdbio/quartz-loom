---
type: research
status: active
date: 2026-07-21
description: "Three SGMD distillation runs are now measured on a shared motion-by-coherence plane; lambda=0.2, not 0.05, owns the strongest observed late operating point, with prompt-specific and coverage caveats explicit."
anchors: ["bench/results/sgmd-training-trajectories-20260721/report.json", "bench/results/sgmd-training-trajectories-20260721/report.md", "bench/results/sgmd-training-trajectories-20260721/trajectory.csv"]
related: ["[[Coherent-Displacement-Metric]]", "[[Coherence-Metric]]", "[[State]]", "[[session-27-sgmd-training-trajectories]]"]
---

# SGMD training trajectories

## Measurement contract

The read-only raw-SGMD, Fisher-normalized lambda=0.1, and normalized lambda
sweep corpora contain 96 unique clips across 24 checkpoints. All expected
lambda=0.2 conditions had arrived before scoring, so the final report has no
condition holes. Each corpus has its own sibling `batch_scores_2axis/` tree
with the native schema-4 report, matrix, and complete per-clip displacement and
coherence records: every successful score retains its full report and every
failure remains an explicit row. In the normalized lambda=0.1 tree this means
37/40 displacement JSONs plus three explicit displacement errors, and 40/40
coherence JSONs.

Trajectory motion is the mean of successful **camera-compensated** cells only.
Screen-space fallbacks are starred, counted separately, and never mixed into
that mean. Coherence averages every successful coherence score because it is an
independent axis. Metric errors remain explicit and are never zero-filled; each
row reports both denominators.

All 11 supplied vehicle anchors reproduce at the stated precision and the
current full displacement reports are byte-identical to the stored independent
spot reports. The provenance check corrected one wording error in the task:
normalized lambda=0.1 vehicle steps 15 and 20 are screen-space fallbacks
(`camera_compensated=false`), even though their numeric anchors 2.456558 and
1.963068 match exactly.

## Lambda result

The registered framing does **not** reproduce. Across equal pooled coverage of
15 compensated cells, lambda=0.05 averages **1.078594 motion / 6.423868
coherence**, while lambda=0.2 averages **1.886850 / 7.033567**. At step 50,
where both runs have 4/4 compensated prompts and zero degraded cells, the
operating points are:

| Run | Step-50 motion | Step-50 coherence |
| --- | ---: | ---: |
| normalized lambda=0.05 | 0.735094 | 6.834893 |
| normalized lambda=0.2 | 1.693540 | 7.403891 |

On the prompts compensated under both lambdas, lambda=0.2 wins motion at four
of five matched checkpoints and loses only step 20. The advantage is
prompt-specific rather than uniform: at step 50 its walker is worse than
lambda=0.05, while its barrel is dramatically better. Very large lambda=0.2
vehicle scores and the step-30 ball score still need frame/blind review because
motion plus coherence does not establish prompt fidelity, direction, or
physics.

Raw SGMD does not trace a stable top-right path: update 25 has only one
compensated prompt (vehicle 2.030809) and three degraded cells; by update 50 its
compensated mean is 0.224814 while coherence rises to 6.871822. Normalized
lambda=0.1 is non-monotonic and early rows have thin motion denominators, but
from step 30 onward it generally occupies 0.61-1.30 compensated motion and
7.10-7.36 coherence.

## Fisher-normalized step-25 vehicle anomaly

The step-25 file is a valid 81/81-frame H.264 decode, not a broken artifact.
The car advances and then reverses, with a visually measured 196-pixel centroid
span versus 279 and 396 pixels at neighboring steps 20 and 30. Coherence is
**8.723019** (temporal 10.0, spatial 7.609106), with segments
**9.812063 -> 8.581635 -> 8.846127** and `degrades_over_time=false`; this is not
a coherence collapse.

The near-zero **0.009206** displacement score is not a clean measurement of
the visible traverse. The car's roughly 196-pixel visual span is **0.236W**,
while the selected CoTracker tracks report only **0.035W**: three persistent
tracks survive, survival is 0.078189, and the DINO appearance guard is zero.
Classify this as a valid-decode transport shortfall/path reversal **plus a
tracker/identity-confidence failure**, not as proof that the clip has no
motion. Do not re-decode the same file as a corruption repair. If checkpoint
promotion hinges on the isolated dip, add another deterministic sample and a
manual or box-assisted track check.

## Runtime safety learned during scoring

A normalized batch initially spent 90 minutes inside one OpenCV
`warpPerspective`. A near-singular/projective-horizon background homography can
send `BORDER_REFLECT101` source coordinates arbitrarily far outside the frame,
making border reflection effectively unbounded. The optional motion-proposal
warp now validates both the fitted mapping and its inverse over the complete
rectangular domain through corner denominator sign/scale and an eight-frame-
span coordinate bound. Inversion failure or any unsafe sampled mapping
discards the whole optional proposal and falls back to the existing compensated
excursion selector; retaining a partial proposal selected the wrong foreground.

All four forward/inverse/singular/partial-proposal guards are red-first
regressions. The normal pilot vehicle report
remains exact, normalized step-10 vehicle returns to its independent
**1.795996** full report, and the complete fast displacement/batch/coherence
suite passes 58 tests (four learned opt-ins skipped).

## Limits

- Early trajectory points with only one or two compensated prompts are not
  directly comparable to full-coverage rows without reading `n/4`.
- A high displacement value can be coherent but semantically wrong; inspect
  the per-prompt matrix before promoting a checkpoint.
- `degrades_over_time=false` can coexist with a moderate middle-segment dip or
  a clip that was already poor early.
- The result ranks these fixed clips, not the full seed distribution of each
  checkpoint.
