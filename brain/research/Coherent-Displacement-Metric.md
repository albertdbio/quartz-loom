---
type: research
status: active
date: 2026-07-21
description: "A local CoTracker3/DINOv2 metric now measures coherent primary-object displacement without rewarding flicker or camera drift. Red-first synthetic gates pass; the 11-label P0 calibration has 10 independent contents and a deduplicated Spearman rho of 0.608 (p=0.062), so it is useful directional signal rather than a replacement for blind review."
anchors: ["bench/displacement_metrics.py", "bench/tests/test_displacement_metrics.py"]
related: ["[[ADR-003-serving-product-model-track-split]]", "[[State]]", "[[SGMD-Training-Trajectories]]", "[[session-21-coherent-displacement-metric]]", "[[session-24-sgmd-degraded-displacement]]", "[[session-27-sgmd-training-trajectories]]"]
---

# Coherent displacement metric

## Purpose and boundary

`bench/displacement_metrics.py` is a CLI and injectable library for exact
81-frame, 480×832 MP4 clips or PNG directories. It measures whether a primary
object travels coherently instead of settling into the near-static behavior now
called **motion-mode collapse**. It is deliberately narrower than general video
quality or articulated-motion scoring and authorizes neither a quality nor a
performance claim.

The default path uses official Torch Hub CoTracker3 offline tracks and DINOv2
ViT-S/14 crop embeddings. A separate dependency environment is pinned in
`requirements-displacement-metrics.txt`; the cached local MPS path scores the
11-label P0 set in about 44 seconds without a GPU pod.

## Measurement

1. CoTracker3 seeds a quasi-dense grid over all 81 frames. An initial
   border-track homography estimates background motion.
2. Primary-object tracks are selected from camera-compensated motion outliers,
   then reduced to a spatially connected persistent component. `--bbox` remains
   an explicit escape hatch for ambiguous multi-object clips.
3. A second homography is fit from tracks outside the selected object. The
   robust object centroid is the per-frame median of persistent compensated
   tracks.
4. The effective trajectory span is the minimum of screen-space span and
   camera-compensated span. This rejects both an ordinary pan over a world-fixed
   object and a screen-locked subject laid over a drifting background.
5. Sparse DINO samples include each base sample and its adjacent frame. Optical
   flow is measured only on adjacent pairs, so a period-two flicker cannot alias
   away under the default ten-frame stride. Raw flow magnitude is diagnostic;
   it never supplies displacement credit.

The reported foreground components include median endpoint displacement in
pixels and frame-width fraction, screen and compensated trajectory spans,
straightness, translation consensus, and survival. The guardrail components
include adjacent flow-warp residual, flow translation coherence, and DINO
median/p10 similarity.

The summary is:

```text
extent = smoothstep(0.015, 0.40, min(screen_span, compensated_span) / width)
coherence = sqrt(track_survival * translation_consensus)
dino = sqrt(smoothstep(0.65, 0.95, median_similarity)
            * smoothstep(0.55, 0.92, p10_similarity))
appearance = cbrt(warp_guard * dino * flow_coherence_guard)
score = 10 * extent * (0.25 + 0.75*coherence)
                    * (0.25 + 0.75*appearance)
```

The conservative boolean additionally requires effective span ≥0.06 frame
width, translation consensus ≥0.50, survival ≥0.50, and score ≥5.0. A calibrated
5.0 threshold keeps the equal-decoder P0 control on the same collapsed side of
the boundary instead of making a 0.111 score difference produce opposite
decisions.

## Red-first gates

Each required behavior was observed failing before its implementation. Four
additional failures found by actual learned-backend review were also pinned
red-first: dense motion-trail contamination, stride-ten flicker aliasing,
low-tail appearance failures, and unpadded PNG temporal order. A symmetric
screen-lock camera case failed at 6.86 before the two-space extent guard.

| Synthetic clip | Score | Effective span/W | Raw flow px | Decision |
|---|---:|---:|---:|---|
| Translating textured square (LK/histogram) | 9.467 | 0.635 | 0.918 | displaced |
| Pulsing/flickering square in place | 0.000 | 0.000 | 2.992 | collapsed |
| Camera pan over world-fixed square | 0.000 | 0.000 | 1.001 | collapsed |
| Camera pan with screen-locked square | 0.018 | 0.025 | 0.064 | collapsed |
| Translating square, official CoTracker3+DINOv2 | 9.809 | 0.676 | 1.192 | displaced |

## P0 calibration

The source worktree remained read-only. Scores below use the official learned
backends with one shared cached model instance. Every clip is 81 frames at
480×832. The `span/W`, survival, consensus, warp, and DINO-p10 columns are score
components rather than independent quality judgments.

| Blind ID / clip | Blind | Score | span/W | Survival | Consensus | Warp | DINO p10 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m07 ring-OFF ball | 5 | 3.890 | 0.216 | 0.982 | 0.903 | 0.021 | 0.712 | collapsed |
| q31 ring-ON ball | 4 | 0.000 | 0.009 | 1.000 | 0.819 | 0.015 | 0.754 | collapsed |
| v18 ring-OFF walker | 3 | 0.050 | 0.049 | 0.957 | 0.831 | 0.013 | 0.395 | collapsed |
| c44 ring-ON walker | 3 | 0.005 | 0.020 | 1.000 | 0.856 | 0.005 | 0.917 | collapsed |
| p62 ring-OFF rolling object | 4 | 0.212 | 0.057 | 0.980 | 0.678 | 0.019 | 0.686 | collapsed |
| h09 ring-ON rolling object | 2 | 0.000 | 0.013 | 0.979 | 0.643 | 0.006 | 0.535 | collapsed |
| x27 ring-OFF vehicle | 1 | 0.000 | 0.007 | 0.977 | 0.863 | 0.004 | 0.963 | collapsed |
| b53 ring-ON vehicle | 1 | 0.071 | 0.040 | 0.984 | 0.957 | 0.009 | 0.610 | collapsed |
| k14 D1 rolling TAEHV | 5 | 3.890 | 0.216 | 0.982 | 0.903 | 0.021 | 0.712 | collapsed |
| s80 D2 reset each block | 3 | 2.115 | 0.206 | 0.902 | 0.817 | 0.046 | 0.568 | collapsed |
| r36 D0 full Wan VAE | 5 | 4.001 | 0.217 | 0.975 | 0.900 | 0.023 | 0.726 | collapsed |

`m07` and `k14` are intentionally byte-identical: Cell-A ring-OFF ball reuses
the retained D1 rolling-TAEHV render. The 11-label correlation is therefore
descriptive only (Spearman ρ=0.701, p=0.016; Kendall τ-b=0.572, p=0.023).
Counting that content once gives the honest inferential result across ten
independent contents: **Spearman ρ=0.608, p=0.062; Kendall τ-b=0.494,
p=0.062**. This is a positive, moderate but not conventionally significant
rank relationship on a very small set.

The expected coarse ordering is directionally present: ring-OFF ball is high,
ring-OFF walker exceeds the ring-OFF vehicle, and the vehicle is collapsed.
The score strongly compresses small in-place gait because it measures coherent
object travel, not limb articulation. That is a known scope boundary, not a
reason to tune the formula until it copies the blind labels.

## P0 regression facts

- Same latent, D1 rolling TAEHV versus D0 Wan: score delta **0.111** (tolerance
  0.5), span-fraction delta **0.00137** (tolerance 0.01), and both decisions are
  collapsed. The decoder remains exonerated on this displacement measurement.
- Where blind review worsened, ring ON never outscored OFF: ball
  `0.000 < 3.890`; rolling object `0.000 < 0.212`. The four-prompt mean is
  `0.019 ON < 1.038 OFF`. The tied vehicle blind pair is not overclaimed as an
  ordered pair.
- The recent-clean ring remains dropped, rolling-state reset remains worse,
  and the serving/measurement track does not re-enter Model/Inference schedule
  experiments owned elsewhere.

## Review and limitations

An actual-file `claude-opus,kimi,grok` CLI panel agreed the implementation is a
sound development displacement meter. Its main concern was the identical
`m07`/`k14` row; the read-only inventory proves that identity is deliberate,
so the correction was to deduplicate inferential correlation rather than alter
the measurement. The panel also caught the knife-edge 4.0 decision mismatch;
the calibrated conservative threshold is now 5.0.

Remaining bounded limitations: automatic primary-object selection is heuristic
for multi-object scenes; frame-zero is the DINO reference; a camera that tracks
a genuinely moving subject can suppress screen-space displacement; and the
small calibration set does not justify replacing blind review. The harness is
the repeatable first screen for future experiments, with blind review retained
for semantics and articulated motion.

## Explicit degraded fallback (2026-07-21)

Some experimental clips dissolve or churn so strongly that too few persistent
background tracks remain for a defensible homography. That condition no longer
makes the whole batch opaque. Only the dedicated camera-compensation failure
now enters a screen-space-only path; other metric-domain errors and unexpected
`OSError`/`RuntimeError` failures still propagate normally.

The fallback reuses the existing object selection, survival, translation,
DINO, flow, and score formula, but cannot claim camera removal. Its complete
report therefore carries mandatory fallback-only markers
`coherence_degraded: true` and `camera_compensated: false`, a concrete
`degradation_reason`, null camera diagnostics and camera-compensated spans, and
`models.camera_compensation = "screen-space-only-v1"`. For extreme churn, the
fallback alone may lower the persistence eligibility floor to two observed
frames and reports the resulting low survival; the compensated path retains
the original 0.5 floor and report shape.

Two red-first synthetics pin both ordinary camera-fit failure and the
low-survival dissolve case. The final scorer suite is **14/14**, and the real
11-clip CoTracker3+DINOv2 P0 replay remains exact, including the absence of
fallback markers from every successful compensated report. A degraded score
is diagnostic screen-space evidence, never interchangeable with the accepted
compensated score.

## Bounded optional proposal warps (2026-07-21)

Real Fisher-normalized batch scoring exposed an OpenCV runtime trap rather than
an ordinary slow clip. A projective horizon in a fitted background homography
can map output pixels arbitrarily far outside the source; with
`BORDER_REFLECT101`, `warpPerspective` may spend effectively unbounded time
reflecting those coordinates. Repeated samples caught one batch inside the same
native call, and a controlled near-singular transform reproduced it.

The motion-proposal path now checks the complete rectangular mapping before
warping: corner denominators must remain finite, bounded away from zero, and
one-signed, and projected corner coordinates must stay within eight frame
spans. For a linear-fractional homography, one-signed denominators make the
corner extrema sufficient over the rectangle. If any sampled transform is
unsafe, the complete optional proposal is discarded and the existing
camera-compensated excursion selector runs instead. Keeping a partial proposal
from only the safe frames is forbidden: it selected the wrong foreground and
changed normalized vehicle step 10 from its independent 1.795996 baseline to
0.040983.

Both the direct unsafe call and the biased-partial-proposal path are red-first
regressions. The corrected step-10 full report is byte-identical to the stored
spot result, the accepted pilot vehicle report is unchanged, and the fast
measurement suite passes 56 tests (four learned opt-ins skipped).
