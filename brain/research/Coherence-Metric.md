---
type: research
status: active
date: 2026-07-21
description: "An independent DINOv2+RAFT temporal and patch-reference spatial axis now measures whether an 81-frame clip stays stable without treating legitimate transport as incoherence; the calibrated motion-by-coherence batch surface is ready for checkpoint comparison."
anchors: ["bench/coherence_metrics.py", "bench/tests/test_coherence_metrics.py", "bench/fixtures/coherence_calibration.json", "bench/results/coherence-calibration-20260721/table.md", "bench/results/sgmd-motion-coherence-batch-20260721/matrix.md"]
related: ["[[Coherent-Displacement-Metric]]", "[[Displacement-Batch-Comparison]]", "[[State]]", "[[session-26-coherence-axis]]"]
---

# Coherence metric

## Scope and contract

`bench/coherence_metrics.py` is a CLI and library for exact 81-frame,
480×832 clips. It measures stability independently from displacement and emits
a deterministic per-clip report with a 0–10 `coherence_score`, temporal and
spatial component scores, three ordered segment reports, and
`degrades_over_time`. It neither gives credit for raw motion nor applies a
displacement penalty.

The default learned path uses DINOv2 ViT-S/14 and TorchVision RAFT Small
`C_T_V2`. DINO encodes all 81 frames. RAFT estimates all 80 adjacent flows at
416×240; frame `i+1` is sampled at `x + flow(i→i+1)` and compared with frame
`i` over in-bounds pixels. Raw flow magnitude remains diagnostic only.

## Components

Temporal coherence combines two adjacent-pair guards:

- DINO cosine similarity uses the median and p10 across all adjacent pairs,
  with smoothstep ranges 0.65–0.95 and 0.55–0.92.
- Flow-warp RGB residual uses the median and p90, with inverse smoothstep
  ranges 0.04–0.18 and 0.07–0.24.

Their geometric combination stays high for content that translates sharply
when RAFT can align it, while flicker or dissolution lowers one or both guards.

Spatial integrity is reference-relative and position-invariant. Each native
frame is divided into non-overlapping 32×32 luma patches. Per patch it measures:

- `high_frequency_energy_ratio × sqrt(high_band_spectral_flatness)` as the
  pointillist index; and
- squared energy removed by a sigma-1 Gaussian blur as structure energy.

Each frame retains both a bulk and localized tail statistic: pointillist q75
and q95, structure q75 and q99. The reference is the median of frames 0–4.
Across a scoring window, pointillist bulk-median and tail-p90 penalize excess
texture relative to the opening; structure bulk/tail median and p10 penalize
both loss (blur/disappearance) and 2–4× excess edge energy (pointillist shards).
Spatial quantiles ignore *where* a sharp object is, so transport across the
frame does not lower the score, while the 5%/1% tails preserve narrow corrupt
bands and small foregrounds that a q75-only summary discarded.

A large coherent object entering the frame may legitimately increase those
structure statistics by more than 4×. The excess-energy exemption therefore
requires the **maximum** adjacent flow-warp residual to be at most 0.04. Overall
scoring uses the complete clip; each trajectory segment uses the causal prefix
through that segment's end. This all-pairs condition is deliberately stricter
than the temporal median/p90 score: one catastrophic transition into a stable
texture soup cannot disappear from a later segment merely because the damaged
frames are mutually static afterward, and a future failure cannot retroactively
poison an earlier clean segment.

Segments are exactly frames 0–20, 20–50, and 50–80; their temporal pair slices
are 0–19, 20–49, and 50–79. The overall score is computed directly from all
frames/pairs rather than averaging segment scores. `degrades_over_time` is true
only when late falls by at least 2.0 points and to at most 75% of early.

## Red-first evidence and motion discriminator

The four required synthetics first failed against an intentional unimplemented
surface, then passed after the smallest scorer implementation: static is high,
a fast sharp square is high, progressive pixel replacement is low with visible
segment decay, and uniform flicker is temporally low while spatially intact.
A checkerboard-to-blur regression then failed at 9.578542 before the structure-
retention guard was added.

The first actual-file review found a real design defect: q75 plus one-sided
guards assigned perfect spatial integrity to high-energy/localized pointillist
failures. A new 20%-width pointillist-band synthetic failed with late spatial
score 10.0 before the bulk/tail, energy-ratio, and two-sided reference fix; it
now passes. This was a correction of a visibly missed failure mode, not an
ordering-tuning loop.

A second review found that the first flow-aligned exemption used the aggregate
median/p90 guard and could erase one catastrophic replacement among 80 pairs.
The actual SGMD barrel counterexample—40 repeats of its clean opening followed
by 41 repeats of its known pointillist final frame—failed at 10/10/10 before
the maximum-residual gate. It now scores low spatially with late
decay, while an exact-flow panorama in which a sharp 72-pixel object genuinely
enters still remains high. A repository-local deterministic stripe replacement
pins the same discriminator in the default synthetic suite.

The final review then caught a trajectory leak: passing the full future maximum
to every segment turned a clip with corruption beginning at frame 70 from
`10/10/10` into `10/0/0`, lowering the clean middle segment. That regression
was watched red before segment evidence became a causal prefix; it now retains
high early/middle scores and lowers only late, while overall still sees the
complete clip.

Motion invariance has two stronger checks beyond the small moving square:

- a full-frame translating texture with an exact 4-pixel flow backend remains
  high while its nonzero raw flow and sub-0.01 aligned residual are asserted;
- the same 81-frame, 480×832 translation through real DINOv2+RAFT scores
  **10.000000 temporal / 10.000000 spatial / 10.000000 overall**, with mean
  half-resolution RAFT magnitude 1.630644 px and warp residual 0.016243.

## Real calibration

The read-only calibration contains 24 fork-grid clips plus SGMD update-25 ball
and vehicle. Spatial thresholds were frozen after the red pointillist repair;
the complete 26-clip learned-backend run was then executed once and was not
iterated to force its expected ordering.

| Condition | Expected tier | Mean | Min | Max | Degrades |
| --- | --- | ---: | ---: | ---: | ---: |
| final_1step | most coherent | 9.631239 | 8.783284 | 10.000000 | 0/4 |
| final_2step | most coherent | 9.961025 | 9.844100 | 10.000000 | 0/4 |
| cd_4step | mid | 7.302360 | 0.000000 | 10.000000 | 1/4 |
| ode_4step | mid | 4.044224 | 0.000000 | 10.000000 | 2/4 |
| cd_1step | low | 1.938434 | 0.000000 | 7.753737 | 2/4 |
| ode_1step | low | 0.000000 | 0.000000 | 0.000000 | 1/4 |

The preregistered group ordering reproduces:
**finals 9.796132 > four-step 5.673292 > one-step 0.969217**. Strict
condition-tier separation also passes. The SGMD ball is correctly catastrophic
at 0.000000. The SGMD vehicle is mid-high at **7.553927**, with clean-to-decayed
segments **10.000000 → 8.763047 → 6.935005** and
`degrades_over_time=true`. The complete per-clip table and reports are under
`bench/results/coherence-calibration-20260721/`; an opt-in real-backend test
re-scores and exactly reproduces all 26 fixture rows.

## Two-axis batch surface

`bench/displacement_batch.py --with-coherence` creates one DINOv2 and one RAFT
backend for the batch, retains the complete coherence report separately under
`coherence/<condition>/<clip>.json`, projects both motion and coherence into
schema 4, and renders both values per cell. A coherence-domain error is isolated
from a valid displacement result; unexpected errors still abort before output
replacement. Without the flag, schema 3 and the complete legacy output tree are
byte-identical and no coherence backend is touched.

The refreshed SGMD 4×4 matrix is a concrete operating-plane example. Its
update-25 vehicle is `(motion 2.030809, coherence 7.553927)`; later vehicle
updates move to `(0.352059, 10.000000)`, `(0.061183, 9.997474)`, and
`(0.227143, 9.920312)`. All 16 displacement reports are byte-identical to the
accepted SGMD batch, while all 16 coherence reports are new independent axis
evidence.

## Interpretation limits

- The calibration is 26 clips from two related experiment families, not a
  broad human-rated coherence corpus.
- The pointillist q95 and structure q99 tails have finite spatial detection
  floors; very small corruption below those patch fractions may be missed.
- Spatial normalization is relative to frames 0–4. A clip already degraded at
  its opening has no clean internal reference and may rely on the temporal axis.
- Catastrophic failures intentionally hit hard guard floors: 18/26 calibration
  rows are exactly 0 or 10. The metric separates clean, mixed, and collapsed
  condition groups, but should not be used to finely rank different kinds of
  already-collapsed texture soup.
- `degrades_over_time=false` can mean the clip was already bad inside the first
  0–20 segment; consult the three segment scores rather than reading false as
  “healthy.”
- Coherence does not establish prompt fidelity, physical plausibility, or
  useful motion. Promotion should seek the top-right motion×coherence direction
  and still retain frame/blind review.
