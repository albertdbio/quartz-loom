---
type: research
status: active
date: 2026-07-21
description: "A deterministic batch wrapper turns displacement reports into condition-by-prompt experiment matrices with explicit error, invalid, missing, and degraded states; optional schema-4 coherence adds the independent second axis while the disabled path stays byte-identical."
anchors: ["bench/displacement_batch.py", "bench/tests/test_displacement_batch.py", "bench/results/p0-displacement-batch-20260721/matrix.md", "bench/results/sgmd-motion-coherence-batch-20260721/matrix.md"]
related: ["[[Coherent-Displacement-Metric]]", "[[Coherence-Metric]]", "[[State]]", "[[session-22-displacement-batch-comparison]]", "[[session-23-fork-grid-displacement-baseline]]", "[[session-24-sgmd-degraded-displacement]]", "[[session-26-coherence-axis]]"]
---

# Displacement batch comparison

## Contract

`bench/displacement_batch.py` consumes a direct
`<root>/<condition>/<clip>.mp4` tree plus a one-name-per-line prompt file. It
calls the accepted `score_clip` function unchanged and constructs the default
CoTracker3 and DINOv2 backends once per batch. Outputs are:

- complete scorer JSON for every present clip under `clips/<condition>/`;
- deterministic `report.json` with a condition-by-prompt matrix containing
  score, span fraction, survival, and displaced boolean;
- `matrix.md` with the same cells and visible `MISSING` values; and
- optional `--compare A B` rows where every delta is `B - A`, each prompt has
  a verdict, and the mean uses comparable prompts only.

Projected floats are rounded to six decimal places with negative zero
normalized. JSON keys and inventories are sorted, and payloads contain no
timestamps. A completed rerun replaces only the tool-owned `output/clips`
subtree so deleted or renamed inputs cannot leave stale per-clip reports.
Choosing the input root itself as the output is rejected.

## Prompt slots and holes

When the union of filenames has exactly the prompt count and at least one
condition supplies the complete filename set, sorted filenames are stable slot
identities. This is the mode that can identify a missing middle clip without
shifting later prompts. Otherwise each condition uses its own sorted positional
order; missing trailing positions are explicit holes with an unknown filename.
No condition may contain more clips than prompt names, and nested MP4s are
rejected rather than silently ignored.

This distinction is deliberate: heterogeneous filenames do not contain enough
information to infer a missing middle position. The report records
`slot_mode` so consumers can see which mapping was used.

## P0 reproduction and example

The archived P0 tree is nested and remained read-only, so the integration test
creates a temporary one-level symlink view with canonical filenames
`00-ball.mp4` through `03-vehicle.mp4`. Five conditions yield 11 scored clips
and nine explicit decoder/prompt holes. The final real default-backend run
reproduces all session-21 fixture scores, span fractions, and displaced
decisions exactly.

The durable example is
`bench/results/p0-displacement-batch-20260721/`. Ring ON minus ring OFF is:

| Prompt | OFF | ON | ON - OFF | Verdict |
|---|---:|---:|---:|---|
| ball | 3.889719 | 0.000000 | -3.889719 | OFF higher |
| walker | 0.050011 | 0.004881 | -0.045130 | OFF higher |
| rolling-object | 0.211627 | 0.000000 | -0.211627 | OFF higher |
| vehicle | 0.000000 | 0.071057 | 0.071057 | ON higher |

Mean `ON - OFF` is **-1.018855** across four comparable prompts. This is a
batch-comparison example, not a new calibration or quality claim.

## Validation and review

The fast batch suite has ten green tests. Required missing-middle behavior was
observed red before implementation. Review-driven red-first cases also pin
distinct singleton filenames, Markdown names, nested uppercase MP4 rejection,
stale output pruning, and output-root refusal. The displacement scorer suite
remains 11/11 green, and the opt-in learned P0 regression passes all 11 clips.

The actual-file CLI panel was partial: Kimi returned a review, Claude Opus 4.8
timed out with useful reasoning recoverable from
`ses_07b9e75c3ffeKA2ElSp4Ouq1BQ`, and Grok 4.5 failed for provider balance.
Kimi and recovered Opus reasoning accepted the core contract and identified
the positional/Markdown and uppercase nested-file gaps that are now fixed. A
separate local actual-file reviewer found stale-output and output-root defects;
both reproduced red and are fixed. This is independent review signal, not a
three-lineage consensus.

## Fork-grid baseline (2026-07-21)

The real decode grid at
`/Users/electric/Documents/areas_of_focus/decart-research/fork-grid-20260721/grid_out/`
remained read-only. The complete outputs live in its new sibling
`batch_scores/`. Filename substrings map the full prompt text to the canonical
order `ball`, `barrel`, `walker`, `vehicle`; this avoids pretending lexical
filename order is prompt order. The wrapper now also accepts repeatable
`--compare` pairs, records metric-domain failures as explicit `ERROR` cells,
and can mark a whole incompatible condition `INVALID` without invoking the
scorer. Unexpected I/O/program failures still abort rather than becoming data.

The 7-condition × 4-prompt inventory has no holes: **17 scored**, **7 ERROR**,
and **4 INVALID** cells. All four `oneforcing_1step` clips are INVALID because
that checkpoint needs its native fork/register-head inference path; they were
never scored and enter no comparison. The seven metric errors are preserved as
null metrics with their exact domain reason rather than zero-filled: two in
`cd_1step`, one in `final_1step`, three in `ode_1step`, and one in
`ode_4step`.

The required comparisons are:

| A | B | Comparable / error | Mean B - A | Verdict |
|---|---|---:|---:|---|
| final_1step | cd_4step | 3 / 1 | +0.234997 | cd_4step higher |
| final_1step | ode_4step | 2 / 2 | +1.305990 | ode_4step higher |
| cd_4step | ode_4step | 3 / 1 | +0.310139 | ode_4step higher |

The spot-check reproduces exactly: vehicle is **1.232953** for `cd_4step`,
**2.128855** for `ode_4step`, and **0.286952** for `final_1step`. Thus the
requested native four-step Stage-2 checkpoints beat `final_1step` on the
comparable-prompt means. This is the usable before-baseline for the next
motion-preserving Stage-3 training comparison.

The stronger expected statement—every pre-DMD condition beats both finals—does
**not** reproduce and must not be cited from this matrix. The forced one-step
Stage-2 conditions have only two and one scored cells, respectively, while
`final_2step` scores **8.133922** on ball and **3.967300** on barrel. Frame
inspection shows why this is not the requested action: the final-two-step ball
has a large one-time lateral recenter but no bounce, and the barrel has slow
drift/pan. The frozen metric measures coherent displacement, including a
transient reposition; it does not prove repeated prompt-faithful action. No
threshold or scoring logic was tuned to make the result agree with the
experiment hypothesis.

Final validation is **16/16** fast batch tests, **11/11** scorer tests, and the
real learned-backend 11-clip P0 exactness replay. An actual-file Claude Opus
4.8 voter returned PASS and independently recomputed all means; Kimi and Grok
failed for provider balance, so that panel is one advisory vote, not consensus.
A separate local adversarial review found the overly broad `OSError` downgrade;
its propagation/abort behavior was watched red and then fixed.

## Error contract, degraded cells, and SGMD pilot (2026-07-21)

Batch schema 3 closes the continuation contract without turning failures into
motion scores. Only `DisplacementMetricError` is caught per clip; its row has
`status: "error"`, deterministic `error_type`, and the verbatim domain message.
Errored prompts have null deltas, increment `error_prompt_count`, and never
enter the comparable-only mean. If none remain, Markdown says `UNAVAILABLE`.
Unexpected `OSError` and `RuntimeError` abort the batch before the completed
output tree is replaced. A success-to-error rerun also removes the obsolete
owned per-clip JSON, so the explicit error row cannot coexist with stale score
evidence.

Successful fallback reports remain scored rows but carry uniform projected
`coherence_degraded` and `camera_compensated` flags. Markdown appends `*` only
to their score. Comparisons exclude them by default and report unavailable
degraded rows; `--include-degraded` is the explicit diagnostic opt-in and
preserves provenance in each verdict. This makes coverage loss visible while
preventing a screen-space-only number from entering a compensated mean.

The read-only SGMD pilot at
`/Users/electric/Documents/areas_of_focus/decart-research/sgmd-pilot-20260721/clips/`
was scored into its new sibling `batch_scores/`. The 4-condition × 4-prompt
matrix has **16 results, 0 ERROR, 0 holes, 8 compensated, and 8 explicitly
degraded** cells. The compensated vehicle column exactly reproduces the
independent reports:

| Condition | Score | span/W | Survival | Decision |
|---|---:|---:|---:|---|
| step_000025 | 2.030809 | 0.699644 | 0.586981 | collapsed |
| step_000050 | 0.352059 | 0.123607 | 0.572310 | collapsed |
| step_000075 | 0.061183 | 0.058446 | 0.566779 | collapsed |
| step_000100 | 0.227143 | 0.095573 | 0.619753 | collapsed |

This supports the working conclusion only in a relative sense: update 25
preserves far more vehicle transport than later SGMD updates and is about
7.08× the `final_1step` fork-grid vehicle floor of 0.286952. It still does not
pass the metric's absolute `displaced` decision. Eight of the twelve
non-vehicle cells are degraded; the four compensated non-vehicle scores are
0.000000, 0.006845, 0.097570, and 0.391858. High degraded walker scores are
screen-space diagnostics of unstable/dissolving output, not evidence of usable
prompt-faithful motion. No threshold or scorer default was tuned to match the
training hypothesis.

Final validation is **20/20** fast batch tests, **14/14** scorer tests, and the
real learned-backend 11-clip P0 exactness replay. The new red-first coverage
pins error exclusion and means, success-to-error stale pruning, unexpected
exception aborts, explicit fallback marking, low-survival churn, default
degraded exclusion, and the opt-in path. The durable SGMD artifacts are
`batch_scores/report.json`, `batch_scores/matrix.md`, and the 16 complete
per-clip reports.

The final actual-file CLI review was partial: Claude Opus 4.8 returned
**NO FINDINGS** and independently verified all five contracts plus the exact
SGMD values; Kimi and Grok failed for provider balance. A separate local
actual-file reviewer also found no issue and byte-compared all eight existing
SGMD spot reports to the new per-clip outputs. This is one external advisory
plus local review, not a completed three-lineage consensus.

## Optional coherence axis (2026-07-21)

`--with-coherence` adds the independent scorer from [[Coherence-Metric]] without
changing displacement behavior. Enabled output is schema 4 and projects
`coherence_score`, temporal score, spatial score, and
`degrades_over_time`; each complete coherence report is retained separately
under `coherence/<condition>/<clip>.json`. One shared DINOv2 instance reaches
both scorers where applicable and one RAFT Small instance serves every
coherence call. A `CoherenceMetricError` marks only that axis `ERROR`, while an
unexpected exception aborts before replacing completed outputs.

The disabled path stays schema 3 and is pinned as a complete byte-for-byte
output-tree comparison: no false/null coherence keys, no coherence directory,
and no coherence backend construction. Enabled reruns are deterministic and
prune stale axis JSON. The batch suite is now **25/25**.

The repository-owned SGMD example at
`bench/results/sgmd-motion-coherence-batch-20260721/` contains all 16 motion and
coherence pairs. Its 16 displacement reports are byte-identical to the accepted
read-only sibling `batch_scores/`; the update-25 vehicle cell is
`(2.030809, 7.553927)`, and the coherence trajectory exposes the late spatial
break that a motion-only table could not represent. The `*` marker still means
camera-compensation degradation on the displacement axis only; it must not be
misread as coherence status.
