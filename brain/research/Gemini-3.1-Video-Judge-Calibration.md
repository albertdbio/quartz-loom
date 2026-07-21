---
type: research
status: active
date: 2026-07-19
description: "Gemini 3.1 Pro video judging is transport-verified at an explicit 16 fps with two repeatable exact-byte passes, but it remains calibration-pending until a preregistered nine-video human-anchored study validates the absolute 7/10 decision boundary."
source_url: "https://ai.google.dev/gemini-api/docs/generate-content/video-understanding"
anchors: ["bench/gemini_video_judge.py#build_gemini_request", "bench/gemini_video_judge.py#parse_gemini_response", "bench/gemini_video_judge.py#run_gemini_plan", "bench/gemini_video_judge.py#ratings_from_gemini_evidence"]
related: ["[[Research]]", "[[State]]", "[[Pegasus-1.5-Video-Judge-Calibration]]", "[[ADR-001-quality-qualified-headline]]", "[[session-5-gemini-calibration-and-runner-preflight]]"]
---

# Gemini 3.1 Pro video-judge calibration

## Bottom line

The Gemini integration is **transport-verified and calibration-pending**. It is
not yet `quality-qualified`.

The strict adapter uploads one MP4 per request, explicitly samples the original
16 fps, requests the registered structured rating at temperature 0, preserves
canonical scrubbed safety/usage metadata, and can replay provider responses
through the full raw-evidence chain. Full-plan execution still refuses the
family until the protocol readiness is `quality-qualified`.

This corrects an important weakness in the archived Gemini CLI evaluation. The
archived `@video.mp4` calls did not set `videoMetadata.fps`; Google's documented
default is 1 fps. Each 5.0625-second, 81-frame clip was therefore represented by
only about five or six sampled frames. Those ratings remain historical signal,
not a full-motion calibration anchor.

## Exact-byte two-pass result

The new calls used the exact three media hashes submitted to the earlier
Pegasus canaries:

| System | Media SHA-256 | Pass 1 | Pass 2 | Two-pass mean |
|---|---|---:|---:|---:|
| CF1 rolling TAEHV | `6c776b72abfd9b5f68cfd58a3881ae95388ae4386ff27e7a20178ed2ec6abda3` | 5.4 | 6.0 | 5.7 |
| CF2 rolling TAEHV | `654664691ab61e64324ce807a8c4a98b153496000108b0491efef5038441f9c7` | 3.8 | 4.4 | 4.1 |
| SF4 Wan | `828795e408b1119f1d73554d8bf430f24a068f6a0845a9873101c18078645bf3` | 5.6 | 5.6 | 5.6 |

Repeatability across the 30 paired core-dimension cells is useful but not by
itself a qualification result:

- mean absolute pass-to-pass error: 0.40;
- maximum cell difference: 1 point;
- 60% exact and 100% within one point;
- per-video mean maximum difference: 0.60;
- only the practically tied CF1/SF4 order flips; CF2 remains last in both
  passes.

The diagnoses are coherent across passes. CF1 is penalized for stiff/sliding
motion and walking instead of running. CF2 is penalized for morphing,
anatomical distortion, and late collapse. SF4 retains spatial/identity quality
but is penalized for a largely static subject and missing requested action.

Evidence lives under `bench/results/calibration/gemini31_16fps_quality_*.json`.
The captured scrubbed real response fixture is
`bench/tests/fixtures/google-gemini31-video-response.json`.

## Exact-byte provider disagreement

The old Pegasus comparison against `cf1_p0.mp4`, `cf2_p0.mp4`, and
`sf4_p0.mp4` was not apples-to-apples: those files differ from the three
Pegasus canary bytes. The new Gemini calls remove that ambiguity.

| System | Gemini two-pass mean | Pegasus canary | Pegasus offset |
|---|---:|---:|---:|
| CF1 | 5.7 | 9.0 | +3.3 |
| CF2 | 4.1 | 8.6 | +4.5 |
| SF4 | 5.6 | 8.0 | +2.4 |

The decisive disagreement is not merely rank. Pegasus passes all three at the
absolute 7/10 bar; Gemini passes none. The rationales also disagree on the
critical motion/anatomy failure channels. Therefore Gemini cannot be qualified
by agreement with Pegasus, and Pegasus remains `calibration-failed`.

## Provider and local contract

- Model: `gemini-3.1-pro-preview`.
- Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent`.
- Media: one inline `video/mp4`, explicit `videoMetadata.fps = 16`, complete
  serialized request limited to 20 MB.
- Output: temperature 0, 2,048-token limit, structured JSON rating with exact
  integer fields.
- Acceptance: exactly one candidate, `finishReason = STOP`, exact model
  version, exact response fields, non-blocked prompt/candidate safety metadata,
  valid usage counters, and exact local schema validation.
- Evidence: `bench/schemas/quality-google-model-evidence-v1.schema.json` plus
  request/media/rubric/adapter hashes and canonical scrubbed raw response.
- Replay: `ratings_from_gemini_evidence` must reproduce submitted ratings;
  the adapter SHA must match the executing adapter, the concrete model identity
  must match the registered rater, and caches/reports containing unknown nested
  response fields fail before propagation. The provider-specific evidence
  schema pins the exact embedded structured-output schema document.

Official references:

- [Video understanding and custom FPS](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)
- [GenerateContent API reference](https://ai.google.dev/api/generate-content)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Cost

The six exact-byte calls used 36,684 input tokens, 1,031 visible candidate
tokens, and 4,704 thinking tokens. At the published sub-200k synchronous rates
of $2/M input and $12/M output including thinking, estimated cost was
**$0.142188**. The separate CF1 P0 transport canary adds about **$0.02268**.
Provider responses report tokens, not the final invoice.

## Pre-registered path to qualification

Run nine fixed blinded videos twice and obtain three independent human ratings
for the same hashes. Include clear positive, midrange, and negative anchors plus
human-confirmed static/missing-action, anatomy/morphing, and late-drift cases.
Keep the family pending unless all of these hold:

1. all 18 Gemini responses replay exactly with no missing/swapped/duplicate
   evidence;
2. core-cell repeat MAE at most 0.75, at least 90% within one point, no cell
   more than two points apart, and per-video mean MAE at most 0.50;
3. human-panel ICC at least 0.75, Gemini-versus-human Spearman at least 0.75,
   absolute MAE at most 1.0, and signed bias magnitude at most 0.5;
4. at least 80% pairwise direction agreement for human gaps of at least one
   point, with motion and temporal MAE each at most 1.25;
5. zero false quality passes for human-consensus items at or below 5.5 and at
   least 80% clear-anchor pass/fail agreement around the 7/10 bar.

Two deterministic passes remain one model family, not two independent judges.
Without human anchors spanning the absolute threshold, the study may validate
repeatability/ranking only; it cannot qualify the gate decision.
