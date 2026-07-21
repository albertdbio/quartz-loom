---
type: handoff
status: active
session: 5
date: 2026-07-19
description: "Verified real Gemini 3.1 Pro video uploads at explicit 16 fps, completed its replayable evidence path and exact-byte two-pass calibration, strengthened both provider caches against nested response poisoning, and added a CPU-only runner preflight that exposes cadence and long-horizon impossibilities."
branch: main
key_commits: []
prior_handoff: "session-4-quality-protocol-hardening"
---

# Session 5 Handoff — Gemini calibration and generation preflight

## TL;DR

- Real Gemini video upload now works through direct `generateContent` with
  `gemini-3.1-pro-preview`, one inline MP4, explicit 16-fps sampling, and strict
  structured output. The first wire attempt exposed a documentation/API enum
  mismatch (`APPLICATION_JSON` is required by the live response-format field);
  the captured real response now pins the working contract.
- Six exact-byte calls (three videos × two passes) are repeatable: CF1 5.4/6.0,
  CF2 3.8/4.4, SF4 5.6/5.6. Gemini is `transport-verified`, not
  `quality-qualified`; see [[Gemini-3.1-Video-Judge-Calibration]].
- The same-byte comparison makes the Pegasus failure stronger: Pegasus scores
  9.0/8.6/8.0 versus Gemini two-pass means 5.7/4.1/5.6, passing all three at the
  absolute bar while Gemini passes none. The old archived-P0 comparison used
  different media bytes and Gemini's default 1-fps sampling.
- Google now has the same full-plan evidence boundary as Pegasus: resumable
  request/media/rubric binding, canonical safety scrubbing, raw response replay,
  common-gate provider dispatch, a provider-specific schema, and a plan CLI that
  refuses unqualified families.
- Adversarial review found and closed a nested cache/report poisoning path in
  both adapters: an unknown secret-bearing provider field plus a recomputed
  response hash can no longer propagate through resume or replay.
- A focused post-hardening review recovered one completed Grok-4.5 final
  (local GO-WITH-FIXES, project NO-GO) before the bridge timeout. Five findings
  reproduced red and are closed: concrete rater identity, exact embedded-schema
  evidence, current-adapter replay binding, local-attention/cache bounds, and
  complete RGB-release coverage. Fable-5 and Kimi-K3 produced no final, so this
  remains partial signal rather than a consensus verdict.
- `bench/generation_preflight.py` and `scripts/generation-preflight` add a
  CPU-only refusal surface. They record honest chunk-release events, enforce
  strict checkpoint keys with a retained key-set hash, bind released RGB-frame
  totals to the rollout, enforce bounded local attention/four distinct
  artifacts, and reject the registered 241-latent sentinel on a fixed
  21-latent global-attention cache.
- The project remains NO-GO for protocol freeze, GPU generation, or headline
  use. No GPU was restarted and no commit was created.

## New implementation

- `bench/gemini_video_judge.py`
- `scripts/quality-gemini-video-judge`
- `bench/tests/test_gemini_video_judge.py`
- `bench/tests/fixtures/google-gemini31-video-response.json`
- `bench/schemas/quality-google-model-evidence-v1.schema.json`
- `bench/generation_preflight.py`
- `scripts/generation-preflight`
- `bench/tests/test_generation_preflight.py`
- `bench/results/calibration/gemini31_16fps_quality_*.json`

The central `validate_raw_evidence_report` now dispatches `google` only through
`ratings_from_gemini_evidence`; edited ratings, request drift, media mismatch,
missing/extra/nested provider fields, version drift, safety blocks, malformed
usage, and poisoned caches fail closed.

## Evidence and accounting

- Exact-byte Gemini repeat MAE across the five core dimensions: 0.40; maximum
  delta: 1; all paired cells within one point.
- Six-call usage: 36,684 input, 1,031 visible candidate, and 4,704 thinking
  tokens. Estimated current synchronous cost: $0.142188.
- Separate CF1-P0 transport canary: about $0.02268.
- Prior Pegasus canaries: about $0.011396.
- Estimated cumulative API calibration spend: about $0.1763. GPU spend remains
  about $11.69 and the RunPod pod remains stopped.

## Runner audit corrections

The recovered H100 runner is still useful executable history, not a valid v1
runner:

1. It measures 21 decoder-chunk readiness events and reports a 37.62ms P95 by
   dividing each four-frame chunk gap by four. That does not prove 81 actual
   frame-ready timestamps. Preserve honest `{ready_ns, frame_count}` releases or
   add a measured presentation scheduler/per-frame decoder.
2. The protocol's 241-latent/961-frame sentinel cannot run with
   `local_attn_size=-1` and the pinned fixed 21-latent cache. Register a
   long-horizon-capable stack or change the sustained contract; do not provision
   a GPU for an impossible plan.
3. No local serving surface combines the corrected 21-block/first-chunk/
   dynamic-trim loop with a client stream. The smallest future demo reuses the
   Self-Forcing Socket.IO shell around the recovered generator core and emits
   batches `[1, 4 × 20]`, with a fake-backend UI test before GPU work.

The source repository typo is also now explicit: GitHub is
`guandeh17/Self-Forcing`; `gdhe17/Self-Forcing` is the Hugging Face checkpoint
namespace.

## Verification

- Full local suite passes **98/98**.
- Python compilation passes for all bench modules and both new CLIs.
- Draft protocol validation passes with Gemini readiness
  `transport-verified`; forced frozen validation still requires every model
  family to be `quality-qualified`.
- All new JSON evidence/schema files parse.
- CodeGraph was synced after the final implementation/docs edits; status is
  up-to-date and a real post-sync query returned the hardened adapter/preflight
  source and call paths.

## Next actions

1. Pre-register the nine-video/two-pass/three-human Gemini calibration thresholds
   in [[Gemini-3.1-Video-Judge-Calibration]], collect human anchors, then either
   qualify or replace Gemini.
2. Do not use Pegasus in gate evidence unless it independently passes the same
   human-anchored calibration.
3. Reconcile the long-horizon stack and cadence semantics in the protocol, then
   extend the CPU preflight to bind real source/config/checkpoint/decoder/Wan
   artifact hashes—and prove the observed checkpoint keys came from those
   bytes—before any CUDA execution.
4. Only after those blockers and a completed diff-fed consensus GO should the
   protocol freeze, the GPU restart, and Round-A/Round-B media generation begin.

[[State]] remains the live authority.
