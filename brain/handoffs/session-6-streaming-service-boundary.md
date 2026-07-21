---
type: handoff
status: active
session: 6
date: 2026-07-19
description: "Implemented and adversarially hardened a dependency-free fake/dev streaming boundary with exact short releases, bounded payload memory, client fencing, hard async deadlines, and fail-closed backend poisoning."
branch: main
key_commits: []
prior_handoff: "session-5-gemini-calibration-and-runner-preflight"
---

# Session 6 Handoff — Streaming service boundary

## TL;DR

- `bench/streaming_service.py` now provides a dependency-free fake/dev service
  core for one startup frame plus twenty four-frame releases: exactly 21 chunks
  and 81 RGB frames for the short job.
- It derives indices and server timestamps, applies credits before backend
  generation, does not retain delivered payloads, serializes a non-reentrant
  backend globally, and fences client replacement/cancellation by `job_id`.
- Hard async deadlines isolate emitter, backend-next, and backend-close tasks.
  An uncooperative transport remains isolated to its client; an uncooperative
  backend poisons the global gate so unsafe cache/stream state is never reused.
- Two independent actual-file reviews reproduced several real races; each is
  now a red-first regression. Both reviewers report no unresolved concrete
  blocker within the stated cooperative-async/process boundary.
- This is not near-real-time generation yet: there is no corrected CUDA
  adapter, real WebSocket/Socket.IO server, browser ACK, or GPU evidence.

## New implementation

- `bench/streaming_service.py`
- `bench/tests/test_streaming_service.py`
- `scripts/streaming-demo-smoke`
- [[Streaming-Service-Boundary]]
- [[Gotcha-Async-Timeouts-Need-Task-Isolation]]

The fake smoke emits targeted NDJSON events without payload bodies. Unit tests
exercise the actual base64 payload serialization and round trip.

## Contract now enforced

1. A short backend yields immutable CPU-owned batches shaped `[1, 4 × 20]`.
   The service—not the backend—assigns chunk and first-frame indices.
2. Readiness is stamped immediately after each async backend yield and must be
   strictly increasing. Cadence is chunk-based; no per-frame time is invented.
3. Bounded credits are acquired before requesting another decoded chunk and
   released only after emission completes.
4. A registry provides client-scoped replacement/cancellation, live job-ID
   uniqueness, per-client emitter serialization, and global backend
   serialization for non-reentrant CUDA state.
5. Emitter and backend exceptions are sanitized. Fire-and-forget and detached
   task outcomes are retrieved.
6. Backend chunk/cleanup timeouts poison the registry gate; future starts fail
   synchronously until the worker is reconstructed.
7. First-ready and wall-e2e metrics are server-side only. The future UI must
   fence the latest `job_id` and send a presentation ACK before claiming
   first-visible latency.

## Adversarial review fixes

The reviews caught and the focused tests now cover:

- backend-internal `CancelledError` and cancellation-suppressing `__anext__`;
- cancellation-suppressing emitter and `aclose` calls;
- immediate cancel leaving a stale registry entry;
- same-client emitter overlap after replacement;
- hung emitter blocking an unrelated client's backend work;
- stale replacement terminals and cleanup descriptor failure;
- payload retention, over-limit rollout acceptance, exception leakage, and
  child-emitter cancellation misclassification.

The durable lesson is [[Gotcha-Async-Timeouts-Need-Task-Isolation]]. Python
cannot hard-stop synchronous event-loop work, so the real CUDA adapter must use
a worker thread/process and yield only synchronized CPU-owned bytes.

## Verification

- Focused streaming suite: **31/31**.
- Complete local suite: **129/129**.
- Python compilation: pass.
- JSON schemas: all **15** parse.
- Draft protocol validator: `valid: cf1-quality-repair-v1 (draft)`.
- Fake NDJSON smoke: pass via its end-to-end test.
- No GPU or external API was used, and no commit was created.

## Next actions

1. Preserve the corrected CF++1 21-block generator locally and adapt its
   dynamic rolling-TAEHV decoder to `StreamingBackend` behind worker/process
   isolation. Synchronize CUDA before yielding CPU-owned chunks.
2. Add the smallest real per-client WebSocket/Socket.IO adapter plus client
   `job_id` fencing and a presentation acknowledgement.
3. Keep the 241-latent sustained run blocked until a compatible bounded
   long-horizon attention/cache policy is registered.
4. In parallel, collect the preregistered nine-video/three-human Gemini
   calibration. No evaluator or GPU gate changes follow from this service work.

[[State]] remains the live authority; the prior evidence/calibration context is
in [[session-5-gemini-calibration-and-runner-preflight]].
