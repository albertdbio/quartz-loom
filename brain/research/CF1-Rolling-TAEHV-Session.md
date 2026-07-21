---
type: research
status: active
date: 2026-07-20
description: "The frozen CF++1 runtime's fail-closed rolling-TAEHV decoder has completed the real 21-chunk/81-PNG smoke, two same-worker jobs, forced-death poison/reap, and genuine browser presentation with exact stream/event, trim, D2H, raster, completion, and ACK semantics."
anchors: ["bench/cf_cuda_session.py#RollingTaehvChunkDecoder", "bench/cf_cuda_session.py#_require_verified_runtime", "bench/cf_cuda_generator.py#CF1LatentPullSession", "bench/cf_cuda_smoke.py#run_cf1_cuda_smoke", "bench/cf_cuda_adapter.py#validate_cf1_runtime_provenance", "bench/generation_preflight.py#rolling_taehv_trim_frames"]
related: ["[[Research]]", "[[State]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[Streaming-Service-Boundary]]", "[[Gotcha-Rolling-TAEHV-Context-Trim]]", "[[session-10-rolling-cuda-session-core]]", "[[session-11-runtime-preflight]]", "[[session-12-latent-pull-and-smoke]]"]
---

# CF++1 rolling TAEHV session core

## Bottom line

`bench/cf_cuda_session.py` is the local, production-shaped decode core for one
short CF++1 pull session. It accepts one exact bfloat16 denoised latent plus its
producer CUDA event, performs corrected rolling TAEHV decode on an owned CUDA
stream, blocks until postprocessing is complete, copies exact uint8 rasters to
CPU, and returns immutable encoded payloads only after strict validation.

The contract was first developed and mutation-tested with fake Torch/CUDA
objects, then executed behind the frozen authorizer on the dedicated H100. The
exact per-block generator and pinned torchvision PNG encoder completed all 21
chunks and 81 decodable 832×480 images; manifest SHA is `d363fbe1…`. The same
decoder path then completed two distinct-seed persistent-worker jobs on one
PID/instance before forced death proved poison/reap. This establishes the real
model/event/D2H/encoder and worker boundaries. It does not establish browser
paint, performance, quality, or sustained operation.

## Runtime identity is a prerequisite, not metadata

Construction accepts only a verified `CF1Runtime`. It recomputes and validates:

- exact stack ID, upstream source commit, asset-lock SHA-256, and effective
  configuration SHA-256;
- exact runtime-lock and stable environment identities, tokenizer sentinel,
  and the actual attention backend selected after upstream import;
- the complete ordered asset inventory with exact paths, byte counts, and
  hashes;
- the bootstrap identity derived from those fields; and
- a canonical guard-bundle manifest digest over
  `cf_cuda_adapter.py`, `cf_cuda_session.py`, `generation_preflight.py`,
  `model_asset_preflight.py`, and `streaming_service.py`.

An arbitrary object with look-alike fields, a forged asset observation, a
changed guard module, an unrelated Torch module, a desired backend copied from
the lock instead of the actual import, or anything other than logical
`cuda:0` is refused before decoder state is created. The source verifier still
has a verify-to-path-reopen window
against a concurrent hostile local writer. That actor is outside the stated
trusted, no-concurrent-writer worker threat model; the boundary must be revisited
if model bytes move to a remote, untrusted, or multi-tenant source.

## Exact rolling contract

Each input must have shape `1×1×16×60×104`, reside on the runtime's explicit
CUDA device, and be exactly bfloat16. The decoder converts it to float16 before
TAEHV, preserves the most recent three latents in chronological order, and
uses the canonical dynamic trims:

- block 0: no context, trim 3 decoded RGB frames, release 1;
- block 1: one context latent, trim 4, release 4;
- block 2: two context latents, trim 8, release 4; and
- blocks 3–20: three context latents, trim 12, release 4 each.

Exactly 21 successful calls therefore yield `[1, 4 × 20]`, or 81 frames. A
22nd call is refused. `finish()` succeeds only after all 21 blocks and releases
the retained tail; incomplete or poisoned sessions cannot finish.

## CUDA ownership and the CPU boundary

The decoder owns an explicit stream on the runtime device. For every block it:

1. requires and waits on the producer's `latent_ready_event`;
2. calls `record_stream` on the producer tensor so allocator ownership follows
   the decode stream;
3. performs float16 TAEHV decode, trimming, clamp/scale/round, and uint8
   conversion on that stream;
4. records a blocking CUDA event after postprocessing and synchronizes it; then
5. performs D2H, verifies CPU ownership plus exact uint8 shape, and calls the
   raster encoder.

This ordering prevents a `DecodedChunk` from claiming readiness while pixels
are still CUDA-owned or while an asynchronous decode/postprocess is unfinished.
The frozen one-block/full smokes and accepted worker now prove the real
producer/decode event behavior for this bounded path; they do not measure or
authorize an overlap-performance claim.

## Encoded payload and failure contract

The encoder must return exactly one payload for block zero and four thereafter.
Every payload is copied into an immutable tuple of nonempty `bytes`, must match
the declared JPEG, PNG, or WebP magic/container boundary, and the aggregate
must stay within the configured chunk-byte limit. The source CPU tensor is
constrained to `3×480×832` uint8 per frame. The full H100 smoke and persistent
acceptance now independently validate PNG decodability and exact 832×480 RGB8
dimensions at the outer boundaries as well.

Any `BaseException` after decode begins poisons the decoder permanently, while
interrupts are re-raised unchanged. State is committed only after successful
D2H and encoding. The persistent worker now translates that poison into
synchronous process-group retirement and bounded reap rather than attempting
warm reuse; the acceptance manifest proves the forced-death path and
post-poison rejection.

## Verification and review

The increment was developed red-first: the session test first failed because
the module did not exist, and every later actionable mocked-core review gap was
made to fail before its fix. Final local validation is:

- complete Python: **278/278**;
- persistent process: **35/35**;
- process + service + NDJSON + WebSocket: **113/113** (**78/78** without
  process);
- asset preflight: **13/13**; runtime preflight: **17/17**; CUDA adapter:
  **21/21**; CUDA session: **9/9**;
- executable Node browser client: **4/4**; and
- Python `compileall` plus every repository JSON parse: green.

The real asset gate remains `ready: true`: the exact source and lock pins are
unchanged, all 11 assets verify, and there are zero unexpected source paths.

Consensus parent `ses_080ea9c12ffejex9NyF4Yi7sk0` timed out without a bridge
synthesis. Direct recovery retained:

- Fable child `ses_080e6c4cbffewgbBPMTGzWaxMV`, concrete
  `claude-fable-5`: GO-WITH-FIXES, confidence 0.78;
- Grok child `ses_080e6c4c9ffeiwnSU0HNzAeM3r`, concrete `grok-4.5`:
  GO-WITH-FIXES, confidence 0.82; and
- Kimi child `ses_080e6c4caffeh0tWtS1ApYcJPr`: stalled with zero tokens.

Fable completed, so the tool correctly did not fall back to Opus. A separate
local Claude CLI review requested Opus, returned GO-WITH-FIXES, and
mutation-tested the new decoder; every reproducible actionable mocked-core gap
was fixed and pinned. The CLI did not expose a concrete model version. This is
strong partial and fallback review signal, not a completed three-voter
consensus and not a project GO.

## Completed H100 and browser proof

The frozen authorizer passed unchanged. One-block and full 21-block smokes then
passed, followed by accepted worker manifest `30bfe91b…`: two seeds, the same
PID/instance, 21 chunks and 81 frames each, distinct output hashes, then an
awaited identity-fenced `SIGKILL`, backend/registry poison, worker reap, and
synchronous post-poison rejection. Warm took 517.196 seconds; output timings
are explicitly non-performance-authorized.

The opt-in real CF++1 browser path now also passes: the genuine H100/frozen-
runtime page reached a visible 832×480 canvas, 81 rendered frames, 21 chunks,
21 post-paint ACKs, server completion, and zero console errors. Do not promote
the serial smoke, worker timings, or browser completion into a benchmark or
headline claim.
