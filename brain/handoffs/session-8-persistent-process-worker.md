---
type: handoff
status: active
session: 8
date: 2026-07-20
description: "Completed and adversarially hardened the persistent fake process-worker boundary; the real CUDA-owning model adapter remains absent."
branch: main
key_commits: []
prior_handoff: "session-7-browser-streaming-transport"
---

# Session 8 Handoff — Persistent fake process worker

## TL;DR

- `bench/streaming_process.py` and `bench/streaming_process_worker.py` now prove
  the persistent subprocess, pull-IPC, fencing, kill/reap, warm-reuse, and
  terminal-evidence seam against fake PNGs.
- Clean jobs reuse one PID and worker instance. Cancellation, close, corruption,
  timeout, malformed payloads, and incomplete terminal state cannot silently
  return the child to the warm pool.
- This is not model inference: no child owns a GPU, imports CF++, executes the
  rolling TAEHV decoder, synchronizes D2H, or emits production model/config/
  checkpoint provenance. The browser demo still uses `TinyPngStreamingBackend`.
- The process design review completed across Fable, Kimi, and Grok. Both later
  actual-code calls timed out and recovered only Grok finals, so implementation
  review is partial and the project-level gate remains NO-GO.

## What changed

- Added `ProcessStreamingBackend`, `WorkerCompletionEvidence`, strict framed
  message parsing, bounded completion storage, and the isolated stdlib-only fake
  worker.
- Added `BackendFatalError` bridging in `bench/streaming_service.py` so a fatal
  process failure poisons the registry rather than allowing unsafe backend reuse.
- Added 32 persistent-process tests, including real subprocess and
  `run_stream_job` integration coverage.
- Documented the lifecycle ownership contract: every owner must eventually
  await idempotent `close`; a reap timeout detaches I/O but deliberately retains
  the `Popen` handle so close can retry the mandatory kill/reap boundary.

## Protocol and lifecycle invariants

- Spawn uses `subprocess.Popen`, a socketpair, an absolute worker script,
  `python -I`, `close_fds`, an explicit inherited descriptor, stdout disabled,
  and an environment allowlist containing only unbuffered-Python configuration.
- HELLO binds the protocol version, unpredictable worker instance, parent stack
  digest, child-computed worker-code digest, isolated-interpreter status, and the
  absence of sensitive environment-name markers.
- START/STARTED precedes exactly one unpredictable `NEXT` credit at a time.
  CHUNK and COMPLETE must echo that credit and match the active worker, job,
  chunk index, first-frame index, and exact `[1, 4 × 20]` topology.
- Canonical JSON headers are length-prefixed; header, frame, chunk, and aggregate
  sizes are checked before body reads. Every payload is SHA-256 verified.
- Completion evidence recomputes prompt, seed, stack/code identity, topology,
  frame count, chunk count, and all 81 payload hashes. Only validated COMPLETE
  permits warm reuse or evidence publication.
- Active ownership retires synchronously before cancellation/close cleanup.
  The whole worker process group is killed before the first await, the IPC fd is
  non-inheritable by helpers, and bounded reap resists repeated task cancellation.
  Stderr callbacks have a finite drain quantum so logging cannot starve timers.
  Fatal protocol and timeout paths poison; prompt/rollout/lifetime
  capacity errors are bounded request failures. Worker stderr, completion
  evidence, job IDs, and maximum latent frames are all bounded.
- `python -I` and the environment allowlist do not constitute an OS sandbox:
  the worker is trusted code and retains filesystem/network access.

## Review and evidence

- Design parent `ses_0816c4b79ffe4XfZf341UmQ6OP` requested aliases
  `claude-opus,kimi-k3,grok` with gpt-5.6-sol as caller. Concrete Fable-5,
  Kimi-K3, and Grok-4.5 all completed GO-WITH-FIXES. Fable ran directly; no Opus
  fallback was needed.
- Actual-code parent `ses_0815374c0fferEiO9rMjeoPomp` timed out. Session recovery
  retained Grok's NO-GO review; Fable and Kimi had zero output. Reap-time I/O
  detachment, cancellation ownership, and unbounded direct rollout findings
  reproduced red before fixes.
- Smaller post-fix parent `ses_08148b660ffeFUDC26pBOL9o5J` also timed out and
  recovered only Grok, now GO-WITH-FIXES. Worker-side rollout limits, lifetime
  job-ID bounds, and stronger close/reap/evidence-capacity coverage were adopted.
  Fable and Kimi again had zero output; no concrete Claude model or Opus fallback
  metadata was returned.
- Claims arising from concatenated review excerpts or ignoring the enforced
  registry/backend timeout nesting were checked against disk and rejected. The
  fake seam is locally accepted with partial independent review, not a completed
  three-family implementation verdict.

## Verification

- Complete Python 3.9 suite: **215/215**.
- Persistent-process suite: **32/32**.
- Process + service + NDJSON + WebSocket: **110/110**.
- Executable Node browser client: **4/4**.
- Python compilation, all 15 JSON schema parses, and draft protocol validation
  pass.
- No GPU was started, no external API was called, no secret value was printed or
  persisted, spend is unchanged, and no commit was created.

## Likely next moves

1. The upstream checkout is now restored cleanly at
   `8db419e341e5fc52542c0b2c4542728420ddfb4a`; `taew2_1.pth` is restored and
   matches SHA-256 `d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e`.
   Restore and pin the still-missing CF++ generator checkpoint plus required Wan
   text/config/tokenizer assets.
2. Replace only the fake child core with the corrected CF++ plus dynamic-trim
   rolling-TAEHV CUDA adapter. Preserve the tested protocol while adding CUDA
   synchronization, D2H, immutable encoded-raster ownership, and exact production
   provenance.
3. In parallel, complete the preregistered nine-video/three-human Gemini
   calibration and choose a feasible long-horizon attention/cache policy. The
   TwelveLabs upload path is available as a second video-understanding signal,
   but Pegasus remains calibration-failed and cannot enter gate evidence.
4. Redesign the timing-evidence contract for simultaneous chunk releases before
   freeze, then collect real Round-A/Round-B evidence and an exact ≥60-second H100
   run only after evaluator qualification and a completed GO review.

## See also

- [[Handoffs]]
- [[Streaming-Service-Boundary]]
- [[Browser-Streaming-Transport]]
- Prior: [[session-7-browser-streaming-transport]]
