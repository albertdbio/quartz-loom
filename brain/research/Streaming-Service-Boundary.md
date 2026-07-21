---
type: research
status: active
date: 2026-07-20
description: "The fake/dev streaming path enforces exact 1+20×4 releases, bounded payload memory, fencing, hard deadlines, persistent isolated pull IPC, NDJSON conformance, and browser ACKs. A provenance-bound rolling CUDA decoder now satisfies the same local chunk contract through synchronized D2H and immutable rasters, but the real model/GPU worker remains unproved and unwired."
anchors: ["bench/streaming_service.py#run_stream_job", "bench/streaming_service.py#StreamingJobRegistry", "bench/streaming_process.py#ProcessStreamingBackend", "bench/streaming_process_worker.py", "bench/model_asset_preflight.py#verify_model_assets", "bench/cf_cuda_adapter.py#build_cf1_runtime", "bench/cf_cuda_session.py#RollingTaehvChunkDecoder", "bench/streaming_transport.py#NDJSONStreamingServer", "bench/streaming_websocket.py#BrowserStreamingServer"]
related: ["[[State]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Rolling-TAEHV-Session]]", "[[Browser-Streaming-Transport]]", "[[Gotcha-Async-Timeouts-Need-Task-Isolation]]", "[[Gotcha-Transport-Write-Is-Not-Presentation]]", "[[Gotcha-Rolling-TAEHV-Context-Trim]]", "[[session-6-streaming-service-boundary]]", "[[session-7-browser-streaming-transport]]", "[[session-8-persistent-process-worker]]", "[[session-9-pinned-assets-and-cuda-bootstrap]]", "[[session-10-rolling-cuda-session-core]]", "[[session-11-runtime-preflight]]"]
---

# Streaming service boundary

## Bottom line

`bench/streaming_service.py` is a tested local service core for the future demo.
It accepts an async backend that yields immutable CPU-owned frame bytes and an
async client-targeted emitter. The included in-process fake backend, NDJSON
transport, and same-origin binary WebSocket demo make the complete 81-frame job
executable through a real browser without CUDA. The browser path measures a
client assertion after draw plus a following animation-frame presentation
opportunity.

`bench/streaming_process.py` separately proves the persistent-process seam. It
exec-spawns the stdlib-only fake `bench/streaming_process_worker.py` through a
socketpair, grants one unpredictable pull credit at a time, validates every
bounded header and frame digest, and retains hash-bound completion evidence only
after a valid terminal `COMPLETE`. Clean jobs reuse the same PID and worker
instance; cancellation and fatal errors kill/reap and cold-start or poison.

The separate minimal-model gate now returns `ready: true` for exact upstream
commit `8db419e341e5fc52542c0b2c4542728420ddfb4a`, zero unexpected paths, and all
11 registered CF++1/TAEHV/Wan-text assets. `bench/cf_cuda_adapter.py`
structurally constructs that pinned stack while bypassing the unnecessary stock
Wan diffusion and VAE weights. `bench/cf_cuda_session.py` adds the verified
rolling decode core: exact runtime/guard identity, producer-event handoff,
explicit decode-stream ownership, dynamic trims, synchronized D2H, and immutable
bounded raster payloads. See [[Pinned-CF1-CUDA-Bootstrap]] and
[[CF1-Rolling-TAEHV-Session]].

This is still not the generation serving surface. The fake worker does not own
a GPU, the adapter has not loaded the real tensors, and the decoder has run only
against fake Torch/CUDA objects and a fake encoder. The browser demo still
instantiates `TinyPngStreamingBackend`, not `ProcessStreamingBackend`, and that
process backend still starts the stdlib fake worker.

## Exact release and timing contract

- A 21-latent short job must release exactly `[1, 4 × 20]`: 21 chunks and 81
  RGB frames. Equal-total but wrongly shaped schedules fail.
- The service derives `chunk_index` and `first_frame_index`; a backend cannot
  inject indices or timestamps.
- `ready_ns` is stamped immediately after each backend `__anext__` returns and
  must be strictly increasing. No synthetic per-frame timestamps are created.
- `first_chunk_ready_s` includes waiting for the global backend gate.
  `wall_e2e_s` ends when the terminal emitter coroutine returns. Neither is a
  browser-paint metric; a real client acknowledgement is required.
- Consumers must ignore every event whose `job_id` is not their latest job,
  because a transport write already in flight cannot be retracted.

## Bounded memory and concurrency

Credits are acquired before requesting the next backend chunk and released
only after that chunk's emitter returns. This bounds decoded payloads across
the producer, queue, and consumer rather than merely bounding a queue after
generation. Delivered payloads are not retained in `StreamSummary`; only
`ChunkReleaseEvent` metadata remains.

The registry enforces a maximum latent count and maximum bytes per chunk,
unique live job IDs, client-scoped replacement/cancellation, one global backend
lock for a non-reentrant CUDA cache/stream, and one emitter lock per client.
Emitter locks retire after the last live call; a detached uncooperative emitter
keeps its client lock until it actually exits, preventing overlap with a
replacement job.

The WebSocket layer adds a second, explicit two-chunk presentation window. A
chunk header and its binary raster messages do not disclose an ACK token. Only
after the full binary group is written and the delivery is registered does the
server emit `chunk_committed` with an unpredictable delivery ID. The client
ACKs that token only after every frame in the group is decoded, drawn, and has
crossed a following `requestAnimationFrame` opportunity.

## Failure isolation

Emitter, backend-next, and backend-close calls run in supervised child tasks
with hard async deadlines. Detached task outcomes are always consumed. A hung
emitter is isolated to its client. A backend operation that will not cancel or
close poisons the registry's global backend gate, and later starts fail
synchronously instead of reusing cache/stream state whose ownership is no
longer known. Restart the worker to recover a poisoned backend.

Python cannot preempt synchronous code that blocks the event-loop thread. The
fake process backend now proves the required isolation shape: `Popen` plus
socketpair rather than threads or asyncio subprocesses, `python -I`, an explicit
environment allowlist, a digest-bound startup HELLO, canonical length-prefixed
headers, aggregate payload limits before reads, a nonce-gated capacity-one pull,
bounded stderr, and exact worker/job/chunk fencing. Only a fully validated
  `COMPLETE` permits warm reuse. Stderr draining has a finite per-callback work
  budget. Cancellation retires ownership and kills the entire worker process
  group before awaiting a bounded reap; the control descriptor cannot flow into
  helpers. A reap timeout detaches I/O, retains the process handle for a later
  idempotent `close`, and poisons reuse. This is Python isolation plus an
  environment-scrubbed trusted child, not a filesystem or network sandbox.

`RollingTaehvChunkDecoder` now encodes the CUDA-side portion of that contract.
It requires a producer event, waits on it from an explicit decoder stream,
records allocator ownership, converts the exact CF++1 bfloat16 latent to
float16, retains the chronological last three latents, applies the canonical
`3/4/8/12…` trims, and synchronizes after GPU postprocessing before D2H and
encoding. It accepts only recomputed runtime provenance binding source, runtime
and asset locks, tokenizer, actual FA2/FA3 backend, exact Torch binding,
effective config, all assets, and a six-module guard bundle. Any post-start
failure poisons the decoder; only 21 complete chunks permit tail release.

The dedicated worker must still own this class, connect its poison to
kill/reap, bind the real encoder and session into HELLO/terminal provenance,
and prove the event and D2H behavior on real CUDA. The session contract is
recorded in [[CF1-Rolling-TAEHV-Session]].

The recovered 0.242-second latency stops at a post-TAEHV GPU event. It is not a
target for `first_chunk_ready_s`: the historical runner performs D2H only after
the complete rollout, while a serving backend must synchronize D2H and create
immutable bytes before yielding. Bare `asyncio.to_thread` cancellation is also
unsafe because the CUDA thread may outlive its cancelled Future; use a
dedicated worker process whose death or quiescence can be proved.

## Network adapters and replacement safety

`bench/streaming_transport.py` is a loopback NDJSON/base64 conformance adapter.
`bench/streaming_websocket.py` is the browser surface: literal-loopback bind,
exact Host/Origin checks, a one-time HttpOnly SameSite cookie nonce, strict
subprotocol, external CSP-safe assets, binary payloads, bounded control sends,
job/delivery lifetime uniqueness, and sanitized terminal errors.

Fencing is not a transport failure. Both adapters may return `False` when an
old job's expected ID no longer matches the current job; retirement must ignore
that stale result rather than disconnect the replacement. Likewise, a binary
group retired during replacement must not receive a commit token, and the
client ignores already-stale commits while retaining fatal handling for a
current-job commit without metadata. These invariants are recorded in
[[Gotcha-Transport-Write-Is-Not-Presentation]].

## Verification

- 35 process tests cover startup identity, pull/nonces, framing and payload
  bounds, hashes, topology, evidence, warm reuse, cancellation, kill/reap,
  poisoning, close, lifetime IDs, false-isolation/sensitive-environment
  rejection, shared protocol limits, bundle identity, and real service
  integration.
- 113 focused tests cover the process, service, NDJSON, and WebSocket release,
  payload, backpressure, replacement, cancellation, lifetime IDs, timeout,
  cleanup, poison, security, and sanitization contracts. Four executable Node
  tests cover stale-commit handling and draw/presentation-before-ACK ordering.
- Latest validation passes **278/278** Python tests: **35/35** process,
  **113/113** combined process/service/NDJSON/WebSocket (**78/78** without
  process), **13/13** asset-preflight, **17/17** runtime-preflight, **21/21**
  CUDA-adapter, and **9/9** CUDA session. Python `compileall`, every repository
  JSON parse, and **4/4**
  executable Node tests also pass.
- The real asset gate reports `ready: true`, exact source commit, zero
  unexpected paths, and all 11 registered assets verified. This is byte and
  structural evidence, not a real PyTorch/CUDA load.
- A real in-app browser repeatedly completed 81 painted frames, 21 chunks, and
  21 presentation ACKs in about 5.1 seconds with zero warnings. An in-flight
  replacement after five frames completed its replacement on the same socket.
- The actual-file consensus bridges timed out. First recovery retained Grok's
  replacement/commit races; each reproduced red and is fixed. The smaller
  post-fix recovery retained Fable and Grok finals agreeing the five defects
  are fixed for the fake milestone. Kimi had no final, so this is not a
  completed panel or a project-level gate.
- The process design panel completed with Fable-5, Kimi-K3, and Grok-4.5 at
  GO-WITH-FIXES. Both later actual-code bridges timed out; session recovery
  found only Grok finals, with Fable and Kimi at zero output. Concrete Grok
  findings around reap-time I/O, cancellation ownership, rollout/lifetime
  bounds, and close coverage reproduced and are fixed. The fake process seam
  is therefore locally verified with partial independent code review, not a
  completed project-level panel.
- A later post-fix actual-artifact panel timed out, but recovery retained
  concrete Fable-5 and Grok-4.5 GO-WITH-FIXES finals; Kimi-K3 produced no final.
  All reproducible in-scope verifier and process findings were fixed. A local
  Claude CLI `--model opus` fallback separately reviewed the exact structural
  bootstrap, returned GO-WITH-FIXES, and did not expose a concrete model
  version. Neither partial review clears the project-level gate.
- The rolling-session parent `ses_080ea9c12ffejex9NyF4Yi7sk0` also timed out
  without synthesis. Recovery retained concrete Fable-5 child
  `ses_080e6c4cbffewgbBPMTGzWaxMV` at GO-WITH-FIXES / 0.78 and concrete
  Grok-4.5 child `ses_080e6c4c9ffeiwnSU0HNzAeM3r` at GO-WITH-FIXES / 0.82;
  Kimi child `ses_080e6c4caffeh0tWtS1ApYcJPr` stalled with zero tokens. Fable
  succeeded, so no Opus fallback occurred. A local Claude CLI Opus review then
  mutation-tested the decoder and every reproducible mocked-core gap was fixed.
  This is not completed three-voter consensus.

The next implementation increment is deliberately narrow. First pin runtime
dependencies and tokenizer sentinel IDs and settle peak CPU RAM/VRAM with a
capacity plan or meta/empty initialization. Then run one dependency-pinned,
dedicated real CUDA worker boot/session/frame smoke: load exact tensors,
exercise the producer/decode event handoff, generate and decode a real latent,
encode/decode-check a 832×480 raster, bind session/encoder identity into the
worker handshake, and prove poison-driven kill/reap. This is not permission for
a broad benchmark, protocol freeze, or headline claim.
