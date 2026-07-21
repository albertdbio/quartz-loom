---
type: gotcha
status: active
date: 2026-07-19
description: "asyncio.wait_for is not a hard deadline when a child suppresses cancellation; isolate the operation in its own task, detach safely at the deadline, and poison shared backend state that can no longer be proven quiescent."
anchors: ["bench/streaming_service.py#_emit_safely", "bench/streaming_service.py#run_stream_job", "bench/streaming_service.py#StreamingJobRegistry"]
related: ["[[Streaming-Service-Boundary]]", "[[State]]", "[[session-6-streaming-service-boundary]]"]
---

# Gotcha — Async timeouts need task isolation

`asyncio.wait_for(coro, timeout)` is cooperative, not a hard wall. At timeout it
cancels the child and then waits for cancellation to finish. If the child
catches `CancelledError` and continues, the caller can remain stuck forever.
This was reproduced independently for the transport emitter, backend
`__anext__`, and async-generator `aclose` paths.

For a hard async deadline, start the operation as a child task and wait on that
task with `asyncio.wait(..., timeout=...)`. At the deadline, cancel and detach
the child, install a done callback that retrieves its eventual exception, and
return control to the supervisor. Do not discard a detached task without
consuming its outcome.

Detachment changes ownership:

- A detached emitter may still be writing to one client. Keep that client's
  emitter lock associated with the rogue task until it exits, so a replacement
  job cannot overlap sends.
- A detached backend-next or backend-close operation may still own shared CUDA
  cache/stream state. Poison the global backend gate and reject new jobs until
  the worker is restarted. A timeout is not proof that the backend stopped.

This pattern still cannot preempt synchronous Python/CUDA code blocking the
event-loop thread. Put blocking adapter work in a thread or process, and use a
process when termination and clean state reconstruction are required.

Do not treat bare `asyncio.to_thread` or executor cancellation as proof that a
CUDA call stopped. Cancelling its asyncio Future can complete immediately while
the underlying thread continues to mutate CUDA/IPC state. That can fool an
async supervisor into treating cancellation as cooperative and avoid the
backend-poison path. A CUDA adapter must instead use cancellable pipe/socket IPC
to a dedicated process, or make its cancellation path terminate and join the
worker before returning. If quiescence cannot be proved, fail closed and
reconstruct the worker.
