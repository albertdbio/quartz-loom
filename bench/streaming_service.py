"""Dependency-free streaming job boundary for the future realtime demo.

The CUDA generator is intentionally not imported here. Backends yield immutable
decoded chunks; this module stamps their service-boundary readiness, applies
bounded backpressure, validates the release sequence, targets events to one
client, and reports chunk cadence without inventing per-frame timestamps.

All timing is server-side. ``first_chunk_ready_s`` includes backend-gate wait,
and ``wall_e2e_s`` ends when the emitter coroutine returns; neither proves that
a browser painted a frame. A real UI needs a client acknowledgement for that.
Every consumer must also fence events by the latest ``job_id`` because a send
already in transport cannot be retracted.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from bench.generation_preflight import ChunkReleaseEvent
from bench.png_validation import is_valid_png


DEFAULT_EMIT_TIMEOUT_S = 2.0
DEFAULT_BACKEND_CLOSE_TIMEOUT_S = 2.0
DEFAULT_BACKEND_CHUNK_TIMEOUT_S = 30.0
DEFAULT_MAX_LATENT_FRAMES = 241
DEFAULT_MAX_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_FRAME_MEDIA_TYPE = "application/octet-stream"
ALLOWED_FRAME_MEDIA_TYPES = frozenset(
    {
        DEFAULT_FRAME_MEDIA_TYPE,
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)


class StreamProtocolError(ValueError):
    """Raised when a backend or streaming job violates the serving contract."""


class BackendFatalError(StreamProtocolError):
    """Raised when process/backend ownership is unsafe and reuse must stop."""


class _StreamEmitterError(StreamProtocolError):
    """Sanitized transport failure that must not be mistaken for backend data."""


class _BackendOperationError(StreamProtocolError):
    """Sanitized backend operation failure with a stable public code."""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StreamProtocolError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamProtocolError(f"{label} must be an integer")
    if positive and value <= 0:
        raise StreamProtocolError(f"{label} must be positive")
    return value


def _require_positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise StreamProtocolError(f"{label} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class StreamRequest:
    client_id: str
    job_id: str
    prompt: str
    seed: int
    latent_frames: int = 21

    def __post_init__(self) -> None:
        _require_nonempty_string(self.client_id, "client_id")
        _require_nonempty_string(self.job_id, "job_id")
        _require_nonempty_string(self.prompt, "prompt")
        _require_int(self.seed, "seed")
        _require_int(self.latent_frames, "latent_frames", positive=True)


@dataclass(frozen=True)
class DecodedChunk:
    """One immutable CPU-owned payload batch returned by a backend."""

    frame_payloads: tuple[bytes, ...]
    frame_media_type: str = DEFAULT_FRAME_MEDIA_TYPE

    @property
    def frame_count(self) -> int:
        return len(self.frame_payloads)


@dataclass(frozen=True)
class _ReleasedChunk:
    chunk_index: int
    first_frame_index: int
    frame_payloads: tuple[bytes, ...]
    frame_media_type: str
    ready_ns: int

    @property
    def frame_count(self) -> int:
        return len(self.frame_payloads)


@dataclass(frozen=True)
class StreamSummary:
    job_id: str
    frame_count: int
    release_count: int
    first_chunk_ready_s: float
    wall_e2e_s: float
    e2e_fps: float
    p95_chunk_release_gap_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "frame_count": self.frame_count,
            "release_count": self.release_count,
            "first_chunk_ready_s": self.first_chunk_ready_s,
            "wall_e2e_s": self.wall_e2e_s,
            "e2e_fps": self.e2e_fps,
            "p95_chunk_release_gap_ms": self.p95_chunk_release_gap_ms,
        }


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    job_id: str
    started_ns: int | None = None
    chunk_index: int | None = None
    first_frame_index: int | None = None
    frame_payloads: tuple[bytes, ...] | None = None
    frame_media_type: str | None = None
    ready_ns: int | None = None
    queue_depth: int | None = None
    summary: StreamSummary | None = None
    error_code: str | None = None

    @property
    def frame_count(self) -> int | None:
        return None if self.frame_payloads is None else len(self.frame_payloads)

    def to_dict(self, *, include_payloads: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "job_id": self.job_id}
        for field, item in (
            ("started_ns", self.started_ns),
            ("chunk_index", self.chunk_index),
            ("first_frame_index", self.first_frame_index),
            ("ready_ns", self.ready_ns),
            ("queue_depth", self.queue_depth),
            ("error_code", self.error_code),
        ):
            if item is not None:
                value[field] = item
        if self.frame_payloads is not None:
            value["frame_count"] = len(self.frame_payloads)
            if self.frame_media_type is not None:
                value["frame_media_type"] = self.frame_media_type
            if include_payloads:
                value["frame_payloads_base64"] = [
                    base64.b64encode(payload).decode("ascii")
                    for payload in self.frame_payloads
                ]
        if self.summary is not None:
            value["summary"] = self.summary.to_dict()
        return value


class StreamingBackend(Protocol):
    def stream(self, request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        """Yield immutable CPU-owned chunks; never block the asyncio event loop.

        Cancellation and ``aclose`` must normally cooperate. The registry uses
        hard async deadlines and poisons its backend gate after an operation is
        detached, but Python cannot preempt synchronous code that blocks the
        event-loop thread. A CUDA adapter must therefore isolate blocking work
        in a dedicated process whose termination and reaping can be proved.
        """


Emitter = Callable[[str, StreamEvent], Awaitable[None]]
Clock = Callable[[], int]
BackendHealthCheck = Callable[[], bool]
BackendPoisoner = Callable[[], None]


def _consume_detached_task(task: asyncio.Future[Any]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


def _cancel_and_detach(task: asyncio.Future[Any]) -> None:
    task.cancel()
    task.add_done_callback(_consume_detached_task)


async def _emit_safely(
    emit: Emitter,
    client_id: str,
    event: StreamEvent,
    *,
    timeout_s: float,
) -> None:
    async def invoke() -> None:
        await emit(client_id, event)

    emission = asyncio.create_task(invoke())
    failed = False
    try:
        done, _pending = await asyncio.wait({emission}, timeout=timeout_s)
    except asyncio.CancelledError:
        _cancel_and_detach(emission)
        raise
    if not done:
        _cancel_and_detach(emission)
        failed = True
    else:
        try:
            emission.result()
        except asyncio.CancelledError:
            failed = True
        except Exception:
            failed = True
    if failed:
        # Do not retain a transport exception: messages may contain credentials
        # or response bodies. Raise the sanitized error outside the except block
        # so it has no implicit exception context.
        raise _StreamEmitterError("stream emitter failed")


@dataclass(frozen=True)
class _QueuedFailure:
    error: StreamProtocolError
    error_code: str


_END = object()


def _expected_rgb_frames(latent_frames: int) -> int:
    return 1 + 4 * (latent_frames - 1)


def _read_clock(clock_ns: Clock, label: str) -> int:
    try:
        value = clock_ns()
    except Exception as exc:
        raise StreamProtocolError(f"{label} clock failed") from exc
    value = _require_int(value, label)
    if value < 0:
        raise StreamProtocolError(f"{label} must be non-negative")
    return value


def _validate_chunk(
    value: Any,
    *,
    chunk_index: int,
    expected_frame_count: int,
    max_chunk_bytes: int,
) -> DecodedChunk:
    if not isinstance(value, DecodedChunk):
        raise StreamProtocolError("streaming backend must yield DecodedChunk values")
    if not isinstance(value.frame_payloads, tuple) or not value.frame_payloads:
        raise StreamProtocolError("decoded chunks require a non-empty payload tuple")
    if any(not isinstance(payload, bytes) or not payload for payload in value.frame_payloads):
        raise StreamProtocolError("decoded frame payloads must be non-empty bytes")
    if value.frame_media_type not in ALLOWED_FRAME_MEDIA_TYPES:
        raise StreamProtocolError(
            "decoded chunk frame_media_type must be one of "
            f"{sorted(ALLOWED_FRAME_MEDIA_TYPES)}"
        )
    for payload in value.frame_payloads:
        if not _payload_matches_media_type(payload, value.frame_media_type):
            raise StreamProtocolError(
                "decoded frame payload does not match "
                f"{value.frame_media_type}"
            )
    if value.frame_count != expected_frame_count:
        raise StreamProtocolError(
            f"chunk {chunk_index} must contain {expected_frame_count} RGB frames; "
            f"received {value.frame_count}"
        )
    payload_bytes = sum(len(payload) for payload in value.frame_payloads)
    if payload_bytes > max_chunk_bytes:
        raise StreamProtocolError(
            f"chunk {chunk_index} exceeds byte limit {max_chunk_bytes}; "
            f"received {payload_bytes}"
        )
    return value


def _payload_matches_media_type(payload: bytes, media_type: str) -> bool:
    """Reject raster-label drift before a client treats bytes as renderable."""

    if media_type == DEFAULT_FRAME_MEDIA_TYPE:
        return True
    if media_type == "image/png":
        return is_valid_png(payload)
    if media_type == "image/jpeg":
        return payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    if media_type == "image/webp":
        if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WEBP":
            return False
        return int.from_bytes(payload[4:8], "little") + 8 == len(payload)
    return False


def _summarize(
    request: StreamRequest,
    *,
    started_ns: int,
    completed_ns: int,
    chunks: Sequence[ChunkReleaseEvent],
) -> StreamSummary:
    if not chunks:
        raise StreamProtocolError("streaming backend produced no decoded chunks")
    expected_frames = _expected_rgb_frames(request.latent_frames)
    frame_count = sum(chunk.frame_count for chunk in chunks)
    if frame_count != expected_frames:
        raise StreamProtocolError(
            f"streaming backend released {frame_count} RGB frames; "
            f"the {request.latent_frames}-latent rollout requires "
            f"{expected_frames} RGB frames"
        )
    if completed_ns < chunks[-1].ready_ns or completed_ns <= started_ns:
        raise StreamProtocolError("job completion time is inconsistent with chunk releases")
    gaps_ns = [
        chunks[index].ready_ns - chunks[index - 1].ready_ns
        for index in range(1, len(chunks))
    ]
    p95_gap_ms: float | None = None
    if gaps_ns:
        ordered = sorted(gaps_ns)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        p95_gap_ms = ordered[p95_index] / 1_000_000
    wall_e2e_s = (completed_ns - started_ns) / 1_000_000_000
    return StreamSummary(
        job_id=request.job_id,
        frame_count=frame_count,
        release_count=len(chunks),
        first_chunk_ready_s=(chunks[0].ready_ns - started_ns) / 1_000_000_000,
        wall_e2e_s=wall_e2e_s,
        e2e_fps=frame_count / wall_e2e_s,
        p95_chunk_release_gap_ms=p95_gap_ms,
    )


async def run_stream_job(
    request: StreamRequest,
    backend: StreamingBackend,
    *,
    emit: Emitter,
    clock_ns: Clock = time.monotonic_ns,
    queue_capacity: int = 2,
    backend_lock: asyncio.Lock | None = None,
    emit_timeout_s: float = DEFAULT_EMIT_TIMEOUT_S,
    backend_close_timeout_s: float = DEFAULT_BACKEND_CLOSE_TIMEOUT_S,
    backend_chunk_timeout_s: float = DEFAULT_BACKEND_CHUNK_TIMEOUT_S,
    max_latent_frames: int = DEFAULT_MAX_LATENT_FRAMES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    backend_health_check: BackendHealthCheck | None = None,
    poison_backend: BackendPoisoner | None = None,
) -> StreamSummary:
    """Run one client-scoped job through a bounded decoded-chunk queue.

    Readiness is stamped immediately after each backend ``__anext__`` returns.
    Chunk credits are acquired before requesting more decoded bytes, and only
    payload-free ``ChunkReleaseEvent`` records survive emission.
    """

    if not isinstance(request, StreamRequest):
        raise StreamProtocolError("request must be a StreamRequest")
    queue_capacity = _require_int(queue_capacity, "queue_capacity", positive=True)
    max_latent_frames = _require_int(
        max_latent_frames, "max_latent_frames", positive=True
    )
    max_chunk_bytes = _require_int(
        max_chunk_bytes, "max_chunk_bytes", positive=True
    )
    emit_timeout_s = _require_positive_number(emit_timeout_s, "emit_timeout_s")
    backend_close_timeout_s = _require_positive_number(
        backend_close_timeout_s, "backend_close_timeout_s"
    )
    backend_chunk_timeout_s = _require_positive_number(
        backend_chunk_timeout_s, "backend_chunk_timeout_s"
    )
    if request.latent_frames > max_latent_frames:
        raise StreamProtocolError(
            f"latent_frames exceeds service limit {max_latent_frames}"
        )
    if backend_health_check is not None and not callable(backend_health_check):
        raise StreamProtocolError("backend_health_check must be callable")
    if poison_backend is not None and not callable(poison_backend):
        raise StreamProtocolError("poison_backend must be callable")
    try:
        started_ns = _read_clock(clock_ns, "started_ns")
    except StreamProtocolError:
        # Registry.start() has already handed a job handle to the transport.
        # Preserve accepted-job terminality even when the first clock read fails.
        with contextlib.suppress(Exception):
            await _emit_safely(
                emit,
                request.client_id,
                StreamEvent(
                    kind="job_failed",
                    job_id=request.job_id,
                    error_code="protocol_error",
                ),
                timeout_s=emit_timeout_s,
            )
        raise
    queue: asyncio.Queue[_ReleasedChunk | _QueuedFailure | object] = asyncio.Queue(
        maxsize=queue_capacity
    )
    credits = asyncio.Semaphore(queue_capacity)
    pending_chunks = 0

    try:
        await _emit_safely(
            emit,
            request.client_id,
            StreamEvent(
                kind="job_started",
                job_id=request.job_id,
                started_ns=started_ns,
            ),
            timeout_s=emit_timeout_s,
        )
    except asyncio.CancelledError:
        # A transport may have delivered the start event before its coroutine
        # was cancelled. Make a bounded best-effort terminal notification.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await _emit_safely(
                emit,
                request.client_id,
                StreamEvent(kind="job_cancelled", job_id=request.job_id),
                timeout_s=emit_timeout_s,
            )
        raise

    def mark_backend_poisoned() -> None:
        if poison_backend is not None:
            poison_backend()

    async def close_iterator(iterator: AsyncIterator[DecodedChunk]) -> bool:
        try:
            closer = getattr(iterator, "aclose", None)
        except Exception:
            mark_backend_poisoned()
            return False
        if not callable(closer):
            return True

        async def invoke_close() -> None:
            await closer()

        try:
            close_operation = asyncio.create_task(invoke_close())
        except Exception:
            mark_backend_poisoned()
            return False
        try:
            done, _pending = await asyncio.wait(
                {close_operation},
                timeout=backend_close_timeout_s,
            )
        except asyncio.CancelledError:
            _cancel_and_detach(close_operation)
            mark_backend_poisoned()
            raise
        if not done:
            _cancel_and_detach(close_operation)
            mark_backend_poisoned()
            return False
        try:
            close_operation.result()
        except asyncio.CancelledError:
            mark_backend_poisoned()
            return False
        except Exception:
            mark_backend_poisoned()
            return False
        return True

    async def next_chunk(iterator: AsyncIterator[DecodedChunk]) -> DecodedChunk:
        async def invoke_next() -> DecodedChunk:
            return await iterator.__anext__()

        try:
            operation = asyncio.create_task(invoke_next())
        except Exception:
            raise _BackendOperationError(
                "streaming backend failed",
                "backend_failure",
            )
        try:
            done, _pending = await asyncio.wait(
                {operation},
                timeout=backend_chunk_timeout_s,
            )
        except asyncio.CancelledError:
            operation.cancel()
            try:
                cancelled, _pending = await asyncio.wait(
                    {operation},
                    timeout=backend_close_timeout_s,
                )
            except asyncio.CancelledError:
                cancelled = set()
            if not cancelled:
                operation.add_done_callback(_consume_detached_task)
                mark_backend_poisoned()
            else:
                _consume_detached_task(operation)
            raise
        if not done:
            _cancel_and_detach(operation)
            mark_backend_poisoned()
            raise _BackendOperationError(
                "streaming backend chunk timed out",
                "backend_timeout",
            )
        try:
            return operation.result()
        except StopAsyncIteration:
            raise
        except asyncio.CancelledError:
            raise _BackendOperationError(
                "streaming backend failed",
                "backend_failure",
            )
        except BackendFatalError:
            mark_backend_poisoned()
            raise _BackendOperationError(
                "streaming backend failed",
                "backend_fatal",
            )
        except Exception:
            raise _BackendOperationError(
                "streaming backend failed",
                "backend_failure",
            )

    async def produce_unlocked() -> None:
        nonlocal pending_chunks
        released_chunk_count = 0
        expected_first_frame_index = 0
        previous_ready_ns: int | None = None
        job_frame_media_type: str | None = None
        iterator: AsyncIterator[DecodedChunk] | None = None
        iterator_closed = False
        try:
            try:
                iterator = backend.stream(request).__aiter__()
            except BackendFatalError:
                mark_backend_poisoned()
                await queue.put(
                    _QueuedFailure(
                        StreamProtocolError("streaming backend failed"),
                        "backend_fatal",
                    )
                )
                return
            except asyncio.CancelledError:
                await queue.put(
                    _QueuedFailure(
                        StreamProtocolError("streaming backend failed"),
                        "backend_failure",
                    )
                )
                return
            except Exception:
                await queue.put(
                    _QueuedFailure(
                        StreamProtocolError("streaming backend failed"),
                        "backend_failure",
                    )
                )
                return
            while True:
                await credits.acquire()
                try:
                    raw = await next_chunk(iterator)
                except StopAsyncIteration:
                    credits.release()
                    break
                except asyncio.CancelledError:
                    credits.release()
                    raise
                except _BackendOperationError as exc:
                    credits.release()
                    await queue.put(
                        _QueuedFailure(
                            StreamProtocolError(str(exc)),
                            exc.error_code,
                        )
                    )
                    return
                try:
                    chunk_index = released_chunk_count
                    if chunk_index >= request.latent_frames:
                        raise StreamProtocolError(
                            f"streaming backend produced extra chunk {chunk_index}"
                        )
                    decoded = _validate_chunk(
                        raw,
                        chunk_index=chunk_index,
                        expected_frame_count=1 if chunk_index == 0 else 4,
                        max_chunk_bytes=max_chunk_bytes,
                    )
                    if job_frame_media_type is None:
                        job_frame_media_type = decoded.frame_media_type
                    elif decoded.frame_media_type != job_frame_media_type:
                        raise StreamProtocolError(
                            "decoded chunk media type cannot change within a job"
                        )
                    ready_ns = _read_clock(clock_ns, "chunk ready_ns")
                    if ready_ns < started_ns:
                        raise StreamProtocolError("chunk ready_ns precedes the job start")
                    if previous_ready_ns is not None and ready_ns <= previous_ready_ns:
                        raise StreamProtocolError(
                            "chunk ready_ns values must be strictly increasing"
                        )
                    chunk = _ReleasedChunk(
                        chunk_index=chunk_index,
                        first_frame_index=expected_first_frame_index,
                        frame_payloads=decoded.frame_payloads,
                        frame_media_type=decoded.frame_media_type,
                        ready_ns=ready_ns,
                    )
                except StreamProtocolError as exc:
                    credits.release()
                    await queue.put(_QueuedFailure(exc, "protocol_error"))
                    return
                await queue.put(chunk)
                pending_chunks += 1
                released_chunk_count += 1
                expected_first_frame_index += chunk.frame_count
                previous_ready_ns = chunk.ready_ns
            expected_frames = _expected_rgb_frames(request.latent_frames)
            if expected_first_frame_index != expected_frames:
                await queue.put(
                    _QueuedFailure(
                        StreamProtocolError(
                            f"streaming backend released {expected_first_frame_index} "
                            f"RGB frames; the {request.latent_frames}-latent rollout "
                            f"requires {expected_frames} RGB frames"
                        ),
                        "protocol_error",
                    )
                )
                return
            if not await close_iterator(iterator):
                iterator_closed = True
                await queue.put(
                    _QueuedFailure(
                        StreamProtocolError("streaming backend cleanup failed"),
                        "backend_cleanup_failure",
                    )
                )
                return
            iterator_closed = True
            await queue.put(_END)
        finally:
            if iterator is not None and not iterator_closed:
                await close_iterator(iterator)

    async def produce() -> None:
        try:
            if backend_lock is None:
                if backend_health_check is not None and not backend_health_check():
                    await queue.put(
                        _QueuedFailure(
                            StreamProtocolError("streaming backend is unavailable"),
                            "backend_unavailable",
                        )
                    )
                    return
                await produce_unlocked()
                return
            async with backend_lock:
                if backend_health_check is not None and not backend_health_check():
                    await queue.put(
                        _QueuedFailure(
                            StreamProtocolError("streaming backend is unavailable"),
                            "backend_unavailable",
                        )
                    )
                    return
                await produce_unlocked()
        except asyncio.CancelledError:
            raise
        except Exception:
            await queue.put(
                _QueuedFailure(
                    StreamProtocolError("streaming backend failed"),
                    "backend_failure",
                )
            )

    producer = asyncio.create_task(produce())
    delivered_chunks: list[ChunkReleaseEvent] = []
    failure_emitted = False
    try:
        while True:
            queued = await queue.get()
            if queued is _END:
                break
            if isinstance(queued, _QueuedFailure):
                await _emit_safely(
                    emit,
                    request.client_id,
                    StreamEvent(
                        kind="job_failed",
                        job_id=request.job_id,
                        error_code=queued.error_code,
                    ),
                    timeout_s=emit_timeout_s,
                )
                failure_emitted = True
                raise queued.error
            if not isinstance(queued, _ReleasedChunk):
                raise StreamProtocolError("streaming queue contained an invalid item")
            pending_chunks -= 1
            if pending_chunks < 0:
                raise StreamProtocolError("streaming chunk accounting underflow")
            delivered_chunks.append(
                ChunkReleaseEvent(
                    chunk_index=queued.chunk_index,
                    first_frame_index=queued.first_frame_index,
                    frame_count=queued.frame_count,
                    ready_ns=queued.ready_ns,
                )
            )
            try:
                await _emit_safely(
                    emit,
                    request.client_id,
                    StreamEvent(
                        kind="chunk_ready",
                        job_id=request.job_id,
                        chunk_index=queued.chunk_index,
                        first_frame_index=queued.first_frame_index,
                        frame_payloads=queued.frame_payloads,
                        frame_media_type=queued.frame_media_type,
                        ready_ns=queued.ready_ns,
                        queue_depth=pending_chunks,
                    ),
                    timeout_s=emit_timeout_s,
                )
            finally:
                credits.release()
        await producer
        completed_ns = _read_clock(clock_ns, "completed_ns")
        summary = _summarize(
            request,
            started_ns=started_ns,
            completed_ns=completed_ns,
            chunks=delivered_chunks,
        )
        await _emit_safely(
            emit,
            request.client_id,
            StreamEvent(
                kind="job_completed",
                job_id=request.job_id,
                summary=summary,
            ),
            timeout_s=emit_timeout_s,
        )
        return summary
    except asyncio.CancelledError:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
        with contextlib.suppress(Exception):
            await _emit_safely(
                emit,
                request.client_id,
                StreamEvent(kind="job_cancelled", job_id=request.job_id),
                timeout_s=emit_timeout_s,
            )
        raise
    except _StreamEmitterError:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
        raise
    except StreamProtocolError:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
        if not failure_emitted:
            with contextlib.suppress(Exception):
                await _emit_safely(
                    emit,
                    request.client_id,
                    StreamEvent(
                        kind="job_failed",
                        job_id=request.job_id,
                        error_code="protocol_error",
                    ),
                    timeout_s=emit_timeout_s,
                )
        raise
    except Exception:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
        raise _StreamEmitterError("stream emitter failed")


class FakeStreamingBackend:
    """Deterministic byte-payload backend for UI/service tests only."""

    def __init__(
        self,
        *,
        frame_counts: Sequence[int],
    ) -> None:
        if isinstance(frame_counts, (str, bytes)) or not isinstance(
            frame_counts, Sequence
        ):
            raise StreamProtocolError("frame_counts must be an array")
        if not frame_counts:
            raise StreamProtocolError("frame_counts must not be empty")
        self.frame_counts = list(frame_counts)

    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        for chunk_index, frame_count in enumerate(self.frame_counts):
            await asyncio.sleep(0)
            frame_count = _require_int(
                frame_count, f"frame_counts[{chunk_index}]", positive=True
            )
            yield DecodedChunk(
                frame_payloads=tuple(
                    f"fake-frame-{chunk_index}-{index}".encode("utf-8")
                    for index in range(frame_count)
                ),
            )


@dataclass(frozen=True)
class JobHandle:
    """Registry handle whose task may be awaited for the terminal result."""

    client_id: str
    job_id: str
    task: asyncio.Task[StreamSummary]


class StreamingJobRegistry:
    """Own one active job per client and serialize a non-reentrant backend.

    A backend operation that exceeds a hard async deadline poisons the global
    backend gate. Subsequent starts then fail synchronously; restart the worker
    rather than risking concurrent use of potentially live CUDA state.
    """

    def __init__(
        self,
        *,
        emit: Emitter,
        clock_ns: Clock = time.monotonic_ns,
        job_id_factory: Callable[[], str] | None = None,
        queue_capacity: int = 2,
        emit_timeout_s: float = DEFAULT_EMIT_TIMEOUT_S,
        backend_close_timeout_s: float = DEFAULT_BACKEND_CLOSE_TIMEOUT_S,
        backend_chunk_timeout_s: float = DEFAULT_BACKEND_CHUNK_TIMEOUT_S,
        max_latent_frames: int = DEFAULT_MAX_LATENT_FRAMES,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    ) -> None:
        self._emit = emit
        self._clock_ns = clock_ns
        self._job_id_factory = job_id_factory or (lambda: uuid.uuid4().hex)
        self._queue_capacity = _require_int(
            queue_capacity, "queue_capacity", positive=True
        )
        self._emit_timeout_s = _require_positive_number(
            emit_timeout_s, "emit_timeout_s"
        )
        self._backend_close_timeout_s = _require_positive_number(
            backend_close_timeout_s, "backend_close_timeout_s"
        )
        self._backend_chunk_timeout_s = _require_positive_number(
            backend_chunk_timeout_s, "backend_chunk_timeout_s"
        )
        self._max_latent_frames = _require_int(
            max_latent_frames, "max_latent_frames", positive=True
        )
        self._max_chunk_bytes = _require_int(
            max_chunk_bytes, "max_chunk_bytes", positive=True
        )
        self._active: dict[str, JobHandle] = {}
        self._live_job_ids: set[tuple[str, str]] = set()
        self._emit_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        # The recovered CUDA pipeline owns mutable caches and CUDA streams.
        # Until a backend proves it is re-entrant, every job shares one gate.
        self._backend_lock = asyncio.Lock()
        self._backend_poisoned = False

    def _backend_is_healthy(self) -> bool:
        return not self._backend_poisoned

    def _poison_backend(self) -> None:
        self._backend_poisoned = True

    def _retire_emit_lock_if_idle(
        self,
        client_id: str,
        emit_lock: asyncio.Lock,
    ) -> None:
        if (
            self._emit_locks.get(client_id) is emit_lock
            and not emit_lock.locked()
            and client_id not in self._active
            and not any(key_client == client_id for key_client, _job_id in self._live_job_ids)
        ):
            self._emit_locks.pop(client_id, None)

    async def start(
        self,
        *,
        client_id: str,
        prompt: str,
        seed: int,
        backend: StreamingBackend,
        latent_frames: int = 21,
    ) -> JobHandle:
        latent_frames = _require_int(latent_frames, "latent_frames", positive=True)
        if latent_frames > self._max_latent_frames:
            raise StreamProtocolError(
                f"latent_frames exceeds service limit {self._max_latent_frames}"
            )
        job_id = _require_nonempty_string(self._job_id_factory(), "job_id")
        request = StreamRequest(
            client_id=client_id,
            job_id=job_id,
            prompt=prompt,
            seed=seed,
            latent_frames=latent_frames,
        )

        async def owned_run(client_emit_lock: asyncio.Lock) -> StreamSummary:
            async def emit_serialized(target_client_id: str, event: StreamEvent) -> None:
                try:
                    async with client_emit_lock:
                        async with self._lock:
                            current = self._active.get(client_id)
                            if current is None and event.kind != "job_cancelled":
                                return
                            if current is not None and current.job_id != job_id:
                                return
                        await self._emit(target_client_id, event)
                finally:
                    self._retire_emit_lock_if_idle(client_id, client_emit_lock)

            try:
                return await run_stream_job(
                    request,
                    backend,
                    emit=emit_serialized,
                    clock_ns=self._clock_ns,
                    queue_capacity=self._queue_capacity,
                    backend_lock=self._backend_lock,
                    emit_timeout_s=self._emit_timeout_s,
                    backend_close_timeout_s=self._backend_close_timeout_s,
                    backend_chunk_timeout_s=self._backend_chunk_timeout_s,
                    max_latent_frames=self._max_latent_frames,
                    max_chunk_bytes=self._max_chunk_bytes,
                    backend_health_check=self._backend_is_healthy,
                    poison_backend=self._poison_backend,
                )
            finally:
                async with self._lock:
                    current = self._active.get(client_id)
                    if (
                        current is not None
                        and current.task is asyncio.current_task()
                    ):
                        self._active.pop(client_id, None)

        async with self._lock:
            if self._backend_poisoned:
                raise StreamProtocolError("streaming backend is unavailable")
            live_job_key = (client_id, job_id)
            if live_job_key in self._live_job_ids:
                raise StreamProtocolError(
                    "job_id must be unique while a client task is live"
                )
            previous = self._active.get(client_id)
            if previous is not None and previous.job_id == job_id:
                raise StreamProtocolError(
                    "job_id must be unique while a client job is active"
                )
            if previous is not None:
                previous.task.cancel()
            client_emit_lock = self._emit_locks.setdefault(
                client_id,
                asyncio.Lock(),
            )
            task = asyncio.create_task(owned_run(client_emit_lock))
            self._live_job_ids.add(live_job_key)

            def retire_task(completed: asyncio.Task[StreamSummary]) -> None:
                self._live_job_ids.discard(live_job_key)
                if not completed.cancelled():
                    with contextlib.suppress(Exception):
                        completed.exception()
                self._retire_emit_lock_if_idle(client_id, client_emit_lock)

            task.add_done_callback(retire_task)
            handle = JobHandle(client_id=client_id, job_id=job_id, task=task)
            self._active[client_id] = handle
        return handle

    async def cancel(self, client_id: str, *, job_id: str | None = None) -> bool:
        _require_nonempty_string(client_id, "client_id")
        async with self._lock:
            handle = self._active.get(client_id)
            if handle is None or (job_id is not None and handle.job_id != job_id):
                return False
            self._active.pop(client_id, None)
            handle.task.cancel()
            return True

    async def active_job_id(self, client_id: str) -> str | None:
        _require_nonempty_string(client_id, "client_id")
        async with self._lock:
            handle = self._active.get(client_id)
            return None if handle is None else handle.job_id
