"""Bounded local TCP/NDJSON transport for :mod:`bench.streaming_service`.

This is a protocol-conformance surface, not a browser-ready serving stack.  One
TCP connection is one server-assigned client; callers cannot choose or resume a
client identity.  Commands and events are newline-delimited JSON objects.

Decoded frame bytes are base64 encoded on this wire. Their allowlisted media
type originates at the backend boundary and remains fixed for a job; the fake
backend's opaque bytes are deliberately marked non-renderable rather than
mislabeled as JPEG/PNG. Base64 adds roughly 33% wire overhead, so a future
same-origin binary WebSocket adapter should replace only this transport layer
once the project has a pinned web dependency.

``presented`` is a client assertion in the client's monotonic clock domain.  A
``presentation_recorded`` reply reports a distinct server receive timestamp and
never treats socket drain as browser paint.  At most two chunk presentations are
outstanding by default; this connection-bound acknowledgement window propagates
client backpressure into the existing bounded service queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from bench.streaming_service import (
    ALLOWED_FRAME_MEDIA_TYPES,
    DEFAULT_EMIT_TIMEOUT_S,
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_LATENT_FRAMES,
    DEFAULT_FRAME_MEDIA_TYPE,
    JobHandle,
    StreamEvent,
    StreamProtocolError,
    StreamingBackend,
    StreamingJobRegistry,
)


PROTOCOL_VERSION = "realtime-video.ndjson.v1"
OPAQUE_PAYLOAD_MEDIA_TYPE = DEFAULT_FRAME_MEDIA_TYPE
RENDERABLE_FRAME_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
DEFAULT_MAX_INPUT_BYTES = 16 * 1024
DEFAULT_MAX_PROMPT_BYTES = 8 * 1024
DEFAULT_MAX_WIRE_EVENT_BYTES = 24 * 1024 * 1024
DEFAULT_PRESENTATION_WINDOW_CHUNKS = 2
DEFAULT_DISCONNECT_TIMEOUT_S = 2.0
DEFAULT_CONTROL_SEND_TIMEOUT_S = 0.5
DEFAULT_MAX_JOBS_PER_CONNECTION = 1024
_WIRE_EVENT_OVERHEAD_BYTES = 8 * 1024
_MAX_JOB_ID_BYTES = 128
_MAX_SIGNED_64 = (1 << 63) - 1


class _CommandError(ValueError):
    def __init__(self, code: str, *, job_id: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.job_id = job_id


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _consume_task(task: asyncio.Future[Any]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


def _validate_job_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _CommandError("invalid_job_id")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise _CommandError("invalid_job_id")
    if len(encoded) > _MAX_JOB_ID_BYTES:
        raise _CommandError("invalid_job_id")
    return value


def _validate_int64(value: Any, code: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CommandError(code)
    lower = 0 if non_negative else -(_MAX_SIGNED_64 + 1)
    if value < lower or value > _MAX_SIGNED_64:
        raise _CommandError(code)
    return value


def _require_fields(
    message: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    if not required.issubset(message) or not set(message).issubset(required | optional):
        raise _CommandError("invalid_command")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


class _ClientSession:
    def __init__(
        self,
        *,
        server: "NDJSONStreamingServer",
        client_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.server = server
        self.client_id = client_id
        self.writer = writer
        self._send_lock = asyncio.Lock()
        self._state = asyncio.Condition()
        self._disconnect_lock = asyncio.Lock()
        self._closed = False
        self._current_job_id: str | None = None
        self._announcement_gate: asyncio.Event | None = None
        self._sent_chunks: dict[int, int] = {}
        self._outstanding_chunks: set[int] = set()
        self._acknowledged_chunks: set[int] = set()
        self._last_client_presented_ns: int | None = None
        self._last_server_received_ns: int | None = None
        self._presentation_records: dict[int, tuple[int, int]] = {}
        self._used_job_ids: set[str] = set()
        self._terminal_jobs: set[str] = set()
        self._cancel_gates: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()

    @property
    def closed(self) -> bool:
        return self._closed

    async def send_message(
        self,
        message: dict[str, Any],
        *,
        expected_job_id: str | None = None,
        after_write: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        try:
            encoded = (
                json.dumps(
                    message,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
            raise ConnectionError("transport serialization failed") from exc
        if len(encoded) > self.server.max_wire_event_bytes:
            raise ConnectionError("outbound event exceeds the wire limit")

        async with self._send_lock:
            if self._closed:
                raise ConnectionError("client disconnected")
            if expected_job_id is not None and self._current_job_id != expected_job_id:
                return False
            self.writer.write(encoded)
            # StreamWriter.write queues bytes synchronously.  Once it returns,
            # a peer may read them even while drain() is still applying local
            # flow control, so delivery eligibility must begin here.
            if after_write is not None:
                await after_write()
            await self.writer.drain()
        return True

    async def send_control_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        expected_job_id: str | None = None,
    ) -> bool:
        async def send_all() -> bool:
            for message in messages:
                sent = await self.send_message(
                    message,
                    expected_job_id=expected_job_id,
                )
                if not sent:
                    return False
            return True

        try:
            return await asyncio.wait_for(
                send_all(),
                timeout=self.server.control_send_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ConnectionError("control send timed out") from exc

    async def send_control_message(
        self,
        message: dict[str, Any],
        *,
        expected_job_id: str | None = None,
    ) -> bool:
        return await self.send_control_messages(
            [message],
            expected_job_id=expected_job_id,
        )

    def _spawn_maintenance(self, operation: Awaitable[None]) -> None:
        task = asyncio.create_task(operation)
        self._maintenance_tasks.add(task)

        def retire(completed: asyncio.Task[Any]) -> None:
            self._maintenance_tasks.discard(completed)
            _consume_task(completed)

        task.add_done_callback(retire)

    def _clear_presentation_state_locked(self) -> None:
        self._sent_chunks.clear()
        self._outstanding_chunks.clear()
        self._acknowledged_chunks.clear()
        self._presentation_records.clear()
        self._last_client_presented_ns = None
        self._last_server_received_ns = None

    async def _retire_failed_job(
        self,
        job_id: str,
        completed: asyncio.Task[Any],
    ) -> None:
        if not completed.cancelled():
            _consume_task(completed)
        async with self._state:
            if self._closed or self._current_job_id != job_id:
                return
            if job_id in self._terminal_jobs:
                return
            had_outstanding_presentation = bool(self._outstanding_chunks)

        error_code = (
            "client_backpressure_timeout"
            if had_outstanding_presentation
            else "transport_failure"
        )
        terminal: dict[str, Any]
        if completed.cancelled():
            terminal = {
                "type": "stream_event",
                "kind": "job_cancelled",
                "job_id": job_id,
            }
        else:
            terminal = {
                "type": "stream_event",
                "kind": "job_failed",
                "job_id": job_id,
                "error_code": error_code,
            }

        send_failed = False
        try:
            await self.send_control_message(
                terminal,
                expected_job_id=job_id,
            )
        except (BrokenPipeError, ConnectionError, OSError):
            send_failed = True
        async with self._state:
            if self._current_job_id == job_id:
                self._current_job_id = None
                self._announcement_gate = None
                self._clear_presentation_state_locked()
                self._terminal_jobs.discard(job_id)
                self._state.notify_all()
        if send_failed:
            await self.disconnect()

    async def emit_event(self, event: StreamEvent) -> None:
        async with self._state:
            if self._closed:
                raise ConnectionError("client disconnected")
            if event.job_id != self._current_job_id:
                return
            gate = self._announcement_gate
        if gate is None:
            return
        await gate.wait()
        if event.kind == "job_cancelled":
            async with self._state:
                cancel_gate = self._cancel_gates.get(event.job_id)
            if cancel_gate is not None:
                await cancel_gate.wait()

        reserved_chunk = False
        if event.kind == "chunk_ready":
            if event.chunk_index is None or event.ready_ns is None:
                raise ConnectionError("chunk event is incomplete")
            async with self._state:
                await self._state.wait_for(
                    lambda: self._closed
                    or self._current_job_id != event.job_id
                    or len(self._outstanding_chunks)
                    < self.server.presentation_window_chunks
                )
                if self._closed:
                    raise ConnectionError("client disconnected")
                if self._current_job_id != event.job_id:
                    return
                self._outstanding_chunks.add(event.chunk_index)
                reserved_chunk = True

        async def mark_chunk_written() -> None:
            if event.chunk_index is None or event.ready_ns is None:
                return
            async with self._state:
                if self._closed or self._current_job_id != event.job_id:
                    self._outstanding_chunks.discard(event.chunk_index)
                else:
                    self._sent_chunks[event.chunk_index] = event.ready_ns
                self._state.notify_all()

        message = {"type": "stream_event", **event.to_dict(include_payloads=True)}
        if event.kind == "chunk_ready":
            if event.frame_media_type not in ALLOWED_FRAME_MEDIA_TYPES:
                raise ConnectionError("chunk event media type is missing or unsupported")
            message.update(
                {
                    "payload_media_type": event.frame_media_type,
                    "payload_encoding": "base64",
                    "renderable": event.frame_media_type
                    in RENDERABLE_FRAME_MEDIA_TYPES,
                }
            )
        try:
            sent = await self.send_message(
                message,
                expected_job_id=event.job_id,
                after_write=mark_chunk_written if reserved_chunk else None,
            )
        except BaseException:
            if reserved_chunk and event.chunk_index is not None:
                async with self._state:
                    self._sent_chunks.pop(event.chunk_index, None)
                    self._outstanding_chunks.discard(event.chunk_index)
                    self._acknowledged_chunks.discard(event.chunk_index)
                    self._presentation_records.pop(event.chunk_index, None)
                    if self._presentation_records:
                        self._last_client_presented_ns = max(
                            item[0] for item in self._presentation_records.values()
                        )
                        self._last_server_received_ns = max(
                            item[1] for item in self._presentation_records.values()
                        )
                    else:
                        self._last_client_presented_ns = None
                        self._last_server_received_ns = None
                    self._state.notify_all()
            raise
        if not sent and reserved_chunk and event.chunk_index is not None:
            async with self._state:
                self._sent_chunks.pop(event.chunk_index, None)
                self._outstanding_chunks.discard(event.chunk_index)
                self._state.notify_all()
        if sent and event.kind in {"job_completed", "job_failed", "job_cancelled"}:
            async with self._state:
                if self._current_job_id == event.job_id:
                    self._terminal_jobs.add(event.job_id)

    async def handle_raw_message(self, raw: bytes) -> None:
        try:
            text = raw.decode("utf-8")
            message = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            raise _CommandError("invalid_json")
        if not isinstance(message, dict):
            raise _CommandError("invalid_command")
        command = message.get("type")
        if command == "start":
            await self._handle_start(message)
        elif command == "cancel":
            await self._handle_cancel(message)
        elif command == "presented":
            await self._handle_presented(message)
        else:
            raise _CommandError("invalid_command")

    async def _handle_start(self, message: dict[str, Any]) -> None:
        _require_fields(
            message,
            required={"type", "prompt", "seed"},
            optional={"latent_frames"},
        )
        prompt = message["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise _CommandError("invalid_prompt")
        try:
            prompt_bytes = prompt.encode("utf-8")
        except UnicodeEncodeError:
            raise _CommandError("invalid_prompt")
        if len(prompt_bytes) > self.server.max_prompt_bytes:
            raise _CommandError("prompt_too_large")
        seed = _validate_int64(message["seed"], "invalid_seed")
        latent_frames = message.get("latent_frames", 21)
        if isinstance(latent_frames, bool) or not isinstance(latent_frames, int):
            raise _CommandError("invalid_latent_frames")
        if latent_frames <= 0 or latent_frames > self.server.max_latent_frames:
            raise _CommandError("invalid_latent_frames")
        async with self._state:
            if len(self._used_job_ids) >= self.server.max_jobs_per_connection:
                raise _CommandError("start_rejected")

        try:
            handle = await self.server.registry.start(
                client_id=self.client_id,
                prompt=prompt,
                seed=seed,
                backend=self.server.backend,
                latent_frames=latent_frames,
            )
        except (StreamProtocolError, ValueError):
            raise _CommandError("start_rejected")
        except Exception:
            raise _CommandError("start_rejected")

        try:
            job_id = _validate_job_id(handle.job_id)
        except _CommandError:
            await self.server.registry.cancel(
                self.client_id,
                job_id=handle.job_id,
            )
            raise _CommandError("start_rejected")
        async with self._state:
            reused_job_id = job_id in self._used_job_ids
            if reused_job_id and self._current_job_id == job_id:
                self._current_job_id = None
                self._announcement_gate = None
                self._clear_presentation_state_locked()
                self._state.notify_all()
        if reused_job_id:
            await self.server.registry.cancel(self.client_id, job_id=job_id)
            raise _CommandError("start_rejected")

        gate = asyncio.Event()
        async with self._state:
            self._used_job_ids.add(job_id)
            self._current_job_id = job_id
            self._announcement_gate = gate
            self._terminal_jobs.clear()
            for cancel_gate in self._cancel_gates.values():
                cancel_gate.set()
            self._cancel_gates.clear()
            self._clear_presentation_state_locked()
            self._state.notify_all()
        self._tasks[job_id] = handle.task

        def retire(completed: asyncio.Task[Any]) -> None:
            if self._tasks.get(job_id) is handle.task:
                self._tasks.pop(job_id, None)
            self._spawn_maintenance(self._retire_failed_job(job_id, completed))

        handle.task.add_done_callback(retire)
        try:
            sent = await self.send_control_message(
                {"type": "start_accepted", "job_id": job_id},
                expected_job_id=job_id,
            )
            if not sent:
                raise ConnectionError("start acknowledgement became stale")
        finally:
            # Service events cannot pass this barrier until socket drain for the
            # job-fenced start acknowledgement has completed.
            gate.set()

    async def _handle_cancel(self, message: dict[str, Any]) -> None:
        _require_fields(message, required={"type", "job_id"})
        job_id = _validate_job_id(message["job_id"])
        cancel_gate = asyncio.Event()
        async with self._state:
            self._cancel_gates[job_id] = cancel_gate
        try:
            # Cancellation starts before any socket wait.  Its terminal event
            # waits on cancel_gate so truthful control replies remain ordered.
            cancelled = await self.server.registry.cancel(
                self.client_id,
                job_id=job_id,
            )
            replies: list[dict[str, Any]] = []
            if cancelled:
                replies.append({"type": "cancel_accepted", "job_id": job_id})
            replies.append(
                {"type": "cancel_result", "job_id": job_id, "cancelled": cancelled}
            )
            await self.send_control_messages(replies)
        finally:
            cancel_gate.set()
            async with self._state:
                if self._cancel_gates.get(job_id) is cancel_gate:
                    self._cancel_gates.pop(job_id, None)

    async def _handle_presented(self, message: dict[str, Any]) -> None:
        _require_fields(
            message,
            required={"type", "job_id", "chunk_index", "client_presented_ns"},
        )
        job_id = _validate_job_id(message["job_id"])
        chunk_index = _validate_int64(
            message["chunk_index"], "invalid_chunk_index", non_negative=True
        )
        client_presented_ns = _validate_int64(
            message["client_presented_ns"],
            "invalid_client_presented_ns",
            non_negative=True,
        )

        try:
            server_received_ns = self.server.clock_ns()
        except Exception:
            raise _CommandError("server_clock_failed", job_id=job_id)
        if (
            isinstance(server_received_ns, bool)
            or not isinstance(server_received_ns, int)
            or server_received_ns < 0
        ):
            raise _CommandError("server_clock_failed", job_id=job_id)

        async with self._state:
            if job_id != self._current_job_id:
                raise _CommandError("stale_job", job_id=job_id)
            if chunk_index not in self._sent_chunks:
                raise _CommandError("chunk_not_sent", job_id=job_id)
            if chunk_index in self._acknowledged_chunks:
                raise _CommandError("duplicate_presentation", job_id=job_id)
            if (
                self._last_client_presented_ns is not None
                and client_presented_ns <= self._last_client_presented_ns
            ):
                raise _CommandError("non_monotonic_presentation", job_id=job_id)
            ready_ns = self._sent_chunks[chunk_index]
            if server_received_ns < ready_ns or (
                self._last_server_received_ns is not None
                and server_received_ns <= self._last_server_received_ns
            ):
                raise _CommandError("server_clock_failed", job_id=job_id)
            self._acknowledged_chunks.add(chunk_index)
            self._outstanding_chunks.discard(chunk_index)
            self._last_client_presented_ns = client_presented_ns
            self._last_server_received_ns = server_received_ns
            self._presentation_records[chunk_index] = (
                client_presented_ns,
                server_received_ns,
            )
            self._state.notify_all()
        await self.send_control_message(
            {
                "type": "presentation_recorded",
                "job_id": job_id,
                "chunk_index": chunk_index,
                "server_chunk_ready_ns": ready_ns,
                "client_presented_ns": client_presented_ns,
                "client_clock_domain": "client_monotonic",
                "server_received_ns": server_received_ns,
                "server_clock_domain": "server_monotonic",
                "timing_semantics": "client_reported_presentation",
                "not_server_emit_completion": True,
            }
        )

    async def disconnect(self) -> None:
        async with self._disconnect_lock:
            if self._closed:
                return
            async with self._state:
                self._closed = True
                gate = self._announcement_gate
                current_job_id = self._current_job_id
                self._current_job_id = None
                self._announcement_gate = None
                for cancel_gate in self._cancel_gates.values():
                    cancel_gate.set()
                self._cancel_gates.clear()
                self._clear_presentation_state_locked()
                self._terminal_jobs.clear()
                self._state.notify_all()
            if gate is not None:
                gate.set()
            self.writer.close()

            with contextlib.suppress(Exception):
                await self.server.registry.cancel(
                    self.client_id,
                    job_id=current_job_id,
                )
            tasks = [task for task in self._tasks.values() if not task.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=self.server.disconnect_timeout_s,
                )
                for task in done:
                    _consume_task(task)
                for task in pending:
                    task.add_done_callback(_consume_task)
            with contextlib.suppress(BrokenPipeError, ConnectionError, OSError):
                await self.writer.wait_closed()


class NDJSONStreamingServer:
    """One-client-per-connection local network adapter.

    The server binds to loopback by default and has no authentication, TLS,
    HTTP, browser origin checks, or resumable identity.  It must not be exposed
    to an untrusted network.
    """

    def __init__(
        self,
        *,
        backend: StreamingBackend,
        host: str = "127.0.0.1",
        port: int = 0,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        max_wire_event_bytes: int = DEFAULT_MAX_WIRE_EVENT_BYTES,
        presentation_window_chunks: int = DEFAULT_PRESENTATION_WINDOW_CHUNKS,
        disconnect_timeout_s: float = DEFAULT_DISCONNECT_TIMEOUT_S,
        control_send_timeout_s: float = DEFAULT_CONTROL_SEND_TIMEOUT_S,
        emit_timeout_s: float = DEFAULT_EMIT_TIMEOUT_S,
        max_jobs_per_connection: int = DEFAULT_MAX_JOBS_PER_CONNECTION,
        max_latent_frames: int = DEFAULT_MAX_LATENT_FRAMES,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        job_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if host != "localhost":
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
            if not is_loopback:
                raise ValueError("unauthenticated transport host must be loopback")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer from 0 through 65535")
        if not callable(getattr(backend, "stream", None)):
            raise ValueError("backend must provide stream(request)")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        max_input_bytes = _require_positive_int(max_input_bytes, "max_input_bytes")
        max_prompt_bytes = _require_positive_int(max_prompt_bytes, "max_prompt_bytes")
        max_wire_event_bytes = _require_positive_int(
            max_wire_event_bytes, "max_wire_event_bytes"
        )
        presentation_window_chunks = _require_positive_int(
            presentation_window_chunks, "presentation_window_chunks"
        )
        max_jobs_per_connection = _require_positive_int(
            max_jobs_per_connection, "max_jobs_per_connection"
        )
        max_latent_frames = _require_positive_int(
            max_latent_frames, "max_latent_frames"
        )
        max_chunk_bytes = _require_positive_int(max_chunk_bytes, "max_chunk_bytes")
        disconnect_timeout_s = _require_positive_number(
            disconnect_timeout_s, "disconnect_timeout_s"
        )
        control_send_timeout_s = _require_positive_number(
            control_send_timeout_s, "control_send_timeout_s"
        )
        emit_timeout_s = _require_positive_number(emit_timeout_s, "emit_timeout_s")
        if max_prompt_bytes > max_input_bytes:
            raise ValueError("max_prompt_bytes cannot exceed max_input_bytes")
        if max_wire_event_bytes <= _WIRE_EVENT_OVERHEAD_BYTES:
            raise ValueError("max_wire_event_bytes is too small for event metadata")
        conservative_payload_capacity = (
            (max_wire_event_bytes - _WIRE_EVENT_OVERHEAD_BYTES) // 4
        ) * 3
        if max_chunk_bytes > conservative_payload_capacity:
            raise ValueError(
                "max_chunk_bytes cannot fit in the configured base64 wire event limit"
            )

        self.backend = backend
        self.host = host
        self.port = port
        self.max_input_bytes = max_input_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_wire_event_bytes = max_wire_event_bytes
        self.presentation_window_chunks = presentation_window_chunks
        self.max_jobs_per_connection = max_jobs_per_connection
        self.disconnect_timeout_s = disconnect_timeout_s
        self.control_send_timeout_s = control_send_timeout_s
        self.max_latent_frames = max_latent_frames
        self.max_chunk_bytes = max_chunk_bytes
        self.clock_ns = clock_ns
        self._server: asyncio.AbstractServer | None = None
        self._clients: dict[str, _ClientSession] = {}
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self.registry = StreamingJobRegistry(
            emit=self._emit_to_client,
            clock_ns=clock_ns,
            job_id_factory=job_id_factory,
            emit_timeout_s=emit_timeout_s,
            max_latent_frames=max_latent_frames,
            max_chunk_bytes=max_chunk_bytes,
        )

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("server is not running")
        socket_name = self._server.sockets[0].getsockname()
        return str(socket_name[0]), int(socket_name[1])

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("server is already running")
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            limit=self.max_input_bytes + 1,
        )

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        sessions = list(self._clients.values())
        if sessions:
            await asyncio.gather(
                *(session.disconnect() for session in sessions),
                return_exceptions=True,
            )
        tasks = [task for task in self._client_tasks if not task.done()]
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.disconnect_timeout_s,
            )
            for task in done:
                _consume_task(task)
            for task in pending:
                task.cancel()
                task.add_done_callback(_consume_task)

    async def _emit_to_client(self, client_id: str, event: StreamEvent) -> None:
        session = self._clients.get(client_id)
        if session is None or session.closed:
            raise ConnectionError("client disconnected")
        await session.emit_event(event)

    async def _send_command_error(
        self,
        session: _ClientSession,
        error: _CommandError,
    ) -> None:
        message: dict[str, Any] = {"type": "command_error", "code": error.code}
        if error.job_id is not None:
            message["job_id"] = error.job_id
        await session.send_control_message(message)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        client_id = uuid.uuid4().hex
        session = _ClientSession(server=self, client_id=client_id, writer=writer)
        self._clients[client_id] = session
        try:
            await session.send_control_message(
                {
                    "type": "connected",
                    "protocol": PROTOCOL_VERSION,
                    "client_id": client_id,
                    "presentation_window_chunks": self.presentation_window_chunks,
                    "payload_encoding": "base64",
                    "media_type_semantics": "backend_declared_per_chunk",
                }
            )
            while True:
                try:
                    raw = await reader.readline()
                except ValueError:
                    with contextlib.suppress(Exception):
                        await self._send_command_error(
                            session,
                            _CommandError("message_too_large"),
                        )
                    break
                if not raw:
                    break
                if len(raw) > self.max_input_bytes:
                    with contextlib.suppress(Exception):
                        await self._send_command_error(
                            session,
                            _CommandError("message_too_large"),
                        )
                    break
                if not raw.endswith(b"\n"):
                    with contextlib.suppress(Exception):
                        await self._send_command_error(
                            session,
                            _CommandError("invalid_framing"),
                        )
                    break
                try:
                    await session.handle_raw_message(raw[:-1])
                except _CommandError as exc:
                    await self._send_command_error(session, exc)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            self._clients.pop(client_id, None)
            await session.disconnect()
            if task is not None:
                self._client_tasks.discard(task)


__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MAX_WIRE_EVENT_BYTES",
    "DEFAULT_PRESENTATION_WINDOW_CHUNKS",
    "DEFAULT_CONTROL_SEND_TIMEOUT_S",
    "DEFAULT_MAX_JOBS_PER_CONNECTION",
    "NDJSONStreamingServer",
    "OPAQUE_PAYLOAD_MEDIA_TYPE",
    "PROTOCOL_VERSION",
    "RENDERABLE_FRAME_MEDIA_TYPES",
]
