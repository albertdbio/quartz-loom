"""Same-origin loopback WebSocket transport for the bounded streaming service.

JSON control messages and event metadata use WebSocket text frames.  A
``chunk`` metadata message is followed atomically by exactly ``frame_count``
binary WebSocket messages.  A delivery becomes acknowledgement-eligible only
after the complete binary group has been written.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import math
import secrets
import socket
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from bench.prompt_resolution import IdentityPromptResolver, PromptResolver
from bench.streaming_service import (
    ALLOWED_FRAME_MEDIA_TYPES,
    DEFAULT_BACKEND_CHUNK_TIMEOUT_S,
    DEFAULT_BACKEND_CLOSE_TIMEOUT_S,
    DEFAULT_MAX_CHUNK_BYTES,
    JobHandle,
    StreamEvent,
    StreamProtocolError,
    StreamingBackend,
    StreamingJobRegistry,
)


PROTOCOL_VERSION = "realtime-video.websocket.v1"
RENDERABLE_FRAME_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
DEFAULT_MAX_CONTROL_BYTES = 16 * 1024
DEFAULT_MAX_PROMPT_BYTES = 4096
DEFAULT_PRESENTATION_WINDOW_CHUNKS = 2
DEFAULT_DISCONNECT_TIMEOUT_S = 2.0
DEFAULT_EMIT_TIMEOUT_S = 2.0
DEFAULT_CONTROL_SEND_TIMEOUT_S = 0.5
DEFAULT_MAX_JOBS_PER_CONNECTION = 1024
DEFAULT_SESSION_TTL_S = 300.0
DEFAULT_MAX_ISSUED_SESSIONS = 64
DEFAULT_PROMPT_RESOLUTION_TIMEOUT_S = 4.0
FIXED_LATENT_FRAMES = 21
EXPECTED_RGB_FRAMES = 81
_SESSION_COOKIE = "rtv_session_nonce"
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_ID_BYTES = 128
_RETIRED_DELIVERY_LIMIT = 64
_MAX_CONCURRENT_PROMPT_RESOLUTIONS = 4
_STATIC_DIR = Path(__file__).with_name("static")
_FAKE_BACKEND_DESCRIPTION = "This local fake backend emits valid PNG frames; it does not run the generation model."
_DEMO_BACKEND_DESCRIPTIONS = {
    "fake": _FAKE_BACKEND_DESCRIPTION,
    "cf1": (
        "This opt-in CF++1 H100 backend streams frames from the frozen generation "
        "runtime."
    ),
}


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


def _validate_int64(value: Any, code: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CommandError(code)
    lower = 0 if non_negative else -(_MAX_SIGNED_64 + 1)
    if value < lower or value > _MAX_SIGNED_64:
        raise _CommandError(code)
    return value


def _validate_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise _CommandError(code)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise _CommandError(code)
    if len(encoded) > _MAX_ID_BYTES:
        raise _CommandError(code)
    return value


def _validate_prompt(value: Any, max_prompt_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _CommandError("invalid_prompt")
    try:
        prompt_bytes = value.encode("utf-8")
    except UnicodeEncodeError:
        raise _CommandError("invalid_prompt")
    if len(prompt_bytes) > max_prompt_bytes:
        raise _CommandError("prompt_too_large")
    return value


def _require_fields(
    message: dict[str, Any],
    *,
    required: set[str],
) -> None:
    if set(message) != required:
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


def _encode_json(message: dict[str, Any]) -> str:
    try:
        return json.dumps(
            message,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ConnectionError("transport serialization failed") from exc


def _consume_task(task: asyncio.Future[Any]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


@dataclass
class _Delivery:
    job_id: str
    chunk_index: int
    ready_ns: int
    acknowledged: bool = False


class _WebSocketSession:
    def __init__(
        self,
        *,
        server: "BrowserStreamingServer",
        client_id: str,
        ws: web.WebSocketResponse,
    ) -> None:
        self.server = server
        self.client_id = client_id
        self.ws = ws
        self._send_lock = asyncio.Lock()
        self._state = asyncio.Condition()
        self._disconnect_lock = asyncio.Lock()
        self._closed = False
        self._current_job_id: str | None = None
        self._announcement_gate: asyncio.Event | None = None
        self._pending_start_resolution: asyncio.Event | None = None
        self._deliveries: dict[str, _Delivery] = {}
        self._outstanding_delivery_ids: set[str] = set()
        self._retired_delivery_ids: deque[str] = deque()
        self._retired_delivery_set: set[str] = set()
        self._last_client_presented_ns: int | None = None
        self._last_server_received_ns: int | None = None
        self._used_job_ids: set[str] = set()
        self._used_delivery_ids: set[str] = set()
        self._used_prompt_resolution_ids: set[str] = set()
        self._active_prompt_resolution_id: str | None = None
        self._prompt_resolution_tasks: set[asyncio.Task[Any]] = set()
        self._resolved_prompts: dict[str, dict[str, Any]] = {}
        self._terminal_jobs: set[str] = set()
        self._cancel_gates: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()

    @property
    def closed(self) -> bool:
        return self._closed

    async def send_json(
        self,
        message: dict[str, Any],
        *,
        expected_job_id: str | None = None,
        expected_prompt_resolution_id: str | None = None,
    ) -> bool:
        encoded = _encode_json(message)

        async def send() -> bool:
            async with self._send_lock:
                async with self._state:
                    if self._closed:
                        raise ConnectionError("client disconnected")
                    if (
                        expected_job_id is not None
                        and self._current_job_id != expected_job_id
                    ):
                        return False
                    if (
                        expected_prompt_resolution_id is not None
                        and self._active_prompt_resolution_id
                        != expected_prompt_resolution_id
                    ):
                        return False
                await self.ws.send_str(encoded)
            return True

        try:
            return await asyncio.wait_for(
                send(),
                timeout=self.server.control_send_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ConnectionError("control send timed out") from exc

    async def _resolve_event_gate(self, job_id: str) -> asyncio.Event | None:
        while True:
            async with self._state:
                if self._closed:
                    raise ConnectionError("client disconnected")
                if self._current_job_id == job_id:
                    return self._announcement_gate
                pending = self._pending_start_resolution
                if pending is None:
                    return None
            await pending.wait()

    def _retire_delivery_id_locked(self, delivery_id: str) -> None:
        if delivery_id in self._retired_delivery_set:
            return
        if len(self._retired_delivery_ids) >= _RETIRED_DELIVERY_LIMIT:
            expired = self._retired_delivery_ids.popleft()
            self._retired_delivery_set.discard(expired)
        self._retired_delivery_ids.append(delivery_id)
        self._retired_delivery_set.add(delivery_id)

    def _retire_current_deliveries_locked(self) -> None:
        for delivery_id in self._deliveries.keys() | self._outstanding_delivery_ids:
            self._retire_delivery_id_locked(delivery_id)
        self._deliveries.clear()
        self._outstanding_delivery_ids.clear()

    def _spawn_maintenance(self, operation: Awaitable[None]) -> None:
        task = asyncio.create_task(operation)
        self._maintenance_tasks.add(task)

        def retire(completed: asyncio.Task[Any]) -> None:
            self._maintenance_tasks.discard(completed)
            _consume_task(completed)

        task.add_done_callback(retire)

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
            had_outstanding_presentation = bool(self._outstanding_delivery_ids)

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
                "error_code": (
                    "client_backpressure_timeout"
                    if had_outstanding_presentation
                    else "transport_failure"
                ),
            }

        send_failed = False
        try:
            await self.send_json(terminal, expected_job_id=job_id)
        except (ConnectionError, OSError, RuntimeError):
            send_failed = True

        async with self._state:
            if self._current_job_id == job_id:
                self._retire_current_deliveries_locked()
                self._current_job_id = None
                self._announcement_gate = None
                self._last_client_presented_ns = None
                self._last_server_received_ns = None
                self._state.notify_all()
        if send_failed:
            await self.disconnect()

    def _new_delivery_id_locked(self) -> str:
        try:
            delivery_id = _validate_id(
                self.server.delivery_id_factory(),
                "invalid_delivery_id",
            )
        except Exception as exc:
            raise ConnectionError(
                "delivery ID factory returned an invalid value"
            ) from exc
        if (
            delivery_id in self._deliveries
            or delivery_id in self._outstanding_delivery_ids
            or delivery_id in self._retired_delivery_set
            or delivery_id in self._used_delivery_ids
        ):
            raise ConnectionError("delivery ID factory returned a reused value")
        self._used_delivery_ids.add(delivery_id)
        return delivery_id

    async def emit_event(self, event: StreamEvent) -> None:
        gate = await self._resolve_event_gate(event.job_id)
        if gate is None:
            return
        await gate.wait()

        if event.kind == "job_cancelled":
            async with self._state:
                cancel_gate = self._cancel_gates.get(event.job_id)
            if cancel_gate is not None:
                await cancel_gate.wait()

        if event.kind != "chunk_ready":
            sent = await self.send_json(
                {"type": "stream_event", **event.to_dict(include_payloads=False)},
                expected_job_id=event.job_id,
            )
            if sent and event.kind in {"job_completed", "job_failed", "job_cancelled"}:
                async with self._state:
                    if self._current_job_id == event.job_id:
                        self._terminal_jobs.add(event.job_id)
            return

        if (
            event.chunk_index is None
            or event.ready_ns is None
            or event.frame_payloads is None
            or event.frame_media_type not in ALLOWED_FRAME_MEDIA_TYPES
        ):
            raise ConnectionError("chunk event is incomplete or unsupported")

        async with self._state:
            await self._state.wait_for(
                lambda: (
                    self._closed
                    or self._current_job_id != event.job_id
                    or len(self._outstanding_delivery_ids)
                    < self.server.presentation_window_chunks
                )
            )
            if self._closed:
                raise ConnectionError("client disconnected")
            if self._current_job_id != event.job_id:
                return
            delivery_id = self._new_delivery_id_locked()
            self._outstanding_delivery_ids.add(delivery_id)

        header = {
            "type": "chunk",
            **event.to_dict(include_payloads=False),
            "payload_encoding": "websocket-binary",
            "renderable": event.frame_media_type in RENDERABLE_FRAME_MEDIA_TYPES,
        }
        encoded_header = _encode_json(header)
        encoded_commit = _encode_json(
            {
                "type": "chunk_committed",
                "job_id": event.job_id,
                "chunk_index": event.chunk_index,
                "delivery_id": delivery_id,
            }
        )
        try:
            async with self._send_lock:
                async with self._state:
                    if self._closed:
                        raise ConnectionError("client disconnected")
                    if self._current_job_id != event.job_id:
                        self._outstanding_delivery_ids.discard(delivery_id)
                        self._retire_delivery_id_locked(delivery_id)
                        self._state.notify_all()
                        return
                await self.ws.send_str(encoded_header)
                for payload in event.frame_payloads:
                    await self.ws.send_bytes(payload)
                should_commit = False
                async with self._state:
                    if self._closed or self._current_job_id != event.job_id:
                        self._outstanding_delivery_ids.discard(delivery_id)
                        self._retire_delivery_id_locked(delivery_id)
                    else:
                        self._deliveries[delivery_id] = _Delivery(
                            job_id=event.job_id,
                            chunk_index=event.chunk_index,
                            ready_ns=event.ready_ns,
                        )
                        should_commit = True
                    self._state.notify_all()
                if should_commit:
                    await self.ws.send_str(encoded_commit)
        except BaseException:
            async with self._state:
                self._deliveries.pop(delivery_id, None)
                self._outstanding_delivery_ids.discard(delivery_id)
                self._retire_delivery_id_locked(delivery_id)
                self._state.notify_all()
            raise

    def _validated_prompt_resolution(
        self,
        value: Any,
        *,
        raw_prompt: str,
    ) -> dict[str, Any]:
        effective_prompt = _validate_prompt(
            getattr(value, "effective_prompt", None),
            self.server.max_prompt_bytes,
        )
        raw_sha256 = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        effective_sha256 = hashlib.sha256(
            effective_prompt.encode("utf-8")
        ).hexdigest()
        transform_id = getattr(value, "transform_id", None)
        try:
            encoded_transform_id = transform_id.encode("utf-8")
        except (AttributeError, UnicodeEncodeError):
            raise ValueError("prompt resolver transform identity is invalid")
        if not transform_id or len(encoded_transform_id) > _MAX_ID_BYTES:
            raise ValueError("prompt resolver transform identity is invalid")
        changed = getattr(value, "changed", None)
        if (
            getattr(value, "raw_prompt", None) != raw_prompt
            or getattr(value, "raw_prompt_sha256", None) != raw_sha256
            or getattr(value, "effective_prompt_sha256", None) != effective_sha256
            or not isinstance(changed, bool)
            or changed != (raw_prompt != effective_prompt)
        ):
            raise ValueError("prompt resolver provenance is invalid")
        return {
            "effective_prompt": effective_prompt,
            "raw_prompt_sha256": raw_sha256,
            "effective_prompt_sha256": effective_sha256,
            "prompt_transform_id": transform_id,
            "prompt_changed": changed,
        }

    async def _run_prompt_resolution(
        self,
        *,
        request_id: str,
        raw_prompt: str,
    ) -> None:
        started_ns = time.monotonic_ns()
        try:
            value = await asyncio.wait_for(
                self.server.prompt_resolver.resolve(raw_prompt),
                timeout=self.server.prompt_resolution_timeout_s,
            )
            resolution = self._validated_prompt_resolution(
                value,
                raw_prompt=raw_prompt,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            sent = await self.send_json(
                {
                    "type": "prompt_resolution_failed",
                    "request_id": request_id,
                    "error_code": "prompt_resolution_failed",
                },
                expected_prompt_resolution_id=request_id,
            )
            if not sent:
                return
        else:
            resolution_ms = max(0.0, (time.monotonic_ns() - started_ns) / 1_000_000)
            async with self._state:
                if self._active_prompt_resolution_id != request_id:
                    return
                self._resolved_prompts[request_id] = dict(resolution)
            sent = await self.send_json(
                {
                    "type": "prompt_resolved",
                    "request_id": request_id,
                    **resolution,
                    "prompt_resolution_ms": resolution_ms,
                },
                expected_prompt_resolution_id=request_id,
            )
            if not sent:
                async with self._state:
                    self._resolved_prompts.pop(request_id, None)
                return
        finally:
            async with self._state:
                if self._active_prompt_resolution_id == request_id:
                    self._active_prompt_resolution_id = None
                self._state.notify_all()

    async def _handle_resolve_prompt(self, message: dict[str, Any]) -> None:
        _require_fields(message, required={"type", "request_id", "prompt"})
        request_id = _validate_id(
            message["request_id"],
            "invalid_prompt_resolution_id",
        )
        raw_prompt = _validate_prompt(message["prompt"], self.server.max_prompt_bytes)
        async with self._state:
            unfinished = sum(
                not task.done() for task in self._prompt_resolution_tasks
            )
            if (
                request_id in self._used_prompt_resolution_ids
                or len(self._used_prompt_resolution_ids)
                >= self.server.max_jobs_per_connection
                or unfinished >= _MAX_CONCURRENT_PROMPT_RESOLUTIONS
            ):
                raise _CommandError("prompt_resolution_rejected")
            previous_tasks = [
                task
                for task in self._prompt_resolution_tasks
                if not task.done()
            ]
            self._used_prompt_resolution_ids.add(request_id)
            self._active_prompt_resolution_id = request_id
            task = asyncio.create_task(
                self._run_prompt_resolution(
                    request_id=request_id,
                    raw_prompt=raw_prompt,
                )
            )
            self._prompt_resolution_tasks.add(task)
        for previous in previous_tasks:
            previous.cancel()

        def retire(completed: asyncio.Task[Any]) -> None:
            self._prompt_resolution_tasks.discard(completed)
            _consume_task(completed)

        task.add_done_callback(retire)

    async def handle_text(self, raw: str) -> None:
        try:
            raw_bytes = raw.encode("utf-8")
        except UnicodeEncodeError:
            raise _CommandError("invalid_json")
        if len(raw_bytes) > self.server.max_control_bytes:
            raise _CommandError("message_too_large")
        try:
            message = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError):
            raise _CommandError("invalid_json")
        if not isinstance(message, dict):
            raise _CommandError("invalid_command")
        command = message.get("type")
        if command == "start":
            await self._handle_start(message)
        elif command == "resolve_prompt":
            await self._handle_resolve_prompt(message)
        elif command == "cancel":
            await self._handle_cancel(message)
        elif command == "presented":
            await self._handle_presented(message)
        else:
            raise _CommandError("invalid_command")

    async def _handle_start(self, message: dict[str, Any]) -> None:
        fields = set(message)
        if fields not in (
            {"type", "prompt", "seed"},
            {"type", "prompt", "prompt_resolution_id", "seed"},
        ):
            raise _CommandError("invalid_command")
        prompt = _validate_prompt(message["prompt"], self.server.max_prompt_bytes)
        seed = _validate_int64(message["seed"], "invalid_seed")
        prompt_resolution_id: str | None = None
        prompt_provenance: dict[str, Any] | None = None
        if "prompt_resolution_id" in message:
            prompt_resolution_id = _validate_id(
                message["prompt_resolution_id"],
                "invalid_prompt_resolution_id",
            )
            async with self._state:
                prompt_provenance = self._resolved_prompts.pop(
                    prompt_resolution_id,
                    None,
                )
            if (
                prompt_provenance is None
                or prompt_provenance.get("effective_prompt") != prompt
            ):
                raise _CommandError("prompt_resolution_mismatch")

        start_resolution = asyncio.Event()
        handle: JobHandle | None = None
        async with self._state:
            if len(self._used_job_ids) >= self.server.max_jobs_per_connection:
                raise _CommandError("start_rejected")
            if self._pending_start_resolution is not None:
                raise _CommandError("start_rejected")
            self._pending_start_resolution = start_resolution
        try:
            handle = await self.server.registry.start(
                client_id=self.client_id,
                prompt=prompt,
                seed=seed,
                backend=self.server.backend,
                latent_frames=FIXED_LATENT_FRAMES,
            )
        except (StreamProtocolError, ValueError):
            raise _CommandError("start_rejected")
        except Exception:
            raise _CommandError("start_rejected")
        finally:
            if handle is None:
                async with self._state:
                    if self._pending_start_resolution is start_resolution:
                        self._pending_start_resolution = None
                    start_resolution.set()
                    self._state.notify_all()

        assert handle is not None
        try:
            job_id = _validate_id(handle.job_id, "start_rejected")
            async with self._state:
                reused_job_id = job_id in self._used_job_ids
        except _CommandError:
            await self.server.registry.cancel(self.client_id, job_id=handle.job_id)
            async with self._state:
                if self._pending_start_resolution is start_resolution:
                    self._pending_start_resolution = None
                start_resolution.set()
                self._state.notify_all()
            raise _CommandError("start_rejected")
        if reused_job_id:
            await self.server.registry.cancel(self.client_id, job_id=job_id)
            async with self._state:
                if self._pending_start_resolution is start_resolution:
                    self._pending_start_resolution = None
                start_resolution.set()
                self._state.notify_all()
            raise _CommandError("start_rejected")

        gate = asyncio.Event()
        async with self._state:
            self._retire_current_deliveries_locked()
            self._used_job_ids.add(job_id)
            self._current_job_id = job_id
            self._announcement_gate = gate
            self._last_client_presented_ns = None
            self._last_server_received_ns = None
            for cancel_gate in self._cancel_gates.values():
                cancel_gate.set()
            self._cancel_gates.clear()
            if self._pending_start_resolution is start_resolution:
                self._pending_start_resolution = None
            start_resolution.set()
            self._state.notify_all()
        self._tasks[job_id] = handle.task

        def retire(completed: asyncio.Task[Any]) -> None:
            if self._tasks.get(job_id) is handle.task:
                self._tasks.pop(job_id, None)
            self._spawn_maintenance(self._retire_failed_job(job_id, completed))

        handle.task.add_done_callback(retire)
        try:
            acceptance = {
                "type": "start_accepted",
                "job_id": job_id,
                "latent_frames": FIXED_LATENT_FRAMES,
                "expected_rgb_frames": EXPECTED_RGB_FRAMES,
            }
            if prompt_resolution_id is not None and prompt_provenance is not None:
                acceptance.update(
                    {
                        "prompt_resolution_id": prompt_resolution_id,
                        "raw_prompt_sha256": prompt_provenance["raw_prompt_sha256"],
                        "effective_prompt_sha256": prompt_provenance[
                            "effective_prompt_sha256"
                        ],
                        "prompt_transform_id": prompt_provenance[
                            "prompt_transform_id"
                        ],
                    }
                )
            sent = await self.send_json(
                acceptance,
                expected_job_id=job_id,
            )
            if not sent:
                raise ConnectionError("start acknowledgement became stale")
        finally:
            gate.set()

    async def _handle_cancel(self, message: dict[str, Any]) -> None:
        _require_fields(message, required={"type", "job_id"})
        job_id = _validate_id(message["job_id"], "invalid_job_id")
        cancel_gate = asyncio.Event()
        async with self._state:
            self._cancel_gates[job_id] = cancel_gate
        try:
            cancelled = await self.server.registry.cancel(
                self.client_id,
                job_id=job_id,
            )
            if cancelled:
                await self.send_json(
                    {"type": "cancel_accepted", "job_id": job_id},
                    expected_job_id=job_id,
                )
            await self.send_json(
                {"type": "cancel_result", "job_id": job_id, "cancelled": cancelled}
            )
        finally:
            cancel_gate.set()
            async with self._state:
                if self._cancel_gates.get(job_id) is cancel_gate:
                    self._cancel_gates.pop(job_id, None)

    async def _handle_presented(self, message: dict[str, Any]) -> None:
        _require_fields(
            message,
            required={
                "type",
                "job_id",
                "chunk_index",
                "delivery_id",
                "client_presented_ns",
            },
        )
        job_id = _validate_id(message["job_id"], "invalid_job_id")
        delivery_id = _validate_id(message["delivery_id"], "invalid_delivery_id")
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
            delivery = self._deliveries.get(delivery_id)
            if delivery is None:
                if delivery_id in self._retired_delivery_set:
                    raise _CommandError("stale_delivery", job_id=job_id)
                raise _CommandError("delivery_not_sent", job_id=job_id)
            if delivery.job_id != job_id or delivery.chunk_index != chunk_index:
                raise _CommandError("delivery_mismatch", job_id=job_id)
            if delivery.acknowledged:
                raise _CommandError("duplicate_presentation", job_id=job_id)
            if (
                self._last_client_presented_ns is not None
                and client_presented_ns <= self._last_client_presented_ns
            ):
                raise _CommandError("non_monotonic_presentation", job_id=job_id)
            if server_received_ns < delivery.ready_ns or (
                self._last_server_received_ns is not None
                and server_received_ns <= self._last_server_received_ns
            ):
                raise _CommandError("server_clock_failed", job_id=job_id)
            delivery.acknowledged = True
            self._outstanding_delivery_ids.discard(delivery_id)
            self._last_client_presented_ns = client_presented_ns
            self._last_server_received_ns = server_received_ns
            self._state.notify_all()
            ready_ns = delivery.ready_ns

        await self.send_json(
            {
                "type": "presentation_recorded",
                "job_id": job_id,
                "chunk_index": chunk_index,
                "delivery_id": delivery_id,
                "server_chunk_ready_ns": ready_ns,
                "client_presented_ns": client_presented_ns,
                "client_clock_domain": "client_monotonic",
                "server_received_ns": server_received_ns,
                "server_clock_domain": "server_monotonic",
                "timing_semantics": "client_reported_presentation",
                "not_server_emit_completion": True,
            },
            expected_job_id=job_id,
        )

    async def disconnect(self) -> None:
        async with self._disconnect_lock:
            if self._closed:
                return
            async with self._state:
                self._closed = True
                current_job_id = self._current_job_id
                announcement = self._announcement_gate
                pending = self._pending_start_resolution
                for cancel_gate in self._cancel_gates.values():
                    cancel_gate.set()
                self._cancel_gates.clear()
                self._state.notify_all()
            if announcement is not None:
                announcement.set()
            if pending is not None:
                pending.set()
            with contextlib.suppress(Exception):
                await self.server.registry.cancel(
                    self.client_id,
                    job_id=current_job_id,
                )
            tasks = [task for task in self._tasks.values() if not task.done()]
            tasks.extend(
                task
                for task in self._prompt_resolution_tasks
                if not task.done() and task not in tasks
            )
            for task in tasks:
                task.cancel()
            if tasks:
                done, pending_tasks = await asyncio.wait(
                    tasks,
                    timeout=self.server.disconnect_timeout_s,
                )
                for task in done:
                    _consume_task(task)
                for task in pending_tasks:
                    task.add_done_callback(_consume_task)
            maintenance = [
                task
                for task in self._maintenance_tasks
                if task is not asyncio.current_task() and not task.done()
            ]
            for task in maintenance:
                task.cancel()
            if maintenance:
                done, pending_tasks = await asyncio.wait(
                    maintenance,
                    timeout=self.server.disconnect_timeout_s,
                )
                for task in done:
                    _consume_task(task)
                for task in pending_tasks:
                    task.add_done_callback(_consume_task)
            with contextlib.suppress(Exception):
                await self.ws.close()


class BrowserStreamingServer:
    """A loopback-only, same-origin aiohttp server for the v1 browser protocol."""

    def __init__(
        self,
        *,
        backend: StreamingBackend,
        host: str = "127.0.0.1",
        port: int = 0,
        max_control_bytes: int = DEFAULT_MAX_CONTROL_BYTES,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        presentation_window_chunks: int = DEFAULT_PRESENTATION_WINDOW_CHUNKS,
        disconnect_timeout_s: float = DEFAULT_DISCONNECT_TIMEOUT_S,
        emit_timeout_s: float = DEFAULT_EMIT_TIMEOUT_S,
        control_send_timeout_s: float = DEFAULT_CONTROL_SEND_TIMEOUT_S,
        prompt_resolver: PromptResolver | None = None,
        prompt_resolution_timeout_s: float = DEFAULT_PROMPT_RESOLUTION_TIMEOUT_S,
        max_jobs_per_connection: int = DEFAULT_MAX_JOBS_PER_CONNECTION,
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
        max_issued_sessions: int = DEFAULT_MAX_ISSUED_SESSIONS,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        backend_chunk_timeout_s: float = DEFAULT_BACKEND_CHUNK_TIMEOUT_S,
        backend_close_timeout_s: float = DEFAULT_BACKEND_CLOSE_TIMEOUT_S,
        demo_backend_kind: str = "fake",
        clock_ns: Callable[[], int] = time.monotonic_ns,
        job_id_factory: Callable[[], str] | None = None,
        delivery_id_factory: Callable[[], str] | None = None,
    ) -> None:
        try:
            host_address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("host must be a literal loopback IP address") from exc
        if not host_address.is_loopback:
            raise ValueError("browser streaming host must be loopback")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError("port must be an integer from 0 through 65535")
        if not callable(getattr(backend, "stream", None)):
            raise ValueError("backend must provide stream(request)")
        if demo_backend_kind not in _DEMO_BACKEND_DESCRIPTIONS:
            raise ValueError("demo_backend_kind must be 'fake' or 'cf1'")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        max_control_bytes = _require_positive_int(
            max_control_bytes, "max_control_bytes"
        )
        max_prompt_bytes = _require_positive_int(max_prompt_bytes, "max_prompt_bytes")
        presentation_window_chunks = _require_positive_int(
            presentation_window_chunks, "presentation_window_chunks"
        )
        max_issued_sessions = _require_positive_int(
            max_issued_sessions, "max_issued_sessions"
        )
        max_jobs_per_connection = _require_positive_int(
            max_jobs_per_connection, "max_jobs_per_connection"
        )
        max_chunk_bytes = _require_positive_int(max_chunk_bytes, "max_chunk_bytes")
        disconnect_timeout_s = _require_positive_number(
            disconnect_timeout_s, "disconnect_timeout_s"
        )
        emit_timeout_s = _require_positive_number(emit_timeout_s, "emit_timeout_s")
        control_send_timeout_s = _require_positive_number(
            control_send_timeout_s, "control_send_timeout_s"
        )
        prompt_resolution_timeout_s = _require_positive_number(
            prompt_resolution_timeout_s,
            "prompt_resolution_timeout_s",
        )
        session_ttl_s = _require_positive_number(session_ttl_s, "session_ttl_s")
        backend_chunk_timeout_s = _require_positive_number(
            backend_chunk_timeout_s, "backend_chunk_timeout_s"
        )
        backend_close_timeout_s = _require_positive_number(
            backend_close_timeout_s, "backend_close_timeout_s"
        )
        if max_prompt_bytes > max_control_bytes:
            raise ValueError("max_prompt_bytes cannot exceed max_control_bytes")
        if prompt_resolver is None:
            prompt_resolver = IdentityPromptResolver()
        if not callable(getattr(prompt_resolver, "resolve", None)):
            raise ValueError("prompt_resolver must provide resolve(raw_prompt)")

        self.backend = backend
        self.host = str(host_address)
        self.port = port
        self.max_control_bytes = max_control_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.presentation_window_chunks = presentation_window_chunks
        self.disconnect_timeout_s = disconnect_timeout_s
        self.emit_timeout_s = emit_timeout_s
        self.control_send_timeout_s = control_send_timeout_s
        self.prompt_resolver = prompt_resolver
        self.prompt_resolution_timeout_s = prompt_resolution_timeout_s
        self.max_jobs_per_connection = max_jobs_per_connection
        self.session_ttl_s = session_ttl_s
        self.max_issued_sessions = max_issued_sessions
        self.max_chunk_bytes = max_chunk_bytes
        self.backend_chunk_timeout_s = backend_chunk_timeout_s
        self.backend_close_timeout_s = backend_close_timeout_s
        self.demo_backend_kind = demo_backend_kind
        self.clock_ns = clock_ns
        self.delivery_id_factory = delivery_id_factory or (
            lambda: secrets.token_urlsafe(18)
        )
        self._address: tuple[str, int] | None = None
        self._expected_host: str | None = None
        self._origin: str | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.SockSite | None = None
        self._clients: dict[str, _WebSocketSession] = {}
        self._issued_nonces: OrderedDict[str, float] = OrderedDict()

        index_html = (_STATIC_DIR / "streaming_demo.html").read_text("utf-8")
        if index_html.count(_FAKE_BACKEND_DESCRIPTION) != 1:
            raise RuntimeError("streaming demo backend description template changed")
        self._index_html = index_html.replace(
            _FAKE_BACKEND_DESCRIPTION,
            _DEMO_BACKEND_DESCRIPTIONS[demo_backend_kind],
        )
        self._demo_js = (_STATIC_DIR / "streaming_demo.js").read_text("utf-8")
        self._demo_css = (_STATIC_DIR / "streaming_demo.css").read_text("utf-8")
        self._app = web.Application(client_max_size=max_control_bytes)
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/demo.js", self._handle_demo_js)
        self._app.router.add_get("/demo.css", self._handle_demo_css)
        self._app.router.add_get("/ws", self._handle_websocket)
        self.registry = StreamingJobRegistry(
            emit=self._emit_to_client,
            clock_ns=clock_ns,
            job_id_factory=job_id_factory,
            emit_timeout_s=emit_timeout_s,
            backend_chunk_timeout_s=backend_chunk_timeout_s,
            backend_close_timeout_s=backend_close_timeout_s,
            max_latent_frames=FIXED_LATENT_FRAMES,
            max_chunk_bytes=max_chunk_bytes,
        )

    @property
    def address(self) -> tuple[str, int]:
        if self._address is None:
            raise RuntimeError("server is not running")
        return self._address

    @property
    def origin(self) -> str:
        if self._origin is None:
            raise RuntimeError("server is not running")
        return self._origin

    @property
    def websocket_url(self) -> str:
        return f"ws{self.origin[4:]}/ws"

    def _host_header_is_valid(self, request: web.Request) -> bool:
        expected = self._expected_host
        actual = request.headers.get("Host")
        return (
            expected is not None
            and actual is not None
            and secrets.compare_digest(actual, expected)
        )

    def _security_headers(self) -> dict[str, str]:
        websocket_origin = None if self._origin is None else f"ws{self._origin[4:]}"
        connect_source = (
            "'self'" if websocket_origin is None else f"'self' {websocket_origin}"
        )
        return {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                f"default-src 'self'; connect-src {connect_source}; "
                "img-src 'self' blob:; "
                "script-src 'self'; style-src 'self'; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        }

    def _purge_expired_nonces(self) -> None:
        now = time.monotonic()
        expired = [
            nonce for nonce, deadline in self._issued_nonces.items() if deadline <= now
        ]
        for nonce in expired:
            self._issued_nonces.pop(nonce, None)

    def _issue_nonce(self) -> str:
        self._purge_expired_nonces()
        while len(self._issued_nonces) >= self.max_issued_sessions:
            self._issued_nonces.popitem(last=False)
        nonce = secrets.token_urlsafe(32)
        self._issued_nonces[nonce] = time.monotonic() + self.session_ttl_s
        return nonce

    async def _handle_index(self, request: web.Request) -> web.StreamResponse:
        if not self._host_header_is_valid(request):
            raise web.HTTPForbidden(text="forbidden")
        response = web.Response(
            text=self._index_html,
            content_type="text/html",
            headers=self._security_headers(),
        )
        response.set_cookie(
            _SESSION_COOKIE,
            self._issue_nonce(),
            httponly=True,
            samesite="Strict",
            path="/",
            max_age=max(1, int(self.session_ttl_s)),
        )
        return response

    async def _handle_demo_js(self, request: web.Request) -> web.StreamResponse:
        if not self._host_header_is_valid(request):
            raise web.HTTPForbidden(text="forbidden")
        return web.Response(
            text=self._demo_js,
            content_type="application/javascript",
            headers=self._security_headers(),
        )

    async def _handle_demo_css(self, request: web.Request) -> web.StreamResponse:
        if not self._host_header_is_valid(request):
            raise web.HTTPForbidden(text="forbidden")
        return web.Response(
            text=self._demo_css,
            content_type="text/css",
            headers=self._security_headers(),
        )

    async def _handle_websocket(self, request: web.Request) -> web.StreamResponse:
        if not self._host_header_is_valid(request):
            raise web.HTTPForbidden(text="forbidden")
        origin = request.headers.get("Origin")
        if (
            self._origin is None
            or origin is None
            or not secrets.compare_digest(origin, self._origin)
        ):
            raise web.HTTPForbidden(text="forbidden")
        requested_protocols = [
            item.strip()
            for item in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if item.strip()
        ]
        if requested_protocols != [PROTOCOL_VERSION]:
            raise web.HTTPBadRequest(text="protocol required")
        self._purge_expired_nonces()
        nonce = request.cookies.get(_SESSION_COOKIE)
        if nonce is None or self._issued_nonces.pop(nonce, None) is None:
            raise web.HTTPForbidden(text="forbidden")

        ws = web.WebSocketResponse(
            protocols=(PROTOCOL_VERSION,),
            max_msg_size=self.max_control_bytes,
            compress=False,
            autoping=True,
        )
        await ws.prepare(request)
        client_id = uuid.uuid4().hex
        session = _WebSocketSession(server=self, client_id=client_id, ws=ws)
        self._clients[client_id] = session
        try:
            await session.send_json(
                {
                    "type": "connected",
                    "protocol": PROTOCOL_VERSION,
                    "client_id": client_id,
                    "latent_frames": FIXED_LATENT_FRAMES,
                    "expected_rgb_frames": EXPECTED_RGB_FRAMES,
                    "presentation_window_chunks": self.presentation_window_chunks,
                    "payload_encoding": "websocket-binary",
                    "media_type_semantics": "backend_declared_per_chunk",
                }
            )
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        await session.handle_text(message.data)
                    except _CommandError as exc:
                        await self._send_command_error(session, exc)
                elif message.type == WSMsgType.BINARY:
                    await self._send_command_error(
                        session,
                        _CommandError("text_controls_required"),
                    )
                elif message.type == WSMsgType.ERROR:
                    break
        except (ConnectionError, RuntimeError):
            pass
        finally:
            self._clients.pop(client_id, None)
            await session.disconnect()
        return ws

    async def _emit_to_client(self, client_id: str, event: StreamEvent) -> None:
        session = self._clients.get(client_id)
        if session is None or session.closed:
            raise ConnectionError("client disconnected")
        await session.emit_event(event)

    async def _send_command_error(
        self,
        session: _WebSocketSession,
        error: _CommandError,
    ) -> None:
        message: dict[str, Any] = {"type": "command_error", "code": error.code}
        if error.job_id is not None:
            message["job_id"] = error.job_id
        await session.send_json(message)

    async def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("server is already running")
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        runner: web.AppRunner | None = None
        try:
            sock.bind((self.host, self.port))
            sock.listen(socket.SOMAXCONN)
            sock.setblocking(False)
            bound = sock.getsockname()
            actual_port = int(bound[1])
            url_host = f"[{self.host}]" if family == socket.AF_INET6 else self.host
            self._address = (self.host, actual_port)
            self._expected_host = f"{url_host}:{actual_port}"
            self._origin = f"http://{self._expected_host}"
            runner = web.AppRunner(self._app, access_log=None)
            await runner.setup()
            site = web.SockSite(runner, sock)
            await site.start()
        except BaseException:
            sock.close()
            self._address = None
            self._expected_host = None
            self._origin = None
            if runner is not None:
                with contextlib.suppress(Exception):
                    await runner.cleanup()
            raise
        assert runner is not None
        self._runner = runner
        self._site = site

    async def close(self) -> None:
        sessions = list(self._clients.values())
        if sessions:
            await asyncio.gather(
                *(session.disconnect() for session in sessions),
                return_exceptions=True,
            )
        runner = self._runner
        self._runner = None
        self._site = None
        if runner is not None:
            await runner.cleanup()
        self._address = None
        self._expected_host = None
        self._origin = None
        self._issued_nonces.clear()


__all__ = [
    "BrowserStreamingServer",
    "DEFAULT_CONTROL_SEND_TIMEOUT_S",
    "DEFAULT_EMIT_TIMEOUT_S",
    "DEFAULT_MAX_JOBS_PER_CONNECTION",
    "DEFAULT_MAX_CONTROL_BYTES",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_PRESENTATION_WINDOW_CHUNKS",
    "EXPECTED_RGB_FRAMES",
    "FIXED_LATENT_FRAMES",
    "PROTOCOL_VERSION",
    "RENDERABLE_FRAME_MEDIA_TYPES",
]
