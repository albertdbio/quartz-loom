"""Persistent, process-isolated streaming backend with a strict pull protocol.

The child used by this module is deliberately a dependency-free fake.  The
supervisor is the production-shaped part: one exec-spawned process is reused
only after a fully validated COMPLETE record, while cancellation tears it down
before any cleanup await.  A later accelerator worker can implement the same
wire contract without weakening process ownership or provenance checks here.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import secrets
import signal
import socket
import struct
import subprocess
import sys
from types import MappingProxyType
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.streaming_process_protocol import (
    WORKER_PROTOCOL_MAX_LATENT_FRAMES,
    WORKER_PROTOCOL_VERSION,
    worker_bundle_sha256,
)
from bench.streaming_service import (
    BackendFatalError,
    DecodedChunk,
    StreamProtocolError,
    StreamRequest,
)


DEFAULT_WORKER_SCRIPT = Path(__file__).with_name("streaming_process_worker.py")
_PACKET_PREFIX_BYTES = 4
_EMPTY_PAYLOAD_KEYS = frozenset({"payload_lengths", "payload_sha256"})
_SHA256_HEX_LENGTH = 64
_MAX_RETAINED_COMPLETIONS = 64
_DEFAULT_MAX_LATENT_FRAMES = WORKER_PROTOCOL_MAX_LATENT_FRAMES
_DEFAULT_MAX_JOB_IDS = 1024
_STDERR_READ_BYTES = 64 * 1024
_MAX_STDERR_READS_PER_CALLBACK = 4
_SENSITIVE_ENVIRONMENT_MARKERS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


class WorkerProtocolError(BackendFatalError):
    """A worker/process ownership violation that permanently poisons reuse."""


@dataclass(frozen=True)
class WorkerCompletionEvidence:
    worker_instance_id: str
    job_id: str
    stack_sha256: str
    worker_code_sha256: str
    prompt_sha256: str
    seed: int
    chunk_count: int
    chunk_frame_counts: tuple[int, ...]
    frame_count: int
    frame_payload_sha256: tuple[str, ...]


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise WorkerProtocolError("packet header is not canonical JSON") from error


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_worker_environment(
    environment: Mapping[str, str],
    *,
    failure_type: type[Exception],
) -> None:
    if environment.get("PYTHONUNBUFFERED") != "1":
        raise failure_type("worker environment must set PYTHONUNBUFFERED=1")
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise failure_type("worker environment contains an invalid entry")
        if any(marker in name.upper() for marker in _SENSITIVE_ENVIRONMENT_MARKERS):
            raise failure_type("worker environment contains a sensitive name")


async def _read_exact(
    sock: socket.socket,
    byte_count: int,
    *,
    deadline: float,
    label: str,
) -> bytes:
    loop = asyncio.get_running_loop()
    parts: list[bytes] = []
    remaining = byte_count
    while remaining:
        timeout_s = deadline - loop.time()
        if timeout_s <= 0:
            raise WorkerProtocolError(f"timed out reading {label}")
        try:
            part = await asyncio.wait_for(
                loop.sock_recv(sock, min(remaining, 64 * 1024)),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as error:
            raise WorkerProtocolError(f"timed out reading {label}") from error
        except OSError as error:
            raise WorkerProtocolError(f"failed reading {label}") from error
        if not part:
            raise WorkerProtocolError(f"truncated {label}")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


async def _send_packet(
    sock: socket.socket,
    header: Mapping[str, Any],
    payloads: Sequence[bytes] = (),
    *,
    timeout_s: float,
    max_header_bytes: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
) -> None:
    """Send one canonical JSON header followed by its raw payload segments."""

    if _EMPTY_PAYLOAD_KEYS.intersection(header):
        raise WorkerProtocolError("packet metadata may not supply payload framing")
    normalized_payloads: list[bytes] = []
    total_bytes = 0
    for payload in payloads:
        if not isinstance(payload, bytes):
            raise WorkerProtocolError("packet payload must be immutable bytes")
        if len(payload) > max_frame_bytes:
            raise WorkerProtocolError("frame length exceeds configured maximum")
        total_bytes += len(payload)
        if total_bytes > max_chunk_bytes:
            raise WorkerProtocolError("chunk length exceeds configured maximum")
        normalized_payloads.append(payload)

    envelope = dict(header)
    envelope["payload_lengths"] = [len(payload) for payload in normalized_payloads]
    envelope["payload_sha256"] = [
        hashlib.sha256(payload).hexdigest() for payload in normalized_payloads
    ]
    encoded_header = _canonical_json(envelope)
    if not encoded_header or len(encoded_header) > max_header_bytes:
        raise WorkerProtocolError("packet header length exceeds configured maximum")
    packet = struct.pack(">I", len(encoded_header)) + encoded_header

    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(loop.sock_sendall(sock, packet), timeout=timeout_s)
        for payload in normalized_payloads:
            await asyncio.wait_for(loop.sock_sendall(sock, payload), timeout=timeout_s)
    except asyncio.TimeoutError as error:
        raise WorkerProtocolError("timed out writing worker packet") from error
    except OSError as error:
        raise WorkerProtocolError("failed writing worker packet") from error


async def _receive_packet(
    sock: socket.socket,
    *,
    timeout_s: float,
    max_header_bytes: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """Receive and authenticate one packet, bounding sizes before body reads."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    prefix = await _read_exact(
        sock,
        _PACKET_PREFIX_BYTES,
        deadline=deadline,
        label="packet header prefix",
    )
    header_length = struct.unpack(">I", prefix)[0]
    if header_length <= 0 or header_length > max_header_bytes:
        raise WorkerProtocolError("packet header length exceeds configured maximum")
    encoded_header = await _read_exact(
        sock,
        header_length,
        deadline=deadline,
        label="packet header",
    )
    try:
        header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WorkerProtocolError("packet header is not valid JSON") from error
    if not isinstance(header, dict):
        raise WorkerProtocolError("packet header must be a JSON object")
    if _canonical_json(header) != encoded_header:
        raise WorkerProtocolError("packet header is not canonical JSON")

    lengths = header.get("payload_lengths")
    digests = header.get("payload_sha256")
    if not isinstance(lengths, list) or not isinstance(digests, list):
        raise WorkerProtocolError("packet payload framing is missing")
    if len(lengths) != len(digests):
        raise WorkerProtocolError("packet payload framing counts disagree")

    total_bytes = 0
    for length in lengths:
        if not _is_plain_int(length) or length < 0:
            raise WorkerProtocolError("frame length must be a non-negative integer")
        if length > max_frame_bytes:
            raise WorkerProtocolError("frame length exceeds configured maximum")
        total_bytes += length
        if total_bytes > max_chunk_bytes:
            raise WorkerProtocolError("chunk length exceeds configured maximum")
    if not all(_is_sha256(digest) for digest in digests):
        raise WorkerProtocolError("payload SHA-256 metadata is invalid")

    payloads: list[bytes] = []
    for frame_index, (length, expected_digest) in enumerate(zip(lengths, digests)):
        payload = await _read_exact(
            sock,
            length,
            deadline=deadline,
            label=f"packet payload {frame_index}",
        )
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise WorkerProtocolError("packet payload SHA-256 mismatch")
        payloads.append(payload)
    return header, tuple(payloads)


def _expect_header_keys(header: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(header)
    expected_with_framing = expected | set(_EMPTY_PAYLOAD_KEYS)
    if actual != expected_with_framing:
        raise WorkerProtocolError(f"{label} fields do not match the protocol")


def _expect_string(value: object, expected: str, label: str) -> None:
    if value != expected or not isinstance(value, str):
        raise WorkerProtocolError(f"{label} does not match the active worker job")


class ProcessStreamingBackend:
    """Own one persistent exec-spawned worker with fail-closed reuse rules.

    The lifecycle owner must eventually await :meth:`close`.  A failed reap
    detaches every I/O handle but deliberately retains the ``Popen`` handle so
    a later idempotent ``close`` can retry the mandatory kill/reap boundary.
    """

    def __init__(
        self,
        *,
        stack_sha256: str,
        worker_script: Path | None = None,
        worker_args: Sequence[str] = (),
        worker_bundle_paths: Sequence[Path] = (),
        worker_environment: Mapping[str, str] | None = None,
        require_warm_start: bool = False,
        startup_timeout_s: float = 1.0,
        io_timeout_s: float = 1.0,
        reap_timeout_s: float = 0.5,
        registry_chunk_timeout_s: float = 30.0,
        registry_close_timeout_s: float = 2.0,
        max_header_bytes: int = 64 * 1024,
        max_frame_bytes: int = 8 * 1024 * 1024,
        max_chunk_bytes: int = 16 * 1024 * 1024,
        max_prompt_bytes: int = 4096,
        max_latent_frames: int = _DEFAULT_MAX_LATENT_FRAMES,
        max_job_ids: int = _DEFAULT_MAX_JOB_IDS,
        stderr_tail_bytes: int = 16 * 1024,
    ) -> None:
        if not _is_sha256(stack_sha256):
            raise ValueError("stack_sha256 must be a lowercase SHA-256 digest")
        self.startup_timeout_s = _positive_finite(
            startup_timeout_s, "startup timeout"
        )
        self.io_timeout_s = _positive_finite(io_timeout_s, "I/O timeout")
        self.reap_timeout_s = _positive_finite(reap_timeout_s, "reap timeout")
        self.registry_chunk_timeout_s = _positive_finite(
            registry_chunk_timeout_s, "registry chunk timeout"
        )
        self.registry_close_timeout_s = _positive_finite(
            registry_close_timeout_s, "registry close timeout"
        )
        if self.io_timeout_s >= self.registry_chunk_timeout_s:
            raise ValueError("I/O timeout must be less than registry chunk timeout")
        if (
            self.startup_timeout_s + self.reap_timeout_s
            >= self.registry_chunk_timeout_s
            or self.io_timeout_s + self.reap_timeout_s
            >= self.registry_chunk_timeout_s
        ):
            raise ValueError(
                "startup/I/O plus reap must fit inside registry chunk timeout"
            )
        if self.reap_timeout_s >= self.registry_close_timeout_s:
            raise ValueError("reap timeout must be less than registry close timeout")

        self.max_header_bytes = _positive_int(max_header_bytes, "max_header_bytes")
        self.max_frame_bytes = _positive_int(max_frame_bytes, "max_frame_bytes")
        self.max_chunk_bytes = _positive_int(max_chunk_bytes, "max_chunk_bytes")
        self.max_prompt_bytes = _positive_int(max_prompt_bytes, "max_prompt_bytes")
        self.max_latent_frames = _positive_int(
            max_latent_frames, "max_latent_frames"
        )
        if self.max_latent_frames > WORKER_PROTOCOL_MAX_LATENT_FRAMES:
            raise ValueError(
                "max_latent_frames exceeds the worker protocol limit "
                f"{WORKER_PROTOCOL_MAX_LATENT_FRAMES}"
            )
        self.max_job_ids = _positive_int(max_job_ids, "max_job_ids")
        self.stderr_tail_bytes = _positive_int(stderr_tail_bytes, "stderr_tail_bytes")
        if self.max_frame_bytes > self.max_chunk_bytes:
            raise ValueError("max_frame_bytes must not exceed max_chunk_bytes")

        self.stack_sha256 = stack_sha256
        self.worker_script = (worker_script or DEFAULT_WORKER_SCRIPT).resolve()
        self.worker_args = tuple(str(argument) for argument in worker_args)
        resolved_bundle_paths = tuple(
            Path(path).resolve() for path in worker_bundle_paths
        )
        if len(set(resolved_bundle_paths)) != len(resolved_bundle_paths):
            raise ValueError("worker bundle paths must be unique")
        self.worker_bundle_paths = resolved_bundle_paths
        environment = (
            {"PYTHONUNBUFFERED": "1"}
            if worker_environment is None
            else dict(worker_environment)
        )
        _validate_worker_environment(environment, failure_type=ValueError)
        self.worker_environment = MappingProxyType(environment)
        if not isinstance(require_warm_start, bool):
            raise ValueError("require_warm_start must be a boolean")
        self.require_warm_start = require_warm_start

        self._claim_lock = asyncio.Lock()
        self._active_job_id: str | None = None
        self._used_job_ids: set[str] = set()
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._worker_instance_id: str | None = None
        self._worker_code_sha256: str | None = None
        self._child_isolated: bool | None = None
        self._child_sensitive_environment_names: tuple[str, ...] | None = None
        self._stderr_tail = b""
        self._stderr_fd: int | None = None
        self._stderr_loop: asyncio.AbstractEventLoop | None = None
        self._completions: OrderedDict[str, WorkerCompletionEvidence] = OrderedDict()
        self._poisoned = False
        self._closed = False
        self._state = "stopped"

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    @property
    def worker_instance_id(self) -> str | None:
        return self._worker_instance_id

    @property
    def child_isolated(self) -> bool | None:
        return self._child_isolated

    @property
    def child_sensitive_environment_names(self) -> tuple[str, ...] | None:
        return self._child_sensitive_environment_names

    @property
    def stderr_tail(self) -> bytes:
        return self._stderr_tail

    @property
    def ready(self) -> bool:
        process = self._process
        return (
            self._state == "ready"
            and process is not None
            and process.poll() is None
            and self._socket is not None
        )

    async def warm(self) -> None:
        """Start and verify the worker before accepting latency-bounded jobs."""

        async with self._claim_lock:
            if self._closed:
                raise BackendFatalError("streaming worker is closed")
            if self._poisoned:
                raise BackendFatalError("streaming worker is poisoned")
            if self._active_job_id is not None:
                raise StreamProtocolError("streaming worker already active")
            try:
                await self._ensure_worker()
            except asyncio.CancelledError:
                if not self._closed:
                    self._state = "stopping"
                process = self._kill_process_sync()
                reaped = await self._reap_resisting_cancellation(process)
                if not reaped:
                    self._poisoned = True
                    self._state = "poisoned"
                elif not self._closed:
                    self._state = "stopped"
                raise
            except BaseException:
                await self._poison_after_failure()
                raise
            if not self.ready:
                await self._poison_after_failure()
                raise WorkerProtocolError("worker did not become ready")

    def stream(self, request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        if self._closed:
            raise BackendFatalError("streaming worker is closed")
        if self._poisoned:
            raise BackendFatalError("streaming worker is poisoned")
        return self._stream(request)

    async def _claim(self, job_id: str) -> None:
        async with self._claim_lock:
            if self._closed:
                raise BackendFatalError("streaming worker is closed")
            if self._poisoned:
                raise BackendFatalError("streaming worker is poisoned")
            if self._active_job_id is not None:
                raise StreamProtocolError("streaming worker already active")
            allowed_states = (
                {"ready"} if self.require_warm_start else {"stopped", "ready"}
            )
            if self._state not in allowed_states:
                if self.require_warm_start and self._state == "stopped":
                    raise StreamProtocolError(
                        "streaming worker must be explicitly prewarmed"
                    )
                raise StreamProtocolError("streaming worker is shutting down")
            if job_id in self._used_job_ids:
                raise StreamProtocolError("job_id may not be reused by this backend")
            if len(self._used_job_ids) >= self.max_job_ids:
                raise StreamProtocolError(
                    "job_id capacity reached; replace the process backend"
                )
            if len(self._completions) >= _MAX_RETAINED_COMPLETIONS:
                raise StreamProtocolError(
                    "completion evidence capacity reached; drain evidence first"
                )
            self._active_job_id = job_id
            self._used_job_ids.add(job_id)
            self._state = "busy"

    async def _release_claim(self, job_id: str) -> None:
        async with self._claim_lock:
            if self._active_job_id == job_id:
                self._active_job_id = None

    def _on_stderr_ready(self, fd: int) -> None:
        # A continuously logging child must not monopolize the event-loop thread.
        # Leave the reader installed so another bounded callback can drain more.
        for _ in range(_MAX_STDERR_READS_PER_CALLBACK):
            try:
                data = os.read(fd, _STDERR_READ_BYTES)
            except BlockingIOError:
                return
            except OSError:
                data = b""
            if not data:
                loop = self._stderr_loop
                if loop is not None:
                    with contextlib.suppress(Exception):
                        loop.remove_reader(fd)
                return
            self._stderr_tail = (self._stderr_tail + data)[-self.stderr_tail_bytes :]

    def _install_stderr_reader(
        self,
        process: subprocess.Popen[bytes],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if process.stderr is None:
            raise WorkerProtocolError("worker stderr pipe was not created")
        fd = process.stderr.fileno()
        os.set_blocking(fd, False)
        self._stderr_fd = fd
        self._stderr_loop = loop
        loop.add_reader(fd, self._on_stderr_ready, fd)

    def _remove_stderr_reader(self, process: subprocess.Popen[bytes]) -> None:
        fd = self._stderr_fd
        loop = self._stderr_loop
        if fd is not None:
            self._on_stderr_ready(fd)
            if loop is not None:
                with contextlib.suppress(Exception):
                    loop.remove_reader(fd)
        if process.stderr is not None:
            with contextlib.suppress(OSError):
                process.stderr.close()
        self._stderr_fd = None
        self._stderr_loop = None

    def _kill_process_sync(self) -> subprocess.Popen[bytes] | None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                # The worker is spawned in its own session. Kill the process
                # group so model servers/helpers cannot outlive GPU ownership.
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
        return process

    async def _reap_process(self, process: subprocess.Popen[bytes]) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.reap_timeout_s
        while process.poll() is None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self._detach_process_io(process)
                return False
            await asyncio.sleep(min(0.005, remaining))
        self._cleanup_process_handles(process)
        return True

    async def _reap_resisting_cancellation(
        self,
        process: subprocess.Popen[bytes] | None,
    ) -> bool:
        if process is None:
            return True
        reap_task = asyncio.create_task(self._reap_process(process))
        while True:
            try:
                return await asyncio.shield(reap_task)
            except asyncio.CancelledError:
                if reap_task.done():
                    return reap_task.result()
                continue

    def _detach_process_io(self, process: subprocess.Popen[bytes]) -> None:
        if self._process is not process:
            return
        self._remove_stderr_reader(process)
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
        self._socket = None
        self._worker_instance_id = None
        self._worker_code_sha256 = None
        self._child_isolated = None
        self._child_sensitive_environment_names = None

    def _cleanup_process_handles(self, process: subprocess.Popen[bytes]) -> None:
        if self._process is not process:
            return
        self._detach_process_io(process)
        self._process = None

    def _retire_active_claim_sync(self, job_id: str | None) -> None:
        if job_id is not None and self._active_job_id == job_id:
            self._active_job_id = None

    async def _abort_for_cancel(self, job_id: str) -> None:
        self._retire_active_claim_sync(job_id)
        if not self._closed:
            self._state = "stopping"
        process = self._kill_process_sync()
        reaped = await self._reap_resisting_cancellation(process)
        if not reaped:
            self._poisoned = True
            self._state = "poisoned"
        elif not self._closed:
            self._state = "stopped"

    async def _poison_after_failure(self, job_id: str | None = None) -> None:
        self._retire_active_claim_sync(job_id)
        self._poisoned = True
        self._state = "closed" if self._closed else "poisoned"
        process = self._kill_process_sync()
        await self._reap_resisting_cancellation(process)

    def _warm_worker_is_clean(self) -> None:
        process = self._process
        sock = self._socket
        if process is None or sock is None:
            raise WorkerProtocolError("worker process handles are incomplete")
        if process.poll() is not None:
            raise WorkerProtocolError("persistent worker exited unexpectedly")
        try:
            pending = sock.recv(1, socket.MSG_PEEK)
        except BlockingIOError:
            return
        except OSError as error:
            raise WorkerProtocolError("failed probing persistent worker") from error
        if pending == b"":
            raise WorkerProtocolError("persistent worker socket closed unexpectedly")
        raise WorkerProtocolError("persistent worker left unsolicited bytes")

    async def _spawn_worker(self) -> None:
        if not self.worker_script.is_file():
            raise WorkerProtocolError("worker script does not exist")
        try:
            launch_environment = dict(self.worker_environment)
        except (TypeError, ValueError) as error:
            raise WorkerProtocolError("worker environment is invalid") from error
        _validate_worker_environment(
            launch_environment,
            failure_type=WorkerProtocolError,
        )
        worker_code_sha256 = worker_bundle_sha256(
            self.worker_script,
            self.worker_bundle_paths,
        )
        parent_socket, child_socket = socket.socketpair()
        parent_socket.setblocking(False)
        process: subprocess.Popen[bytes] | None = None
        try:
            command = [
                str(Path(sys.executable).resolve()),
                "-I",
                "-S",
                str(self.worker_script),
                "--ipc-fd",
                str(child_socket.fileno()),
                "--stack-sha256",
                self.stack_sha256,
                *self.worker_args,
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(child_socket.fileno(),),
                close_fds=True,
                env=launch_environment,
                start_new_session=True,
            )
        except (OSError, ValueError) as error:
            parent_socket.close()
            raise WorkerProtocolError("failed to exec streaming worker") from error
        finally:
            child_socket.close()

        self._process = process
        self._socket = parent_socket
        self._worker_code_sha256 = worker_code_sha256
        self._state = "starting"
        self._install_stderr_reader(process, asyncio.get_running_loop())
        header, payloads = await _receive_packet(
            parent_socket,
            timeout_s=self.startup_timeout_s,
            max_header_bytes=self.max_header_bytes,
            max_frame_bytes=self.max_frame_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )
        _expect_header_keys(
            header,
            {
                "type",
                "protocol_version",
                "worker_instance_id",
                "stack_sha256",
                "worker_code_sha256",
                "max_latent_frames",
                "isolated",
                "sensitive_environment_names",
            },
            "worker handshake",
        )
        if payloads:
            raise WorkerProtocolError("worker handshake may not contain payloads")
        _expect_string(header["type"], "HELLO", "worker handshake type")
        _expect_string(
            header["protocol_version"],
            WORKER_PROTOCOL_VERSION,
            "worker protocol version",
        )
        _expect_string(header["stack_sha256"], self.stack_sha256, "stack digest")
        _expect_string(
            header["worker_code_sha256"], worker_code_sha256, "worker code digest"
        )
        instance_id = header["worker_instance_id"]
        if not _is_sha256(instance_id):
            raise WorkerProtocolError("worker instance id is invalid")
        worker_max_latent_frames = header["max_latent_frames"]
        if (
            not _is_plain_int(worker_max_latent_frames)
            or worker_max_latent_frames != WORKER_PROTOCOL_MAX_LATENT_FRAMES
        ):
            raise WorkerProtocolError(
                "worker latent frame limit does not match protocol"
            )
        if header["isolated"] is not True:
            raise WorkerProtocolError("worker interpreter is not isolated")
        sensitive_names = header["sensitive_environment_names"]
        if not isinstance(sensitive_names, list) or not all(
            isinstance(name, str) for name in sensitive_names
        ):
            raise WorkerProtocolError("worker environment report is invalid")
        if sensitive_names:
            raise WorkerProtocolError("sensitive environment names reached worker")
        self._worker_instance_id = instance_id
        self._child_isolated = True
        self._child_sensitive_environment_names = tuple(sensitive_names)
        self._state = "ready"

    async def _ensure_worker(self) -> None:
        if self._process is None:
            await self._spawn_worker()
        else:
            self._warm_worker_is_clean()

    async def _send(self, header: Mapping[str, Any]) -> None:
        sock = self._socket
        if sock is None:
            raise WorkerProtocolError("worker socket is unavailable")
        await _send_packet(
            sock,
            header,
            timeout_s=self.io_timeout_s,
            max_header_bytes=self.max_header_bytes,
            max_frame_bytes=self.max_frame_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )

    async def _receive(self) -> tuple[dict[str, Any], tuple[bytes, ...]]:
        sock = self._socket
        if sock is None:
            raise WorkerProtocolError("worker socket is unavailable")
        return await _receive_packet(
            sock,
            timeout_s=self.io_timeout_s,
            max_header_bytes=self.max_header_bytes,
            max_frame_bytes=self.max_frame_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )

    def _validate_job_envelope(
        self,
        header: Mapping[str, Any],
        *,
        request: StreamRequest,
        expected_type: str,
        expected_chunk_index: int,
        expected_credit_nonce: str | None,
        extra_keys: set[str],
    ) -> None:
        envelope_keys = {
            "type",
            "protocol_version",
            "worker_instance_id",
            "job_id",
            "chunk_index",
        }
        if expected_credit_nonce is not None:
            envelope_keys.add("credit_nonce")
        _expect_header_keys(
            header,
            envelope_keys | extra_keys,
            expected_type,
        )
        _expect_string(header["type"], expected_type, "worker response type")
        _expect_string(
            header["protocol_version"],
            WORKER_PROTOCOL_VERSION,
            "worker protocol version",
        )
        if self._worker_instance_id is None:
            raise WorkerProtocolError("worker instance is unavailable")
        _expect_string(
            header["worker_instance_id"],
            self._worker_instance_id,
            "worker instance id",
        )
        _expect_string(header["job_id"], request.job_id, "job id")
        if (
            not _is_plain_int(header["chunk_index"])
            or header["chunk_index"] != expected_chunk_index
        ):
            raise WorkerProtocolError("chunk index does not match NEXT request")
        if expected_credit_nonce is not None:
            _expect_string(
                header["credit_nonce"], expected_credit_nonce, "NEXT credit nonce"
            )

    def _validated_frame_media_type(
        self,
        value: object,
        payloads: Sequence[bytes],
    ) -> str:
        """Validate the default fake/dev worker's lossless PNG contract."""

        if value != "image/png":
            raise WorkerProtocolError("worker frame media type is not image/png")
        if any(
            not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or not payload.endswith(b"IEND\xaeB`\x82")
            for payload in payloads
        ):
            raise WorkerProtocolError("worker emitted an invalid PNG payload")
        return "image/png"

    async def _run_claimed_job(
        self,
        request: StreamRequest,
    ) -> AsyncIterator[DecodedChunk]:
        await self._ensure_worker()
        if self._worker_instance_id is None or self._worker_code_sha256 is None:
            raise WorkerProtocolError("worker handshake provenance is unavailable")
        prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        await self._send(
            {
                "type": "START",
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "worker_instance_id": self._worker_instance_id,
                "job_id": request.job_id,
                "chunk_index": -1,
                "stack_sha256": self.stack_sha256,
                "worker_code_sha256": self._worker_code_sha256,
                "prompt": request.prompt,
                "prompt_sha256": prompt_sha256,
                "seed": request.seed,
                "latent_frames": request.latent_frames,
            }
        )
        started, started_payloads = await self._receive()
        self._validate_job_envelope(
            started,
            request=request,
            expected_type="STARTED",
            expected_chunk_index=-1,
            expected_credit_nonce=None,
            extra_keys=set(),
        )
        if started_payloads:
            raise WorkerProtocolError("STARTED response may not contain payloads")

        frame_hashes: list[str] = []
        chunk_frame_counts: list[int] = []
        first_frame_index = 0
        for chunk_index in range(request.latent_frames):
            credit_nonce = secrets.token_hex(16)
            await self._send(
                {
                    "type": "NEXT",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "worker_instance_id": self._worker_instance_id,
                    "job_id": request.job_id,
                    "chunk_index": chunk_index,
                    "credit_nonce": credit_nonce,
                }
            )
            chunk_header, payloads = await self._receive()
            self._validate_job_envelope(
                chunk_header,
                request=request,
                expected_type="CHUNK",
                expected_chunk_index=chunk_index,
                expected_credit_nonce=credit_nonce,
                extra_keys={"first_frame_index", "frame_count", "frame_media_type"},
            )
            expected_frame_count = 1 if chunk_index == 0 else 4
            if (
                not _is_plain_int(chunk_header["first_frame_index"])
                or chunk_header["first_frame_index"] != first_frame_index
            ):
                raise WorkerProtocolError("first frame index is not contiguous")
            if (
                not _is_plain_int(chunk_header["frame_count"])
                or chunk_header["frame_count"] != expected_frame_count
                or len(payloads) != expected_frame_count
            ):
                raise WorkerProtocolError("chunk frame count violates release profile")
            frame_media_type = self._validated_frame_media_type(
                chunk_header["frame_media_type"],
                payloads,
            )

            chunk_frame_counts.append(len(payloads))
            frame_hashes.extend(
                hashlib.sha256(payload).hexdigest() for payload in payloads
            )
            first_frame_index += len(payloads)
            # All framing, hashes, counts, and raster signatures are checked before
            # the chunk becomes observable to the service boundary.
            yield DecodedChunk(payloads, frame_media_type=frame_media_type)

        completion_credit_nonce = secrets.token_hex(16)
        await self._send(
            {
                "type": "NEXT",
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "worker_instance_id": self._worker_instance_id,
                "job_id": request.job_id,
                "chunk_index": request.latent_frames,
                "credit_nonce": completion_credit_nonce,
            }
        )
        complete, complete_payloads = await self._receive()
        self._validate_job_envelope(
            complete,
            request=request,
            expected_type="COMPLETE",
            expected_chunk_index=request.latent_frames,
            expected_credit_nonce=completion_credit_nonce,
            extra_keys={
                "stack_sha256",
                "worker_code_sha256",
                "prompt_sha256",
                "seed",
                "chunk_count",
                "chunk_frame_counts",
                "frame_count",
                "frame_payload_sha256",
            },
        )
        if complete_payloads:
            raise WorkerProtocolError("COMPLETE response may not contain payloads")
        _expect_string(complete["stack_sha256"], self.stack_sha256, "stack digest")
        _expect_string(
            complete["worker_code_sha256"],
            self._worker_code_sha256,
            "worker code digest",
        )
        _expect_string(complete["prompt_sha256"], prompt_sha256, "prompt digest")
        if not _is_plain_int(complete["seed"]) or complete["seed"] != request.seed:
            raise WorkerProtocolError("completion seed does not match request")
        if (
            not _is_plain_int(complete["chunk_count"])
            or complete["chunk_count"] != request.latent_frames
        ):
            raise WorkerProtocolError("completion chunk count is invalid")
        if complete["chunk_frame_counts"] != chunk_frame_counts:
            raise WorkerProtocolError("completion chunk frame counts are invalid")
        if (
            not _is_plain_int(complete["frame_count"])
            or complete["frame_count"] != len(frame_hashes)
        ):
            raise WorkerProtocolError("completion frame count is invalid")
        if complete["frame_payload_sha256"] != frame_hashes:
            raise WorkerProtocolError("completion payload hashes are invalid")

        evidence = WorkerCompletionEvidence(
            worker_instance_id=self._worker_instance_id,
            job_id=request.job_id,
            stack_sha256=self.stack_sha256,
            worker_code_sha256=self._worker_code_sha256,
            prompt_sha256=prompt_sha256,
            seed=request.seed,
            chunk_count=request.latent_frames,
            chunk_frame_counts=tuple(chunk_frame_counts),
            frame_count=len(frame_hashes),
            frame_payload_sha256=tuple(frame_hashes),
        )
        self._completions[request.job_id] = evidence
        self._completions.move_to_end(request.job_id)
        self._state = "ready"

    async def _stream(self, request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        try:
            prompt_bytes = request.prompt.encode("utf-8")
        except UnicodeError as error:
            raise StreamProtocolError("prompt must be valid UTF-8") from error
        if len(prompt_bytes) > self.max_prompt_bytes:
            raise StreamProtocolError("prompt exceeds worker byte limit")
        if request.latent_frames > self.max_latent_frames:
            raise StreamProtocolError(
                f"latent_frames exceeds worker limit {self.max_latent_frames}"
            )

        claimed = False
        completed = False
        try:
            await self._claim(request.job_id)
            claimed = True
            async for chunk in self._run_claimed_job(request):
                yield chunk
            completed = True
        except (asyncio.CancelledError, GeneratorExit):
            if claimed and not completed:
                # SIGKILL is issued synchronously inside this call, before its first
                # cleanup await.  Reaping then resists repeated task cancellation.
                await self._abort_for_cancel(request.job_id)
            raise
        except StreamProtocolError:
            if claimed and not completed:
                await self._poison_after_failure(request.job_id)
            raise
        except Exception as error:
            if claimed and not completed:
                await self._poison_after_failure(request.job_id)
            raise WorkerProtocolError("streaming worker lifecycle failed") from error
        except BaseException:
            if claimed and not completed:
                await self._poison_after_failure(request.job_id)
            raise
        finally:
            if claimed:
                await self._release_claim(request.job_id)

    def drain_completion_evidence(
        self,
        job_id: str,
    ) -> WorkerCompletionEvidence | None:
        return self._completions.pop(job_id, None)

    async def close(self) -> None:
        if self._closed and self._process is None:
            return
        self._closed = True
        self._state = "closed"
        self._retire_active_claim_sync(self._active_job_id)
        process = self._kill_process_sync()
        reaped = await self._reap_resisting_cancellation(process)
        if not reaped:
            self._poisoned = True
