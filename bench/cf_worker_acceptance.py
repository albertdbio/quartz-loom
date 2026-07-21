"""Non-authorizing lifecycle acceptance for the persistent CF++1 H100 worker.

The runner intentionally exercises the production ``StreamingJobRegistry``
boundary.  It retains hashes and timing metadata, never frame payloads, and
publishes a single success-only manifest with atomic cooperative no-clobber
semantics. A post-link durability or identity failure is explicitly
indeterminate and never triggers a destructive rollback of the published name.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.cf_streaming_worker import (
    CF1_LATENT_FRAMES,
    CF1_MAX_SEED,
    REAL_WORKER_BUNDLE_PATHS,
    REAL_WORKER_SCRIPT,
    AcceptanceWorkerTerminationEvidence,
    build_cf1_process_streaming_backend,
)
from bench.streaming_process import WorkerCompletionEvidence, WorkerProtocolError
from bench.streaming_process_protocol import (
    WORKER_PROTOCOL_VERSION,
    worker_bundle_sha256,
)
from bench.png_validation import is_valid_png
from bench.streaming_service import (
    StreamEvent,
    StreamProtocolError,
    StreamSummary,
    StreamingJobRegistry,
)


_EXPECTED_CHUNK_FRAME_COUNTS = (1,) + (4,) * (CF1_LATENT_FRAMES - 1)
_EXPECTED_FRAME_COUNT = sum(_EXPECTED_CHUNK_FRAME_COUNTS)
_EXPECTED_FRAME_WIDTH = 832
_EXPECTED_FRAME_HEIGHT = 480
_CLIENT_ID = "cf1-persistent-worker-acceptance"
_JOB_IDS = (
    "cf1-acceptance-job-1",
    "cf1-acceptance-job-2",
    "cf1-acceptance-death-probe",
    "cf1-acceptance-post-poison-probe",
)
_MAX_PROMPT_BYTES = 4096
_WORKER_FATAL_PREFIX = b"real worker fatal: "
_MAX_FAILURE_TYPE_BYTES = 128
_PARENT_WARM_FAILURE_CODES = {
    "timed out reading packet header prefix": "hello_timeout",
    "failed reading packet header prefix": "hello_read_failed",
    "truncated packet header prefix": "hello_truncated",
    "packet header length exceeds configured maximum": "hello_invalid_packet",
    "packet header is not valid JSON": "hello_invalid_packet",
    "packet header must be a JSON object": "hello_invalid_packet",
    "packet header is not canonical JSON": "hello_invalid_packet",
    "packet payload framing is missing": "hello_invalid_packet",
    "packet payload framing counts disagree": "hello_invalid_packet",
    "payload SHA-256 metadata is invalid": "hello_invalid_packet",
    "worker handshake fields do not match the protocol": "hello_fields",
    "worker handshake may not contain payloads": "hello_payloads",
    "worker handshake type does not match the active worker job": "hello_type",
    "worker protocol version does not match the active worker job": "protocol_version",
    "stack digest does not match the active worker job": "stack_identity",
    "worker code digest does not match the active worker job": "worker_code_identity",
    "worker instance id is invalid": "worker_instance_identity",
    "worker latent frame limit does not match protocol": "latent_frame_limit",
    "worker interpreter is not isolated": "interpreter_isolation",
    "worker environment report is invalid": "environment_report",
    "sensitive environment names reached worker": "sensitive_environment_names",
    "worker did not become ready": "worker_not_ready",
    "persistent worker exited unexpectedly": "worker_exited",
    "persistent worker socket closed unexpectedly": "worker_socket_closed",
    "persistent worker left unsolicited bytes": "worker_unsolicited_bytes",
}


class AcceptanceError(ValueError):
    """The real-worker lifecycle did not satisfy the acceptance contract."""


class AcceptancePublicationIndeterminate(AcceptanceError):
    """Publication may have become visible without a provable durable outcome."""


class _WorkerWarmRefusal(AcceptanceError):
    """A sanitized warm failure whose arbitrary diagnostic text stays private."""

    def __init__(
        self,
        *,
        parent_failure_type: str,
        parent_failure_code: str | None,
        worker_fatal_type: str | None,
    ) -> None:
        super().__init__("persistent worker warm-up refused")
        self.parent_failure_type = parent_failure_type
        self.parent_failure_code = parent_failure_code
        self.worker_fatal_type = worker_fatal_type


def _safe_failure_type(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("ascii", errors="ignore")) > _MAX_FAILURE_TYPE_BYTES
        or not ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z" or value[0] == "_")
        or not all(
            "A" <= character <= "Z"
            or "a" <= character <= "z"
            or "0" <= character <= "9"
            or character == "_"
            for character in value
        )
    ):
        return None
    return value


def _worker_fatal_type(stderr_tail: object) -> str | None:
    if not isinstance(stderr_tail, bytes):
        return None
    for line in reversed(stderr_tail.splitlines()):
        if not line.startswith(_WORKER_FATAL_PREFIX):
            continue
        try:
            candidate = line[len(_WORKER_FATAL_PREFIX) :].decode("ascii")
        except UnicodeError:
            return None
        return _safe_failure_type(candidate)
    return None


def _parent_warm_failure_code(error: BaseException) -> str | None:
    if not isinstance(error, WorkerProtocolError):
        return None
    return _PARENT_WARM_FAILURE_CODES.get(str(error))


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_oci_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


def _require_uint32(value: object, label: str) -> int:
    if not _is_plain_int(value) or value < 0 or value > CF1_MAX_SEED:
        raise AcceptanceError(f"{label} must be an unsigned 32-bit integer")
    return value


def _absolute_output_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise AcceptanceError("output_manifest must be a Path")
    if value.is_symlink():
        raise AcceptanceError("output manifest must not be a symlink")
    output = Path(os.path.abspath(value))
    if output.exists() or output.is_symlink():
        raise AcceptanceError("output manifest already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AcceptanceError("output manifest parent must be an existing directory")
    return output


def _validate_request(
    *,
    prompt: object,
    first_seed: object,
    second_seed: object,
    expected_stack_sha256: object,
    expected_worker_code_sha256: object,
    runtime_image_index_digest: object,
    runtime_image_digest: object,
    runtime_image_config_digest: object,
    runtime_environment_root: object,
    runtime_distribution_path: object,
    runtime_wheelhouse: object,
    output_manifest: object,
) -> tuple[str, int, int, str, str, str, str, str, str, str, str, Path]:
    output = _absolute_output_path(output_manifest)
    if not isinstance(prompt, str) or not prompt.strip():
        raise AcceptanceError("prompt must be a non-empty string")
    try:
        prompt.encode("utf-8")
    except UnicodeError as error:
        raise AcceptanceError("prompt must be valid UTF-8") from error
    normalized_first_seed = _require_uint32(first_seed, "first_seed")
    normalized_second_seed = _require_uint32(second_seed, "second_seed")
    if normalized_first_seed == normalized_second_seed:
        raise AcceptanceError("acceptance seeds must be distinct")
    if not _is_sha256(expected_stack_sha256):
        raise AcceptanceError("expected_stack_sha256 must be a lowercase SHA-256")
    if not _is_sha256(expected_worker_code_sha256):
        raise AcceptanceError("expected_worker_code_sha256 must be a lowercase SHA-256")
    if (
        not _is_oci_sha256(runtime_image_index_digest)
        or not _is_oci_sha256(runtime_image_digest)
        or not _is_oci_sha256(runtime_image_config_digest)
        or runtime_image_index_digest == runtime_image_digest
    ):
        raise AcceptanceError("runtime OCI assertions are invalid")
    runtime_paths = (
        runtime_environment_root,
        runtime_distribution_path,
        runtime_wheelhouse,
    )
    if any(
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
        for value in runtime_paths
    ):
        raise AcceptanceError("runtime evidence paths must be absolute")
    return (
        prompt,
        normalized_first_seed,
        normalized_second_seed,
        expected_stack_sha256,
        expected_worker_code_sha256,
        runtime_image_index_digest,
        runtime_image_digest,
        runtime_image_config_digest,
        runtime_environment_root,
        runtime_distribution_path,
        runtime_wheelhouse,
        output,
    )


def _canonical_manifest(report: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise AcceptanceError("acceptance report is not canonical JSON") from error


def _fsync_directory(directory: Path | int) -> None:
    if isinstance(directory, int) and not isinstance(directory, bool):
        os.fsync(directory)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("manifest write made no progress")
        offset += written


def _read_bounded(descriptor: int, limit: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _same_regular_inode(observed: os.stat_result, expected: os.stat_result) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_dev == expected.st_dev
        and observed.st_ino == expected.st_ino
    )


def _verify_published_inode(
    parent_descriptor: int,
    output_name: str,
    staged: os.stat_result,
    encoded: bytes,
    *,
    expected_link_count: int,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AcceptanceError("no-follow manifest verification is unavailable")
    descriptor = os.open(
        output_name,
        os.O_RDONLY | nofollow,
        dir_fd=parent_descriptor,
    )
    try:
        observed_stat = os.fstat(descriptor)
        if (
            not _same_regular_inode(observed_stat, staged)
            or observed_stat.st_nlink != expected_link_count
        ):
            raise AcceptanceError("published acceptance manifest identity changed")
        observed = _read_bounded(descriptor, len(encoded))
    finally:
        os.close(descriptor)
    if observed != encoded:
        raise AcceptanceError("published acceptance manifest bytes changed")
    return observed


def _published_name_matches(
    parent_descriptor: int,
    output_name: str,
    staged: os.stat_result,
    *,
    expected_link_count: int,
) -> bool:
    try:
        observed = os.stat(
            output_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        _same_regular_inode(observed, staged)
        and observed.st_nlink == expected_link_count
    )


def _cleanup_staged_temporary(
    parent_descriptor: int,
    temporary_name: str,
    staged: os.stat_result,
) -> None:
    try:
        observed = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return
    if not _same_regular_inode(observed, staged):
        return
    with contextlib.suppress(OSError):
        os.unlink(temporary_name, dir_fd=parent_descriptor)


def _publish_manifest_no_replace(output: Path, report: dict[str, Any]) -> str:
    """Durably publish one regular file without replacing a cooperative writer.

    The final no-follow name stat is the success linearization point. Portable
    POSIX APIs cannot prevent an uncooperative same-user process from replacing
    a directory entry afterward, so every failure after the hard link is
    indeterminate and this function never rolls that name back.
    """

    if output.exists() or output.is_symlink():
        raise AcceptanceError("output manifest already exists")
    encoded = _canonical_manifest(report)
    temporary_name = f".{output.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AcceptanceError("no-follow manifest publication is unavailable")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    descriptor: int | None = None
    parent_descriptor: int | None = None
    staged: os.stat_result | None = None
    linked = False
    try:
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow,
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        if not stat.S_ISREG(staged.st_mode) or staged.st_nlink != 1:
            raise AcceptanceError("staged acceptance manifest is not a private file")
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        linked = True
        _verify_published_inode(
            parent_descriptor,
            output.name,
            staged,
            encoded,
            expected_link_count=2,
        )
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        _fsync_directory(parent_descriptor)
        observed = _verify_published_inode(
            parent_descriptor,
            output.name,
            staged,
            encoded,
            expected_link_count=1,
        )
        if not _published_name_matches(
            parent_descriptor,
            output.name,
            staged,
            expected_link_count=1,
        ):
            raise AcceptanceError("published acceptance manifest identity changed")
    except FileExistsError as error:
        raise AcceptanceError("output manifest already exists") from error
    except BaseException as error:
        if linked:
            raise AcceptancePublicationIndeterminate(
                "acceptance manifest publication outcome is indeterminate"
            ) from error
        if isinstance(error, AcceptanceError):
            raise
        if isinstance(error, Exception):
            raise AcceptanceError(
                "acceptance manifest could not be published"
            ) from error
        raise
    finally:
        if parent_descriptor is not None and staged is not None:
            _cleanup_staged_temporary(
                parent_descriptor,
                temporary_name,
                staged,
            )
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if parent_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(parent_descriptor)
    return hashlib.sha256(observed).hexdigest()


@dataclass
class _CapturedJob:
    job_id: str
    started_ns: int | None = None
    chunk_frame_counts: list[int] = field(default_factory=list)
    chunk_ready_ns: list[int] = field(default_factory=list)
    chunk_validated_ns: list[int] = field(default_factory=list)
    frame_payload_sha256: list[str] = field(default_factory=list)
    terminal_kind: str | None = None
    failure_code: str | None = None
    terminal_summary: StreamSummary | None = None

    def accept(self, event: StreamEvent) -> None:
        if event.job_id != self.job_id:
            raise AcceptanceError("emitted job identity changed")
        if self.terminal_kind is not None:
            raise AcceptanceError("job emitted an event after its terminal event")
        if event.kind == "job_started":
            if (
                self.started_ns is not None
                or self.chunk_frame_counts
                or not _is_plain_int(event.started_ns)
            ):
                raise AcceptanceError("job start event is invalid or duplicated")
            if event.started_ns < 0:
                raise AcceptanceError("job start time is negative")
            self.started_ns = event.started_ns
            return
        if event.kind == "chunk_ready":
            expected_chunk_index = len(self.chunk_frame_counts)
            expected_frame_count = (
                _EXPECTED_CHUNK_FRAME_COUNTS[expected_chunk_index]
                if expected_chunk_index < len(_EXPECTED_CHUNK_FRAME_COUNTS)
                else None
            )
            payloads = event.frame_payloads
            if (
                self.started_ns is None
                or self.terminal_kind is not None
                or event.chunk_index != expected_chunk_index
                or event.first_frame_index != len(self.frame_payload_sha256)
                or expected_frame_count is None
                or not isinstance(payloads, tuple)
                or len(payloads) != expected_frame_count
                or event.frame_media_type != "image/png"
                or not _is_plain_int(event.ready_ns)
                or event.ready_ns <= self.started_ns
            ):
                raise AcceptanceError("emitted chunk violates the acceptance topology")
            if self.chunk_ready_ns and event.ready_ns <= self.chunk_ready_ns[-1]:
                raise AcceptanceError("emitted chunk readiness is not monotonic")
            for payload in payloads:
                if not is_valid_png(
                    payload,
                    expected_width=_EXPECTED_FRAME_WIDTH,
                    expected_height=_EXPECTED_FRAME_HEIGHT,
                    require_rgb8=True,
                ):
                    raise AcceptanceError("emitted frame is not a renderable PNG")
                self.frame_payload_sha256.append(hashlib.sha256(payload).hexdigest())
            validated_ns = time.perf_counter_ns()
            if validated_ns < event.ready_ns:
                raise AcceptanceError(
                    "acceptance validation time precedes service readiness"
                )
            self.chunk_frame_counts.append(len(payloads))
            self.chunk_ready_ns.append(event.ready_ns)
            self.chunk_validated_ns.append(validated_ns)
            return
        if event.kind in {"job_completed", "job_failed", "job_cancelled"}:
            if self.started_ns is None:
                raise AcceptanceError("job terminal event preceded its start")
            if event.kind == "job_completed":
                if (
                    tuple(self.chunk_frame_counts) != _EXPECTED_CHUNK_FRAME_COUNTS
                    or len(self.frame_payload_sha256) != _EXPECTED_FRAME_COUNT
                    or not isinstance(event.summary, StreamSummary)
                    or event.summary.job_id != self.job_id
                    or event.summary.release_count != CF1_LATENT_FRAMES
                    or event.summary.frame_count != _EXPECTED_FRAME_COUNT
                    or event.error_code is not None
                ):
                    raise AcceptanceError("job completion event is inconsistent")
            elif event.summary is not None:
                raise AcceptanceError("failed job carried a completion summary")
            elif event.kind == "job_failed" and (
                not isinstance(event.error_code, str) or not event.error_code
            ):
                raise AcceptanceError("failed job omitted its failure code")
            elif event.kind == "job_cancelled" and event.error_code is not None:
                raise AcceptanceError("cancelled job carried a failure code")
            self.terminal_kind = event.kind
            self.failure_code = event.error_code
            self.terminal_summary = event.summary
            return
        raise AcceptanceError("registry emitted an unknown acceptance event")


class _AcceptanceEmitter:
    def __init__(
        self,
        client_id: str,
        allowed_job_ids: tuple[str, ...] = _JOB_IDS,
    ) -> None:
        self.client_id = client_id
        self.allowed_job_ids = frozenset(allowed_job_ids)
        self.captures: dict[str, _CapturedJob] = {}

    async def __call__(self, client_id: str, event: StreamEvent) -> None:
        if client_id != self.client_id:
            raise AcceptanceError("registry emitted to the wrong client")
        if event.job_id not in self.allowed_job_ids:
            raise AcceptanceError("registry emitted an unsolicited job identity")
        capture = self.captures.setdefault(event.job_id, _CapturedJob(event.job_id))
        capture.accept(event)


def _require_warm_identity(backend: Any) -> tuple[int, str]:
    pid = backend.worker_pid
    instance_id = backend.worker_instance_id
    if (
        not _is_plain_int(pid)
        or pid <= 0
        or pid == os.getpid()
        or not _is_sha256(instance_id)
        or backend.ready is not True
        or backend.poisoned is not False
        or backend.child_isolated is not True
        or backend.child_sensitive_environment_names != ()
    ):
        raise AcceptanceError("warm worker identity or isolation evidence is invalid")
    return pid, instance_id


def _validate_completion(
    *,
    evidence: object,
    summary: StreamSummary,
    capture: _CapturedJob,
    job_id: str,
    prompt_sha256: str,
    seed: int,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    expected_worker_instance_id: str,
) -> WorkerCompletionEvidence:
    if not isinstance(evidence, WorkerCompletionEvidence):
        raise AcceptanceError("worker completion evidence is missing")
    if (
        evidence.worker_instance_id != expected_worker_instance_id
        or evidence.job_id != job_id
        or evidence.stack_sha256 != expected_stack_sha256
        or evidence.worker_code_sha256 != expected_worker_code_sha256
        or evidence.prompt_sha256 != prompt_sha256
        or evidence.seed != seed
        or evidence.chunk_count != CF1_LATENT_FRAMES
        or evidence.chunk_frame_counts != _EXPECTED_CHUNK_FRAME_COUNTS
        or evidence.frame_count != _EXPECTED_FRAME_COUNT
        or evidence.frame_payload_sha256 != tuple(capture.frame_payload_sha256)
    ):
        raise AcceptanceError("worker completion evidence does not match emitted bytes")
    if (
        capture.started_ns is None
        or capture.terminal_kind != "job_completed"
        or capture.terminal_summary != summary
        or tuple(capture.chunk_frame_counts) != _EXPECTED_CHUNK_FRAME_COUNTS
        or len(capture.chunk_ready_ns) != CF1_LATENT_FRAMES
        or len(capture.chunk_validated_ns) != CF1_LATENT_FRAMES
        or len(capture.frame_payload_sha256) != _EXPECTED_FRAME_COUNT
        or summary.job_id != job_id
        or summary.release_count != CF1_LATENT_FRAMES
        or summary.frame_count != _EXPECTED_FRAME_COUNT
    ):
        raise AcceptanceError("registry summary does not match emitted bytes")
    return evidence


async def _run_accepted_job(
    *,
    registry: StreamingJobRegistry,
    emitter: _AcceptanceEmitter,
    backend: Any,
    prompt: str,
    prompt_sha256: str,
    seed: int,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    warm_pid: int,
    warm_instance_id: str,
) -> dict[str, Any]:
    wall_start_ns = time.perf_counter_ns()
    handle = await registry.start(
        client_id=_CLIENT_ID,
        prompt=prompt,
        seed=seed,
        backend=backend,
        latent_frames=CF1_LATENT_FRAMES,
    )
    try:
        summary = await handle.task
    except Exception as error:
        raise AcceptanceError("acceptance job did not complete") from error
    wall_terminal_ns = time.perf_counter_ns()
    capture = emitter.captures.get(handle.job_id)
    if capture is None:
        raise AcceptanceError("registry emitted no job evidence")
    evidence = _validate_completion(
        evidence=backend.drain_completion_evidence(handle.job_id),
        summary=summary,
        capture=capture,
        job_id=handle.job_id,
        prompt_sha256=prompt_sha256,
        seed=seed,
        expected_stack_sha256=expected_stack_sha256,
        expected_worker_code_sha256=expected_worker_code_sha256,
        expected_worker_instance_id=warm_instance_id,
    )
    observed_pid, observed_instance_id = _require_warm_identity(backend)
    if observed_pid != warm_pid or observed_instance_id != warm_instance_id:
        raise AcceptanceError("persistent worker identity changed between jobs")
    if capture.started_ns is None:
        raise AcceptanceError("registry start timing is missing")
    service_offsets = [
        ready_ns - capture.started_ns for ready_ns in capture.chunk_ready_ns
    ]
    validation_offsets = [
        validated_ns - capture.started_ns for validated_ns in capture.chunk_validated_ns
    ]
    duration_ns = wall_terminal_ns - wall_start_ns
    if (
        duration_ns <= 0
        or any(offset <= 0 for offset in service_offsets)
        or any(offset <= 0 for offset in validation_offsets)
    ):
        raise AcceptanceError("acceptance timing is invalid")
    return {
        "job_id": handle.job_id,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "worker_pid": observed_pid,
        "worker_instance_id": observed_instance_id,
        "stack_sha256": evidence.stack_sha256,
        "worker_code_sha256": evidence.worker_code_sha256,
        "chunk_count": evidence.chunk_count,
        "chunk_frame_counts": list(evidence.chunk_frame_counts),
        "frame_count": evidence.frame_count,
        "frame_dimensions": {
            "width": _EXPECTED_FRAME_WIDTH,
            "height": _EXPECTED_FRAME_HEIGHT,
        },
        "frame_media_type": "image/png",
        "frame_payload_sha256": list(evidence.frame_payload_sha256),
        "completion_evidence_reconciled": True,
        "parent_received_chunks": [
            {
                "chunk_index": chunk_index,
                "first_frame_index": sum(capture.chunk_frame_counts[:chunk_index]),
                "frame_count": capture.chunk_frame_counts[chunk_index],
                "service_validated_ready_ns": capture.chunk_ready_ns[chunk_index],
                "acceptance_validated_ns": capture.chunk_validated_ns[chunk_index],
                "frame_payload_sha256": list(
                    evidence.frame_payload_sha256[
                        sum(capture.chunk_frame_counts[:chunk_index]) : sum(
                            capture.chunk_frame_counts[: chunk_index + 1]
                        )
                    ]
                ),
            }
            for chunk_index in range(CF1_LATENT_FRAMES)
        ],
        "timing": {
            "clock": "monotonic nanoseconds",
            "registry_start_to_terminal_ns": duration_ns,
            "service_validated_chunk_ready_offsets_ns": service_offsets,
            "acceptance_validated_chunk_offsets_ns": validation_offsets,
            "first_parent_validated_frame_ready_ns": validation_offsets[0],
            "all_parent_validated_frames_ready_ns": validation_offsets[-1],
        },
    }


async def run_cf1_worker_acceptance(
    *,
    prompt: str,
    first_seed: int,
    second_seed: int,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    runtime_image_index_digest: str,
    runtime_image_digest: str,
    runtime_image_config_digest: str,
    runtime_environment_root: str,
    runtime_distribution_path: str,
    runtime_wheelhouse: str,
    output_manifest: Path,
) -> dict[str, Any]:
    """Run two reused-worker jobs plus one unexpected-death poison proof."""

    (
        prompt,
        first_seed,
        second_seed,
        expected_stack_sha256,
        expected_worker_code_sha256,
        runtime_image_index_digest,
        runtime_image_digest,
        runtime_image_config_digest,
        runtime_environment_root,
        runtime_distribution_path,
        runtime_wheelhouse,
        output,
    ) = _validate_request(
        prompt=prompt,
        first_seed=first_seed,
        second_seed=second_seed,
        expected_stack_sha256=expected_stack_sha256,
        expected_worker_code_sha256=expected_worker_code_sha256,
        runtime_image_index_digest=runtime_image_index_digest,
        runtime_image_digest=runtime_image_digest,
        runtime_image_config_digest=runtime_image_config_digest,
        runtime_environment_root=runtime_environment_root,
        runtime_distribution_path=runtime_distribution_path,
        runtime_wheelhouse=runtime_wheelhouse,
        output_manifest=output_manifest,
    )
    observed_worker_code_sha256 = worker_bundle_sha256(
        REAL_WORKER_SCRIPT,
        REAL_WORKER_BUNDLE_PATHS,
    )
    if observed_worker_code_sha256 != expected_worker_code_sha256:
        raise AcceptanceError("current worker bundle does not match the frozen digest")

    backend = build_cf1_process_streaming_backend(
        expected_stack_sha256=expected_stack_sha256,
        expected_worker_code_sha256=expected_worker_code_sha256,
        runtime_image_index_digest=runtime_image_index_digest,
        runtime_image_digest=runtime_image_digest,
        runtime_image_config_digest=runtime_image_config_digest,
        runtime_environment_root=runtime_environment_root,
        runtime_distribution_path=runtime_distribution_path,
        runtime_wheelhouse=runtime_wheelhouse,
    )
    report: dict[str, Any] | None = None
    try:
        if (
            backend.stack_sha256 != expected_stack_sha256
            or backend.expected_worker_code_sha256 != expected_worker_code_sha256
        ):
            raise AcceptanceError("backend launch configuration changed")
        try:
            prompt_bytes = prompt.encode("utf-8")
        except UnicodeError as error:  # already checked, retained at the boundary
            raise AcceptanceError("prompt must be valid UTF-8") from error
        if len(prompt_bytes) > _MAX_PROMPT_BYTES:
            raise AcceptanceError("prompt exceeds the frozen worker byte limit")

        warm_start_ns = time.perf_counter_ns()
        try:
            await backend.warm()
        except Exception as error:
            raise _WorkerWarmRefusal(
                parent_failure_type=(
                    _safe_failure_type(type(error).__name__) or "Exception"
                ),
                parent_failure_code=_parent_warm_failure_code(error),
                worker_fatal_type=_worker_fatal_type(
                    getattr(backend, "stderr_tail", b"")
                ),
            ) from error
        warm_verified_ns = time.perf_counter_ns()
        warm_pid, warm_instance_id = _require_warm_identity(backend)
        warm_duration_ns = warm_verified_ns - warm_start_ns
        if warm_duration_ns <= 0:
            raise AcceptanceError("warm timing is invalid")

        emitter = _AcceptanceEmitter(_CLIENT_ID)
        job_ids = iter(_JOB_IDS)
        registry = StreamingJobRegistry(
            emit=emitter,
            job_id_factory=lambda: next(job_ids),
            queue_capacity=2,
            backend_chunk_timeout_s=backend.registry_chunk_timeout_s,
            backend_close_timeout_s=backend.registry_close_timeout_s,
            max_latent_frames=CF1_LATENT_FRAMES,
            max_chunk_bytes=backend.max_chunk_bytes,
        )
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        jobs = []
        for seed in (first_seed, second_seed):
            jobs.append(
                await _run_accepted_job(
                    registry=registry,
                    emitter=emitter,
                    backend=backend,
                    prompt=prompt,
                    prompt_sha256=prompt_sha256,
                    seed=seed,
                    expected_stack_sha256=expected_stack_sha256,
                    expected_worker_code_sha256=expected_worker_code_sha256,
                    warm_pid=warm_pid,
                    warm_instance_id=warm_instance_id,
                )
            )
        if jobs[0]["frame_payload_sha256"] == jobs[1]["frame_payload_sha256"]:
            raise AcceptanceError("distinct seeds produced identical frame payloads")

        termination = await backend.terminate_idle_worker_for_acceptance(
            expected_pid=warm_pid,
            expected_worker_instance_id=warm_instance_id,
        )
        if (
            not isinstance(termination, AcceptanceWorkerTerminationEvidence)
            or termination.pid != warm_pid
            or termination.worker_instance_id != warm_instance_id
            or termination.process_group_id != warm_pid
            or termination.session_id != warm_pid
            or termination.signal_name != "SIGKILL"
        ):
            raise AcceptanceError("worker termination evidence is invalid")

        death_probe_start_ns = time.perf_counter_ns()
        death_handle = await registry.start(
            client_id=_CLIENT_ID,
            prompt=prompt,
            seed=first_seed,
            backend=backend,
            latent_frames=CF1_LATENT_FRAMES,
        )
        try:
            await death_handle.task
        except StreamProtocolError as death_error:
            death_failure_type = type(death_error).__name__
        else:
            raise AcceptanceError("job unexpectedly succeeded after worker death")
        death_probe_terminal_ns = time.perf_counter_ns()
        death_capture = emitter.captures.get(death_handle.job_id)
        if (
            death_capture is None
            or death_capture.terminal_kind != "job_failed"
            or death_capture.failure_code != "backend_fatal"
            or death_capture.frame_payload_sha256
            or backend.poisoned is not True
            or backend.ready is not False
            or backend.worker_pid is not None
            or backend.worker_instance_id is not None
        ):
            raise AcceptanceError("dead worker was not poisoned and reaped")

        try:
            unexpected_handle = await registry.start(
                client_id=_CLIENT_ID,
                prompt=prompt,
                seed=second_seed,
                backend=backend,
                latent_frames=CF1_LATENT_FRAMES,
            )
        except StreamProtocolError:
            registry_rejected_after_poison = True
        else:
            await registry.cancel(_CLIENT_ID, job_id=unexpected_handle.job_id)
            with contextlib.suppress(BaseException):
                await unexpected_handle.task
            raise AcceptanceError("registry accepted a job after backend poison")

        if set(emitter.captures) != set(_JOB_IDS[:3]):
            raise AcceptanceError(
                "registry event identities do not match acceptance jobs"
            )

        report = {
            "schema_version": 1,
            "kind": "cf1-persistent-worker-acceptance",
            "purpose": "development-lifecycle-acceptance",
            "ready": True,
            "status": "accepted",
            "lifecycle_acceptance_passed": True,
            "authorizes_quality_claim": False,
            "authorizes_performance_claim": False,
            "authorizes_browser_visibility_claim": False,
            "authorizes_provider_upload": False,
            "authorizes_additional_gpu_execution": False,
            "performance_gate_evaluated": False,
            "gpu_execution_performed": True,
            "provider_api_calls_performed": False,
            "inputs": {
                "prompt_sha256": prompt_sha256,
                "seeds": [first_seed, second_seed],
                "expected_stack_sha256": expected_stack_sha256,
                "expected_worker_code_sha256": expected_worker_code_sha256,
                "configured_runtime_image_index_digest": (runtime_image_index_digest),
                "configured_runtime_image_digest": runtime_image_digest,
                "configured_runtime_image_config_digest": (runtime_image_config_digest),
                "configured_runtime_environment_root": (runtime_environment_root),
                "configured_runtime_distribution_path": (runtime_distribution_path),
                "configured_runtime_wheelhouse": runtime_wheelhouse,
                "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            },
            "service_boundary": {
                "kind": "StreamingJobRegistry",
                "queue_capacity": 2,
                "backend_chunk_timeout_s": backend.registry_chunk_timeout_s,
                "backend_close_timeout_s": backend.registry_close_timeout_s,
                "max_chunk_bytes": backend.max_chunk_bytes,
            },
            "warm": {
                "duration_ns": warm_duration_ns,
                "worker_pid": warm_pid,
                "worker_instance_id": warm_instance_id,
                "child_isolated": True,
                "sensitive_environment_names": [],
            },
            "jobs": jobs,
            "reuse": {
                "same_worker_pid": True,
                "same_worker_instance": True,
                "distinct_seed_outputs": True,
            },
            "forced_death": {
                "fault_injector": "acceptance harness",
                "worker_pid": termination.pid,
                "worker_instance_id": termination.worker_instance_id,
                "process_group_id": termination.process_group_id,
                "session_id": termination.session_id,
                "signal": termination.signal_name,
                "death_probe_job_id": death_handle.job_id,
                "death_probe_failure_type": death_failure_type,
                "death_probe_duration_ns": (
                    death_probe_terminal_ns - death_probe_start_ns
                ),
                "death_probe_awaited": True,
                "backend_poisoned": True,
                "backend_ready": False,
                "worker_reaped": True,
                "registry_poisoned": registry_rejected_after_poison,
                "post_poison_start_rejected": registry_rejected_after_poison,
            },
            "timing_scope": {
                "warm_is_separate": True,
                "parent_chunk_ready_includes": [
                    "job START and STARTED",
                    "generator and decoder initialization",
                    "CUDA generation and decode",
                    "device-to-host transfer",
                    "PNG encoding",
                    "worker IPC",
                    "supervisor and service validation",
                ],
                "excludes": [
                    "MP4 encoding",
                    "network transport",
                    "provider upload",
                    "browser presentation and paint",
                ],
                "interpretation": (
                    "bounded serial parent-validated renderable PNG readiness only"
                ),
            },
        }
    finally:
        await backend.close()

    if report is None:
        raise AcceptanceError("acceptance report was not produced")
    _publish_manifest_no_replace(output, report)
    return report


async def cf1_worker_acceptance_report(**kwargs: Any) -> dict[str, Any]:
    """Return a sanitized refusal record while publishing only successful runs."""

    try:
        return await run_cf1_worker_acceptance(**kwargs)
    except AcceptancePublicationIndeterminate as error:
        return {
            "schema_version": 1,
            "kind": "cf1-persistent-worker-acceptance",
            "ready": False,
            "status": "publication_indeterminate",
            "failure_type": type(error).__name__,
        }
    except _WorkerWarmRefusal as error:
        report = {
            "schema_version": 1,
            "kind": "cf1-persistent-worker-acceptance",
            "ready": False,
            "status": "refused",
            "failure_type": error.parent_failure_type,
            "failure_stage": "warm",
        }
        if error.parent_failure_code is not None:
            report["parent_failure_code"] = error.parent_failure_code
        if error.worker_fatal_type is not None:
            report["worker_fatal_type"] = error.worker_fatal_type
        return report
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "cf1-persistent-worker-acceptance",
            "ready": False,
            "status": "refused",
            "failure_type": type(error).__name__,
        }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-seed", required=True, type=int)
    parser.add_argument("--second-seed", required=True, type=int)
    parser.add_argument("--expected-stack-sha256", required=True)
    parser.add_argument("--expected-worker-code-sha256", required=True)
    parser.add_argument("--runtime-image-index-digest", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--runtime-image-config-digest", required=True)
    parser.add_argument("--runtime-environment-root", required=True)
    parser.add_argument("--runtime-distribution-path", required=True)
    parser.add_argument("--runtime-wheelhouse", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(
        cf1_worker_acceptance_report(
            prompt=args.prompt,
            first_seed=args.first_seed,
            second_seed=args.second_seed,
            expected_stack_sha256=args.expected_stack_sha256,
            expected_worker_code_sha256=args.expected_worker_code_sha256,
            runtime_image_index_digest=args.runtime_image_index_digest,
            runtime_image_digest=args.runtime_image_digest,
            runtime_image_config_digest=args.runtime_image_config_digest,
            runtime_environment_root=args.runtime_environment_root,
            runtime_distribution_path=args.runtime_distribution_path,
            runtime_wheelhouse=args.runtime_wheelhouse,
            output_manifest=args.output,
        )
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if report.get("ready") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
