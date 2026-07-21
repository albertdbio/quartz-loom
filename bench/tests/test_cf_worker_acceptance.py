from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any
from unittest import mock

from bench import cf_streaming_process_worker, cf_streaming_worker
from bench.cf_streaming_worker import AcceptanceWorkerTerminationEvidence
from bench.cf_worker_acceptance import (
    AcceptanceError,
    AcceptancePublicationIndeterminate,
    _AcceptanceEmitter,
    _CapturedJob,
    _canonical_manifest,
    _publish_manifest_no_replace,
    _read_bounded,
    _validate_completion,
    cf1_worker_acceptance_report,
    run_cf1_worker_acceptance,
)
from bench.streaming_process import WorkerCompletionEvidence, WorkerProtocolError
from bench.streaming_process_protocol import worker_bundle_sha256
from bench.streaming_service import (
    DecodedChunk,
    StreamEvent,
    StreamRequest,
    StreamSummary,
    StreamingJobRegistry,
)


STACK_SHA256 = "7" * 64
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE_INDEX_DIGEST = "sha256:" + "b" * 64
IMAGE_CONFIG_DIGEST = "sha256:" + "c" * 64
RUNTIME_LAUNCH = {
    "runtime_image_index_digest": IMAGE_INDEX_DIGEST,
    "runtime_image_digest": IMAGE_DIGEST,
    "runtime_image_config_digest": IMAGE_CONFIG_DIGEST,
    "runtime_environment_root": "/runtime/venv",
    "runtime_distribution_path": "/runtime/venv/site-packages",
    "runtime_wheelhouse": "/runtime/wheelhouse",
}
WORKER_CODE_SHA256 = worker_bundle_sha256(
    Path(cf_streaming_process_worker.__file__).resolve(),
    cf_streaming_worker.REAL_WORKER_BUNDLE_PATHS,
)
PROMPT = "A red fox runs through snow."
BACKEND_CHUNK_TIMEOUT_S = 123.25
BACKEND_CLOSE_TIMEOUT_S = 9.75
BACKEND_MAX_CHUNK_BYTES = 8 * 1024 * 1024 + 17
FIRST_JOB_ID = "cf1-acceptance-job-1"
SECOND_JOB_ID = "cf1-acceptance-job-2"
DEATH_PROBE_JOB_ID = "cf1-acceptance-death-probe"
POST_POISON_PROBE_JOB_ID = "cf1-acceptance-post-poison-probe"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def make_tiny_png(
    red: int,
    green: int,
    blue: int,
    *,
    width: int = 832,
    height: int = 480,
) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = bytes((0,)) + bytes((red, green, blue)) * width
    pixels = zlib.compress(scanline * height, level=1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _summary(job_id: str, *, wall_e2e_s: float = 1.0) -> StreamSummary:
    return StreamSummary(
        job_id=job_id,
        frame_count=81,
        release_count=21,
        first_chunk_ready_s=0.1,
        wall_e2e_s=wall_e2e_s,
        e2e_fps=81.0 / wall_e2e_s,
        p95_chunk_release_gap_ms=50.0,
    )


def _complete_capture(job_id: str, terminal_summary: StreamSummary) -> _CapturedJob:
    capture = _CapturedJob(job_id)
    capture.accept(StreamEvent(kind="job_started", job_id=job_id, started_ns=100))
    payload = make_tiny_png(1, 2, 3)
    frame_index = 0
    for chunk_index in range(21):
        frame_count = 1 if chunk_index == 0 else 4
        capture.accept(
            StreamEvent(
                kind="chunk_ready",
                job_id=job_id,
                chunk_index=chunk_index,
                first_frame_index=frame_index,
                frame_payloads=(payload,) * frame_count,
                frame_media_type="image/png",
                ready_ns=101 + chunk_index,
            )
        )
        frame_index += frame_count
    capture.accept(
        StreamEvent(kind="job_completed", job_id=job_id, summary=terminal_summary)
    )
    return capture


class RegistryAudit:
    """Wrap the real registry while retaining hashes, never frame payloads."""

    def __init__(self) -> None:
        self.constructor_calls: list[dict[str, Any]] = []
        self.registries: list[StreamingJobRegistry] = []
        self.generated_job_ids: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.failure_codes: dict[str, str | None] = {}
        self.parent_received_chunks: dict[str, list[dict[str, Any]]] = {}

    def build(self, **kwargs: Any) -> StreamingJobRegistry:
        self.constructor_calls.append(dict(kwargs))
        downstream_emit = kwargs["emit"]
        downstream_job_id_factory = kwargs["job_id_factory"]

        def audited_job_id_factory() -> str:
            job_id = downstream_job_id_factory()
            self.generated_job_ids.append(job_id)
            return job_id

        async def audited_emit(client_id: str, event: StreamEvent) -> None:
            self.events.append((event.job_id, event.kind))
            if event.kind == "job_failed":
                self.failure_codes[event.job_id] = event.error_code
            if event.kind == "chunk_ready":
                if event.frame_payloads is None:
                    raise AssertionError("chunk_ready must carry transient payloads")
                self.parent_received_chunks.setdefault(event.job_id, []).append(
                    {
                        "chunk_index": event.chunk_index,
                        "first_frame_index": event.first_frame_index,
                        "frame_count": len(event.frame_payloads),
                        "ready_ns": event.ready_ns,
                        "frame_payload_sha256": [
                            hashlib.sha256(payload).hexdigest()
                            for payload in event.frame_payloads
                        ],
                    }
                )
            await downstream_emit(client_id, event)

        kwargs["emit"] = audited_emit
        kwargs["job_id_factory"] = audited_job_id_factory
        registry = StreamingJobRegistry(**kwargs)
        self.registries.append(registry)
        return registry


class FakeAcceptanceBackend:
    def __init__(
        self,
        *,
        pid_drift: bool = False,
        instance_drift: bool = False,
        evidence_mismatch: bool = False,
        poison_on_dead_probe: bool = True,
        ignore_seed: bool = False,
        invalid_png: bool = False,
        wrong_dimensions: bool = False,
        warm_error: BaseException | None = None,
        stderr_tail: bytes = b"",
    ) -> None:
        self.stack_sha256 = STACK_SHA256
        self.expected_worker_code_sha256 = WORKER_CODE_SHA256
        # Deliberately differ from StreamingJobRegistry defaults. The acceptance
        # path must copy the real backend's immutable bounds into the registry.
        self.registry_chunk_timeout_s = BACKEND_CHUNK_TIMEOUT_S
        self.registry_close_timeout_s = BACKEND_CLOSE_TIMEOUT_S
        self.max_chunk_bytes = BACKEND_MAX_CHUNK_BYTES
        self._pid = 4242
        self._instance_id = "d" * 64
        self._ready = False
        self._dead = False
        self._poisoned = False
        self._closed = False
        self._active_job_id: str | None = None
        self._job_count = 0
        self._completions: dict[str, WorkerCompletionEvidence] = {}
        self.pid_drift = pid_drift
        self.instance_drift = instance_drift
        self.evidence_mismatch = evidence_mismatch
        self.poison_on_dead_probe = poison_on_dead_probe
        self.ignore_seed = ignore_seed
        self.invalid_png = invalid_png
        self.wrong_dimensions = wrong_dimensions
        self.warm_error = warm_error
        self._stderr_tail = stderr_tail
        self.warm_calls = 0
        self.close_calls = 0
        self.stream_requests: list[StreamRequest] = []
        self.lifecycle: list[tuple[str, str]] = []
        self.drain_calls: list[str] = []
        self.termination_calls: list[tuple[int, str]] = []

    @property
    def ready(self) -> bool:
        return self._ready and not self._dead and not self._poisoned

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def worker_pid(self) -> int | None:
        return None if self._dead or self._closed else self._pid

    @property
    def worker_instance_id(self) -> str | None:
        return self._instance_id

    @property
    def child_isolated(self) -> bool | None:
        return True

    @property
    def child_sensitive_environment_names(self) -> tuple[str, ...] | None:
        return ()

    @property
    def stderr_tail(self) -> bytes:
        return self._stderr_tail

    async def warm(self) -> None:
        self.warm_calls += 1
        if self.warm_error is not None:
            raise self.warm_error
        self._ready = True

    def stream(self, request: StreamRequest):
        self.stream_requests.append(request)

        async def generate():
            if self._active_job_id is not None:
                raise AssertionError("acceptance jobs overlapped")
            self._active_job_id = request.job_id
            self.lifecycle.append((request.job_id, "started"))
            try:
                if self._dead:
                    if self.poison_on_dead_probe:
                        self._poisoned = True
                        self._ready = False
                        self._instance_id = None
                    self.lifecycle.append((request.job_id, "failed"))
                    raise WorkerProtocolError("persistent worker exited")

                hashes: list[str] = []
                counts: list[int] = []
                frame_index = 0
                effective_seed = 0 if self.ignore_seed else request.seed
                for chunk_index in range(21):
                    count = 1 if chunk_index == 0 else 4
                    if self.invalid_png:
                        payloads = tuple(
                            b"\x89PNG\r\n\x1a\n"
                            + bytes(
                                (
                                    effective_seed & 0xFF,
                                    (frame_index + offset) & 0xFF,
                                )
                            )
                            + b"garbage\x00\x00\x00\x00IEND\xaeB`\x82"
                            for offset in range(count)
                        )
                    elif self.wrong_dimensions:
                        payloads = tuple(
                            make_tiny_png(
                                (effective_seed + frame_index + offset) & 0xFF,
                                (effective_seed * 3 + frame_index + offset) & 0xFF,
                                (255 - effective_seed - frame_index - offset) & 0xFF,
                                width=1,
                                height=1,
                            )
                            for offset in range(count)
                        )
                    else:
                        payloads = tuple(
                            make_tiny_png(
                                (effective_seed + frame_index + offset) & 0xFF,
                                (effective_seed * 3 + frame_index + offset) & 0xFF,
                                (255 - effective_seed - frame_index - offset) & 0xFF,
                            )
                            for offset in range(count)
                        )
                    frame_index += count
                    counts.append(count)
                    hashes.extend(
                        hashlib.sha256(payload).hexdigest() for payload in payloads
                    )
                    yield DecodedChunk(payloads, frame_media_type="image/png")

                evidence_hashes = list(hashes)
                if self.evidence_mismatch:
                    evidence_hashes[-1] = "0" * 64
                completion_instance_id = self._instance_id
                self._completions[request.job_id] = WorkerCompletionEvidence(
                    worker_instance_id=completion_instance_id,
                    job_id=request.job_id,
                    stack_sha256=self.stack_sha256,
                    worker_code_sha256=self.expected_worker_code_sha256,
                    prompt_sha256=hashlib.sha256(
                        request.prompt.encode("utf-8")
                    ).hexdigest(),
                    seed=request.seed,
                    chunk_count=21,
                    chunk_frame_counts=tuple(counts),
                    frame_count=81,
                    frame_payload_sha256=tuple(evidence_hashes),
                )
                self._job_count += 1
                if self.pid_drift and self._job_count == 1:
                    self._pid += 1
                if self.instance_drift and self._job_count == 1:
                    self._instance_id = "e" * 64
                self.lifecycle.append((request.job_id, "completed"))
            finally:
                self._active_job_id = None

        return generate()

    def drain_completion_evidence(self, job_id: str) -> WorkerCompletionEvidence | None:
        self.drain_calls.append(job_id)
        if self.drain_calls.count(job_id) != 1:
            raise AssertionError("completion evidence was drained more than once")
        return self._completions.pop(job_id, None)

    async def terminate_idle_worker_for_acceptance(
        self,
        *,
        expected_pid: int,
        expected_worker_instance_id: str,
    ) -> AcceptanceWorkerTerminationEvidence:
        if not self.ready or self._active_job_id is not None:
            raise AssertionError("acceptance termination requires an idle ready worker")
        if expected_pid != self._pid:
            raise AssertionError("acceptance runner targeted the wrong PID")
        if expected_worker_instance_id != self._instance_id:
            raise AssertionError("acceptance runner targeted the wrong worker instance")
        self.termination_calls.append((expected_pid, expected_worker_instance_id))
        # Deliberately do not poison here. The awaited registry death probe must
        # observe the killed worker and make both poison gates monotonic.
        self._dead = True
        return AcceptanceWorkerTerminationEvidence(
            pid=expected_pid,
            worker_instance_id=expected_worker_instance_id,
            process_group_id=expected_pid,
            session_id=expected_pid,
            signal_name="SIGKILL",
        )

    async def close(self) -> None:
        self.close_calls += 1
        self._closed = True
        self._ready = False


def _flatten_chunk_hashes(chunks: list[dict[str, Any]]) -> list[str]:
    return [
        payload_hash
        for chunk in chunks
        for payload_hash in chunk["frame_payload_sha256"]
    ]


class CapturedJobMutationTests(unittest.IsolatedAsyncioTestCase):
    def test_terminal_event_cannot_precede_start(self) -> None:
        capture = _CapturedJob(FIRST_JOB_ID)
        with self.assertRaises(AcceptanceError):
            capture.accept(
                StreamEvent(
                    kind="job_completed",
                    job_id=FIRST_JOB_ID,
                    summary=_summary(FIRST_JOB_ID),
                )
            )

    def test_completion_requires_the_exact_release_topology_and_summary(self) -> None:
        capture = _CapturedJob(FIRST_JOB_ID)
        capture.accept(
            StreamEvent(kind="job_started", job_id=FIRST_JOB_ID, started_ns=100)
        )
        with self.assertRaises(AcceptanceError):
            capture.accept(
                StreamEvent(
                    kind="job_completed",
                    job_id=FIRST_JOB_ID,
                    summary=_summary(FIRST_JOB_ID),
                )
            )

    def test_no_event_is_accepted_after_a_terminal_event(self) -> None:
        capture = _CapturedJob(DEATH_PROBE_JOB_ID)
        capture.accept(
            StreamEvent(kind="job_started", job_id=DEATH_PROBE_JOB_ID, started_ns=100)
        )
        capture.accept(
            StreamEvent(
                kind="job_failed",
                job_id=DEATH_PROBE_JOB_ID,
                error_code="backend_fatal",
            )
        )
        with self.assertRaises(AcceptanceError):
            capture.accept(
                StreamEvent(
                    kind="chunk_ready",
                    job_id=DEATH_PROBE_JOB_ID,
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(make_tiny_png(1, 2, 3),),
                    frame_media_type="image/png",
                    ready_ns=101,
                )
            )

    async def test_emitter_rejects_unsolicited_cross_job_events(self) -> None:
        emitter = _AcceptanceEmitter("expected-client")
        with self.assertRaises(AcceptanceError):
            await emitter(
                "expected-client",
                StreamEvent(
                    kind="job_started",
                    job_id="unsolicited-cross-job",
                    started_ns=100,
                ),
            )

    def test_returned_summary_must_equal_the_terminal_event_summary(self) -> None:
        emitted_summary = _summary(FIRST_JOB_ID, wall_e2e_s=1.0)
        returned_summary = _summary(FIRST_JOB_ID, wall_e2e_s=2.0)
        capture = _complete_capture(FIRST_JOB_ID, emitted_summary)
        hashes = tuple(capture.frame_payload_sha256)
        evidence = WorkerCompletionEvidence(
            worker_instance_id="d" * 64,
            job_id=FIRST_JOB_ID,
            stack_sha256=STACK_SHA256,
            worker_code_sha256=WORKER_CODE_SHA256,
            prompt_sha256=hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            seed=11,
            chunk_count=21,
            chunk_frame_counts=(1,) + (4,) * 20,
            frame_count=81,
            frame_payload_sha256=hashes,
        )
        with self.assertRaises(AcceptanceError):
            _validate_completion(
                evidence=evidence,
                summary=returned_summary,
                capture=capture,
                job_id=FIRST_JOB_ID,
                prompt_sha256=hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
                seed=11,
                expected_stack_sha256=STACK_SHA256,
                expected_worker_code_sha256=WORKER_CODE_SHA256,
                expected_worker_instance_id="d" * 64,
            )


class ManifestPublicationMutationTests(unittest.TestCase):
    def test_same_byte_symlink_swap_is_not_accepted(self) -> None:
        report = {"ready": True, "status": "accepted"}
        encoded = _canonical_manifest(report)
        real_link = os.link
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "acceptance.json"
            racer = root / "racer.json"
            racer.write_bytes(encoded)

            def swap_after_link(
                source: Any, destination: Any, *args: Any, **kwargs: Any
            ) -> None:
                real_link(source, destination, *args, **kwargs)
                destination_directory = kwargs["dst_dir_fd"]
                os.unlink(destination, dir_fd=destination_directory)
                os.symlink(racer, destination, dir_fd=destination_directory)

            with mock.patch(
                "bench.cf_worker_acceptance.os.link",
                side_effect=swap_after_link,
            ):
                with self.assertRaises(AcceptanceError):
                    _publish_manifest_no_replace(output, report)
            self.assertTrue(output.is_symlink())
            self.assertEqual(racer.read_bytes(), encoded)

    def test_swap_during_final_read_is_caught_by_name_linearization(self) -> None:
        report = {"ready": True, "status": "accepted"}
        encoded = _canonical_manifest(report)
        calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "acceptance.json"
            racer = root / "racer.json"
            racer.write_bytes(encoded)

            def swap_on_second_read(descriptor: int, limit: int) -> bytes:
                nonlocal calls
                calls += 1
                observed = _read_bounded(descriptor, limit)
                if calls == 2:
                    os.unlink(output)
                    os.symlink(racer, output)
                return observed

            with mock.patch(
                "bench.cf_worker_acceptance._read_bounded",
                side_effect=swap_on_second_read,
            ):
                with self.assertRaises(AcceptancePublicationIndeterminate):
                    _publish_manifest_no_replace(output, report)
            self.assertTrue(output.is_symlink())
            self.assertEqual(racer.read_bytes(), encoded)

    def test_post_link_fsync_failure_is_indeterminate_and_never_rolls_back(
        self,
    ) -> None:
        report = {"ready": True, "status": "accepted"}
        real_fsync_directory = __import__(
            "bench.cf_worker_acceptance", fromlist=["_fsync_directory"]
        )._fsync_directory
        calls = 0

        def fail_once(directory: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated directory fsync failure")
            real_fsync_directory(directory)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with mock.patch(
                "bench.cf_worker_acceptance._fsync_directory",
                side_effect=fail_once,
            ):
                with self.assertRaises(AcceptancePublicationIndeterminate):
                    _publish_manifest_no_replace(output, report)
            self.assertEqual(output.read_bytes(), _canonical_manifest(report))
            self.assertFalse(output.is_symlink())

    def test_rollback_never_unlinks_a_competing_inode(self) -> None:
        report = {"ready": True, "status": "accepted"}
        competitor = b"competitor-owned-bytes"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            calls = 0

            def replace_then_fail(_directory: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    os.unlink(output)
                    output.write_bytes(competitor)
                    raise OSError("simulated post-link race")

            with mock.patch(
                "bench.cf_worker_acceptance._fsync_directory",
                side_effect=replace_then_fail,
            ):
                with self.assertRaises(AcceptanceError):
                    _publish_manifest_no_replace(output, report)
            self.assertEqual(output.read_bytes(), competitor)


class CF1WorkerAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_registry_two_jobs_then_forced_death_poisons(self) -> None:
        backend = FakeAcceptanceBackend()
        registry_audit = RegistryAudit()
        fsync_targets: list[str] = []
        link_calls: list[tuple[Path, Path]] = []
        real_fsync = os.fsync
        real_link = os.link

        def audited_fsync(file_descriptor: int) -> None:
            mode = os.fstat(file_descriptor).st_mode
            if stat.S_ISREG(mode):
                fsync_targets.append("file")
            elif stat.S_ISDIR(mode):
                fsync_targets.append("directory")
            else:
                fsync_targets.append("other")
            real_fsync(file_descriptor)

        def audited_link(
            source: Any, destination: Any, *args: Any, **kwargs: Any
        ) -> None:
            link_calls.append((Path(source), Path(destination)))
            real_link(source, destination, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with (
                mock.patch(
                    "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                    return_value=backend,
                ) as build,
                mock.patch(
                    "bench.cf_worker_acceptance.StreamingJobRegistry",
                    side_effect=registry_audit.build,
                ) as registry_constructor,
                mock.patch(
                    "bench.cf_worker_acceptance.os.fsync",
                    side_effect=audited_fsync,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance.os.link",
                    side_effect=audited_link,
                ),
            ):
                report = await run_cf1_worker_acceptance(
                    prompt=PROMPT,
                    first_seed=11,
                    second_seed=12,
                    expected_stack_sha256=STACK_SHA256,
                    expected_worker_code_sha256=WORKER_CODE_SHA256,
                    **RUNTIME_LAUNCH,
                    output_manifest=output,
                )

            build.assert_called_once_with(
                expected_stack_sha256=STACK_SHA256,
                expected_worker_code_sha256=WORKER_CODE_SHA256,
                **RUNTIME_LAUNCH,
            )
            self.assertEqual(registry_constructor.call_count, 1)
            self.assertEqual(len(registry_audit.constructor_calls), 1)
            registry_options = registry_audit.constructor_calls[0]
            self.assertEqual(
                registry_options["backend_chunk_timeout_s"],
                backend.registry_chunk_timeout_s,
            )
            self.assertEqual(
                registry_options["backend_close_timeout_s"],
                backend.registry_close_timeout_s,
            )
            self.assertEqual(
                registry_options["max_chunk_bytes"], backend.max_chunk_bytes
            )
            self.assertEqual(registry_options["max_latent_frames"], 21)
            self.assertEqual(registry_options["queue_capacity"], 2)
            self.assertTrue(callable(registry_options["emit"]))
            self.assertTrue(callable(registry_options["job_id_factory"]))

            self.assertEqual(backend.warm_calls, 1)
            self.assertEqual(backend.close_calls, 1)
            self.assertTrue(backend._closed)
            self.assertEqual(
                backend.termination_calls,
                [(4242, "d" * 64)],
            )
            self.assertEqual(
                registry_audit.generated_job_ids,
                [
                    FIRST_JOB_ID,
                    SECOND_JOB_ID,
                    DEATH_PROBE_JOB_ID,
                    POST_POISON_PROBE_JOB_ID,
                ],
            )
            self.assertEqual(
                [request.job_id for request in backend.stream_requests],
                [FIRST_JOB_ID, SECOND_JOB_ID, DEATH_PROBE_JOB_ID],
            )
            self.assertEqual(
                backend.lifecycle,
                [
                    (FIRST_JOB_ID, "started"),
                    (FIRST_JOB_ID, "completed"),
                    (SECOND_JOB_ID, "started"),
                    (SECOND_JOB_ID, "completed"),
                    (DEATH_PROBE_JOB_ID, "started"),
                    (DEATH_PROBE_JOB_ID, "failed"),
                ],
            )
            self.assertEqual(backend.drain_calls, [FIRST_JOB_ID, SECOND_JOB_ID])
            self.assertEqual(backend._completions, {})
            self.assertTrue(backend.poisoned)
            self.assertEqual(len(registry_audit.registries), 1)
            self.assertTrue(registry_audit.registries[0]._backend_poisoned)

            self.assertEqual(json.loads(output.read_text("utf-8")), report)
            self.assertEqual(len(link_calls), 1)
            self.assertEqual(link_calls[0][1], Path(output.name))
            self.assertGreaterEqual(len(fsync_targets), 2)
            self.assertEqual(fsync_targets[0], "file")
            self.assertEqual(fsync_targets[-1], "directory")
            self.assertEqual(list(Path(temporary).iterdir()), [output])

            self.assertEqual(report["kind"], "cf1-persistent-worker-acceptance")
            self.assertEqual(report["status"], "accepted")
            self.assertTrue(report["ready"])
            self.assertFalse(report["authorizes_quality_claim"])
            self.assertFalse(report["authorizes_performance_claim"])
            self.assertFalse(report["authorizes_browser_visibility_claim"])
            self.assertEqual(report["warm"]["worker_pid"], 4242)
            self.assertEqual(report["warm"]["worker_instance_id"], "d" * 64)
            self.assertEqual(len(report["jobs"]), 2)

            for job, expected_job_id, expected_seed in zip(
                report["jobs"],
                (FIRST_JOB_ID, SECOND_JOB_ID),
                (11, 12),
            ):
                audited_chunks = registry_audit.parent_received_chunks[expected_job_id]
                self.assertEqual(job["job_id"], expected_job_id)
                self.assertEqual(job["seed"], expected_seed)
                self.assertEqual(job["chunk_count"], 21)
                self.assertEqual(job["frame_count"], 81)
                self.assertTrue(job["completion_evidence_reconciled"])
                self.assertEqual(
                    len(job["parent_received_chunks"]), len(audited_chunks)
                )
                for reported_chunk, audited_chunk in zip(
                    job["parent_received_chunks"], audited_chunks
                ):
                    self.assertEqual(
                        reported_chunk["chunk_index"], audited_chunk["chunk_index"]
                    )
                    self.assertEqual(
                        reported_chunk["first_frame_index"],
                        audited_chunk["first_frame_index"],
                    )
                    self.assertEqual(
                        reported_chunk["frame_count"], audited_chunk["frame_count"]
                    )
                    self.assertEqual(
                        reported_chunk["service_validated_ready_ns"],
                        audited_chunk["ready_ns"],
                    )
                    self.assertGreaterEqual(
                        reported_chunk["acceptance_validated_ns"],
                        reported_chunk["service_validated_ready_ns"],
                    )
                    self.assertEqual(
                        reported_chunk["frame_payload_sha256"],
                        audited_chunk["frame_payload_sha256"],
                    )
                self.assertEqual(
                    [chunk["chunk_index"] for chunk in audited_chunks],
                    list(range(21)),
                )
                self.assertEqual(
                    [chunk["first_frame_index"] for chunk in audited_chunks],
                    [0] + [1 + 4 * index for index in range(20)],
                )
                self.assertEqual(
                    [chunk["frame_count"] for chunk in audited_chunks],
                    [1] + [4] * 20,
                )
                self.assertEqual(len(_flatten_chunk_hashes(audited_chunks)), 81)

            first_hashes = _flatten_chunk_hashes(
                registry_audit.parent_received_chunks[FIRST_JOB_ID]
            )
            second_hashes = _flatten_chunk_hashes(
                registry_audit.parent_received_chunks[SECOND_JOB_ID]
            )
            self.assertNotEqual(first_hashes, second_hashes)
            self.assertTrue(report["reuse"]["same_worker_pid"])
            self.assertTrue(report["reuse"]["same_worker_instance"])
            self.assertTrue(report["reuse"]["distinct_seed_outputs"])
            self.assertEqual(report["forced_death"]["worker_pid"], 4242)
            self.assertEqual(report["forced_death"]["worker_instance_id"], "d" * 64)
            self.assertTrue(report["forced_death"]["death_probe_awaited"])
            self.assertTrue(report["forced_death"]["backend_poisoned"])
            self.assertTrue(report["forced_death"]["registry_poisoned"])
            self.assertTrue(report["forced_death"]["post_poison_start_rejected"])
            self.assertTrue(report["forced_death"]["worker_reaped"])
            self.assertFalse(report["forced_death"]["backend_ready"])
            self.assertIn((DEATH_PROBE_JOB_ID, "job_failed"), registry_audit.events)
            self.assertEqual(
                registry_audit.failure_codes[DEATH_PROBE_JOB_ID],
                "backend_fatal",
            )
            self.assertNotIn(
                (POST_POISON_PROBE_JOB_ID, "job_started"), registry_audit.events
            )

    async def test_identity_output_or_evidence_mismatch_never_publishes(self) -> None:
        cases = (
            ("pid-drift", FakeAcceptanceBackend(pid_drift=True)),
            ("instance-drift", FakeAcceptanceBackend(instance_drift=True)),
            ("evidence-mismatch", FakeAcceptanceBackend(evidence_mismatch=True)),
            ("identical-output", FakeAcceptanceBackend(ignore_seed=True)),
            ("invalid-png", FakeAcceptanceBackend(invalid_png=True)),
            ("wrong-dimensions", FakeAcceptanceBackend(wrong_dimensions=True)),
        )
        for label, backend in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "acceptance.json"
                registry_audit = RegistryAudit()
                with (
                    mock.patch(
                        "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                        return_value=backend,
                    ),
                    mock.patch(
                        "bench.cf_worker_acceptance.StreamingJobRegistry",
                        side_effect=registry_audit.build,
                    ),
                ):
                    with self.assertRaises(AcceptanceError):
                        await run_cf1_worker_acceptance(
                            prompt=PROMPT,
                            first_seed=11,
                            second_seed=12,
                            expected_stack_sha256=STACK_SHA256,
                            expected_worker_code_sha256=WORKER_CODE_SHA256,
                            **RUNTIME_LAUNCH,
                            output_manifest=output,
                        )
                self.assertFalse(output.exists())
                self.assertEqual(backend.close_calls, 1)
                self.assertTrue(backend._closed)

    async def test_unprovable_publish_rollback_is_not_reported_as_refusal(self) -> None:
        backend = FakeAcceptanceBackend()
        registry_audit = RegistryAudit()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with (
                mock.patch(
                    "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                    return_value=backend,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance.StreamingJobRegistry",
                    side_effect=registry_audit.build,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance._fsync_directory",
                    side_effect=OSError("simulated persistent fsync failure"),
                ),
            ):
                report = await cf1_worker_acceptance_report(
                    prompt=PROMPT,
                    first_seed=11,
                    second_seed=12,
                    expected_stack_sha256=STACK_SHA256,
                    expected_worker_code_sha256=WORKER_CODE_SHA256,
                    **RUNTIME_LAUNCH,
                    output_manifest=output,
                )
            self.assertFalse(report["ready"])
            self.assertEqual(report["status"], "publication_indeterminate")
            self.assertEqual(backend.close_calls, 1)

    async def test_death_probe_must_make_backend_poison_monotonic(self) -> None:
        backend = FakeAcceptanceBackend(poison_on_dead_probe=False)
        registry_audit = RegistryAudit()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with (
                mock.patch(
                    "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                    return_value=backend,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance.StreamingJobRegistry",
                    side_effect=registry_audit.build,
                ),
            ):
                with self.assertRaisesRegex(AcceptanceError, "poison"):
                    await run_cf1_worker_acceptance(
                        prompt=PROMPT,
                        first_seed=11,
                        second_seed=12,
                        expected_stack_sha256=STACK_SHA256,
                        expected_worker_code_sha256=WORKER_CODE_SHA256,
                        **RUNTIME_LAUNCH,
                        output_manifest=output,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(backend.close_calls, 1)
            self.assertFalse(backend.poisoned)
            self.assertTrue(registry_audit.registries[0]._backend_poisoned)

    async def test_invalid_or_existing_output_refuses_before_backend_build(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            output.write_text("do-not-replace", encoding="utf-8")
            with mock.patch(
                "bench.cf_worker_acceptance.build_cf1_process_streaming_backend"
            ) as build:
                report = await cf1_worker_acceptance_report(
                    prompt=PROMPT,
                    first_seed=11,
                    second_seed=12,
                    expected_stack_sha256=STACK_SHA256,
                    expected_worker_code_sha256=WORKER_CODE_SHA256,
                    **RUNTIME_LAUNCH,
                    output_manifest=output,
                )
            self.assertFalse(report["ready"])
            build.assert_not_called()
            self.assertEqual(output.read_text("utf-8"), "do-not-replace")

        for first_seed, second_seed in ((11, 11), (-1, 12), (11, 2**32)):
            with (
                self.subTest(seeds=(first_seed, second_seed)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary) / "acceptance.json"
                with mock.patch(
                    "bench.cf_worker_acceptance.build_cf1_process_streaming_backend"
                ) as build:
                    report = await cf1_worker_acceptance_report(
                        prompt=PROMPT,
                        first_seed=first_seed,
                        second_seed=second_seed,
                        expected_stack_sha256=STACK_SHA256,
                        expected_worker_code_sha256=WORKER_CODE_SHA256,
                        **RUNTIME_LAUNCH,
                        output_manifest=output,
                    )
                self.assertFalse(report["ready"])
                self.assertFalse(output.exists())
                build.assert_not_called()

    async def test_publish_race_cannot_clobber_a_competing_file(self) -> None:
        backend = FakeAcceptanceBackend()
        registry_audit = RegistryAudit()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"

            def competing_link(
                _source: Any, destination: Any, *_args: Any, **_kwargs: Any
            ) -> None:
                output.write_text("won-by-racer", encoding="utf-8")
                raise FileExistsError("simulated publish race")

            with (
                mock.patch(
                    "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                    return_value=backend,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance.StreamingJobRegistry",
                    side_effect=registry_audit.build,
                ),
                mock.patch(
                    "bench.cf_worker_acceptance.os.link",
                    side_effect=competing_link,
                ),
            ):
                with self.assertRaises(AcceptanceError):
                    await run_cf1_worker_acceptance(
                        prompt=PROMPT,
                        first_seed=11,
                        second_seed=12,
                        expected_stack_sha256=STACK_SHA256,
                        expected_worker_code_sha256=WORKER_CODE_SHA256,
                        **RUNTIME_LAUNCH,
                        output_manifest=output,
                    )
            self.assertEqual(output.read_text("utf-8"), "won-by-racer")
            self.assertEqual(list(Path(temporary).iterdir()), [output])
            self.assertEqual(backend.close_calls, 1)

    async def test_candidate_refusal_is_sanitized_and_closes_built_backend(
        self,
    ) -> None:
        secret = "Bearer TOP-SECRET-CANDIDATE-TOKEN"
        cases: tuple[tuple[str, FakeAcceptanceBackend | None], ...] = (
            ("build", None),
            (
                "warm",
                FakeAcceptanceBackend(
                    warm_error=WorkerProtocolError(
                        f"candidate runtime refused: {secret}"
                    )
                ),
            ),
        )
        for label, backend in cases:
            with self.subTest(stage=label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "acceptance.json"
                builder_result: Any
                if backend is None:
                    builder_result = mock.DEFAULT
                    builder_patch = mock.patch(
                        "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                        side_effect=WorkerProtocolError(
                            f"candidate runtime refused: {secret}"
                        ),
                    )
                else:
                    builder_result = backend
                    builder_patch = mock.patch(
                        "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                        return_value=builder_result,
                    )
                with builder_patch:
                    report = await cf1_worker_acceptance_report(
                        prompt=PROMPT,
                        first_seed=11,
                        second_seed=12,
                        expected_stack_sha256=STACK_SHA256,
                        expected_worker_code_sha256=WORKER_CODE_SHA256,
                        **RUNTIME_LAUNCH,
                        output_manifest=output,
                    )
                expected_report = {
                    "schema_version": 1,
                    "kind": "cf1-persistent-worker-acceptance",
                    "ready": False,
                    "status": "refused",
                    "failure_type": "WorkerProtocolError",
                }
                if backend is not None:
                    expected_report["failure_stage"] = "warm"
                self.assertEqual(report, expected_report)
                self.assertNotIn(secret, json.dumps(report))
                self.assertFalse(output.exists())
                if backend is not None:
                    self.assertEqual(backend.close_calls, 1)

    async def test_warm_refusal_reports_only_whitelisted_child_failure_type(
        self,
    ) -> None:
        secret = "Bearer TOP-SECRET-WORKER-TOKEN"
        backend = FakeAcceptanceBackend(
            warm_error=WorkerProtocolError(f"worker startup failed: {secret}"),
            stderr_tail=(
                f"third-party diagnostic containing {secret}\n".encode("utf-8")
                + b"real worker fatal: RuntimeError\n"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with mock.patch(
                "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                return_value=backend,
            ):
                report = await cf1_worker_acceptance_report(
                    prompt=PROMPT,
                    first_seed=11,
                    second_seed=12,
                    expected_stack_sha256=STACK_SHA256,
                    expected_worker_code_sha256=WORKER_CODE_SHA256,
                    **RUNTIME_LAUNCH,
                    output_manifest=output,
                )
        self.assertEqual(
            report,
            {
                "schema_version": 1,
                "kind": "cf1-persistent-worker-acceptance",
                "ready": False,
                "status": "refused",
                "failure_type": "WorkerProtocolError",
                "failure_stage": "warm",
                "worker_fatal_type": "RuntimeError",
            },
        )
        self.assertNotIn(secret, json.dumps(report))
        self.assertFalse(output.exists())
        self.assertEqual(backend.close_calls, 1)

    async def test_warm_refusal_maps_only_an_exact_parent_failure_code(self) -> None:
        backend = FakeAcceptanceBackend(
            warm_error=WorkerProtocolError(
                "sensitive environment names reached worker"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance.json"
            with mock.patch(
                "bench.cf_worker_acceptance.build_cf1_process_streaming_backend",
                return_value=backend,
            ):
                report = await cf1_worker_acceptance_report(
                    prompt=PROMPT,
                    first_seed=11,
                    second_seed=12,
                    expected_stack_sha256=STACK_SHA256,
                    expected_worker_code_sha256=WORKER_CODE_SHA256,
                    **RUNTIME_LAUNCH,
                    output_manifest=output,
                )
        self.assertEqual(report["failure_stage"], "warm")
        self.assertEqual(report["failure_type"], "WorkerProtocolError")
        self.assertEqual(
            report["parent_failure_code"],
            "sensitive_environment_names",
        )
        self.assertNotIn(
            "sensitive environment names reached worker", json.dumps(report)
        )
        self.assertFalse(output.exists())
        self.assertEqual(backend.close_calls, 1)
