from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import socket
import struct
import threading
import unittest
from pathlib import Path
from unittest import mock

from bench import streaming_process, streaming_process_worker
from bench.streaming_process import (
    DEFAULT_WORKER_SCRIPT,
    ProcessStreamingBackend,
    WorkerProtocolError,
    _receive_packet,
)
from bench.streaming_process_protocol import (
    WORKER_PROTOCOL_MAX_LATENT_FRAMES,
    WORKER_PROTOCOL_VERSION,
)
from bench.streaming_process_worker import ProtocolFailure, validate_start
from bench.streaming_service import (
    BackendFatalError,
    StreamProtocolError,
    StreamRequest,
    run_stream_job,
)


STACK_SHA256 = hashlib.sha256(b"fake-stack-v1").hexdigest()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DEFINITION = DEFAULT_WORKER_SCRIPT.with_name(
    "streaming_process_protocol.py"
)


def expected_worker_bundle_sha256(worker_script: Path) -> str:
    digest = hashlib.sha256()
    for label, path in (
        (b"protocol", PROTOCOL_DEFINITION),
        (b"worker", worker_script),
    ):
        payload = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def request(job_id: str, *, seed: int = 7) -> StreamRequest:
    return StreamRequest(
        client_id="process-client",
        job_id=job_id,
        prompt="A red fox runs through snow.",
        seed=seed,
        latent_frames=21,
    )


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def collect(backend: ProcessStreamingBackend, job_id: str, *, seed: int = 7):
    chunks = []
    async for chunk in backend.stream(request(job_id, seed=seed)):
        chunks.append(chunk)
    return chunks


class ProcessStreamingBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.backends: list[ProcessStreamingBackend] = []

    async def asyncTearDown(self) -> None:
        await asyncio.gather(
            *(backend.close() for backend in self.backends),
            return_exceptions=True,
        )

    def backend(self, **kwargs: object) -> ProcessStreamingBackend:
        backend = ProcessStreamingBackend(stack_sha256=STACK_SHA256, **kwargs)
        self.backends.append(backend)
        return backend

    async def test_clean_jobs_reuse_pid_and_isolate_completion_evidence(self) -> None:
        backend = self.backend()
        first = await collect(backend, "job-one", seed=7)
        first_pid = backend.worker_pid
        first_instance = backend.worker_instance_id
        self.assertIsNotNone(first_pid)
        self.assertIsNotNone(first_instance)

        self.assertEqual([chunk.frame_count for chunk in first], [1] + [4] * 20)
        self.assertEqual(sum(chunk.frame_count for chunk in first), 81)
        self.assertTrue(all(chunk.frame_media_type == "image/png" for chunk in first))
        self.assertTrue(
            all(
                isinstance(payload, bytes)
                and payload.startswith(b"\x89PNG\r\n\x1a\n")
                and payload.endswith(b"IEND\xaeB`\x82")
                for chunk in first
                for payload in chunk.frame_payloads
            )
        )
        evidence_one = backend.drain_completion_evidence("job-one")
        self.assertIsNotNone(evidence_one)
        assert evidence_one is not None
        delivered_payloads = tuple(
            payload for chunk in first for payload in chunk.frame_payloads
        )
        self.assertEqual(evidence_one.worker_instance_id, first_instance)
        self.assertEqual(evidence_one.job_id, "job-one")
        self.assertEqual(evidence_one.stack_sha256, STACK_SHA256)
        self.assertEqual(
            evidence_one.worker_code_sha256,
            expected_worker_bundle_sha256(DEFAULT_WORKER_SCRIPT),
        )
        self.assertEqual(
            evidence_one.prompt_sha256,
            hashlib.sha256(request("job-one").prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(evidence_one.seed, 7)
        self.assertEqual(evidence_one.chunk_count, 21)
        self.assertEqual(evidence_one.chunk_frame_counts, (1, *([4] * 20)))
        self.assertEqual(evidence_one.frame_count, 81)
        self.assertEqual(
            evidence_one.frame_payload_sha256,
            tuple(hashlib.sha256(payload).hexdigest() for payload in delivered_payloads),
        )
        self.assertIsNone(backend.drain_completion_evidence("job-one"))

        second = await collect(backend, "job-two", seed=8)
        self.assertEqual(sum(chunk.frame_count for chunk in second), 81)
        self.assertEqual(backend.worker_pid, first_pid)
        self.assertEqual(backend.worker_instance_id, first_instance)
        evidence_two = backend.drain_completion_evidence("job-two")
        self.assertIsNotNone(evidence_two)
        assert evidence_two is not None
        self.assertEqual(evidence_two.worker_code_sha256, evidence_one.worker_code_sha256)
        self.assertEqual(evidence_two.stack_sha256, evidence_one.stack_sha256)
        self.assertIsNone(backend.drain_completion_evidence("job-one"))

    async def test_real_service_runner_consumes_process_backend_and_keeps_warm_worker(self) -> None:
        backend = self.backend()
        delivered = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        summary = await run_stream_job(
            request("job-service-integration"),
            backend,
            emit=emit,
        )
        self.assertEqual(summary.frame_count, 81)
        self.assertEqual(summary.release_count, 21)
        self.assertIsNotNone(backend.worker_pid)
        self.assertFalse(backend.poisoned)
        self.assertEqual(delivered[0].kind, "job_started")
        self.assertEqual(
            [event.kind for event in delivered].count("chunk_ready"),
            21,
        )
        self.assertEqual(delivered[-1].kind, "job_completed")
        evidence = backend.drain_completion_evidence("job-service-integration")
        self.assertIsNotNone(evidence)

    async def test_created_but_uniterated_generator_does_not_claim_worker(self) -> None:
        backend = self.backend()
        abandoned = backend.stream(request("abandoned"))
        chunks = await collect(backend, "real-job")
        self.assertEqual(sum(chunk.frame_count for chunk in chunks), 81)
        await abandoned.aclose()

    async def test_concurrent_stream_is_rejected_atomically(self) -> None:
        backend = self.backend()
        first = backend.stream(request("job-one"))
        first_chunk = await first.__anext__()
        self.assertEqual(first_chunk.frame_count, 1)
        second = backend.stream(request("job-two"))
        with self.assertRaisesRegex(StreamProtocolError, "already active"):
            await second.__anext__()
        next_first_chunk = await first.__anext__()
        self.assertEqual(next_first_chunk.frame_count, 4)
        third = backend.stream(request("job-three"))
        with self.assertRaisesRegex(StreamProtocolError, "already active"):
            await third.__anext__()
        await first.aclose()

    async def test_aclose_kills_and_reaps_before_cold_respawn(self) -> None:
        backend = self.backend()
        iterator = backend.stream(request("job-one"))
        await iterator.__anext__()
        old_pid = backend.worker_pid
        old_instance = backend.worker_instance_id
        self.assertIsNotNone(old_pid)
        assert old_pid is not None
        await asyncio.wait_for(
            iterator.aclose(),
            timeout=backend.registry_close_timeout_s,
        )
        self.assertFalse(pid_exists(old_pid))
        self.assertFalse(backend.poisoned)

        replacement = backend.stream(request("job-two"))
        await replacement.__anext__()
        self.assertNotEqual(backend.worker_pid, old_pid)
        self.assertNotEqual(backend.worker_instance_id, old_instance)
        await replacement.aclose()

    async def test_prompt_limit_is_job_error_without_spawning_or_poisoning(self) -> None:
        backend = self.backend(max_prompt_bytes=4)
        too_long = StreamRequest(
            client_id="process-client",
            job_id="job-too-long",
            prompt="12345",
            seed=7,
            latent_frames=21,
        )
        with self.assertRaisesRegex(StreamProtocolError, "prompt"):
            async for _chunk in backend.stream(too_long):
                pass
        self.assertIsNone(backend.worker_pid)
        self.assertFalse(backend.poisoned)

        short = StreamRequest(
            client_id="process-client",
            job_id="job-short",
            prompt="okay",
            seed=7,
            latent_frames=21,
        )
        chunks = []
        async for chunk in backend.stream(short):
            chunks.append(chunk)
        self.assertEqual(sum(chunk.frame_count for chunk in chunks), 81)

    async def test_cancel_storm_sends_kill_before_cleanup_awaits(self) -> None:
        backend = self.backend(worker_args=("--fault", "hang-next"), reap_timeout_s=0.2)
        iterator = backend.stream(request("job-hang"))
        pending = asyncio.create_task(iterator.__anext__())
        for _ in range(100):
            if backend.worker_pid is not None:
                break
            await asyncio.sleep(0.001)
        pid = backend.worker_pid
        self.assertIsNotNone(pid)
        assert pid is not None
        pending.cancel()
        asyncio.get_running_loop().call_soon(pending.cancel)
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(pid_exists(pid))

    async def test_injected_baseexception_kills_reaps_and_poisons(self) -> None:
        for fatal in (KeyboardInterrupt(), SystemExit(9)):
            with self.subTest(fatal=type(fatal).__name__):
                backend = self.backend()
                iterator = backend.stream(
                    request(f"fatal-{type(fatal).__name__.lower()}")
                )
                first = await iterator.__anext__()
                self.assertEqual(first.frame_count, 1)
                pid = backend.worker_pid
                self.assertIsNotNone(pid)
                assert pid is not None

                with self.assertRaises(type(fatal)):
                    await iterator.athrow(fatal)

                self.assertFalse(pid_exists(pid))
                self.assertTrue(backend.poisoned)
                with self.assertRaisesRegex(BackendFatalError, "poisoned"):
                    backend.stream(request(f"after-{type(fatal).__name__.lower()}"))

    async def test_cancel_during_contended_release_cannot_strand_job_ownership(self) -> None:
        backend = self.backend(worker_args=("--fault", "hang-next"), reap_timeout_s=0.2)
        iterator = backend.stream(request("job-contended-cancel"))
        pending = asyncio.create_task(iterator.__anext__())
        for _ in range(100):
            if backend.worker_pid is not None:
                break
            await asyncio.sleep(0.001)
        self.assertIsNotNone(backend.worker_pid)

        await backend._claim_lock.acquire()
        try:
            pending.cancel()
            for _ in range(100):
                if backend.worker_pid is None:
                    break
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        finally:
            backend._claim_lock.release()

        self.assertIsNone(backend._active_job_id)
        backend.worker_args = ()
        replacement = backend.stream(request("job-after-contended-cancel"))
        await replacement.__anext__()
        await replacement.aclose()

    async def test_protocol_corruption_poisons_and_later_stream_fails_synchronously(self) -> None:
        backend = self.backend(worker_args=("--fault", "wrong-job"))
        with self.assertRaises(BackendFatalError):
            await collect(backend, "job-corrupt")
        self.assertTrue(backend.poisoned)
        with self.assertRaisesRegex(BackendFatalError, "poisoned"):
            backend.stream(request("job-after-poison"))

    async def test_epoch_index_and_early_complete_corruption_each_poison(self) -> None:
        for fault in ("wrong-worker", "wrong-index", "early-complete", "extra-chunk"):
            with self.subTest(fault=fault):
                backend = self.backend(worker_args=("--fault", fault))
                with self.assertRaises(BackendFatalError):
                    await collect(backend, f"job-{fault}")
                self.assertTrue(backend.poisoned)

    async def test_unsolicited_or_wrong_type_response_cannot_spend_future_credit(self) -> None:
        for fault in ("unsolicited", "wrong-type"):
            with self.subTest(fault=fault):
                backend = self.backend(worker_args=("--fault", fault))
                with self.assertRaises(BackendFatalError):
                    await collect(backend, f"job-{fault}")
                self.assertTrue(backend.poisoned)

    async def test_ipc_timeout_kills_reaps_and_poisons(self) -> None:
        backend = self.backend(
            worker_args=("--fault", "hang-next"),
            io_timeout_s=0.05,
            reap_timeout_s=0.2,
        )
        iterator = backend.stream(request("job-timeout"))
        pending = asyncio.create_task(iterator.__anext__())
        for _ in range(100):
            if backend.worker_pid is not None:
                break
            await asyncio.sleep(0.001)
        pid = backend.worker_pid
        self.assertIsNotNone(pid)
        assert pid is not None
        with self.assertRaises(BackendFatalError):
            await pending
        self.assertTrue(backend.poisoned)
        self.assertFalse(pid_exists(pid))

    async def test_forged_payload_and_completion_evidence_poison(self) -> None:
        for fault in ("bad-frame-hash", "bad-completion"):
            with self.subTest(fault=fault):
                backend = self.backend(worker_args=("--fault", fault))
                with self.assertRaises(BackendFatalError):
                    await collect(backend, f"job-{fault}")
                self.assertTrue(backend.poisoned)

    async def test_handshake_mismatch_and_hang_fail_closed(self) -> None:
        mismatch = self.backend(worker_args=("--fault", "wrong-digest"))
        with self.assertRaises(BackendFatalError):
            await collect(mismatch, "job-mismatch")
        self.assertTrue(mismatch.poisoned)

        hung = self.backend(
            worker_args=("--fault", "hang-handshake"),
            startup_timeout_s=0.05,
            reap_timeout_s=0.2,
        )
        with self.assertRaises(BackendFatalError):
            await collect(hung, "job-hung")
        self.assertTrue(hung.poisoned)

    async def test_warm_start_rejects_stray_bytes_after_complete(self) -> None:
        backend = self.backend(worker_args=("--fault", "stray-after-complete"))
        await collect(backend, "job-one")
        with self.assertRaises(BackendFatalError):
            await collect(backend, "job-two")
        self.assertTrue(backend.poisoned)

    async def test_all_chunks_without_complete_kills_worker_and_publishes_no_evidence(self) -> None:
        backend = self.backend()
        iterator = backend.stream(request("job-no-complete"))
        chunks = [await iterator.__anext__() for _ in range(21)]
        self.assertEqual(sum(chunk.frame_count for chunk in chunks), 81)
        old_instance = backend.worker_instance_id
        old_pid = backend.worker_pid
        self.assertIsNotNone(old_instance)
        self.assertIsNotNone(old_pid)
        await iterator.aclose()
        assert old_pid is not None
        self.assertFalse(pid_exists(old_pid))
        self.assertIsNone(backend.drain_completion_evidence("job-no-complete"))
        self.assertFalse(backend.poisoned)

        replacement = backend.stream(request("job-after-incomplete"))
        await replacement.__anext__()
        self.assertNotEqual(backend.worker_instance_id, old_instance)
        await replacement.aclose()

    async def test_environment_is_allowlisted_and_interpreter_is_isolated(self) -> None:
        backend = self.backend()
        with mock.patch.dict(
            os.environ,
            {
                "TWELVELABS_API_KEY": "not-forwarded",
                "GEMINI_API_KEY": "not-forwarded",
                "PRIVATE_TOKEN": "not-forwarded",
            },
            clear=False,
        ):
            await collect(backend, "job-env")
        self.assertTrue(backend.child_isolated)
        self.assertEqual(backend.child_sensitive_environment_names, ())

    async def test_parent_rejects_false_isolation_and_sensitive_worker_environment(
        self,
    ) -> None:
        for fault, message in (
            ("false-isolation", "worker interpreter is not isolated"),
            ("sensitive-environment", "sensitive environment names reached worker"),
        ):
            with self.subTest(fault=fault):
                backend = self.backend(worker_args=("--fault", fault))
                with self.assertRaisesRegex(WorkerProtocolError, message):
                    await collect(backend, f"job-{fault}")
                self.assertTrue(backend.poisoned)

    async def test_stderr_flood_is_drained_without_threads_or_deadlock(self) -> None:
        before = {thread.ident for thread in threading.enumerate()}
        backend = self.backend(worker_args=("--fault", "stderr-flood"))
        with mock.patch.object(
            threading.Thread,
            "start",
            autospec=True,
            side_effect=AssertionError("process supervisor must not start threads"),
        ):
            chunks = await collect(backend, "job-stderr")
        after = {thread.ident for thread in threading.enumerate()}
        self.assertEqual(sum(chunk.frame_count for chunk in chunks), 81)
        self.assertLessEqual(len(backend.stderr_tail), backend.stderr_tail_bytes)
        self.assertEqual(after, before)

    async def test_stderr_reader_callback_has_finite_work_budget(self) -> None:
        backend = self.backend()
        with mock.patch(
            "bench.streaming_process.os.read",
            side_effect=[b"x"] * 5 + [BlockingIOError()],
        ) as read:
            backend._on_stderr_ready(123)
        self.assertLessEqual(read.call_count, 4)

    async def test_cancel_kills_the_worker_process_group(self) -> None:
        backend = self.backend(worker_args=("--fault", "fork-helper"))
        iterator = backend.stream(request("job-process-group"))
        await iterator.__anext__()
        leader_pid = backend.worker_pid
        self.assertIsNotNone(leader_pid)
        helper_pid = None
        for _ in range(200):
            match = re.search(rb"helper-pid=(\d+)", backend.stderr_tail)
            if match is not None:
                helper_pid = int(match.group(1))
                break
            await asyncio.sleep(0.005)
        self.assertIsNotNone(helper_pid)
        assert leader_pid is not None and helper_pid is not None

        helper_gone = False
        try:
            await iterator.aclose()
            for _ in range(200):
                if not pid_exists(helper_pid):
                    helper_gone = True
                    break
                await asyncio.sleep(0.005)
        finally:
            if pid_exists(helper_pid):
                with self.subTest(cleanup="orphan helper"):
                    os.kill(helper_pid, signal.SIGKILL)
        self.assertFalse(pid_exists(leader_pid))
        self.assertTrue(helper_gone, "worker helper survived supervisor cancellation")

    async def test_deadline_ordering_is_constructor_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "registry chunk timeout"):
            ProcessStreamingBackend(
                stack_sha256=STACK_SHA256,
                io_timeout_s=1.0,
                registry_chunk_timeout_s=1.0,
            )

    async def test_direct_backend_rejects_overlong_rollout_before_spawning(self) -> None:
        backend = self.backend()
        overlong = StreamRequest(
            client_id="process-client",
            job_id="job-overlong",
            prompt="bounded rollout",
            seed=7,
            latent_frames=22,
        )
        with self.assertRaisesRegex(StreamProtocolError, "latent_frames"):
            async for _chunk in backend.stream(overlong):
                pass
        self.assertIsNone(backend.worker_pid)
        self.assertFalse(backend.poisoned)

    async def test_constructor_rejects_cap_above_worker_protocol_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker protocol limit"):
            ProcessStreamingBackend(
                stack_sha256=STACK_SHA256,
                max_latent_frames=22,
            )

    async def test_parent_and_isolated_worker_share_protocol_limits(self) -> None:
        self.assertEqual(
            streaming_process.WORKER_PROTOCOL_VERSION,
            WORKER_PROTOCOL_VERSION,
        )
        self.assertEqual(
            streaming_process.WORKER_PROTOCOL_MAX_LATENT_FRAMES,
            WORKER_PROTOCOL_MAX_LATENT_FRAMES,
        )
        self.assertEqual(
            streaming_process_worker.PROTOCOL_VERSION,
            WORKER_PROTOCOL_VERSION,
        )
        self.assertEqual(
            streaming_process_worker.MAX_LATENT_FRAMES,
            WORKER_PROTOCOL_MAX_LATENT_FRAMES,
        )

    async def test_parent_rejects_worker_with_drifted_latent_cap(self) -> None:
        backend = self.backend(worker_args=("--fault", "wrong-latent-cap"))
        with self.assertRaisesRegex(
            WorkerProtocolError,
            "worker latent frame limit does not match protocol",
        ):
            await collect(backend, "job-drifted-cap")
        self.assertTrue(backend.poisoned)

    async def test_worker_rejects_overlong_start_defense_in_depth(self) -> None:
        worker_code_sha256 = expected_worker_bundle_sha256(DEFAULT_WORKER_SCRIPT)
        instance_id = hashlib.sha256(b"worker-instance").hexdigest()
        prompt = "bounded rollout"
        with self.assertRaisesRegex(ProtocolFailure, "latent frame count"):
            validate_start(
                {
                    "type": "START",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "worker_instance_id": instance_id,
                    "job_id": "job-worker-overlong",
                    "chunk_index": -1,
                    "stack_sha256": STACK_SHA256,
                    "worker_code_sha256": worker_code_sha256,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "seed": 7,
                    "latent_frames": 22,
                    "payload_lengths": [],
                    "payload_sha256": [],
                },
                instance_id,
                STACK_SHA256,
                worker_code_sha256,
            )

    async def test_lifetime_job_id_ledger_is_bounded(self) -> None:
        backend = self.backend(max_job_ids=2)
        await collect(backend, "job-ledger-one")
        backend.drain_completion_evidence("job-ledger-one")
        await collect(backend, "job-ledger-two")
        backend.drain_completion_evidence("job-ledger-two")
        with self.assertRaisesRegex(StreamProtocolError, "job_id capacity"):
            async for _chunk in backend.stream(request("job-ledger-three")):
                pass
        self.assertFalse(backend.poisoned)

    async def test_undrained_completion_evidence_applies_bounded_backpressure(self) -> None:
        backend = self.backend()
        with mock.patch("bench.streaming_process._MAX_RETAINED_COMPLETIONS", 1):
            await collect(backend, "job-evidence-one")
            warm_pid = backend.worker_pid
            with self.assertRaisesRegex(StreamProtocolError, "completion evidence capacity"):
                async for _chunk in backend.stream(request("job-evidence-two")):
                    pass
            self.assertFalse(backend.poisoned)
            self.assertEqual(backend.worker_pid, warm_pid)
            self.assertIsNotNone(
                backend.drain_completion_evidence("job-evidence-one")
            )
            await collect(backend, "job-evidence-two")
            self.assertEqual(backend.worker_pid, warm_pid)

    async def test_reap_timeout_detaches_io_without_forgetting_process(self) -> None:
        backend = self.backend(
            reap_timeout_s=0.01,
            registry_close_timeout_s=0.1,
        )
        process = mock.Mock()
        process.pid = 424242
        process.poll.return_value = None
        process.stderr = None
        worker_socket = mock.Mock()
        backend._process = process
        backend._socket = worker_socket
        try:
            with mock.patch.object(
                backend, "_remove_stderr_reader"
            ) as remove_reader, mock.patch(
                "bench.streaming_process.os.killpg"
            ) as killpg:
                await backend._poison_after_failure()
            self.assertTrue(backend.poisoned)
            killpg.assert_called_once_with(process.pid, signal.SIGKILL)
            remove_reader.assert_called_once_with(process)
            worker_socket.close.assert_called_once_with()
            self.assertIsNone(backend._socket)
            self.assertIs(backend._process, process)
        finally:
            backend._process = None
            backend._socket = None
        with self.assertRaisesRegex(ValueError, "registry close timeout"):
            ProcessStreamingBackend(
                stack_sha256=STACK_SHA256,
                reap_timeout_s=2.0,
                registry_close_timeout_s=2.0,
            )
        with self.assertRaisesRegex(ValueError, "registry chunk timeout"):
            ProcessStreamingBackend(
                stack_sha256=STACK_SHA256,
                startup_timeout_s=0.6,
                io_timeout_s=0.6,
                reap_timeout_s=0.5,
                registry_chunk_timeout_s=1.0,
            )

    async def test_parent_rejects_sensitive_launch_environment_before_spawn(
        self,
    ) -> None:
        with mock.patch.object(streaming_process.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "sensitive"):
                ProcessStreamingBackend(
                    stack_sha256=STACK_SHA256,
                    worker_environment={
                        "PYTHONUNBUFFERED": "1",
                        "TWELVELABS_API_KEY": "must-not-reach-child",
                    },
                )
        popen.assert_not_called()

    async def test_parent_revalidates_mutated_environment_at_spawn_boundary(
        self,
    ) -> None:
        backend = self.backend()
        backend.worker_environment = {
            "PYTHONUNBUFFERED": "1",
            "TWELVELABS_API_KEY": "must-not-reach-child",
        }
        with mock.patch.object(
            streaming_process.subprocess,
            "Popen",
            side_effect=AssertionError("sensitive environment reached Popen"),
        ) as popen:
            with self.assertRaisesRegex(WorkerProtocolError, "sensitive"):
                await backend.warm()
        popen.assert_not_called()
        self.assertTrue(backend.poisoned)

    async def test_close_is_idempotent_and_terminal_while_worker_is_active(self) -> None:
        backend = self.backend()
        iterator = backend.stream(request("job-active-close"))
        await iterator.__anext__()
        pid = backend.worker_pid
        self.assertIsNotNone(pid)
        await backend.close()
        assert pid is not None
        self.assertFalse(pid_exists(pid))
        await backend.close()
        with self.assertRaisesRegex(BackendFatalError, "closed"):
            backend.stream(request("job-after-close"))
        await iterator.aclose()

    async def test_close_while_next_is_pending_reaps_and_retires_owner(self) -> None:
        backend = self.backend(worker_args=("--fault", "hang-next"), reap_timeout_s=0.2)
        iterator = backend.stream(request("job-close-pending"))
        pending = asyncio.create_task(iterator.__anext__())
        for _ in range(100):
            if backend.worker_pid is not None:
                break
            await asyncio.sleep(0.001)
        pid = backend.worker_pid
        self.assertIsNotNone(pid)
        await asyncio.wait_for(backend.close(), timeout=1)
        with self.assertRaises(BackendFatalError):
            await pending
        assert pid is not None
        self.assertFalse(pid_exists(pid))
        self.assertIsNone(backend._active_job_id)

    async def test_oversized_and_truncated_headers_fail_before_body_read(self) -> None:
        left, right = socket.socketpair()
        left.setblocking(False)
        right.sendall(struct.pack(">I", 65_537))
        try:
            with self.assertRaisesRegex(WorkerProtocolError, "header length"):
                await _receive_packet(
                    left,
                    timeout_s=0.1,
                    max_header_bytes=65_536,
                    max_frame_bytes=1024,
                    max_chunk_bytes=4096,
                )
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        left.setblocking(False)
        right.sendall(struct.pack(">I", 10) + b"{}")
        right.close()
        try:
            with self.assertRaisesRegex(WorkerProtocolError, "truncated"):
                await _receive_packet(
                    left,
                    timeout_s=0.1,
                    max_header_bytes=65_536,
                    max_frame_bytes=1024,
                    max_chunk_bytes=4096,
                )
        finally:
            left.close()

    async def test_payload_lengths_are_bounded_before_body_read(self) -> None:
        async def assert_rejected(header: dict[str, object], pattern: str) -> None:
            encoded = json.dumps(
                header,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            left, right = socket.socketpair()
            left.setblocking(False)
            right.sendall(struct.pack(">I", len(encoded)) + encoded)
            try:
                with self.assertRaisesRegex(WorkerProtocolError, pattern):
                    await _receive_packet(
                        left,
                        timeout_s=0.1,
                        max_header_bytes=65_536,
                        max_frame_bytes=1024,
                        max_chunk_bytes=4096,
                    )
            finally:
                left.close()
                right.close()

        digest = hashlib.sha256(b"not-sent").hexdigest()
        await assert_rejected(
            {
                "type": "CHUNK",
                "payload_lengths": [1025],
                "payload_sha256": [digest],
            },
            "frame length",
        )
        await assert_rejected(
            {
                "type": "CHUNK",
                "payload_lengths": [800, 800, 800, 800, 800, 800],
                "payload_sha256": [digest, digest, digest, digest, digest, digest],
            },
            "chunk length",
        )

        encoded = json.dumps(
            {
                "type": "CHUNK",
                "payload_lengths": [10],
                "payload_sha256": [hashlib.sha256(b"0123456789").hexdigest()],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        left, right = socket.socketpair()
        left.setblocking(False)
        right.sendall(struct.pack(">I", len(encoded)) + encoded + b"01")
        right.close()
        try:
            with self.assertRaisesRegex(WorkerProtocolError, "truncated"):
                await _receive_packet(
                    left,
                    timeout_s=0.1,
                    max_header_bytes=65_536,
                    max_frame_bytes=1024,
                    max_chunk_bytes=4096,
                )
        finally:
            left.close()
            right.close()

    async def test_worker_script_exists_and_is_not_a_model_or_cuda_adapter(self) -> None:
        self.assertEqual(DEFAULT_WORKER_SCRIPT, PROJECT_ROOT / "bench" / "streaming_process_worker.py")
        self.assertTrue(DEFAULT_WORKER_SCRIPT.is_file())
        source = DEFAULT_WORKER_SCRIPT.read_text("utf-8")
        self.assertNotIn("torch", source)
        self.assertNotIn("cuda", source.lower())
        self.assertNotIn("asyncio.to_thread", source)
        supervisor_source = (
            PROJECT_ROOT / "bench" / "streaming_process.py"
        ).read_text("utf-8")
        self.assertNotIn("asyncio.create_subprocess_exec", supervisor_source)
        self.assertNotIn("run_in_executor", supervisor_source)
        self.assertNotIn("asyncio.to_thread", supervisor_source)


if __name__ == "__main__":
    unittest.main()
