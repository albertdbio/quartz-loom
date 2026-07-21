from __future__ import annotations

import asyncio
import base64
import json
import struct
import subprocess
import sys
import unittest
import zlib
from pathlib import Path

from bench.streaming_service import (
    BackendFatalError,
    DecodedChunk,
    FakeStreamingBackend,
    StreamProtocolError,
    StreamRequest,
    StreamingJobRegistry,
    run_stream_job,
)


START_NS = 1_000_000_000
FRAME_COUNTS = [1, *([4] * 20)]
READY_NS = [START_NS + 100_000_000 + index * 50_000_000 for index in range(21)]
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _rgb_png_with_idat(idat: bytes) -> bytes:
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


class SequenceClock:
    def __init__(self, *values: int):
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def request(client_id: str = "client-a", job_id: str = "job-a") -> StreamRequest:
    return StreamRequest(
        client_id=client_id,
        job_id=job_id,
        prompt="A red fox runs through snow.",
        seed=20260719,
        latent_frames=21,
    )


class StreamingJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_backend_emits_exact_chunks_and_honest_metrics(self) -> None:
        delivered: list[tuple[str, object]] = []

        async def emit(client_id: str, event: object) -> None:
            delivered.append((client_id, event))

        summary = await run_stream_job(
            request(),
            FakeStreamingBackend(frame_counts=FRAME_COUNTS),
            emit=emit,
            clock_ns=SequenceClock(
                START_NS,
                *READY_NS,
                READY_NS[-1] + 100_000_000,
            ),
            queue_capacity=2,
        )

        self.assertEqual(summary.frame_count, 81)
        self.assertEqual(summary.release_count, 21)
        self.assertAlmostEqual(summary.first_chunk_ready_s, 0.1)
        self.assertAlmostEqual(summary.p95_chunk_release_gap_ms, 50.0)
        self.assertAlmostEqual(summary.wall_e2e_s, 1.2)
        self.assertAlmostEqual(summary.e2e_fps, 67.5)
        self.assertNotIn("first_visible_rgb_s", summary.to_dict())
        self.assertEqual({client_id for client_id, _ in delivered}, {"client-a"})
        chunk_events = [event for _, event in delivered if event.kind == "chunk_ready"]
        self.assertEqual([event.frame_count for event in chunk_events], FRAME_COUNTS)
        self.assertEqual(
            [event.first_frame_index for event in chunk_events],
            [0, *[1 + 4 * index for index in range(20)]],
        )
        for _, event in delivered:
            public = event.to_dict(include_payloads=False)
            self.assertNotIn("rgb_ready_ns", public)
            self.assertNotIn("frame_ready_ns", public)
        self.assertEqual(delivered[0][1].kind, "job_started")
        self.assertEqual(delivered[-1][1].kind, "job_completed")
        self.assertEqual(chunk_events[-1].queue_depth, 0)
        encoded_first = chunk_events[0].to_dict()["frame_payloads_base64"][0]
        self.assertEqual(base64.b64decode(encoded_first), b"fake-frame-0-0")

    async def test_incomplete_release_schedule_fails_without_completion(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        with self.assertRaisesRegex(StreamProtocolError, "5.*81.*RGB frames"):
            await run_stream_job(
                request(),
                FakeStreamingBackend(
                    frame_counts=[1, 4],
                ),
                emit=emit,
                clock_ns=SequenceClock(START_NS, *READY_NS[:2]),
            )

        self.assertEqual(delivered[-1].kind, "job_failed")
        self.assertNotIn("message", delivered[-1].to_dict())
        self.assertNotIn("job_completed", [event.kind for event in delivered])

    async def test_bounded_queue_applies_backpressure(self) -> None:
        unblock_sender = asyncio.Event()
        first_chunk_seen = asyncio.Event()

        class ProbeBackend:
            def __init__(self) -> None:
                self.resumed_after_yield = 0

            async def stream(self, _request: StreamRequest):
                first_frame_index = 0
                for chunk_index, frame_count in enumerate(FRAME_COUNTS):
                    yield DecodedChunk(
                        frame_payloads=tuple(
                            f"frame-{chunk_index}-{index}".encode()
                            for index in range(frame_count)
                        ),
                    )
                    self.resumed_after_yield += 1
                    first_frame_index += frame_count

        backend = ProbeBackend()

        async def emit(_client_id: str, event: object) -> None:
            if event.kind == "chunk_ready" and event.chunk_index == 0:
                first_chunk_seen.set()
                await unblock_sender.wait()

        task = asyncio.create_task(
            run_stream_job(
                request(),
                backend,
                emit=emit,
                clock_ns=SequenceClock(
                    START_NS,
                    *READY_NS,
                    READY_NS[-1] + 100_000_000,
                ),
                queue_capacity=1,
            )
        )
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=1)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(backend.resumed_after_yield, 0)
        self.assertFalse(task.done())

        unblock_sender.set()
        await asyncio.wait_for(task, timeout=1)

    async def test_delivered_payloads_are_not_retained_for_summary(self) -> None:
        finalized: list[int] = []

        class TrackedBytes(bytes):
            def __new__(cls, value: bytes, chunk_index: int):
                instance = super().__new__(cls, value)
                instance.chunk_index = chunk_index
                return instance

            def __del__(self) -> None:
                finalized.append(self.chunk_index)

        class ReleasingBackend:
            async def stream(self, _request: StreamRequest):
                for chunk_index, frame_count in enumerate([1, 4, 4]):
                    yield DecodedChunk(
                        frame_payloads=tuple(
                            TrackedBytes(
                                f"frame-{chunk_index}-{frame_index}".encode(),
                                chunk_index,
                            )
                            for frame_index in range(frame_count)
                        )
                    )

        async def emit(_client_id: str, event: object) -> None:
            if event.kind == "chunk_ready" and event.chunk_index == 2:
                self.assertIn(0, finalized)

        short_request = StreamRequest(
            client_id="client-a",
            job_id="release-payloads",
            prompt="test",
            seed=1,
            latent_frames=3,
        )
        await run_stream_job(
            short_request,
            ReleasingBackend(),
            emit=emit,
            clock_ns=SequenceClock(
                START_NS,
                START_NS + 1,
                START_NS + 2,
                START_NS + 3,
                START_NS + 4,
            ),
            queue_capacity=1,
        )

    async def test_completion_clock_failure_emits_terminal_failure(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="bad-completion-clock",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "completion time"):
            await run_stream_job(
                one_chunk_request,
                FakeStreamingBackend(frame_counts=[1]),
                emit=emit,
                clock_ns=SequenceClock(START_NS, START_NS + 2, START_NS + 1),
            )

        self.assertEqual(delivered[-1].kind, "job_failed")
        self.assertEqual(delivered[-1].error_code, "protocol_error")

    async def test_initial_clock_failure_emits_terminal_failure(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        def broken_clock() -> int:
            raise RuntimeError("private clock failure")

        with self.assertRaisesRegex(StreamProtocolError, "started_ns clock failed"):
            await run_stream_job(
                request(job_id="bad-start-clock"),
                FakeStreamingBackend(frame_counts=FRAME_COUNTS),
                emit=emit,
                clock_ns=broken_clock,
            )

        self.assertEqual([event.kind for event in delivered], ["job_failed"])
        self.assertEqual(delivered[0].error_code, "protocol_error")

    async def test_equal_chunk_ready_time_fails_without_fabricating_time(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        two_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="equal-clock",
            prompt="test",
            seed=1,
            latent_frames=2,
        )
        with self.assertRaisesRegex(StreamProtocolError, "strictly increasing"):
            await run_stream_job(
                two_chunk_request,
                FakeStreamingBackend(frame_counts=[1, 4]),
                emit=emit,
                clock_ns=SequenceClock(
                    START_NS,
                    START_NS + 1,
                    START_NS + 1,
                ),
            )

        chunk_events = [event for event in delivered if event.kind == "chunk_ready"]
        self.assertEqual([event.ready_ns for event in chunk_events], [START_NS + 1])
        self.assertEqual(delivered[-1].kind, "job_failed")

    async def test_backend_cancelled_error_is_sanitized_and_does_not_hang(self) -> None:
        delivered: list[object] = []

        class InternallyCancelledBackend:
            async def stream(self, _request: StreamRequest):
                if False:
                    yield None
                raise asyncio.CancelledError

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        with self.assertRaisesRegex(StreamProtocolError, "backend failed"):
            await asyncio.wait_for(
                run_stream_job(
                    request(),
                    InternallyCancelledBackend(),
                    emit=emit,
                    clock_ns=SequenceClock(START_NS),
                ),
                timeout=0.1,
            )

        self.assertEqual(delivered[-1].kind, "job_failed")
        self.assertEqual(delivered[-1].error_code, "backend_failure")

    async def test_backend_chunk_deadline_fails_and_poisons_registry_gate(self) -> None:
        backend_entered = asyncio.Event()
        delivered: list[object] = []

        class HungBackend:
            async def stream(self, _request: StreamRequest):
                backend_entered.set()
                await asyncio.Event().wait()
                if False:
                    yield None

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        registry = StreamingJobRegistry(
            emit=emit,
            backend_chunk_timeout_s=0.01,
        )
        handle = await registry.start(
            client_id="client-a",
            prompt="timeout",
            seed=1,
            backend=HungBackend(),
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1)
        with self.assertRaisesRegex(StreamProtocolError, "chunk timed out"):
            await asyncio.wait_for(handle.task, timeout=0.2)

        self.assertTrue(registry._backend_poisoned)
        self.assertEqual(delivered[-1].error_code, "backend_timeout")

    async def test_backend_fatal_error_poisons_registry_gate(self) -> None:
        delivered: list[object] = []
        replacement_started = False

        class FatalBackend:
            async def stream(self, _request: StreamRequest):
                raise BackendFatalError("worker epoch is unsafe")
                yield DecodedChunk((b"unreachable",))

        class ReplacementBackend:
            async def stream(self, _request: StreamRequest):
                nonlocal replacement_started
                replacement_started = True
                yield DecodedChunk((PNG_1X1,), frame_media_type="image/png")

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        job_ids = iter(["fatal-worker", "blocked-replacement"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
        )
        first = await registry.start(
            client_id="client-a",
            prompt="fatal",
            seed=1,
            backend=FatalBackend(),
        )

        with self.assertRaisesRegex(StreamProtocolError, "backend failed"):
            await asyncio.wait_for(first.task, timeout=1)
        self.assertTrue(registry._backend_poisoned)
        self.assertEqual(delivered[-1].error_code, "backend_fatal")
        with self.assertRaisesRegex(StreamProtocolError, "backend is unavailable"):
            await registry.start(
                client_id="client-b",
                prompt="blocked",
                seed=2,
                backend=ReplacementBackend(),
            )
        self.assertFalse(replacement_started)

    async def test_initial_emitter_error_is_sanitized(self) -> None:
        async def emit(_client_id: str, _event: object) -> None:
            raise RuntimeError("transport-private-token")

        with self.assertRaisesRegex(StreamProtocolError, "emitter failed") as context:
            await run_stream_job(
                request(),
                FakeStreamingBackend(frame_counts=FRAME_COUNTS),
                emit=emit,
                clock_ns=SequenceClock(START_NS),
            )

        self.assertNotIn("private-token", str(context.exception))

    async def test_cancel_during_started_emit_attempts_terminal_cancellation(self) -> None:
        delivered: list[str] = []
        started_emit_entered = asyncio.Event()

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event.kind)
            if event.kind == "job_started":
                started_emit_entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(
            run_stream_job(
                request(),
                FakeStreamingBackend(frame_counts=FRAME_COUNTS),
                emit=emit,
                clock_ns=SequenceClock(START_NS),
                emit_timeout_s=1,
            )
        )
        await asyncio.wait_for(started_emit_entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)

        self.assertEqual(delivered, ["job_started", "job_cancelled"])

    async def test_chunk_emitter_protocol_error_is_sanitized(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)
            if event.kind == "chunk_ready":
                raise StreamProtocolError("transport-private-token")

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="emitter-protocol-error",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "emitter failed") as context:
            await run_stream_job(
                one_chunk_request,
                FakeStreamingBackend(frame_counts=[1]),
                emit=emit,
                clock_ns=SequenceClock(START_NS, START_NS + 1),
            )

        self.assertNotIn("private-token", str(context.exception))

    async def test_emitter_internal_cancel_is_sanitized_not_job_cancel(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)
            if event.kind == "chunk_ready":
                raise asyncio.CancelledError

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="emitter-internal-cancel",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "emitter failed"):
            await run_stream_job(
                one_chunk_request,
                FakeStreamingBackend(frame_counts=[1]),
                emit=emit,
                clock_ns=SequenceClock(START_NS, START_NS + 1),
            )

        self.assertNotIn("job_cancelled", [event.kind for event in delivered])

    async def test_emitter_deadline_is_hard_when_child_suppresses_cancel(self) -> None:
        release_rogue_emit = asyncio.Event()
        calls = 0

        async def emit(_client_id: str, _event: object) -> None:
            nonlocal calls
            calls += 1
            if calls != 1:
                return
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_rogue_emit.wait()

        task = asyncio.create_task(
            run_stream_job(
                request(),
                FakeStreamingBackend(frame_counts=FRAME_COUNTS),
                emit=emit,
                emit_timeout_s=0.01,
            )
        )
        await asyncio.sleep(0.04)
        finished_by_deadline = task.done()
        release_rogue_emit.set()
        if not task.done():
            await asyncio.wait_for(task, timeout=1)

        self.assertTrue(finished_by_deadline)
        with self.assertRaisesRegex(StreamProtocolError, "emitter failed"):
            await task

    async def test_chunk_payload_limit_fails_before_emission(self) -> None:
        delivered: list[object] = []

        class OversizedBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk(frame_payloads=(b"12345",))

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="oversized-payload",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "chunk 0.*byte limit"):
            await run_stream_job(
                one_chunk_request,
                OversizedBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS),
                max_chunk_bytes=4,
            )

        self.assertNotIn("chunk_ready", [event.kind for event in delivered])
        self.assertEqual(delivered[-1].kind, "job_failed")

    async def test_registry_replacement_is_client_scoped_and_cancellable(self) -> None:
        events: list[tuple[str, str, str]] = []
        release_b = asyncio.Event()
        started_a = asyncio.Event()
        started_b = asyncio.Event()

        class GatedBackend:
            def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
                self.started = started
                self.release = release
                self.cancelled = asyncio.Event()

            async def stream(self, _request: StreamRequest):
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS,
                ).stream(_request):
                    yield chunk

        release_a = asyncio.Event()
        old_a_backend = GatedBackend(started_a, release_a)
        client_b_backend = GatedBackend(started_b, release_b)

        async def emit(client_id: str, event: object) -> None:
            events.append((client_id, event.job_id, event.kind))

        job_ids = iter(["job-a-old", "job-b", "job-a-new"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
            queue_capacity=2,
        )
        old_a = await registry.start(
            client_id="client-a",
            prompt="old",
            seed=1,
            backend=old_a_backend,
            latent_frames=21,
        )
        job_b = await registry.start(
            client_id="client-b",
            prompt="other client",
            seed=2,
            backend=client_b_backend,
            latent_frames=21,
        )
        await asyncio.wait_for(started_a.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertFalse(started_b.is_set())

        new_a = await registry.start(
            client_id="client-a",
            prompt="replacement",
            seed=3,
            backend=FakeStreamingBackend(
                frame_counts=FRAME_COUNTS,
            ),
            latent_frames=21,
        )

        await asyncio.wait_for(old_a_backend.cancelled.wait(), timeout=1)
        self.assertFalse(job_b.task.done())
        await asyncio.wait_for(started_b.wait(), timeout=1)
        release_b.set()
        await asyncio.gather(new_a.task, job_b.task)
        with self.assertRaises(asyncio.CancelledError):
            await old_a.task

        self.assertIn(("client-a", "job-a-new", "job_completed"), events)
        self.assertIn(("client-b", "job-b", "job_completed"), events)
        new_start_index = events.index(("client-a", "job-a-new", "job_started"))
        self.assertFalse(
            any(
                client == "client-a" and job_id == "job-a-old"
                for client, job_id, _kind in events[new_start_index + 1 :]
            )
        )
        self.assertFalse(
            any(
                client == "client-b" and job_id.startswith("job-a")
                for client, job_id, _ in events
            )
        )

    async def test_registry_serializes_non_reentrant_backends_across_clients(self) -> None:
        release_a = asyncio.Event()
        started_a = asyncio.Event()
        started_b = asyncio.Event()

        class SerializedProbeBackend:
            def __init__(self, started: asyncio.Event, release: asyncio.Event | None) -> None:
                self.started = started
                self.release = release

            async def stream(self, request: StreamRequest):
                self.started.set()
                if self.release is not None:
                    await self.release.wait()
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS
                ).stream(request):
                    yield chunk

        async def emit(_client_id: str, _event: object) -> None:
            return None

        job_ids = iter(["job-a", "job-b"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
        )
        job_a = await registry.start(
            client_id="client-a",
            prompt="a",
            seed=1,
            backend=SerializedProbeBackend(started_a, release_a),
        )
        await asyncio.wait_for(started_a.wait(), timeout=1)
        job_b = await registry.start(
            client_id="client-b",
            prompt="b",
            seed=2,
            backend=SerializedProbeBackend(started_b, None),
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertFalse(started_b.is_set())
        release_a.set()
        await asyncio.wait_for(job_a.task, timeout=1)
        await asyncio.wait_for(started_b.wait(), timeout=1)
        await asyncio.wait_for(job_b.task, timeout=1)

    async def test_emitter_timeout_releases_global_backend_for_next_client(self) -> None:
        blocked_chunk_emit = asyncio.Event()
        started_b = asyncio.Event()

        class ObservedBackend:
            def __init__(self, started: asyncio.Event | None = None) -> None:
                self.started = started

            async def stream(self, request: StreamRequest):
                if self.started is not None:
                    self.started.set()
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS
                ).stream(request):
                    yield chunk

        async def emit(client_id: str, event: object) -> None:
            if client_id == "client-a" and event.kind == "chunk_ready":
                blocked_chunk_emit.set()
                await asyncio.Event().wait()

        job_ids = iter(["timeout-a", "after-timeout-b"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
            emit_timeout_s=0.02,
            backend_close_timeout_s=0.02,
        )
        job_a = await registry.start(
            client_id="client-a",
            prompt="a",
            seed=1,
            backend=ObservedBackend(),
        )
        await asyncio.wait_for(blocked_chunk_emit.wait(), timeout=1)
        job_b = await registry.start(
            client_id="client-b",
            prompt="b",
            seed=2,
            backend=ObservedBackend(started_b),
        )

        with self.assertRaisesRegex(StreamProtocolError, "emitter failed"):
            await asyncio.wait_for(job_a.task, timeout=0.2)
        await asyncio.wait_for(started_b.wait(), timeout=0.2)
        await asyncio.wait_for(job_b.task, timeout=1)

    async def test_backend_close_timeout_bounds_cancellation(self) -> None:
        backend_entered = asyncio.Event()
        close_entered = asyncio.Event()

        class HungIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                backend_entered.set()
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                close_entered.set()
                await asyncio.Event().wait()

        class HungCloseBackend:
            def stream(self, _request: StreamRequest):
                return HungIterator()

        async def emit(_client_id: str, _event: object) -> None:
            return None

        task = asyncio.create_task(
            run_stream_job(
                request(),
                HungCloseBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS),
                backend_close_timeout_s=0.02,
            )
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        self.assertTrue(close_entered.is_set())

    async def test_backend_close_deadline_is_hard_when_close_suppresses_cancel(self) -> None:
        backend_entered = asyncio.Event()
        close_entered = asyncio.Event()
        release_rogue_close = asyncio.Event()

        class CancellationSuppressingIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                backend_entered.set()
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                close_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release_rogue_close.wait()

        class Backend:
            def stream(self, _request: StreamRequest):
                return CancellationSuppressingIterator()

        async def emit(_client_id: str, _event: object) -> None:
            return None

        task = asyncio.create_task(
            run_stream_job(
                request(),
                Backend(),
                emit=emit,
                backend_close_timeout_s=0.01,
            )
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        await asyncio.sleep(0.04)
        finished_by_deadline = task.done()
        release_rogue_close.set()
        if not task.done():
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)

        self.assertTrue(finished_by_deadline)
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_cleanup_timeout_poisons_registry_backend_gate(self) -> None:
        backend_entered = asyncio.Event()
        close_entered = asyncio.Event()
        release_rogue_close = asyncio.Event()
        replacement_backend_started = asyncio.Event()
        delivered: list[object] = []

        class PoisoningIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                backend_entered.set()
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                close_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release_rogue_close.wait()

        class PoisoningBackend:
            def stream(self, _request: StreamRequest):
                return PoisoningIterator()

        class ReplacementBackend:
            async def stream(self, request: StreamRequest):
                replacement_backend_started.set()
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS
                ).stream(request):
                    yield chunk

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        job_ids = iter(["poisoning-job", "blocked-replacement"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
            backend_close_timeout_s=0.01,
        )
        first = await registry.start(
            client_id="client-a",
            prompt="poison",
            seed=1,
            backend=PoisoningBackend(),
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1)
        self.assertTrue(await registry.cancel("client-a", job_id=first.job_id))
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(first.task, timeout=0.2)

        with self.assertRaisesRegex(StreamProtocolError, "backend is unavailable"):
            await registry.start(
                client_id="client-b",
                prompt="must fail closed",
                seed=2,
                backend=ReplacementBackend(),
            )
        self.assertFalse(replacement_backend_started.is_set())
        release_rogue_close.set()

    async def test_cancel_suppressing_backend_next_is_detached_and_poisons_gate(self) -> None:
        backend_entered = asyncio.Event()
        release_rogue_next = asyncio.Event()
        replacement_started = asyncio.Event()

        class SuppressingIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                backend_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release_rogue_next.wait()

        class SuppressingBackend:
            def stream(self, _request: StreamRequest):
                return SuppressingIterator()

        class ReplacementBackend:
            async def stream(self, request: StreamRequest):
                replacement_started.set()
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS
                ).stream(request):
                    yield chunk

        async def emit(_client_id: str, _event: object) -> None:
            return None

        job_ids = iter(["suppresses-cancel", "must-not-run"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
            backend_close_timeout_s=0.01,
        )
        first = await registry.start(
            client_id="client-a",
            prompt="poison",
            seed=1,
            backend=SuppressingBackend(),
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1)
        self.assertTrue(await registry.cancel("client-a", job_id=first.job_id))
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(first.task, timeout=0.2)
        self.assertTrue(registry._backend_poisoned)

        with self.assertRaisesRegex(StreamProtocolError, "backend is unavailable"):
            await registry.start(
                client_id="client-b",
                prompt="blocked",
                seed=2,
                backend=ReplacementBackend(),
            )
        self.assertFalse(replacement_started.is_set())
        release_rogue_next.set()

    async def test_cleanup_descriptor_failure_poisons_registry_backend_gate(self) -> None:
        replacement_started = asyncio.Event()

        class BadCloseIterator:
            def __init__(self) -> None:
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index == 0:
                    self.index += 1
                    return DecodedChunk(frame_payloads=(b"frame",))
                raise StopAsyncIteration

            @property
            def aclose(self):
                raise RuntimeError("malformed cleanup descriptor")

        class BadCloseBackend:
            def stream(self, _request: StreamRequest):
                return BadCloseIterator()

        class ReplacementBackend:
            async def stream(self, request: StreamRequest):
                replacement_started.set()
                async for chunk in FakeStreamingBackend(frame_counts=[1]).stream(
                    request
                ):
                    yield chunk

        async def emit(_client_id: str, _event: object) -> None:
            return None

        job_ids = iter(["bad-close", "must-not-run"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
            max_latent_frames=1,
        )
        first = await registry.start(
            client_id="client-a",
            prompt="bad cleanup",
            seed=1,
            backend=BadCloseBackend(),
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "cleanup failed"):
            await asyncio.wait_for(first.task, timeout=1)
        self.assertTrue(registry._backend_poisoned)
        with self.assertRaisesRegex(StreamProtocolError, "backend is unavailable"):
            await registry.start(
                client_id="client-b",
                prompt="blocked",
                seed=2,
                backend=ReplacementBackend(),
                latent_frames=1,
            )
        self.assertFalse(replacement_started.is_set())

    async def test_registry_rejects_duplicate_job_id_without_replacing_active(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class GatedBackend:
            async def stream(self, request: StreamRequest):
                started.set()
                await release.wait()
                async for chunk in FakeStreamingBackend(
                    frame_counts=FRAME_COUNTS
                ).stream(request):
                    yield chunk

        async def emit(_client_id: str, _event: object) -> None:
            return None

        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: "duplicate-job-id",
        )
        original = await registry.start(
            client_id="client-a",
            prompt="original",
            seed=1,
            backend=GatedBackend(),
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        with self.assertRaisesRegex(StreamProtocolError, "job_id.*unique"):
            await registry.start(
                client_id="client-a",
                prompt="replacement",
                seed=2,
                backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
            )

        self.assertEqual(await registry.active_job_id("client-a"), original.job_id)
        self.assertFalse(original.task.done())
        release.set()
        await asyncio.wait_for(original.task, timeout=1)

    async def test_registry_rejects_over_limit_rollout_before_returning_handle(self) -> None:
        delivered: list[object] = []

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        registry = StreamingJobRegistry(
            emit=emit,
            max_latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "service limit 1"):
            await registry.start(
                client_id="client-a",
                prompt="too long",
                seed=1,
                backend=FakeStreamingBackend(frame_counts=[1, 4]),
                latent_frames=2,
            )

        self.assertEqual(delivered, [])

    async def test_registry_retires_per_client_emitter_lock(self) -> None:
        async def emit(_client_id: str, _event: object) -> None:
            return None

        registry = StreamingJobRegistry(emit=emit)
        handle = await registry.start(
            client_id="one-shot-client",
            prompt="test",
            seed=1,
            backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
        )
        await asyncio.wait_for(handle.task, timeout=1)
        await asyncio.sleep(0)

        self.assertNotIn("one-shot-client", registry._emit_locks)

    async def test_immediate_registry_cancel_does_not_leave_stale_active_job(self) -> None:
        async def emit(_client_id: str, _event: object) -> None:
            return None

        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: "cancel-before-first-poll",
        )
        handle = await registry.start(
            client_id="client-a",
            prompt="test",
            seed=1,
            backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
        )
        self.assertTrue(await registry.cancel("client-a", job_id=handle.job_id))
        await asyncio.sleep(0)

        self.assertIsNone(await registry.active_job_id("client-a"))
        with self.assertRaises(asyncio.CancelledError):
            await handle.task

    async def test_cancelled_job_id_cannot_be_reused_until_old_task_finishes(self) -> None:
        backend_started = asyncio.Event()
        cancel_emit_entered = asyncio.Event()
        release_cancel_emit = asyncio.Event()

        class BlockingBackend:
            async def stream(self, _request: StreamRequest):
                backend_started.set()
                await asyncio.Event().wait()
                if False:
                    yield None

        async def emit(_client_id: str, event: object) -> None:
            if event.kind == "job_cancelled":
                cancel_emit_entered.set()
                await release_cancel_emit.wait()

        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: "reused-id",
        )
        old = await registry.start(
            client_id="client-a",
            prompt="old",
            seed=1,
            backend=BlockingBackend(),
        )
        await asyncio.wait_for(backend_started.wait(), timeout=1)
        self.assertTrue(await registry.cancel("client-a", job_id=old.job_id))
        await asyncio.wait_for(cancel_emit_entered.wait(), timeout=1)

        with self.assertRaisesRegex(StreamProtocolError, "job_id.*unique"):
            await registry.start(
                client_id="client-a",
                prompt="too early",
                seed=2,
                backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
            )

        release_cancel_emit.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(old.task, timeout=1)
        await asyncio.sleep(0)
        replacement = await registry.start(
            client_id="client-a",
            prompt="safe reuse after completion",
            seed=3,
            backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
        )
        await asyncio.wait_for(replacement.task, timeout=1)

    async def test_same_client_emitter_calls_do_not_overlap_after_cancel(self) -> None:
        old_cancel_entered = asyncio.Event()
        release_old_cancel = asyncio.Event()
        new_start_entered = asyncio.Event()
        active_emitters = 0
        maximum_emitters = 0

        async def emit(_client_id: str, event: object) -> None:
            nonlocal active_emitters, maximum_emitters
            active_emitters += 1
            maximum_emitters = max(maximum_emitters, active_emitters)
            try:
                if event.job_id == "old" and event.kind == "job_cancelled":
                    old_cancel_entered.set()
                    await release_old_cancel.wait()
                if event.job_id == "new" and event.kind == "job_started":
                    new_start_entered.set()
            finally:
                active_emitters -= 1

        backend_started = asyncio.Event()

        class BlockingBackend:
            async def stream(self, _request: StreamRequest):
                backend_started.set()
                await asyncio.Event().wait()
                if False:
                    yield None

        job_ids = iter(["old", "new"])
        registry = StreamingJobRegistry(
            emit=emit,
            job_id_factory=lambda: next(job_ids),
        )
        old = await registry.start(
            client_id="client-a",
            prompt="old",
            seed=1,
            backend=BlockingBackend(),
        )
        await asyncio.wait_for(backend_started.wait(), timeout=1)
        self.assertTrue(await registry.cancel("client-a", job_id=old.job_id))
        await asyncio.wait_for(old_cancel_entered.wait(), timeout=1)

        new = await registry.start(
            client_id="client-a",
            prompt="new",
            seed=2,
            backend=FakeStreamingBackend(frame_counts=FRAME_COUNTS),
        )
        await asyncio.sleep(0)
        self.assertFalse(new_start_entered.is_set())
        self.assertEqual(maximum_emitters, 1)

        release_old_cancel.set()
        await asyncio.wait_for(new_start_entered.wait(), timeout=1)
        await asyncio.wait_for(new.task, timeout=1)
        with self.assertRaises(asyncio.CancelledError):
            await old.task

    async def test_wrong_chunk_shape_fails_even_when_total_frame_count_matches(self) -> None:
        delivered: list[object] = []

        class WrongShapeBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk(frame_payloads=(b"a", b"b"))
                yield DecodedChunk(frame_payloads=(b"c", b"d", b"e"))

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        short_request = StreamRequest(
            client_id="client-a",
            job_id="wrong-shape",
            prompt="test",
            seed=1,
            latent_frames=2,
        )
        with self.assertRaisesRegex(StreamProtocolError, "chunk 0.*1.*2"):
            await run_stream_job(
                short_request,
                WrongShapeBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, START_NS + 1),
            )
        self.assertEqual(delivered[-1].kind, "job_failed")

    async def test_frame_media_type_is_backend_owned_and_job_invariant(self) -> None:
        delivered: list[object] = []

        class FormatChangingBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk((PNG_1X1,), frame_media_type="image/png")
                yield DecodedChunk(
                    tuple(b"opaque-frame" for _ in range(4)),
                    frame_media_type="application/octet-stream",
                )

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        short_request = StreamRequest(
            client_id="client-a",
            job_id="format-change",
            prompt="test",
            seed=1,
            latent_frames=2,
        )
        with self.assertRaisesRegex(StreamProtocolError, "media type.*change"):
            await run_stream_job(
                short_request,
                FormatChangingBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, *READY_NS[:2]),
            )

        chunks = [event for event in delivered if event.kind == "chunk_ready"]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].frame_media_type, "image/png")
        self.assertEqual(chunks[0].to_dict()["frame_media_type"], "image/png")
        self.assertEqual(delivered[-1].kind, "job_failed")

    async def test_unsafe_frame_media_type_fails_before_chunk_emission(self) -> None:
        delivered: list[object] = []

        class UnsafeFormatBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk(
                    (b"active-content",),
                    frame_media_type="image/svg+xml",
                )

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="unsafe-format",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "frame_media_type"):
            await run_stream_job(
                one_chunk_request,
                UnsafeFormatBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, READY_NS[0]),
            )
        self.assertFalse(any(event.kind == "chunk_ready" for event in delivered))

    async def test_declared_raster_media_type_must_match_payload_bytes(self) -> None:
        delivered: list[object] = []

        class MislabeledRasterBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk(
                    (b"not-a-png",),
                    frame_media_type="image/png",
                )

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="mislabeled-raster",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "does not match image/png"):
            await run_stream_job(
                one_chunk_request,
                MislabeledRasterBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, READY_NS[0]),
            )
        self.assertFalse(any(event.kind == "chunk_ready" for event in delivered))

    async def test_png_envelope_without_valid_chunks_is_rejected(self) -> None:
        delivered: list[object] = []
        invalid_png = (
            b"\x89PNG\r\n\x1a\n"
            b"garbage"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        class CorruptPngBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk((invalid_png,), frame_media_type="image/png")

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="corrupt-png",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "does not match image/png"):
            await run_stream_job(
                one_chunk_request,
                CorruptPngBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, READY_NS[0], READY_NS[0] + 1),
            )
        self.assertFalse(any(event.kind == "chunk_ready" for event in delivered))

    async def test_png_with_corrupt_crc_is_rejected(self) -> None:
        delivered: list[object] = []
        corrupt_png = bytearray(PNG_1X1)
        corrupt_png[29] ^= 0x01

        class CorruptPngBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk((bytes(corrupt_png),), frame_media_type="image/png")

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        one_chunk_request = StreamRequest(
            client_id="client-a",
            job_id="corrupt-png-crc",
            prompt="test",
            seed=1,
            latent_frames=1,
        )
        with self.assertRaisesRegex(StreamProtocolError, "does not match image/png"):
            await run_stream_job(
                one_chunk_request,
                CorruptPngBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, READY_NS[0], READY_NS[0] + 1),
            )
        self.assertFalse(any(event.kind == "chunk_ready" for event in delivered))

    async def test_png_with_invalid_idat_or_scanline_is_rejected(self) -> None:
        cases = {
            "invalid-zlib": _rgb_png_with_idat(b"not-zlib"),
            "invalid-filter": _rgb_png_with_idat(
                zlib.compress(bytes((5, 1, 2, 3)))
            ),
            "short-scanline": _rgb_png_with_idat(zlib.compress(bytes((0, 1, 2)))),
        }
        for label, invalid_png in cases.items():
            with self.subTest(case=label):
                delivered: list[object] = []

                class CorruptPngBackend:
                    async def stream(self, _request: StreamRequest):
                        yield DecodedChunk(
                            (invalid_png,), frame_media_type="image/png"
                        )

                async def emit(_client_id: str, event: object) -> None:
                    delivered.append(event)

                one_chunk_request = StreamRequest(
                    client_id="client-a",
                    job_id=f"corrupt-png-{label}",
                    prompt="test",
                    seed=1,
                    latent_frames=1,
                )
                with self.assertRaisesRegex(
                    StreamProtocolError, "does not match image/png"
                ):
                    await run_stream_job(
                        one_chunk_request,
                        CorruptPngBackend(),
                        emit=emit,
                        clock_ns=SequenceClock(
                            START_NS, READY_NS[0], READY_NS[0] + 1
                        ),
                    )
                self.assertFalse(
                    any(event.kind == "chunk_ready" for event in delivered)
                )

    async def test_png_with_malformed_ancillary_chunk_is_rejected(self) -> None:
        header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        invalid_png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"sRGB", b"")
            + _png_chunk(b"IDAT", zlib.compress(bytes((0, 1, 2, 3))))
            + _png_chunk(b"IEND", b"")
        )
        delivered: list[object] = []

        class MalformedMetadataBackend:
            async def stream(self, _request: StreamRequest):
                yield DecodedChunk((invalid_png,), frame_media_type="image/png")

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        with self.assertRaisesRegex(StreamProtocolError, "does not match image/png"):
            await run_stream_job(
                StreamRequest("client-a", "malformed-metadata", "test", 1, 1),
                MalformedMetadataBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS, READY_NS[0], READY_NS[0] + 1),
            )
        self.assertFalse(any(event.kind == "chunk_ready" for event in delivered))

    async def test_backend_error_is_not_echoed_to_the_client(self) -> None:
        delivered: list[object] = []

        class FailingBackend:
            async def stream(self, _request: StreamRequest):
                if False:
                    yield None
                raise RuntimeError("provider echoed private-token")

        async def emit(_client_id: str, event: object) -> None:
            delivered.append(event)

        with self.assertRaisesRegex(StreamProtocolError, "backend failed") as context:
            await run_stream_job(
                request(),
                FailingBackend(),
                emit=emit,
                clock_ns=SequenceClock(START_NS),
            )

        self.assertNotIn("private-token", str(context.exception))
        self.assertEqual(delivered[-1].to_dict()["error_code"], "backend_failure")
        self.assertNotIn("private-token", str(delivered[-1].to_dict()))

    async def test_fake_demo_smoke_cli_exercises_the_complete_stream(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "streaming-demo-smoke")],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(events), 23)
        self.assertEqual(events[0]["kind"], "job_started")
        self.assertEqual(events[-1]["kind"], "job_completed")
        self.assertEqual(events[-1]["summary"]["frame_count"], 81)
        self.assertEqual(
            [event["frame_count"] for event in events if event["kind"] == "chunk_ready"],
            FRAME_COUNTS,
        )
        self.assertTrue(all(event["client_id"] == "fake-client" for event in events))
        self.assertNotIn("frame_ready_ns", completed.stdout)
        self.assertNotIn("rgb_ready_ns", completed.stdout)


if __name__ == "__main__":
    unittest.main()
