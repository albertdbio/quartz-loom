from __future__ import annotations

import asyncio
import base64
import json
import unittest
from collections.abc import AsyncIterator
from typing import Any

from bench.streaming_service import (
    DecodedChunk,
    FakeStreamingBackend,
    StreamEvent,
    StreamRequest,
)
from bench.streaming_transport import (
    NDJSONStreamingServer,
    _ClientSession,
    _CommandError,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def send_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(
        json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    await writer.drain()


async def receive_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    if not line:
        raise AssertionError("connection closed before the expected message")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise AssertionError("transport emitted a non-object JSON value")
    return value


async def receive_until(
    reader: asyncio.StreamReader,
    predicate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    while True:
        message = await receive_message(reader)
        seen.append(message)
        if predicate(message):
            return message, seen


async def wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


class BlockingBackend:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        try:
            self.entered.set()
            await asyncio.Future()
            yield DecodedChunk((b"unreachable",))
        finally:
            self.cancelled.set()


class OversizedPayloadBackend:
    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        yield DecodedChunk((b"12345",))


class PngPayloadBackend:
    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        yield DecodedChunk((PNG_1X1,), frame_media_type="image/png")


class GateWriter:
    def __init__(self) -> None:
        self.wrote = asyncio.Event()
        self.allow_drain = asyncio.Event()
        self.payloads: list[bytes] = []
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.payloads.append(payload)
        self.wrote.set()

    async def drain(self) -> None:
        await self.allow_drain.wait()

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class StreamingTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.servers: list[NDJSONStreamingServer] = []
        self.writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self.writers:
            if not writer.is_closing():
                writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionError):
                pass
        for server in reversed(self.servers):
            await server.close()

    async def start_server(self, backend: Any, **kwargs: Any) -> NDJSONStreamingServer:
        server = NDJSONStreamingServer(backend=backend, **kwargs)
        await server.start()
        self.servers.append(server)
        return server

    async def connect(
        self,
        server: NDJSONStreamingServer,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]]:
        host, port = server.address
        reader, writer = await asyncio.open_connection(host, port)
        self.writers.append(writer)
        hello = await receive_message(reader)
        self.assertEqual(hello["type"], "connected")
        self.assertEqual(hello["protocol"], "realtime-video.ndjson.v1")
        return reader, writer, hello

    async def test_real_socket_stream_is_job_fenced_and_payload_is_opaque(self) -> None:
        job_ids = iter(["job-one"])
        server = await self.start_server(
            FakeStreamingBackend(frame_counts=[1, 4, 4]),
            job_id_factory=lambda: next(job_ids),
        )
        reader, writer, _hello = await self.connect(server)

        await send_message(
            writer,
            {"type": "start", "prompt": "fox", "seed": 7, "latent_frames": 3},
        )
        messages: list[dict[str, Any]] = []
        while True:
            message = await receive_message(reader)
            messages.append(message)
            if message.get("type") == "stream_event" and message.get("kind") == "chunk_ready":
                await send_message(
                    writer,
                    {
                        "type": "presented",
                        "job_id": "job-one",
                        "chunk_index": message["chunk_index"],
                        "client_presented_ns": 123 + message["chunk_index"],
                    },
                )
            if message.get("type") == "stream_event" and message.get("kind") == "job_completed":
                break

        job_messages = [message for message in messages if "job_id" in message]
        self.assertTrue(job_messages)
        self.assertEqual({message["job_id"] for message in job_messages}, {"job-one"})
        accepted_index = next(
            index for index, message in enumerate(messages) if message["type"] == "start_accepted"
        )
        first_event_index = next(
            index for index, message in enumerate(messages) if message["type"] == "stream_event"
        )
        self.assertLess(accepted_index, first_event_index)
        chunks = [
            message
            for message in messages
            if message.get("type") == "stream_event"
            and message.get("kind") == "chunk_ready"
        ]
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [0, 1, 2])
        self.assertEqual([chunk["frame_count"] for chunk in chunks], [1, 4, 4])
        for chunk in chunks:
            self.assertEqual(chunk["payload_media_type"], "application/octet-stream")
            self.assertEqual(chunk["payload_encoding"], "base64")
            self.assertFalse(chunk["renderable"])
        self.assertEqual(
            base64.b64decode(chunks[0]["frame_payloads_base64"][0]),
            b"fake-frame-0-0",
        )
        recorded = next(message for message in messages if message["type"] == "presentation_recorded")
        self.assertEqual(recorded["type"], "presentation_recorded")
        self.assertEqual(recorded["job_id"], "job-one")
        self.assertEqual(recorded["client_clock_domain"], "client_monotonic")
        self.assertEqual(recorded["server_clock_domain"], "server_monotonic")
        self.assertEqual(recorded["timing_semantics"], "client_reported_presentation")
        self.assertTrue(recorded["not_server_emit_completion"])

    async def test_transport_propagates_backend_owned_raster_media_type(self) -> None:
        server = await self.start_server(
            PngPayloadBackend(),
            job_id_factory=lambda: "job-png",
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "png", "seed": 8, "latent_frames": 1},
        )
        chunk, _ = await receive_until(
            reader,
            lambda item: item.get("kind") == "chunk_ready",
        )

        self.assertEqual(chunk["frame_media_type"], "image/png")
        self.assertEqual(chunk["payload_media_type"], "image/png")
        self.assertTrue(chunk["renderable"])

    async def test_two_chunk_presentation_window_applies_client_backpressure(self) -> None:
        server = await self.start_server(
            FakeStreamingBackend(frame_counts=[1, 4, 4]),
            job_id_factory=lambda: "job-window",
            presentation_window_chunks=2,
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "window", "seed": 9, "latent_frames": 3},
        )
        chunks: list[dict[str, Any]] = []
        while len(chunks) < 2:
            message = await receive_message(reader)
            if message.get("kind") == "chunk_ready":
                chunks.append(message)

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.readline(), timeout=0.05)

        await send_message(
            writer,
            {
                "type": "presented",
                "job_id": "job-window",
                "chunk_index": chunks[0]["chunk_index"],
                "client_presented_ns": 100,
            },
        )
        third, _seen = await receive_until(
            reader,
            lambda item: item.get("kind") == "chunk_ready" and item.get("chunk_index") == 2,
        )
        self.assertEqual(third["job_id"], "job-window")

    async def test_cancel_is_scoped_to_connection_and_requires_job_id(self) -> None:
        backend = BlockingBackend()
        job_ids = iter(["job-owner"])
        server = await self.start_server(
            backend,
            job_id_factory=lambda: next(job_ids),
            disconnect_timeout_s=0.2,
        )
        owner_reader, owner_writer, _ = await self.connect(server)
        other_reader, other_writer, _ = await self.connect(server)
        await send_message(
            owner_writer,
            {"type": "start", "prompt": "hold", "seed": 1, "latent_frames": 1},
        )
        await receive_until(
            owner_reader,
            lambda item: item.get("type") == "start_accepted",
        )
        await asyncio.wait_for(backend.entered.wait(), timeout=1.0)

        await send_message(
            other_writer,
            {
                "type": "presented",
                "job_id": "job-owner",
                "chunk_index": 0,
                "client_presented_ns": 1,
            },
        )
        cross_client_ack = await receive_message(other_reader)
        self.assertEqual(cross_client_ack["code"], "stale_job")
        self.assertEqual(cross_client_ack["job_id"], "job-owner")

        await send_message(other_writer, {"type": "cancel", "job_id": "job-owner"})
        other_result = await receive_message(other_reader)
        self.assertEqual(other_result["type"], "cancel_result")
        self.assertEqual(other_result["job_id"], "job-owner")
        self.assertFalse(other_result["cancelled"])

        await send_message(owner_writer, {"type": "cancel", "job_id": "job-owner"})
        owner_result, seen = await receive_until(
            owner_reader,
            lambda item: item.get("type") == "cancel_result",
        )
        accepted_index = next(
            index for index, item in enumerate(seen) if item.get("type") == "cancel_accepted"
        )
        cancelled_index = next(
            (
                index
                for index, item in enumerate(seen)
                if item.get("kind") == "job_cancelled"
            ),
            len(seen),
        )
        self.assertLess(accepted_index, cancelled_index)
        self.assertEqual(owner_result["job_id"], "job-owner")
        self.assertTrue(owner_result["cancelled"])
        await asyncio.wait_for(backend.cancelled.wait(), timeout=1.0)

    async def test_ack_is_eligible_after_write_before_socket_drain_completes(self) -> None:
        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
        )
        writer = GateWriter()
        session = _ClientSession(
            server=server,
            client_id="client-gated",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-gated"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        emission = asyncio.create_task(
            session.emit_event(
                StreamEvent(
                    kind="chunk_ready",
                    job_id="job-gated",
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(b"opaque",),
                    frame_media_type="application/octet-stream",
                    ready_ns=10,
                    queue_depth=0,
                )
            )
        )
        await asyncio.wait_for(writer.wrote.wait(), timeout=1.0)
        acknowledgement = asyncio.create_task(
            session.handle_raw_message(
                b'{"type":"presented","job_id":"job-gated",'
                b'"chunk_index":0,"client_presented_ns":12}'
            )
        )
        await wait_until(lambda: 0 in session._acknowledged_chunks)
        self.assertFalse(acknowledgement.done())

        writer.allow_drain.set()
        await asyncio.wait_for(emission, timeout=1.0)
        await asyncio.wait_for(acknowledgement, timeout=1.0)

    async def test_failed_chunk_drain_rolls_back_ack_eligibility(self) -> None:
        class FailingDrainWriter(GateWriter):
            async def drain(self) -> None:
                raise ConnectionError("synthetic drain failure")

        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
        )
        writer = FailingDrainWriter()
        session = _ClientSession(
            server=server,
            client_id="client-failed-drain",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-failed-drain"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()

        with self.assertRaisesRegex(ConnectionError, "synthetic drain failure"):
            await session.emit_event(
                StreamEvent(
                    kind="chunk_ready",
                    job_id="job-failed-drain",
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(b"opaque",),
                    frame_media_type="application/octet-stream",
                    ready_ns=10,
                    queue_depth=0,
                )
            )
        self.assertNotIn(0, session._sent_chunks)
        self.assertNotIn(0, session._outstanding_chunks)

    async def test_stale_job_retirement_does_not_disconnect_replacement(self) -> None:
        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-replacement",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-a"
        disconnected = False

        async def stale_terminal_send(
            _message: dict[str, Any],
            *,
            expected_job_id: str | None = None,
        ) -> bool:
            self.assertEqual(expected_job_id, "job-a")
            async with session._state:
                session._current_job_id = "job-b"
            return False

        async def record_disconnect() -> None:
            nonlocal disconnected
            disconnected = True

        session.send_control_message = stale_terminal_send  # type: ignore[method-assign]
        session.disconnect = record_disconnect  # type: ignore[method-assign]
        completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        completed.set_exception(RuntimeError("backend failed"))

        await session._retire_failed_job("job-a", completed)

        self.assertFalse(disconnected)
        self.assertEqual(session._current_job_id, "job-b")

    async def test_server_clock_failure_does_not_release_presentation_credit(self) -> None:
        def broken_clock() -> int:
            raise RuntimeError("private clock failure")

        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
            clock_ns=broken_clock,
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-clock",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-clock"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        await session.emit_event(
            StreamEvent(
                kind="chunk_ready",
                job_id="job-clock",
                chunk_index=0,
                first_frame_index=0,
                frame_payloads=(b"opaque",),
                frame_media_type="application/octet-stream",
                ready_ns=10,
                queue_depth=0,
            )
        )

        with self.assertRaisesRegex(_CommandError, "server_clock_failed"):
            await session.handle_raw_message(
                b'{"type":"presented","job_id":"job-clock",'
                b'"chunk_index":0,"client_presented_ns":12}'
            )
        self.assertNotIn(0, session._acknowledged_chunks)
        self.assertIn(0, session._outstanding_chunks)

    async def test_server_receive_clock_cannot_precede_chunk_readiness(self) -> None:
        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
            clock_ns=lambda: 9,
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-clock-before-ready",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-clock-before-ready"
        session._sent_chunks[0] = 10
        session._outstanding_chunks.add(0)

        with self.assertRaisesRegex(_CommandError, "server_clock_failed"):
            await session.handle_raw_message(
                b'{"type":"presented","job_id":"job-clock-before-ready",'
                b'"chunk_index":0,"client_presented_ns":12}'
            )
        self.assertNotIn(0, session._acknowledged_chunks)
        self.assertIn(0, session._outstanding_chunks)

    async def test_server_receive_clock_must_be_strictly_monotonic(self) -> None:
        readings = iter([20, 20])
        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
            clock_ns=lambda: next(readings),
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-clock-monotonic",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-clock-monotonic"
        session._sent_chunks.update({0: 10, 1: 11})
        session._outstanding_chunks.update({0, 1})

        await session.handle_raw_message(
            b'{"type":"presented","job_id":"job-clock-monotonic",'
            b'"chunk_index":0,"client_presented_ns":12}'
        )
        with self.assertRaisesRegex(_CommandError, "server_clock_failed"):
            await session.handle_raw_message(
                b'{"type":"presented","job_id":"job-clock-monotonic",'
                b'"chunk_index":1,"client_presented_ns":13}'
            )
        self.assertIn(0, session._acknowledged_chunks)
        self.assertNotIn(1, session._acknowledged_chunks)
        self.assertIn(1, session._outstanding_chunks)

    def test_unauthenticated_transport_refuses_non_loopback_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            NDJSONStreamingServer(
                backend=FakeStreamingBackend(frame_counts=[1]),
                host="0.0.0.0",
            )

    async def test_presentation_ack_rejects_unsent_and_stale_jobs(self) -> None:
        backend = BlockingBackend()
        server = await self.start_server(
            backend,
            job_id_factory=lambda: "job-current",
            disconnect_timeout_s=0.2,
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "hold", "seed": 2, "latent_frames": 1},
        )
        await receive_until(reader, lambda item: item.get("type") == "start_accepted")

        await send_message(
            writer,
            {
                "type": "presented",
                "job_id": "job-current",
                "chunk_index": 0,
                "client_presented_ns": 10,
            },
        )
        unsent, _ = await receive_until(
            reader,
            lambda item: item.get("type") == "command_error"
            and item.get("code") == "chunk_not_sent",
        )
        self.assertEqual(unsent["type"], "command_error")
        self.assertEqual(unsent["code"], "chunk_not_sent")
        self.assertEqual(unsent["job_id"], "job-current")

        await send_message(
            writer,
            {
                "type": "presented",
                "job_id": "job-stale",
                "chunk_index": 0,
                "client_presented_ns": 11,
            },
        )
        stale, _ = await receive_until(
            reader,
            lambda item: item.get("type") == "command_error"
            and item.get("code") == "stale_job",
        )
        self.assertEqual(stale["code"], "stale_job")
        self.assertEqual(stale["job_id"], "job-stale")

    async def test_disconnect_cancels_the_connection_job(self) -> None:
        backend = BlockingBackend()
        server = await self.start_server(
            backend,
            job_id_factory=lambda: "job-disconnect",
            disconnect_timeout_s=0.2,
        )
        reader, writer, _hello = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "hold", "seed": 3, "latent_frames": 1},
        )
        await receive_until(
            reader,
            lambda item: item.get("type") == "stream_event"
            and item.get("kind") == "job_started",
        )
        await asyncio.wait_for(backend.entered.wait(), timeout=1.0)

        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(backend.cancelled.wait(), timeout=1.0)

    async def test_prompt_and_wire_input_are_bounded(self) -> None:
        server = await self.start_server(
            FakeStreamingBackend(frame_counts=[1]),
            max_prompt_bytes=8,
            max_input_bytes=128,
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "012345678", "seed": 1, "latent_frames": 1},
        )
        prompt_error = await receive_message(reader)
        self.assertEqual(prompt_error, {"type": "command_error", "code": "prompt_too_large"})

        await send_message(
            writer,
            {
                "type": "start",
                "client_id": "caller-chosen",
                "prompt": "ok",
                "seed": 1,
                "latent_frames": 1,
            },
        )
        identity_error = await receive_message(reader)
        self.assertEqual(identity_error, {"type": "command_error", "code": "invalid_command"})

        writer.write(b"{" + b"x" * 256 + b"}\n")
        await writer.drain()
        size_error = await receive_message(reader)
        self.assertEqual(size_error, {"type": "command_error", "code": "message_too_large"})
        self.assertEqual(await asyncio.wait_for(reader.read(), timeout=1.0), b"")

    async def test_lone_surrogates_and_recursive_json_are_stable_command_errors(self) -> None:
        server = await self.start_server(
            BlockingBackend(),
            job_id_factory=lambda: "job-json-safety",
        )
        reader, writer, _ = await self.connect(server)

        writer.write(
            json.dumps(
                {"type": "start", "prompt": "\ud800", "seed": 1},
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        self.assertEqual(
            await receive_message(reader),
            {"type": "command_error", "code": "invalid_prompt"},
        )

        writer.write(
            b'{"type":"cancel","job_id":"\\ud800"}\n'
        )
        await writer.drain()
        self.assertEqual(
            await receive_message(reader),
            {"type": "command_error", "code": "invalid_job_id"},
        )

        writer.write(b"[" * 1100 + b"0" + b"]" * 1100 + b"\n")
        await writer.drain()
        self.assertEqual(
            await receive_message(reader),
            {"type": "command_error", "code": "invalid_json"},
        )

        await send_message(writer, {"type": "cancel", "job_id": "not-current"})
        self.assertEqual(
            await receive_message(reader),
            {
                "type": "cancel_result",
                "job_id": "not-current",
                "cancelled": False,
            },
        )

    async def test_unacknowledged_window_timeout_emits_terminal_and_retires_job(self) -> None:
        server = await self.start_server(
            FakeStreamingBackend(frame_counts=[1, 4]),
            job_id_factory=lambda: "job-backpressure-timeout",
            presentation_window_chunks=1,
            emit_timeout_s=0.05,
            control_send_timeout_s=0.2,
        )
        reader, writer, hello = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "slow client", "seed": 2, "latent_frames": 2},
        )
        await receive_until(reader, lambda item: item.get("kind") == "chunk_ready")
        terminal, _ = await receive_until(
            reader,
            lambda item: item.get("kind") == "job_failed",
        )
        self.assertEqual(terminal["job_id"], "job-backpressure-timeout")
        self.assertEqual(terminal["error_code"], "client_backpressure_timeout")

        session = server._clients[hello["client_id"]]
        await wait_until(lambda: session._current_job_id is None)
        self.assertFalse(session._sent_chunks)
        self.assertFalse(session._outstanding_chunks)

    async def test_cancel_begins_before_blocked_socket_drain_and_orders_terminal(self) -> None:
        backend = BlockingBackend()
        server = NDJSONStreamingServer(
            backend=backend,
            job_id_factory=lambda: "job-prompt-cancel",
            control_send_timeout_s=0.5,
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-prompt-cancel",
            writer=writer,  # type: ignore[arg-type]
        )
        server._clients[session.client_id] = session
        session._current_job_id = "job-prompt-cancel"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        handle = await server.registry.start(
            client_id=session.client_id,
            prompt="hold",
            seed=3,
            backend=backend,
            latent_frames=1,
        )
        await asyncio.wait_for(backend.entered.wait(), timeout=1.0)

        writer.allow_drain.clear()
        writer.wrote.clear()
        blocked_send = asyncio.create_task(session.send_message({"type": "blocked"}))
        await asyncio.wait_for(writer.wrote.wait(), timeout=1.0)
        cancel = asyncio.create_task(
            session.handle_raw_message(
                b'{"type":"cancel","job_id":"job-prompt-cancel"}'
            )
        )
        await asyncio.wait_for(backend.cancelled.wait(), timeout=0.1)
        self.assertFalse(cancel.done())

        writer.allow_drain.set()
        await asyncio.wait_for(blocked_send, timeout=1.0)
        await asyncio.wait_for(cancel, timeout=1.0)
        with self.assertRaises(asyncio.CancelledError):
            await handle.task
        messages = [json.loads(payload) for payload in writer.payloads]
        accepted_index = next(
            index for index, item in enumerate(messages)
            if item.get("type") == "cancel_accepted"
        )
        result_index = next(
            index for index, item in enumerate(messages)
            if item.get("type") == "cancel_result"
        )
        terminal_index = next(
            index for index, item in enumerate(messages)
            if item.get("kind") == "job_cancelled"
        )
        self.assertLess(accepted_index, result_index)
        self.assertLess(result_index, terminal_index)

    async def test_cancel_accepted_is_never_emitted_for_false_result(self) -> None:
        server = NDJSONStreamingServer(
            backend=FakeStreamingBackend(frame_counts=[1]),
        )
        writer = GateWriter()
        writer.allow_drain.set()
        session = _ClientSession(
            server=server,
            client_id="client-false-cancel",
            writer=writer,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-already-retired"

        await session.handle_raw_message(
            b'{"type":"cancel","job_id":"job-already-retired"}'
        )
        messages = [json.loads(payload) for payload in writer.payloads]
        self.assertFalse(any(item.get("type") == "cancel_accepted" for item in messages))
        self.assertEqual(
            messages,
            [
                {
                    "type": "cancel_result",
                    "job_id": "job-already-retired",
                    "cancelled": False,
                }
            ],
        )

    async def test_job_ids_are_unique_for_connection_lifetime(self) -> None:
        server = await self.start_server(
            FakeStreamingBackend(frame_counts=[1]),
            job_id_factory=lambda: "job-reused",
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "first", "seed": 4, "latent_frames": 1},
        )
        while True:
            message = await receive_message(reader)
            if message.get("kind") == "chunk_ready":
                await send_message(
                    writer,
                    {
                        "type": "presented",
                        "job_id": "job-reused",
                        "chunk_index": 0,
                        "client_presented_ns": 1,
                    },
                )
            if message.get("kind") == "job_completed":
                break

        await send_message(
            writer,
            {"type": "start", "prompt": "second", "seed": 5, "latent_frames": 1},
        )
        rejected, _ = await receive_until(
            reader,
            lambda item: item.get("type") == "command_error",
        )
        self.assertEqual(rejected, {"type": "command_error", "code": "start_rejected"})

    async def test_generated_job_id_is_validated_before_start_acceptance(self) -> None:
        generated = iter(["x" * 129, "\ud800", "job-valid"])
        server = await self.start_server(
            BlockingBackend(),
            job_id_factory=lambda: next(generated),
        )
        reader, writer, _ = await self.connect(server)

        for prompt in ("too long", "surrogate"):
            await send_message(
                writer,
                {"type": "start", "prompt": prompt, "seed": 6, "latent_frames": 1},
            )
            self.assertEqual(
                await receive_message(reader),
                {"type": "command_error", "code": "start_rejected"},
            )

        await send_message(
            writer,
            {"type": "start", "prompt": "valid", "seed": 7, "latent_frames": 1},
        )
        accepted, seen = await receive_until(
            reader,
            lambda item: item.get("type") == "start_accepted",
        )
        self.assertEqual(accepted["job_id"], "job-valid")
        self.assertFalse(any(item.get("job_id") in {"x" * 129, "\ud800"} for item in seen))

    async def test_connection_lifetime_job_id_ledger_is_bounded(self) -> None:
        backend = BlockingBackend()
        generated = iter(["job-ledger-one", "job-ledger-two"])
        server = await self.start_server(
            backend,
            job_id_factory=lambda: next(generated),
            max_jobs_per_connection=1,
        )
        reader, writer, hello = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "first", "seed": 8, "latent_frames": 1},
        )
        await receive_until(reader, lambda item: item.get("type") == "start_accepted")

        await send_message(
            writer,
            {"type": "start", "prompt": "second", "seed": 9, "latent_frames": 1},
        )
        rejected, _ = await receive_until(
            reader,
            lambda item: item.get("type") == "command_error",
        )
        self.assertEqual(rejected, {"type": "command_error", "code": "start_rejected"})
        self.assertEqual(
            await server.registry.active_job_id(hello["client_id"]),
            "job-ledger-one",
        )

    async def test_decoded_payload_limit_fails_before_network_delivery(self) -> None:
        server = await self.start_server(
            OversizedPayloadBackend(),
            job_id_factory=lambda: "job-too-large",
            max_chunk_bytes=4,
        )
        reader, writer, _ = await self.connect(server)
        await send_message(
            writer,
            {"type": "start", "prompt": "limit", "seed": 4, "latent_frames": 1},
        )
        failed, seen = await receive_until(
            reader,
            lambda item: item.get("type") == "stream_event"
            and item.get("kind") == "job_failed",
        )
        self.assertEqual(failed["job_id"], "job-too-large")
        self.assertEqual(failed["error_code"], "protocol_error")
        self.assertFalse(any(item.get("kind") == "chunk_ready" for item in seen))


if __name__ == "__main__":
    unittest.main()
