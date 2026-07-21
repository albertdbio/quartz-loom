from __future__ import annotations

import asyncio
import itertools
import json
import hashlib
import subprocess
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aiohttp import (
    ClientSession,
    CookieJar,
    WSCloseCode,
    WSMsgType,
    WSServerHandshakeError,
)

from bench.streaming_service import DecodedChunk, StreamEvent, StreamRequest
from bench.streaming_web_demo import TinyPngStreamingBackend
from bench.streaming_websocket import (
    BrowserStreamingServer,
    PROTOCOL_VERSION,
    _CommandError,
    _WebSocketSession,
)


class OpaqueBackend:
    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        for chunk_index, frame_count in enumerate([1] + [4] * 20):
            yield DecodedChunk(
                tuple(
                    f"opaque-{chunk_index}-{frame_index}".encode("ascii")
                    for frame_index in range(frame_count)
                )
            )


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


class IncompleteBackend:
    async def stream(self, _request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        yield DecodedChunk((b"only-one-frame",))


class RecordingPromptBackend:
    def __init__(self) -> None:
        self.requests: list[StreamRequest] = []

    async def stream(self, request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        self.requests.append(request)
        for chunk_index, frame_count in enumerate([1] + [4] * 20):
            yield DecodedChunk(
                tuple(
                    f"resolved-{chunk_index}-{frame_index}".encode("ascii")
                    for frame_index in range(frame_count)
                )
            )


class FakePromptResolution:
    def __init__(self, raw_prompt: str, effective_prompt: str, transform_id: str) -> None:
        self.raw_prompt = raw_prompt
        self.effective_prompt = effective_prompt
        self.raw_prompt_sha256 = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        self.effective_prompt_sha256 = hashlib.sha256(
            effective_prompt.encode("utf-8")
        ).hexdigest()
        self.transform_id = transform_id
        self.changed = raw_prompt != effective_prompt


class StaticPromptResolver:
    async def resolve(self, raw_prompt: str) -> FakePromptResolution:
        return FakePromptResolution(
            raw_prompt,
            f"Temporal expansion: {raw_prompt}",
            "test-temporal-v1",
        )


class FailingPromptResolver:
    async def resolve(self, _raw_prompt: str) -> FakePromptResolution:
        raise RuntimeError("provider response body must stay private")


class ReplacementPromptResolver:
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def resolve(self, raw_prompt: str) -> FakePromptResolution:
        if raw_prompt == "first prompt":
            self.first_started.set()
            try:
                await self.release_first.wait()
            except asyncio.CancelledError:
                # Model a provider client that suppresses cancellation. The session's
                # request-id fence must still prevent this stale result from escaping.
                await self.release_first.wait()
        return FakePromptResolution(
            raw_prompt,
            f"Resolved: {raw_prompt}",
            "test-replacement-v1",
        )


class GatedWebSocket:
    def __init__(self) -> None:
        self.header_sent = asyncio.Event()
        self.allow_binary = asyncio.Event()
        self.strings: list[str] = []
        self.binary: list[bytes] = []

    async def send_str(self, value: str) -> None:
        self.strings.append(value)
        self.header_sent.set()

    async def send_bytes(self, value: bytes) -> None:
        await self.allow_binary.wait()
        self.binary.append(value)

    async def close(self) -> None:
        return None


class CommitGatedWebSocket:
    def __init__(self) -> None:
        self.binary_observable = asyncio.Event()
        self.allow_binary_return = asyncio.Event()
        self.commit_observable = asyncio.Event()
        self.allow_commit_return = asyncio.Event()
        self.strings: list[str] = []
        self.binary: list[bytes] = []

    async def send_str(self, value: str) -> None:
        self.strings.append(value)
        message = json.loads(value)
        if message.get("type") == "chunk_committed":
            self.commit_observable.set()
            await self.allow_commit_return.wait()

    async def send_bytes(self, value: bytes) -> None:
        # Model aiohttp making the complete payload peer-visible before the
        # send coroutine returns to the emitter.
        self.binary.append(value)
        self.binary_observable.set()
        await self.allow_binary_return.wait()

    async def close(self) -> None:
        return None


class StalledControlWebSocket:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def send_str(self, _value: str) -> None:
        self.entered.set()
        await asyncio.Future()

    async def close(self) -> None:
        return None


class SessionHarness:
    presentation_window_chunks = 2
    max_control_bytes = 16 * 1024
    control_send_timeout_s = 0.02
    delivery_id_factory = staticmethod(lambda: "gated-delivery")
    clock_ns = staticmethod(lambda: 500)


class RecordingWebSocket:
    def __init__(self) -> None:
        self.strings: list[str] = []
        self.close_calls = 0

    async def send_str(self, value: str) -> None:
        self.strings.append(value)

    async def send_bytes(self, _value: bytes) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1


async def receive_json(ws: Any, *, timeout: float = 2.0) -> dict[str, Any]:
    message = await ws.receive(timeout=timeout)
    if message.type != WSMsgType.TEXT:
        raise AssertionError(f"expected text message, got {message.type!r}")
    value = json.loads(message.data)
    if not isinstance(value, dict):
        raise AssertionError("server emitted a non-object JSON value")
    return value


async def receive_chunk(
    ws: Any,
    *,
    seen_json: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[bytes], dict[str, Any]]:
    while True:
        header = await receive_json(ws)
        if seen_json is not None:
            seen_json.append(header)
        if header.get("type") != "chunk":
            continue
        payloads: list[bytes] = []
        for _ in range(header["frame_count"]):
            message = await ws.receive(timeout=2.0)
            if message.type != WSMsgType.BINARY:
                raise AssertionError(
                    f"expected binary frame payload, got {message.type!r}"
                )
            payloads.append(bytes(message.data))
        commit = await receive_json(ws)
        if seen_json is not None:
            seen_json.append(commit)
        if commit.get("type") != "chunk_committed":
            raise AssertionError(f"expected chunk commit, got {commit!r}")
        if commit.get("job_id") != header.get("job_id"):
            raise AssertionError("chunk commit job does not match its header")
        if commit.get("chunk_index") != header.get("chunk_index"):
            raise AssertionError("chunk commit index does not match its header")
        return header, payloads, commit


async def receive_until_json(
    ws: Any, predicate: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    while True:
        message = await receive_json(ws)
        seen.append(message)
        if message.get("type") == "chunk":
            for _ in range(message["frame_count"]):
                binary = await ws.receive(timeout=2.0)
                if binary.type != WSMsgType.BINARY:
                    raise AssertionError("chunk binary sequence was interrupted")
            commit = await receive_json(ws)
            if commit.get("type") != "chunk_committed":
                raise AssertionError("chunk binary sequence lacked its commit")
            seen.append(commit)
        if predicate(message):
            return message, seen


class BrowserStreamingServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.servers: list[BrowserStreamingServer] = []
        self.clients: list[ClientSession] = []

    async def asyncTearDown(self) -> None:
        for client in self.clients:
            await client.close()
        for server in reversed(self.servers):
            await server.close()

    async def start_server(self, backend: Any, **kwargs: Any) -> BrowserStreamingServer:
        server = BrowserStreamingServer(backend=backend, **kwargs)
        await server.start()
        self.servers.append(server)
        return server

    async def client(self) -> ClientSession:
        client = ClientSession(cookie_jar=CookieJar(unsafe=True))
        self.clients.append(client)
        return client

    async def connect(
        self,
        server: BrowserStreamingServer,
        *,
        client: ClientSession | None = None,
    ) -> tuple[ClientSession, Any, dict[str, Any]]:
        client = client or await self.client()
        response = await client.get(f"{server.origin}/")
        self.assertEqual(response.status, 200)
        html = await response.text()
        self.assertIn("Realtime video streaming demo", html)
        ws = await client.ws_connect(
            server.websocket_url,
            origin=server.origin,
            protocols=(PROTOCOL_VERSION,),
        )
        connected = await receive_json(ws)
        self.assertEqual(connected["type"], "connected")
        self.assertEqual(connected["protocol"], PROTOCOL_VERSION)
        self.assertRegex(connected["client_id"], r"^[0-9a-f]{32}$")
        return client, ws, connected

    async def test_page_nonce_origin_and_host_are_required(self) -> None:
        server = await self.start_server(TinyPngStreamingBackend())
        client = await self.client()

        hostile_host = await client.get(
            f"{server.origin}/",
            headers={"Host": "attacker.invalid"},
        )
        self.assertEqual(hostile_host.status, 403)

        with self.assertRaises(WSServerHandshakeError) as missing_nonce:
            await client.ws_connect(
                server.websocket_url,
                origin=server.origin,
                protocols=(PROTOCOL_VERSION,),
            )
        self.assertEqual(missing_nonce.exception.status, 403)

        page = await client.get(f"{server.origin}/")
        self.assertEqual(page.status, 200)
        await page.read()
        with self.assertRaises(WSServerHandshakeError) as hostile_origin:
            await client.ws_connect(
                server.websocket_url,
                origin="https://attacker.invalid",
                protocols=(PROTOCOL_VERSION,),
            )
        self.assertEqual(hostile_origin.exception.status, 403)

        ws = await client.ws_connect(
            server.websocket_url,
            origin=server.origin,
            protocols=(PROTOCOL_VERSION,),
        )
        await receive_json(ws)
        with self.assertRaises(WSServerHandshakeError) as replayed_nonce:
            await client.ws_connect(
                server.websocket_url,
                origin=server.origin,
                protocols=(PROTOCOL_VERSION,),
            )
        self.assertEqual(replayed_nonce.exception.status, 403)
        await ws.close()

    async def test_prompt_resolution_is_disclosed_before_explicit_generation(self) -> None:
        backend = RecordingPromptBackend()
        server = await self.start_server(
            backend,
            prompt_resolver=StaticPromptResolver(),
            prompt_resolution_timeout_s=0.5,
        )
        _client, ws, _connected = await self.connect(server)

        await ws.send_json(
            {
                "type": "resolve_prompt",
                "request_id": "resolution-one",
                "prompt": "a bouncing ball",
            }
        )
        resolved = await receive_json(ws)
        self.assertEqual(resolved["type"], "prompt_resolved")
        self.assertEqual(resolved["request_id"], "resolution-one")
        self.assertEqual(resolved["effective_prompt"], "Temporal expansion: a bouncing ball")
        self.assertEqual(resolved["prompt_transform_id"], "test-temporal-v1")
        self.assertTrue(resolved["prompt_changed"])
        self.assertRegex(resolved["raw_prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(resolved["effective_prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(resolved["prompt_resolution_ms"], float)
        self.assertGreaterEqual(resolved["prompt_resolution_ms"], 0)
        self.assertEqual(backend.requests, [])

        await ws.send_json(
            {
                "type": "start",
                "prompt": resolved["effective_prompt"],
                "prompt_resolution_id": resolved["request_id"],
                "seed": 7,
            }
        )
        accepted = await receive_json(ws)
        self.assertEqual(accepted["type"], "start_accepted")
        self.assertEqual(accepted["prompt_resolution_id"], "resolution-one")
        self.assertEqual(
            accepted["effective_prompt_sha256"],
            resolved["effective_prompt_sha256"],
        )
        self.assertEqual(
            accepted["raw_prompt_sha256"],
            resolved["raw_prompt_sha256"],
        )
        self.assertEqual(
            accepted["prompt_transform_id"],
            resolved["prompt_transform_id"],
        )
        for _ in range(100):
            if backend.requests:
                break
            await asyncio.sleep(0)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(backend.requests[0].prompt, resolved["effective_prompt"])
        await ws.close()

    async def test_prompt_resolution_failure_is_sanitized_and_starts_no_job(self) -> None:
        backend = RecordingPromptBackend()
        server = await self.start_server(
            backend,
            prompt_resolver=FailingPromptResolver(),
            prompt_resolution_timeout_s=0.5,
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json(
            {
                "type": "resolve_prompt",
                "request_id": "resolution-failure",
                "prompt": "a bouncing ball",
            }
        )
        failed = await receive_json(ws)
        self.assertEqual(
            failed,
            {
                "type": "prompt_resolution_failed",
                "request_id": "resolution-failure",
                "error_code": "prompt_resolution_failed",
            },
        )
        self.assertEqual(backend.requests, [])
        await ws.close()

    async def test_replaced_prompt_resolution_cannot_publish_a_stale_result(self) -> None:
        resolver = ReplacementPromptResolver()
        backend = RecordingPromptBackend()
        server = await self.start_server(
            backend,
            prompt_resolver=resolver,
            prompt_resolution_timeout_s=0.5,
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json(
            {
                "type": "resolve_prompt",
                "request_id": "resolution-first",
                "prompt": "first prompt",
            }
        )
        await asyncio.wait_for(resolver.first_started.wait(), timeout=0.5)
        await ws.send_json(
            {
                "type": "resolve_prompt",
                "request_id": "resolution-second",
                "prompt": "second prompt",
            }
        )
        resolved = await receive_json(ws)
        self.assertEqual(resolved["type"], "prompt_resolved")
        self.assertEqual(resolved["request_id"], "resolution-second")
        self.assertEqual(resolved["effective_prompt"], "Resolved: second prompt")
        resolver.release_first.set()
        with self.assertRaises(asyncio.TimeoutError):
            await ws.receive(timeout=0.05)
        self.assertEqual(backend.requests, [])
        await ws.close()

    async def test_png_stream_uses_binary_frames_and_start_barrier(self) -> None:
        delivery_ids = (f"delivery-{index}" for index in itertools.count())
        server = await self.start_server(
            TinyPngStreamingBackend(),
            job_id_factory=lambda: "job-one",
            delivery_id_factory=lambda: next(delivery_ids),
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "fox", "seed": 7})

        seen: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        binary_payloads: list[bytes] = []
        while True:
            message = await receive_json(ws)
            seen.append(message)
            if message.get("type") == "chunk":
                chunks.append(message)
                for _ in range(message["frame_count"]):
                    payload = await ws.receive(timeout=2.0)
                    self.assertEqual(payload.type, WSMsgType.BINARY)
                    binary_payloads.append(bytes(payload.data))
                commit = await receive_json(ws)
                seen.append(commit)
                self.assertEqual(commit["type"], "chunk_committed")
                self.assertNotIn("delivery_id", message)
                self.assertEqual(commit["job_id"], message["job_id"])
                self.assertEqual(commit["chunk_index"], message["chunk_index"])
                await ws.send_json(
                    {
                        "type": "presented",
                        "job_id": message["job_id"],
                        "chunk_index": message["chunk_index"],
                        "delivery_id": commit["delivery_id"],
                        "client_presented_ns": 10_000 + message["chunk_index"],
                    }
                )
            if (
                message.get("type") == "stream_event"
                and message.get("kind") == "job_completed"
            ):
                break

        accepted_index = next(
            index for index, item in enumerate(seen) if item["type"] == "start_accepted"
        )
        first_job_message = next(
            index
            for index, item in enumerate(seen)
            if item["type"] in {"chunk", "stream_event"}
        )
        self.assertLess(accepted_index, first_job_message)
        self.assertEqual(seen[accepted_index]["latent_frames"], 21)
        self.assertEqual(seen[accepted_index]["expected_rgb_frames"], 81)
        self.assertEqual([item["frame_count"] for item in chunks], [1] + [4] * 20)
        commits = [item for item in seen if item.get("type") == "chunk_committed"]
        self.assertEqual(len({item["delivery_id"] for item in commits}), 21)
        self.assertEqual(len(binary_payloads), 81)
        self.assertTrue(
            all(payload.startswith(b"\x89PNG\r\n\x1a\n") for payload in binary_payloads)
        )
        self.assertTrue(all(item["frame_media_type"] == "image/png" for item in chunks))
        self.assertTrue(all(item["renderable"] is True for item in chunks))
        self.assertTrue(
            all(item["payload_encoding"] == "websocket-binary" for item in chunks)
        )
        await ws.close()

    async def test_opaque_backend_is_not_mislabeled_renderable(self) -> None:
        server = await self.start_server(
            OpaqueBackend(),
            job_id_factory=lambda: "opaque-job",
            delivery_id_factory=lambda: "opaque-delivery",
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "opaque", "seed": 1})
        header, payloads, _commit = await receive_chunk(ws)
        self.assertEqual(header["frame_media_type"], "application/octet-stream")
        self.assertIs(header["renderable"], False)
        self.assertEqual(payloads, [b"opaque-0-0"])
        await ws.close()

    async def test_presentation_window_releases_only_after_full_chunk_ack(self) -> None:
        delivery_ids = (f"window-{index}" for index in itertools.count())
        server = await self.start_server(
            TinyPngStreamingBackend(),
            job_id_factory=lambda: "window-job",
            delivery_id_factory=lambda: next(delivery_ids),
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "window", "seed": 2})
        first, _first_payloads, first_commit = await receive_chunk(ws)
        second, _second_payloads, _second_commit = await receive_chunk(ws)
        with self.assertRaises(asyncio.TimeoutError):
            await ws.receive(timeout=0.05)

        await ws.send_json(
            {
                "type": "presented",
                "job_id": first["job_id"],
                "chunk_index": first["chunk_index"],
                "delivery_id": first_commit["delivery_id"],
                "client_presented_ns": 100,
            }
        )
        third, _third_payloads, _third_commit = await receive_chunk(ws)
        self.assertEqual(third["chunk_index"], 2)
        self.assertEqual(second["chunk_index"], 1)
        await ws.close()

    async def test_ack_before_complete_binary_group_is_rejected(self) -> None:
        ws = GatedWebSocket()
        session = _WebSocketSession(
            server=SessionHarness(),  # type: ignore[arg-type]
            client_id="client",
            ws=ws,  # type: ignore[arg-type]
        )
        session._current_job_id = "job"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        emit = asyncio.create_task(
            session.emit_event(
                StreamEvent(
                    kind="chunk_ready",
                    job_id="job",
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(b"one", b"two"),
                    frame_media_type="application/octet-stream",
                    ready_ns=100,
                    queue_depth=0,
                )
            )
        )
        await asyncio.wait_for(ws.header_sent.wait(), timeout=1.0)
        with self.assertRaises(_CommandError) as early:
            await asyncio.wait_for(
                session.handle_text(
                    json.dumps(
                        {
                            "type": "presented",
                            "job_id": "job",
                            "chunk_index": 0,
                            "delivery_id": "gated-delivery",
                            "client_presented_ns": 200,
                        }
                    )
                ),
                timeout=0.1,
            )
        self.assertEqual(early.exception.code, "delivery_not_sent")
        ws.allow_binary.set()
        await asyncio.wait_for(emit, timeout=1.0)

    async def test_delivery_is_registered_before_commit_becomes_peer_visible(
        self,
    ) -> None:
        ws = CommitGatedWebSocket()
        session = _WebSocketSession(
            server=SessionHarness(),  # type: ignore[arg-type]
            client_id="client",
            ws=ws,  # type: ignore[arg-type]
        )
        session._current_job_id = "job"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        emit = asyncio.create_task(
            session.emit_event(
                StreamEvent(
                    kind="chunk_ready",
                    job_id="job",
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(b"large-final-frame",),
                    frame_media_type="application/octet-stream",
                    ready_ns=100,
                    queue_depth=0,
                )
            )
        )

        await asyncio.wait_for(ws.binary_observable.wait(), timeout=1.0)
        header = json.loads(ws.strings[0])
        self.assertEqual(ws.binary, [b"large-final-frame"])
        self.assertEqual(header["type"], "chunk")
        self.assertNotIn("delivery_id", header)
        self.assertFalse(ws.commit_observable.is_set())
        self.assertNotIn("gated-delivery", session._deliveries)
        with self.assertRaises(_CommandError) as early:
            await session.handle_text(
                json.dumps(
                    {
                        "type": "presented",
                        "job_id": "job",
                        "chunk_index": 0,
                        "delivery_id": "gated-delivery",
                        "client_presented_ns": 150,
                    }
                )
            )
        self.assertEqual(early.exception.code, "delivery_not_sent")

        ws.allow_binary_return.set()
        await asyncio.wait_for(ws.commit_observable.wait(), timeout=1.0)
        commit = json.loads(ws.strings[1])
        self.assertEqual(commit["type"], "chunk_committed")
        self.assertIn(commit["delivery_id"], session._deliveries)

        acknowledged = asyncio.create_task(
            session.handle_text(
                json.dumps(
                    {
                        "type": "presented",
                        "job_id": "job",
                        "chunk_index": 0,
                        "delivery_id": commit["delivery_id"],
                        "client_presented_ns": 200,
                    }
                )
            )
        )
        await asyncio.sleep(0)
        self.assertNotIn(commit["delivery_id"], session._outstanding_delivery_ids)
        ws.allow_commit_return.set()
        await asyncio.wait_for(emit, timeout=1.0)
        await asyncio.wait_for(acknowledged, timeout=1.0)

    async def test_retiring_stale_job_does_not_disconnect_replacement(self) -> None:
        ws = RecordingWebSocket()
        session = _WebSocketSession(
            server=SessionHarness(),  # type: ignore[arg-type]
            client_id="client",
            ws=ws,  # type: ignore[arg-type]
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

        session.send_json = stale_terminal_send  # type: ignore[method-assign]
        session.disconnect = record_disconnect  # type: ignore[method-assign]
        completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        completed.set_exception(RuntimeError("backend failed"))

        await session._retire_failed_job("job-a", completed)

        self.assertFalse(disconnected)
        self.assertEqual(session._current_job_id, "job-b")

    async def test_replaced_chunk_does_not_emit_commit_for_retired_delivery(
        self,
    ) -> None:
        ws = CommitGatedWebSocket()
        ws.allow_commit_return.set()
        session = _WebSocketSession(
            server=SessionHarness(),  # type: ignore[arg-type]
            client_id="client",
            ws=ws,  # type: ignore[arg-type]
        )
        session._current_job_id = "job-a"
        session._announcement_gate = asyncio.Event()
        session._announcement_gate.set()
        emit = asyncio.create_task(
            session.emit_event(
                StreamEvent(
                    kind="chunk_ready",
                    job_id="job-a",
                    chunk_index=0,
                    first_frame_index=0,
                    frame_payloads=(b"frame",),
                    frame_media_type="application/octet-stream",
                    ready_ns=100,
                    queue_depth=0,
                )
            )
        )

        await asyncio.wait_for(ws.binary_observable.wait(), timeout=1.0)
        async with session._state:
            session._current_job_id = "job-b"
            session._state.notify_all()
        ws.allow_binary_return.set()
        await asyncio.wait_for(emit, timeout=1.0)

        messages = [json.loads(value) for value in ws.strings]
        self.assertEqual([message["type"] for message in messages], ["chunk"])

    async def test_delivery_ids_cannot_be_reused_after_retirement_window(self) -> None:
        harness = SessionHarness()
        generated = iter([f"delivery-{index}" for index in range(65)] + ["delivery-0"])
        harness.delivery_id_factory = lambda: next(generated)  # type: ignore[method-assign]
        session = _WebSocketSession(
            server=harness,  # type: ignore[arg-type]
            client_id="client",
            ws=RecordingWebSocket(),  # type: ignore[arg-type]
        )

        for _ in range(65):
            delivery_id = session._new_delivery_id_locked()
            session._outstanding_delivery_ids.add(delivery_id)
            session._retire_current_deliveries_locked()

        with self.assertRaisesRegex(ConnectionError, "reused"):
            session._new_delivery_id_locked()

    async def test_control_send_timeout_is_bounded(self) -> None:
        ws = StalledControlWebSocket()
        session = _WebSocketSession(
            server=SessionHarness(),  # type: ignore[arg-type]
            client_id="client",
            ws=ws,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ConnectionError, "control send timed out"):
            await asyncio.wait_for(
                session.send_json({"type": "test"}),
                timeout=0.2,
            )

    async def test_delivery_ack_rejects_duplicate_stale_and_cross_job(self) -> None:
        job_ids = iter(["job-one", "job-two"])
        delivery_ids = (f"ack-{index}" for index in itertools.count())
        server = await self.start_server(
            TinyPngStreamingBackend(),
            job_id_factory=lambda: next(job_ids),
            delivery_id_factory=lambda: next(delivery_ids),
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "one", "seed": 1})
        first, _payloads, first_commit = await receive_chunk(ws)
        ack = {
            "type": "presented",
            "job_id": first["job_id"],
            "chunk_index": first["chunk_index"],
            "delivery_id": first_commit["delivery_id"],
            "client_presented_ns": 100,
        }
        await ws.send_json(ack)
        recorded, _seen = await receive_until_json(
            ws, lambda item: item.get("type") == "presentation_recorded"
        )
        self.assertEqual(recorded["delivery_id"], first_commit["delivery_id"])

        await ws.send_json({**ack, "client_presented_ns": 101})
        duplicate, _seen = await receive_until_json(
            ws, lambda item: item.get("type") == "command_error"
        )
        self.assertEqual(duplicate["code"], "duplicate_presentation")

        await ws.send_json({"type": "start", "prompt": "two", "seed": 2})
        accepted, _seen = await receive_until_json(
            ws,
            lambda item: (
                item.get("type") == "start_accepted" and item.get("job_id") == "job-two"
            ),
        )
        self.assertEqual(accepted["job_id"], "job-two")

        await ws.send_json({**ack, "client_presented_ns": 102})
        stale, _seen = await receive_until_json(
            ws, lambda item: item.get("type") == "command_error"
        )
        self.assertEqual(stale["code"], "stale_job")

        await ws.send_json(
            {
                **ack,
                "job_id": "job-two",
                "client_presented_ns": 103,
            }
        )
        cross_job, _seen = await receive_until_json(
            ws, lambda item: item.get("type") == "command_error"
        )
        self.assertEqual(cross_job["code"], "stale_delivery")
        await ws.close()

    async def test_cancel_acceptance_precedes_cancelled_event(self) -> None:
        backend = BlockingBackend()
        server = await self.start_server(
            backend,
            job_id_factory=lambda: "blocking-job",
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "wait", "seed": 3})
        await receive_until_json(
            ws,
            lambda item: item.get("type") == "start_accepted",
        )
        await asyncio.wait_for(backend.entered.wait(), timeout=2.0)
        await ws.send_json({"type": "cancel", "job_id": "blocking-job"})
        _cancelled, seen = await receive_until_json(
            ws,
            lambda item: (
                item.get("type") == "stream_event"
                and item.get("kind") == "job_cancelled"
            ),
        )
        accepted_index = next(
            index
            for index, item in enumerate(seen)
            if item["type"] == "cancel_accepted"
        )
        cancelled_index = next(
            index
            for index, item in enumerate(seen)
            if item.get("kind") == "job_cancelled"
        )
        self.assertLess(accepted_index, cancelled_index)
        await asyncio.wait_for(backend.cancelled.wait(), timeout=2.0)
        await ws.close()

    async def test_cancel_acceptance_is_not_emitted_for_completed_job(self) -> None:
        server = await self.start_server(
            IncompleteBackend(),
            job_id_factory=lambda: "failed-job",
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "short", "seed": 4})
        await receive_until_json(
            ws,
            lambda item: (
                item.get("type") == "stream_event" and item.get("kind") == "job_failed"
            ),
        )
        await ws.send_json({"type": "cancel", "job_id": "failed-job"})
        result, seen = await receive_until_json(
            ws,
            lambda item: item.get("type") == "cancel_result",
        )
        self.assertIs(result["cancelled"], False)
        self.assertFalse(any(item.get("type") == "cancel_accepted" for item in seen))
        await ws.close()

    async def test_unacknowledged_client_gets_terminal_and_can_start_fresh_job(
        self,
    ) -> None:
        job_ids = iter(["backpressured-job", "fresh-job"])
        delivery_ids = (f"backpressure-{index}" for index in itertools.count())
        server = await self.start_server(
            TinyPngStreamingBackend(),
            job_id_factory=lambda: next(job_ids),
            delivery_id_factory=lambda: next(delivery_ids),
            emit_timeout_s=0.05,
            control_send_timeout_s=0.2,
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "stall", "seed": 5})

        failed, seen = await receive_until_json(
            ws,
            lambda item: (
                item.get("type") == "stream_event" and item.get("kind") == "job_failed"
            ),
        )
        self.assertEqual(failed["job_id"], "backpressured-job")
        self.assertEqual(failed["error_code"], "client_backpressure_timeout")
        self.assertEqual(
            len([item for item in seen if item.get("type") == "chunk"]),
            2,
        )

        await ws.send_json({"type": "start", "prompt": "fresh", "seed": 6})
        accepted, _seen = await receive_until_json(
            ws,
            lambda item: (
                item.get("type") == "start_accepted"
                and item.get("job_id") == "fresh-job"
            ),
        )
        self.assertEqual(accepted["job_id"], "fresh-job")
        await ws.close()

    async def test_connection_lifetime_job_id_ledger_is_capped(self) -> None:
        issued = 0

        def next_job_id() -> str:
            nonlocal issued
            issued += 1
            return f"bounded-job-{issued}"

        server = await self.start_server(
            TinyPngStreamingBackend(),
            job_id_factory=next_job_id,
            max_jobs_per_connection=1,
        )
        _client, ws, _connected = await self.connect(server)
        await ws.send_json({"type": "start", "prompt": "one", "seed": 1})
        accepted, _seen = await receive_until_json(
            ws,
            lambda item: item.get("type") == "start_accepted",
        )
        self.assertEqual(accepted["job_id"], "bounded-job-1")

        await ws.send_json({"type": "start", "prompt": "two", "seed": 2})
        rejected, _seen = await receive_until_json(
            ws,
            lambda item: item.get("type") == "command_error",
        )
        self.assertEqual(rejected["code"], "start_rejected")
        self.assertEqual(issued, 1)
        await ws.close()

    async def test_static_demo_has_strict_external_css_and_presentation_contract(
        self,
    ) -> None:
        server = await self.start_server(TinyPngStreamingBackend())
        client = await self.client()
        page = await client.get(f"{server.origin}/")
        self.assertEqual(page.status, 200)
        html = await page.text()
        self.assertNotIn("<style", html)
        self.assertIn('rel="stylesheet" href="/demo.css"', html)
        self.assertIn("fake backend", html)
        csp = page.headers["Content-Security-Policy"]
        self.assertIn("style-src 'self'", csp)
        self.assertNotIn("'unsafe-inline'", csp)

        css = await client.get(f"{server.origin}/demo.css")
        self.assertEqual(css.status, 200)
        self.assertEqual(css.content_type, "text/css")
        self.assertIn("canvas", await css.text())

        script = await client.get(f"{server.origin}/demo.js")
        self.assertEqual(script.status, 200)
        javascript = await script.text()
        self.assertIn('message.type === "chunk_committed"', javascript)
        self.assertIn("1000 / 16", javascript)
        self.assertIn("requestAnimationFrame", javascript)
        self.assertIn("dataset.renderedFrames", javascript)
        self.assertIn("dataset.renderedChunks", javascript)
        self.assertIn("dataset.ackCount", javascript)
        self.assertIn("serverCompleted", javascript)
        self.assertIn("expectedFrames", javascript)
        self.assertIn("presentationChain.catch", javascript)

    async def test_cf1_mode_labels_page_and_forwards_backend_timeouts(self) -> None:
        server = await self.start_server(
            TinyPngStreamingBackend(),
            demo_backend_kind="cf1",
            backend_chunk_timeout_s=123.0,
            backend_close_timeout_s=7.0,
        )
        self.assertEqual(server.backend_chunk_timeout_s, 123.0)
        self.assertEqual(server.backend_close_timeout_s, 7.0)
        self.assertEqual(server.registry._backend_chunk_timeout_s, 123.0)
        self.assertEqual(server.registry._backend_close_timeout_s, 7.0)

        client = await self.client()
        page = await client.get(f"{server.origin}/")
        self.assertEqual(page.status, 200)
        html = await page.text()
        self.assertIn("CF++1 H100 backend", html)
        self.assertNotIn("local fake backend", html)

    def test_unknown_demo_backend_kind_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "demo_backend_kind"):
            BrowserStreamingServer(
                backend=TinyPngStreamingBackend(),
                demo_backend_kind="unreviewed",
            )

    def test_browser_client_protocol_is_executed_under_node(self) -> None:
        test_file = Path(__file__).with_name("streaming_demo_client.test.mjs")
        completed = subprocess.run(
            ["node", "--test", str(test_file)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"node client tests failed:\n{completed.stdout}\n{completed.stderr}",
        )

    async def test_start_contract_is_fixed_and_prompt_is_bounded(self) -> None:
        server = await self.start_server(TinyPngStreamingBackend())
        _client, ws, _connected = await self.connect(server)
        await ws.send_json(
            {
                "type": "start",
                "prompt": "fox",
                "seed": 1,
                "latent_frames": 20,
            }
        )
        invalid_shape = await receive_json(ws)
        self.assertEqual(invalid_shape["code"], "invalid_command")

        await ws.send_json({"type": "start", "prompt": "x" * 4097, "seed": 1})
        oversized_prompt = await receive_json(ws)
        self.assertEqual(oversized_prompt["code"], "prompt_too_large")

        await ws.send_str(r'{"type":"start","prompt":"\ud800","seed":1}')
        invalid_unicode = await receive_json(ws)
        self.assertEqual(invalid_unicode["code"], "invalid_prompt")
        await ws.close()

    async def test_control_message_over_16_kib_closes_the_socket(self) -> None:
        server = await self.start_server(TinyPngStreamingBackend())
        _client, ws, _connected = await self.connect(server)
        await ws.send_str(
            json.dumps(
                {
                    "type": "start",
                    "prompt": "x",
                    "seed": 1,
                    "padding": "y" * (17 * 1024),
                }
            )
        )
        message = await ws.receive(timeout=2.0)
        self.assertIn(message.type, {WSMsgType.CLOSE, WSMsgType.CLOSED})
        self.assertEqual(ws.close_code, WSCloseCode.MESSAGE_TOO_BIG)

    async def test_non_loopback_binding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            BrowserStreamingServer(
                backend=TinyPngStreamingBackend(),
                host="0.0.0.0",
            )


if __name__ == "__main__":
    unittest.main()
