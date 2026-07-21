#!/usr/bin/env python3
"""White-box protocol net for openstudio-server — pytest, no GPU, no subprocess.

contract_test.py is the black-box executable spec: it boots `server.py
--pipeline fake` as a subprocess and drives it like a browser would. This suite
is the white-box half: it runs the REAL handler/worker/mailbox/outbox stack
in-process over a real localhost WebSocket, but monkeypatches StreamPipeline
with an instrumented stub so tests can

  (a) exercise the `--pipeline stream` build path without torch/CUDA/model,
  (b) FREEZE the "GPU" mid-frame (a threading gate) to make backpressure and
      prompt-coalescing assertions deterministic instead of timing-dependent,
  (c) assert what the pipeline actually saw — cover-fit geometry, coalesced
      prompt lists — invariants a black-box client cannot reach.

Boot-path config (serve_forever's compression=None / max_size / signal
handling) stays contract_test.py's job; this file binds the handler directly.

    uv venv .venv-test --python 3.12
    uv pip install -p .venv-test/bin/python \
        pytest pytest-asyncio websockets==15.0.1 numpy opencv-python-headless
    .venv-test/bin/python -m pytest test_server_protocol.py -q
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402

HDR_IN = struct.Struct("<BId")
HDR_OUT = struct.Struct("<BIdf")


# --------------------------------------------------------------------------- #
# instrumented stub pipeline
# --------------------------------------------------------------------------- #

class StubPipeline:
    """Stands in for StreamPipeline. Inverts pixels (a restyle a test can see),
    records every input shape and applied prompt, and can hold mid-`process`
    on a gate so a test controls exactly when the "GPU" finishes a frame."""

    name = "stub"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.processed_shapes: list[tuple[int, ...]] = []
        self.prompts_applied: list[str] = []
        self.warmed = -1
        self.started = threading.Event()  # a process() call has begun
        self._gate = threading.Event()
        self._gate.set()  # open by default; hold() closes it

    def hold(self) -> None:
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    def process(self, rgb: np.ndarray) -> np.ndarray:
        self.processed_shapes.append(rgb.shape)
        self.started.set()
        assert self._gate.wait(timeout=10), "test forgot to release() the stub gate"
        return 255 - rgb

    def set_prompt(self, text: str) -> float:
        self.prompts_applied.append(text)
        self.prompt = text
        return 0.42

    def warmup(self, n: int) -> None:
        self.warmed = n


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def stub_server(monkeypatch: pytest.MonkeyPatch):
    """Real Worker + real handler on a real localhost WS; stubbed pipeline."""
    box: dict[str, StubPipeline] = {}

    def factory(*, model_id, prompt, width, height, t_index_list, use_tiny_vae, use_xformers,
                **levers):  # quality levers (cfg_type, guidance_scale, …) — accepted, unused
        box["stub"] = StubPipeline(prompt)
        return box["stub"]

    monkeypatch.setattr(server, "StreamPipeline", factory)
    args = server.parse_args(["--pipeline", "stream", "--warmup-frames", "0"])
    pipeline = server.build_pipeline(args)  # goes through the stream branch
    stub = box["stub"]
    assert pipeline is stub and stub.warmed == 0

    worker = server.Worker(server.FrameProcessor(pipeline, args.width, args.height, args.jpeg_quality))
    worker.start()
    try:
        handler = server._make_handler(server.ServerState(worker=worker, args=args))
        async with serve(handler, "127.0.0.1", 0, max_size=8 * 1024 * 1024, compression=None) as srv:
            port = srv.sockets[0].getsockname()[1]
            yield f"ws://127.0.0.1:{port}", stub, worker
    finally:
        stub.release()  # never leave the worker thread blocked on the gate
        worker.stop()


def frame_bytes(seq: int, ts: float, value: int = 200, w: int = 1280, h: int = 720) -> bytes:
    """A solid-color 720p input: cover-fit keeps it solid, so the stub's
    inversion is checkable through JPEG loss (mean ≈ 255 - value)."""
    img = np.full((h, w, 3), value, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    return HDR_IN.pack(0x01, seq, ts) + jpg.tobytes()


async def _next(ws, want, timeout: float, what: str):
    """Next message matching `want`, else AssertionError — ALWAYS AssertionError
    on timeout (never TimeoutError), so negative probes can pytest.raises it."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"no {what} arrived within {timeout}s")
        try:
            msg = await asyncio.wait_for(ws.recv(), remaining)
        except TimeoutError:
            raise AssertionError(f"no {what} arrived within {timeout}s") from None
        found = want(msg)
        if found is not None:
            return found


async def next_binary(ws, timeout: float = 5.0) -> bytes:
    return await _next(
        ws, lambda m: bytes(m) if isinstance(m, (bytes, bytearray)) else None, timeout, "binary frame"
    )


async def next_json(ws, wanted: str, timeout: float = 5.0) -> dict:
    def match(m):
        if isinstance(m, str):
            obj = json.loads(m)
            if obj.get("type") == wanted:
                return obj
        return None

    return await _next(ws, match, timeout, f"{wanted!r} message")


async def poll(cond, timeout: float = 5.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hello_handshake(monkeypatch):
    """§2: hello is the FIRST message and advertises the live pipeline."""
    async with stub_server(monkeypatch) as (uri, stub, _worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert hello == {
                "type": "hello",
                "server": "openstudio-server",
                "proto": 1,
                "pipeline": "stub",  # hello reflects pipeline.name, not the CLI flag
                "model": "stabilityai/sd-turbo",
                "width": 512,
                "height": 512,
                "t_index": [35, 45],
                "prompt": stub.prompt,
                "jpeg_quality": 80,
                # boot-time quality levers, defaults (additive since proto 1)
                "vae": "taesd",
                "cfg_type": "none",
                "guidance_scale": 1.0,
                "delta": 1.0,
                "noise_mode": "add",
                "similar_filter": None,
                "similar_max_skip": 10,
                "lcm_lora": False,
                "seed": 2,
            }


@pytest.mark.asyncio
async def test_frame_roundtrip_geometry_and_echo(monkeypatch):
    """§3: 720p in → cover-fit 512x512 through the pipeline → 512x512 JPEG out,
    with seq + capture_ts echoed bit-exactly and infer_ms attached."""
    async with stub_server(monkeypatch) as (uri, stub, _worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            await ws.recv()  # hello
            ts = 123456.789
            await ws.send(frame_bytes(7, ts, value=200))
            out = await next_binary(ws)
            magic, seq, ts_echo, infer_ms = HDR_OUT.unpack_from(out)
            assert magic == 0x02 and seq == 7
            assert ts_echo == ts  # f64 must round-trip exactly, not approximately
            assert infer_ms >= 0.0
            img = cv2.imdecode(np.frombuffer(out[HDR_OUT.size:], np.uint8), cv2.IMREAD_COLOR)
            assert img is not None and img.shape == (512, 512, 3)
            assert abs(float(img.mean()) - 55.0) < 4.0  # 255-200, through JPEG loss
            assert stub.processed_shapes == [(512, 512, 3)]  # cover-fit BEFORE pipeline


@pytest.mark.asyncio
async def test_prompt_swap_acks_and_survives_reconnect(monkeypatch):
    """§4: prompt hot-swap hits the pipeline once and is acked; a later
    session's hello advertises the NEW prompt (pipeline outlives sessions)."""
    async with stub_server(monkeypatch) as (uri, stub, _worker):
        async with connect(uri) as ws:
            await ws.recv()  # hello
            await ws.send(json.dumps({"type": "prompt", "text": "an oil painting"}))
            ack = await next_json(ws, "prompt_applied")
            assert ack["text"] == "an oil painting" and ack["ms"] == 0.42
            assert stub.prompts_applied == ["an oil painting"]
        await poll(lambda: True, 0.1)  # let the server run its disconnect path
        for _ in range(20):  # slot frees asynchronously; retry busy briefly
            async with connect(uri) as ws2:
                first = json.loads(await asyncio.wait_for(ws2.recv(), 5))
                if first["type"] == "hello":
                    assert first["prompt"] == "an oil painting"
                    return
            await asyncio.sleep(0.1)
        raise AssertionError("session slot never freed")


@pytest.mark.asyncio
async def test_prompt_coalescing_under_load(monkeypatch):
    """§4: several prompts queued while the GPU is busy → only the NEWEST is
    applied, and only that one is acked."""
    async with stub_server(monkeypatch) as (uri, stub, worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            await ws.recv()  # hello
            stub.hold()
            await ws.send(frame_bytes(0, 1.0))
            await poll(stub.started.is_set, what="worker to enter process()")
            for text in ("a", "b", "c"):
                await ws.send(json.dumps({"type": "prompt", "text": text}))
            await poll(lambda: len(worker._control) == 3, what="3 queued prompt ops")
            stub.release()
            ack = await next_json(ws, "prompt_applied")
            assert ack["text"] == "c"
            assert stub.prompts_applied == ["c"]  # a and b never touched the pipeline
            with pytest.raises(AssertionError):  # and no second ack ever arrives
                await next_json(ws, "prompt_applied", timeout=0.7)


@pytest.mark.asyncio
async def test_newest_frame_wins_under_burst(monkeypatch):
    """§5: 20 frames against a frozen GPU → the mailbox keeps only the newest;
    exactly frames 0 and 19 come back and 18 drops are counted."""
    async with stub_server(monkeypatch) as (uri, stub, worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            await ws.recv()  # hello
            stub.hold()
            await ws.send(frame_bytes(0, 100.0))
            await poll(stub.started.is_set, what="worker to enter process()")  # 0 is ON the GPU
            for seq in range(1, 20):
                await ws.send(frame_bytes(seq, 100.0 + seq))
            await poll(lambda: worker.frames_in == 20, what="all 20 frames ingested")
            stub.release()

            seqs = [HDR_OUT.unpack_from(await next_binary(ws))[1] for _ in range(2)]
            assert seqs == [0, 19]
            with pytest.raises(AssertionError):  # frames 1..18 must NEVER appear
                await next_binary(ws, timeout=0.7)
            assert [s[0] for s in stub.processed_shapes] == [512, 512]  # exactly 2 processed

            stats = await next_json(ws, "stats")
            while stats.get("dropped_stale") != 18:
                stats = await next_json(ws, "stats")
            assert stats["dropped_outbox"] == 0  # client kept reading; egress never dropped
            assert stats["decode_failures"] == 0


@pytest.mark.asyncio
async def test_second_client_busy_then_slot_release(monkeypatch):
    """§1: one live session; a second client gets busy JSON + close 1013 and
    the first session keeps working; the slot frees on disconnect."""
    async with stub_server(monkeypatch) as (uri, _stub, _worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            await ws.recv()  # hello
            async with connect(uri) as ws2:
                busy = json.loads(await asyncio.wait_for(ws2.recv(), 5))
                assert busy["type"] == "busy"
                with pytest.raises(ConnectionClosed) as closed:
                    await ws2.recv()
                assert closed.value.rcvd is not None and closed.value.rcvd.code == 1013
            await ws.send(frame_bytes(1, 2.0))  # first session unaffected
            assert HDR_OUT.unpack_from(await next_binary(ws))[1] == 1
        for _ in range(20):
            async with connect(uri) as ws3:
                if json.loads(await asyncio.wait_for(ws3.recv(), 5))["type"] == "hello":
                    return
            await asyncio.sleep(0.1)
        raise AssertionError("session slot never freed")


@pytest.mark.asyncio
async def test_malformed_binary_is_nonfatal(monkeypatch):
    """§3/§6: bad magic and truncated headers get error JSON; the session and
    the frame path stay alive (error is informational, only close is fatal)."""
    async with stub_server(monkeypatch) as (uri, _stub, worker):
        async with connect(uri, max_size=8 * 1024 * 1024) as ws:
            await ws.recv()  # hello
            await ws.send(b"\x03" + b"\x00" * 20)  # wrong magic
            assert "header" in (await next_json(ws, "error"))["message"]
            await ws.send(HDR_IN.pack(0x01, 5, 1.0))  # header only, no JPEG payload
            assert "header" in (await next_json(ws, "error"))["message"]
            await ws.send(json.dumps({"type": "nope"}))  # unknown control type
            assert "nope" in (await next_json(ws, "error"))["message"]
            await ws.send(frame_bytes(6, 3.0))
            assert HDR_OUT.unpack_from(await next_binary(ws))[1] == 6
            assert worker.frames_in == 1  # the malformed blobs never reached the worker


@pytest.mark.asyncio
async def test_ping_pong_and_config_clamp(monkeypatch):
    """§4: pong echoes t; jpeg_quality clamps to [30, 95] and acks FIFO."""
    async with stub_server(monkeypatch) as (uri, _stub, worker):
        async with connect(uri) as ws:
            await ws.recv()  # hello
            await ws.send(json.dumps({"type": "ping", "t": 3.25}))
            assert (await next_json(ws, "pong"))["t"] == 3.25
            acks = []
            for q in (200, 5, 60):
                await ws.send(json.dumps({"type": "config", "jpeg_quality": q}))
            for _ in range(3):
                acks.append((await next_json(ws, "config_applied"))["jpeg_quality"])
            assert acks == [95, 30, 60]
            assert worker.processor.jpeg_quality == 60
