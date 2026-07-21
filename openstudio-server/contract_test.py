#!/usr/bin/env python3
"""Executable wire-contract spec for openstudio-server (no GPU needed).

Boots `server.py --pipeline fake` as a subprocess, then drives the full v1
contract from a real WebSocket client: hello, binary frame round-trip with
header echo, 512x512 output, prompt hot-swap, config clamping, ping/pong,
bad-frame error handling, single-session busy policy, slot release, stats —
plus the backpressure invariant (firehose 60 frames at a 50 ms pipeline: only
the newest survives, the rest count as dropped_stale).

    uv run --with opencv-python-headless --with numpy --with websockets \
        python contract_test.py

Deps: same as local dev (websockets, opencv-python-headless, numpy).
This is the regression net for the transport layer: run it before touching
server.py, and keep it green. GPU behavior is covered by `--selfcheck` on the
pod, not here.

Tunnel mode: with CONTRACT_TEST_PORT pointed at an ssh -L tunnel to a live pod
(the local subprocess then fails to bind and every check runs against the real
server), set CONTRACT_TEST_SKIP_FIREHOSE=1. The firehose burst is ~6 MB of
JPEG; on a narrow uplink the burst itself saturates the link until the test
client's own keepalive dies (1011) — it validates the server's drop behavior,
which only measures cleanly on localhost. Real clients never firehose: wire
contract §5 gates sends on ws.bufferedAmount.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import struct
import subprocess
import sys
import time

import cv2
import numpy as np
from websockets.asyncio.client import connect

HDR_IN = struct.Struct("<BId")
HDR_OUT = struct.Struct("<BIdf")
PORT = int(os.environ.get("CONTRACT_TEST_PORT", "8871"))
URI = f"ws://127.0.0.1:{PORT}"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("ok   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def frame_payload(seq: int, ts: float) -> bytes:
    img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    assert ok
    return HDR_IN.pack(0x01, seq, ts) + jpg.tobytes()


async def recv_json_of_type(ws, wanted: str, tries: int = 40) -> dict:
    for _ in range(tries):
        m = await asyncio.wait_for(ws.recv(), 5)
        if isinstance(m, str):
            o = json.loads(m)
            if o.get("type") == wanted:
                return o
    return {}


async def contract_checks() -> None:
    async with connect(URI, max_size=8 * 1024 * 1024) as ws:
        hello = json.loads(await ws.recv())
        check("hello first", hello.get("type") == "hello" and hello.get("proto") == 1)
        check("hello geometry", hello.get("width") == 512 and hello.get("height") == 512)

        ts = time.time() * 1000.0
        await ws.send(frame_payload(7, ts))
        out = None
        for _ in range(20):
            m = await asyncio.wait_for(ws.recv(), 5)
            if isinstance(m, (bytes, bytearray)):
                out = m
                break
        check("frame returned", out is not None)
        if out:
            magic, seq, ts_echo, infer_ms = HDR_OUT.unpack_from(out)
            check("out header echo", magic == 0x02 and seq == 7 and abs(ts_echo - ts) < 1e-6,
                  f"infer_ms={infer_ms:.2f}")
            dec = cv2.imdecode(np.frombuffer(out[HDR_OUT.size:], np.uint8), cv2.IMREAD_COLOR)
            check("out jpeg 512x512", dec is not None and dec.shape[:2] == (512, 512))

        await ws.send(json.dumps({"type": "prompt", "text": "an oil painting"}))
        applied = await recv_json_of_type(ws, "prompt_applied")
        check("prompt_applied", applied.get("text") == "an oil painting", f"ms={applied.get('ms')}")

        await ws.send(json.dumps({"type": "config", "jpeg_quality": 200}))  # must clamp
        check("config clamped to 95",
              (await recv_json_of_type(ws, "config_applied")).get("jpeg_quality") == 95)

        await ws.send(json.dumps({"type": "ping", "t": 42}))
        check("pong", (await recv_json_of_type(ws, "pong")).get("t") == 42)

        await ws.send(b"\xff garbage")
        check("bad frame -> error json", bool(await recv_json_of_type(ws, "error")))

        async with connect(URI) as ws2:
            check("second client busy", json.loads(await ws2.recv()).get("type") == "busy")

        check("stats flow", bool(await recv_json_of_type(ws, "stats")))

    async with connect(URI) as ws3:
        check("slot freed after disconnect", json.loads(await ws3.recv()).get("type") == "hello")


async def backpressure_checks() -> None:
    """Server runs the fake pipeline at 50 ms/frame; we firehose 60 frames."""
    async with connect(URI, max_size=8 * 1024 * 1024) as ws:
        await ws.recv()  # hello
        for seq in range(60):
            await ws.send(frame_payload(seq, time.time() * 1000.0))
        got: list[int] = []
        dropped = 0
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            try:
                m = await asyncio.wait_for(ws.recv(), 1)
            except asyncio.TimeoutError:
                continue
            if isinstance(m, (bytes, bytearray)):
                got.append(HDR_OUT.unpack_from(m)[1])
            else:
                o = json.loads(m)
                if o.get("type") == "stats":
                    dropped = max(dropped, o.get("dropped_stale", 0))
                    if 59 in got and dropped:
                        break
        check("firehose: most frames dropped", 0 < len(got) < 30, f"returned={len(got)}")
        check("firehose: newest frame wins", bool(got) and got[-1] == 59, f"last={got[-1] if got else None}")
        check("firehose: drops counted", dropped >= 30, f"dropped_stale={dropped}")


async def main() -> int:
    server = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"),
         "--pipeline", "fake", "--fake-delay-ms", "50", "--warmup-frames", "1",
         "--port", str(PORT), "--log-level", "WARNING"],
    )
    try:
        for _ in range(50):  # wait for the port
            try:
                async with connect(URI) as probe:
                    await probe.recv()
                break
            except OSError:
                await asyncio.sleep(0.2)
        else:
            print("FAIL server never came up", file=sys.stderr)
            return 1
        await contract_checks()
        if os.environ.get("CONTRACT_TEST_SKIP_FIREHOSE"):
            print("skip firehose backpressure block (CONTRACT_TEST_SKIP_FIREHOSE set)")
        else:
            await backpressure_checks()
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=10)
    print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
