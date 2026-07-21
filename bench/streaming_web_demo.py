"""Runnable loopback demo backed by valid, generated one-pixel PNG frames."""

from __future__ import annotations

import argparse
import asyncio
import struct
import zlib
from collections.abc import AsyncIterator, Sequence

from bench.streaming_service import DecodedChunk, StreamRequest
from bench.streaming_websocket import BrowserStreamingServer


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def make_tiny_png(red: int, green: int, blue: int) -> bytes:
    """Return a valid 1x1 RGB PNG, useful as honest renderable demo payload."""

    channels = (red, green, blue)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 255
        for value in channels
    ):
        raise ValueError("PNG channels must be integers from 0 through 255")
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


class TinyPngStreamingBackend:
    """Deterministic demo backend; it does not perform video generation."""

    async def stream(self, request: StreamRequest) -> AsyncIterator[DecodedChunk]:
        frame_counts: Sequence[int] = [1] + [4] * (request.latent_frames - 1)
        frame_number = 0
        for frame_count in frame_counts:
            await asyncio.sleep(0)
            payloads: list[bytes] = []
            for _ in range(frame_count):
                offset = (request.seed + frame_number * 17) & 0xFF
                payloads.append(
                    make_tiny_png(offset, (offset * 3) & 0xFF, (255 - offset) & 0xFF)
                )
                frame_number += 1
            yield DecodedChunk(tuple(payloads), frame_media_type="image/png")


async def _serve(host: str, port: int) -> None:
    server = BrowserStreamingServer(
        backend=TinyPngStreamingBackend(),
        host=host,
        port=port,
    )
    await server.start()
    print(f"Demo listening at {server.origin}/", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
