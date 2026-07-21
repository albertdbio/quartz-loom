"""Bounded validation for the metadata-free PNG subset used by this project.

The validator checks the complete PNG container, critical-chunk ordering and
CRCs, rejects all optional metadata chunks, inflates the IDAT stream to its
exact expected size, and validates every scanline filter byte. The deliberately
narrow subset avoids claiming support for ancillary structures this parser does
not interpret. It returns only a boolean so malformed provider bytes cannot
leak parser details across serving boundaries.
"""

from __future__ import annotations

import struct
import zlib


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_DIMENSION = 32_768
_MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
_VALID_DEPTHS = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


def _pass_size(length: int, start: int, step: int) -> int:
    if length <= start:
        return 0
    return (length - start + step - 1) // step


def _scanline_layout(
    width: int,
    height: int,
    bits_per_pixel: int,
    interlace: int,
) -> tuple[tuple[int, int], ...] | None:
    passes = ((0, 0, 1, 1),) if interlace == 0 else _ADAM7_PASSES
    layout: list[tuple[int, int]] = []
    total = 0
    for x_start, y_start, x_step, y_step in passes:
        pass_width = _pass_size(width, x_start, x_step)
        pass_height = _pass_size(height, y_start, y_step)
        if pass_width == 0 or pass_height == 0:
            continue
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        total += pass_height * (row_bytes + 1)
        if total > _MAX_DECOMPRESSED_BYTES:
            return None
        layout.append((pass_height, row_bytes))
    return tuple(layout) if layout else None


def _valid_chunk_name(kind: bytes) -> bool:
    return (
        len(kind) == 4
        and all(
            65 <= character <= 90 or 97 <= character <= 122
            for character in kind
        )
        and not (kind[2] & 0x20)
    )


def _inflate_exact(compressed: bytes, expected_size: int) -> bytes | None:
    try:
        inflater = zlib.decompressobj()
        decoded = inflater.decompress(compressed, expected_size + 1)
    except zlib.error:
        return None
    if (
        len(decoded) != expected_size
        or not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
    ):
        return None
    try:
        if inflater.flush():
            return None
    except zlib.error:
        return None
    return decoded


def is_valid_png(
    payload: object,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    require_rgb8: bool = False,
) -> bool:
    """Return whether *payload* is one complete PNG in the supported subset."""

    if not isinstance(payload, bytes) or not payload.startswith(PNG_SIGNATURE):
        return False

    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    idat_closed = False
    saw_iend = False
    chunk_index = 0

    while offset < len(payload):
        if len(payload) - offset < 12:
            return False
        length = int.from_bytes(payload[offset : offset + 4], "big")
        kind = payload[offset + 4 : offset + 8]
        if not _valid_chunk_name(kind):
            return False
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if data_end < data_start or crc_end > len(payload):
            return False
        data = payload[data_start:data_end]
        expected_crc = int.from_bytes(payload[data_end:crc_end], "big")
        observed_crc = zlib.crc32(kind)
        observed_crc = zlib.crc32(data, observed_crc) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            return False
        offset = crc_end

        if chunk_index == 0 and kind != b"IHDR":
            return False
        chunk_index += 1

        if kind == b"IHDR":
            if ihdr is not None or length != 13:
                return False
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", data)
            )
            if (
                width == 0
                or height == 0
                or width > _MAX_DIMENSION
                or height > _MAX_DIMENSION
                or color_type not in _VALID_DEPTHS
                or bit_depth not in _VALID_DEPTHS[color_type]
                or compression != 0
                or filtering != 0
                or interlace not in {0, 1}
            ):
                return False
            ihdr = (width, height, bit_depth, color_type, interlace)
            continue

        if ihdr is None:
            return False
        if kind == b"IDAT":
            if idat_closed:
                return False
            idat_parts.append(data)
            continue

        if idat_parts:
            idat_closed = True

        if kind == b"IEND":
            if length != 0 or not idat_parts or offset != len(payload):
                return False
            saw_iend = True
            break
        return False

    if not saw_iend or ihdr is None:
        return False
    width, height, bit_depth, color_type, interlace = ihdr
    if expected_width is not None and width != expected_width:
        return False
    if expected_height is not None and height != expected_height:
        return False
    if color_type == 3:
        return False
    if require_rgb8 and (bit_depth != 8 or color_type != 2):
        return False

    layout = _scanline_layout(
        width,
        height,
        _CHANNELS[color_type] * bit_depth,
        interlace,
    )
    if layout is None:
        return False
    expected_size = sum(rows * (row_bytes + 1) for rows, row_bytes in layout)
    decoded = _inflate_exact(b"".join(idat_parts), expected_size)
    if decoded is None:
        return False

    decoded_offset = 0
    for rows, row_bytes in layout:
        for _ in range(rows):
            if decoded[decoded_offset] > 4:
                return False
            decoded_offset += row_bytes + 1
    return decoded_offset == len(decoded)
