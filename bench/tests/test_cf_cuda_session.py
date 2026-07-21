from __future__ import annotations

import base64
import struct
import unittest
import zlib
from dataclasses import replace

from bench.cf_cuda_adapter import (
    CF1_ASSET_LOCK_SHA256,
    CF1_EFFECTIVE_CONFIG_SHA256,
    CF1_RUNTIME_LOCK_SHA256,
    CF1_SOURCE_COMMIT,
    CF1_STACK_ID,
    CF1_TOKENIZER_SENTINEL_SHA256,
    CF1AssetIdentity,
    CF1BootstrapProvenance,
    CF1Runtime,
    _cf1_guard_bundle_sha256,
    _provenance_identity_sha256,
)
from bench.cf_runtime_preflight import CF1_RUNTIME_ID, RuntimePreflightIdentity
from bench.cf_runtime_evidence import (
    DEFAULT_RUNTIME_EVIDENCE_PATH,
    load_runtime_evidence_snapshot,
    runtime_evidence_locked_identities,
)
from bench.cf_cuda_session import (
    CudaSessionError,
    RollingTaehvChunkDecoder,
    _payload_matches_media_type,
)
from bench.model_asset_preflight import DEFAULT_LOCK_PATH, load_asset_lock_snapshot


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _rgb_png(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x01\x02\x03" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline * height, level=1))
        + _png_chunk(b"IEND", b"")
    )


PNG_832X480 = _rgb_png(832, 480)
_LOCKED_NATIVE_IDENTITIES = runtime_evidence_locked_identities(
    load_runtime_evidence_snapshot(DEFAULT_RUNTIME_EVIDENCE_PATH)
)


class FakeDevice:
    def __init__(self, kind: str, index: int | None = None) -> None:
        self.type = kind
        self.index = index

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeTensor:
    cpu_shape_override = None

    def __init__(self, shape, *, dtype, device, log, timeline=None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.log = log
        self.timeline = None if timeline is None else tuple(timeline)

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and args:
            dtype = args[0]
        self.log.append(("to", dtype))
        return FakeTensor(
            self.shape,
            dtype=dtype if dtype is not None else self.dtype,
            device=self.device,
            log=self.log,
            timeline=self.timeline,
        )

    def __getitem__(self, key):
        self.log.append(("getitem", key))
        if not isinstance(key, tuple):
            key = (key,)
        shape = list(self.shape)
        output = []
        for axis, selector in enumerate(key):
            size = shape[axis]
            if isinstance(selector, int):
                continue
            if isinstance(selector, slice):
                start, stop, step = selector.indices(size)
                output.append(len(range(start, stop, step)))
                continue
            raise AssertionError(f"unsupported fake selector: {selector!r}")
        output.extend(shape[len(key) :])
        timeline = self.timeline
        if timeline is not None and len(self.shape) >= 2:
            temporal_selector = key[1] if len(key) > 1 else slice(None)
            if isinstance(temporal_selector, slice):
                timeline = timeline[temporal_selector]
        return FakeTensor(
            output,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            timeline=timeline,
        )

    def clamp(self, minimum, maximum):
        self.log.append(("clamp", minimum, maximum))
        return self

    def mul(self, value):
        self.log.append(("mul", value))
        return self

    def round(self):
        self.log.append(("round",))
        return self

    def cpu(self):
        self.log.append(("cpu",))
        return FakeTensor(
            self.cpu_shape_override or self.shape,
            dtype=self.dtype,
            device=FakeDevice("cpu"),
            log=self.log,
            timeline=self.timeline,
        )

    def record_stream(self, stream):
        self.log.append(("tensor-record-stream", stream))


class FakeEvent:
    def __init__(self, log) -> None:
        self.log = log

    def record(self, stream=None) -> None:
        self.log.append(("event-record", stream))

    def synchronize(self) -> None:
        self.log.append(("event-sync",))


class FakeCuda:
    def __init__(self, log) -> None:
        self.log = log

    def Event(self, **kwargs):
        self.log.append(("event-create", kwargs))
        return FakeEvent(self.log)

    def Stream(self, *, device):
        self.log.append(("stream-create", str(device)))
        return FakeStream(self.log)

    def device(self, device):
        return FakeContext(self.log, "device", str(device))

    def stream(self, stream):
        return FakeContext(self.log, "stream", stream)


class FakeContext:
    def __init__(self, log, kind, value) -> None:
        self.log = log
        self.kind = kind
        self.value = value

    def __enter__(self):
        self.log.append((f"{self.kind}-enter", self.value))
        return self.value

    def __exit__(self, exc_type, exc, traceback):
        self.log.append((f"{self.kind}-exit", self.value))
        return False


class FakeStream:
    def __init__(self, log) -> None:
        self.log = log

    def wait_event(self, event) -> None:
        self.log.append(("stream-wait-event", event))


class FakeReadyEvent:
    pass


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    uint8 = "uint8"

    def __init__(self, log) -> None:
        self.cuda = FakeCuda(log)
        self.log = log

    def cat(self, tensors, dim):
        self.log.append(("cat", dim, tuple(tensor.shape for tensor in tensors)))
        timelines = tuple(tensor.timeline for tensor in tensors)
        self.log.append(("cat-timeline", timelines))
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        return FakeTensor(
            shape,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
            log=self.log,
            timeline=tuple(
                item
                for timeline in timelines
                for item in (() if timeline is None else timeline)
            ),
        )


class FakeTAEHV:
    def __init__(self, log) -> None:
        self.log = log
        self.decode_latent_counts = []
        self.decode_timelines = []
        self.output_dtype = "float16"
        self.output_device = None
        self.error = None

    def decode_video(self, value, *, parallel, show_progress_bar):
        if self.error is not None:
            raise self.error
        self.log.append(("decode", value.dtype, parallel, show_progress_bar))
        self.decode_latent_counts.append(value.shape[1])
        self.decode_timelines.append(value.timeline)
        return FakeTensor(
            (1, value.shape[1] * 4, 3, 480, 832),
            dtype=self.output_dtype,
            device=self.output_device or value.device,
            log=self.log,
            timeline=tuple(
                (latent_id, frame_index)
                for latent_id in value.timeline
                for frame_index in range(4)
            ),
        )


class RollingTaehvChunkDecoderTests(unittest.TestCase):
    def provenance(self, **changes):
        lock = load_asset_lock_snapshot(DEFAULT_LOCK_PATH).parsed()
        value = CF1BootstrapProvenance(
            stack_id=CF1_STACK_ID,
            source_commit=CF1_SOURCE_COMMIT,
            asset_lock_sha256=CF1_ASSET_LOCK_SHA256,
            runtime_lock_sha256=CF1_RUNTIME_LOCK_SHA256,
            runtime_evidence_sha256=(
                "8209043b4ebecc85f0e844f9c040b54f"
                "c1685104fe9e0b361ce9ee6d060b0c6c"
            ),
            static_environment_sha256="c" * 64,
            runtime_environment_sha256="e" * 64,
            runtime_native_environment_sha256=(
                _LOCKED_NATIVE_IDENTITIES.runtime_environment_sha256
            ),
            native_identity_sha256=(
                _LOCKED_NATIVE_IDENTITIES.native_identity_sha256
            ),
            attention_probe_identity_sha256="3" * 64,
            effective_config_sha256=CF1_EFFECTIVE_CONFIG_SHA256,
            tokenizer_sentinel_sha256=CF1_TOKENIZER_SENTINEL_SHA256,
            attention_backend="flash-attention-2",
            guard_bundle_sha256=_cf1_guard_bundle_sha256(),
            bootstrap_identity_sha256="0" * 64,
            assets=tuple(
                CF1AssetIdentity(
                    id=asset["id"],
                    relative_path=asset["relative_path"],
                    size_bytes=asset["size_bytes"],
                    sha256=asset["sha256"],
                )
                for asset in lock["assets"]
            ),
        )
        value = replace(value, **changes)
        return replace(
            value,
            bootstrap_identity_sha256=_provenance_identity_sha256(value),
        )

    def make_decoder(self):
        log = []
        torch = FakeTorch(log)
        taehv = FakeTAEHV(log)
        runtime = CF1Runtime(
            pipeline=object(),
            taehv=taehv,
            effective_config=object(),
            effective_config_sha256=CF1_EFFECTIVE_CONFIG_SHA256,
            device=FakeDevice("cuda", 0),
            torch=torch,
            attention_backend="flash-attention-2",
            runtime_identity=RuntimePreflightIdentity(
                runtime_id=CF1_RUNTIME_ID,
                runtime_lock_sha256=CF1_RUNTIME_LOCK_SHA256,
                runtime_evidence_sha256=(
                    "8209043b4ebecc85f0e844f9c040b54f"
                    "c1685104fe9e0b361ce9ee6d060b0c6c"
                ),
                static_environment_sha256="c" * 64,
                environment_sha256="e" * 64,
                effective_host_headroom_bytes=96 * 1024**3,
                gpu_total_bytes=80_000_000_000,
                gpu_free_bytes=64 * 1024**3,
            ),
            runtime_native_environment_sha256=(
                _LOCKED_NATIVE_IDENTITIES.runtime_environment_sha256
            ),
            native_identity_sha256=(
                _LOCKED_NATIVE_IDENTITIES.native_identity_sha256
            ),
            attention_probe_identity_sha256="3" * 64,
            tokenizer_sentinel_sha256=CF1_TOKENIZER_SENTINEL_SHA256,
            provenance=self.provenance(),
        )

        def encode_frames(frames):
            log.append(("encode", frames.shape, frames.dtype, str(frames.device)))
            return tuple(
                b"\xff\xd8" + f"frame-{index}".encode("ascii") + b"\xff\xd9"
                for index in range(frames.shape[0])
            )

        decoder = RollingTaehvChunkDecoder(
            runtime=runtime,
            torch=torch,
            encode_frames=encode_frames,
            frame_media_type="image/jpeg",
        )
        return decoder, taehv, log

    def decode_one(self, decoder, latent):
        return decoder.decode(latent, latent_ready_event=FakeReadyEvent())

    def latent(
        self,
        log,
        *,
        shape=(1, 1, 16, 60, 104),
        device=None,
        dtype="bfloat16",
        latent_id=0,
    ):
        return FakeTensor(
            shape,
            dtype=dtype,
            device=device or FakeDevice("cuda", 0),
            log=log,
            timeline=(latent_id,),
        )

    def test_exact_21_chunk_fp16_context_trim_and_release_contract(self) -> None:
        decoder, taehv, log = self.make_decoder()

        chunks = [
            self.decode_one(decoder, self.latent(log, latent_id=latent_id))
            for latent_id in range(21)
        ]

        self.assertEqual([chunk.frame_count for chunk in chunks], [1] + [4] * 20)
        self.assertTrue(all(chunk.frame_media_type == "image/jpeg" for chunk in chunks))
        self.assertEqual(taehv.decode_latent_counts, [1, 2, 3] + [4] * 18)
        self.assertEqual(
            taehv.decode_timelines,
            [(0,), (0, 1), (0, 1, 2)]
            + [tuple(range(max(0, index - 3), index + 1)) for index in range(3, 21)],
        )
        self.assertEqual(decoder.trim_history, (3, 4, 8) + (12,) * 18)
        self.assertEqual(
            [entry for entry in log if entry[:1] == ("decode",)],
            [("decode", "float16", True, False)] * 21,
        )
        self.assertEqual(
            len([entry for entry in log if entry == ("to", "float16")]),
            21,
        )
        tail_slices = [
            entry
            for entry in log
            if entry[0] == "getitem"
            and isinstance(entry[1], tuple)
            and len(entry[1]) == 2
            and entry[1][1] == slice(-3, None)
        ]
        trim_slices = [
            entry[1][1].start
            for entry in log
            if entry[0] == "getitem"
            and isinstance(entry[1], tuple)
            and len(entry[1]) == 2
            and isinstance(entry[1][1], slice)
            and entry[1][1].stop is None
            and entry[1][1].start in {3, 4, 8, 12}
        ]
        self.assertEqual(len(tail_slices), 21)
        self.assertEqual(trim_slices, [3, 4, 8] + [12] * 18)
        self.assertTrue(decoder.complete)
        decoder.finish()
        self.assertIsNone(decoder._tail)
        with self.assertRaisesRegex(CudaSessionError, "complete"):
            self.decode_one(decoder, self.latent(log))

    def test_sync_precedes_d2h_and_encoding_for_each_chunk(self) -> None:
        decoder, _taehv, log = self.make_decoder()

        chunk = self.decode_one(decoder, self.latent(log))

        self.assertEqual(chunk.frame_payloads, (b"\xff\xd8frame-0\xff\xd9",))
        wait_index = next(
            index for index, entry in enumerate(log) if entry[:1] == ("stream-wait-event",)
        )
        decode_index = next(
            index for index, entry in enumerate(log) if entry[:1] == ("decode",)
        )
        uint8_index = log.index(("to", "uint8"))
        record_index = next(
            index for index, entry in enumerate(log) if entry[:1] == ("event-record",)
        )
        sync_index = log.index(("event-sync",))
        cpu_index = log.index(("cpu",))
        encode_index = next(
            index for index, entry in enumerate(log) if entry[:1] == ("encode",)
        )
        self.assertLess(wait_index, decode_index)
        self.assertLess(log.index(("device-enter", "cuda:0")), decode_index)
        self.assertLess(
            next(index for index, entry in enumerate(log) if entry[:1] == ("stream-enter",)),
            decode_index,
        )
        self.assertLess(
            decode_index,
            next(index for index, entry in enumerate(log) if entry[:1] == ("stream-exit",)),
        )
        self.assertLess(decode_index, log.index(("device-exit", "cuda:0")))
        self.assertLess(uint8_index, record_index)
        self.assertLess(record_index, sync_index)
        self.assertLess(sync_index, cpu_index)
        self.assertLess(cpu_index, encode_index)
        self.assertIn(("clamp", 0, 1), log)
        self.assertIn(("event-create", {"blocking": True}), log)
        self.assertTrue(
            any(entry[:1] == ("tensor-record-stream",) for entry in log)
        )

    def test_shape_device_and_provenance_fail_before_decode(self) -> None:
        decoder, taehv, log = self.make_decoder()
        verified_runtime = decoder.runtime
        for latent in (
            self.latent(log, shape=(1, 2, 16, 60, 104)),
            self.latent(log, device=FakeDevice("cuda", 1)),
            self.latent(log, dtype="float32"),
        ):
            with self.assertRaises(CudaSessionError):
                self.decode_one(decoder, latent)
            self.assertFalse(decoder.poisoned)
        self.assertEqual(taehv.decode_latent_counts, [])

        runtime = CF1Runtime(
            pipeline=object(),
            taehv=taehv,
            effective_config=object(),
            effective_config_sha256=CF1_EFFECTIVE_CONFIG_SHA256,
            device=FakeDevice("cuda", 0),
            torch=FakeTorch(log),
            attention_backend="flash-attention-2",
            provenance=None,
        )
        with self.assertRaisesRegex(CudaSessionError, "provenance"):
            RollingTaehvChunkDecoder(
                runtime=runtime,
                torch=FakeTorch(log),
                encode_frames=lambda _frames: (),
            )

        for forged in (
            object(),
            replace(
                verified_runtime,
                provenance=self.provenance(stack_id="other"),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(asset_lock_sha256="0" * 64),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(effective_config_sha256="0" * 64),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(source_commit="wrong-source"),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(guard_bundle_sha256="c" * 64),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(runtime_lock_sha256="d" * 64),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(
                    tokenizer_sentinel_sha256="d" * 64
                ),
            ),
            replace(
                verified_runtime,
                tokenizer_sentinel_sha256="d" * 64,
            ),
            replace(
                verified_runtime,
                attention_backend="torch-sdpa",
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(attention_backend="torch-sdpa"),
            ),
            replace(
                verified_runtime,
                provenance=self.provenance(
                    runtime_environment_sha256="d" * 64
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    runtime_id="other-runtime",
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    runtime_lock_sha256="d" * 64,
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    environment_sha256="d" * 64,
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    effective_host_headroom_bytes=56 * 1024**3 - 1,
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    gpu_total_bytes=80_000_000_000 - 1,
                ),
            ),
            replace(
                verified_runtime,
                runtime_identity=replace(
                    verified_runtime.runtime_identity,
                    gpu_free_bytes=36 * 1024**3 - 1,
                ),
            ),
        ):
            with self.assertRaisesRegex(CudaSessionError, "identity|runtime"):
                RollingTaehvChunkDecoder(
                    runtime=forged,
                    torch=verified_runtime.torch,
                    encode_frames=lambda _frames: (),
                )

        with self.assertRaisesRegex(CudaSessionError, "Torch binding"):
            RollingTaehvChunkDecoder(
                runtime=verified_runtime,
                torch=FakeTorch(log),
                encode_frames=lambda _frames: (),
            )

    def test_constructor_guards_and_all_renderable_signatures(self) -> None:
        decoder, _taehv, log = self.make_decoder()
        runtime = decoder.runtime
        torch = runtime.torch
        for kwargs, message in (
            ({"encode_frames": None}, "callable"),
            ({"frame_media_type": "application/octet-stream"}, "renderable"),
            ({"max_chunk_bytes": True}, "max_chunk_bytes"),
            ({"max_chunk_bytes": 0}, "max_chunk_bytes"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                CudaSessionError, message
            ):
                arguments = dict(
                    runtime=runtime,
                    torch=torch,
                    encode_frames=lambda _frames: (),
                )
                arguments.update(kwargs)
                RollingTaehvChunkDecoder(**arguments)

        valid = {
            "image/jpeg": b"\xff\xd8body\xff\xd9",
            "image/png": PNG_832X480,
            "image/webp": b"RIFF\x04\x00\x00\x00WEBP",
        }
        for media_type, payload in valid.items():
            with self.subTest(media_type=media_type):
                self.assertTrue(_payload_matches_media_type(payload, media_type))
                self.assertFalse(
                    _payload_matches_media_type(b"not-renderable", media_type)
                )
        self.assertFalse(_payload_matches_media_type(PNG_1X1, "image/png"))

    def test_context_count_must_match_block_index(self) -> None:
        decoder, _taehv, log = self.make_decoder()
        decoder._block_index = 3
        decoder._tail = self.latent(log, latent_id=1).to(dtype="float16")

        with self.assertRaisesRegex(CudaSessionError, "context"):
            self.decode_one(decoder, self.latent(log, latent_id=3))
        self.assertTrue(decoder.poisoned)

    def test_cpu_shape_and_chunk_byte_limit_fail_closed(self) -> None:
        decoder, _taehv, log = self.make_decoder()
        decoder.max_chunk_bytes = 3
        with self.assertRaisesRegex(CudaSessionError, "byte limit"):
            self.decode_one(decoder, self.latent(log))
        self.assertTrue(decoder.poisoned)

        decoder, _taehv, log = self.make_decoder()
        FakeTensor.cpu_shape_override = (2, 3, 480, 832)
        try:
            with self.assertRaisesRegex(CudaSessionError, "ownership"):
                self.decode_one(decoder, self.latent(log))
        finally:
            FakeTensor.cpu_shape_override = None
        self.assertTrue(decoder.poisoned)

        decoder, _taehv, log = self.make_decoder()
        decoder.encode_frames = lambda _frames: b"\xff\xd8whole-batch\xff\xd9"
        with self.assertRaisesRegex(CudaSessionError, "invalid batch"):
            self.decode_one(decoder, self.latent(log))
        self.assertTrue(decoder.poisoned)

    def test_encoding_failure_permanently_poison_decoder(self) -> None:
        decoder, _taehv, log = self.make_decoder()
        decoder.encode_frames = lambda _frames: (b"",)

        with self.assertRaisesRegex(CudaSessionError, "encoded frame"):
            self.decode_one(decoder, self.latent(log))
        self.assertTrue(decoder.poisoned)
        with self.assertRaisesRegex(CudaSessionError, "poisoned"):
            self.decode_one(decoder, self.latent(log))

    def test_decode_output_device_dtype_and_payload_signature_fail_closed(self) -> None:
        cases = (
            ("float32", FakeDevice("cuda", 0), None, "dtype"),
            ("float16", FakeDevice("cpu"), None, "device"),
            ("float16", FakeDevice("cuda", 0), (b"not-a-jpeg",), "payload"),
        )
        for dtype, device, encoded, message in cases:
            with self.subTest(message=message):
                decoder, taehv, log = self.make_decoder()
                taehv.output_dtype = dtype
                taehv.output_device = device
                if encoded is not None:
                    decoder.encode_frames = lambda _frames, value=encoded: value
                with self.assertRaisesRegex(CudaSessionError, message):
                    self.decode_one(decoder, self.latent(log))
                self.assertTrue(decoder.poisoned)

    def test_decode_and_encoder_exceptions_poison_and_finish_requires_21(self) -> None:
        decoder, taehv, log = self.make_decoder()
        with self.assertRaisesRegex(CudaSessionError, "incomplete"):
            decoder.finish()
        self.assertFalse(decoder.poisoned)

        taehv.error = RuntimeError("decoder detail must not be normalized here")
        with self.assertRaises(RuntimeError):
            self.decode_one(decoder, self.latent(log))
        self.assertTrue(decoder.poisoned)

    def test_base_exception_during_decode_permanently_poisons_decoder(self) -> None:
        decoder, taehv, log = self.make_decoder()
        taehv.error = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.decode_one(decoder, self.latent(log))

        self.assertTrue(decoder.poisoned)
        with self.assertRaisesRegex(CudaSessionError, "poisoned"):
            self.decode_one(decoder, self.latent(log))

        decoder, _taehv, log = self.make_decoder()

        def fail_encoder(_frames):
            raise RuntimeError("encoder failed")

        decoder.encode_frames = fail_encoder
        with self.assertRaises(RuntimeError):
            self.decode_one(decoder, self.latent(log))
        self.assertTrue(decoder.poisoned)


if __name__ == "__main__":
    unittest.main()
