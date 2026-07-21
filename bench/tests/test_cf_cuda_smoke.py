from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.cf_cuda_smoke import (
    CudaSmokeError,
    build_png_encoder,
    cf1_cuda_smoke_report,
    run_cf1_cuda_smoke,
)
from bench.streaming_service import DecodedChunk


PNG_PREFIX = b"\x89PNG\r\n\x1a\n"
PNG_SUFFIX = b"\x00\x00\x00\x00IEND\xaeB`\x82"
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
        PNG_PREFIX
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline * height, level=1))
        + _png_chunk(b"IEND", b"")
    )


PNG_832X480 = _rgb_png(832, 480)


class FakeArray:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def tobytes(self) -> bytes:
        return self.payload


class FakeEncoded:
    def __init__(self, payload: bytes, torch) -> None:
        self.payload = payload
        self.shape = (len(payload),)
        self.dtype = torch.uint8
        self.device = SimpleNamespace(type="cpu")

    def numpy(self):
        return FakeArray(self.payload)


class FakeFrame:
    def __init__(self, index, torch) -> None:
        self.index = index
        self.shape = (3, 480, 832)
        self.dtype = torch.uint8
        self.device = SimpleNamespace(type="cpu")

    def contiguous(self):
        return self


class FakeFrames:
    def __init__(self, count, torch) -> None:
        self.shape = (count, 3, 480, 832)
        self.dtype = torch.uint8
        self.device = SimpleNamespace(type="cpu")
        self.torch = torch

    def __getitem__(self, index):
        return FakeFrame(index, self.torch)


class FakeSession:
    def __init__(self, *, runtime, prompt, seed, log) -> None:
        self.runtime = runtime
        self.prompt = prompt
        self.seed = seed
        self.log = log
        self.index = 0
        self.finished = False
        log.append(("session-init", prompt, seed))

    @property
    def complete(self):
        return self.index == 21

    def pull(self):
        index = self.index
        self.index += 1
        self.log.append(("pull", index))
        return SimpleNamespace(
            block_index=index,
            denoised_latent=f"latent-{index}",
            latent_ready_event=f"event-{index}",
        )

    def finish(self):
        self.log.append(("session-finish",))
        if not self.complete:
            raise AssertionError("incomplete fake session")
        self.finished = True


class FakeDecoder:
    def __init__(self, *, runtime, torch, encode_frames, frame_media_type, log):
        self.log = log
        self.index = 0
        self.finished = False
        self.log.append(("decoder-init", frame_media_type, encode_frames))

    @property
    def complete(self):
        return self.index == 21

    def decode(self, latent, *, latent_ready_event):
        index = self.index
        self.index += 1
        self.log.append(("decode", index, latent, latent_ready_event))
        count = 1 if index == 0 else 4
        payloads = (PNG_832X480,) * count
        return DecodedChunk(payloads, frame_media_type="image/png")

    def finish(self):
        self.log.append(("decoder-finish",))
        if self.index != 21:
            raise AssertionError("incomplete fake decoder")
        self.finished = True


class CudaSmokeTests(unittest.TestCase):
    def runtime(self, log=None):
        if log is None:
            log = []

        def synchronize(device):
            log.append(("runtime-sync", device))

        return SimpleNamespace(
            device="cuda:0",
            torch=SimpleNamespace(
                uint8="uint8",
                cuda=SimpleNamespace(synchronize=synchronize),
            ),
            provenance=SimpleNamespace(
                bootstrap_identity_sha256="b" * 64,
                runtime_environment_sha256="e" * 64,
                guard_bundle_sha256="g" * 64,
            ),
        )

    def run_with_fakes(self, output, blocks):
        log = []
        runtime = self.runtime(log)

        def session_factory(**kwargs):
            return FakeSession(**kwargs, log=log)

        def decoder_factory(**kwargs):
            return FakeDecoder(**kwargs, log=log)

        with mock.patch(
            "bench.cf_cuda_smoke.build_cf1_runtime",
            return_value=runtime,
        ), mock.patch(
            "bench.cf_cuda_smoke.build_png_encoder",
            return_value=lambda _frames: (),
        ), mock.patch(
            "bench.cf_cuda_smoke.CF1LatentPullSession",
            side_effect=session_factory,
        ), mock.patch(
            "bench.cf_cuda_smoke.RollingTaehvChunkDecoder",
            side_effect=decoder_factory,
        ):
            report = run_cf1_cuda_smoke(
                prompt="A red fox runs.",
                seed=7,
                blocks=blocks,
                output_directory=output,
            )
        return report, log

    def test_one_block_smoke_writes_one_frame_and_marks_runtime_nonreusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first-chunk"
            report, log = self.run_with_fakes(output, 1)

            self.assertEqual(report["status"], "bounded-first-chunk")
            self.assertEqual(report["block_count"], 1)
            self.assertEqual(report["frame_count"], 1)
            self.assertFalse(report["runtime_reusable"])
            self.assertEqual(report["chunk_frame_counts"], [1])
            self.assertTrue((output / "frame-000000.png").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest, report)
            self.assertNotIn("A red fox", json.dumps(report))
            self.assertNotIn(("decoder-finish",), log)
            self.assertNotIn(("session-finish",), log)

    def test_full_smoke_writes_81_frames_and_finishes_decoder_before_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "complete"
            report, log = self.run_with_fakes(output, 21)

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["frame_count"], 81)
            self.assertEqual(report["chunk_frame_counts"], [1] + [4] * 20)
            self.assertTrue(report["runtime_reusable"])
            self.assertEqual(report["timing"]["stream_mode"], "serial")
            self.assertIn("first_chunk_encoded_s", report["timing"])
            self.assertIn("first_frame_written_s", report["timing"])
            self.assertNotIn("first_frame_persisted_s", report["timing"])
            self.assertNotIn("first_visible_s", report["timing"])
            self.assertEqual(len(list(output.glob("frame-*.png"))), 81)
            self.assertLess(log.index(("decoder-finish",)), log.index(("session-finish",)))
            self.assertEqual(report["bootstrap_identity_sha256"], "b" * 64)

    def test_timing_starts_after_runtime_sync_before_session_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timed"
            log = []
            runtime = self.runtime(log)
            clock_value = 0

            def perf_counter():
                nonlocal clock_value
                clock_value += 1
                log.append(("clock", clock_value))
                return float(clock_value)

            with mock.patch(
                "bench.cf_cuda_smoke.build_cf1_runtime",
                return_value=runtime,
            ), mock.patch(
                "bench.cf_cuda_smoke.build_png_encoder",
                return_value=lambda _frames: (),
            ), mock.patch(
                "bench.cf_cuda_smoke.CF1LatentPullSession",
                side_effect=lambda **kwargs: FakeSession(**kwargs, log=log),
            ), mock.patch(
                "bench.cf_cuda_smoke.RollingTaehvChunkDecoder",
                side_effect=lambda **kwargs: FakeDecoder(**kwargs, log=log),
            ), mock.patch(
                "bench.cf_cuda_smoke.time.perf_counter",
                side_effect=perf_counter,
            ):
                report = run_cf1_cuda_smoke(
                    prompt="A red fox runs.",
                    seed=7,
                    blocks=1,
                    output_directory=output,
                )

            sync_index = log.index(("runtime-sync", "cuda:0"))
            first_clock_index = next(
                index for index, entry in enumerate(log) if entry[0] == "clock"
            )
            session_index = log.index(("session-init", "A red fox runs.", 7))
            self.assertLess(sync_index, first_clock_index)
            self.assertLess(first_clock_index, session_index)
            timing = report["timing"]
            self.assertEqual(
                timing["origin"],
                "after_runtime_bootstrap_sync_before_session_initialization",
            )
            self.assertEqual(timing["clock"], "time.perf_counter")
            self.assertIn(
                "session_initialization",
                timing["first_chunk_encoded_includes"],
            )
            self.assertIn(
                "decoder_construction",
                timing["first_chunk_encoded_includes"],
            )
            self.assertEqual(
                timing["excludes"],
                [
                    "runtime_bootstrap",
                    "model_weight_load",
                    "png_encoder_construction",
                    "output_directory_creation",
                    "manifest_write",
                ],
            )

    def test_invalid_blocks_and_existing_output_fail_before_runtime_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for blocks in (0, 2, 20, 22):
                with self.subTest(blocks=blocks), mock.patch(
                    "bench.cf_cuda_smoke.build_cf1_runtime"
                ) as build, self.assertRaises(CudaSmokeError):
                    run_cf1_cuda_smoke(
                        prompt="prompt",
                        seed=0,
                        blocks=blocks,
                        output_directory=root / f"bad-{blocks}",
                    )
                build.assert_not_called()

            existing = root / "existing"
            existing.mkdir()
            with mock.patch(
                "bench.cf_cuda_smoke.build_cf1_runtime"
            ) as build, self.assertRaisesRegex(CudaSmokeError, "already exists"):
                run_cf1_cuda_smoke(
                    prompt="prompt",
                    seed=0,
                    blocks=1,
                    output_directory=existing,
                )
            build.assert_not_called()

    def test_smoke_report_sanitizes_unexpected_failure(self) -> None:
        with mock.patch(
            "bench.cf_cuda_smoke.run_cf1_cuda_smoke",
            side_effect=RuntimeError("sensitive detail"),
        ):
            report = cf1_cuda_smoke_report(
                prompt="prompt",
                seed=0,
                blocks=1,
                output_directory=Path("unused"),
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["failure"], "unexpected smoke error: RuntimeError")
        self.assertNotIn("sensitive", json.dumps(report))

    def test_post_frame_failure_leaves_an_explicit_failed_manifest(self) -> None:
        runtime = self.runtime()
        log = []

        class FailingDecoder(FakeDecoder):
            def decode(self, latent, *, latent_ready_event):
                if self.index == 1:
                    raise RuntimeError("synthetic decode failure")
                return super().decode(
                    latent,
                    latent_ready_event=latent_ready_event,
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with mock.patch(
                "bench.cf_cuda_smoke.build_cf1_runtime",
                return_value=runtime,
            ), mock.patch(
                "bench.cf_cuda_smoke.build_png_encoder",
                return_value=lambda _frames: (),
            ), mock.patch(
                "bench.cf_cuda_smoke.CF1LatentPullSession",
                side_effect=lambda **kwargs: FakeSession(**kwargs, log=log),
            ), mock.patch(
                "bench.cf_cuda_smoke.RollingTaehvChunkDecoder",
                side_effect=lambda **kwargs: FailingDecoder(**kwargs, log=log),
            ):
                report = cf1_cuda_smoke_report(
                    prompt="A red fox runs.",
                    seed=7,
                    blocks=21,
                    output_directory=output,
                )

            self.assertFalse(report["ready"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["runtime_reusable"])
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["failure_type"], "RuntimeError")
            self.assertEqual(manifest["bootstrap_identity_sha256"], "b" * 64)
            self.assertEqual(manifest["runtime_environment_sha256"], "e" * 64)
            self.assertEqual(manifest["guard_bundle_sha256"], "g" * 64)
            self.assertNotIn("synthetic decode failure", json.dumps(manifest))

    def test_keyboard_interrupt_after_a_frame_still_writes_failed_manifest(self) -> None:
        runtime = self.runtime()
        log = []

        class InterruptedDecoder(FakeDecoder):
            def decode(self, latent, *, latent_ready_event):
                if self.index == 1:
                    raise KeyboardInterrupt
                return super().decode(
                    latent,
                    latent_ready_event=latent_ready_event,
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted"
            with mock.patch(
                "bench.cf_cuda_smoke.build_cf1_runtime",
                return_value=runtime,
            ), mock.patch(
                "bench.cf_cuda_smoke.build_png_encoder",
                return_value=lambda _frames: (),
            ), mock.patch(
                "bench.cf_cuda_smoke.CF1LatentPullSession",
                side_effect=lambda **kwargs: FakeSession(**kwargs, log=log),
            ), mock.patch(
                "bench.cf_cuda_smoke.RollingTaehvChunkDecoder",
                side_effect=lambda **kwargs: InterruptedDecoder(**kwargs, log=log),
            ), self.assertRaises(KeyboardInterrupt):
                run_cf1_cuda_smoke(
                    prompt="A red fox runs.",
                    seed=7,
                    blocks=21,
                    output_directory=output,
                )

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["failure_type"], "KeyboardInterrupt")
            self.assertEqual(manifest["frame_count"], 1)

    def test_png_encoder_requires_same_torch_and_returns_real_png_bytes(self) -> None:
        log = []
        torch = SimpleNamespace(uint8="uint8")
        runtime = SimpleNamespace(torch=torch)

        def encode_png(frame, *, compression_level):
            log.append((frame.index, compression_level))
            return FakeEncoded(PNG_832X480, torch)

        modules = {
            "torch": torch,
            "torchvision": SimpleNamespace(__version__="0.23.0+cu128"),
            "torchvision.io": SimpleNamespace(encode_png=encode_png),
        }
        with mock.patch(
            "bench.cf_cuda_smoke._require_verified_runtime",
            side_effect=lambda value: value,
        ), mock.patch(
            "bench.cf_cuda_smoke._expected_torchvision_version",
            return_value="0.23.0+cu128",
        ), mock.patch(
            "bench.cf_cuda_smoke.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            encoder = build_png_encoder(runtime)
            payloads = encoder(FakeFrames(4, torch))

        self.assertEqual(log, [(0, 1), (1, 1), (2, 1), (3, 1)])
        self.assertEqual(len(payloads), 4)
        self.assertTrue(all(payload.startswith(PNG_PREFIX) for payload in payloads))
        self.assertTrue(all(payload.endswith(PNG_SUFFIX) for payload in payloads))

    def test_png_encoder_rejects_valid_png_with_wrong_dimensions(self) -> None:
        torch = SimpleNamespace(uint8="uint8")
        runtime = SimpleNamespace(torch=torch)

        def encode_png(_frame, *, compression_level):
            self.assertEqual(compression_level, 1)
            return FakeEncoded(PNG_1X1, torch)

        modules = {
            "torch": torch,
            "torchvision": SimpleNamespace(__version__="0.23.0+cu128"),
            "torchvision.io": SimpleNamespace(encode_png=encode_png),
        }
        with mock.patch(
            "bench.cf_cuda_smoke._require_verified_runtime",
            side_effect=lambda value: value,
        ), mock.patch(
            "bench.cf_cuda_smoke._expected_torchvision_version",
            return_value="0.23.0+cu128",
        ), mock.patch(
            "bench.cf_cuda_smoke.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            encoder = build_png_encoder(runtime)
            with self.assertRaisesRegex(CudaSmokeError, "PNG payload"):
                encoder(FakeFrames(1, torch))


if __name__ == "__main__":
    unittest.main()
