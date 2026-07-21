from __future__ import annotations

import binascii
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from bench.cf_video_artifact import (
    ArtifactAssemblyError,
    ToolIdentity,
    _canonical_ffmpeg_arguments,
    _decode_media,
    _encode_once,
    _probe_media,
    _publish_directory_noreplace,
    _sanitized_subprocess_environment,
    assemble_cf1_video_artifact,
    validate_cf_video_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLE_CLI = PROJECT_ROOT / "scripts" / "cf-video-assemble"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def png_bytes(index: int) -> bytes:
    red = index % 256
    green = (index * 3) % 256
    blue = (index * 7) % 256
    row = b"\x00" + bytes((red, green, blue)) * 832
    raw = row * 480
    ihdr = struct.pack(">IIBBBBB", 832, 480, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=1))
        + _png_chunk(b"IEND", b"")
    )


def write_complete_smoke(directory: Path) -> dict:
    directory.mkdir()
    frame_hashes: list[str] = []
    for index in range(81):
        payload = png_bytes(index)
        (directory / f"frame-{index:06d}.png").write_bytes(payload)
        frame_hashes.append(hashlib.sha256(payload).hexdigest())
    manifest = {
        "schema_version": 1,
        "kind": "cf1-cuda-smoke",
        "ready": True,
        "status": "complete",
        "runtime_reusable": True,
        "prompt_sha256": hashlib.sha256(b"A red fox runs through snow.").hexdigest(),
        "seed": 20260719,
        "block_count": 21,
        "chunk_frame_counts": [1] + [4] * 20,
        "frame_count": 81,
        "frame_payload_sha256": frame_hashes,
        "frame_media_type": "image/png",
        "resolution": {"width": 832, "height": 480},
        "fps_contract": 16,
        "timing": {"stream_mode": "serial", "effective_fps": 9999},
        "bootstrap_identity_sha256": "a" * 64,
        "runtime_environment_sha256": "b" * 64,
        "guard_bundle_sha256": "c" * 64,
        "output_directory": "/provider/original/smoke-output",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def exact_probe() -> dict:
    return {
        "width": 832,
        "height": 480,
        "fps": 16.0,
        "decoded_frames": 81,
        "duration_s": 5.0625,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "video_streams": 1,
        "audio_streams": 0,
    }


def tool(name: str) -> ToolIdentity:
    return ToolIdentity(
        resolved_path=Path(f"/verified/{name}"),
        executable_sha256=("d" if name == "ffmpeg" else "e") * 64,
        version_sha256=("f" if name == "ffmpeg" else "1") * 64,
        version_first_line=f"{name} version pinned",
    )


class CFVideoArtifactUnitTests(unittest.TestCase):
    def assemble_with_fakes(self, smoke: Path, output: Path):
        encodes: list[tuple[Path, Path]] = []

        def encode_once(*, frame_directory, output_path, ffmpeg, environment):
            self.assertNotIn("GEMINI_API_KEY", environment)
            self.assertNotIn("TWELVELABS_API_KEY", environment)
            encodes.append((frame_directory, output_path))
            output_path.write_bytes(b"deterministic-mp4-bytes")

        with mock.patch(
            "bench.cf_video_artifact._resolve_tool_identity",
            side_effect=[tool("ffmpeg"), tool("ffprobe")],
        ), mock.patch(
            "bench.cf_video_artifact._encode_once", side_effect=encode_once
        ), mock.patch(
            "bench.cf_video_artifact._probe_media", return_value=exact_probe()
        ), mock.patch(
            "bench.cf_video_artifact._decode_media"
        ):
            report = assemble_cf1_video_artifact(
                smoke_directory=smoke,
                output_directory=output,
            )
        return report, encodes

    def test_rejects_noncomplete_smoke_before_resolving_tools_or_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            manifest = write_complete_smoke(smoke)
            output = root / "artifact"
            for mutation in (
                {"ready": False},
                {"status": "bounded-first-chunk"},
                {"runtime_reusable": False},
                {"block_count": 1},
            ):
                with self.subTest(mutation=mutation):
                    changed = dict(manifest)
                    changed.update(mutation)
                    (smoke / "manifest.json").write_text(json.dumps(changed))
                    with mock.patch(
                        "bench.cf_video_artifact._resolve_tool_identity"
                    ) as resolve, self.assertRaises(ArtifactAssemblyError):
                        assemble_cf1_video_artifact(
                            smoke_directory=smoke,
                            output_directory=output,
                        )
                    resolve.assert_not_called()
                    self.assertFalse(output.exists())

    def test_rejects_missing_extra_symlinked_or_hash_drifted_frames(self) -> None:
        mutations = ("missing", "extra", "symlink", "hash")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                smoke = root / "smoke"
                write_complete_smoke(smoke)
                if mutation == "missing":
                    (smoke / "frame-000080.png").unlink()
                elif mutation == "extra":
                    (smoke / "frame-000081.png").write_bytes(png_bytes(81))
                elif mutation == "symlink":
                    target = smoke / "frame-000080.png"
                    target.unlink()
                    target.symlink_to(smoke / "frame-000079.png")
                else:
                    (smoke / "frame-000080.png").write_bytes(png_bytes(200))

                output = root / "artifact"
                with mock.patch(
                    "bench.cf_video_artifact._resolve_tool_identity"
                ) as resolve, self.assertRaises(ArtifactAssemblyError):
                    assemble_cf1_video_artifact(
                        smoke_directory=smoke,
                        output_directory=output,
                    )
                resolve.assert_not_called()
                self.assertFalse(output.exists())

    def test_success_double_encodes_then_atomically_publishes_hash_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            source = write_complete_smoke(smoke)
            output = root / "artifact"

            report, encodes = self.assemble_with_fakes(smoke, output)

            self.assertEqual(len(encodes), 2)
            self.assertEqual(report["kind"], "cf1-development-video-artifact")
            self.assertEqual(
                report["purpose"],
                "development-video-understanding-not-gate-evidence",
            )
            self.assertFalse(report["authorizes_quality_claim"])
            self.assertFalse(report["authorizes_performance_claim"])
            self.assertEqual(report["status"], "assembled-and-verified")
            self.assertEqual(report["source"]["generation_prompt_sha256"], source["prompt_sha256"])
            self.assertEqual(report["source"]["frame_payload_sha256"], source["frame_payload_sha256"])
            self.assertNotIn("timing", report["source"])
            self.assertNotIn("effective_fps", json.dumps(report))
            self.assertTrue(report["encoding"]["repeat_encode_byte_equal"])
            self.assertEqual(
                report["encoding"]["repeat_encode_sha256"],
                report["media"]["media_sha256"],
            )
            self.assertEqual((output / "video.mp4").read_bytes(), b"deterministic-mp4-bytes")
            self.assertEqual(
                json.loads((output / "manifest.json").read_text()), report
            )
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"manifest.json", "video.mp4"}
            )
            self.assertFalse(any(path.name.startswith(f".{output.name}.") for path in root.iterdir()))

    def test_repeat_encode_mismatch_or_probe_drift_never_publishes_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            write_complete_smoke(smoke)

            for mode in ("repeat-mismatch", "probe-mismatch"):
                output = root / mode
                writes = 0

                def encode_once(*, output_path, **_kwargs):
                    nonlocal writes
                    writes += 1
                    suffix = writes if mode == "repeat-mismatch" else 1
                    output_path.write_bytes(f"video-{suffix}".encode())

                probe = exact_probe()
                if mode == "probe-mismatch":
                    probe["decoded_frames"] = 80
                with mock.patch(
                    "bench.cf_video_artifact._resolve_tool_identity",
                    side_effect=[tool("ffmpeg"), tool("ffprobe")],
                ), mock.patch(
                    "bench.cf_video_artifact._encode_once", side_effect=encode_once
                ), mock.patch(
                    "bench.cf_video_artifact._probe_media", return_value=probe
                ), self.assertRaises(ArtifactAssemblyError):
                    assemble_cf1_video_artifact(
                        smoke_directory=smoke,
                        output_directory=output,
                    )
                self.assertFalse(output.exists())
                self.assertFalse(any(path.name.startswith(f".{output.name}.") for path in root.iterdir()))

    def test_no_replace_publication_preserves_a_destination_that_wins_the_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            output = root / "artifact"
            stage.mkdir()
            (stage / "staged").write_text("new")
            output.mkdir()
            (output / "winner").write_text("existing")

            with self.assertRaisesRegex(ArtifactAssemblyError, "appeared"):
                _publish_directory_noreplace(stage, output)

            self.assertEqual((output / "winner").read_text(), "existing")
            self.assertEqual((stage / "staged").read_text(), "new")

    def test_ffmpeg_arguments_and_subprocess_environment_are_exact_and_secret_free(self) -> None:
        self.assertEqual(
            _canonical_ffmpeg_arguments(),
            [
                "<FFMPEG>",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                "16",
                "-start_number",
                "0",
                "-i",
                "<FRAME_DIRECTORY>/frame-%06d.png",
                "-map",
                "0:v:0",
                "-frames:v",
                "81",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "16",
                "-fps_mode",
                "cfr",
                "-threads",
                "1",
                "-x264-params",
                "threads=1:lookahead_threads=1:sliced_threads=0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-movflags",
                "+faststart",
                "-video_track_timescale",
                "16000",
                "-y",
                "<OUTPUT>",
            ],
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {
                "PATH": "/secret/bin",
                "GEMINI_API_KEY": "gemini-secret",
                "TWELVELABS_API_KEY": "twelve-secret",
                "UNRELATED_SECRET": "also-secret",
            },
            clear=True,
        ):
            environment = _sanitized_subprocess_environment(Path(temporary))
        self.assertEqual(
            environment,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "SOURCE_DATE_EPOCH": "0",
                "TMPDIR": str(Path(temporary).resolve()),
                "TZ": "UTC",
            },
        )
        self.assertNotIn("secret", json.dumps(environment).lower())

    def test_ffprobe_decoder_error_on_stderr_fails_even_with_exact_stdout_contract(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 832,
                    "height": 480,
                    "r_frame_rate": "16/1",
                    "nb_read_frames": "81",
                    "duration": "5.062500",
                }
            ],
            "format": {"duration": "5.062500"},
        }
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload).encode(),
            stderr=b"[h264] error while decoding macroblock",
        )
        with mock.patch(
            "bench.cf_video_artifact.subprocess.run", return_value=completed
        ), self.assertRaisesRegex(ArtifactAssemblyError, "FFprobe"):
            _probe_media(
                Path("video.mp4"),
                ffprobe=tool("ffprobe"),
                environment={},
            )

    def test_full_decode_requires_xerror_success_and_empty_output(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=183,
            stdout=b"",
            stderr=b"[h264] corrupt decoded frame",
        )
        with mock.patch(
            "bench.cf_video_artifact.subprocess.run", return_value=completed
        ), self.assertRaisesRegex(ArtifactAssemblyError, "validation decode"):
            _decode_media(
                Path("video.mp4"),
                ffmpeg=tool("ffmpeg"),
                environment={},
            )

    def test_encode_rejects_error_output_even_when_ffmpeg_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "video.mp4"

            def encode_with_error(*_args, **_kwargs):
                output.write_bytes(b"bad-video")
                return subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=b"",
                    stderr=b"decoder error",
                )

            with mock.patch(
                "bench.cf_video_artifact.subprocess.run", side_effect=encode_with_error
            ), self.assertRaisesRegex(ArtifactAssemblyError, "FFmpeg encode"):
                _encode_once(
                    frame_directory=Path(temporary),
                    output_path=output,
                    ffmpeg=tool("ffmpeg"),
                    environment={},
                )

    def test_exported_validator_returns_exact_media_bytes_hash_path_and_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            source = write_complete_smoke(smoke)
            output = root / "artifact"
            self.assemble_with_fakes(smoke, output)

            verified_paths: list[Path] = []

            def verify_private_copy(path: Path, **_kwargs):
                verified_paths.append(path)
                self.assertNotEqual(path, output / "video.mp4")
                self.assertEqual(path.read_bytes(), b"deterministic-mp4-bytes")
                return exact_probe()

            with mock.patch(
                "bench.cf_video_artifact._resolve_tool_identity",
                side_effect=[tool("ffmpeg"), tool("ffprobe")],
            ), mock.patch(
                "bench.cf_video_artifact._probe_media", side_effect=verify_private_copy
            ), mock.patch(
                "bench.cf_video_artifact._decode_media",
                side_effect=lambda path, **_kwargs: verified_paths.append(path),
            ):
                validated = validate_cf_video_artifact(output / "manifest.json")

            self.assertEqual(
                set(validated),
                {
                    "artifact_manifest",
                    "artifact_manifest_sha256",
                    "media_path",
                    "media_sha256",
                    "media_bytes",
                    "generation_prompt_sha256",
                    "fps",
                },
            )
            self.assertEqual(validated["media_path"], (output / "video.mp4").resolve())
            self.assertEqual(validated["media_bytes"], b"deterministic-mp4-bytes")
            self.assertEqual(
                validated["media_sha256"],
                hashlib.sha256(validated["media_bytes"]).hexdigest(),
            )
            self.assertEqual(
                validated["generation_prompt_sha256"], source["prompt_sha256"]
            )
            self.assertEqual(validated["fps"], 16.0)
            self.assertEqual(len(verified_paths), 2)
            self.assertEqual(verified_paths[0], verified_paths[1])

    def test_exported_validator_requires_the_recorded_ffprobe_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            write_complete_smoke(smoke)
            output = root / "artifact"
            self.assemble_with_fakes(smoke, output)

            with mock.patch(
                "bench.cf_video_artifact._resolve_tool_identity",
                side_effect=[tool("ffmpeg"), tool("ffmpeg")],
            ), mock.patch(
                "bench.cf_video_artifact._probe_media", return_value=exact_probe()
            ), mock.patch(
                "bench.cf_video_artifact._decode_media"
            ), self.assertRaisesRegex(ArtifactAssemblyError, "FFprobe identity"):
                validate_cf_video_artifact(output / "manifest.json")

    def test_cli_is_executable_and_requires_nonexisting_output(self) -> None:
        self.assertTrue(os.access(ASSEMBLE_CLI, os.X_OK))
        completed = subprocess.run(
            [str(ASSEMBLE_CLI), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--smoke-dir", completed.stdout)
        self.assertIn("--output", completed.stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class CFVideoArtifactRealSurfaceTests(unittest.TestCase):
    def test_exact_encoder_repeats_bytes_and_meets_the_81_frame_contract(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg is not None
        encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        if encoders.returncode != 0 or "libx264" not in encoders.stdout:
            self.skipTest("libx264 unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            write_complete_smoke(smoke)
            first = assemble_cf1_video_artifact(
                smoke_directory=smoke,
                output_directory=root / "artifact-a",
            )
            second = assemble_cf1_video_artifact(
                smoke_directory=smoke,
                output_directory=root / "artifact-b",
            )

            self.assertEqual(first, second)
            self.assertEqual(
                (root / "artifact-a" / "video.mp4").read_bytes(),
                (root / "artifact-b" / "video.mp4").read_bytes(),
            )
            self.assertEqual(first["media"]["decoded_frames"], 81)
            self.assertEqual(first["media"]["fps"], 16.0)
            self.assertEqual(first["media"]["duration_s"], 5.0625)
            self.assertEqual(first["media"]["codec"], "h264")
            self.assertEqual(first["media"]["pixel_format"], "yuv420p")


if __name__ == "__main__":
    unittest.main()
