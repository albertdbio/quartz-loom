"""Assemble a complete CF++1 PNG smoke into a verified development MP4.

This module deliberately stops at a local, hash-bound artifact.  It does not
load provider credentials, call a video-understanding service, or turn smoke
timings into performance or quality evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


_FRAME_COUNT = 81
_WIDTH = 832
_HEIGHT = 480
_FPS = 16
_DURATION_SECONDS = _FRAME_COUNT / _FPS
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SMOKE_MANIFEST_FIELDS = {
    "schema_version",
    "kind",
    "ready",
    "status",
    "runtime_reusable",
    "prompt_sha256",
    "seed",
    "block_count",
    "chunk_frame_counts",
    "frame_count",
    "frame_payload_sha256",
    "frame_media_type",
    "resolution",
    "fps_contract",
    "timing",
    "bootstrap_identity_sha256",
    "runtime_environment_sha256",
    "guard_bundle_sha256",
    "output_directory",
}
_ARTIFACT_MANIFEST_FIELDS = {
    "schema_version",
    "kind",
    "purpose",
    "status",
    "authorizes_quality_claim",
    "authorizes_performance_claim",
    "source",
    "encoding",
    "media",
}
_SOURCE_FIELDS = {
    "smoke_manifest_sha256",
    "generation_prompt_sha256",
    "seed",
    "bootstrap_identity_sha256",
    "runtime_environment_sha256",
    "guard_bundle_sha256",
    "frame_count",
    "frame_payload_sha256",
}
_ENCODING_FIELDS = {
    "fps",
    "encoder",
    "preset",
    "crf",
    "codec",
    "pixel_format",
    "ffmpeg_executable_sha256",
    "ffmpeg_version_sha256",
    "ffmpeg_version_first_line",
    "ffprobe_executable_sha256",
    "ffprobe_version_sha256",
    "ffprobe_version_first_line",
    "command_arguments",
    "repeat_encode_byte_equal",
    "repeat_encode_sha256",
}
_MEDIA_FIELDS = {
    "file",
    "media_sha256",
    "byte_count",
    "width",
    "height",
    "fps",
    "decoded_frames",
    "duration_s",
    "codec",
    "pixel_format",
    "video_streams",
    "audio_streams",
}


class ArtifactAssemblyError(ValueError):
    """The source frames or assembled development artifact are not trustworthy."""


@dataclass(frozen=True)
class ToolIdentity:
    resolved_path: Path
    executable_sha256: str
    version_sha256: str
    version_first_line: str


@dataclass(frozen=True)
class _SmokeSource:
    directory: Path
    manifest_sha256: str
    generation_prompt_sha256: str
    seed: int
    bootstrap_identity_sha256: str
    runtime_environment_sha256: str
    guard_bundle_sha256: str
    frame_payload_sha256: tuple[str, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactAssemblyError("artifact input could not be hashed") from error
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ArtifactAssemblyError("JSON input contains a duplicate key")
        parsed[key] = value
    return parsed


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactAssemblyError(f"{label} must be a regular file")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except ArtifactAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactAssemblyError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ArtifactAssemblyError(f"{label} must be a JSON object")
    return parsed, raw


def _require_exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ArtifactAssemblyError(f"{label} fields do not match schema")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactAssemblyError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ArtifactAssemblyError(f"{label} must equal {expected}")
    return value


def _validate_png_header(payload: bytes) -> None:
    if (
        len(payload) < 45
        or not payload.startswith(_PNG_SIGNATURE)
        or not payload.endswith(_PNG_IEND)
        or payload[8:12] != struct.pack(">I", 13)
        or payload[12:16] != b"IHDR"
    ):
        raise ArtifactAssemblyError("smoke frame is not a complete PNG")
    try:
        width, height, depth, color, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", payload[16:29]
        )
    except struct.error as error:
        raise ArtifactAssemblyError("smoke PNG header is malformed") from error
    if (
        width != _WIDTH
        or height != _HEIGHT
        or depth != 8
        or color != 2
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise ArtifactAssemblyError("smoke PNG does not match the RGB frame contract")


def _validate_smoke_directory(smoke_directory: str | Path) -> _SmokeSource:
    supplied = Path(smoke_directory)
    if supplied.is_symlink() or not supplied.is_dir():
        raise ArtifactAssemblyError("smoke directory must be a regular directory")
    directory = supplied.resolve()
    expected_frame_names = [f"frame-{index:06d}.png" for index in range(_FRAME_COUNT)]
    expected_names = {"manifest.json", *expected_frame_names}
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise ArtifactAssemblyError("smoke directory could not be enumerated") from error
    if {entry.name for entry in entries} != expected_names:
        raise ArtifactAssemblyError("smoke directory entries do not match the 81-frame contract")
    if any(entry.is_symlink() for entry in entries):
        raise ArtifactAssemblyError("smoke directory must not contain symlinks")

    manifest, manifest_bytes = _read_json_object(
        directory / "manifest.json", "smoke manifest"
    )
    _require_exact_fields(manifest, _SMOKE_MANIFEST_FIELDS, "smoke manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "cf1-cuda-smoke"
        or manifest["ready"] is not True
        or manifest["status"] != "complete"
        or manifest["runtime_reusable"] is not True
    ):
        raise ArtifactAssemblyError("smoke manifest is not a complete reusable proof")
    _require_int(manifest["block_count"], 21, "smoke block_count")
    _require_int(manifest["frame_count"], _FRAME_COUNT, "smoke frame_count")
    _require_int(manifest["fps_contract"], _FPS, "smoke fps_contract")
    if manifest["chunk_frame_counts"] != [1] + [4] * 20:
        raise ArtifactAssemblyError("smoke chunk frame counts do not match the release contract")
    if manifest["frame_media_type"] != "image/png":
        raise ArtifactAssemblyError("smoke media type must be image/png")
    if manifest["resolution"] != {"width": _WIDTH, "height": _HEIGHT}:
        raise ArtifactAssemblyError("smoke resolution does not match the frame contract")
    timing = manifest["timing"]
    if not isinstance(timing, dict) or timing.get("stream_mode") != "serial":
        raise ArtifactAssemblyError("smoke timing must identify the current serial path")
    if not isinstance(manifest["output_directory"], str) or not manifest[
        "output_directory"
    ]:
        raise ArtifactAssemblyError("smoke output_directory is required")
    seed = manifest["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > 2**32 - 1
    ):
        raise ArtifactAssemblyError("smoke seed must be unsigned 32-bit")
    frame_hashes = manifest["frame_payload_sha256"]
    if not isinstance(frame_hashes, list) or len(frame_hashes) != _FRAME_COUNT:
        raise ArtifactAssemblyError("smoke frame hashes must cover exactly 81 frames")
    expected_hashes = tuple(
        _require_sha256(value, "smoke frame hash") for value in frame_hashes
    )

    for index, (name, expected_hash) in enumerate(zip(expected_frame_names, expected_hashes)):
        frame = directory / name
        if frame.is_symlink() or not frame.is_file():
            raise ArtifactAssemblyError("smoke frame must be a regular file")
        try:
            payload = frame.read_bytes()
        except OSError as error:
            raise ArtifactAssemblyError("smoke frame could not be read") from error
        _validate_png_header(payload)
        if _sha256_bytes(payload) != expected_hash:
            raise ArtifactAssemblyError(f"smoke frame hash mismatch at index {index}")

    return _SmokeSource(
        directory=directory,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        generation_prompt_sha256=_require_sha256(
            manifest["prompt_sha256"], "smoke prompt_sha256"
        ),
        seed=seed,
        bootstrap_identity_sha256=_require_sha256(
            manifest["bootstrap_identity_sha256"],
            "smoke bootstrap_identity_sha256",
        ),
        runtime_environment_sha256=_require_sha256(
            manifest["runtime_environment_sha256"],
            "smoke runtime_environment_sha256",
        ),
        guard_bundle_sha256=_require_sha256(
            manifest["guard_bundle_sha256"], "smoke guard_bundle_sha256"
        ),
        frame_payload_sha256=expected_hashes,
    )


def _rehash_smoke_frames(source: _SmokeSource) -> None:
    for index, expected in enumerate(source.frame_payload_sha256):
        frame = source.directory / f"frame-{index:06d}.png"
        if frame.is_symlink() or not frame.is_file() or _sha256_file(frame) != expected:
            raise ArtifactAssemblyError("smoke frames changed during assembly")


def _canonical_ffmpeg_arguments() -> list[str]:
    """Return the exact path-independent command recorded in every artifact."""

    return [
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
    ]


def _sanitized_subprocess_environment(temporary_directory: Path) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "SOURCE_DATE_EPOCH": "0",
        "TMPDIR": str(temporary_directory.resolve()),
        "TZ": "UTC",
    }


def _resolve_tool_identity(
    executable: str | Path,
    *,
    label: str,
    environment: Mapping[str, str],
) -> ToolIdentity:
    candidate_text = str(executable)
    if isinstance(executable, Path) or os.sep in candidate_text:
        candidate = Path(candidate_text)
    else:
        located = shutil.which(candidate_text)
        if located is None:
            raise ArtifactAssemblyError(f"{label} executable is unavailable")
        candidate = Path(located)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ArtifactAssemblyError(f"{label} executable could not be resolved") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ArtifactAssemblyError(f"{label} executable is not a regular executable")
    try:
        completed = subprocess.run(
            [str(resolved), "-version"],
            capture_output=True,
            check=False,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactAssemblyError(f"{label} version probe failed") from error
    version_bytes = completed.stdout + completed.stderr
    if completed.returncode != 0 or not version_bytes or len(version_bytes) > 1_000_000:
        raise ArtifactAssemblyError(f"{label} version probe was invalid")
    try:
        lines = version_bytes.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ArtifactAssemblyError(f"{label} version output was not UTF-8") from error
    if not lines or not lines[0].strip():
        raise ArtifactAssemblyError(f"{label} version output was empty")
    return ToolIdentity(
        resolved_path=resolved,
        executable_sha256=_sha256_file(resolved),
        version_sha256=_sha256_bytes(version_bytes),
        version_first_line=lines[0].strip(),
    )


def _ffmpeg_command(
    *, frame_directory: Path, output_path: Path, ffmpeg: ToolIdentity
) -> list[str]:
    replacements = {
        "<FFMPEG>": str(ffmpeg.resolved_path),
        "<FRAME_DIRECTORY>/frame-%06d.png": str(
            frame_directory / "frame-%06d.png"
        ),
        "<OUTPUT>": str(output_path),
    }
    return [replacements.get(value, value) for value in _canonical_ffmpeg_arguments()]


def _encode_once(
    *,
    frame_directory: Path,
    output_path: Path,
    ffmpeg: ToolIdentity,
    environment: Mapping[str, str],
) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise ArtifactAssemblyError("staged video output already exists")
    try:
        completed = subprocess.run(
            _ffmpeg_command(
                frame_directory=frame_directory,
                output_path=output_path,
                ffmpeg=ffmpeg,
            ),
            capture_output=True,
            check=False,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactAssemblyError("FFmpeg encode could not be executed") from error
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise ArtifactAssemblyError(
            f"FFmpeg encode exited {completed.returncode}; output withheld"
        )
    if output_path.is_symlink() or not output_path.is_file():
        raise ArtifactAssemblyError("FFmpeg did not produce a regular MP4")


def _parse_probe_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        streams = payload["streams"]
        format_info = payload["format"]
        if (
            not isinstance(streams, list)
            or len(streams) != 1
            or not isinstance(streams[0], dict)
            or not isinstance(format_info, dict)
            or "mp4" not in str(format_info.get("format_name", "")).split(",")
        ):
            raise TypeError
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if len(videos) != 1:
            raise ValueError
        stream = videos[0]
        rate = Fraction(stream["r_frame_rate"])
        frames = stream.get("nb_read_frames") or stream.get("nb_frames")
        duration = stream.get("duration") or format_info.get("duration")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(rate),
            "decoded_frames": int(frames),
            "duration_s": float(duration),
            "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"],
            "video_streams": len(videos),
            "audio_streams": len(audios),
        }
    except (ArtifactAssemblyError, AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactAssemblyError("FFprobe response did not match the media schema") from error


def _probe_media(
    path: Path,
    *,
    ffprobe: ToolIdentity,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        str(ffprobe.resolved_path),
        "-v",
        "error",
        "-err_detect",
        "explode",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,"
            "nb_read_frames,nb_frames,duration:format=format_name,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactAssemblyError("FFprobe could not be executed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > 1_000_000
    ):
        raise ArtifactAssemblyError("FFprobe failed; output withheld")
    return _parse_probe_payload(completed.stdout)


def _decode_media(
    path: Path,
    *,
    ffmpeg: ToolIdentity,
    environment: Mapping[str, str],
) -> None:
    command = [
        str(ffmpeg.resolved_path),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "81",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactAssemblyError("FFmpeg validation decode could not be executed") from error
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise ArtifactAssemblyError("FFmpeg validation decode failed; output withheld")


def _validate_probe_contract(probe: Any) -> dict[str, Any]:
    expected_fields = {
        "width",
        "height",
        "fps",
        "decoded_frames",
        "duration_s",
        "codec",
        "pixel_format",
        "video_streams",
        "audio_streams",
    }
    value = _require_exact_fields(probe, expected_fields, "media probe")
    for field, expected in (
        ("width", _WIDTH),
        ("height", _HEIGHT),
        ("decoded_frames", _FRAME_COUNT),
        ("video_streams", 1),
        ("audio_streams", 0),
    ):
        _require_int(value[field], expected, f"media {field}")
    fps = value["fps"]
    duration = value["duration_s"]
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or not math.isclose(float(fps), float(_FPS), abs_tol=1e-9)
    ):
        raise ArtifactAssemblyError("media fps must equal 16")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or not math.isclose(float(duration), _DURATION_SECONDS, abs_tol=1e-6)
    ):
        raise ArtifactAssemblyError("media duration does not match 81 frames at 16 fps")
    if value["codec"] != "h264" or value["pixel_format"] != "yuv420p":
        raise ArtifactAssemblyError("media codec must be H.264 yuv420p")
    return {
        **value,
        "fps": float(fps),
        "duration_s": float(duration),
    }


def _validate_artifact_manifest_schema(manifest: Any) -> dict[str, Any]:
    value = _require_exact_fields(
        manifest, _ARTIFACT_MANIFEST_FIELDS, "artifact manifest"
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != "cf1-development-video-artifact"
        or value["purpose"] != "development-video-understanding-not-gate-evidence"
        or value["status"] != "assembled-and-verified"
        or value["authorizes_quality_claim"] is not False
        or value["authorizes_performance_claim"] is not False
    ):
        raise ArtifactAssemblyError("artifact manifest identity or claim boundary is invalid")

    source = _require_exact_fields(value["source"], _SOURCE_FIELDS, "artifact source")
    for field in (
        "smoke_manifest_sha256",
        "generation_prompt_sha256",
        "bootstrap_identity_sha256",
        "runtime_environment_sha256",
        "guard_bundle_sha256",
    ):
        _require_sha256(source[field], f"artifact source {field}")
    _require_int(source["frame_count"], _FRAME_COUNT, "artifact source frame_count")
    seed = source["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > 2**32 - 1
    ):
        raise ArtifactAssemblyError("artifact source seed must be unsigned 32-bit")
    frame_hashes = source["frame_payload_sha256"]
    if not isinstance(frame_hashes, list) or len(frame_hashes) != _FRAME_COUNT:
        raise ArtifactAssemblyError("artifact source must bind exactly 81 frame hashes")
    for frame_hash in frame_hashes:
        _require_sha256(frame_hash, "artifact source frame hash")

    encoding = _require_exact_fields(
        value["encoding"], _ENCODING_FIELDS, "artifact encoding"
    )
    expected_encoding = {
        "fps": _FPS,
        "encoder": "libx264",
        "preset": "medium",
        "crf": 18,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "command_arguments": _canonical_ffmpeg_arguments(),
        "repeat_encode_byte_equal": True,
    }
    if any(encoding.get(field) != expected for field, expected in expected_encoding.items()):
        raise ArtifactAssemblyError("artifact encoding contract is invalid")
    for field in (
        "ffmpeg_executable_sha256",
        "ffmpeg_version_sha256",
        "ffprobe_executable_sha256",
        "ffprobe_version_sha256",
        "repeat_encode_sha256",
    ):
        _require_sha256(encoding[field], f"artifact encoding {field}")
    for field in ("ffmpeg_version_first_line", "ffprobe_version_first_line"):
        if not isinstance(encoding[field], str) or not encoding[field]:
            raise ArtifactAssemblyError(f"artifact encoding {field} is required")

    media = _require_exact_fields(value["media"], _MEDIA_FIELDS, "artifact media")
    if media["file"] != "video.mp4":
        raise ArtifactAssemblyError("artifact media file must equal video.mp4")
    media_sha256 = _require_sha256(media["media_sha256"], "artifact media SHA-256")
    if encoding["repeat_encode_sha256"] != media_sha256:
        raise ArtifactAssemblyError("repeat encode hash does not match artifact media")
    if (
        isinstance(media["byte_count"], bool)
        or not isinstance(media["byte_count"], int)
        or media["byte_count"] < 1
    ):
        raise ArtifactAssemblyError("artifact media byte_count must be positive")
    _validate_probe_contract({field: media[field] for field in _MEDIA_FIELDS - {"file", "media_sha256", "byte_count"}})
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError as error:
        raise ArtifactAssemblyError("artifact directory could not be synchronized") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactAssemblyError("artifact media must be a regular file")
        os.fsync(descriptor)
    except OSError as error:
        raise ArtifactAssemblyError("artifact media could not be synchronized") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_directory_noreplace(stage: Path, output: Path) -> None:
    """Atomically publish ``stage`` while refusing any destination race."""

    if stage.parent != output.parent:
        raise ArtifactAssemblyError("artifact publication must stay in one directory")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd: int | None = None
    try:
        parent_fd = os.open(stage.parent, parent_flags)
        library = ctypes.CDLL(None, use_errno=True)
        old_name = os.fsencode(stage.name)
        new_name = os.fsencode(output.name)
        if sys.platform == "darwin":
            function = getattr(library, "renameatx_np", None)
            if function is None:
                raise ArtifactAssemblyError("no-replace directory publication is unavailable")
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            result = function(parent_fd, old_name, parent_fd, new_name, 0x00000004)
        elif sys.platform.startswith("linux"):
            function = getattr(library, "renameat2", None)
            if function is None:
                raise ArtifactAssemblyError("no-replace directory publication is unavailable")
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            result = function(parent_fd, old_name, parent_fd, new_name, 0x00000001)
        else:
            raise ArtifactAssemblyError("no-replace directory publication is unavailable")
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise ArtifactAssemblyError("artifact output appeared during assembly")
            raise ArtifactAssemblyError("artifact directory could not be published")
        os.fsync(parent_fd)
    except OSError as error:
        raise ArtifactAssemblyError("artifact directory could not be published") from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _validated_output_path(output_directory: str | Path, smoke: Path) -> Path:
    supplied = Path(output_directory)
    if supplied.exists() or supplied.is_symlink():
        raise ArtifactAssemblyError("artifact output directory already exists")
    try:
        parent = supplied.parent.resolve(strict=True)
    except OSError as error:
        raise ArtifactAssemblyError("artifact output parent does not exist") from error
    if not parent.is_dir() or not supplied.name or supplied.name in {".", ".."}:
        raise ArtifactAssemblyError("artifact output path is invalid")
    output = parent / supplied.name
    if output == smoke or smoke in output.parents:
        raise ArtifactAssemblyError("artifact output must not be inside the smoke directory")
    return output


def assemble_cf1_video_artifact(
    *,
    smoke_directory: str | Path,
    output_directory: str | Path,
    ffmpeg_executable: str | Path = "ffmpeg",
    ffprobe_executable: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Double-encode, verify, and atomically publish one development MP4."""

    source = _validate_smoke_directory(smoke_directory)
    output = _validated_output_path(output_directory, source.directory)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    ).resolve()
    environment = _sanitized_subprocess_environment(stage)
    try:
        ffmpeg = _resolve_tool_identity(
            ffmpeg_executable, label="FFmpeg", environment=environment
        )
        ffprobe = _resolve_tool_identity(
            ffprobe_executable, label="FFprobe", environment=environment
        )
        video = stage / "video.mp4"
        repeated = stage / "repeat.mp4"
        _encode_once(
            frame_directory=source.directory,
            output_path=video,
            ffmpeg=ffmpeg,
            environment=environment,
        )
        _encode_once(
            frame_directory=source.directory,
            output_path=repeated,
            ffmpeg=ffmpeg,
            environment=environment,
        )
        video_sha256 = _sha256_file(video)
        repeated_sha256 = _sha256_file(repeated)
        try:
            same_bytes = (
                video.stat().st_size == repeated.stat().st_size
                and video.read_bytes() == repeated.read_bytes()
            )
        except OSError as error:
            raise ArtifactAssemblyError("repeat encodes could not be compared") from error
        if not same_bytes or video_sha256 != repeated_sha256:
            raise ArtifactAssemblyError("repeat FFmpeg encodes were not byte-identical")
        _rehash_smoke_frames(source)
        probe = _validate_probe_contract(
            _probe_media(video, ffprobe=ffprobe, environment=environment)
        )
        _decode_media(video, ffmpeg=ffmpeg, environment=environment)
        _fsync_regular_file(video)
        try:
            byte_count = video.stat().st_size
        except OSError as error:
            raise ArtifactAssemblyError("assembled MP4 size is unavailable") from error
        if byte_count < 1:
            raise ArtifactAssemblyError("assembled MP4 is empty")

        report = {
            "schema_version": 1,
            "kind": "cf1-development-video-artifact",
            "purpose": "development-video-understanding-not-gate-evidence",
            "status": "assembled-and-verified",
            "authorizes_quality_claim": False,
            "authorizes_performance_claim": False,
            "source": {
                "smoke_manifest_sha256": source.manifest_sha256,
                "generation_prompt_sha256": source.generation_prompt_sha256,
                "seed": source.seed,
                "bootstrap_identity_sha256": source.bootstrap_identity_sha256,
                "runtime_environment_sha256": source.runtime_environment_sha256,
                "guard_bundle_sha256": source.guard_bundle_sha256,
                "frame_count": _FRAME_COUNT,
                "frame_payload_sha256": list(source.frame_payload_sha256),
            },
            "encoding": {
                "fps": _FPS,
                "encoder": "libx264",
                "preset": "medium",
                "crf": 18,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "ffmpeg_executable_sha256": ffmpeg.executable_sha256,
                "ffmpeg_version_sha256": ffmpeg.version_sha256,
                "ffmpeg_version_first_line": ffmpeg.version_first_line,
                "ffprobe_executable_sha256": ffprobe.executable_sha256,
                "ffprobe_version_sha256": ffprobe.version_sha256,
                "ffprobe_version_first_line": ffprobe.version_first_line,
                "command_arguments": _canonical_ffmpeg_arguments(),
                "repeat_encode_byte_equal": True,
                "repeat_encode_sha256": repeated_sha256,
            },
            "media": {
                "file": "video.mp4",
                "media_sha256": video_sha256,
                "byte_count": byte_count,
                **probe,
            },
        }
        _validate_artifact_manifest_schema(report)
        try:
            repeated.unlink()
            _write_json_atomic(stage / "manifest.json", report)
        except OSError as error:
            raise ArtifactAssemblyError("artifact manifest could not be staged") from error
        if {path.name for path in stage.iterdir()} != {"manifest.json", "video.mp4"}:
            raise ArtifactAssemblyError("staged artifact contains unexpected files")
        if output.exists() or output.is_symlink():
            raise ArtifactAssemblyError("artifact output appeared during assembly")
        _publish_directory_noreplace(stage, output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def validate_cf_video_artifact(manifest_path: str | Path) -> dict[str, Any]:
    """Revalidate a published artifact and return its exact provider input bytes."""

    supplied = Path(manifest_path)
    if supplied.name != "manifest.json" or supplied.is_symlink():
        raise ArtifactAssemblyError("artifact manifest path must name manifest.json")
    try:
        path = supplied.resolve(strict=True)
    except OSError as error:
        raise ArtifactAssemblyError("artifact manifest is unavailable") from error
    manifest, raw_manifest = _read_json_object(path, "artifact manifest")
    manifest = _validate_artifact_manifest_schema(manifest)
    directory = path.parent
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise ArtifactAssemblyError("artifact directory could not be enumerated") from error
    if {entry.name for entry in entries} != {"manifest.json", "video.mp4"}:
        raise ArtifactAssemblyError("artifact directory entries do not match schema")
    if any(entry.is_symlink() for entry in entries):
        raise ArtifactAssemblyError("artifact directory must not contain symlinks")
    media_path = directory / "video.mp4"
    try:
        media_bytes = media_path.read_bytes()
    except OSError as error:
        raise ArtifactAssemblyError("artifact media could not be read") from error
    media = manifest["media"]
    media_sha256 = _sha256_bytes(media_bytes)
    if (
        not media_bytes
        or len(media_bytes) != media["byte_count"]
        or media_sha256 != media["media_sha256"]
    ):
        raise ArtifactAssemblyError("artifact media bytes do not match the manifest")
    with tempfile.TemporaryDirectory(prefix="cf-video-validate-") as temporary:
        private_directory = Path(temporary).resolve()
        private_media = private_directory / "video.mp4"
        try:
            with private_media.open("xb") as handle:
                handle.write(media_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise ArtifactAssemblyError("artifact validation copy could not be staged") from error
        environment = _sanitized_subprocess_environment(private_directory)
        ffmpeg = _resolve_tool_identity(
            "ffmpeg", label="FFmpeg", environment=environment
        )
        ffprobe = _resolve_tool_identity(
            "ffprobe", label="FFprobe", environment=environment
        )
        encoding = manifest["encoding"]
        if (
            ffmpeg.executable_sha256 != encoding["ffmpeg_executable_sha256"]
            or ffmpeg.version_sha256 != encoding["ffmpeg_version_sha256"]
            or ffmpeg.version_first_line != encoding["ffmpeg_version_first_line"]
        ):
            raise ArtifactAssemblyError("FFmpeg identity does not match the artifact")
        if (
            ffprobe.executable_sha256 != encoding["ffprobe_executable_sha256"]
            or ffprobe.version_sha256 != encoding["ffprobe_version_sha256"]
            or ffprobe.version_first_line != encoding["ffprobe_version_first_line"]
        ):
            raise ArtifactAssemblyError("FFprobe identity does not match the artifact")
        actual_probe = _validate_probe_contract(
            _probe_media(private_media, ffprobe=ffprobe, environment=environment)
        )
        _decode_media(private_media, ffmpeg=ffmpeg, environment=environment)
    declared_probe = {
        field: media[field]
        for field in _MEDIA_FIELDS - {"file", "media_sha256", "byte_count"}
    }
    if actual_probe != declared_probe:
        raise ArtifactAssemblyError("artifact media probe no longer matches the manifest")
    return {
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": _sha256_bytes(raw_manifest),
        "media_path": media_path.resolve(),
        "media_sha256": media_sha256,
        "media_bytes": media_bytes,
        "generation_prompt_sha256": manifest["source"][
            "generation_prompt_sha256"
        ],
        "fps": float(media["fps"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    arguments = parser.parse_args(argv)
    try:
        report = assemble_cf1_video_artifact(
            smoke_directory=arguments.smoke_dir,
            output_directory=arguments.output,
            ffmpeg_executable=arguments.ffmpeg,
            ffprobe_executable=arguments.ffprobe,
        )
    except ArtifactAssemblyError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:
        print(
            f"error: unexpected artifact assembly failure: {type(error).__name__}",
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
