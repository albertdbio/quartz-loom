"""Bounded real-CUDA proof for CF++1 pull generation and rolling TAEHV.

The default one-block mode proves the first visible PNG while intentionally
leaving the in-process runtime non-reusable; the CLI process must exit.  The
21-block mode exercises the complete 81-frame release contract and cleanly
finishes both generator and decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from bench.cf_cuda_adapter import RuntimeBootstrapError, build_cf1_runtime
from bench.cf_cuda_generator import (
    CF1LatentPullSession,
    CudaGenerationError,
)
from bench.cf_cuda_session import (
    CudaSessionError,
    RollingTaehvChunkDecoder,
    _require_verified_runtime,
)
from bench.cf_runtime_preflight import (
    DEFAULT_RUNTIME_LOCK_PATH,
    RuntimePreflightError,
    load_runtime_lock_snapshot,
    validate_runtime_lock,
)
from bench.png_validation import is_valid_png


_FRAME_SHAPE = (3, 480, 832)
_FULL_FRAME_SHAPE = (None, 3, 480, 832)
_ALLOWED_BLOCK_COUNTS = frozenset({1, 21})
_MAX_SEED = 2**32 - 1


class CudaSmokeError(ValueError):
    """The bounded CUDA smoke cannot produce trustworthy evidence."""


def _normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _expected_torchvision_version() -> str:
    snapshot = load_runtime_lock_snapshot(DEFAULT_RUNTIME_LOCK_PATH)
    lock = snapshot.parsed()
    validate_runtime_lock(lock, require_frozen=True)
    matches = [
        package["version"]
        for package in lock["packages"]
        if _normalize_distribution(package["distribution"]) == "torchvision"
    ]
    if len(matches) != 1:
        raise CudaSmokeError("runtime torchvision pin is ambiguous")
    return matches[0]


def _device_type(value: Any) -> object:
    return getattr(value, "type", None)


def build_png_encoder(runtime: Any) -> Callable[[Any], Sequence[bytes]]:
    """Build a CPU PNG encoder from the exact runtime's torchvision module."""

    try:
        runtime = _require_verified_runtime(runtime)
    except CudaSessionError as error:
        raise CudaSmokeError("verified CF++1 runtime is required") from error
    loaded_torch = importlib.import_module("torch")
    if loaded_torch is not runtime.torch:
        raise CudaSmokeError("PNG encoder Torch binding does not match runtime")
    torchvision = importlib.import_module("torchvision")
    if getattr(torchvision, "__version__", None) != _expected_torchvision_version():
        raise CudaSmokeError("loaded torchvision version does not match runtime")
    torchvision_io = importlib.import_module("torchvision.io")
    encode_png = getattr(torchvision_io, "encode_png", None)
    if not callable(encode_png):
        raise CudaSmokeError("torchvision PNG encoder is unavailable")

    def encode_frames(frames: Any) -> tuple[bytes, ...]:
        try:
            shape = tuple(frames.shape)
        except (AttributeError, TypeError) as error:
            raise CudaSmokeError("CPU frame batch has no tensor shape") from error
        if (
            len(shape) != 4
            or shape[0] not in {1, 4}
            or shape[1:] != _FRAME_SHAPE
            or getattr(frames, "dtype", None) != runtime.torch.uint8
            or _device_type(getattr(frames, "device", None)) != "cpu"
        ):
            raise CudaSmokeError("CPU frame batch does not match PNG contract")
        payloads: list[bytes] = []
        for index in range(shape[0]):
            frame = frames[index]
            try:
                encoded = encode_png(frame.contiguous(), compression_level=1)
                encoded_shape = tuple(encoded.shape)
                payload = encoded.numpy().tobytes()
            except Exception as error:
                raise CudaSmokeError("PNG frame encoding failed") from error
            if (
                len(encoded_shape) != 1
                or getattr(encoded, "dtype", None) != runtime.torch.uint8
                or _device_type(getattr(encoded, "device", None)) != "cpu"
                or not isinstance(payload, bytes)
                or not is_valid_png(
                    payload,
                    expected_width=_FRAME_SHAPE[2],
                    expected_height=_FRAME_SHAPE[1],
                    require_rgb8=True,
                )
            ):
                raise CudaSmokeError("encoded PNG payload is invalid")
            payloads.append(payload)
        return tuple(payloads)

    return encode_frames


def _validate_smoke_request(
    *,
    prompt: Any,
    seed: Any,
    blocks: Any,
    output_directory: Any,
) -> Path:
    if blocks not in _ALLOWED_BLOCK_COUNTS or isinstance(blocks, bool):
        raise CudaSmokeError("blocks must be exactly 1 or 21")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CudaSmokeError("prompt must be a non-empty string")
    try:
        prompt.encode("utf-8")
    except UnicodeError as error:
        raise CudaSmokeError("prompt must be valid UTF-8") from error
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > _MAX_SEED
    ):
        raise CudaSmokeError("seed must be an unsigned 32-bit integer")
    if not isinstance(output_directory, Path):
        raise CudaSmokeError("output directory must be a Path")
    output = output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise CudaSmokeError("output directory already exists")
    return output


def _positive_duration(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise CudaSmokeError(f"{label} is invalid")
    return value


def _write_smoke_manifest(output: Path, report: dict[str, Any]) -> None:
    try:
        (output / "manifest.json").write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise CudaSmokeError("smoke manifest could not be persisted") from error


def run_cf1_cuda_smoke(
    *,
    prompt: str,
    seed: int,
    blocks: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Run one bounded or complete real pull/decode proof and persist PNGs."""

    output = _validate_smoke_request(
        prompt=prompt,
        seed=seed,
        blocks=blocks,
        output_directory=output_directory,
    )
    runtime = build_cf1_runtime()
    provenance = getattr(runtime, "provenance", None)
    if provenance is None:
        raise CudaSmokeError("runtime provenance is unavailable")
    encode_frames = build_png_encoder(runtime)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise CudaSmokeError("output directory could not be created") from error

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    frame_hashes: list[str] = []
    chunk_frame_counts: list[int] = []
    block_durations: list[float] = []
    frame_index = 0
    try:
        runtime.torch.cuda.synchronize(runtime.device)
        wall_start = time.perf_counter()
        session = CF1LatentPullSession(
            runtime=runtime,
            prompt=prompt,
            seed=seed,
        )
        decoder = RollingTaehvChunkDecoder(
            runtime=runtime,
            torch=runtime.torch,
            encode_frames=encode_frames,
            frame_media_type="image/png",
        )
        first_chunk_encoded_s: float | None = None
        first_frame_written_s: float | None = None
        for expected_block_index in range(blocks):
            block_start = time.perf_counter()
            generated = session.pull()
            if generated.block_index != expected_block_index:
                raise CudaSmokeError("generated block index is not contiguous")
            chunk = decoder.decode(
                generated.denoised_latent,
                latent_ready_event=generated.latent_ready_event,
            )
            expected_frames = 1 if expected_block_index == 0 else 4
            if (
                chunk.frame_media_type != "image/png"
                or chunk.frame_count != expected_frames
            ):
                raise CudaSmokeError("decoded chunk violates the release contract")
            if first_chunk_encoded_s is None:
                first_chunk_encoded_s = _positive_duration(
                    time.perf_counter() - wall_start,
                    "first encoded chunk latency",
                )
            for payload in chunk.frame_payloads:
                if not is_valid_png(
                    payload,
                    expected_width=_FRAME_SHAPE[2],
                    expected_height=_FRAME_SHAPE[1],
                    require_rgb8=True,
                ):
                    raise CudaSmokeError("decoded frame is not a PNG payload")
                frame_path = output / f"frame-{frame_index:06d}.png"
                try:
                    written = frame_path.write_bytes(payload)
                except OSError as error:
                    raise CudaSmokeError(
                        "decoded frame could not be persisted"
                    ) from error
                if written != len(payload):
                    raise CudaSmokeError("decoded frame write was incomplete")
                frame_hashes.append(hashlib.sha256(payload).hexdigest())
                frame_index += 1
                if first_frame_written_s is None:
                    first_frame_written_s = _positive_duration(
                        time.perf_counter() - wall_start,
                        "first written frame latency",
                    )
            chunk_frame_counts.append(chunk.frame_count)
            elapsed = _positive_duration(
                time.perf_counter() - block_start,
                "block duration",
            )
            block_durations.append(elapsed)

        if blocks == 21:
            decoder.finish()
            session.finish()
            if not decoder.complete or not session.complete:
                raise CudaSmokeError("complete runtime did not finish cleanly")
            status = "complete"
            runtime_reusable = True
        else:
            status = "bounded-first-chunk"
            runtime_reusable = False

        wall_s = _positive_duration(
            time.perf_counter() - wall_start,
            "wall duration",
        )
        frame_count = len(frame_hashes)
        report = {
            "schema_version": 1,
            "kind": "cf1-cuda-smoke",
            "ready": True,
            "status": status,
            "runtime_reusable": runtime_reusable,
            "prompt_sha256": prompt_sha256,
            "seed": seed,
            "block_count": blocks,
            "chunk_frame_counts": chunk_frame_counts,
            "frame_count": frame_count,
            "frame_payload_sha256": frame_hashes,
            "frame_media_type": "image/png",
            "resolution": {"width": 832, "height": 480},
            "fps_contract": 16,
            "timing": {
                "clock": "time.perf_counter",
                "origin": (
                    "after_runtime_bootstrap_sync_before_session_initialization"
                ),
                "stream_mode": "serial",
                "first_chunk_encoded_s": first_chunk_encoded_s,
                "first_chunk_encoded_includes": [
                    "session_initialization",
                    "decoder_construction",
                    "generation",
                    "context_refresh",
                    "taehv_decode",
                    "d2h",
                    "png_encode",
                ],
                "first_frame_written_s": first_frame_written_s,
                "wall_s": wall_s,
                "effective_fps": frame_count / wall_s if wall_s > 0 else None,
                "block_durations_s": block_durations,
                "excludes": [
                    "runtime_bootstrap",
                    "model_weight_load",
                    "png_encoder_construction",
                    "output_directory_creation",
                    "manifest_write",
                ],
            },
            "bootstrap_identity_sha256": provenance.bootstrap_identity_sha256,
            "runtime_environment_sha256": provenance.runtime_environment_sha256,
            "guard_bundle_sha256": provenance.guard_bundle_sha256,
            "output_directory": str(output),
        }
        _write_smoke_manifest(output, report)
    except BaseException as error:
        failed_report = {
            "schema_version": 1,
            "kind": "cf1-cuda-smoke",
            "ready": False,
            "status": "failed",
            "runtime_reusable": False,
            "failure_type": type(error).__name__,
            "prompt_sha256": prompt_sha256,
            "seed": seed,
            "intended_block_count": blocks,
            "completed_block_count": len(chunk_frame_counts),
            "chunk_frame_counts": chunk_frame_counts,
            "frame_count": len(frame_hashes),
            "frame_payload_sha256": frame_hashes,
            "frame_media_type": "image/png",
            "bootstrap_identity_sha256": provenance.bootstrap_identity_sha256,
            "runtime_environment_sha256": provenance.runtime_environment_sha256,
            "guard_bundle_sha256": provenance.guard_bundle_sha256,
            "output_directory": str(output),
        }
        try:
            _write_smoke_manifest(output, failed_report)
        except CudaSmokeError:
            pass
        raise
    return report


def cf1_cuda_smoke_report(
    *,
    prompt: str,
    seed: int,
    blocks: int,
    output_directory: Path,
) -> dict[str, Any]:
    try:
        return run_cf1_cuda_smoke(
            prompt=prompt,
            seed=seed,
            blocks=blocks,
            output_directory=output_directory,
        )
    except (
        CudaSmokeError,
        CudaGenerationError,
        CudaSessionError,
        RuntimeBootstrapError,
        RuntimePreflightError,
    ) as error:
        return {
            "schema_version": 1,
            "kind": "cf1-cuda-smoke",
            "ready": False,
            "failure": str(error),
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "cf1-cuda-smoke",
            "ready": False,
            "failure": f"unexpected smoke error: {type(error).__name__}",
        }


def _seed_argument(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed <= _MAX_SEED:
        raise argparse.ArgumentTypeError("seed must be unsigned 32-bit")
    return seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=_seed_argument, default=0)
    parser.add_argument("--blocks", type=int, choices=(1, 21), default=1)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = cf1_cuda_smoke_report(
        prompt=arguments.prompt,
        seed=arguments.seed,
        blocks=arguments.blocks,
        output_directory=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
