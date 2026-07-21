"""Fail-closed rolling-TAEHV chunk decode for the pinned CF++1 worker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from bench.cf_cuda_adapter import (
    CF1_ASSET_LOCK_SHA256,
    CF1_EFFECTIVE_CONFIG_SHA256,
    CF1_RUNTIME_LOCK_SHA256,
    CF1_STACK_ID,
    CF1_TOKENIZER_SENTINEL_SHA256,
    CF1BootstrapProvenance,
    CF1Runtime,
    RuntimeBootstrapError,
    validate_cf1_runtime_provenance,
)
from bench.cf_runtime_preflight import CF1_RUNTIME_ID, RuntimePreflightIdentity
from bench.generation_preflight import PreflightError, rolling_taehv_trim_frames
from bench.png_validation import is_valid_png
from bench.streaming_service import DEFAULT_MAX_CHUNK_BYTES, DecodedChunk


_LATENT_SHAPE = (1, 1, 16, 60, 104)
_MAX_BLOCKS = 21
_FRAME_WIDTH = 832
_FRAME_HEIGHT = 480
# Deliberately excludes the service's opaque development media type.
_RENDERABLE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class CudaSessionError(ValueError):
    """The CUDA session cannot continue without uncertain decoder ownership."""


def _device_identity(value: Any) -> tuple[object, object]:
    return (getattr(value, "type", None), getattr(value, "index", None))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_verified_runtime(runtime: Any) -> CF1Runtime:
    if not isinstance(runtime, CF1Runtime):
        raise CudaSessionError("decoder runtime must be a verified CF1Runtime")
    provenance = runtime.provenance
    if not isinstance(provenance, CF1BootstrapProvenance):
        raise CudaSessionError("verified runtime provenance is required")
    if (
        provenance.stack_id != CF1_STACK_ID
        or provenance.asset_lock_sha256 != CF1_ASSET_LOCK_SHA256
        or provenance.runtime_lock_sha256 != CF1_RUNTIME_LOCK_SHA256
        or not _is_sha256(provenance.runtime_evidence_sha256)
        or not _is_sha256(provenance.static_environment_sha256)
        or not _is_sha256(provenance.runtime_environment_sha256)
        or not _is_sha256(provenance.runtime_native_environment_sha256)
        or not _is_sha256(provenance.native_identity_sha256)
        or not _is_sha256(provenance.attention_probe_identity_sha256)
        or provenance.effective_config_sha256 != CF1_EFFECTIVE_CONFIG_SHA256
        or provenance.tokenizer_sentinel_sha256
        != CF1_TOKENIZER_SENTINEL_SHA256
        or runtime.effective_config_sha256 != CF1_EFFECTIVE_CONFIG_SHA256
        or runtime.tokenizer_sentinel_sha256 != CF1_TOKENIZER_SENTINEL_SHA256
        or provenance.effective_config_sha256 != runtime.effective_config_sha256
        or runtime.torch is None
        or runtime.attention_backend
        not in {"flash-attention-3", "flash-attention-2"}
        or provenance.attention_backend != runtime.attention_backend
        or not isinstance(runtime.runtime_identity, RuntimePreflightIdentity)
        or runtime.runtime_identity.runtime_id != CF1_RUNTIME_ID
        or runtime.runtime_identity.runtime_lock_sha256
        != CF1_RUNTIME_LOCK_SHA256
        or runtime.runtime_identity.runtime_evidence_sha256
        != provenance.runtime_evidence_sha256
        or runtime.runtime_identity.static_environment_sha256
        != provenance.static_environment_sha256
        or runtime.runtime_identity.environment_sha256
        != provenance.runtime_environment_sha256
        or runtime.runtime_native_environment_sha256
        != provenance.runtime_native_environment_sha256
        or runtime.native_identity_sha256 != provenance.native_identity_sha256
        or runtime.attention_probe_identity_sha256
        != provenance.attention_probe_identity_sha256
        or not _is_sha256(provenance.guard_bundle_sha256)
        or not _is_sha256(provenance.bootstrap_identity_sha256)
    ):
        raise CudaSessionError("CF++1 runtime identity does not match the pin")
    try:
        validate_cf1_runtime_provenance(runtime)
    except RuntimeBootstrapError as error:
        raise CudaSessionError("CF++1 runtime identity is not verified") from error
    device_type, device_index = _device_identity(runtime.device)
    if (
        device_type != "cuda"
        or isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index != 0
    ):
        raise CudaSessionError("CF++1 runtime requires an explicit CUDA device")
    return runtime


def _payload_matches_media_type(payload: bytes, media_type: str) -> bool:
    if media_type == "image/jpeg":
        return payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    if media_type == "image/png":
        return is_valid_png(
            payload,
            expected_width=_FRAME_WIDTH,
            expected_height=_FRAME_HEIGHT,
            require_rgb8=True,
        )
    if media_type == "image/webp":
        return (
            len(payload) >= 12
            and payload[:4] == b"RIFF"
            and payload[8:12] == b"WEBP"
            and int.from_bytes(payload[4:8], "little") + 8 == len(payload)
        )
    return False


class RollingTaehvChunkDecoder:
    """Decode one CF++ latent per pull into immutable CPU-owned raster bytes.

    This class does not create a CUDA runtime or generate latents. It owns only
    the rolling three-latent TAEHV state. Any failure after decode begins
    poisons the instance; the process owner must then kill/reap the worker.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        torch: Any,
        encode_frames: Callable[[Any], Sequence[bytes]],
        frame_media_type: str = "image/jpeg",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    ) -> None:
        runtime = _require_verified_runtime(runtime)
        if torch is not runtime.torch:
            raise CudaSessionError(
                "Torch binding does not match the verified runtime"
            )
        if not callable(encode_frames):
            raise CudaSessionError("frame encoder must be callable")
        if frame_media_type not in _RENDERABLE_MEDIA_TYPES:
            raise CudaSessionError("frame media type is not renderable")
        if (
            isinstance(max_chunk_bytes, bool)
            or not isinstance(max_chunk_bytes, int)
            or max_chunk_bytes <= 0
        ):
            raise CudaSessionError("max_chunk_bytes must be a positive integer")
        self.runtime = runtime
        self.torch = torch
        self.encode_frames = encode_frames
        self.frame_media_type = frame_media_type
        self.max_chunk_bytes = max_chunk_bytes
        try:
            self._decode_stream = torch.cuda.Stream(device=runtime.device)
        except Exception as error:
            raise CudaSessionError("CUDA decode stream could not be created") from error
        self._tail: Any | None = None
        self._block_index = 0
        self._trim_history: list[int] = []
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def trim_history(self) -> tuple[int, ...]:
        return tuple(self._trim_history)

    @property
    def complete(self) -> bool:
        return self._block_index == _MAX_BLOCKS and not self._poisoned

    def finish(self) -> None:
        if self._poisoned:
            raise CudaSessionError("rolling TAEHV decoder is poisoned")
        if self._block_index != _MAX_BLOCKS:
            raise CudaSessionError("rolling TAEHV decode is incomplete")
        self._tail = None

    def _validate_latent(self, latent: Any) -> None:
        try:
            shape = tuple(latent.shape)
        except (AttributeError, TypeError) as error:
            raise CudaSessionError("denoised latent has no tensor shape") from error
        if shape != _LATENT_SHAPE:
            raise CudaSessionError("denoised latent shape does not match CF++1")
        if _device_identity(getattr(latent, "device", None)) != _device_identity(
            self.runtime.device
        ):
            raise CudaSessionError("denoised latent is on the wrong CUDA device")
        if getattr(latent, "dtype", None) != self.torch.bfloat16:
            raise CudaSessionError("denoised latent must be CF++1 bfloat16 output")

    def decode(
        self,
        denoised_latent: Any,
        *,
        latent_ready_event: Any,
    ) -> DecodedChunk:
        if self._poisoned:
            raise CudaSessionError("rolling TAEHV decoder is poisoned")
        if self._block_index >= _MAX_BLOCKS:
            raise CudaSessionError("rolling TAEHV decode is complete")
        self._validate_latent(denoised_latent)

        try:
            with self.torch.cuda.device(self.runtime.device):
                with self.torch.cuda.stream(self._decode_stream):
                    self._decode_stream.wait_event(latent_ready_event)
                    denoised_latent.record_stream(self._decode_stream)
                    current = denoised_latent.to(dtype=self.torch.float16)
                    prior_context = (
                        0 if self._tail is None else int(self._tail.shape[1])
                    )
                    try:
                        trim_frames = rolling_taehv_trim_frames(
                            self._block_index,
                            prior_context,
                        )
                    except PreflightError as error:
                        raise CudaSessionError(
                            "rolling TAEHV context does not match block index"
                        ) from error
                    decode_input = (
                        current
                        if self._tail is None
                        else self.torch.cat((self._tail, current), dim=1)
                    )
                    next_tail = decode_input[:, -3:]
                    pixels_untrimmed = self.runtime.taehv.decode_video(
                        decode_input,
                        parallel=True,
                        show_progress_bar=False,
                    )
                    if (
                        _device_identity(getattr(pixels_untrimmed, "device", None))
                        != _device_identity(self.runtime.device)
                    ):
                        raise CudaSessionError(
                            "rolling TAEHV output is on the wrong device"
                        )
                    if getattr(pixels_untrimmed, "dtype", None) != self.torch.float16:
                        raise CudaSessionError(
                            "rolling TAEHV output dtype must be float16"
                        )
                    pixels = pixels_untrimmed[:, trim_frames:]
                    expected_frames = 1 if self._block_index == 0 else 4
                    expected_shape = (1, expected_frames, 3, 480, 832)
                    if tuple(pixels.shape) != expected_shape:
                        raise CudaSessionError(
                            "rolling TAEHV output shape does not match release contract"
                        )
                    gpu_frames = (
                        pixels[0]
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .to(dtype=self.torch.uint8)
                    )
                    decoded_event = self.torch.cuda.Event(blocking=True)
                    decoded_event.record(self._decode_stream)

            decoded_event.synchronize()
            cpu_frames = gpu_frames.cpu()
            if (
                tuple(cpu_frames.shape) != expected_shape[1:]
                or getattr(cpu_frames.device, "type", None) != "cpu"
                or cpu_frames.dtype != self.torch.uint8
            ):
                raise CudaSessionError("D2H raster ownership is invalid")

            encoded = self.encode_frames(cpu_frames)
            if isinstance(encoded, (str, bytes, bytearray)) or not isinstance(
                encoded, Sequence
            ):
                raise CudaSessionError("frame encoder returned an invalid batch")
            payloads = tuple(encoded)
            if len(payloads) != expected_frames or any(
                not isinstance(payload, bytes) or not payload for payload in payloads
            ):
                raise CudaSessionError("encoded frame payloads are invalid")
            if any(
                not _payload_matches_media_type(payload, self.frame_media_type)
                for payload in payloads
            ):
                raise CudaSessionError("encoded frame payload format is invalid")
            if sum(len(payload) for payload in payloads) > self.max_chunk_bytes:
                raise CudaSessionError("encoded frame payload batch exceeds byte limit")
        except BaseException:
            self._poisoned = True
            raise

        self._tail = next_tail
        self._block_index += 1
        self._trim_history.append(trim_frames)
        return DecodedChunk(
            frame_payloads=payloads,
            frame_media_type=self.frame_media_type,
        )
