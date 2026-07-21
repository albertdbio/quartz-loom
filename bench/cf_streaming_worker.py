"""Fail-closed CF++1 CUDA engine for the persistent process protocol.

This module is a local scaffold for the first frozen H100 worker.  It cannot
weaken the candidate runtime lock: production construction always calls the
unchanged ``build_cf1_runtime`` authorizer, and the process entrypoint emits no
HELLO until the observed bootstrap identity matches the caller's frozen pin.
"""

from __future__ import annotations

import importlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from bench.cf_cuda_adapter import build_cf1_runtime
from bench.cf_cuda_generator import CF1LatentPullSession
from bench.cf_cuda_session import RollingTaehvChunkDecoder, _require_verified_runtime
from bench.cf_cuda_smoke import build_png_encoder
from bench.streaming_process import ProcessStreamingBackend, WorkerProtocolError
from bench.streaming_process_protocol import worker_bundle_sha256
from bench.streaming_service import (
    BackendFatalError,
    DecodedChunk,
    StreamProtocolError,
    StreamRequest,
)


REAL_WORKER_SCRIPT = Path(__file__).with_name("cf_streaming_process_worker.py")
REAL_WORKER_BUNDLE_PATHS = (
    Path(__file__).with_name("__init__.py"),
    Path(__file__),
    Path(__file__).with_name("streaming_process.py"),
    Path(__file__).with_name("streaming_process_worker.py"),
    Path(__file__).with_name("cf_cuda_adapter.py"),
    Path(__file__).with_name("cf_cuda_generator.py"),
    Path(__file__).with_name("cf_cuda_session.py"),
    Path(__file__).with_name("cf_cuda_smoke.py"),
    Path(__file__).with_name("cf_attention_probe.py"),
    Path(__file__).with_name("cf_runtime_evidence.py"),
    Path(__file__).with_name("cf_runtime_preflight.py"),
    Path(__file__).with_name("generation_preflight.py"),
    Path(__file__).with_name("model_asset_preflight.py"),
    Path(__file__).with_name("png_validation.py"),
    Path(__file__).with_name("streaming_service.py"),
)
CF1_LATENT_FRAMES = 21
CF1_MAX_SEED = 2**32 - 1
CF1_PNG_FRAME_ENCODING_PROFILE = "png-c1-lossless-v1"
CF1_JPEG_FRAME_ENCODING_PROFILE = "jpeg-q90-cpu-v1"
CF1_BROWSER_FRAME_ENCODING_PROFILE = CF1_JPEG_FRAME_ENCODING_PROFILE
CF1_FRAME_ENCODING_PROFILES = frozenset(
    {
        CF1_PNG_FRAME_ENCODING_PROFILE,
        CF1_JPEG_FRAME_ENCODING_PROFILE,
    }
)
_CF1_FRAME_SHAPE = (3, 480, 832)
_CF1_JPEG_QUALITY = 90
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


@dataclass(frozen=True)
class AcceptanceWorkerTerminationEvidence:
    """Sanitized identity of the exact isolated worker group signaled in a test."""

    pid: int
    worker_instance_id: str
    process_group_id: int
    session_id: int
    signal_name: str


class CF1StreamingWorkerError(ValueError):
    """The real CUDA worker cannot continue or safely reuse its runtime."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_oci_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


def _device_type(value: Any) -> object:
    return getattr(value, "type", None)


def _is_valid_cf1_jpeg(payload: bytes) -> bool:
    """Validate one bounded baseline JPEG with the exact CF++1 dimensions."""

    if (
        not isinstance(payload, bytes)
        or len(payload) < 32
        or not payload.startswith(b"\xff\xd8")
        or not payload.endswith(b"\xff\xd9")
    ):
        return False
    offset = 2
    saw_sof = False
    saw_scan = False
    payload_end = len(payload) - 2
    while offset < payload_end:
        if payload[offset] != 0xFF:
            return False
        while offset < payload_end and payload[offset] == 0xFF:
            offset += 1
        if offset >= payload_end:
            return False
        marker = payload[offset]
        offset += 1
        if marker in {0x00, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            return False
        if offset + 2 > payload_end:
            return False
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > payload_end:
            return False
        segment = payload[offset + 2 : offset + segment_length]
        if marker in _JPEG_SOF_MARKERS:
            if (
                saw_sof
                or marker != 0xC0
                or len(segment) != 15
                or segment[0] != 8
                or int.from_bytes(segment[1:3], "big") != _CF1_FRAME_SHAPE[1]
                or int.from_bytes(segment[3:5], "big") != _CF1_FRAME_SHAPE[2]
                or segment[5] != 3
            ):
                return False
            saw_sof = True
        if marker == 0xDA:
            if not saw_sof or saw_scan or len(segment) < 6:
                return False
            saw_scan = True
            entropy_start = offset + segment_length
            return entropy_start < payload_end
        offset += segment_length
    return False


def build_cpu_jpeg_encoder(runtime: Any) -> Callable[[Any], Sequence[bytes]]:
    """Build the fixed q90 CPU JPEG profile from the frozen Torchvision runtime."""

    # Reuse the lossless encoder's exact runtime/Torch/Torchvision authorization.
    build_png_encoder(runtime)
    torchvision_io = importlib.import_module("torchvision.io")
    encode_jpeg = getattr(torchvision_io, "encode_jpeg", None)
    if not callable(encode_jpeg):
        raise CF1StreamingWorkerError("Torchvision JPEG encoder is unavailable")

    def encode_frames(frames: Any) -> tuple[bytes, ...]:
        try:
            shape = tuple(frames.shape)
        except (AttributeError, TypeError) as error:
            raise CF1StreamingWorkerError("CPU frame batch has no tensor shape") from error
        if (
            len(shape) != 4
            or shape[0] not in {1, 4}
            or shape[1:] != _CF1_FRAME_SHAPE
            or getattr(frames, "dtype", None) != runtime.torch.uint8
            or _device_type(getattr(frames, "device", None)) != "cpu"
        ):
            raise CF1StreamingWorkerError(
                "CPU frame batch does not match JPEG profile"
            )
        payloads: list[bytes] = []
        for index in range(shape[0]):
            try:
                encoded = encode_jpeg(
                    frames[index].contiguous(),
                    quality=_CF1_JPEG_QUALITY,
                )
                encoded_shape = tuple(encoded.shape)
                payload = encoded.numpy().tobytes()
            except Exception as error:
                raise CF1StreamingWorkerError("JPEG frame encoding failed") from error
            if (
                len(encoded_shape) != 1
                or getattr(encoded, "dtype", None) != runtime.torch.uint8
                or _device_type(getattr(encoded, "device", None)) != "cpu"
                or not _is_valid_cf1_jpeg(payload)
            ):
                raise CF1StreamingWorkerError("encoded JPEG payload is invalid")
            payloads.append(payload)
        return tuple(payloads)

    return encode_frames


def _frame_encoding_contract(
    runtime: Any,
    frame_encoding_profile: str,
) -> tuple[str, Callable[[Any], Sequence[bytes]]]:
    if frame_encoding_profile == CF1_PNG_FRAME_ENCODING_PROFILE:
        return "image/png", build_png_encoder(runtime)
    if frame_encoding_profile == CF1_JPEG_FRAME_ENCODING_PROFILE:
        return "image/jpeg", build_cpu_jpeg_encoder(runtime)
    raise CF1StreamingWorkerError("unsupported frame encoding profile")


def _payload_matches_frame_encoding_profile(
    payload: bytes,
    frame_encoding_profile: str,
) -> bool:
    if frame_encoding_profile == CF1_PNG_FRAME_ENCODING_PROFILE:
        return payload.startswith(b"\x89PNG\r\n\x1a\n") and payload.endswith(
            b"IEND\xaeB`\x82"
        )
    if frame_encoding_profile == CF1_JPEG_FRAME_ENCODING_PROFILE:
        return _is_valid_cf1_jpeg(payload)
    return False


def _validate_start_request(*, prompt: Any, seed: Any, latent_frames: Any) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise CF1StreamingWorkerError("prompt must be a non-empty string")
    try:
        prompt.encode("utf-8")
    except UnicodeError as error:
        raise CF1StreamingWorkerError("prompt must be valid UTF-8") from error
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > CF1_MAX_SEED
    ):
        raise CF1StreamingWorkerError("seed must be an unsigned 32-bit integer")
    if (
        isinstance(latent_frames, bool)
        or not isinstance(latent_frames, int)
        or latent_frames != CF1_LATENT_FRAMES
    ):
        raise CF1StreamingWorkerError("latent_frames must be exactly 21")


class CF1StreamingWorker:
    """Own one warm runtime and fresh exact generator/decoder state per job."""

    def __init__(
        self,
        runtime: Any,
        *,
        frame_encoding_profile: str = CF1_PNG_FRAME_ENCODING_PROFILE,
    ) -> None:
        self.runtime = _require_verified_runtime(runtime)
        provenance = self.runtime.provenance
        stack_sha256 = provenance.bootstrap_identity_sha256
        if not _is_sha256(stack_sha256):
            raise CF1StreamingWorkerError("runtime bootstrap identity is invalid")
        self.stack_sha256 = stack_sha256
        self.frame_encoding_profile = frame_encoding_profile
        self.frame_media_type, self._encode_frames = _frame_encoding_contract(
            self.runtime,
            frame_encoding_profile,
        )
        self._session: Any | None = None
        self._decoder: Any | None = None
        self._next_index = 0
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def active(self) -> bool:
        return self._session is not None or self._decoder is not None

    def _require_usable(self) -> None:
        if self._poisoned:
            raise CF1StreamingWorkerError("CF++1 streaming worker is poisoned")

    def _fatal(self, message: str) -> CF1StreamingWorkerError:
        self._poisoned = True
        return CF1StreamingWorkerError(message)

    def start(self, *, prompt: Any, seed: Any, latent_frames: Any) -> None:
        self._require_usable()
        _validate_start_request(
            prompt=prompt,
            seed=seed,
            latent_frames=latent_frames,
        )
        if self.active:
            raise self._fatal("CF++1 streaming worker already owns an active job")
        try:
            session = CF1LatentPullSession(
                runtime=self.runtime,
                prompt=prompt,
                seed=seed,
            )
            decoder = RollingTaehvChunkDecoder(
                runtime=self.runtime,
                torch=self.runtime.torch,
                encode_frames=self._encode_frames,
                frame_media_type=self.frame_media_type,
            )
        except BaseException:
            self._poisoned = True
            raise
        self._session = session
        self._decoder = decoder
        self._next_index = 0

    def pull(self, chunk_index: Any) -> DecodedChunk:
        self._require_usable()
        if not self.active or self._session is None or self._decoder is None:
            raise self._fatal("CF++1 streaming worker has no active job")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index != self._next_index
            or chunk_index >= CF1_LATENT_FRAMES
        ):
            raise self._fatal("NEXT chunk index is not contiguous")
        try:
            generated = self._session.pull()
            if (
                isinstance(getattr(generated, "block_index", None), bool)
                or getattr(generated, "block_index", None) != chunk_index
            ):
                raise CF1StreamingWorkerError(
                    "generated block index does not match NEXT credit"
                )
            chunk = self._decoder.decode(
                generated.denoised_latent,
                latent_ready_event=generated.latent_ready_event,
            )
            expected_frames = 1 if chunk_index == 0 else 4
            if (
                not isinstance(chunk, DecodedChunk)
                or chunk.frame_media_type != self.frame_media_type
                or chunk.frame_count != expected_frames
                or any(
                    not isinstance(payload, bytes)
                    or not _payload_matches_frame_encoding_profile(
                        payload,
                        self.frame_encoding_profile,
                    )
                    for payload in chunk.frame_payloads
                )
            ):
                raise CF1StreamingWorkerError(
                    "decoded chunk violates the CF++1 frame encoding profile"
                )
        except BaseException:
            self._poisoned = True
            raise
        self._next_index += 1
        return chunk

    def finish(self, chunk_index: Any) -> None:
        self._require_usable()
        if not self.active or self._session is None or self._decoder is None:
            raise self._fatal("CF++1 streaming worker has no active job")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index != CF1_LATENT_FRAMES
            or self._next_index != CF1_LATENT_FRAMES
        ):
            raise self._fatal("terminal completion index requires 21 clean pulls")
        try:
            self._decoder.finish()
            self._session.finish()
            if self._decoder.complete is not True or self._session.complete is not True:
                raise CF1StreamingWorkerError(
                    "CF++1 generator/decoder did not finish cleanly"
                )
        except BaseException:
            self._poisoned = True
            raise
        self._session = None
        self._decoder = None
        self._next_index = 0


def build_cf1_streaming_worker(
    *,
    expected_stack_sha256: str,
    frame_encoding_profile: str = CF1_PNG_FRAME_ENCODING_PROFILE,
) -> CF1StreamingWorker:
    """Build, synchronize, and identity-check the production CUDA engine."""

    if not _is_sha256(expected_stack_sha256):
        raise CF1StreamingWorkerError(
            "expected stack identity must be a lowercase SHA-256 digest"
        )
    runtime = build_cf1_runtime()
    worker = CF1StreamingWorker(
        runtime,
        frame_encoding_profile=frame_encoding_profile,
    )
    if worker.stack_sha256 != expected_stack_sha256:
        raise CF1StreamingWorkerError(
            "observed runtime bootstrap identity does not match the expected identity"
        )
    try:
        runtime.torch.cuda.synchronize(runtime.device)
    except BaseException:
        raise
    return worker


class CF1ProcessStreamingBackend(ProcessStreamingBackend):
    """A real-worker supervisor that refuses hidden cold starts in jobs."""

    _IMMUTABLE_LAUNCH_FIELDS = frozenset(
        {
            "stack_sha256",
            "worker_script",
            "worker_args",
            "worker_bundle_paths",
            "worker_environment",
            "require_warm_start",
            "max_latent_frames",
            "expected_worker_code_sha256",
            "frame_encoding_profile",
            "startup_timeout_s",
            "io_timeout_s",
            "reap_timeout_s",
            "registry_chunk_timeout_s",
            "registry_close_timeout_s",
            "max_header_bytes",
            "max_frame_bytes",
            "max_chunk_bytes",
            "max_prompt_bytes",
            "max_job_ids",
            "stderr_tail_bytes",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_cf1_launch_locked", False) and (
            name in self._IMMUTABLE_LAUNCH_FIELDS
            or name == "_cf1_launch_locked"
        ):
            raise AttributeError("CF++1 launch configuration is immutable")
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        expected_stack_sha256: str,
        expected_worker_code_sha256: str,
        runtime_image_index_digest: str,
        runtime_image_digest: str,
        runtime_image_config_digest: str,
        runtime_environment_root: str,
        runtime_distribution_path: str,
        runtime_wheelhouse: str,
        frame_encoding_profile: str = CF1_PNG_FRAME_ENCODING_PROFILE,
        startup_timeout_s: float = 900.0,
        io_timeout_s: float = 180.0,
        reap_timeout_s: float = 5.0,
        registry_chunk_timeout_s: float = 930.0,
        registry_close_timeout_s: float = 10.0,
    ) -> None:
        if not _is_oci_sha256(runtime_image_digest):
            raise ValueError(
                "runtime_image_digest must be a canonical sha256 OCI digest"
            )
        if (
            not _is_oci_sha256(runtime_image_index_digest)
            or not _is_oci_sha256(runtime_image_config_digest)
            or runtime_image_index_digest == runtime_image_digest
        ):
            raise ValueError("runtime OCI index/config assertions are invalid")
        runtime_paths = {
            "CF1_RUNTIME_ENVIRONMENT_ROOT": runtime_environment_root,
            "CF1_RUNTIME_DISTRIBUTION_PATH": runtime_distribution_path,
            "CF1_RUNTIME_WHEELHOUSE": runtime_wheelhouse,
        }
        if any(
            not isinstance(value, str)
            or not value
            or not Path(value).is_absolute()
            or "\x00" in value
            for value in runtime_paths.values()
        ):
            raise ValueError("runtime evidence paths must be absolute")
        if not _is_sha256(expected_worker_code_sha256):
            raise ValueError(
                "expected_worker_code_sha256 must be a lowercase SHA-256 digest"
            )
        if frame_encoding_profile not in CF1_FRAME_ENCODING_PROFILES:
            raise ValueError("unsupported frame_encoding_profile")
        self.expected_worker_code_sha256 = expected_worker_code_sha256
        self.frame_encoding_profile = frame_encoding_profile
        super().__init__(
            stack_sha256=expected_stack_sha256,
            worker_script=REAL_WORKER_SCRIPT,
            worker_args=(
                "--worker-code-sha256",
                expected_worker_code_sha256,
                "--frame-encoding-profile",
                frame_encoding_profile,
            ),
            worker_bundle_paths=REAL_WORKER_BUNDLE_PATHS,
            worker_environment={
                "PYTHONUNBUFFERED": "1",
                "CF1_RUNTIME_IMAGE_INDEX_DIGEST": runtime_image_index_digest,
                "CF1_RUNTIME_IMAGE_DIGEST": runtime_image_digest,
                "CF1_RUNTIME_IMAGE_CONFIG_DIGEST": runtime_image_config_digest,
                **runtime_paths,
            },
            require_warm_start=True,
            startup_timeout_s=startup_timeout_s,
            io_timeout_s=io_timeout_s,
            reap_timeout_s=reap_timeout_s,
            registry_chunk_timeout_s=registry_chunk_timeout_s,
            registry_close_timeout_s=registry_close_timeout_s,
            max_latent_frames=CF1_LATENT_FRAMES,
        )
        self._cf1_launch_locked = True

    def _validated_frame_media_type(
        self,
        value: object,
        payloads: Sequence[bytes],
    ) -> str:
        """Validate only the encoding profile bound into this worker's boot."""

        if self.frame_encoding_profile == CF1_PNG_FRAME_ENCODING_PROFILE:
            return super()._validated_frame_media_type(value, payloads)
        if self.frame_encoding_profile != CF1_JPEG_FRAME_ENCODING_PROFILE:
            raise WorkerProtocolError("worker frame encoding profile is invalid")
        if value != "image/jpeg":
            raise WorkerProtocolError("worker frame media type is not image/jpeg")
        if any(not _is_valid_cf1_jpeg(payload) for payload in payloads):
            raise WorkerProtocolError("worker emitted an invalid JPEG payload")
        return "image/jpeg"

    async def _spawn_worker(self) -> None:
        observed_worker_code_sha256 = worker_bundle_sha256(
            self.worker_script,
            self.worker_bundle_paths,
        )
        if observed_worker_code_sha256 != self.expected_worker_code_sha256:
            raise WorkerProtocolError(
                "worker bundle does not match expected digest"
            )
        await super()._spawn_worker()

    async def terminate_idle_worker_for_acceptance(
        self,
        *,
        expected_pid: int,
        expected_worker_instance_id: str,
    ) -> AcceptanceWorkerTerminationEvidence:
        """SIGKILL one exact idle worker group without consuming its stale state.

        This is deliberately narrower than the supervisor's normal kill path.  It
        exists only so an acceptance run can make the *next* ordinary job observe
        the dead persistent child and exercise the normal poison/reap path.
        """

        if (
            isinstance(expected_pid, bool)
            or not isinstance(expected_pid, int)
            or expected_pid <= 0
        ):
            raise StreamProtocolError("expected_pid must be a positive integer")
        if (
            not isinstance(expected_worker_instance_id, str)
            or not expected_worker_instance_id
        ):
            raise StreamProtocolError(
                "expected_worker_instance_id must be a non-empty string"
            )

        async with self._claim_lock:
            if self._closed:
                raise BackendFatalError("streaming worker is closed")
            if self._poisoned:
                raise BackendFatalError("streaming worker is poisoned")
            if self._active_job_id is not None:
                raise StreamProtocolError(
                    "acceptance termination requires an idle streaming worker"
                )
            if self._state != "ready":
                raise StreamProtocolError(
                    "acceptance termination requires a ready streaming worker"
                )

            process = self._process
            if process is None or self._socket is None:
                raise StreamProtocolError(
                    "acceptance termination requires a live streaming worker"
                )
            observed_pid = process.pid
            observed_worker_instance_id = self._worker_instance_id
            if observed_pid != expected_pid:
                raise StreamProtocolError("acceptance worker PID does not match")
            if observed_worker_instance_id != expected_worker_instance_id:
                raise StreamProtocolError(
                    "acceptance worker instance does not match"
                )
            if process.poll() is not None:
                raise StreamProtocolError(
                    "acceptance termination requires a live streaming worker"
                )

            try:
                process_group_id = os.getpgid(observed_pid)
                session_id = os.getsid(observed_pid)
            except OSError as error:
                raise StreamProtocolError(
                    "acceptance worker process group could not be verified"
                ) from error
            if process_group_id != observed_pid or session_id != observed_pid:
                raise StreamProtocolError(
                    "acceptance worker does not own its process group and session"
                )

            try:
                os.killpg(observed_pid, signal.SIGKILL)
            except OSError as error:
                raise StreamProtocolError(
                    "acceptance worker process group could not be signaled"
                ) from error
            return AcceptanceWorkerTerminationEvidence(
                pid=observed_pid,
                worker_instance_id=observed_worker_instance_id,
                process_group_id=process_group_id,
                session_id=session_id,
                signal_name="SIGKILL",
            )

    def stream(self, request: StreamRequest):
        if self._closed:
            raise BackendFatalError("streaming worker is closed")
        if self._poisoned:
            raise BackendFatalError("streaming worker is poisoned")
        if not isinstance(request, StreamRequest):
            raise StreamProtocolError("request must be a StreamRequest")
        try:
            _validate_start_request(
                prompt=request.prompt,
                seed=request.seed,
                latent_frames=request.latent_frames,
            )
        except CF1StreamingWorkerError as error:
            raise StreamProtocolError(str(error)) from error
        if self._state == "stopped":
            raise StreamProtocolError(
                "real CF++1 worker must be explicitly prewarmed"
            )
        if self._state not in {"ready", "busy"}:
            raise StreamProtocolError("real CF++1 worker is not ready")
        return super().stream(request)


def build_cf1_process_streaming_backend(
    *,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    runtime_image_index_digest: str,
    runtime_image_digest: str,
    runtime_image_config_digest: str,
    runtime_environment_root: str,
    runtime_distribution_path: str,
    runtime_wheelhouse: str,
    frame_encoding_profile: str = CF1_PNG_FRAME_ENCODING_PROFILE,
) -> CF1ProcessStreamingBackend:
    """Return the exact cold supervisor; callers must await ``warm()``."""

    return CF1ProcessStreamingBackend(
        expected_stack_sha256=expected_stack_sha256,
        expected_worker_code_sha256=expected_worker_code_sha256,
        runtime_image_index_digest=runtime_image_index_digest,
        runtime_image_digest=runtime_image_digest,
        runtime_image_config_digest=runtime_image_config_digest,
        runtime_environment_root=runtime_environment_root,
        runtime_distribution_path=runtime_distribution_path,
        runtime_wheelhouse=runtime_wheelhouse,
        frame_encoding_profile=frame_encoding_profile,
    )
