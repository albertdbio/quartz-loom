"""Exact per-block CF++1 latent generation for the dedicated CUDA worker.

This ports the generation portion of the pinned upstream inference loop and
the recovered H100 rolling runner.  It deliberately yields only after the
clean context-cache refresh for a block has completed on the generation
stream.  The caller owns decoding and transport.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bench.cf_cuda_session import CudaSessionError, _require_verified_runtime


_BLOCK_COUNT = 21
_LATENT_SHAPE = (1, 1, 16, 60, 104)
_NOISE_SHAPE = (1, _BLOCK_COUNT, 16, 60, 104)
_PROMPT_EMBED_SHAPE = (1, 512, 4096)
_TRANSFORMER_BLOCK_COUNT = 30
_FRAME_SEQUENCE_LENGTH = 1560
_FINAL_CACHE_END = _BLOCK_COUNT * _FRAME_SEQUENCE_LENGTH
_MAX_PROMPT_UTF8_BYTES = 32 * 1024
_MAX_NUMPY_SEED = 2**32 - 1


class CudaGenerationError(ValueError):
    """The exact CF++1 pull session cannot continue safely."""


@dataclass(frozen=True)
class GeneratedCF1Latent:
    block_index: int
    denoised_latent: Any
    latent_ready_event: Any


def _device_identity(value: Any) -> tuple[object, object]:
    return (getattr(value, "type", None), getattr(value, "index", None))


def _seed_python_and_numpy(seed: int) -> None:
    """Reproduce the non-Torch portion of upstream ``set_seed`` lazily."""

    random.seed(seed)
    numpy = importlib.import_module("numpy")
    numpy.random.seed(seed)


def _validate_request(prompt: Any, seed: Any) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise CudaGenerationError("prompt must be a non-empty string")
    try:
        encoded_prompt = prompt.encode("utf-8")
    except UnicodeError as error:
        raise CudaGenerationError("prompt must be valid UTF-8") from error
    if len(encoded_prompt) > _MAX_PROMPT_UTF8_BYTES:
        raise CudaGenerationError("prompt exceeds the bounded UTF-8 size")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > _MAX_NUMPY_SEED
    ):
        raise CudaGenerationError("seed must be an unsigned 32-bit integer")


def _schedule_length(value: Any, expected: int, label: str) -> None:
    if value is None:
        raise CudaGenerationError(f"{label} is missing")
    try:
        length = len(value)
    except (TypeError, ValueError) as error:
        raise CudaGenerationError(f"{label} is invalid") from error
    if length != expected:
        raise CudaGenerationError(f"{label} length does not match CF++1")


def _validate_pipeline_contract(pipeline: Any) -> None:
    if getattr(pipeline, "num_frame_per_block", None) != 1:
        raise CudaGenerationError("num_frame_per_block does not match CF++1")
    if getattr(pipeline, "independent_first_frame", None) is not False:
        raise CudaGenerationError("independent_first_frame does not match CF++1")
    if getattr(pipeline, "num_transformer_blocks", None) != _TRANSFORMER_BLOCK_COUNT:
        raise CudaGenerationError("transformer block count does not match CF++1")
    if getattr(pipeline, "frame_seq_length", None) != _FRAME_SEQUENCE_LENGTH:
        raise CudaGenerationError("frame sequence length does not match CF++1")
    if getattr(pipeline, "local_attn_size", None) != -1:
        raise CudaGenerationError("local attention size does not match CF++1")
    context_noise = getattr(getattr(pipeline, "args", None), "context_noise", None)
    if isinstance(context_noise, bool) or context_noise != 0:
        raise CudaGenerationError("context noise does not match CF++1")
    _schedule_length(
        getattr(pipeline, "denoising_step_list_first_chunk", None),
        4,
        "first-chunk denoising schedule",
    )
    _schedule_length(
        getattr(pipeline, "denoising_step_list", None),
        1,
        "ordinary denoising schedule",
    )
    for field in ("generator", "text_encoder"):
        if not callable(getattr(pipeline, field, None)):
            raise CudaGenerationError(f"pipeline {field} is unavailable")
    if getattr(pipeline, "scheduler", None) is None or not callable(
        getattr(pipeline.scheduler, "add_noise", None)
    ):
        raise CudaGenerationError("pipeline scheduler is unavailable")
    if not callable(getattr(pipeline, "_initialize_kv_cache", None)) or not callable(
        getattr(pipeline, "_initialize_crossattn_cache", None)
    ):
        raise CudaGenerationError("pipeline cache initializers are unavailable")


def _validate_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
    label: str,
) -> None:
    try:
        observed_shape = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise CudaGenerationError(f"{label} has no tensor shape") from error
    if observed_shape != shape:
        raise CudaGenerationError(f"{label} shape does not match CF++1")
    if getattr(value, "dtype", None) != dtype:
        raise CudaGenerationError(f"{label} dtype does not match CF++1")
    if _device_identity(getattr(value, "device", None)) != _device_identity(device):
        raise CudaGenerationError(f"{label} device does not match CF++1")


def _cache_entries(pipeline: Any) -> tuple[list[Any], list[Any]]:
    kv_cache = getattr(pipeline, "kv_cache1", None)
    crossattn_cache = getattr(pipeline, "crossattn_cache", None)
    if (
        not isinstance(kv_cache, list)
        or len(kv_cache) != _TRANSFORMER_BLOCK_COUNT
        or not isinstance(crossattn_cache, list)
        or len(crossattn_cache) != _TRANSFORMER_BLOCK_COUNT
    ):
        raise CudaGenerationError("pipeline cache inventory does not match CF++1")
    return kv_cache, crossattn_cache


class CF1LatentPullSession:
    """Own one exact 21-block CF++1 T2V generation session.

    Once initialization starts mutating CUDA/RNG/cache state, any failure is
    fatal to reuse.  A poisoned session intentionally retains its exclusive
    pipeline lease so the process owner must kill and reap the worker.
    """

    def __init__(self, *, runtime: Any, prompt: str, seed: int) -> None:
        _validate_request(prompt, seed)
        try:
            runtime = _require_verified_runtime(runtime)
        except CudaSessionError as error:
            raise CudaGenerationError("verified CF++1 runtime is required") from error
        pipeline = runtime.pipeline
        _validate_pipeline_contract(pipeline)
        if hasattr(pipeline, "_cf1_pull_session_poisoned"):
            raise CudaGenerationError("pipeline is poisoned and must not be reused")
        if hasattr(pipeline, "_cf1_pull_session_owner"):
            raise CudaGenerationError("pipeline already owns a pull session")

        self.runtime = runtime
        self.pipeline = pipeline
        self.torch = runtime.torch
        self.prompt = prompt
        self.seed = seed
        self._owner = object()
        self._block_index = 0
        self._poisoned = False
        self._finished = False
        self._noise: Any | None = None
        self._conditional_dict: dict[str, Any] | None = None
        self._generation_stream: Any | None = None
        pipeline._cf1_pull_session_owner = self._owner

        try:
            with self.torch.cuda.device(runtime.device):
                self._generation_stream = self.torch.cuda.default_stream(
                    device=runtime.device
                )
                with self.torch.cuda.stream(self._generation_stream):
                    _seed_python_and_numpy(seed)
                    self.torch.manual_seed(seed)
                    self.torch.cuda.manual_seed_all(seed)
                    initial_noise_generator = self.torch.Generator(
                        device=runtime.device
                    ).manual_seed(seed)
                    self._noise = self.torch.randn(
                        _NOISE_SHAPE,
                        device=runtime.device,
                        dtype=self.torch.bfloat16,
                        generator=initial_noise_generator,
                    )
                    _validate_tensor(
                        self._noise,
                        shape=_NOISE_SHAPE,
                        dtype=self.torch.bfloat16,
                        device=runtime.device,
                        label="initial noise",
                    )
                    self._conditional_dict = self._encode_prompt()
                    self._reset_caches()
        except BaseException as error:
            self._poisoned = True
            pipeline._cf1_pull_session_poisoned = True
            if not isinstance(error, Exception):
                raise
            if isinstance(error, CudaGenerationError):
                raise CudaGenerationError(
                    "CF++1 pull session initialization failed"
                ) from error
            raise CudaGenerationError(
                "CF++1 pull session initialization failed"
            ) from error

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def complete(self) -> bool:
        return self._block_index == _BLOCK_COUNT and not self._poisoned

    def _require_ownership(self) -> None:
        if hasattr(self.pipeline, "_cf1_pull_session_poisoned"):
            self._poisoned = True
            raise CudaGenerationError("pipeline is poisoned and must not be reused")
        if getattr(self.pipeline, "_cf1_pull_session_owner", None) is not self._owner:
            self._poisoned = True
            self.pipeline._cf1_pull_session_poisoned = True
            raise CudaGenerationError("CF++1 pull session ownership changed")

    def _encode_prompt(self) -> dict[str, Any]:
        conditional = self.pipeline.text_encoder(text_prompts=[self.prompt])
        if not isinstance(conditional, Mapping) or set(conditional) != {
            "prompt_embeds"
        }:
            raise CudaGenerationError("text conditioning does not match CF++1")
        try:
            prompt_embeds = conditional["prompt_embeds"].to(
                dtype=self.torch.bfloat16
            )
        except Exception as error:
            raise CudaGenerationError("text conditioning could not be cast") from error
        _validate_tensor(
            prompt_embeds,
            shape=_PROMPT_EMBED_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.runtime.device,
            label="prompt embeddings",
        )
        return {"prompt_embeds": prompt_embeds}

    def _reset_caches(self) -> None:
        if self.pipeline.kv_cache1 is None:
            self.pipeline._initialize_kv_cache(
                batch_size=1,
                dtype=self.torch.bfloat16,
                device=self.runtime.device,
            )
            self.pipeline._initialize_crossattn_cache(
                batch_size=1,
                dtype=self.torch.bfloat16,
                device=self.runtime.device,
            )
            _cache_entries(self.pipeline)
            return

        kv_cache, crossattn_cache = _cache_entries(self.pipeline)
        for cache in crossattn_cache:
            if not isinstance(cache, Mapping) or "is_init" not in cache:
                raise CudaGenerationError("cross-attention cache is invalid")
            cache["is_init"] = False
        for cache in kv_cache:
            if not isinstance(cache, Mapping) or not {
                "global_end_index",
                "local_end_index",
            }.issubset(cache):
                raise CudaGenerationError("KV cache is invalid")
            for field in ("global_end_index", "local_end_index"):
                zero = getattr(cache[field], "zero_", None)
                if not callable(zero):
                    raise CudaGenerationError("KV cache index is invalid")
                zero()

    def _generator_forward(self, noisy_input: Any, timestep: Any) -> Any:
        result = self.pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=self._conditional_dict,
            timestep=timestep,
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=self._block_index * _FRAME_SEQUENCE_LENGTH,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise CudaGenerationError("generator output does not match CF++1")
        denoised = result[1]
        _validate_tensor(
            denoised,
            shape=_LATENT_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.runtime.device,
            label="denoised latent",
        )
        return denoised

    def pull(self) -> GeneratedCF1Latent:
        if self._poisoned:
            raise CudaGenerationError("CF++1 pull session is poisoned")
        if self._finished or self._block_index >= _BLOCK_COUNT:
            raise CudaGenerationError("CF++1 pull session is complete")
        self._require_ownership()
        block_index = self._block_index
        try:
            with self.torch.cuda.device(self.runtime.device):
                with self.torch.cuda.stream(self._generation_stream):
                    noisy_input = self._noise[:, block_index : block_index + 1]
                    current_steps = (
                        self.pipeline.denoising_step_list_first_chunk
                        if block_index == 0
                        else self.pipeline.denoising_step_list
                    )
                    denoised_pred = None
                    timestep = None
                    for step_index, current_timestep in enumerate(current_steps):
                        timestep = self.torch.ones(
                            (1, 1),
                            device=self.runtime.device,
                            dtype=self.torch.int64,
                        ) * current_timestep
                        denoised_pred = self._generator_forward(
                            noisy_input,
                            timestep,
                        )
                        if step_index < len(current_steps) - 1:
                            next_timestep = current_steps[step_index + 1]
                            denoised_flat = denoised_pred.flatten(0, 1)
                            transition_noise = self.torch.randn_like(
                                denoised_pred.flatten(0, 1)
                            )
                            noisy_input = self.pipeline.scheduler.add_noise(
                                denoised_flat,
                                transition_noise,
                                next_timestep
                                * self.torch.ones(
                                    (1,),
                                    device=self.runtime.device,
                                    dtype=self.torch.long,
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                            _validate_tensor(
                                noisy_input,
                                shape=_LATENT_SHAPE,
                                dtype=self.torch.bfloat16,
                                device=self.runtime.device,
                                label="scheduled noisy latent",
                            )
                    if denoised_pred is None or timestep is None:
                        raise CudaGenerationError("denoising schedule is empty")
                    released_latent = denoised_pred.detach().clone()
                    _validate_tensor(
                        released_latent,
                        shape=_LATENT_SHAPE,
                        dtype=self.torch.bfloat16,
                        device=self.runtime.device,
                        label="released denoised latent",
                    )
                    context_timestep = (
                        self.torch.ones_like(timestep)
                        * self.pipeline.args.context_noise
                    )
                    self._generator_forward(denoised_pred, context_timestep)
                    latent_ready = self.torch.cuda.Event()
                    latent_ready.record(self._generation_stream)
            self._block_index += 1
            return GeneratedCF1Latent(
                block_index=block_index,
                denoised_latent=released_latent,
                latent_ready_event=latent_ready,
            )
        except BaseException as error:
            self._poisoned = True
            self.pipeline._cf1_pull_session_poisoned = True
            if not isinstance(error, Exception):
                raise
            raise CudaGenerationError("CF++1 latent generation failed") from error

    def _validate_final_cache_indices(self) -> None:
        kv_cache, _crossattn_cache = _cache_entries(self.pipeline)
        for cache in kv_cache:
            if not isinstance(cache, Mapping):
                raise CudaGenerationError("KV cache is invalid")
            for field in ("global_end_index", "local_end_index"):
                try:
                    value = cache[field].item()
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise CudaGenerationError("KV cache index is invalid") from error
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value != _FINAL_CACHE_END
                ):
                    raise CudaGenerationError(
                        "final KV cache index does not match CF++1"
                    )

    def finish(self) -> None:
        if self._poisoned:
            raise CudaGenerationError("CF++1 pull session is poisoned")
        if self._finished:
            raise CudaGenerationError("CF++1 pull session is already finished")
        if self._block_index != _BLOCK_COUNT:
            raise CudaGenerationError("CF++1 pull session is incomplete")
        self._require_ownership()
        try:
            self._validate_final_cache_indices()
        except BaseException as error:
            self._poisoned = True
            self.pipeline._cf1_pull_session_poisoned = True
            if not isinstance(error, Exception):
                raise
            raise CudaGenerationError("CF++1 pull session finalization failed") from error
        self._noise = None
        self._conditional_dict = None
        delattr(self.pipeline, "_cf1_pull_session_owner")
        self._finished = True
