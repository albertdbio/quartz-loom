"""Pinned CF++1 + rolling-TAEHV runtime bootstrap for the CUDA worker.

Third-party/CUDA imports are deferred until after the immutable asset preflight.
The generator is constructed from the pinned Wan architecture config and then
strictly loaded from the CF++ checkpoint, so boot never opens the stock Wan
diffusion checkpoint or Wan VAE checkpoint.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from bench.cf_attention_probe import attention_probe_report
from bench.cf_runtime_evidence import (
    DEFAULT_RUNTIME_EVIDENCE_PATH,
    RuntimeEvidenceError,
    RuntimeEvidenceSnapshot,
    load_runtime_evidence_snapshot,
    runtime_evidence_locked_identities,
)
from bench.cf_runtime_preflight import (
    CF1_RUNTIME_ID,
    CF1_RUNTIME_LOCK_SHA256,
    CF1_TOKENIZER_SENTINEL_SHA256,
    DEFAULT_RUNTIME_LOCK_PATH,
    RuntimeLockSnapshot,
    RuntimePreflightError,
    RuntimePreflightIdentity,
    load_runtime_lock_snapshot,
    preflight_current_runtime,
    validate_current_host_capacity,
    validate_cf1_tokenizer_sentinel,
    validate_loaded_cuda_capacity,
)
from bench.generation_preflight import (
    normalize_fsdp_generator_state_dict,
    validate_strict_checkpoint_keys,
)
from bench.model_asset_preflight import (
    AssetLockSnapshot,
    DEFAULT_CHECKOUT,
    DEFAULT_LOCK_PATH,
    load_asset_lock,
    load_asset_lock_snapshot,
    verify_model_assets_snapshot,
)


_MODEL_CONFIG_FIELDS = {
    "dim",
    "eps",
    "ffn_dim",
    "freq_dim",
    "in_dim",
    "model_type",
    "num_heads",
    "num_layers",
    "out_dim",
    "text_len",
}
_STACK_ID = "cf1-rolling-taehv-v1"
_CF1_DENOISING_STEPS = [1000]
_CF1_FIRST_CHUNK_DENOISING_STEPS = [1000, 750, 500, 250]
_CF1_TIMESTEP_SHIFT = 5.0
_EXPECTED_EFFECTIVE_CONFIG_SHA256 = (
    "54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06"
)
_EXPECTED_ASSET_LOCK_SHA256 = (
    "0aee8671f8e3b30286b689a16f6f4a355f917772c16e599cec75a49e89057967"
)

# Public identity constants are consumed by the session layer. Keeping one
# definition prevents the decoder from silently accepting a nearby stack.
CF1_STACK_ID = _STACK_ID
CF1_EFFECTIVE_CONFIG_SHA256 = _EXPECTED_EFFECTIVE_CONFIG_SHA256
CF1_ASSET_LOCK_SHA256 = _EXPECTED_ASSET_LOCK_SHA256
CF1_SOURCE_COMMIT = "8db419e341e5fc52542c0b2c4542728420ddfb4a"


class RuntimeBootstrapError(ValueError):
    """The pinned runtime cannot be constructed without weakening a guard."""


@dataclass(frozen=True)
class CF1RuntimePaths:
    checkout: Path
    default_config: Path
    candidate_config: Path
    model_config: Path
    generator_checkpoint: Path
    text_encoder_checkpoint: Path
    tokenizer_directory: Path
    tokenizer_files: tuple[Path, ...]
    taehv_checkpoint: Path


@dataclass(frozen=True)
class CF1AssetIdentity:
    id: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CF1BootstrapProvenance:
    """Byte identity of the locally verified bootstrap, before worker deps."""

    stack_id: str
    source_commit: str
    asset_lock_sha256: str
    runtime_lock_sha256: str
    runtime_evidence_sha256: str
    static_environment_sha256: str
    runtime_environment_sha256: str
    runtime_native_environment_sha256: str
    native_identity_sha256: str
    attention_probe_identity_sha256: str
    effective_config_sha256: str
    tokenizer_sentinel_sha256: str
    attention_backend: str
    guard_bundle_sha256: str
    bootstrap_identity_sha256: str
    assets: tuple[CF1AssetIdentity, ...]


@dataclass(frozen=True)
class RuntimeBindings:
    """Late-bound upstream/CUDA symbols; injectable for CPU unit tests."""

    torch: Any
    OmegaConf: Any
    CausalWanModel: Any
    WanDiffusionWrapper: Any
    FlowMatchScheduler: Any
    WanTextEncoder: Any
    HuggingfaceTokenizer: Any
    umt5_xxl: Any
    CausalInferencePipeline: Any
    TAEHV: Any
    gpu: Any
    attention_backend: str


@dataclass(frozen=True)
class CF1Runtime:
    pipeline: Any
    taehv: Any
    effective_config: Any
    effective_config_sha256: str
    device: Any
    torch: Any
    attention_backend: str
    runtime_identity: RuntimePreflightIdentity | None = None
    runtime_native_environment_sha256: str | None = None
    native_identity_sha256: str | None = None
    attention_probe_identity_sha256: str | None = None
    tokenizer_sentinel_sha256: str | None = None
    provenance: CF1BootstrapProvenance | None = None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeBootstrapError(f"duplicate model config key: {key}")
        value[key] = item
    return value


def _load_model_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeBootstrapError("Wan model config could not be parsed") from error
    if not isinstance(value, dict):
        raise RuntimeBootstrapError("Wan model config must be an object")
    metadata = {"_class_name", "_diffusers_version"}
    if set(value) != _MODEL_CONFIG_FIELDS | metadata:
        raise RuntimeBootstrapError("Wan model config fields do not match the pin")
    if value["_class_name"] != "WanModel":
        raise RuntimeBootstrapError("Wan model config class does not match WanModel")
    return {field: value[field] for field in sorted(_MODEL_CONFIG_FIELDS)}


def _asset_paths(
    lock_or_path: Mapping[str, Any] | Path, checkout: Path
) -> CF1RuntimePaths:
    lock = (
        load_asset_lock(lock_or_path)
        if isinstance(lock_or_path, Path)
        else lock_or_path
    )
    if lock["stack_id"] != _STACK_ID:
        raise RuntimeBootstrapError(
            f"CF++1 adapter requires stack_id {_STACK_ID}"
        )
    by_id = {asset["id"]: asset["relative_path"] for asset in lock["assets"]}
    required = {
        "default-config",
        "cf1-config",
        "wan-config",
        "cf1-generator",
        "wan-text-encoder",
        "wan-special-tokens",
        "wan-spiece",
        "wan-tokenizer",
        "wan-tokenizer-config",
        "taehv-weight",
    }
    missing = sorted(required - set(by_id))
    if missing:
        raise RuntimeBootstrapError(
            f"asset lock is missing runtime asset id {missing[0]}"
        )
    checkout = checkout.resolve()

    def pinned(asset_id: str) -> Path:
        return checkout / by_id[asset_id]

    tokenizer_assets = (
        pinned("wan-special-tokens"),
        pinned("wan-spiece"),
        pinned("wan-tokenizer"),
        pinned("wan-tokenizer-config"),
    )
    tokenizer_parents = {path.parent for path in tokenizer_assets}
    if len(tokenizer_parents) != 1:
        raise RuntimeBootstrapError("Wan tokenizer assets must share one directory")

    return CF1RuntimePaths(
        checkout=checkout,
        default_config=pinned("default-config"),
        candidate_config=pinned("cf1-config"),
        model_config=pinned("wan-config"),
        generator_checkpoint=pinned("cf1-generator"),
        text_encoder_checkpoint=pinned("wan-text-encoder"),
        tokenizer_directory=tokenizer_parents.pop(),
        tokenizer_files=tokenizer_assets,
        taehv_checkpoint=pinned("taehv-weight"),
    )


def _validate_tokenizer_inventory(paths: CF1RuntimePaths) -> None:
    expected = {path.name for path in paths.tokenizer_files}
    try:
        entries = tuple(paths.tokenizer_directory.iterdir())
    except OSError as error:
        raise RuntimeBootstrapError("tokenizer inventory could not be read") from error
    actual = {entry.name for entry in entries}
    if actual != expected or any(
        entry.parent != paths.tokenizer_directory or not entry.is_file()
        for entry in entries
    ):
        raise RuntimeBootstrapError(
            "tokenizer inventory does not match the four pinned files"
        )


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_checkout_module(module: Any, checkout: Path, label: str) -> Path:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeBootstrapError(f"upstream module {label} has no source path")
    resolved = Path(module_file).resolve()
    try:
        relative = resolved.relative_to(checkout.resolve())
    except ValueError as error:
        raise RuntimeBootstrapError(
            f"upstream module {label} resolved outside verified checkout"
        ) from error
    if not relative.parts or not resolved.is_file():
        raise RuntimeBootstrapError(
            f"upstream module {label} is not a verified checkout file"
        )
    return resolved


def _disable_upstream_bytecode_writes() -> None:
    # The post-load source inventory intentionally rejects ignored pyc files.
    # Keep this process-wide in the dedicated worker because upstream modules
    # can import lazily after bootstrap returns.
    sys.dont_write_bytecode = True


def _active_attention_backend(attention_module: Any) -> str:
    fa3 = getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", None)
    fa2 = getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", None)
    if not isinstance(fa3, bool) or not isinstance(fa2, bool):
        raise RuntimeBootstrapError("upstream attention backend flags are invalid")
    if fa3:
        return "flash-attention-3"
    if fa2:
        return "flash-attention-2"
    raise RuntimeBootstrapError(
        "the pinned cross-attention path requires FlashAttention 2 or 3"
    )


def _load_runtime_bindings(
    checkout: Path, cuda_device_index: int
) -> RuntimeBindings:
    _disable_upstream_bytecode_writes()
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    with _working_directory(checkout):
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeBootstrapError("CUDA is unavailable")
        device_count = torch.cuda.device_count()
        if device_count != 1:
            raise RuntimeBootstrapError("CF++1 requires exactly one visible CUDA GPU")
        if cuda_device_index >= device_count:
            raise RuntimeBootstrapError("CUDA device index is unavailable")
        torch.cuda.set_device(cuda_device_index)
        if torch.cuda.current_device() != cuda_device_index:
            raise RuntimeBootstrapError("CUDA device selection did not take effect")
        omegaconf_module = importlib.import_module("omegaconf")
        upstream_names = (
            "demo_utils.memory",
            "demo_utils.taehv",
            "pipeline",
            "utils.scheduler",
            "utils.wan_wrapper",
            "wan.modules.attention",
            "wan.modules.causal_model",
            "wan.modules.t5",
            "wan.modules.tokenizers",
        )
        cached = [name for name in upstream_names if name in sys.modules]
        if cached:
            raise RuntimeBootstrapError(
                f"upstream module was already cached: {cached[0]}"
            )
        upstream_modules = {
            "demo_utils.memory": importlib.import_module("demo_utils.memory"),
            "demo_utils.taehv": importlib.import_module("demo_utils.taehv"),
            "pipeline": importlib.import_module("pipeline"),
            "utils.scheduler": importlib.import_module("utils.scheduler"),
            "utils.wan_wrapper": importlib.import_module("utils.wan_wrapper"),
            "wan.modules.attention": importlib.import_module(
                "wan.modules.attention"
            ),
            "wan.modules.causal_model": importlib.import_module(
                "wan.modules.causal_model"
            ),
            "wan.modules.t5": importlib.import_module("wan.modules.t5"),
            "wan.modules.tokenizers": importlib.import_module(
                "wan.modules.tokenizers"
            ),
        }
        for label, module in upstream_modules.items():
            _require_checkout_module(module, checkout, label)
        expected_gpu = torch.device(f"cuda:{cuda_device_index}")
        if upstream_modules["demo_utils.memory"].gpu != expected_gpu:
            raise RuntimeBootstrapError("upstream CUDA device does not match selection")
        attention_backend = _active_attention_backend(
            upstream_modules["wan.modules.attention"]
        )

    return RuntimeBindings(
        torch=torch,
        OmegaConf=omegaconf_module.OmegaConf,
        CausalWanModel=upstream_modules[
            "wan.modules.causal_model"
        ].CausalWanModel,
        WanDiffusionWrapper=upstream_modules[
            "utils.wan_wrapper"
        ].WanDiffusionWrapper,
        FlowMatchScheduler=upstream_modules["utils.scheduler"].FlowMatchScheduler,
        WanTextEncoder=upstream_modules["utils.wan_wrapper"].WanTextEncoder,
        HuggingfaceTokenizer=upstream_modules[
            "wan.modules.tokenizers"
        ].HuggingfaceTokenizer,
        umt5_xxl=upstream_modules["wan.modules.t5"].umt5_xxl,
        CausalInferencePipeline=upstream_modules["pipeline"].CausalInferencePipeline,
        TAEHV=upstream_modules["demo_utils.taehv"].TAEHV,
        gpu=upstream_modules["demo_utils.memory"].gpu,
        attention_backend=attention_backend,
    )


def _build_checkpoint_only_generator(
    bindings: RuntimeBindings,
    model: Any,
    *,
    timestep_shift: float,
) -> Any:
    """Construct the upstream wrapper without its ``from_pretrained`` path."""

    class CheckpointOnlyWanDiffusionWrapper(bindings.WanDiffusionWrapper):
        def __init__(self) -> None:
            bindings.torch.nn.Module.__init__(self)
            self.model = model.eval()
            self.uniform_timestep = False
            self.scheduler = bindings.FlowMatchScheduler(
                shift=timestep_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.scheduler.set_timesteps(1000, training=True)
            self.seq_len = 32760
            self.post_init()

    return CheckpointOnlyWanDiffusionWrapper()


def _latent_passthrough(bindings: RuntimeBindings) -> Any:
    class LatentPassthrough(bindings.torch.nn.Module):
        def decode_to_pixel(self, latents: Any, use_cache: bool = False) -> Any:
            del use_cache
            return latents

    return LatentPassthrough()


def _build_verified_tokenizer(
    bindings: RuntimeBindings,
    paths: CF1RuntimePaths,
) -> tuple[Any, str]:
    """Construct the exact local tokenizer and prove its production behavior."""

    tokenizer = bindings.HuggingfaceTokenizer(
        name=str(paths.tokenizer_directory),
        seq_len=512,
        clean="whitespace",
        local_files_only=True,
        use_fast=True,
    )
    try:
        digest = validate_cf1_tokenizer_sentinel(
            tokenizer,
            expected_dtype=bindings.torch.int64,
        )
    except RuntimePreflightError as error:
        raise RuntimeBootstrapError(
            "Wan tokenizer sentinel does not match the runtime pin"
        ) from error
    return tokenizer, digest


def _build_pinned_text_encoder(
    bindings: RuntimeBindings,
    paths: CF1RuntimePaths,
    tokenizer: Any,
) -> Any:
    """Construct the upstream encoder contract with a safe explicit load."""

    class PinnedWanTextEncoder(bindings.WanTextEncoder):
        def __init__(self) -> None:
            bindings.torch.nn.Module.__init__(self)
            self.text_encoder = bindings.umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=bindings.torch.float32,
                device=bindings.torch.device("cpu"),
            ).eval().requires_grad_(False)
            state_dict = bindings.torch.load(
                paths.text_encoder_checkpoint,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            if not isinstance(state_dict, Mapping):
                raise RuntimeBootstrapError(
                    "Wan text encoder checkpoint must be a state dictionary"
                )
            validate_strict_checkpoint_keys(
                expected_keys=self.text_encoder.state_dict().keys(),
                checkpoint_keys=state_dict.keys(),
            )
            self.text_encoder.load_state_dict(
                state_dict,
                strict=True,
                assign=True,
            )
            self.tokenizer = tokenizer

    return PinnedWanTextEncoder()


def _config_value(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        if field not in config:
            raise RuntimeBootstrapError(f"effective config requires {field}")
        return config[field]
    try:
        return getattr(config, field)
    except (AttributeError, KeyError) as error:
        raise RuntimeBootstrapError(f"effective config requires {field}") from error


def _timestep_list(value: Any, field: str) -> list[int]:
    if isinstance(value, (str, bytes)):
        raise RuntimeBootstrapError(f"effective {field} must be a timestep array")
    try:
        timesteps = list(value)
    except TypeError as error:
        raise RuntimeBootstrapError(
            f"effective {field} must be a timestep array"
        ) from error
    if not timesteps or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 < item <= 1000
        for item in timesteps
    ):
        raise RuntimeBootstrapError(
            f"effective {field} must contain timesteps in [1, 1000]"
        )
    return timesteps


def _validate_effective_config(config: Any) -> float:
    denoising_steps = _timestep_list(
        _config_value(config, "denoising_step_list"),
        "denoising_step_list",
    )
    if denoising_steps != _CF1_DENOISING_STEPS:
        raise RuntimeBootstrapError(
            "effective denoising_step_list does not match CF++1"
        )
    first_chunk_steps = _timestep_list(
        _config_value(config, "denoising_step_list_first_chunk"),
        "denoising_step_list_first_chunk",
    )
    if first_chunk_steps != _CF1_FIRST_CHUNK_DENOISING_STEPS:
        raise RuntimeBootstrapError(
            "effective denoising_step_list_first_chunk does not match CF++1"
        )
    if _config_value(config, "warp_denoising_step") is not True:
        raise RuntimeBootstrapError("effective warp_denoising_step must be true")
    if _config_value(config, "independent_first_frame") is not False:
        raise RuntimeBootstrapError(
            "effective independent_first_frame must be false"
        )
    num_frame_per_block = _config_value(config, "num_frame_per_block")
    if (
        isinstance(num_frame_per_block, bool)
        or not isinstance(num_frame_per_block, int)
        or num_frame_per_block != 1
    ):
        raise RuntimeBootstrapError("effective num_frame_per_block must equal one")
    context_noise = _config_value(config, "context_noise")
    if (
        isinstance(context_noise, bool)
        or not isinstance(context_noise, (int, float))
        or not math.isfinite(context_noise)
        or context_noise != 0
    ):
        raise RuntimeBootstrapError("effective context_noise must equal zero")
    model_kwargs = _config_value(config, "model_kwargs")
    if not isinstance(model_kwargs, Mapping) or "timestep_shift" not in model_kwargs:
        raise RuntimeBootstrapError("effective model_kwargs.timestep_shift is required")
    timestep_shift = model_kwargs["timestep_shift"]
    if (
        isinstance(timestep_shift, bool)
        or not isinstance(timestep_shift, (int, float))
        or not math.isfinite(timestep_shift)
        or timestep_shift != _CF1_TIMESTEP_SHIFT
    ):
        raise RuntimeBootstrapError("effective timestep_shift must equal CF++1")
    return float(timestep_shift)


def _effective_config_sha256(bindings: RuntimeBindings, config: Any) -> str:
    """Bind the complete resolved OmegaConf merge, not only selected fields."""

    try:
        resolved = bindings.OmegaConf.to_container(
            config,
            resolve=True,
            enum_to_str=True,
        )
    except Exception as error:
        raise RuntimeBootstrapError(
            "effective config could not be resolved"
        ) from error
    if not isinstance(resolved, Mapping):
        raise RuntimeBootstrapError("resolved effective config must be an object")
    try:
        encoded = json.dumps(
            resolved,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuntimeBootstrapError(
            "resolved effective config is not canonical JSON"
        ) from error
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != _EXPECTED_EFFECTIVE_CONFIG_SHA256:
        raise RuntimeBootstrapError("effective config digest does not match CF++1")
    return digest


def _guard_bundle_sha256(paths: Mapping[str, Path]) -> str:
    """Hash every module that defines bootstrap/session guard semantics."""

    if not paths or any(
        not isinstance(name, str)
        or not name
        or not isinstance(path, Path)
        for name, path in paths.items()
    ):
        raise RuntimeBootstrapError("CUDA guard bundle paths are invalid")
    records: list[dict[str, Any]] = []
    for name in sorted(paths):
        try:
            encoded = paths[name].resolve().read_bytes()
        except OSError as error:
            raise RuntimeBootstrapError(
                f"CUDA guard module bytes could not be read: {name}"
            ) from error
        records.append(
            {
                "name": name,
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cf1_guard_bundle_sha256() -> str:
    bench_root = Path(__file__).resolve().parent
    return _guard_bundle_sha256(
        {
            "bench/__init__.py": bench_root / "__init__.py",
            "bench/cf_cuda_adapter.py": bench_root / "cf_cuda_adapter.py",
            "bench/cf_cuda_generator.py": bench_root / "cf_cuda_generator.py",
            "bench/cf_cuda_session.py": bench_root / "cf_cuda_session.py",
            "bench/cf_cuda_smoke.py": bench_root / "cf_cuda_smoke.py",
            "bench/cf_attention_probe.py": bench_root / "cf_attention_probe.py",
            "bench/cf_runtime_evidence.py": bench_root / "cf_runtime_evidence.py",
            "bench/cf_runtime_preflight.py": bench_root / "cf_runtime_preflight.py",
            "bench/generation_preflight.py": bench_root / "generation_preflight.py",
            "bench/model_asset_preflight.py": bench_root / "model_asset_preflight.py",
            "bench/png_validation.py": bench_root / "png_validation.py",
            "bench/streaming_service.py": bench_root / "streaming_service.py",
        }
    )


def _lock_snapshot_unchanged(
    lock_path: Path, snapshot: AssetLockSnapshot
) -> bool:
    try:
        return lock_path.read_bytes() == snapshot.encoded
    except OSError:
        return False


def _runtime_lock_snapshot_unchanged(
    lock_path: Path, snapshot: RuntimeLockSnapshot
) -> bool:
    try:
        return lock_path.read_bytes() == snapshot.encoded
    except OSError:
        return False


def _runtime_evidence_snapshot_unchanged(
    evidence_path: Path, snapshot: RuntimeEvidenceSnapshot
) -> bool:
    try:
        return evidence_path.read_bytes() == snapshot.encoded
    except OSError:
        return False


def _validated_attention_probe(
    report: Mapping[str, Any],
    runtime_identity: RuntimePreflightIdentity,
    *,
    expected_runtime_environment_sha256: str,
    expected_native_identity_sha256: str,
) -> tuple[str, str, str]:
    probe_identity = report.get("probe_identity_sha256")
    native_environment = report.get("runtime_environment_sha256")
    native_identity = report.get("native_identity_sha256")
    if (
        report.get("probe_succeeded") is not True
        or report.get("authorizes_boot") is not False
        or report.get("ready") is not False
        or report.get("probe_mode") != "bound-verification"
        or report.get("gpu_execution_performed") is not True
        or report.get("runtime_lock_sha256")
        != runtime_identity.runtime_lock_sha256
        or report.get("runtime_evidence_sha256")
        != runtime_identity.runtime_evidence_sha256
        or report.get("static_environment_sha256")
        != runtime_identity.static_environment_sha256
        or report.get("attention_backend")
        not in {"flash-attention-2", "flash-attention-3"}
        or not _is_sha256(probe_identity)
        or not _is_sha256(native_environment)
        or not _is_sha256(native_identity)
        or not _is_sha256(expected_runtime_environment_sha256)
        or not _is_sha256(expected_native_identity_sha256)
        or native_environment != expected_runtime_environment_sha256
        or native_identity != expected_native_identity_sha256
    ):
        raise RuntimeBootstrapError("executed attention probe did not verify")
    return probe_identity, native_environment, native_identity


def _bootstrap_provenance(
    snapshot: AssetLockSnapshot,
    runtime_identity: RuntimePreflightIdentity,
    runtime: CF1Runtime,
    guard_bundle_sha256: str,
    report: Mapping[str, Any],
) -> CF1BootstrapProvenance:
    lock = snapshot.parsed()
    source = report.get("source")
    reported_assets = report.get("assets")
    if (
        report.get("ready") is not True
        or report.get("stack_id") != lock["stack_id"]
        or report.get("lock_sha256") != snapshot.sha256
        or not isinstance(source, Mapping)
        or source.get("status") != "verified"
        or source.get("observed_commit") != lock["source"]["commit"]
        or not isinstance(reported_assets, list)
    ):
        raise RuntimeBootstrapError(
            "observed source report does not match bootstrap lock"
        )
    reported_by_id: dict[str, Mapping[str, Any]] = {}
    for item in reported_assets:
        if not isinstance(item, Mapping):
            raise RuntimeBootstrapError("observed asset report is invalid")
        asset_id = item.get("id")
        if not isinstance(asset_id, str) or asset_id in reported_by_id:
            raise RuntimeBootstrapError("observed asset report is ambiguous")
        reported_by_id[asset_id] = item
    if set(reported_by_id) != {asset["id"] for asset in lock["assets"]}:
        raise RuntimeBootstrapError("observed asset set does not match lock")
    identities: list[CF1AssetIdentity] = []
    for asset in lock["assets"]:
        observed = reported_by_id[asset["id"]]
        if (
            observed.get("status") != "verified"
            or observed.get("relative_path") != asset["relative_path"]
            or observed.get("expected_size_bytes") != asset["size_bytes"]
            or observed.get("observed_size_bytes") != asset["size_bytes"]
            or observed.get("expected_sha256") != asset["sha256"]
            or observed.get("observed_sha256") != asset["sha256"]
        ):
            raise RuntimeBootstrapError(
                f"observed asset does not match lock: {asset['id']}"
            )
        identities.append(
            CF1AssetIdentity(
                id=asset["id"],
                relative_path=observed["relative_path"],
                size_bytes=observed["observed_size_bytes"],
                sha256=observed["observed_sha256"],
            )
        )
    assets = tuple(identities)
    if (
        runtime_identity.runtime_id != CF1_RUNTIME_ID
        or runtime_identity.runtime_lock_sha256 != CF1_RUNTIME_LOCK_SHA256
        or not _is_sha256(runtime_identity.runtime_evidence_sha256)
        or not _is_sha256(runtime_identity.static_environment_sha256)
        or runtime.torch is None
        or runtime.attention_backend
        not in {"flash-attention-3", "flash-attention-2"}
        or not isinstance(runtime.tokenizer_sentinel_sha256, str)
        or runtime.tokenizer_sentinel_sha256 != CF1_TOKENIZER_SENTINEL_SHA256
        or not _is_sha256(runtime.runtime_native_environment_sha256)
        or not _is_sha256(runtime.native_identity_sha256)
        or not _is_sha256(runtime.attention_probe_identity_sha256)
    ):
        raise RuntimeBootstrapError("observed runtime identity does not match CF++1")
    provenance = CF1BootstrapProvenance(
        stack_id=lock["stack_id"],
        source_commit=lock["source"]["commit"],
        asset_lock_sha256=snapshot.sha256,
        runtime_lock_sha256=runtime_identity.runtime_lock_sha256,
        runtime_evidence_sha256=runtime_identity.runtime_evidence_sha256,
        static_environment_sha256=runtime_identity.static_environment_sha256,
        runtime_environment_sha256=runtime_identity.environment_sha256,
        runtime_native_environment_sha256=(
            runtime.runtime_native_environment_sha256
        ),
        native_identity_sha256=runtime.native_identity_sha256,
        attention_probe_identity_sha256=(
            runtime.attention_probe_identity_sha256
        ),
        effective_config_sha256=runtime.effective_config_sha256,
        tokenizer_sentinel_sha256=runtime.tokenizer_sentinel_sha256,
        attention_backend=runtime.attention_backend,
        guard_bundle_sha256=guard_bundle_sha256,
        bootstrap_identity_sha256="",
        assets=assets,
    )
    return replace(
        provenance,
        bootstrap_identity_sha256=_provenance_identity_sha256(provenance),
    )


def _provenance_identity_sha256(provenance: CF1BootstrapProvenance) -> str:
    material = {
        "schema_version": 2,
        "stack_id": provenance.stack_id,
        "source_commit": provenance.source_commit,
        "asset_lock_sha256": provenance.asset_lock_sha256,
        "runtime_lock_sha256": provenance.runtime_lock_sha256,
        "runtime_evidence_sha256": provenance.runtime_evidence_sha256,
        "static_environment_sha256": provenance.static_environment_sha256,
        "runtime_environment_sha256": provenance.runtime_environment_sha256,
        "runtime_native_environment_sha256": (
            provenance.runtime_native_environment_sha256
        ),
        "native_identity_sha256": provenance.native_identity_sha256,
        "attention_probe_identity_sha256": (
            provenance.attention_probe_identity_sha256
        ),
        "effective_config_sha256": provenance.effective_config_sha256,
        "tokenizer_sentinel_sha256": provenance.tokenizer_sentinel_sha256,
        "attention_backend": provenance.attention_backend,
        "guard_bundle_sha256": provenance.guard_bundle_sha256,
        "assets": [
            {
                "id": asset.id,
                "relative_path": asset.relative_path,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
            }
            for asset in provenance.assets
        ],
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_cf1_runtime_provenance(runtime: Any) -> CF1BootstrapProvenance:
    """Recompute the complete pinned provenance before a CUDA session starts."""

    if not isinstance(runtime, CF1Runtime) or not isinstance(
        runtime.provenance, CF1BootstrapProvenance
    ):
        raise RuntimeBootstrapError("verified CF++1 runtime provenance is required")
    provenance = runtime.provenance
    try:
        snapshot = load_asset_lock_snapshot(DEFAULT_LOCK_PATH)
        runtime_snapshot = load_runtime_lock_snapshot(DEFAULT_RUNTIME_LOCK_PATH)
        evidence_snapshot = load_runtime_evidence_snapshot(
            DEFAULT_RUNTIME_EVIDENCE_PATH
        )
        locked_native_identities = runtime_evidence_locked_identities(
            evidence_snapshot
        )
    except Exception as error:
        raise RuntimeBootstrapError(
            "pinned CF++1 lock could not be reloaded"
        ) from error
    if snapshot.sha256 != CF1_ASSET_LOCK_SHA256:
        raise RuntimeBootstrapError("pinned CF++1 asset lock digest changed")
    lock = snapshot.parsed()
    runtime_lock = runtime_snapshot.parsed()
    expected_assets = tuple(
        CF1AssetIdentity(
            id=asset["id"],
            relative_path=asset["relative_path"],
            size_bytes=asset["size_bytes"],
            sha256=asset["sha256"],
        )
        for asset in lock["assets"]
    )
    if (
        lock["stack_id"] != CF1_STACK_ID
        or lock["source"]["commit"] != CF1_SOURCE_COMMIT
        or provenance.stack_id != CF1_STACK_ID
        or provenance.source_commit != CF1_SOURCE_COMMIT
        or provenance.asset_lock_sha256 != snapshot.sha256
        or runtime_snapshot.sha256 != CF1_RUNTIME_LOCK_SHA256
        or not isinstance(runtime.runtime_identity, RuntimePreflightIdentity)
        or runtime.runtime_identity.runtime_id != CF1_RUNTIME_ID
        or runtime.runtime_identity.runtime_lock_sha256 != runtime_snapshot.sha256
        or runtime_lock["evidence_lock_sha256"] != evidence_snapshot.sha256
        or runtime.runtime_identity.runtime_evidence_sha256
        != evidence_snapshot.sha256
        or provenance.runtime_evidence_sha256 != evidence_snapshot.sha256
        or provenance.static_environment_sha256
        != runtime.runtime_identity.static_environment_sha256
        or not _is_sha256(provenance.static_environment_sha256)
        or runtime.torch is None
        or runtime.attention_backend
        not in {"flash-attention-3", "flash-attention-2"}
        or (
            runtime_lock["target"]["attention_backend"] is not None
            and runtime.attention_backend
            != runtime_lock["target"]["attention_backend"]
        )
        or runtime.runtime_identity.effective_host_headroom_bytes
        < runtime_lock["capacity"]["minimum_host_headroom_bytes"]
        or runtime.runtime_identity.gpu_total_bytes
        < runtime_lock["capacity"]["minimum_gpu_total_bytes"]
        or runtime.runtime_identity.gpu_free_bytes
        < runtime_lock["capacity"]["minimum_gpu_free_bytes"]
        or provenance.runtime_lock_sha256 != runtime_snapshot.sha256
        or provenance.runtime_environment_sha256
        != runtime.runtime_identity.environment_sha256
        or not _is_sha256(provenance.runtime_environment_sha256)
        or provenance.runtime_native_environment_sha256
        != runtime.runtime_native_environment_sha256
        or runtime.runtime_native_environment_sha256
        != locked_native_identities.runtime_environment_sha256
        or not _is_sha256(provenance.runtime_native_environment_sha256)
        or provenance.native_identity_sha256 != runtime.native_identity_sha256
        or runtime.native_identity_sha256
        != locked_native_identities.native_identity_sha256
        or not _is_sha256(provenance.native_identity_sha256)
        or provenance.attention_probe_identity_sha256
        != runtime.attention_probe_identity_sha256
        or not _is_sha256(provenance.attention_probe_identity_sha256)
        or provenance.effective_config_sha256 != CF1_EFFECTIVE_CONFIG_SHA256
        or runtime.effective_config_sha256 != CF1_EFFECTIVE_CONFIG_SHA256
        or provenance.effective_config_sha256 != runtime.effective_config_sha256
        or runtime.tokenizer_sentinel_sha256 != CF1_TOKENIZER_SENTINEL_SHA256
        or provenance.tokenizer_sentinel_sha256
        != runtime.tokenizer_sentinel_sha256
        or provenance.attention_backend != runtime.attention_backend
        or provenance.assets != expected_assets
        or provenance.guard_bundle_sha256 != _cf1_guard_bundle_sha256()
        or provenance.bootstrap_identity_sha256
        != _provenance_identity_sha256(provenance)
    ):
        raise RuntimeBootstrapError("CF++1 runtime provenance identity is invalid")
    return provenance


def _checkpoint_generator(
    bindings: RuntimeBindings,
    generator: Any,
    checkpoint_path: Path,
) -> None:
    checkpoint = bindings.torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeBootstrapError("CF++ checkpoint must be an object")
    state_dict = checkpoint.get("generator_ema")
    if not isinstance(state_dict, Mapping):
        raise RuntimeBootstrapError("CF++ checkpoint generator_ema is required")
    normalized = normalize_fsdp_generator_state_dict(
        state_dict,
        expected_keys=generator.state_dict().keys(),
    )
    generator.load_state_dict(normalized, strict=True, assign=True)


def _build_pinned_taehv(
    bindings: RuntimeBindings,
    checkpoint_path: Path,
) -> Any:
    """Construct TAEHV without its path loader, then safe-load exact keys."""

    taehv = bindings.TAEHV(checkpoint_path=None)
    state_dict = bindings.torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state_dict, Mapping):
        raise RuntimeBootstrapError("TAEHV checkpoint must be an object")
    validate_strict_checkpoint_keys(
        expected_keys=taehv.state_dict().keys(),
        checkpoint_keys=state_dict.keys(),
    )
    patched = taehv.patch_tgrow_layers(state_dict)
    if not isinstance(patched, Mapping):
        raise RuntimeBootstrapError("TAEHV patched checkpoint must be an object")
    validate_strict_checkpoint_keys(
        expected_keys=taehv.state_dict().keys(),
        checkpoint_keys=patched.keys(),
    )
    taehv.load_state_dict(patched, strict=True, assign=True)
    return taehv


def _build_verified_runtime(
    paths: CF1RuntimePaths,
    bindings: RuntimeBindings,
) -> CF1Runtime:
    """Build from paths whose bytes were already verified by the asset gate."""

    bindings.torch.set_grad_enabled(False)
    with _working_directory(paths.checkout):
        effective_config = bindings.OmegaConf.merge(
            bindings.OmegaConf.load(paths.default_config),
            bindings.OmegaConf.load(paths.candidate_config),
        )
        timestep_shift = _validate_effective_config(effective_config)
        effective_config_sha256 = _effective_config_sha256(
            bindings, effective_config
        )
        model_kwargs = _load_model_config(paths.model_config)
        tokenizer, tokenizer_sentinel_sha256 = _build_verified_tokenizer(
            bindings,
            paths,
        )
        model_kwargs.update(local_attn_size=-1, sink_size=0)
        model = bindings.CausalWanModel(**model_kwargs)
        generator = _build_checkpoint_only_generator(
            bindings,
            model,
            timestep_shift=float(timestep_shift),
        )
        _checkpoint_generator(bindings, generator, paths.generator_checkpoint)
        text_encoder = _build_pinned_text_encoder(
            bindings,
            paths,
            tokenizer,
        )
        pipeline = bindings.CausalInferencePipeline(
            effective_config,
            device=bindings.gpu,
            generator=generator,
            text_encoder=text_encoder,
            vae=_latent_passthrough(bindings),
        )
        pipeline = pipeline.to(dtype=bindings.torch.bfloat16)
        pipeline.text_encoder.to(bindings.gpu)
        pipeline.generator.to(bindings.gpu)
        pipeline.eval()
        taehv = _build_pinned_taehv(
            bindings,
            paths.taehv_checkpoint,
        ).to(device=bindings.gpu, dtype=bindings.torch.float16).eval()
    return CF1Runtime(
        pipeline=pipeline,
        taehv=taehv,
        effective_config=effective_config,
        effective_config_sha256=effective_config_sha256,
        device=bindings.gpu,
        torch=bindings.torch,
        attention_backend=bindings.attention_backend,
        tokenizer_sentinel_sha256=tokenizer_sentinel_sha256,
    )


def _preflight_failures(report: Mapping[str, Any]) -> list[str]:
    source = report.get("source")
    source_status = source.get("status") if isinstance(source, Mapping) else "invalid"
    failures = [f"source:{source_status}"] if source_status != "verified" else []
    assets = report.get("assets")
    if not isinstance(assets, list):
        return [*failures, "assets:invalid"]
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            failures.append(f"assets[{index}]:invalid")
        elif asset.get("status") != "verified":
            failures.append(f"{asset.get('id')}:{asset.get('status')}")
    return failures


def build_cf1_runtime(
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    runtime_lock_path: Path = DEFAULT_RUNTIME_LOCK_PATH,
    runtime_evidence_path: Path = DEFAULT_RUNTIME_EVIDENCE_PATH,
    checkout: Path = DEFAULT_CHECKOUT,
    cuda_device_index: int = 0,
) -> CF1Runtime:
    """Verify every pinned byte, then construct the minimal CUDA runtime."""

    if (
        isinstance(cuda_device_index, bool)
        or not isinstance(cuda_device_index, int)
        or cuda_device_index != 0
    ):
        raise ValueError(
            "the exact single-GPU runtime requires CUDA index zero"
        )
    try:
        runtime_snapshot = load_runtime_lock_snapshot(runtime_lock_path)
        if runtime_snapshot.sha256 != CF1_RUNTIME_LOCK_SHA256:
            raise RuntimeBootstrapError("runtime lock digest does not match CF++1")
        runtime_identity = preflight_current_runtime(
            runtime_snapshot,
            gpu_index=cuda_device_index,
            evidence_path=runtime_evidence_path,
        )
        runtime_evidence_snapshot = load_runtime_evidence_snapshot(
            runtime_evidence_path
        )
        if (
            runtime_evidence_snapshot.sha256
            != runtime_identity.runtime_evidence_sha256
        ):
            raise RuntimeBootstrapError(
                "runtime evidence digest does not match preflight"
            )
        locked_native_identities = runtime_evidence_locked_identities(
            runtime_evidence_snapshot
        )
    except (RuntimePreflightError, RuntimeEvidenceError) as error:
        raise RuntimeBootstrapError("runtime environment preflight failed") from error
    snapshot = load_asset_lock_snapshot(lock_path)
    if snapshot.sha256 != _EXPECTED_ASSET_LOCK_SHA256:
        raise RuntimeBootstrapError("asset lock digest does not match CF++1")
    lock = snapshot.parsed()
    report = verify_model_assets_snapshot(snapshot, checkout)
    if not report["ready"]:
        raise RuntimeBootstrapError(
            "model asset preflight is not ready: "
            + ", ".join(_preflight_failures(report))
        )
    paths = _asset_paths(lock, checkout)
    _validate_tokenizer_inventory(paths)
    guard_bundle_sha256 = _cf1_guard_bundle_sha256()
    probe_report = attention_probe_report(
        runtime_lock_path,
        asset_lock_path=lock_path,
        evidence_path=runtime_evidence_path,
        checkout=checkout,
        gpu_index=cuda_device_index,
    )
    (
        attention_probe_identity_sha256,
        runtime_native_environment_sha256,
        native_identity_sha256,
    ) = _validated_attention_probe(
        probe_report,
        runtime_identity,
        expected_runtime_environment_sha256=(
            locked_native_identities.runtime_environment_sha256
        ),
        expected_native_identity_sha256=(
            locked_native_identities.native_identity_sha256
        ),
    )
    if (
        probe_report["attention_backend"]
        != runtime_snapshot.parsed()["target"]["attention_backend"]
    ):
        raise RuntimeBootstrapError(
            "executed attention backend does not match runtime lock"
        )
    bindings = _load_runtime_bindings(paths.checkout, cuda_device_index)
    try:
        loaded_host_headroom = validate_current_host_capacity(runtime_snapshot)
        loaded_gpu_free, loaded_gpu_total = validate_loaded_cuda_capacity(
            bindings.torch,
            runtime_snapshot,
            device=bindings.gpu,
            attention_backend=bindings.attention_backend,
        )
    except RuntimePreflightError as error:
        raise RuntimeBootstrapError(
            "loaded host/CUDA capacity preflight failed"
        ) from error
    runtime_identity = replace(
        runtime_identity,
        effective_host_headroom_bytes=loaded_host_headroom,
        gpu_total_bytes=loaded_gpu_total,
        gpu_free_bytes=loaded_gpu_free,
    )
    runtime = _build_verified_runtime(paths, bindings)
    if (
        runtime.torch is not bindings.torch
        or runtime.attention_backend != bindings.attention_backend
    ):
        raise RuntimeBootstrapError(
            "constructed runtime bindings do not match the verified bindings"
        )
    runtime = replace(
        runtime,
        runtime_identity=runtime_identity,
        runtime_native_environment_sha256=(
            runtime_native_environment_sha256
        ),
        native_identity_sha256=native_identity_sha256,
        attention_probe_identity_sha256=(
            attention_probe_identity_sha256
        ),
    )
    if not _lock_snapshot_unchanged(lock_path, snapshot):
        raise RuntimeBootstrapError("asset lock changed during runtime bootstrap")
    if not _runtime_lock_snapshot_unchanged(runtime_lock_path, runtime_snapshot):
        raise RuntimeBootstrapError("runtime lock changed during runtime bootstrap")
    if not _runtime_evidence_snapshot_unchanged(
        runtime_evidence_path, runtime_evidence_snapshot
    ):
        raise RuntimeBootstrapError(
            "runtime evidence changed during runtime bootstrap"
        )
    if _cf1_guard_bundle_sha256() != guard_bundle_sha256:
        raise RuntimeBootstrapError("CUDA guard bundle changed during runtime bootstrap")
    post_report = verify_model_assets_snapshot(snapshot, checkout)
    if not post_report["ready"]:
        raise RuntimeBootstrapError(
            "model assets changed during runtime bootstrap: "
            + ", ".join(_preflight_failures(post_report))
        )
    return replace(
        runtime,
        provenance=_bootstrap_provenance(
            snapshot,
            runtime_identity,
            runtime,
            guard_bundle_sha256,
            post_report,
        ),
    )
