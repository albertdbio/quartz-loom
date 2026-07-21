"""Candidate-safe executed attention probe for the exact CF++1 H100 stack.

The diagnostic validates immutable runtime/source identities before importing
Torch.  It executes the verified ``wan/modules/attention.py`` bytes directly,
so Python never runs the ``wan`` package initializers that import model,
tokenizer, and VAE code.  Success is evidence only and never authorizes boot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping

from bench.cf_runtime_evidence import (
    DEFAULT_RUNTIME_EVIDENCE_PATH,
    RuntimeEvidenceError,
    verify_runtime_evidence,
)
from bench.cf_runtime_preflight import (
    CF1_RUNTIME_ID,
    CF1_RUNTIME_LOCK_SHA256,
    DEFAULT_RUNTIME_LOCK_PATH,
    RuntimeEvidenceContext,
    RuntimePreflightError,
    load_runtime_lock_snapshot,
    validate_runtime_lock,
    verify_bound_static_runtime_evidence,
)
from bench.model_asset_preflight import (
    AssetLockError,
    DEFAULT_CHECKOUT,
    DEFAULT_LOCK_PATH,
    load_asset_lock_snapshot,
    verify_model_source_snapshot,
)


CF1_ASSET_LOCK_SHA256 = (
    "0aee8671f8e3b30286b689a16f6f4a355f917772c16e599cec75a49e89057967"
)
CF1_SOURCE_COMMIT = "8db419e341e5fc52542c0b2c4542728420ddfb4a"
ATTENTION_RELATIVE_PATH = Path("wan/modules/attention.py")
LOADED_LIBRARY_IDENTITY_POLICY = "explicit-runtime-roots-v1"
ATTENTION_INPUT_SHAPE = (1, 16, 12, 128)
_MAX_ATTENTION_SOURCE_BYTES = 64 * 1024
_READ_BYTES = 1024 * 1024


class AttentionProbeError(ValueError):
    """The diagnostic cannot prove the exact production attention route."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_is_verified(report: Mapping[str, Any]) -> bool:
    source = report.get("source")
    return (
        report.get("ready") is True
        and isinstance(source, Mapping)
        and source.get("status") == "verified"
        and source.get("observed_commit") == CF1_SOURCE_COMMIT
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_verified_attention_source(checkout: Path) -> tuple[bytes, str]:
    """Read the bounded tracked source without following path symlinks."""

    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise AttentionProbeError("no-follow source reads are unsupported")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | close_on_exec
    directory_fd = os.open(checkout, directory_flags)
    file_fd: int | None = None
    try:
        for component in ATTENTION_RELATIVE_PATH.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            ATTENTION_RELATIVE_PATH.parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_ATTENTION_SOURCE_BYTES
        ):
            raise AttentionProbeError("attention source file identity is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(file_fd, min(64 * 1024, remaining))
            if not block:
                raise AttentionProbeError("attention source changed while read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(file_fd, 1):
            raise AttentionProbeError("attention source changed while read")
        after = os.fstat(file_fd)
        if _stat_fingerprint(after) != _stat_fingerprint(before):
            raise AttentionProbeError("attention source changed while read")
    except OSError as error:
        raise AttentionProbeError("attention source could not be read safely") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)
    encoded = b"".join(chunks)
    return encoded, hashlib.sha256(encoded).hexdigest()


def _execute_attention_source(
    encoded: bytes,
    source_path: Path,
    *,
    module_name: str = "cf1_verified_attention_probe",
) -> Any:
    """Execute one verified source snapshot without package import hooks."""

    if not isinstance(encoded, bytes) or not encoded:
        raise AttentionProbeError("attention source snapshot is empty")
    if module_name in sys.modules:
        raise AttentionProbeError("attention probe module name is already cached")
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    try:
        code = compile(encoded, str(source_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except (AttentionProbeError, RuntimePreflightError):
        raise
    except Exception as error:
        raise AttentionProbeError("verified attention source could not load") from error
    return module


def _relevant_module_loaded() -> str | None:
    prefixes = (
        "torch",
        "flash_attn",
        "flash_attn_2_cuda",
        "flash_attn_interface",
    )
    for name in sorted(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            return name
    return None


def _require_isolated_import_context() -> None:
    flags = sys.flags
    if getattr(flags, "isolated", 0) != 1 or getattr(flags, "no_site", 0) != 1:
        raise AttentionProbeError(
            "attention probe requires an isolated no-site interpreter"
        )
    loaded = _relevant_module_loaded()
    if loaded is not None:
        raise AttentionProbeError(
            "Torch or FlashAttention was loaded before runtime verification"
        )


def _probe_distribution_paths(
    explicit: tuple[Path, ...],
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path in explicit:
        if not path.is_absolute():
            raise AttentionProbeError(
                "trusted runtime distribution path is invalid"
            )
        try:
            value = path.resolve(strict=True)
        except OSError as error:
            raise AttentionProbeError(
                "trusted runtime distribution path is unavailable"
            ) from error
        if not value.is_dir() or value in resolved:
            raise AttentionProbeError(
                "trusted runtime distribution path is invalid"
            )
        resolved.append(value)
    if not resolved:
        raise AttentionProbeError("trusted runtime distribution paths are empty")
    return tuple(resolved)


def _probe_identity_roots(
    evidence_context: RuntimeEvidenceContext | None,
    *,
    unbound_environment_root: Path | None,
    unbound_stdlib_root: Path | None,
    unbound_distribution_paths: tuple[Path, ...] | None,
) -> tuple[Path, Path, tuple[Path, ...]]:
    if evidence_context is not None:
        return (
            evidence_context.environment_root,
            evidence_context.stdlib_root,
            evidence_context.distribution_paths,
        )
    if unbound_environment_root is None:
        raise AttentionProbeError(
            "trusted runtime environment root assertion is missing"
        )
    if unbound_stdlib_root is None:
        raise AttentionProbeError("trusted Python stdlib root assertion is missing")
    if unbound_distribution_paths is None:
        raise AttentionProbeError(
            "trusted runtime distribution path assertion is missing"
        )
    if (
        not unbound_environment_root.is_absolute()
        or not unbound_stdlib_root.is_absolute()
    ):
        raise AttentionProbeError("trusted runtime identity root is invalid")
    try:
        environment_root = unbound_environment_root.resolve(strict=True)
        stdlib_root = unbound_stdlib_root.resolve(strict=True)
    except OSError as error:
        raise AttentionProbeError("trusted runtime identity root is unavailable") from error
    if not environment_root.is_dir() or not stdlib_root.is_dir():
        raise AttentionProbeError("trusted runtime identity root is invalid")
    distribution_paths = _probe_distribution_paths(unbound_distribution_paths)
    if any(not _is_within(path, environment_root) for path in distribution_paths):
        raise AttentionProbeError(
            "trusted runtime distribution path escapes the environment"
        )
    return environment_root, stdlib_root, distribution_paths


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_torch(
    distribution_paths: tuple[Path, ...],
    *,
    stdlib_root: Path,
) -> Any:
    """Import Torch with reviewed site-packages ahead of all project paths."""

    _require_isolated_import_context()
    distributions = _probe_distribution_paths(distribution_paths)
    try:
        stdlib = stdlib_root.resolve(strict=True)
    except OSError as error:
        raise AttentionProbeError("Python stdlib path is unavailable") from error
    if not stdlib.is_dir():
        raise AttentionProbeError("Python stdlib path is unavailable")
    stdlib_zip_names = {
        f"python{sys.version_info.major}{sys.version_info.minor}.zip",
        f"python{sys.version_info.major}.{sys.version_info.minor}.zip",
    }
    original = list(sys.path)
    standard_paths: list[str] = []
    for item in original:
        if not item:
            continue
        path = Path(item)
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            continue
        if _is_within(resolved, stdlib) or (
            resolved.parent == stdlib.parent
            and resolved.name in stdlib_zip_names
        ):
            standard_paths.append(str(path))
    trusted = [str(path) for path in distributions]
    trusted.extend(path for path in standard_paths if path not in trusted)
    if not trusted:
        raise AttentionProbeError("trusted runtime import path is empty")
    sys.path[:] = trusted
    sys.dont_write_bytecode = True
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        raise AttentionProbeError("verified Torch package could not import") from error
    finally:
        sys.path[:] = trusted + [path for path in original if path not in trusted]
    origin = getattr(torch, "__file__", None)
    if not isinstance(origin, str):
        raise AttentionProbeError("loaded Torch origin is unavailable")
    try:
        resolved_origin = Path(origin).resolve(strict=True)
    except OSError as error:
        raise AttentionProbeError("loaded Torch origin is unavailable") from error
    if not any(_is_within(resolved_origin, root) for root in distributions):
        raise AttentionProbeError("loaded Torch origin is outside reviewed packages")
    return torch


def _validate_attention_origins(
    attention_module: Any,
    distribution_paths: tuple[Path, ...],
) -> None:
    roots = _probe_distribution_paths(distribution_paths)
    candidates = [getattr(attention_module, "flash_attn", None)]
    if getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False):
        candidates.append(getattr(attention_module, "flash_attn_interface", None))
    for module in candidates:
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise AttentionProbeError("loaded FlashAttention origin is unavailable")
        try:
            resolved = Path(origin).resolve(strict=True)
        except OSError as error:
            raise AttentionProbeError(
                "loaded FlashAttention origin is unavailable"
            ) from error
        if not any(_is_within(resolved, root) for root in roots):
            raise AttentionProbeError(
                "loaded FlashAttention origin is outside reviewed packages"
            )


def _validate_loaded_runtime(
    torch: Any,
    lock: Mapping[str, Any],
    *,
    gpu_index: int,
) -> str:
    if gpu_index != 0:
        raise AttentionProbeError("the probe requires CUDA index zero")
    expected_torch = next(
        (
            package["version"]
            for package in lock["packages"]
            if package["distribution"].lower().replace("_", "-") == "torch"
        ),
        None,
    )
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
    except Exception as error:
        raise AttentionProbeError("CUDA availability could not be observed") from error
    if available is not True or count != 1:
        raise AttentionProbeError("the probe requires exactly one CUDA GPU")
    try:
        torch.cuda.set_device(gpu_index)
        current = torch.cuda.current_device()
        name = torch.cuda.get_device_name(gpu_index)
        capability = tuple(torch.cuda.get_device_capability(gpu_index))
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception as error:
        raise AttentionProbeError("selected CUDA device could not be observed") from error
    if (
        current != gpu_index
        or name != lock["target"]["gpu_name"]
        or capability != tuple(lock["target"]["compute_capability"])
        or bf16_supported is not True
        or getattr(torch, "__version__", None) != expected_torch
        or getattr(getattr(torch, "version", None), "cuda", None)
        != lock["target"]["cuda_runtime"]
    ):
        raise AttentionProbeError("loaded Torch/CUDA identity does not match the pin")
    return f"cuda:{gpu_index}"


def _backend_callables(attention_module: Any) -> tuple[str, list[tuple[str, Any, str]]]:
    fa3 = getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", None)
    fa2 = getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", None)
    if not isinstance(fa3, bool) or not isinstance(fa2, bool):
        raise AttentionProbeError("attention backend flags are invalid")
    if not fa3 and not fa2:
        raise AttentionProbeError("FlashAttention 2 or 3 is required")
    callables: list[tuple[str, Any, str]] = []
    if fa3:
        owner = getattr(attention_module, "flash_attn_interface", None)
        function = getattr(owner, "flash_attn_varlen_func", None)
        if owner is None or not callable(function):
            raise AttentionProbeError("FlashAttention 3 callable is unavailable")
        callables.append(
            ("flash_attn_interface.flash_attn_varlen_func", owner, "flash-attention-3")
        )
    if fa2:
        owner = getattr(attention_module, "flash_attn", None)
        function = getattr(owner, "flash_attn_varlen_func", None)
        if owner is None or not callable(function):
            raise AttentionProbeError("FlashAttention 2 callable is unavailable")
        callables.append(
            ("flash_attn.flash_attn_varlen_func", owner, "flash-attention-2")
        )
    selected = "flash-attention-3" if fa3 else "flash-attention-2"
    return selected, callables


def _execute_attention_kernel(
    torch: Any,
    attention_module: Any,
    *,
    device: str,
) -> dict[str, Any]:
    """Delegate through the production router and record the exact real callable."""

    selected, callables = _backend_callables(attention_module)
    calls: list[str] = []
    originals: list[tuple[Any, Any]] = []
    for callable_name, owner, _backend in callables:
        original = owner.flash_attn_varlen_func

        def delegated(*args: Any, __name: str = callable_name, __fn: Any = original, **kwargs: Any) -> Any:
            calls.append(__name)
            return __fn(*args, **kwargs)

        originals.append((owner, original))
        owner.flash_attn_varlen_func = delegated
    try:
        q = torch.full(ATTENTION_INPUT_SHAPE, 0.125, device=device, dtype=torch.bfloat16)
        k = torch.full(ATTENTION_INPUT_SHAPE, -0.25, device=device, dtype=torch.bfloat16)
        v = torch.full(ATTENTION_INPUT_SHAPE, 0.5, device=device, dtype=torch.bfloat16)
        output = attention_module.attention(
            q,
            k,
            v,
            dtype=torch.bfloat16,
            fa_version=None,
        )
        torch.cuda.synchronize()
    finally:
        for owner, original in originals:
            owner.flash_attn_varlen_func = original

    expected_callable = (
        "flash_attn_interface.flash_attn_varlen_func"
        if selected == "flash-attention-3"
        else "flash_attn.flash_attn_varlen_func"
    )
    output_device = getattr(output, "device", None)
    try:
        finite = torch.isfinite(output).all().item()
    except Exception as error:
        raise AttentionProbeError("attention output finiteness could not be checked") from error
    if (
        calls != [expected_callable]
        or tuple(getattr(output, "shape", ())) != ATTENTION_INPUT_SHAPE
        or getattr(output, "dtype", None) != torch.bfloat16
        or getattr(output_device, "type", None) != "cuda"
        or getattr(output_device, "index", None) not in {0, None}
        or finite is not True
    ):
        raise AttentionProbeError("executed attention output contract is invalid")
    return {
        "gpu_execution_performed": True,
        "attention_backend": selected,
        "executed_callable": expected_callable,
        "call_count": 1,
        "input_shape": list(ATTENTION_INPUT_SHAPE),
        "output_shape": list(ATTENTION_INPUT_SHAPE),
        "output_dtype": str(torch.bfloat16),
        "output_finite": True,
    }


def _sha256_file(path: Path, label: str) -> tuple[str, int]:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise AttentionProbeError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(_READ_BYTES)
                if not block:
                    break
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise AttentionProbeError(f"{label} could not be hashed") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise AttentionProbeError(f"{label} changed while being hashed")
    return digest.hexdigest(), before.st_size


def _readelf_metadata(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/usr/bin/readelf", "--wide", "-h", "-d", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AttentionProbeError("critical ELF metadata could not be read") from error
    elf_class: str | None = None
    elf_machine: str | None = None
    needed: list[str] = []
    search_paths: dict[str, list[str]] = {"rpath": [], "runpath": []}

    def append_unique(values: list[str], value: str, label: str) -> None:
        if value in values:
            raise AttentionProbeError(f"critical ELF {label} contains duplicates")
        values.append(value)

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Class:"):
            elf_class = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Machine:"):
            elf_machine = stripped.split(":", 1)[1].strip()
        elif "(NEEDED)" in stripped and "[" in stripped and "]" in stripped:
            append_unique(
                needed,
                stripped.rsplit("[", 1)[1].split("]", 1)[0],
                "NEEDED list",
            )
        else:
            for tag in ("rpath", "runpath"):
                if f"({tag.upper()})" in stripped and "[" in stripped and "]" in stripped:
                    raw = stripped.rsplit("[", 1)[1].split("]", 1)[0]
                    for item in (value for value in raw.split(":") if value):
                        append_unique(search_paths[tag], item, tag.upper())
    if not elf_class or not elf_machine:
        raise AttentionProbeError("critical ELF identity is incomplete")
    return {
        "elf_class": elf_class,
        "elf_machine": elf_machine,
        "needed": needed,
        "rpath": search_paths["rpath"],
        "runpath": search_paths["runpath"],
    }


def _critical_module_row(module: str, path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AttentionProbeError("critical native module is unavailable") from error
    # Read the complete file now so an unreadable or changing native module
    # cannot be represented as loaded evidence.  The evidence capturer hashes
    # the verified path again into its bounded snapshot.
    _sha256_file(resolved, "critical native module")
    return {
        "module": module,
        "path": str(resolved),
        **_readelf_metadata(resolved),
    }


def _critical_native_modules(
    torch: Any,
    attention_module: Any,
    *,
    attention_backend: str,
) -> list[dict[str, Any]]:
    torch_c = getattr(torch, "_C", None)
    torch_c_path = getattr(torch_c, "__file__", None)
    torch_path = getattr(torch, "__file__", None)
    if not isinstance(torch_c_path, str) or not isinstance(torch_path, str):
        raise AttentionProbeError("loaded Torch native origins are unavailable")
    rows = [
        _critical_module_row("torch._C", Path(torch_c_path)),
        _critical_module_row(
            "torch.lib.libtorch_cuda",
            Path(torch_path).resolve().parent / "lib" / "libtorch_cuda.so",
        ),
    ]
    if attention_backend == "flash-attention-2":
        extension = sys.modules.get("flash_attn_2_cuda")
        extension_name = "flash_attn_2_cuda"
    else:
        extension = getattr(attention_module, "flash_attn_interface", None)
        extension_name = getattr(extension, "__name__", "flash_attn_interface")
    extension_path = getattr(extension, "__file__", None)
    if not isinstance(extension_path, str):
        raise AttentionProbeError("loaded FlashAttention extension origin is unavailable")
    rows.append(_critical_module_row(extension_name, Path(extension_path)))
    rows.sort(key=lambda row: row["module"])
    if len({row["module"] for row in rows}) != len(rows):
        raise AttentionProbeError("critical native module inventory is ambiguous")
    return rows


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _loaded_library_identity(
    *,
    environment_root: Path,
    stdlib_root: Path,
    maps_path: Path = Path("/proc/self/maps"),
) -> tuple[str, int, list[dict[str, Any]]]:
    try:
        lines = maps_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AttentionProbeError("loaded library mappings could not be read") from error
    paths: set[Path] = set()
    for line in lines:
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        raw_path = parts[5]
        if not raw_path.startswith("/"):
            continue
        # CUDA uses deleted /dev/zero mappings as anonymous shared memory.
        # They have no executable file identity and are not libraries.  Any
        # other deleted file-backed mapping remains a hard failure.
        if raw_path == "/dev/zero (deleted)":
            continue
        if raw_path.endswith(" (deleted)"):
            raise AttentionProbeError("loaded library mapping is deleted")
        try:
            path = Path(raw_path).resolve(strict=True)
        except OSError as error:
            raise AttentionProbeError("loaded library mapping is unavailable") from error
        if path.is_file():
            paths.add(path)
    if not paths:
        raise AttentionProbeError("loaded library mapping inventory is empty")
    try:
        environment_root = environment_root.resolve(strict=True)
        stdlib_root = stdlib_root.resolve(strict=True)
    except OSError as error:
        raise AttentionProbeError("loaded library identity root is unavailable") from error
    if not environment_root.is_dir() or not stdlib_root.is_dir():
        raise AttentionProbeError("loaded library identity root is invalid")
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=str):
        digest, size = _sha256_file(path, "loaded library")
        if _path_within(path, environment_root):
            scope = "environment"
            identity = path.relative_to(environment_root).as_posix()
        elif _path_within(path, stdlib_root):
            scope = "stdlib"
            identity = path.relative_to(stdlib_root).as_posix()
        else:
            scope = "system"
            identity = path.name
        rows.append(
            {
                "scope": scope,
                "identity": identity,
                "size": size,
                "sha256": digest,
            }
        )
    rows.sort(
        key=lambda row: (
            row["scope"],
            row["identity"],
            row["sha256"],
        )
    )
    identities = [(row["scope"], row["identity"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise AttentionProbeError("loaded library identities are ambiguous")
    return _canonical_sha256(rows), len(rows), rows


def _version_string(value: object, label: str) -> str:
    if isinstance(value, tuple) and value and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        return ".".join(str(item) for item in value)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        digits = str(value)
        if len(digits) < 4:
            return digits
        return f"{digits[0]}.{int(digits[1:-2])}.{int(digits[-2:])}"
    if isinstance(value, str) and value:
        return value
    raise AttentionProbeError(f"loaded {label} version is invalid")


def _capture_loaded_facts(
    torch: Any,
    attention_module: Any,
    *,
    attention_backend: str,
    environment_root: Path,
    stdlib_root: Path,
) -> dict[str, Any]:
    try:
        cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        cudnn = torch.backends.cudnn.version()
        nccl = torch.cuda.nccl.version()
        arches = sorted(set(torch.cuda.get_arch_list()))
    except Exception as error:
        raise AttentionProbeError("loaded CUDA ABI facts could not be observed") from error
    if not isinstance(cxx11_abi, bool) or "sm_90" not in arches:
        raise AttentionProbeError("loaded CUDA ABI does not include the H100 target")
    critical_modules = _critical_native_modules(
        torch,
        attention_module,
        attention_backend=attention_backend,
    )
    libraries_sha256, library_count, libraries = _loaded_library_identity(
        environment_root=environment_root,
        stdlib_root=stdlib_root,
    )
    return {
        "loaded_library_identity_policy": LOADED_LIBRARY_IDENTITY_POLICY,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cxx11_abi": cxx11_abi,
        "cudnn_version": _version_string(cudnn, "cuDNN"),
        "nccl_version": _version_string(nccl, "NCCL"),
        "cuda_arch_list": arches,
        "import_order": [
            "torch",
            "verified-attention-source",
            attention_backend,
        ],
        "critical_modules": critical_modules,
        "loaded_libraries_manifest_sha256": libraries_sha256,
        "loaded_library_count": library_count,
        "loaded_libraries": libraries,
    }


def attention_probe_report(
    lock_path: Path = DEFAULT_RUNTIME_LOCK_PATH,
    *,
    asset_lock_path: Path = DEFAULT_LOCK_PATH,
    evidence_path: Path = DEFAULT_RUNTIME_EVIDENCE_PATH,
    checkout: Path = DEFAULT_CHECKOUT,
    gpu_index: int = 0,
    allow_unbound_evidence_capture: bool = False,
    unbound_environment_root: Path | None = None,
    unbound_stdlib_root: Path | None = None,
    unbound_distribution_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Run a non-authorizing exact-source CUDA diagnostic."""

    base: dict[str, Any] = {
        "schema_version": 2,
        "kind": "cf1-executed-attention-probe",
        "probe_succeeded": False,
        "authorizes_boot": False,
        "ready": False,
        "gpu_execution_performed": False,
        "runtime_id": CF1_RUNTIME_ID,
        "runtime_lock_sha256": None,
        "runtime_evidence_sha256": None,
        "asset_lock_sha256": None,
    }
    try:
        runtime_snapshot = load_runtime_lock_snapshot(lock_path)
        base["runtime_lock_sha256"] = runtime_snapshot.sha256
        if runtime_snapshot.sha256 != CF1_RUNTIME_LOCK_SHA256:
            raise AttentionProbeError("runtime lock digest changed")
        lock = runtime_snapshot.parsed()
        validate_runtime_lock(lock, require_frozen=False)
        evidence_context: RuntimeEvidenceContext | None = None
        if allow_unbound_evidence_capture:
            if (
                lock["status"] != "candidate"
                or lock["evidence_lock_sha256"] is not None
            ):
                raise AttentionProbeError(
                    "unbound evidence capture requires an unbound candidate lock"
                )
            probe_mode = "unbound-evidence-capture"
        else:
            if lock["evidence_lock_sha256"] is None:
                raise AttentionProbeError("runtime evidence lock is not bound")
            evidence_context = verify_bound_static_runtime_evidence(
                lock,
                evidence_path=evidence_path,
            )
            base["runtime_evidence_sha256"] = (
                evidence_context.snapshot.sha256
            )
            probe_mode = "bound-verification"
        _require_isolated_import_context()

        asset_snapshot = load_asset_lock_snapshot(asset_lock_path)
        base["asset_lock_sha256"] = asset_snapshot.sha256
        if asset_snapshot.sha256 != CF1_ASSET_LOCK_SHA256:
            raise AttentionProbeError("asset lock digest changed")
        source_report = verify_model_source_snapshot(asset_snapshot, checkout)
        if not _source_is_verified(source_report):
            raise AttentionProbeError("verified source checkout is required")
        source, source_sha256 = _read_verified_attention_source(checkout)

        environment_root, stdlib_root, distribution_paths = _probe_identity_roots(
            evidence_context,
            unbound_environment_root=unbound_environment_root,
            unbound_stdlib_root=unbound_stdlib_root,
            unbound_distribution_paths=unbound_distribution_paths,
        )
        torch = _load_torch(distribution_paths, stdlib_root=stdlib_root)
        device = _validate_loaded_runtime(torch, lock, gpu_index=gpu_index)
        attention_module = _execute_attention_source(
            source,
            checkout / ATTENTION_RELATIVE_PATH,
        )
        _validate_attention_origins(
            attention_module,
            distribution_paths,
        )
        execution = _execute_attention_kernel(
            torch,
            attention_module,
            device=device,
        )
        loaded_facts = _capture_loaded_facts(
            torch,
            attention_module,
            attention_backend=execution["attention_backend"],
            environment_root=environment_root,
            stdlib_root=stdlib_root,
        )
        base["gpu_execution_performed"] = True
        expected_backend = lock["target"]["attention_backend"]
        if (
            expected_backend is not None
            and execution["attention_backend"] != expected_backend
        ):
            raise AttentionProbeError("executed attention backend does not match the pin")

        runtime_environment_sha256: str | None = None
        native_identity_sha256: str | None = None
        static_environment_sha256: str | None = None
        if evidence_context is not None:
            try:
                evidence_identity = verify_runtime_evidence(
                    evidence_context.snapshot,
                    observed_oci=evidence_context.observed_oci,
                    python_executable=evidence_context.python_executable,
                    python_implementation=platform.python_implementation(),
                    python_version=platform.python_version(),
                    python_build=tuple(platform.python_build()),
                    stdlib_root=evidence_context.stdlib_root,
                    environment_root=evidence_context.environment_root,
                    distribution_paths=evidence_context.distribution_paths,
                    wheelhouse=evidence_context.wheelhouse,
                    loaded_facts=loaded_facts,
                )
            except RuntimeEvidenceError as error:
                raise AttentionProbeError(
                    "loaded runtime evidence does not verify"
                ) from error
            if (
                evidence_identity.runtime_id != lock["runtime_id"]
                or evidence_identity.runtime_evidence_sha256
                != lock["evidence_lock_sha256"]
                or evidence_identity.image_manifest_digest
                != lock["image"]["digest"]
            ):
                raise AttentionProbeError(
                    "loaded runtime evidence identity is inconsistent"
                )
            runtime_environment_sha256 = evidence_identity.environment_sha256
            native_identity_sha256 = evidence_identity.native_identity_sha256
            static_environment_sha256 = (
                evidence_context.static_identity.static_environment_sha256
            )

        post_source_report = verify_model_source_snapshot(asset_snapshot, checkout)
        post_source, post_sha256 = _read_verified_attention_source(checkout)
        if (
            not _source_is_verified(post_source_report)
            or post_source != source
            or post_sha256 != source_sha256
        ):
            raise AttentionProbeError("attention source changed during probe")
        identity_material = {
            "schema_version": 3,
            "probe_mode": probe_mode,
            "runtime_lock_sha256": runtime_snapshot.sha256,
            "runtime_evidence_sha256": base["runtime_evidence_sha256"],
            "static_environment_sha256": static_environment_sha256,
            "runtime_environment_sha256": runtime_environment_sha256,
            "native_identity_sha256": native_identity_sha256,
            "asset_lock_sha256": asset_snapshot.sha256,
            "source_commit": CF1_SOURCE_COMMIT,
            "attention_source_sha256": source_sha256,
            "loaded_facts_sha256": _canonical_sha256(loaded_facts),
            **execution,
        }
        identity_sha256 = _canonical_sha256(identity_material)
    except (
        AttentionProbeError,
        RuntimePreflightError,
        RuntimeEvidenceError,
        AssetLockError,
    ) as error:
        return {**base, "failure": str(error)}
    except Exception as error:
        return {
            **base,
            "failure": f"unexpected probe error: {type(error).__name__}",
        }
    return {
        **base,
        "probe_succeeded": True,
        "gpu_execution_performed": True,
        "probe_mode": probe_mode,
        "source_commit": CF1_SOURCE_COMMIT,
        "attention_source_sha256": source_sha256,
        "loaded_facts": loaded_facts,
        "loaded_facts_sha256": _canonical_sha256(loaded_facts),
        "static_environment_sha256": static_environment_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "native_identity_sha256": native_identity_sha256,
        **execution,
        "probe_identity_sha256": identity_sha256,
    }


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_RUNTIME_LOCK_PATH)
    parser.add_argument("--asset-lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument(
        "--evidence-lock",
        type=Path,
        default=DEFAULT_RUNTIME_EVIDENCE_PATH,
    )
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--gpu-index", type=_nonnegative_int, default=0)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--stdlib-root", type=Path)
    parser.add_argument("--distribution-path", type=Path, action="append")
    parser.add_argument(
        "--bootstrap-unbound-evidence",
        action="store_true",
        help=(
            "explicitly capture non-authorizing loaded facts for an unbound "
            "candidate lock"
        ),
    )
    arguments = parser.parse_args(argv)
    report = attention_probe_report(
        arguments.lock,
        asset_lock_path=arguments.asset_lock,
        evidence_path=arguments.evidence_lock,
        checkout=arguments.checkout,
        gpu_index=arguments.gpu_index,
        allow_unbound_evidence_capture=arguments.bootstrap_unbound_evidence,
        unbound_environment_root=arguments.environment_root,
        unbound_stdlib_root=arguments.stdlib_root,
        unbound_distribution_paths=(
            tuple(arguments.distribution_path)
            if arguments.distribution_path is not None
            else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["probe_succeeded"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
