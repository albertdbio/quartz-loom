"""Cheap, fail-closed runtime/capacity checks for the pinned CF++1 worker.

This module is intentionally stdlib-only.  It can reject an unfrozen package
set or an undersized cgroup before importing Torch or streaming 17 GiB of model
assets through the page cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import struct
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from bench.cf_runtime_evidence import (
    DEFAULT_RUNTIME_EVIDENCE_PATH,
    RuntimeEvidenceError,
    RuntimeEvidenceSnapshot,
    RuntimeEvidenceStaticIdentity,
    load_runtime_evidence_snapshot,
    verify_static_runtime_evidence,
)

DEFAULT_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve().parent
    / "runtime"
    / "cf1-h100-cu128-v1.lock.json"
)

CF1_RUNTIME_ID = "cf1-h100-cu128-v1"
CF1_RUNTIME_LOCK_SHA256 = (
    "d4d163d635ecbafb5b11bbe54cca7bdd5e9f80c1edc23ed96972821940ecc692"
)
CF1_TOKENIZER_SENTINEL_SHA256 = (
    "2ab00c08615e582d62b163a0d13d305c04c3f3c99a45e034483413a7efb2210f"
)
CF1_TOKENIZER_SENTINEL_PROMPTS = (
    "  A  red\tfox\njumps.  ",
    "Caf\u00e9\u00a0\u732b",
    "<extra_id_0>",
)

_GIB = 1024**3
_MAX_LOCK_BYTES = 64 * 1024
_EXPECTED_TARGET = {
    "platform_system": "Linux",
    "machine": "x86_64",
    "python_implementation": "CPython",
    "python_major_minor": "3.12",
    "cuda_runtime": "12.8",
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "gpu_count": 1,
    "compute_capability": [9, 0],
}
_EXPECTED_CAPACITY = {
    "load_strategy": "fp32-construct-mmap-assign-cpu-cast",
    "minimum_host_headroom_bytes": 56 * _GIB,
    "minimum_gpu_total_bytes": 80_000_000_000,
    "minimum_gpu_free_bytes": 36 * _GIB,
    "locked_asset_file_bytes": 17_082_290_819,
    "known_host_active_storage_bytes": 39_761_449_216,
    "known_host_no_reclamation_envelope_bytes": 48_364_777_331,
    "gpu_weight_storage_bytes": 14_222_445_350,
    "gpu_cache_ready_lower_bound_bytes": 20_355_140_390,
    "archived_rolling_peak_allocated_bytes": 24_822_243_328,
    "archived_rolling_peak_reserved_bytes": 29_360_128_000,
    "archived_full_batch_peak_allocated_bytes": 38_896_813_056,
    "archived_full_batch_peak_reserved_bytes": 50_545_557_504,
    "swap_counts_toward_headroom": False,
}
_EXPECTED_TOKENIZER = {
    "sentinel_version": 1,
    "rows": 3,
    "columns": 512,
    "vocab_size": 256300,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "unk_token_id": 3,
    "extra_id_0": 256299,
    "sentinel_sha256": CF1_TOKENIZER_SENTINEL_SHA256,
}
_TOKENIZER_PREFIXES = (
    (320, 4062, 273, 56209, 48150, 281, 274, 1),
    (25382, 273, 14985, 1),
    (256299, 1),
)
_EVIDENCE_STATES = frozenset({"observed", "selected", "reconstructed"})
_REQUIRED_PACKAGE_NAMES = frozenset(
    {
        "diffusers",
        "easydict",
        "einops",
        "flash-attn",
        "ftfy",
        "huggingface-hub",
        "numpy",
        "omegaconf",
        "regex",
        "safetensors",
        "sentencepiece",
        "tokenizers",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
    }
)
_ATTENTION_BACKENDS = frozenset(
    {"flash-attention-2", "flash-attention-3"}
)
_ROOT_FIELDS = {
    "schema_version",
    "runtime_id",
    "status",
    "image",
    "target",
    "packages",
    "capacity",
    "tokenizer",
    "unresolved",
    "evidence_lock_sha256",
}


class RuntimePreflightError(ValueError):
    """The worker environment is not safe or reproducible enough to boot."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimePreflightError(f"duplicate runtime lock key: {key}")
        value[key] = item
    return value


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


def _normalize_oci_sha256(value: str | None) -> str | None:
    if isinstance(value, str) and _is_sha256(value):
        return f"sha256:{value}"
    return value


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


@dataclass(frozen=True)
class RuntimeLockSnapshot:
    encoded: bytes
    sha256: str

    def parsed(self) -> dict[str, Any]:
        try:
            value = json.loads(
                self.encoded.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimePreflightError("runtime lock is not valid JSON") from error
        if not isinstance(value, dict):
            raise RuntimePreflightError("runtime lock must be an object")
        return value


@dataclass
class RuntimeObservation:
    image_digest: str | None
    platform_system: str
    machine: str
    python_implementation: str
    python_version: str
    python_build: tuple[str, str]
    package_versions: dict[str, str]
    mem_available_bytes: int
    cgroup_state: str
    cgroup_headroom_bytes: int | None
    swap_total_bytes: int
    nvidia_driver_version: str
    gpu_name: str
    gpu_compute_capability: tuple[int, int]
    gpu_total_bytes: int
    gpu_free_bytes: int


@dataclass(frozen=True)
class HostMemoryObservation:
    mem_available_bytes: int
    cgroup_state: str
    cgroup_headroom_bytes: int | None
    swap_total_bytes: int


@dataclass(frozen=True)
class RuntimePreflightIdentity:
    runtime_id: str
    runtime_lock_sha256: str
    runtime_evidence_sha256: str
    static_environment_sha256: str
    environment_sha256: str
    effective_host_headroom_bytes: int
    gpu_total_bytes: int
    gpu_free_bytes: int


@dataclass(frozen=True)
class RuntimeEvidenceContext:
    snapshot: RuntimeEvidenceSnapshot
    static_identity: RuntimeEvidenceStaticIdentity
    observed_oci: Mapping[str, Any]
    python_executable: Path
    stdlib_root: Path
    environment_root: Path
    distribution_paths: tuple[Path, ...]
    wheelhouse: Path


def load_runtime_lock_snapshot(
    path: Path = DEFAULT_RUNTIME_LOCK_PATH,
) -> RuntimeLockSnapshot:
    try:
        size = path.stat().st_size
        if not 0 < size <= _MAX_LOCK_BYTES:
            raise RuntimePreflightError("runtime lock size is invalid")
        encoded = path.read_bytes()
    except OSError as error:
        raise RuntimePreflightError("runtime lock could not be read") from error
    if len(encoded) != size:
        raise RuntimePreflightError("runtime lock changed while being read")
    snapshot = RuntimeLockSnapshot(
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
    snapshot.parsed()
    return snapshot


def validate_runtime_lock(
    lock: Mapping[str, Any],
    *,
    require_frozen: bool,
) -> None:
    if not isinstance(lock, Mapping) or set(lock) != _ROOT_FIELDS:
        raise RuntimePreflightError("runtime lock fields do not match schema v2")
    if lock["schema_version"] != 2 or lock["runtime_id"] != CF1_RUNTIME_ID:
        raise RuntimePreflightError("runtime lock identity does not match CF++1")
    status = lock["status"]
    if status not in {"candidate", "frozen"}:
        raise RuntimePreflightError("runtime lock status is invalid")

    image = lock["image"]
    if (
        not isinstance(image, Mapping)
        or set(image) != {"tag", "digest"}
        or not isinstance(image["tag"], str)
        or not image["tag"]
        or (image["digest"] is not None and not _is_oci_sha256(image["digest"]))
    ):
        raise RuntimePreflightError("runtime image identity is invalid")

    target = lock["target"]
    expected_target_fields = set(_EXPECTED_TARGET) | {
        "python_version",
        "python_build",
        "nvidia_driver_version",
        "attention_backend",
    }
    if not isinstance(target, Mapping) or set(target) != expected_target_fields:
        raise RuntimePreflightError("runtime target fields are invalid")
    for field, expected in _EXPECTED_TARGET.items():
        if target[field] != expected:
            raise RuntimePreflightError(f"runtime target {field} changed")
    python_version = target["python_version"]
    if python_version is not None and (
        not isinstance(python_version, str)
        or not python_version.startswith(target["python_major_minor"] + ".")
    ):
        raise RuntimePreflightError("runtime Python version is invalid")
    python_build = target["python_build"]
    if python_build is not None and (
        not isinstance(python_build, list)
        or len(python_build) != 2
        or any(not isinstance(item, str) or not item for item in python_build)
    ):
        raise RuntimePreflightError("runtime Python build is invalid")
    driver_version = target["nvidia_driver_version"]
    if driver_version is not None and (
        not isinstance(driver_version, str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver_version) is None
    ):
        raise RuntimePreflightError("runtime NVIDIA driver version is invalid")
    attention_backend = target["attention_backend"]
    if attention_backend is not None and attention_backend not in _ATTENTION_BACKENDS:
        raise RuntimePreflightError("runtime attention backend is invalid")

    packages = lock["packages"]
    if not isinstance(packages, list) or not packages:
        raise RuntimePreflightError("runtime package pins are required")
    names: set[str] = set()
    for package in packages:
        if not isinstance(package, Mapping) or set(package) != {
            "distribution",
            "version",
            "evidence",
        }:
            raise RuntimePreflightError("runtime package pin is invalid")
        distribution = package["distribution"]
        version = package["version"]
        evidence = package["evidence"]
        if (
            not isinstance(distribution, str)
            or not distribution
            or not isinstance(version, str)
            or not version
            or any(character.isspace() for character in version)
            or evidence not in _EVIDENCE_STATES
        ):
            raise RuntimePreflightError("runtime package pin is invalid")
        normalized = _normalize_distribution(distribution)
        if normalized in names:
            raise RuntimePreflightError(
                f"duplicate runtime package pin: {distribution}"
            )
        names.add(normalized)
    if not _REQUIRED_PACKAGE_NAMES.issubset(names):
        raise RuntimePreflightError("runtime package inventory is incomplete")

    capacity = lock["capacity"]
    if not isinstance(capacity, Mapping) or dict(capacity) != _EXPECTED_CAPACITY:
        raise RuntimePreflightError("runtime capacity contract changed")
    if any(
        not _positive_int(value)
        for key, value in capacity.items()
        if key.endswith("_bytes")
    ):
        raise RuntimePreflightError("runtime capacity bytes are invalid")

    tokenizer = lock["tokenizer"]
    if not isinstance(tokenizer, Mapping) or dict(tokenizer) != _EXPECTED_TOKENIZER:
        raise RuntimePreflightError("runtime tokenizer contract changed")

    unresolved = lock["unresolved"]
    if (
        not isinstance(unresolved, list)
        or any(not isinstance(item, str) or not item for item in unresolved)
        or len(set(unresolved)) != len(unresolved)
    ):
        raise RuntimePreflightError("runtime unresolved list is invalid")

    evidence_lock_sha256 = lock["evidence_lock_sha256"]
    if evidence_lock_sha256 is not None and not _is_sha256(
        evidence_lock_sha256
    ):
        raise RuntimePreflightError("runtime evidence lock digest is invalid")

    if require_frozen and (
        status != "frozen"
        or unresolved
        or image["digest"] is None
        or python_version is None
        or python_build is None
        or driver_version is None
        or attention_backend is None
        or evidence_lock_sha256 is None
        or any(package["evidence"] != "observed" for package in packages)
    ):
        raise RuntimePreflightError("runtime lock is not frozen")


def _required_environment_value(name: str) -> str:
    value = os.environ.get(name)
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimePreflightError(
            f"trusted launcher assertion {name} is missing or invalid"
        )
    return value


def _required_oci_environment_value(name: str) -> str:
    value = _required_environment_value(name)
    if not _is_oci_sha256(value):
        raise RuntimePreflightError(
            f"trusted launcher assertion {name} is not an OCI digest"
        )
    return value


def _required_absolute_environment_path(name: str) -> Path:
    value = _required_environment_value(name)
    path = Path(value)
    if not path.is_absolute():
        raise RuntimePreflightError(
            f"trusted launcher assertion {name} is not an absolute path"
        )
    return path


def _bound_evidence_contract(
    lock: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    if evidence["runtime_id"] != lock["runtime_id"]:
        raise RuntimePreflightError("runtime evidence ID does not match lock")
    oci = evidence["oci"]
    if (
        oci["tag"] != lock["image"]["tag"]
        or oci["manifest_digest"] != lock["image"]["digest"]
    ):
        raise RuntimePreflightError("runtime evidence OCI identity does not match lock")
    python_evidence = evidence["python"]
    target = lock["target"]
    if (
        python_evidence["implementation"] != target["python_implementation"]
        or python_evidence["version"] != target["python_version"]
        or python_evidence["build"] != target["python_build"]
    ):
        raise RuntimePreflightError(
            "runtime evidence Python identity does not match lock"
        )
    expected_packages = {
        _normalize_distribution(package["distribution"]): package["version"]
        for package in lock["packages"]
    }
    evidence_packages = {
        package["distribution"]: package["version"]
        for package in evidence["environment"]["packages"]
    }
    if evidence_packages != expected_packages:
        raise RuntimePreflightError(
            "runtime evidence package inventory does not match lock"
        )
    native = evidence["native"]
    expected_torch = expected_packages.get("torch")
    if (
        native["torch_version"] != expected_torch
        or native["cuda_runtime"] != target["cuda_runtime"]
        or native["import_order"]
        != ["torch", "verified-attention-source", target["attention_backend"]]
        or "sm_90" not in native["cuda_arch_list"]
    ):
        raise RuntimePreflightError(
            "runtime evidence native identity does not match lock"
        )


def verify_bound_static_runtime_evidence(
    lock: Mapping[str, Any],
    *,
    evidence_path: Path = DEFAULT_RUNTIME_EVIDENCE_PATH,
) -> RuntimeEvidenceContext:
    """Verify the separately reviewed byte evidence before GPU/Torch work."""

    validate_runtime_lock(lock, require_frozen=False)
    expected_sha256 = lock["evidence_lock_sha256"]
    if expected_sha256 is None:
        raise RuntimePreflightError("runtime evidence lock is not bound")
    try:
        snapshot = load_runtime_evidence_snapshot(evidence_path)
    except RuntimeEvidenceError as error:
        raise RuntimePreflightError("runtime evidence lock could not be loaded") from error
    if snapshot.sha256 != expected_sha256:
        raise RuntimePreflightError("runtime evidence lock digest changed")
    evidence = snapshot.parsed()
    _bound_evidence_contract(lock, evidence)

    observed_oci = {
        "tag": lock["image"]["tag"],
        "index_digest": _required_oci_environment_value(
            "CF1_RUNTIME_IMAGE_INDEX_DIGEST"
        ),
        "platform": {"os": "linux", "architecture": "amd64"},
        "manifest_digest": _required_oci_environment_value(
            "CF1_RUNTIME_IMAGE_DIGEST"
        ),
        "config_digest": _required_oci_environment_value(
            "CF1_RUNTIME_IMAGE_CONFIG_DIGEST"
        ),
    }
    if observed_oci["manifest_digest"] != lock["image"]["digest"]:
        raise RuntimePreflightError(
            "trusted launcher child image digest does not match lock"
        )
    environment_root = _required_absolute_environment_path(
        "CF1_RUNTIME_ENVIRONMENT_ROOT"
    )
    distribution_paths = (
        _required_absolute_environment_path("CF1_RUNTIME_DISTRIBUTION_PATH"),
    )
    wheelhouse = _required_absolute_environment_path("CF1_RUNTIME_WHEELHOUSE")
    stdlib_value = sysconfig.get_path("stdlib")
    if not isinstance(stdlib_value, str) or not stdlib_value:
        raise RuntimePreflightError("runtime stdlib path is unavailable")
    try:
        static_identity = verify_static_runtime_evidence(
            snapshot,
            observed_oci=observed_oci,
            python_executable=Path(sys.executable),
            python_implementation=platform.python_implementation(),
            python_version=platform.python_version(),
            python_build=tuple(platform.python_build()),
            stdlib_root=Path(stdlib_value),
            environment_root=environment_root,
            distribution_paths=distribution_paths,
            wheelhouse=wheelhouse,
        )
    except RuntimeEvidenceError as error:
        raise RuntimePreflightError("static runtime evidence does not verify") from error
    if (
        static_identity.runtime_id != lock["runtime_id"]
        or static_identity.runtime_evidence_sha256 != expected_sha256
        or static_identity.image_manifest_digest != lock["image"]["digest"]
    ):
        raise RuntimePreflightError("static runtime evidence identity is inconsistent")
    return RuntimeEvidenceContext(
        snapshot=snapshot,
        static_identity=static_identity,
        observed_oci=observed_oci,
        python_executable=Path(sys.executable),
        stdlib_root=Path(stdlib_value),
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_host_headroom(
    observation: RuntimeObservation | HostMemoryObservation,
) -> int:
    if observation.cgroup_state not in {"bounded", "unbounded"}:
        raise RuntimePreflightError("runtime cgroup memory state is ambiguous")
    if not _positive_int(observation.mem_available_bytes):
        raise RuntimePreflightError("runtime MemAvailable is invalid")
    if observation.cgroup_state == "bounded":
        if not _positive_int(observation.cgroup_headroom_bytes):
            raise RuntimePreflightError("runtime cgroup headroom is invalid")
        return min(
            observation.mem_available_bytes,
            observation.cgroup_headroom_bytes,
        )
    if observation.cgroup_headroom_bytes is not None:
        raise RuntimePreflightError("unbounded cgroup reported finite headroom")
    return observation.mem_available_bytes


def preflight_runtime_environment(
    snapshot: RuntimeLockSnapshot,
    observation: RuntimeObservation,
    *,
    static_identity: RuntimeEvidenceStaticIdentity | None = None,
) -> RuntimePreflightIdentity:
    if hashlib.sha256(snapshot.encoded).hexdigest() != snapshot.sha256:
        raise RuntimePreflightError("runtime lock snapshot digest is invalid")
    lock = snapshot.parsed()
    validate_runtime_lock(lock, require_frozen=True)
    if (
        not isinstance(static_identity, RuntimeEvidenceStaticIdentity)
        or static_identity.runtime_id != lock["runtime_id"]
        or static_identity.runtime_evidence_sha256
        != lock["evidence_lock_sha256"]
        or static_identity.image_manifest_digest != lock["image"]["digest"]
    ):
        raise RuntimePreflightError("bound static runtime evidence is required")
    target = lock["target"]

    if observation.image_digest != lock["image"]["digest"]:
        raise RuntimePreflightError("runtime image digest does not match the pin")

    expected_scalars = {
        "platform_system": observation.platform_system,
        "machine": observation.machine,
        "python_implementation": observation.python_implementation,
        "gpu_name": observation.gpu_name,
    }
    for field, observed in expected_scalars.items():
        if observed != target[field]:
            raise RuntimePreflightError(f"runtime {field} does not match the pin")
    if tuple(target["compute_capability"]) != tuple(
        observation.gpu_compute_capability
    ):
        raise RuntimePreflightError("runtime GPU compute capability does not match")
    if target["python_version"] is not None:
        python_matches = observation.python_version == target["python_version"]
    else:
        python_matches = observation.python_version.startswith(
            target["python_major_minor"] + "."
        )
    if not python_matches:
        raise RuntimePreflightError("runtime Python version does not match the pin")
    if tuple(target["python_build"]) != tuple(observation.python_build):
        raise RuntimePreflightError("runtime Python build does not match the pin")
    if observation.nvidia_driver_version != target["nvidia_driver_version"]:
        raise RuntimePreflightError(
            "runtime NVIDIA driver version does not match the pin"
        )

    observed_packages: dict[str, str] = {}
    for distribution, version in observation.package_versions.items():
        name = _normalize_distribution(distribution)
        if name in observed_packages:
            raise RuntimePreflightError(
                f"runtime package observation is ambiguous: {distribution}"
            )
        observed_packages[name] = version
    expected_packages = {
        _normalize_distribution(package["distribution"])
        for package in lock["packages"]
    }
    if set(observed_packages) != expected_packages:
        raise RuntimePreflightError("runtime package inventory does not match the pin")
    for package in lock["packages"]:
        name = _normalize_distribution(package["distribution"])
        if observed_packages.get(name) != package["version"]:
            raise RuntimePreflightError(
                f"runtime package does not match pin: {package['distribution']}"
            )

    effective_host = _effective_host_headroom(observation)
    capacity = lock["capacity"]
    if effective_host < capacity["minimum_host_headroom_bytes"]:
        raise RuntimePreflightError("runtime host headroom is below 56 GiB")
    if (
        not _positive_int(observation.gpu_total_bytes)
        or observation.gpu_total_bytes < capacity["minimum_gpu_total_bytes"]
    ):
        raise RuntimePreflightError("runtime GPU total memory is below the pin")
    if (
        not _positive_int(observation.gpu_free_bytes)
        or observation.gpu_free_bytes < capacity["minimum_gpu_free_bytes"]
    ):
        raise RuntimePreflightError("runtime GPU free memory is below 36 GiB")
    if observation.gpu_free_bytes > observation.gpu_total_bytes:
        raise RuntimePreflightError("runtime GPU memory accounting is invalid")

    environment = {
        "runtime_lock_sha256": snapshot.sha256,
        "runtime_evidence_sha256": static_identity.runtime_evidence_sha256,
        "static_environment_sha256": static_identity.static_environment_sha256,
        "image_tag": lock["image"]["tag"],
        "image_digest": observation.image_digest,
        "platform_system": observation.platform_system,
        "machine": observation.machine,
        "python_implementation": observation.python_implementation,
        "python_version": observation.python_version,
        "python_build": list(observation.python_build),
        "packages": sorted(observed_packages.items()),
        "nvidia_driver_version": observation.nvidia_driver_version,
        "gpu_name": observation.gpu_name,
        "gpu_compute_capability": list(observation.gpu_compute_capability),
    }
    return RuntimePreflightIdentity(
        runtime_id=lock["runtime_id"],
        runtime_lock_sha256=snapshot.sha256,
        runtime_evidence_sha256=static_identity.runtime_evidence_sha256,
        static_environment_sha256=static_identity.static_environment_sha256,
        environment_sha256=_canonical_sha256(environment),
        effective_host_headroom_bytes=effective_host,
        gpu_total_bytes=observation.gpu_total_bytes,
        gpu_free_bytes=observation.gpu_free_bytes,
    )


def _parse_meminfo(path: Path) -> tuple[int, int]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimePreflightError("/proc/meminfo could not be read") from error
    values: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit() and parts[2] == "kB":
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    if "MemAvailable" not in values or "SwapTotal" not in values:
        raise RuntimePreflightError("/proc/meminfo is missing capacity fields")
    return values["MemAvailable"], values["SwapTotal"]


def _read_cgroup_integer(path: Path, label: str) -> int:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimePreflightError(f"cgroup {label} could not be read") from error
    if not value.isdigit():
        raise RuntimePreflightError(f"cgroup {label} is invalid")
    parsed = int(value)
    if parsed < 0:
        raise RuntimePreflightError(f"cgroup {label} is invalid")
    return parsed


def _cgroup_membership(
    proc_root: Path,
    *,
    hierarchy: str,
) -> tuple[str, ...]:
    try:
        lines = (proc_root / "self" / "cgroup").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimePreflightError(
            f"cgroup {hierarchy} membership is ambiguous"
        ) from error
    matches: list[str] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if hierarchy == "v2" and parts[0] == "0" and parts[1] == "":
            matches.append(parts[2])
        elif hierarchy == "v1" and "memory" in parts[1].split(","):
            matches.append(parts[2])
    if len(matches) != 1:
        raise RuntimePreflightError(
            f"cgroup {hierarchy} memory membership is ambiguous"
        )
    raw = matches[0]
    parsed = PurePosixPath(raw)
    if not raw.startswith("/") or any(
        part in {"", ".", ".."} for part in parsed.parts[1:]
    ):
        raise RuntimePreflightError(
            f"cgroup {hierarchy} memory membership is invalid"
        )
    return tuple(parsed.parts[1:])


def _cgroup_ancestors(root: Path, relative: tuple[str, ...]) -> tuple[Path, ...]:
    current = root.joinpath(*relative)
    paths: list[Path] = []
    while True:
        paths.append(current)
        if current == root:
            return tuple(paths)
        parent = current.parent
        if parent == current:
            raise RuntimePreflightError("cgroup hierarchy escaped its mount")
        current = parent


def _v2_cgroup_headroom(
    proc_root: Path,
    cgroup_root: Path,
) -> tuple[str, int | None]:
    relative = _cgroup_membership(proc_root, hierarchy="v2")
    headrooms: list[int] = []
    for base in _cgroup_ancestors(cgroup_root, relative):
        try:
            maximum_text = (base / "memory.max").read_text(
                encoding="ascii"
            ).strip()
        except (OSError, UnicodeError) as error:
            raise RuntimePreflightError(
                "cgroup v2 memory state is ambiguous"
            ) from error
        current = _read_cgroup_integer(base / "memory.current", "memory.current")
        if maximum_text == "max":
            continue
        if not maximum_text.isdigit():
            raise RuntimePreflightError("cgroup v2 memory.max is invalid")
        maximum = int(maximum_text)
        if maximum <= 0 or current > maximum:
            raise RuntimePreflightError("cgroup v2 memory accounting is invalid")
        headrooms.append(maximum - current)
    if not headrooms:
        return "unbounded", None
    return "bounded", min(headrooms)


def _v1_cgroup_headroom(
    proc_root: Path,
    cgroup_root: Path,
) -> tuple[str, int | None]:
    relative = _cgroup_membership(proc_root, hierarchy="v1")
    hierarchy_root = cgroup_root / "memory"
    headrooms: list[int] = []
    for base in _cgroup_ancestors(hierarchy_root, relative):
        maximum = _read_cgroup_integer(
            base / "memory.limit_in_bytes",
            "memory.limit_in_bytes",
        )
        current = _read_cgroup_integer(
            base / "memory.usage_in_bytes",
            "memory.usage_in_bytes",
        )
        if maximum >= 1 << 60:
            continue
        if maximum <= 0 or current > maximum:
            raise RuntimePreflightError("cgroup v1 memory accounting is invalid")
        headrooms.append(maximum - current)
    if not headrooms:
        return "unbounded", None
    return "bounded", min(headrooms)


def observe_host_memory(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> HostMemoryObservation:
    mem_available, swap_total = _parse_meminfo(proc_root / "meminfo")
    if (cgroup_root / "cgroup.controllers").exists():
        state, headroom = _v2_cgroup_headroom(proc_root, cgroup_root)
    else:
        state, headroom = _v1_cgroup_headroom(proc_root, cgroup_root)
    return HostMemoryObservation(
        mem_available_bytes=mem_available,
        cgroup_state=state,
        cgroup_headroom_bytes=headroom,
        swap_total_bytes=swap_total,
    )


def _observe_gpu(
    gpu_index: int,
) -> tuple[str, str, tuple[int, int], int, int]:
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    if gpu_index != 0:
        raise RuntimePreflightError(
            "the pinned runtime requires exactly one GPU at logical index zero"
        )
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimePreflightError("selected GPU could not be observed") from error
    rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimePreflightError("selected GPU report is ambiguous")
    name, driver_version, capability_text, total_mib, free_mib = (
        value.strip() for value in rows[0]
    )
    capability_parts = capability_text.split(".")
    if (
        len(capability_parts) != 2
        or not all(part.isdigit() for part in capability_parts)
        or not total_mib.isdigit()
        or not free_mib.isdigit()
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver_version) is None
    ):
        raise RuntimePreflightError("selected GPU report is invalid")
    total_bytes = int(total_mib) * 1024**2
    free_bytes = int(free_mib) * 1024**2
    if free_bytes > total_bytes:
        raise RuntimePreflightError("selected GPU memory accounting is invalid")
    return (
        name,
        driver_version,
        (int(capability_parts[0]), int(capability_parts[1])),
        total_bytes,
        free_bytes,
    )


def _collect_package_versions(
    distribution_paths: tuple[Path, ...] | None,
) -> dict[str, str]:
    if distribution_paths is None:
        distributions = importlib.metadata.distributions()
    else:
        resolved_paths: list[Path] = []
        for distribution_path in distribution_paths:
            try:
                resolved = distribution_path.resolve(strict=True)
            except OSError as error:
                raise RuntimePreflightError(
                    "trusted runtime distribution path is unavailable"
                ) from error
            if not resolved.is_dir() or resolved in resolved_paths:
                raise RuntimePreflightError(
                    "trusted runtime distribution path is invalid"
                )
            resolved_paths.append(resolved)
        if not resolved_paths:
            raise RuntimePreflightError(
                "trusted runtime distribution paths are empty"
            )
        distributions = importlib.metadata.distributions(
            path=[str(path) for path in resolved_paths]
        )
    packages: dict[str, str] = {}
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not name or not isinstance(version, str):
            raise RuntimePreflightError("installed package metadata is invalid")
        normalized = _normalize_distribution(name)
        if normalized in packages:
            raise RuntimePreflightError(
                f"installed package metadata is ambiguous: {name}"
            )
        packages[normalized] = version
    return packages


def _collect_runtime_observation(
    *,
    gpu_index: int,
    distribution_paths: tuple[Path, ...] | None = None,
) -> RuntimeObservation:
    """Collect stdlib-only facts without deciding whether they authorize boot."""

    image_digest = _normalize_oci_sha256(
        os.environ.get("CF1_RUNTIME_IMAGE_DIGEST")
    )
    if image_digest is not None and not _is_oci_sha256(image_digest):
        raise RuntimePreflightError(
            "trusted runtime image digest assertion is invalid"
        )
    packages = _collect_package_versions(distribution_paths)
    host = observe_host_memory()
    gpu_name, driver_version, capability, gpu_total, gpu_free = _observe_gpu(
        gpu_index
    )
    return RuntimeObservation(
        # This is a trusted-launcher assertion, not in-container attestation.
        # The launcher must derive it from provider-observed deployment metadata.
        image_digest=image_digest,
        platform_system=platform.system(),
        machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_build=tuple(platform.python_build()),
        package_versions=packages,
        mem_available_bytes=host.mem_available_bytes,
        cgroup_state=host.cgroup_state,
        cgroup_headroom_bytes=host.cgroup_headroom_bytes,
        swap_total_bytes=host.swap_total_bytes,
        nvidia_driver_version=driver_version,
        gpu_name=gpu_name,
        gpu_compute_capability=capability,
        gpu_total_bytes=gpu_total,
        gpu_free_bytes=gpu_free,
    )


def observe_runtime_environment(
    lock: Mapping[str, Any],
    *,
    gpu_index: int,
    distribution_paths: tuple[Path, ...],
) -> RuntimeObservation:
    """Collect facts only after the lock is already frozen and valid."""

    validate_runtime_lock(lock, require_frozen=True)
    return _collect_runtime_observation(
        gpu_index=gpu_index,
        distribution_paths=distribution_paths,
    )


def capture_runtime_environment(
    lock: Mapping[str, Any],
    *,
    gpu_index: int,
) -> RuntimeObservation:
    """Collect candidate facts without comparing them or authorizing boot."""

    validate_runtime_lock(lock, require_frozen=False)
    return _collect_runtime_observation(gpu_index=gpu_index)


def preflight_current_runtime(
    snapshot: RuntimeLockSnapshot,
    *,
    gpu_index: int,
    evidence_path: Path = DEFAULT_RUNTIME_EVIDENCE_PATH,
) -> RuntimePreflightIdentity:
    lock = snapshot.parsed()
    validate_runtime_lock(lock, require_frozen=True)
    evidence_context = verify_bound_static_runtime_evidence(
        lock,
        evidence_path=evidence_path,
    )
    return preflight_runtime_environment(
        snapshot,
        observe_runtime_environment(
            lock,
            gpu_index=gpu_index,
            distribution_paths=evidence_context.distribution_paths,
        ),
        static_identity=evidence_context.static_identity,
    )


def validate_current_host_capacity(snapshot: RuntimeLockSnapshot) -> int:
    """Recheck cgroup-aware host headroom immediately before constructors."""

    if hashlib.sha256(snapshot.encoded).hexdigest() != snapshot.sha256:
        raise RuntimePreflightError("runtime lock snapshot digest is invalid")
    lock = snapshot.parsed()
    validate_runtime_lock(lock, require_frozen=True)
    effective_host = _effective_host_headroom(observe_host_memory())
    if effective_host < lock["capacity"]["minimum_host_headroom_bytes"]:
        raise RuntimePreflightError("runtime host headroom is below 56 GiB")
    return effective_host


def validate_loaded_cuda_capacity(
    torch: Any,
    snapshot: RuntimeLockSnapshot,
    *,
    device: Any,
    attention_backend: str,
) -> tuple[int, int]:
    """Re-check the selected CUDA device after Torch creates its context."""

    lock = snapshot.parsed()
    validate_runtime_lock(lock, require_frozen=True)
    expected_torch = next(
        (
            package["version"]
            for package in lock["packages"]
            if _normalize_distribution(package["distribution"]) == "torch"
        ),
        None,
    )
    if expected_torch is None:
        raise RuntimePreflightError("runtime Torch pin is missing")
    if getattr(torch, "__version__", None) != expected_torch:
        raise RuntimePreflightError("loaded Torch version does not match the pin")
    if getattr(getattr(torch, "version", None), "cuda", None) != lock["target"][
        "cuda_runtime"
    ]:
        raise RuntimePreflightError("loaded Torch CUDA runtime does not match the pin")
    if attention_backend != lock["target"]["attention_backend"]:
        raise RuntimePreflightError(
            "loaded attention backend does not match the pin"
        )
    try:
        name = torch.cuda.get_device_name(device)
        capability = tuple(torch.cuda.get_device_capability(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception as error:
        raise RuntimePreflightError("loaded CUDA device could not be observed") from error
    if name != lock["target"]["gpu_name"]:
        raise RuntimePreflightError("loaded CUDA device name does not match the pin")
    if capability != tuple(lock["target"]["compute_capability"]):
        raise RuntimePreflightError("loaded CUDA compute capability does not match")
    capacity = lock["capacity"]
    if (
        not _positive_int(total_bytes)
        or total_bytes < capacity["minimum_gpu_total_bytes"]
    ):
        raise RuntimePreflightError("loaded CUDA total memory is below the pin")
    if (
        not _positive_int(free_bytes)
        or free_bytes < capacity["minimum_gpu_free_bytes"]
    ):
        raise RuntimePreflightError("loaded CUDA free memory is below 36 GiB")
    if free_bytes > total_bytes:
        raise RuntimePreflightError("loaded CUDA memory accounting is invalid")
    return int(free_bytes), int(total_bytes)


def _tensor_matrix(
    value: Any,
    *,
    label: str,
    expected_dtype: object,
) -> list[list[int]]:
    if tuple(getattr(value, "shape", ())) != (3, 512):
        raise RuntimePreflightError(f"tokenizer {label} shape is invalid")
    if getattr(value, "dtype", None) != expected_dtype:
        raise RuntimePreflightError(f"tokenizer {label} dtype is invalid")
    if getattr(getattr(value, "device", None), "type", None) != "cpu":
        raise RuntimePreflightError(f"tokenizer {label} must remain on CPU")
    try:
        rows = value.tolist()
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimePreflightError(f"tokenizer {label} is not materialized") from error
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or any(not isinstance(row, list) or len(row) != 512 for row in rows)
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for row in rows
            for item in row
        )
    ):
        raise RuntimePreflightError(f"tokenizer {label} matrix is invalid")
    return rows


def _tokenizer_sentinel_sha256(
    ids: list[list[int]],
    masks: list[list[int]],
) -> str:
    digest = hashlib.sha256(b"cf1-tokenizer-sentinel-v1\0")
    digest.update(struct.pack(">II", 3, 512))
    try:
        for row in ids:
            for item in row:
                digest.update(struct.pack(">I", item))
        for row in masks:
            digest.update(bytes(row))
    except (OverflowError, struct.error, ValueError) as error:
        raise RuntimePreflightError("tokenizer sentinel values are invalid") from error
    return digest.hexdigest()


def validate_cf1_tokenizer_sentinel(
    tokenizer: Any,
    *,
    expected_dtype: object,
) -> str:
    inner = getattr(tokenizer, "tokenizer", None)
    try:
        length = len(inner)
        extra_id = inner.convert_tokens_to_ids("<extra_id_0>")
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimePreflightError("tokenizer implementation is invalid") from error
    if getattr(inner, "is_fast", None) is not True:
        raise RuntimePreflightError("tokenizer must use the pinned fast implementation")
    if (
        getattr(inner, "vocab_size", None) != 256300
        or length != 256300
        or getattr(inner, "pad_token_id", None) != 0
        or getattr(inner, "eos_token_id", None) != 1
        or getattr(inner, "unk_token_id", None) != 3
        or extra_id != 256299
    ):
        raise RuntimePreflightError("tokenizer special-token identity changed")
    try:
        ids_tensor, masks_tensor = tokenizer(
            CF1_TOKENIZER_SENTINEL_PROMPTS,
            return_mask=True,
            add_special_tokens=True,
        )
    except Exception as error:
        raise RuntimePreflightError("tokenizer sentinel call failed") from error
    ids = _tensor_matrix(
        ids_tensor,
        label="input IDs",
        expected_dtype=expected_dtype,
    )
    masks = _tensor_matrix(
        masks_tensor,
        label="attention mask",
        expected_dtype=expected_dtype,
    )
    for index, prefix in enumerate(_TOKENIZER_PREFIXES):
        expected_ids = list(prefix) + [0] * (512 - len(prefix))
        expected_mask = [1] * len(prefix) + [0] * (512 - len(prefix))
        if ids[index] != expected_ids or masks[index] != expected_mask:
            raise RuntimePreflightError(
                f"tokenizer sentinel row {index} does not match the pin"
            )
    digest = _tokenizer_sentinel_sha256(ids, masks)
    if digest != CF1_TOKENIZER_SENTINEL_SHA256:
        raise RuntimePreflightError("tokenizer sentinel digest does not match the pin")
    return digest


def runtime_preflight_report(
    path: Path = DEFAULT_RUNTIME_LOCK_PATH,
    *,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """Return a machine-readable boot authorization without importing Torch."""

    snapshot: RuntimeLockSnapshot | None = None
    try:
        snapshot = load_runtime_lock_snapshot(path)
        if snapshot.sha256 != CF1_RUNTIME_LOCK_SHA256:
            raise RuntimePreflightError("runtime lock digest changed")
        identity = preflight_current_runtime(snapshot, gpu_index=gpu_index)
    except RuntimePreflightError as error:
        return {
            "ready": False,
            "runtime_id": CF1_RUNTIME_ID,
            "runtime_lock_sha256": (
                snapshot.sha256 if snapshot is not None else None
            ),
            "failure": str(error),
        }
    except Exception as error:
        return {
            "ready": False,
            "runtime_id": CF1_RUNTIME_ID,
            "runtime_lock_sha256": (
                snapshot.sha256 if snapshot is not None else None
            ),
            "failure": f"unexpected preflight error: {type(error).__name__}",
        }
    return {
        "ready": True,
        "runtime_id": identity.runtime_id,
        "runtime_lock_sha256": identity.runtime_lock_sha256,
        "runtime_evidence_sha256": identity.runtime_evidence_sha256,
        "static_environment_sha256": identity.static_environment_sha256,
        "environment_sha256": identity.environment_sha256,
        "capacity": {
            "effective_host_headroom_bytes": (
                identity.effective_host_headroom_bytes
            ),
            "gpu_total_bytes": identity.gpu_total_bytes,
            "gpu_free_bytes": identity.gpu_free_bytes,
        },
    }


def _capture_package_rows(observation: RuntimeObservation) -> list[dict[str, str]]:
    normalized: dict[str, str] = {}
    for distribution, version in observation.package_versions.items():
        if (
            not isinstance(distribution, str)
            or not distribution
            or not isinstance(version, str)
            or not version
        ):
            raise RuntimePreflightError("captured package metadata is invalid")
        name = _normalize_distribution(distribution)
        if name in normalized:
            raise RuntimePreflightError(
                f"captured package metadata is ambiguous: {distribution}"
            )
        normalized[name] = version
    return [
        {
            "distribution": distribution,
            "version": normalized[distribution],
            "evidence": "observed",
        }
        for distribution in sorted(normalized)
    ]


def _runtime_capture_observation(
    lock: Mapping[str, Any],
    observation: RuntimeObservation,
    *,
    gpu_index: int,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise RuntimePreflightError(
            "the pinned runtime requires exactly one GPU at logical index zero"
        )
    if observation.image_digest is not None and not _is_oci_sha256(
        observation.image_digest
    ):
        raise RuntimePreflightError(
            "trusted runtime image digest assertion is invalid"
        )
    effective_headroom = _effective_host_headroom(observation)
    if (
        not _positive_int(observation.gpu_total_bytes)
        or not _positive_int(observation.gpu_free_bytes)
        or observation.gpu_free_bytes > observation.gpu_total_bytes
    ):
        raise RuntimePreflightError("captured GPU memory accounting is invalid")
    return {
        "image": {
            "lock_tag": lock["image"]["tag"],
            "trusted_launcher_digest_assertion": observation.image_digest,
        },
        "platform": {
            "system": observation.platform_system,
            "machine": observation.machine,
        },
        "python": {
            "implementation": observation.python_implementation,
            "version": observation.python_version,
            "build": list(observation.python_build),
        },
        "packages": _capture_package_rows(observation),
        "host_memory": {
            "mem_available_bytes": observation.mem_available_bytes,
            "cgroup_state": observation.cgroup_state,
            "cgroup_headroom_bytes": observation.cgroup_headroom_bytes,
            "effective_headroom_bytes": effective_headroom,
            "swap_total_bytes": observation.swap_total_bytes,
        },
        "gpu": {
            "logical_index": gpu_index,
            "visible_gpu_count": 1,
            "name": observation.gpu_name,
            "driver_version": observation.nvidia_driver_version,
            "compute_capability": list(observation.gpu_compute_capability),
            "total_bytes": observation.gpu_total_bytes,
            "free_bytes": observation.gpu_free_bytes,
        },
    }


def runtime_capture_report(
    path: Path = DEFAULT_RUNTIME_LOCK_PATH,
    *,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """Return a non-authorizing environment capture without importing Torch."""

    snapshot: RuntimeLockSnapshot | None = None
    base: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cf1-runtime-environment-capture",
        "capture_succeeded": False,
        "authorizes_boot": False,
        "ready": False,
        "runtime_id": CF1_RUNTIME_ID,
        "runtime_lock_sha256": None,
    }
    try:
        snapshot = load_runtime_lock_snapshot(path)
        base["runtime_lock_sha256"] = snapshot.sha256
        if snapshot.sha256 != CF1_RUNTIME_LOCK_SHA256:
            raise RuntimePreflightError("runtime lock digest changed")
        lock = snapshot.parsed()
        validate_runtime_lock(lock, require_frozen=False)
        observation = capture_runtime_environment(
            lock,
            gpu_index=gpu_index,
        )
        payload = _runtime_capture_observation(
            lock,
            observation,
            gpu_index=gpu_index,
        )
    except RuntimePreflightError as error:
        return {**base, "failure": str(error)}
    except Exception as error:
        return {
            **base,
            "failure": f"unexpected capture error: {type(error).__name__}",
        }
    return {
        **base,
        "capture_succeeded": True,
        "lock_status": lock["status"],
        "observation": payload,
        "lock_unresolved": list(lock["unresolved"]),
        "not_observed_without_imports": [
            "loaded_torch_cuda_runtime",
            "active_attention_backend",
            "wheel_hashes",
            "cuda_abi_inventory",
        ],
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
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_RUNTIME_LOCK_PATH,
        help="exact runtime lock (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu-index",
        type=_nonnegative_int,
        default=0,
        help="CUDA device index to inspect (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    report = runtime_preflight_report(
        arguments.lock,
        gpu_index=arguments.gpu_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] is True else 2


def capture_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture CF++1 H100 runtime facts without authorizing a model boot."
        )
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_RUNTIME_LOCK_PATH,
        help="exact candidate runtime lock (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu-index",
        type=_nonnegative_int,
        default=0,
        help="CUDA device index to inspect (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    report = runtime_capture_report(
        arguments.lock,
        gpu_index=arguments.gpu_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["capture_succeeded"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
