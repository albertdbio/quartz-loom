"""Capture and verify byte-level CF++1 runtime evidence without importing Torch.

The main runtime lock is the authorization boundary.  This module only creates
and verifies a separately reviewed evidence lock; its CLI always reports
``authorizes_boot: false``.  Absolute installation paths and PEP 610 local URLs
are reduced to bounded, path-independent identities before they can enter the
evidence document.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit


MAX_RUNTIME_EVIDENCE_BYTES = 64 * 1024
DEFAULT_RUNTIME_EVIDENCE_PATH = (
    Path(__file__).resolve().parent
    / "runtime"
    / "cf1-h100-cu128-v1.evidence.json"
)
_MAX_AUXILIARY_JSON_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_READ_BYTES = 1024 * 1024
_TREE_POLICY = "cf-runtime-tree-v2"
_INSTALLED_POLICY = "cf-installed-record-v2"
_LOADED_LIBRARY_IDENTITY_POLICY = "explicit-runtime-roots-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
_NORMALIZED_DISTRIBUTION_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_WHEEL_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
_MAX_WHEEL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 * 1024
_READELF_PATH = "/usr/bin/readelf"


class RuntimeEvidenceError(ValueError):
    """Runtime evidence is malformed, unsafe, incomplete, or no longer exact."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeEvidenceError(f"duplicate runtime evidence key: {key}")
        value[key] = item
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeEvidenceError("runtime evidence is not canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_oci_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonempty_string(value: object, label: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeEvidenceError(f"{label} is invalid")
    return value


def _normalize_distribution(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if _NORMALIZED_DISTRIBUTION_PATTERN.fullmatch(normalized) is None:
        raise RuntimeEvidenceError("installed distribution name is invalid")
    return normalized


def _exact_fields(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeEvidenceError(f"{label} fields do not match schema")
    return value


def _safe_relative_path(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeEvidenceError(f"{label} is not a safe relative path")
    return text


def _safe_filename(value: object, label: str, *, suffix: str | None = None) -> str:
    text = _nonempty_string(value, label, maximum=512)
    if PurePosixPath(text).name != text or text in {".", ".."}:
        raise RuntimeEvidenceError(f"{label} is not a safe filename")
    if suffix is not None and not text.endswith(suffix):
        raise RuntimeEvidenceError(f"{label} has the wrong suffix")
    return text


def _ordered_unique_strings(
    value: object,
    label: str,
    validator: Any,
) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeEvidenceError(f"{label} list is invalid")
    result = [validator(item) for item in value]
    if len(result) != len(set(result)):
        raise RuntimeEvidenceError(f"{label} entries are duplicated")
    return result


def _validate_loaded_library_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeEvidenceError("loaded library inventory is empty")
    rows: list[Mapping[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for value_row in value:
        row = _exact_fields(
            value_row,
            {"scope", "identity", "size", "sha256"},
            "loaded library",
        )
        scope = row["scope"]
        if scope not in {"environment", "stdlib", "system"}:
            raise RuntimeEvidenceError("loaded library scope is invalid")
        if scope == "system":
            identity = _safe_filename(row["identity"], "system library identity")
        else:
            identity = _safe_relative_path(
                row["identity"], "loaded library relative identity"
            )
        if not _positive_int(row["size"]):
            raise RuntimeEvidenceError("loaded library size is invalid")
        if not _is_sha256(row["sha256"]):
            raise RuntimeEvidenceError("loaded library SHA-256 is invalid")
        rows.append(row)
        identities.append((scope, identity))
    if len(identities) != len(set(identities)):
        raise RuntimeEvidenceError("loaded library identities are ambiguous")
    expected_order = sorted(
        rows,
        key=lambda row: (str(row["scope"]), str(row["identity"]), str(row["sha256"])),
    )
    if rows != expected_order:
        raise RuntimeEvidenceError("loaded library inventory is not sorted")
    return rows


def _manifest_shape(value: object, label: str) -> Mapping[str, Any]:
    manifest = _exact_fields(
        value,
        {"policy", "file_count", "total_bytes", "sha256"},
        label,
    )
    if manifest["policy"] != _TREE_POLICY:
        raise RuntimeEvidenceError(f"{label} policy is invalid")
    if not _nonnegative_int(manifest["file_count"]):
        raise RuntimeEvidenceError(f"{label} file_count is invalid")
    if not _nonnegative_int(manifest["total_bytes"]):
        raise RuntimeEvidenceError(f"{label} total_bytes is invalid")
    if not _is_sha256(manifest["sha256"]):
        raise RuntimeEvidenceError(f"{label} SHA-256 is invalid")
    return manifest


def _validate_oci(value: object) -> Mapping[str, Any]:
    oci = _exact_fields(
        value,
        {"tag", "index_digest", "platform", "manifest_digest", "config_digest"},
        "OCI evidence",
    )
    _nonempty_string(oci["tag"], "OCI tag")
    for field in ("index_digest", "manifest_digest", "config_digest"):
        if not _is_oci_sha256(oci[field]):
            raise RuntimeEvidenceError(f"OCI {field} is invalid")
    platform_value = _exact_fields(
        oci["platform"], {"os", "architecture"}, "OCI platform"
    )
    if platform_value["os"] != "linux" or platform_value["architecture"] != "amd64":
        raise RuntimeEvidenceError("OCI platform is not the pinned linux/amd64 target")
    if oci["index_digest"] == oci["manifest_digest"]:
        raise RuntimeEvidenceError("OCI index and child manifest digests are ambiguous")
    return oci


def _validate_package_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeEvidenceError("runtime evidence package inventory is empty")
    rows: list[Mapping[str, Any]] = []
    names: list[str] = []
    for row in value:
        package = _exact_fields(
            row,
            {
                "distribution",
                "version",
                "wheel",
                "installed_file_count",
                "installed_manifest_sha256",
                "install_metadata_sha256",
            },
            "runtime evidence package",
        )
        distribution = _normalize_distribution(
            _nonempty_string(package["distribution"], "distribution", maximum=256)
        )
        if distribution != package["distribution"]:
            raise RuntimeEvidenceError("distribution name is not normalized")
        _nonempty_string(package["version"], "distribution version", maximum=256)
        wheel = _exact_fields(
            package["wheel"], {"filenames", "sha256"}, "wheel evidence"
        )
        filenames = wheel["filenames"]
        if not isinstance(filenames, list) or not filenames:
            raise RuntimeEvidenceError("wheel filename inventory is empty")
        validated_filenames = [
            _safe_filename(filename, "wheel filename", suffix=".whl")
            for filename in filenames
        ]
        if validated_filenames != sorted(set(validated_filenames)):
            raise RuntimeEvidenceError("wheel filenames are not unique and sorted")
        if not _is_sha256(wheel["sha256"]):
            raise RuntimeEvidenceError("wheel SHA-256 is invalid")
        if not _positive_int(package["installed_file_count"]):
            raise RuntimeEvidenceError("installed file count is invalid")
        for field in ("installed_manifest_sha256", "install_metadata_sha256"):
            if not _is_sha256(package[field]):
                raise RuntimeEvidenceError(f"package {field} is invalid")
        names.append(distribution)
        rows.append(package)
    if names != sorted(set(names)):
        raise RuntimeEvidenceError("package inventory is not unique and sorted")
    return rows


def _validate_native(value: object) -> Mapping[str, Any]:
    native = _exact_fields(
        value,
        {
            "loaded_library_identity_policy",
            "torch_version",
            "cuda_runtime",
            "cxx11_abi",
            "cudnn_version",
            "nccl_version",
            "cuda_arch_list",
            "import_order",
            "critical_modules",
            "loaded_libraries",
            "loaded_libraries_manifest_sha256",
            "loaded_library_count",
        },
        "native evidence",
    )
    if native["loaded_library_identity_policy"] != _LOADED_LIBRARY_IDENTITY_POLICY:
        raise RuntimeEvidenceError("loaded library identity policy is invalid")
    for field in (
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "nccl_version",
    ):
        _nonempty_string(native[field], f"native {field}", maximum=256)
    if not isinstance(native["cxx11_abi"], bool):
        raise RuntimeEvidenceError("native CXX11 ABI value is invalid")
    arches = native["cuda_arch_list"]
    if not isinstance(arches, list) or not arches:
        raise RuntimeEvidenceError("native CUDA architecture list is empty")
    validated_arches = [
        _nonempty_string(arch, "CUDA architecture", maximum=64) for arch in arches
    ]
    if validated_arches != sorted(set(validated_arches)):
        raise RuntimeEvidenceError("CUDA architectures are not unique and sorted")
    import_order = native["import_order"]
    if import_order not in (
        ["torch", "verified-attention-source", "flash-attention-2"],
        ["torch", "verified-attention-source", "flash-attention-3"],
    ):
        raise RuntimeEvidenceError("loaded attention import order is invalid")
    modules = native["critical_modules"]
    if not isinstance(modules, list) or not modules:
        raise RuntimeEvidenceError("critical native module inventory is empty")
    module_names: list[str] = []
    for row in modules:
        module = _exact_fields(
            row,
            {
                "module",
                "origin_scope",
                "relative_path",
                "sha256",
                "elf_class",
                "elf_machine",
                "needed",
                "rpath",
                "runpath",
            },
            "critical native module",
        )
        name = _nonempty_string(module["module"], "critical module name", maximum=256)
        if module["origin_scope"] not in {"environment", "stdlib"}:
            raise RuntimeEvidenceError("critical module origin scope is invalid")
        _safe_relative_path(module["relative_path"], "critical module relative path")
        if not _is_sha256(module["sha256"]):
            raise RuntimeEvidenceError("critical module SHA-256 is invalid")
        _nonempty_string(module["elf_class"], "ELF class", maximum=64)
        _nonempty_string(module["elf_machine"], "ELF machine", maximum=256)
        _ordered_unique_strings(
            module["needed"],
            "ELF needed library",
            lambda item: _safe_filename(item, "ELF needed library"),
        )
        for field in ("rpath", "runpath"):
            _ordered_unique_strings(
                module[field],
                f"ELF {field}",
                lambda item, field=field: _nonempty_string(
                    item, f"ELF {field} entry", maximum=4096
                ),
            )
        module_names.append(name)
    if module_names != sorted(set(module_names)):
        raise RuntimeEvidenceError("critical modules are not unique and sorted")
    loaded_libraries = _validate_loaded_library_rows(native["loaded_libraries"])
    if not _is_sha256(native["loaded_libraries_manifest_sha256"]):
        raise RuntimeEvidenceError("loaded library manifest SHA-256 is invalid")
    if native["loaded_libraries_manifest_sha256"] != _canonical_sha256(
        loaded_libraries
    ):
        raise RuntimeEvidenceError("loaded library manifest SHA-256 does not match rows")
    if (
        not _positive_int(native["loaded_library_count"])
        or native["loaded_library_count"] != len(loaded_libraries)
    ):
        raise RuntimeEvidenceError("loaded library count is invalid")
    return native


def validate_runtime_evidence(value: object) -> None:
    root = _exact_fields(
        value,
        {"schema_version", "runtime_id", "oci", "python", "environment", "native"},
        "runtime evidence",
    )
    if root["schema_version"] != 2 or isinstance(root["schema_version"], bool):
        raise RuntimeEvidenceError("runtime evidence schema version is invalid")
    _nonempty_string(root["runtime_id"], "runtime evidence ID", maximum=256)
    _validate_oci(root["oci"])
    python_value = _exact_fields(
        root["python"],
        {"implementation", "version", "build", "executable_sha256", "stdlib_manifest"},
        "Python evidence",
    )
    _nonempty_string(python_value["implementation"], "Python implementation")
    _nonempty_string(python_value["version"], "Python version", maximum=256)
    build = python_value["build"]
    if (
        not isinstance(build, list)
        or len(build) != 2
        or any(not isinstance(item, str) or not item for item in build)
    ):
        raise RuntimeEvidenceError("Python build evidence is invalid")
    if not _is_sha256(python_value["executable_sha256"]):
        raise RuntimeEvidenceError("Python executable SHA-256 is invalid")
    _manifest_shape(python_value["stdlib_manifest"], "stdlib manifest")
    environment = _exact_fields(
        root["environment"], {"tree_manifest", "packages"}, "environment evidence"
    )
    _manifest_shape(environment["tree_manifest"], "environment tree manifest")
    packages = _validate_package_rows(environment["packages"])
    native = _validate_native(root["native"])
    torch_rows = [
        row for row in packages if row["distribution"] == "torch"
    ]
    if torch_rows and torch_rows[0]["version"] != native["torch_version"]:
        raise RuntimeEvidenceError("native Torch version does not match package inventory")


def _parse_runtime_evidence(encoded: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except UnicodeError as error:
        raise RuntimeEvidenceError("runtime evidence is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise RuntimeEvidenceError("runtime evidence is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeEvidenceError("runtime evidence must be an object")
    validate_runtime_evidence(value)
    return value


@dataclass(frozen=True)
class RuntimeEvidenceSnapshot:
    encoded: bytes
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.encoded, bytes)
            or not self.encoded
            or len(self.encoded) > MAX_RUNTIME_EVIDENCE_BYTES
        ):
            raise RuntimeEvidenceError("runtime evidence size is invalid")
        observed = hashlib.sha256(self.encoded).hexdigest()
        if self.sha256 != observed:
            raise RuntimeEvidenceError("runtime evidence SHA-256 does not match bytes")
        _parse_runtime_evidence(self.encoded)

    def parsed(self) -> dict[str, Any]:
        return _parse_runtime_evidence(self.encoded)


@dataclass(frozen=True)
class RuntimeEvidenceStaticIdentity:
    runtime_id: str
    runtime_evidence_sha256: str
    static_environment_sha256: str
    image_manifest_digest: str
    python_executable_sha256: str
    stdlib_manifest_sha256: str
    environment_tree_sha256: str
    package_inventory_sha256: str


@dataclass(frozen=True)
class RuntimeEvidenceIdentity:
    runtime_id: str
    runtime_evidence_sha256: str
    environment_sha256: str
    image_manifest_digest: str
    python_executable_sha256: str
    stdlib_manifest_sha256: str
    environment_tree_sha256: str
    native_identity_sha256: str


@dataclass(frozen=True)
class RuntimeEvidenceLockedIdentities:
    runtime_environment_sha256: str
    native_identity_sha256: str


def load_runtime_evidence_snapshot(path: Path) -> RuntimeEvidenceSnapshot:
    try:
        size = path.stat().st_size
        if not 0 < size <= MAX_RUNTIME_EVIDENCE_BYTES:
            raise RuntimeEvidenceError("runtime evidence size is invalid")
        encoded = path.read_bytes()
    except OSError as error:
        raise RuntimeEvidenceError("runtime evidence could not be read") from error
    if len(encoded) != size:
        raise RuntimeEvidenceError("runtime evidence changed while being read")
    return RuntimeEvidenceSnapshot(
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def runtime_evidence_locked_identities(
    snapshot: RuntimeEvidenceSnapshot,
) -> RuntimeEvidenceLockedIdentities:
    """Derive the live-native identities committed by exact evidence bytes."""

    locked = snapshot.parsed()
    runtime_environment_sha256 = _canonical_sha256(
        {
            "runtime_evidence_sha256": snapshot.sha256,
            "oci": locked["oci"],
            "python": locked["python"],
            "environment": locked["environment"],
            "native": locked["native"],
        }
    )
    native_identity_sha256 = _canonical_sha256(locked["native"])
    return RuntimeEvidenceLockedIdentities(
        runtime_environment_sha256=runtime_environment_sha256,
        native_identity_sha256=native_identity_sha256,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeEvidenceError(f"{label} is unavailable") from error
    if not resolved.is_dir():
        raise RuntimeEvidenceError(f"{label} is not a directory")
    return resolved


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


def _open_nofollow_regular(path: Path, label: str) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeEvidenceError("no-follow runtime reads are unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
    except OSError as error:
        raise RuntimeEvidenceError(f"{label} could not be opened without links") from error
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise RuntimeEvidenceError(f"{label} is not a regular file")
    return descriptor


def _sha256_open_file(
    descriptor: int,
    label: str,
) -> tuple[str, int, int, tuple[int, ...]]:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeEvidenceError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(_READ_BYTES, before.st_size - offset), offset)
            if not chunk:
                raise RuntimeEvidenceError(f"{label} changed while being hashed")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise RuntimeEvidenceError(f"{label} could not be hashed") from error
    identity_before = _stat_fingerprint(before)
    identity_after = _stat_fingerprint(after)
    if identity_before != identity_after:
        raise RuntimeEvidenceError(f"{label} changed while being hashed")
    return (
        digest.hexdigest(),
        before.st_size,
        stat.S_IMODE(before.st_mode),
        identity_before,
    )


def _sha256_regular_file(path: Path, label: str) -> tuple[str, int, int]:
    descriptor = _open_nofollow_regular(path, label)
    try:
        digest, size, mode, _identity = _sha256_open_file(descriptor, label)
    finally:
        os.close(descriptor)
    return digest, size, mode


def _parse_readelf_output(encoded: str) -> dict[str, object]:
    elf_class: str | None = None
    elf_machine: str | None = None
    needed: list[str] = []
    search_paths: dict[str, list[str]] = {"rpath": [], "runpath": []}
    dynamic_pattern = re.compile(r"\((NEEDED|RPATH|RUNPATH)\).*\[([^\]]*)\]")
    for line in encoded.splitlines():
        stripped = line.strip()
        if stripped.startswith("Class:"):
            elf_class = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Machine:"):
            elf_machine = stripped.split(":", 1)[1].strip()
            continue
        match = dynamic_pattern.search(stripped)
        if match is None:
            continue
        tag, raw = match.groups()
        if tag == "NEEDED":
            needed.append(raw)
            continue
        values = raw.split(":") if raw else []
        if any(not value for value in values):
            raise RuntimeEvidenceError("ELF search path contains an empty entry")
        search_paths[tag.lower()].extend(values)
    if not elf_class or not elf_machine:
        raise RuntimeEvidenceError("critical ELF identity is incomplete")
    needed_value = _ordered_unique_strings(
        needed,
        "ELF needed library",
        lambda item: _safe_filename(item, "ELF needed library"),
    )
    validated_paths: dict[str, list[str]] = {}
    for field in ("rpath", "runpath"):
        validated_paths[field] = _ordered_unique_strings(
            search_paths[field],
            f"ELF {field}",
            lambda item, field=field: _nonempty_string(
                item, f"ELF {field} entry", maximum=4096
            ),
        )
    return {
        "elf_class": _nonempty_string(elf_class, "ELF class", maximum=64),
        "elf_machine": _nonempty_string(elf_machine, "ELF machine", maximum=256),
        "needed": needed_value,
        "rpath": validated_paths["rpath"],
        "runpath": validated_paths["runpath"],
    }


def _readelf_open_file(descriptor: int, identity: tuple[int, ...]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                _READELF_PATH,
                "--wide",
                "--file-header",
                "--dynamic",
                f"/proc/self/fd/{descriptor}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            pass_fds=(descriptor,),
        )
        after = os.fstat(descriptor)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeEvidenceError("critical ELF metadata could not be read") from error
    if _stat_fingerprint(after) != identity:
        raise RuntimeEvidenceError("critical ELF changed while metadata was read")
    return _parse_readelf_output(completed.stdout)


def _observe_elf_file(path: Path, label: str) -> dict[str, object]:
    """Hash and inspect one ELF through the same no-follow descriptor."""

    descriptor = _open_nofollow_regular(path, label)
    try:
        digest, size, mode, identity = _sha256_open_file(descriptor, label)
        metadata = _readelf_open_file(descriptor, identity)
        final_identity = _stat_fingerprint(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if final_identity != identity:
        raise RuntimeEvidenceError(f"{label} changed during ELF observation")
    return {
        "sha256": digest,
        "size": size,
        "mode": mode,
        **metadata,
    }


def _load_json_bytes(encoded: bytes, label: str) -> object:
    if not encoded or len(encoded) > _MAX_AUXILIARY_JSON_BYTES:
        raise RuntimeEvidenceError(f"{label} size is invalid")
    try:
        return json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeEvidenceError(f"{label} is not valid JSON") from error


def _archive_sha256(value: Mapping[str, Any]) -> str:
    archive = value.get("archive_info")
    if not isinstance(archive, Mapping):
        raise RuntimeEvidenceError("direct_url archive_info is invalid")
    hashes = archive.get("hashes")
    digest: object = hashes.get("sha256") if isinstance(hashes, Mapping) else None
    if digest is None:
        legacy = archive.get("hash")
        if isinstance(legacy, str) and legacy.startswith("sha256="):
            digest = legacy.removeprefix("sha256=")
    if not _is_sha256(digest):
        raise RuntimeEvidenceError("direct_url archive SHA-256 is invalid")
    return str(digest)


def _sanitize_direct_url_bytes(encoded: bytes) -> dict[str, str]:
    value = _load_json_bytes(encoded, "direct_url")
    if not isinstance(value, Mapping):
        raise RuntimeEvidenceError("direct_url must be an object")
    url = value.get("url")
    if not isinstance(url, str) or not url:
        raise RuntimeEvidenceError("direct_url URL is invalid")
    parts = urlsplit(url)
    if (
        parts.scheme != "file"
        or parts.netloc not in {"", "localhost"}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise RuntimeEvidenceError("direct_url must be a credential-free local wheel")
    filename = _safe_filename(
        PurePosixPath(unquote(parts.path)).name,
        "direct_url archive filename",
        suffix=".whl",
    )
    return {
        "archive_filename": filename,
        "archive_sha256": _archive_sha256(value),
    }


def _tree_entry(path: Path, root: Path) -> tuple[dict[str, object], int]:
    relative = path.relative_to(root).as_posix()
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeEvidenceError("runtime tree entry could not be observed") from error
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = os.readlink(path)
            resolved_target = path.resolve(strict=True)
            after = path.lstat()
        except OSError as error:
            raise RuntimeEvidenceError("runtime tree symlink could not be observed") from error
        if _stat_fingerprint(metadata) != _stat_fingerprint(after):
            raise RuntimeEvidenceError("runtime tree symlink changed during capture")
        encoded = os.fsencode(target)
        if not _is_within(resolved_target, root):
            try:
                target_metadata = resolved_target.lstat()
            except OSError as error:
                raise RuntimeEvidenceError(
                    "runtime tree symlink target could not be observed"
                ) from error
            if not stat.S_ISREG(target_metadata.st_mode):
                raise RuntimeEvidenceError(
                    "runtime tree symlink directory escapes its root"
                )
            target_digest, target_size, target_mode = _sha256_regular_file(
                resolved_target, "runtime tree external symlink target"
            )
            try:
                final_link = path.lstat()
            except OSError as error:
                raise RuntimeEvidenceError(
                    "runtime tree symlink could not be rechecked"
                ) from error
            if _stat_fingerprint(metadata) != _stat_fingerprint(final_link):
                raise RuntimeEvidenceError(
                    "runtime tree symlink changed during target hashing"
                )
            material = {
                "link_sha256": hashlib.sha256(encoded).hexdigest(),
                "target_sha256": target_digest,
                "target_size": target_size,
                "target_mode": target_mode,
            }
            return (
                {
                    "path": relative,
                    "kind": "external-file-symlink",
                    "mode": mode,
                    "size": len(encoded) + target_size,
                    "sha256": _canonical_sha256(material),
                },
                len(encoded) + target_size,
            )
        return (
            {
                "path": relative,
                "kind": "symlink",
                "mode": mode,
                "size": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            len(encoded),
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeEvidenceError("runtime tree contains a special file")
    if path.name == "direct_url.json":
        encoded = _read_bounded_file(path, "direct_url")
        sanitized = _canonical_bytes(_sanitize_direct_url_bytes(encoded))
        digest = hashlib.sha256(sanitized).hexdigest()
        size = len(sanitized)
    else:
        digest, size, mode = _sha256_regular_file(path, "runtime tree file")
    return (
        {
            "path": relative,
            "kind": "file",
            "mode": mode,
            "size": size,
            "sha256": digest,
        },
        size,
    )


def _tree_manifest(root: Path) -> dict[str, object]:
    resolved = _resolved_directory(root, "runtime tree root")
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(
        resolved, topdown=True, followlinks=False
    ):
        directory_names[:] = sorted(directory_names)
        file_names = sorted(file_names)
        directory_path = Path(directory)
        for name in tuple(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                directory_names.remove(name)
                row, size = _tree_entry(candidate, resolved)
                entries.append(row)
                total_bytes += size
        for name in file_names:
            row, size = _tree_entry(directory_path / name, resolved)
            entries.append(row)
            total_bytes += size
    entries.sort(key=lambda row: str(row["path"]))
    return {
        "policy": _TREE_POLICY,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256": _canonical_sha256(entries),
    }


def _read_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, label: str
) -> bytes:
    if info.file_size <= 0 or info.file_size > _MAX_METADATA_BYTES:
        raise RuntimeEvidenceError(f"wheel {label} size is invalid")
    try:
        encoded = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise RuntimeEvidenceError(f"wheel {label} could not be read") from error
    if len(encoded) != info.file_size:
        raise RuntimeEvidenceError(f"wheel {label} changed while being read")
    return encoded


def _safe_archive_path(value: str, label: str) -> PurePosixPath:
    if not value or "\x00" in value or "\\" in value:
        raise RuntimeEvidenceError(f"{label} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeEvidenceError(f"{label} is invalid")
    return path


def _hash_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    label: str,
) -> tuple[str, int]:
    if info.file_size < 0 or info.file_size > _MAX_WHEEL_MEMBER_BYTES:
        raise RuntimeEvidenceError(f"wheel {label} size is invalid")
    digest = hashlib.sha256()
    observed = 0
    try:
        with archive.open(info, "r") as handle:
            while True:
                block = handle.read(_READ_BYTES)
                if not block:
                    break
                observed += len(block)
                if observed > info.file_size:
                    raise RuntimeEvidenceError(f"wheel {label} expands beyond its size")
                digest.update(block)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise RuntimeEvidenceError(f"wheel {label} could not be read") from error
    if observed != info.file_size:
        raise RuntimeEvidenceError(f"wheel {label} changed while being read")
    return digest.hexdigest(), observed


def _wheel_payload_path(
    path: PurePosixPath,
    dist_info_directory: str,
) -> str | None:
    if path.parts[0] == dist_info_directory:
        return None
    if path.parts[0].endswith((".dist-info", ".egg-info")):
        raise RuntimeEvidenceError("wheel contains foreign distribution metadata")
    if path.parts[0].endswith(".data"):
        if len(path.parts) < 3:
            raise RuntimeEvidenceError("wheel .data payload path is invalid")
        if path.parts[1] not in {"purelib", "platlib"}:
            return None
        path = PurePosixPath(*path.parts[2:])
    return _safe_relative_path(path.as_posix(), "wheel payload path")


def _wheel_inventory(path: Path) -> tuple[str, str, dict[str, dict[str, object]]]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeEvidenceError("wheel contains duplicate archive paths")
            if sum(info.file_size for info in infos) > _MAX_WHEEL_UNCOMPRESSED_BYTES:
                raise RuntimeEvidenceError("wheel uncompressed size is invalid")
            for info in infos:
                archive_path = info.filename[:-1] if info.is_dir() else info.filename
                _safe_archive_path(archive_path, "wheel archive path")
                archive_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(archive_mode):
                    raise RuntimeEvidenceError("wheel contains an unsupported symlink")
            metadata_infos = [
                info
                for info in infos
                if PurePosixPath(info.filename).name == "METADATA"
                and len(PurePosixPath(info.filename).parts) == 2
                and PurePosixPath(info.filename).parts[0].endswith(".dist-info")
            ]
            record_infos = [
                info
                for info in infos
                if PurePosixPath(info.filename).name == "RECORD"
                and len(PurePosixPath(info.filename).parts) == 2
                and PurePosixPath(info.filename).parts[0].endswith(".dist-info")
            ]
            if len(metadata_infos) != 1 or len(record_infos) != 1:
                raise RuntimeEvidenceError("wheel metadata inventory is ambiguous")
            encoded = _read_zip_member(archive, metadata_infos[0], "METADATA")
            record_bytes = _read_zip_member(archive, record_infos[0], "RECORD")
            try:
                record_rows = list(
                    csv.reader(record_bytes.decode("utf-8").splitlines())
                )
            except (UnicodeError, csv.Error) as error:
                raise RuntimeEvidenceError("wheel RECORD is invalid") from error
            if not record_rows:
                raise RuntimeEvidenceError("wheel RECORD is empty")
            info_by_name = {info.filename: info for info in infos if not info.is_dir()}
            record_names: set[str] = set()
            payloads: dict[str, dict[str, object]] = {}
            dist_info_directory = PurePosixPath(metadata_infos[0].filename).parts[0]
            record_name = record_infos[0].filename
            for row in record_rows:
                if len(row) != 3:
                    raise RuntimeEvidenceError("wheel RECORD row is invalid")
                member_name, declared_digest, declared_size = row
                member_path = _safe_archive_path(member_name, "wheel RECORD path")
                if member_name in record_names:
                    raise RuntimeEvidenceError("wheel RECORD path is duplicated")
                record_names.add(member_name)
                info = info_by_name.get(member_name)
                if info is None:
                    raise RuntimeEvidenceError("wheel RECORD member is missing")
                actual_digest, actual_size = _hash_zip_member(
                    archive, info, "RECORD member"
                )
                if member_name == record_name:
                    if declared_digest or declared_size:
                        raise RuntimeEvidenceError("wheel RECORD self-row is not blank")
                else:
                    if _decode_record_digest(declared_digest) != actual_digest:
                        raise RuntimeEvidenceError("wheel RECORD digest does not match payload")
                    if not declared_size.isdigit() or int(declared_size) != actual_size:
                        raise RuntimeEvidenceError("wheel RECORD size does not match payload")
                payload_path = _wheel_payload_path(member_path, dist_info_directory)
                if payload_path is not None:
                    if payload_path in payloads:
                        raise RuntimeEvidenceError("wheel payload install path is ambiguous")
                    payloads[payload_path] = {
                        "sha256": actual_digest,
                        "size": actual_size,
                    }
            if set(info_by_name) != record_names:
                raise RuntimeEvidenceError("wheel RECORD does not own every archive file")
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeEvidenceError("wheel archive is invalid") from error
    message = BytesParser(policy=compat32).parsebytes(encoded)
    name = message.get("Name")
    version = message.get("Version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise RuntimeEvidenceError("wheel Name/Version metadata is missing")
    return (
        _normalize_distribution(name),
        _nonempty_string(version, "wheel version", maximum=256),
        payloads,
    )


def _scan_wheelhouse(
    wheelhouse: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    root = _resolved_directory(wheelhouse, "wheelhouse")
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as error:
        raise RuntimeEvidenceError("wheelhouse could not be enumerated") from error
    if not entries:
        raise RuntimeEvidenceError("wheelhouse is empty")
    for entry in entries:
        filename = _safe_filename(entry.name, "wheel filename", suffix=".whl")
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RuntimeEvidenceError("wheelhouse entry is not a regular wheel")
        path = Path(entry.path)
        distribution, version, payloads = _wheel_inventory(path)
        digest, _, _ = _sha256_regular_file(path, "wheel artifact")
        key = (distribution, version)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "filenames": [filename],
                "sha256": digest,
                "payloads": payloads,
            }
        else:
            if existing["sha256"] != digest:
                raise RuntimeEvidenceError(
                    "wheelhouse has ambiguous bytes for one distribution version"
                )
            if existing["payloads"] != payloads:
                raise RuntimeEvidenceError("wheel payload inventory is ambiguous")
            filenames = existing["filenames"]
            assert isinstance(filenames, list)
            filenames.append(filename)
    return grouped


def _read_bounded_file(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeEvidenceError(f"{label} is not a regular file")
        if not 0 < before.st_size <= _MAX_METADATA_BYTES:
            raise RuntimeEvidenceError(f"{label} size is invalid")
        encoded = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise RuntimeEvidenceError(f"{label} could not be read") from error
    if len(encoded) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeEvidenceError(f"{label} changed while being read")
    return encoded


def _decode_record_digest(value: str) -> str:
    if not value.startswith("sha256="):
        raise RuntimeEvidenceError("installed RECORD digest algorithm is unsupported")
    encoded = value.removeprefix("sha256=")
    try:
        digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, binascii.Error) as error:
        raise RuntimeEvidenceError("installed RECORD digest is invalid") from error
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeEvidenceError("installed RECORD digest is invalid")
    return digest.hex()


def _record_path(
    distribution: importlib.metadata.Distribution,
    value: str,
    environment_root: Path,
    *,
    allow_missing: bool = False,
) -> tuple[Path, str, bool]:
    if not value or "\x00" in value or "\\" in value:
        raise RuntimeEvidenceError("installed RECORD path is invalid")
    posix = PurePosixPath(value)
    if posix.is_absolute() or posix.as_posix() != value:
        raise RuntimeEvidenceError("installed RECORD path is invalid")
    candidate = Path(distribution.locate_file(value))
    exists = True
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        if not allow_missing:
            raise RuntimeEvidenceError("installed RECORD path is missing") from error
        ancestor = candidate
        while not ancestor.exists():
            if ancestor.is_symlink() or ancestor.parent == ancestor:
                raise RuntimeEvidenceError("installed RECORD path is missing") from error
            ancestor = ancestor.parent
        try:
            if not ancestor.is_dir():
                raise RuntimeEvidenceError("installed RECORD path is missing")
            ancestor.resolve(strict=True)
            path = candidate.resolve(strict=False)
        except OSError as missing_error:
            raise RuntimeEvidenceError(
                "installed RECORD path is missing"
            ) from missing_error
        exists = False
    if not _is_within(path, environment_root):
        raise RuntimeEvidenceError("installed RECORD path escapes the environment")
    return path, path.relative_to(environment_root).as_posix(), exists


def _is_distribution_metadata_path(path: PurePosixPath) -> bool:
    return bool(path.parts) and path.parts[0].endswith((".dist-info", ".egg-info"))


def _is_generated_bytecode(path: PurePosixPath) -> bool:
    return path.suffix in {".pyc", ".pyo"}


def _is_importable_payload(path: PurePosixPath) -> bool:
    if not path.parts or ".." in path.parts or _is_distribution_metadata_path(path):
        return False
    name = path.name.lower()
    return (
        name.endswith(
            (
                ".py",
                ".pyc",
                ".pyo",
                ".pyi",
                ".pth",
                ".egg-link",
                ".so",
                ".pyd",
                ".dll",
                ".dylib",
                ".zip",
            )
        )
        or ".so." in name
    )


def _installed_package_row(
    distribution: importlib.metadata.Distribution,
    *,
    environment_root: Path,
    wheel: Mapping[str, object],
) -> tuple[dict[str, object], set[str]]:
    dist_path_value = getattr(distribution, "_path", None)
    if not isinstance(dist_path_value, Path):
        dist_path_value = Path(dist_path_value) if dist_path_value is not None else None
    if not isinstance(dist_path_value, Path):
        raise RuntimeEvidenceError("installed distribution metadata path is unavailable")
    try:
        dist_info = dist_path_value.resolve(strict=True)
    except OSError as error:
        raise RuntimeEvidenceError("installed distribution metadata is unavailable") from error
    if (
        not dist_info.name.endswith(".dist-info")
        or not dist_info.is_dir()
        or not _is_within(dist_info, environment_root)
    ):
        raise RuntimeEvidenceError("installed distribution metadata path is invalid")
    record_path = dist_info / "RECORD"
    record_bytes = _read_bounded_file(record_path, "installed RECORD")
    try:
        record_text = record_bytes.decode("utf-8")
        rows = list(csv.reader(record_text.splitlines()))
    except (UnicodeError, csv.Error) as error:
        raise RuntimeEvidenceError("installed RECORD is invalid") from error
    if not rows:
        raise RuntimeEvidenceError("installed RECORD is empty")
    manifest: list[dict[str, object]] = []
    seen: set[str] = set()
    record_relative = record_path.resolve().relative_to(environment_root).as_posix()
    direct_url: dict[str, str] | None = None
    owned_importable: set[str] = set()
    installed_payloads: set[str] = set()
    wheel_payloads = wheel.get("payloads")
    if not isinstance(wheel_payloads, Mapping):
        raise RuntimeEvidenceError("wheel payload inventory is unavailable")
    for row in rows:
        if len(row) != 3:
            raise RuntimeEvidenceError("installed RECORD row is invalid")
        path_text, declared_digest, declared_size = row
        install_path = PurePosixPath(path_text)
        optional_missing_bytecode = (
            _is_generated_bytecode(install_path)
            and not declared_digest
            and not declared_size
        )
        path, relative, path_exists = _record_path(
            distribution,
            path_text,
            environment_root,
            allow_missing=optional_missing_bytecode,
        )
        if relative in seen:
            raise RuntimeEvidenceError("installed RECORD path is duplicated")
        seen.add(relative)
        if not path_exists:
            continue
        actual_digest, actual_size, mode = _sha256_regular_file(
            path, "installed RECORD file"
        )
        if declared_digest:
            if _decode_record_digest(declared_digest) != actual_digest:
                raise RuntimeEvidenceError("installed RECORD digest does not match file")
        elif relative != record_relative and not _is_generated_bytecode(
            PurePosixPath(path_text)
        ):
            raise RuntimeEvidenceError("installed RECORD omits a required file digest")
        if declared_size:
            if not declared_size.isdigit() or int(declared_size) != actual_size:
                raise RuntimeEvidenceError("installed RECORD size does not match file")
        elif relative != record_relative and not _is_generated_bytecode(
            PurePosixPath(path_text)
        ):
            raise RuntimeEvidenceError("installed RECORD omits a required file size")
        if path.name == "direct_url.json":
            direct_url = _sanitize_direct_url_bytes(
                _read_bounded_file(path, "direct_url")
            )
            canonical = _canonical_bytes(direct_url)
            actual_digest = hashlib.sha256(canonical).hexdigest()
            actual_size = len(canonical)
        if (
            not install_path.is_absolute()
            and ".." not in install_path.parts
            and not _is_distribution_metadata_path(install_path)
        ):
            expected_payload = wheel_payloads.get(install_path.as_posix())
            if _is_generated_bytecode(install_path) and expected_payload is None:
                pass
            else:
                if not isinstance(expected_payload, Mapping):
                    raise RuntimeEvidenceError(
                        "installed payload is absent from reviewed wheel payload"
                    )
                if (
                    expected_payload.get("sha256") != actual_digest
                    or expected_payload.get("size") != actual_size
                ):
                    raise RuntimeEvidenceError(
                        "installed payload does not match reviewed wheel payload"
                    )
                installed_payloads.add(install_path.as_posix())
        if _is_importable_payload(install_path):
            owned_importable.add(relative)
        manifest.append(
            {
                "path": relative,
                "mode": mode,
                "size": actual_size,
                "sha256": actual_digest,
            }
        )
    required = {
        (dist_info / name).resolve().relative_to(environment_root).as_posix()
        for name in ("METADATA", "WHEEL", "INSTALLER", "RECORD")
    }
    if not required.issubset(seen):
        raise RuntimeEvidenceError("installed RECORD omits required metadata")
    wheel_filenames = wheel["filenames"]
    wheel_sha256 = wheel["sha256"]
    assert isinstance(wheel_filenames, list) and isinstance(wheel_sha256, str)
    if direct_url is not None and (
        direct_url["archive_filename"] not in wheel_filenames
        or direct_url["archive_sha256"] != wheel_sha256
    ):
        raise RuntimeEvidenceError("direct_url does not match the reviewed wheel")
    wheel_bytes = _read_bounded_file(dist_info / "WHEEL", "installed WHEEL")
    installer_bytes = _read_bounded_file(
        dist_info / "INSTALLER", "installed INSTALLER"
    )
    metadata_identity = {
        "wheel_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "installer_sha256": hashlib.sha256(installer_bytes).hexdigest(),
        "direct_url": direct_url,
    }
    manifest.sort(key=lambda item: str(item["path"]))
    if installed_payloads != set(wheel_payloads):
        raise RuntimeEvidenceError(
            "installed payload inventory does not match reviewed wheel payload"
        )
    name = _normalize_distribution(
        _nonempty_string(distribution.metadata.get("Name"), "installed distribution")
    )
    version = _nonempty_string(
        distribution.version, "installed distribution version", maximum=256
    )
    return (
        {
            "distribution": name,
            "version": version,
            "wheel": {
                "filenames": sorted(str(item) for item in wheel_filenames),
                "sha256": wheel_sha256,
            },
            "installed_file_count": len(manifest),
            "installed_manifest_sha256": _canonical_sha256(
                {"policy": _INSTALLED_POLICY, "files": manifest}
            ),
            "install_metadata_sha256": _canonical_sha256(metadata_identity),
        },
        owned_importable,
    )


def _capture_packages(
    *,
    environment_root: Path,
    distribution_paths: Sequence[Path],
    wheelhouse: Path,
) -> list[dict[str, object]]:
    resolved_paths: list[Path] = []
    for path in distribution_paths:
        resolved = _resolved_directory(path, "distribution path")
        if not _is_within(resolved, environment_root):
            raise RuntimeEvidenceError("distribution path escapes the environment")
        if resolved not in resolved_paths:
            resolved_paths.append(resolved)
    if not resolved_paths:
        raise RuntimeEvidenceError("distribution paths are empty")
    wheels = _scan_wheelhouse(wheelhouse)
    installed: dict[tuple[str, str], importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(
        path=[str(path) for path in resolved_paths]
    ):
        name_value = distribution.metadata.get("Name")
        version_value = distribution.version
        if not isinstance(name_value, str) or not isinstance(version_value, str):
            raise RuntimeEvidenceError("installed distribution metadata is invalid")
        key = (_normalize_distribution(name_value), version_value)
        if key in installed:
            raise RuntimeEvidenceError("installed distribution inventory is ambiguous")
        installed[key] = distribution
    if set(installed) != set(wheels):
        raise RuntimeEvidenceError(
            "installed distribution inventory does not match wheelhouse"
        )
    rows: list[dict[str, object]] = []
    owned_importable: set[str] = set()
    for key in sorted(installed):
        row, owned = _installed_package_row(
            installed[key], environment_root=environment_root, wheel=wheels[key]
        )
        rows.append(row)
        owned_importable.update(owned)
    observed_importable: set[str] = set()
    for distribution_root in resolved_paths:
        for directory, directory_names, file_names in os.walk(
            distribution_root, topdown=True, followlinks=False
        ):
            directory_names[:] = sorted(directory_names)
            directory_path = Path(directory)
            for name in tuple(directory_names):
                candidate = directory_path / name
                if candidate.is_symlink():
                    directory_names.remove(name)
                    raise RuntimeEvidenceError(
                        "installed importable package directory is a symlink"
                    )
            for name in sorted(file_names):
                candidate = directory_path / name
                relative_to_distribution = candidate.relative_to(distribution_root)
                if not _is_importable_payload(
                    PurePosixPath(relative_to_distribution.as_posix())
                ):
                    continue
                try:
                    metadata = candidate.lstat()
                except OSError as error:
                    raise RuntimeEvidenceError(
                        "installed importable payload could not be observed"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeEvidenceError(
                        "installed importable payload is a symlink"
                    )
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeEvidenceError(
                        "installed importable payload is not a regular file"
                    )
                observed_importable.add(
                    candidate.relative_to(environment_root).as_posix()
                )
    unowned = observed_importable - owned_importable
    if unowned:
        raise RuntimeEvidenceError("runtime contains an unowned importable payload")
    return rows


def _capture_native(
    loaded_facts: Mapping[str, Any],
    *,
    environment_root: Path,
    stdlib_root: Path,
) -> dict[str, object]:
    facts = _exact_fields(
        loaded_facts,
        {
            "loaded_library_identity_policy",
            "torch_version",
            "cuda_runtime",
            "cxx11_abi",
            "cudnn_version",
            "nccl_version",
            "cuda_arch_list",
            "import_order",
            "critical_modules",
            "loaded_libraries",
            "loaded_libraries_manifest_sha256",
            "loaded_library_count",
        },
        "loaded native facts",
    )
    if facts["loaded_library_identity_policy"] != _LOADED_LIBRARY_IDENTITY_POLICY:
        raise RuntimeEvidenceError("loaded library identity policy is invalid")
    for field in (
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "nccl_version",
    ):
        _nonempty_string(facts[field], f"loaded native {field}", maximum=256)
    if not isinstance(facts["cxx11_abi"], bool):
        raise RuntimeEvidenceError("loaded native CXX11 ABI value is invalid")
    arches_value = facts["cuda_arch_list"]
    if not isinstance(arches_value, list) or not arches_value:
        raise RuntimeEvidenceError("loaded CUDA architectures are invalid")
    arches = sorted(
        {
            _nonempty_string(item, "loaded CUDA architecture", maximum=64)
            for item in arches_value
        }
    )
    import_order = facts["import_order"]
    if import_order not in (
        ["torch", "verified-attention-source", "flash-attention-2"],
        ["torch", "verified-attention-source", "flash-attention-3"],
    ):
        raise RuntimeEvidenceError("loaded attention import order is invalid")
    module_values = facts["critical_modules"]
    if not isinstance(module_values, list) or not module_values:
        raise RuntimeEvidenceError("loaded critical modules are empty")
    modules: list[dict[str, object]] = []
    for item in module_values:
        module = _exact_fields(
            item,
            {
                "module",
                "path",
                "elf_class",
                "elf_machine",
                "needed",
                "rpath",
                "runpath",
            },
            "loaded critical module",
        )
        name = _nonempty_string(module["module"], "loaded module name", maximum=256)
        try:
            path = Path(_nonempty_string(module["path"], "loaded module path")).resolve(
                strict=True
            )
        except OSError as error:
            raise RuntimeEvidenceError("loaded critical module is unavailable") from error
        if _is_within(path, environment_root):
            scope = "environment"
            relative = path.relative_to(environment_root).as_posix()
        elif _is_within(path, stdlib_root):
            scope = "stdlib"
            relative = path.relative_to(stdlib_root).as_posix()
        else:
            raise RuntimeEvidenceError(
                "loaded critical module is outside reviewed runtime roots"
            )
        _safe_relative_path(relative, "loaded module relative path")
        needed = _ordered_unique_strings(
            module["needed"],
            "loaded ELF needed library",
            lambda value: _safe_filename(value, "ELF needed library"),
        )
        search_paths: dict[str, list[str]] = {}
        for field in ("rpath", "runpath"):
            search_paths[field] = _ordered_unique_strings(
                module[field],
                f"loaded ELF {field}",
                lambda value, field=field: _nonempty_string(
                    value,
                    f"loaded ELF {field} entry",
                    maximum=4096,
                ),
            )
        supplied_metadata = {
            "elf_class": _nonempty_string(
                module["elf_class"], "loaded ELF class", maximum=64
            ),
            "elf_machine": _nonempty_string(
                module["elf_machine"], "loaded ELF machine", maximum=256
            ),
            "needed": needed,
            "rpath": search_paths["rpath"],
            "runpath": search_paths["runpath"],
        }
        observed_file = _observe_elf_file(path, "loaded critical module")
        observed_metadata = {
            field: observed_file[field]
            for field in ("elf_class", "elf_machine", "needed", "rpath", "runpath")
        }
        if observed_metadata != supplied_metadata:
            raise RuntimeEvidenceError(
                "loaded critical ELF facts do not match independent readelf observation"
            )
        modules.append(
            {
                "module": name,
                "origin_scope": scope,
                "relative_path": relative,
                "sha256": observed_file["sha256"],
                **observed_metadata,
            }
        )
    modules.sort(key=lambda item: str(item["module"]))
    if len({str(item["module"]) for item in modules}) != len(modules):
        raise RuntimeEvidenceError("loaded critical module inventory is ambiguous")
    loaded_libraries = _validate_loaded_library_rows(facts["loaded_libraries"])
    if not _is_sha256(facts["loaded_libraries_manifest_sha256"]):
        raise RuntimeEvidenceError("loaded library manifest SHA-256 is invalid")
    if facts["loaded_libraries_manifest_sha256"] != _canonical_sha256(
        loaded_libraries
    ):
        raise RuntimeEvidenceError("loaded library manifest does not match rows")
    if (
        not _positive_int(facts["loaded_library_count"])
        or facts["loaded_library_count"] != len(loaded_libraries)
    ):
        raise RuntimeEvidenceError("loaded library count is invalid")
    return {
        "loaded_library_identity_policy": _LOADED_LIBRARY_IDENTITY_POLICY,
        "torch_version": facts["torch_version"],
        "cuda_runtime": facts["cuda_runtime"],
        "cxx11_abi": facts["cxx11_abi"],
        "cudnn_version": facts["cudnn_version"],
        "nccl_version": facts["nccl_version"],
        "cuda_arch_list": arches,
        "import_order": list(import_order),
        "critical_modules": modules,
        "loaded_libraries": [dict(row) for row in loaded_libraries],
        "loaded_libraries_manifest_sha256": facts[
            "loaded_libraries_manifest_sha256"
        ],
        "loaded_library_count": facts["loaded_library_count"],
    }


def _verify_locked_native_files(
    native: Mapping[str, Any],
    *,
    environment_root: Path,
    stdlib_root: Path,
) -> None:
    for module in native["critical_modules"]:
        root = (
            environment_root
            if module["origin_scope"] == "environment"
            else stdlib_root
        )
        try:
            path = (root / module["relative_path"]).resolve(strict=True)
        except OSError as error:
            raise RuntimeEvidenceError(
                "locked critical native module is unavailable"
            ) from error
        if not _is_within(path, root):
            raise RuntimeEvidenceError(
                "locked critical native module escapes its runtime root"
            )
        observed = _observe_elf_file(path, "locked critical native module")
        locked_file = {
            "sha256": module["sha256"],
            "elf_class": module["elf_class"],
            "elf_machine": module["elf_machine"],
            "needed": module["needed"],
            "rpath": module["rpath"],
            "runpath": module["runpath"],
        }
        observed_file = {
            field: observed[field]
            for field in locked_file
        }
        if observed_file != locked_file:
            raise RuntimeEvidenceError(
                "locked critical native module does not match file or ELF metadata"
            )
    for library in native["loaded_libraries"]:
        if library["scope"] == "system":
            continue
        root = environment_root if library["scope"] == "environment" else stdlib_root
        try:
            path = (root / library["identity"]).resolve(strict=True)
        except OSError as error:
            raise RuntimeEvidenceError("locked loaded library is unavailable") from error
        if not _is_within(path, root):
            raise RuntimeEvidenceError("locked loaded library escapes its runtime root")
        digest, size, _mode = _sha256_regular_file(path, "locked loaded library")
        if digest != library["sha256"] or size != library["size"]:
            raise RuntimeEvidenceError("locked loaded library does not match file")


def _capture_static_runtime_evidence(
    *,
    runtime_id: str,
    oci: Mapping[str, Any],
    python_executable: Path,
    python_implementation: str,
    python_version: str,
    python_build: Sequence[str],
    stdlib_root: Path,
    environment_root: Path,
    distribution_paths: Sequence[Path],
    wheelhouse: Path,
) -> dict[str, object]:
    runtime_id_value = _nonempty_string(runtime_id, "runtime evidence ID")
    oci_value = dict(_validate_oci(oci))
    if (
        not isinstance(python_build, Sequence)
        or isinstance(python_build, (str, bytes))
        or len(python_build) != 2
    ):
        raise RuntimeEvidenceError("Python build evidence is invalid")
    build = [
        _nonempty_string(item, "Python build value", maximum=4096)
        for item in python_build
    ]
    try:
        executable = Path(python_executable).resolve(strict=True)
    except OSError as error:
        raise RuntimeEvidenceError("Python executable is unavailable") from error
    executable_digest, _, _ = _sha256_regular_file(
        executable, "Python executable"
    )
    environment = _resolved_directory(environment_root, "runtime environment")
    stdlib = _resolved_directory(stdlib_root, "Python stdlib")
    packages = _capture_packages(
        environment_root=environment,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    return {
        "schema_version": 2,
        "runtime_id": runtime_id_value,
        "oci": oci_value,
        "python": {
            "implementation": _nonempty_string(
                python_implementation, "Python implementation"
            ),
            "version": _nonempty_string(
                python_version, "Python version", maximum=256
            ),
            "build": build,
            "executable_sha256": executable_digest,
            "stdlib_manifest": _tree_manifest(stdlib),
        },
        "environment": {
            "tree_manifest": _tree_manifest(environment),
            "packages": packages,
        },
    }


def capture_runtime_evidence(
    *,
    runtime_id: str,
    oci: Mapping[str, Any],
    python_executable: Path,
    python_implementation: str,
    python_version: str,
    python_build: Sequence[str],
    stdlib_root: Path,
    environment_root: Path,
    distribution_paths: Sequence[Path],
    wheelhouse: Path,
    loaded_facts: Mapping[str, Any],
) -> dict[str, object]:
    """Capture a deterministic, path-redacted evidence document.

    ``loaded_facts`` is produced by a separate loaded-runtime probe.  This
    stdlib-only function verifies and hashes its critical module paths but never
    imports Torch or any model code itself.
    """

    evidence = _capture_static_runtime_evidence(
        runtime_id=runtime_id,
        oci=oci,
        python_executable=python_executable,
        python_implementation=python_implementation,
        python_version=python_version,
        python_build=python_build,
        stdlib_root=stdlib_root,
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    evidence["native"] = _capture_native(
        loaded_facts,
        environment_root=_resolved_directory(
            environment_root, "runtime environment"
        ),
        stdlib_root=_resolved_directory(stdlib_root, "Python stdlib"),
    )
    _verify_locked_native_files(
        evidence["native"],
        environment_root=_resolved_directory(
            environment_root, "runtime environment"
        ),
        stdlib_root=_resolved_directory(stdlib_root, "Python stdlib"),
    )
    post_native_static = _capture_static_runtime_evidence(
        runtime_id=runtime_id,
        oci=oci,
        python_executable=python_executable,
        python_implementation=python_implementation,
        python_version=python_version,
        python_build=python_build,
        stdlib_root=stdlib_root,
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    if post_native_static != {
        field: evidence[field]
        for field in ("schema_version", "runtime_id", "oci", "python", "environment")
    }:
        raise RuntimeEvidenceError("static runtime changed during native capture")
    validate_runtime_evidence(evidence)
    if len(_canonical_bytes(evidence)) > MAX_RUNTIME_EVIDENCE_BYTES:
        raise RuntimeEvidenceError("runtime evidence size is invalid")
    return evidence


def verify_static_runtime_evidence(
    snapshot: RuntimeEvidenceSnapshot,
    *,
    observed_oci: Mapping[str, Any],
    python_executable: Path,
    python_implementation: str,
    python_version: str,
    python_build: Sequence[str],
    stdlib_root: Path,
    environment_root: Path,
    distribution_paths: Sequence[Path],
    wheelhouse: Path,
) -> RuntimeEvidenceStaticIdentity:
    """Verify every non-loaded byte before GPU observation or Torch import.

    Native ABI values remain reviewed lock facts at this phase.  Their module
    bytes are already covered by the environment tree; a later loaded probe
    must call :func:`verify_runtime_evidence` to check the actual loaded facts.
    """

    if hashlib.sha256(snapshot.encoded).hexdigest() != snapshot.sha256:
        raise RuntimeEvidenceError("runtime evidence snapshot digest is invalid")
    locked = snapshot.parsed()
    environment = _resolved_directory(environment_root, "runtime environment")
    stdlib = _resolved_directory(stdlib_root, "Python stdlib")
    _verify_locked_native_files(
        locked["native"],
        environment_root=environment,
        stdlib_root=stdlib,
    )
    observed = _capture_static_runtime_evidence(
        runtime_id=locked["runtime_id"],
        oci=observed_oci,
        python_executable=python_executable,
        python_implementation=python_implementation,
        python_version=python_version,
        python_build=python_build,
        stdlib_root=stdlib_root,
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    locked_static = {
        field: locked[field]
        for field in ("schema_version", "runtime_id", "oci", "python", "environment")
    }
    if observed != locked_static:
        raise RuntimeEvidenceError("static runtime evidence does not match lock")
    return RuntimeEvidenceStaticIdentity(
        runtime_id=locked["runtime_id"],
        runtime_evidence_sha256=snapshot.sha256,
        static_environment_sha256=_canonical_sha256(
            {
                "runtime_evidence_sha256": snapshot.sha256,
                **locked_static,
            }
        ),
        image_manifest_digest=locked["oci"]["manifest_digest"],
        python_executable_sha256=locked["python"]["executable_sha256"],
        stdlib_manifest_sha256=locked["python"]["stdlib_manifest"]["sha256"],
        environment_tree_sha256=locked["environment"]["tree_manifest"]["sha256"],
        package_inventory_sha256=_canonical_sha256(
            locked["environment"]["packages"]
        ),
    )


def verify_runtime_evidence(
    snapshot: RuntimeEvidenceSnapshot,
    *,
    observed_oci: Mapping[str, Any],
    python_executable: Path,
    python_implementation: str,
    python_version: str,
    python_build: Sequence[str],
    stdlib_root: Path,
    environment_root: Path,
    distribution_paths: Sequence[Path],
    wheelhouse: Path,
    loaded_facts: Mapping[str, Any],
) -> RuntimeEvidenceIdentity:
    initial_static = verify_static_runtime_evidence(
        snapshot,
        observed_oci=observed_oci,
        python_executable=python_executable,
        python_implementation=python_implementation,
        python_version=python_version,
        python_build=python_build,
        stdlib_root=stdlib_root,
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    locked = snapshot.parsed()
    observed_native = _capture_native(
        loaded_facts,
        environment_root=_resolved_directory(
            environment_root, "runtime environment"
        ),
        stdlib_root=_resolved_directory(stdlib_root, "Python stdlib"),
    )
    if observed_native != locked["native"]:
        raise RuntimeEvidenceError("loaded runtime evidence does not match lock")
    post_native_static = verify_static_runtime_evidence(
        snapshot,
        observed_oci=observed_oci,
        python_executable=python_executable,
        python_implementation=python_implementation,
        python_version=python_version,
        python_build=python_build,
        stdlib_root=stdlib_root,
        environment_root=environment_root,
        distribution_paths=distribution_paths,
        wheelhouse=wheelhouse,
    )
    if post_native_static != initial_static:
        raise RuntimeEvidenceError("static runtime changed during native verification")
    locked_identities = runtime_evidence_locked_identities(snapshot)
    return RuntimeEvidenceIdentity(
        runtime_id=locked["runtime_id"],
        runtime_evidence_sha256=snapshot.sha256,
        environment_sha256=locked_identities.runtime_environment_sha256,
        image_manifest_digest=locked["oci"]["manifest_digest"],
        python_executable_sha256=locked["python"]["executable_sha256"],
        stdlib_manifest_sha256=locked["python"]["stdlib_manifest"]["sha256"],
        environment_tree_sha256=locked["environment"]["tree_manifest"]["sha256"],
        native_identity_sha256=locked_identities.native_identity_sha256,
    )


def runtime_evidence_capture_report(**arguments: Any) -> dict[str, object]:
    """Capture evidence while making the lack of boot authority explicit."""

    base: dict[str, object] = {
        "schema_version": 1,
        "kind": "cf1-runtime-evidence-capture",
        "capture_succeeded": False,
        "authorizes_boot": False,
        "ready": False,
        "runtime_evidence_sha256": None,
    }
    try:
        evidence = capture_runtime_evidence(**arguments)
        encoded = _canonical_bytes(evidence)
    except RuntimeEvidenceError as error:
        return {**base, "failure": str(error)}
    except Exception as error:
        return {
            **base,
            "failure": f"unexpected evidence capture error: {type(error).__name__}",
        }
    return {
        **base,
        "capture_succeeded": True,
        "runtime_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "evidence": evidence,
    }


def _validate_bootstrap_probe(value: object, *, runtime_id: str) -> Mapping[str, Any]:
    probe = _exact_fields(
        value,
        {
            "schema_version",
            "kind",
            "probe_succeeded",
            "authorizes_boot",
            "ready",
            "probe_mode",
            "gpu_execution_performed",
            "runtime_id",
            "runtime_lock_sha256",
            "runtime_evidence_sha256",
            "asset_lock_sha256",
            "source_commit",
            "attention_source_sha256",
            "loaded_facts",
            "loaded_facts_sha256",
            "static_environment_sha256",
            "runtime_environment_sha256",
            "native_identity_sha256",
            "attention_backend",
            "executed_callable",
            "call_count",
            "input_shape",
            "output_shape",
            "output_dtype",
            "output_finite",
            "probe_identity_sha256",
        },
        "bootstrap attention probe",
    )
    if (
        probe["schema_version"] != 2
        or probe["kind"] != "cf1-executed-attention-probe"
        or probe["probe_succeeded"] is not True
        or probe["authorizes_boot"] is not False
        or probe["ready"] is not False
        or probe["probe_mode"] != "unbound-evidence-capture"
        or probe["gpu_execution_performed"] is not True
        or probe["runtime_id"] != runtime_id
        or probe["runtime_evidence_sha256"] is not None
        or probe["static_environment_sha256"] is not None
        or probe["runtime_environment_sha256"] is not None
        or probe["native_identity_sha256"] is not None
    ):
        raise RuntimeEvidenceError("bootstrap attention probe is not a complete success")
    for field in (
        "runtime_lock_sha256",
        "asset_lock_sha256",
        "attention_source_sha256",
        "loaded_facts_sha256",
        "probe_identity_sha256",
    ):
        if not _is_sha256(probe[field]):
            raise RuntimeEvidenceError(f"bootstrap probe {field} is invalid")
    source_commit = probe["source_commit"]
    if not isinstance(source_commit, str) or _GIT_SHA1_PATTERN.fullmatch(source_commit) is None:
        raise RuntimeEvidenceError("bootstrap probe source commit is invalid")
    backend = probe["attention_backend"]
    expected_callable = {
        "flash-attention-2": "flash_attn.flash_attn_varlen_func",
        "flash-attention-3": "flash_attn_interface.flash_attn_varlen_func",
    }.get(backend)
    if (
        expected_callable is None
        or probe["executed_callable"] != expected_callable
        or probe["call_count"] != 1
        or isinstance(probe["call_count"], bool)
        or probe["input_shape"] != [1, 16, 12, 128]
        or probe["output_shape"] != [1, 16, 12, 128]
        or probe["output_dtype"] != "torch.bfloat16"
        or probe["output_finite"] is not True
    ):
        raise RuntimeEvidenceError("bootstrap attention execution contract is invalid")
    loaded_facts = probe["loaded_facts"]
    facts = _exact_fields(
        loaded_facts,
        {
            "loaded_library_identity_policy",
            "torch_version",
            "cuda_runtime",
            "cxx11_abi",
            "cudnn_version",
            "nccl_version",
            "cuda_arch_list",
            "import_order",
            "critical_modules",
            "loaded_libraries",
            "loaded_libraries_manifest_sha256",
            "loaded_library_count",
        },
        "bootstrap loaded facts",
    )
    if facts["loaded_library_identity_policy"] != _LOADED_LIBRARY_IDENTITY_POLICY:
        raise RuntimeEvidenceError("bootstrap loaded library policy is invalid")
    if facts["import_order"] != ["torch", "verified-attention-source", backend]:
        raise RuntimeEvidenceError("bootstrap loaded import order is invalid")
    libraries = _validate_loaded_library_rows(facts["loaded_libraries"])
    if (
        facts["loaded_library_count"] != len(libraries)
        or facts["loaded_libraries_manifest_sha256"] != _canonical_sha256(libraries)
    ):
        raise RuntimeEvidenceError("bootstrap loaded library identity is invalid")
    loaded_sha = _canonical_sha256(loaded_facts)
    if probe["loaded_facts_sha256"] != loaded_sha:
        raise RuntimeEvidenceError("bootstrap loaded facts SHA-256 does not match")
    execution = {
        field: probe[field]
        for field in (
            "gpu_execution_performed",
            "attention_backend",
            "executed_callable",
            "call_count",
            "input_shape",
            "output_shape",
            "output_dtype",
            "output_finite",
        )
    }
    identity_material = {
        "schema_version": 3,
        "probe_mode": "unbound-evidence-capture",
        "runtime_lock_sha256": probe["runtime_lock_sha256"],
        "runtime_evidence_sha256": None,
        "static_environment_sha256": None,
        "runtime_environment_sha256": None,
        "native_identity_sha256": None,
        "asset_lock_sha256": probe["asset_lock_sha256"],
        "source_commit": source_commit,
        "attention_source_sha256": probe["attention_source_sha256"],
        "loaded_facts_sha256": loaded_sha,
        **execution,
    }
    if probe["probe_identity_sha256"] != _canonical_sha256(identity_material):
        raise RuntimeEvidenceError("bootstrap probe identity SHA-256 does not match")
    return probe


def _load_bootstrap_probe(path: Path, *, runtime_id: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if not 0 < size <= _MAX_AUXILIARY_JSON_BYTES:
            raise RuntimeEvidenceError("bootstrap probe size is invalid")
        encoded = path.read_bytes()
    except OSError as error:
        raise RuntimeEvidenceError("bootstrap probe could not be read") from error
    if len(encoded) != size:
        raise RuntimeEvidenceError("bootstrap probe changed while being read")
    value = _load_json_bytes(encoded, "bootstrap probe")
    return _validate_bootstrap_probe(value, runtime_id=runtime_id)


def _atomic_write_evidence(path: Path, encoded: bytes) -> None:
    if not encoded or len(encoded) > MAX_RUNTIME_EVIDENCE_BYTES:
        raise RuntimeEvidenceError("runtime evidence size is invalid")
    name = _safe_filename(path.name, "runtime evidence output filename")
    parent = _resolved_directory(path.parent, "runtime evidence output directory")
    target = parent / name
    try:
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeEvidenceError(
                    "runtime evidence output target is not a regular file"
                )
        temporary = parent / f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(encoded)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise RuntimeEvidenceError("runtime evidence output write failed")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except RuntimeEvidenceError:
        raise
    except OSError as error:
        raise RuntimeEvidenceError("runtime evidence output could not be written") from error
    finally:
        temporary_value = locals().get("temporary")
        if isinstance(temporary_value, Path):
            try:
                temporary_value.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-index-digest", required=True)
    parser.add_argument("--image-manifest-digest", required=True)
    parser.add_argument("--image-config-digest", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--bootstrap-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--stdlib-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument(
        "--distribution-path", type=Path, action="append", required=True
    )
    arguments = parser.parse_args(argv)
    try:
        bootstrap_probe = _load_bootstrap_probe(
            arguments.bootstrap_probe,
            runtime_id=arguments.runtime_id,
        )
    except RuntimeEvidenceError as error:
        report: dict[str, object] = {
            "schema_version": 1,
            "kind": "cf1-runtime-evidence-capture",
            "capture_succeeded": False,
            "authorizes_boot": False,
            "ready": False,
            "runtime_evidence_sha256": None,
            "failure": str(error),
        }
    else:
        try:
            evidence = capture_runtime_evidence(
                runtime_id=arguments.runtime_id,
                oci={
                    "tag": arguments.image_tag,
                    "index_digest": arguments.image_index_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                    "manifest_digest": arguments.image_manifest_digest,
                    "config_digest": arguments.image_config_digest,
                },
                python_executable=arguments.python_executable,
                python_implementation=platform.python_implementation(),
                python_version=platform.python_version(),
                python_build=tuple(platform.python_build()),
                stdlib_root=arguments.stdlib_root,
                environment_root=arguments.environment_root,
                distribution_paths=tuple(arguments.distribution_path),
                wheelhouse=arguments.wheelhouse,
                loaded_facts=bootstrap_probe["loaded_facts"],
            )
            encoded = _canonical_bytes(evidence)
            _atomic_write_evidence(arguments.output, encoded)
        except RuntimeEvidenceError as error:
            report = {
                "schema_version": 1,
                "kind": "cf1-runtime-evidence-capture",
                "capture_succeeded": False,
                "authorizes_boot": False,
                "ready": False,
                "runtime_evidence_sha256": None,
                "failure": str(error),
            }
        except Exception as error:
            report = {
                "schema_version": 1,
                "kind": "cf1-runtime-evidence-capture",
                "capture_succeeded": False,
                "authorizes_boot": False,
                "ready": False,
                "runtime_evidence_sha256": None,
                "failure": f"unexpected evidence capture error: {type(error).__name__}",
            }
        else:
            report = {
                "schema_version": 1,
                "kind": "cf1-runtime-evidence-capture",
                "capture_succeeded": True,
                "authorizes_boot": False,
                "ready": False,
                "runtime_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["capture_succeeded"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
