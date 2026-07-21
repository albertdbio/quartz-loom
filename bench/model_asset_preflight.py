"""Fail-closed byte and source-pin checks for the real CF++ streaming stack."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parent
    / "model_assets"
    / "cf1-rolling-taehv-v1.lock.json"
)
DEFAULT_CHECKOUT = PROJECT_ROOT / ".upstream" / "Causal-Forcing"
_SHA256_LENGTH = 64
_READ_BYTES = 1024 * 1024
_MAX_LOCK_BYTES = 1024 * 1024


class AssetLockError(ValueError):
    """The lock itself is malformed or permits an ambiguous asset path."""


@dataclass(frozen=True)
class AssetLockSnapshot:
    """One immutable read of the lock bytes used throughout a runtime boot."""

    encoded: bytes
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.encoded, bytes)
            or not self.encoded
            or len(self.encoded) > _MAX_LOCK_BYTES
        ):
            raise AssetLockError("asset lock snapshot size is outside bounds")
        observed = hashlib.sha256(self.encoded).hexdigest()
        if self.sha256 != observed:
            raise AssetLockError("asset lock snapshot SHA-256 does not match bytes")
        _parse_asset_lock(self.encoded)

    def parsed(self) -> Dict[str, Any]:
        return _parse_asset_lock(self.encoded)


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AssetLockError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    if set(value) != set(expected):
        raise AssetLockError(f"{label} fields do not match the lock schema")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssetLockError(f"{label} must be a non-empty string")
    return value


def _validate_relative_path(value: object, label: str) -> str:
    path_text = _require_nonempty_string(value, label)
    path = Path(path_text)
    if "\x00" in path_text or path.is_absolute() or path_text != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise AssetLockError(f"{label} must be a normalized safe relative_path")
    return path_text


def _parse_asset_lock(encoded: bytes) -> Dict[str, Any]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as error:
        raise AssetLockError("asset lock is not valid UTF-8") from error
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise AssetLockError("asset lock is not valid JSON") from error
    if not isinstance(value, dict):
        raise AssetLockError("asset lock must be an object")
    _expect_exact_keys(
        value,
        ("schema_version", "stack_id", "source", "assets"),
        "asset lock",
    )
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise AssetLockError("unsupported asset lock schema_version")
    _require_nonempty_string(value["stack_id"], "stack_id")

    source = value["source"]
    if not isinstance(source, dict):
        raise AssetLockError("source must be an object")
    _expect_exact_keys(source, ("repository", "commit"), "source")
    _require_nonempty_string(source["repository"], "source.repository")
    if not _is_commit(source["commit"]):
        raise AssetLockError("source.commit must be a lowercase 40-character commit")

    assets = value["assets"]
    if not isinstance(assets, list) or not assets:
        raise AssetLockError("assets must be a non-empty array")
    seen_ids = set()
    seen_paths = set()
    for index, asset in enumerate(assets):
        label = f"assets[{index}]"
        if not isinstance(asset, dict):
            raise AssetLockError(f"{label} must be an object")
        _expect_exact_keys(
            asset,
            ("id", "relative_path", "size_bytes", "sha256", "source"),
            label,
        )
        asset_id = _require_nonempty_string(asset["id"], f"{label}.id")
        relative_path = _validate_relative_path(
            asset["relative_path"], f"{label}.relative_path"
        )
        size = asset["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise AssetLockError(f"{label}.size_bytes must be a positive integer")
        if not _is_sha256(asset["sha256"]):
            raise AssetLockError(f"{label}.sha256 must be a lowercase SHA-256")
        _require_nonempty_string(asset["source"], f"{label}.source")
        if asset_id in seen_ids:
            raise AssetLockError(f"duplicate asset id: {asset_id}")
        if relative_path in seen_paths:
            raise AssetLockError(f"duplicate asset relative_path: {relative_path}")
        seen_ids.add(asset_id)
        seen_paths.add(relative_path)
    return value


def load_asset_lock_snapshot(path: Path) -> AssetLockSnapshot:
    """Read and validate one bounded lock snapshot for an entire boot."""

    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise AssetLockError("asset lock could not be read") from error
    if not encoded or len(encoded) > _MAX_LOCK_BYTES:
        raise AssetLockError("asset lock size is outside bounds")
    return AssetLockSnapshot(
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def load_asset_lock(path: Path) -> Dict[str, Any]:
    """Load and strictly validate an immutable model-asset lock."""

    return load_asset_lock_snapshot(path).parsed()


def _run_git(checkout: Path, arguments: Sequence[str]) -> str:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env=environment,
    )
    return result.stdout.strip()


def _unexpected_checkout_paths(
    checkout: Path,
    allowed_untracked_paths: set[str],
) -> Tuple[int, List[str]]:
    tracked_paths = {
        path
        for path in _run_git(checkout, ("ls-files", "-z")).split("\x00")
        if path
    }
    unexpected: List[str] = []
    unexpected_count = 0

    def record(path: Path) -> None:
        nonlocal unexpected_count
        relative = path.relative_to(checkout).as_posix()
        if relative in tracked_paths or relative in allowed_untracked_paths:
            return
        unexpected_count += 1
        if len(unexpected) < 64:
            unexpected.append(relative)

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        checkout,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        directory_path = Path(directory)
        if directory_path == checkout and ".git" in directory_names:
            directory_names.remove(".git")
        for name in tuple(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                directory_names.remove(name)
                record(candidate)
        for name in file_names:
            record(directory_path / name)
    return unexpected_count, sorted(unexpected)


def _source_report(
    checkout: Path,
    expected_commit: str,
    allowed_untracked_paths: set[str],
) -> Dict[str, Any]:
    observed_commit = None
    worktree_clean = False
    unexpected_path_count = None
    unexpected_paths: List[str] = []
    status = "unavailable"
    if checkout.is_dir():
        try:
            observed_commit = _run_git(checkout, ("rev-parse", "HEAD"))
            tracked_clean = not _run_git(
                checkout, ("status", "--porcelain", "--untracked-files=no")
            )
            unexpected_path_count, unexpected_paths = _unexpected_checkout_paths(
                checkout,
                allowed_untracked_paths,
            )
            worktree_clean = tracked_clean and unexpected_path_count == 0
            status = (
                "verified"
                if observed_commit == expected_commit and worktree_clean
                else "mismatch"
            )
        except (OSError, subprocess.SubprocessError):
            status = "unavailable"
    return {
        "expected_commit": expected_commit,
        "observed_commit": observed_commit,
        "worktree_clean": worktree_clean,
        "unexpected_path_count": unexpected_path_count,
        "unexpected_paths": unexpected_paths,
        "status": status,
    }


def _sha256_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(_READ_BYTES)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _open_asset_nofollow(
    checkout: Path, relative_path: str
) -> Tuple[int, int, str]:
    """Open an asset and each parent directory without following symlinks."""

    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise OSError(errno.ENOTSUP, "no-follow asset opens are unsupported")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | close_on_exec
    components = Path(relative_path).parts
    parent_fd = os.open(checkout, directory_flags)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            previous_fd = parent_fd
            parent_fd = next_fd
            os.close(previous_fd)
        asset_fd = os.open(components[-1], file_flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    return asset_fd, parent_fd, components[-1]


def _stat_fingerprint(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _asset_report(checkout: Path, asset: Mapping[str, Any]) -> Dict[str, Any]:
    relative_path = asset["relative_path"]
    observed_size = None
    observed_sha256 = None
    status = "missing"
    asset_fd = None
    parent_fd = None
    leaf_name = None
    try:
        asset_fd, parent_fd, leaf_name = _open_asset_nofollow(
            checkout, relative_path
        )
    except FileNotFoundError:
        status = "missing"
    except OSError as error:
        status = (
            "not_regular_file"
            if error.errno in {errno.ELOOP, errno.ENOTDIR}
            else "unreadable"
        )
    if asset_fd is not None and parent_fd is not None and leaf_name is not None:
        try:
            before = os.fstat(asset_fd)
            observed_size = before.st_size
            if not stat.S_ISREG(before.st_mode):
                status = "not_regular_file"
            elif observed_size != asset["size_bytes"]:
                status = "size_mismatch"
            else:
                with os.fdopen(os.dup(asset_fd), "rb") as handle:
                    observed_sha256 = _sha256_file(handle)
                after = os.fstat(asset_fd)
                try:
                    path_after = os.stat(
                        leaf_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    status = "changed_during_verification"
                else:
                    unchanged = (
                        stat.S_ISREG(after.st_mode)
                        and stat.S_ISREG(path_after.st_mode)
                        and _stat_fingerprint(before) == _stat_fingerprint(after)
                        and _stat_fingerprint(after)
                        == _stat_fingerprint(path_after)
                    )
                    if not unchanged:
                        status = "changed_during_verification"
                    else:
                        status = (
                            "verified"
                            if observed_sha256 == asset["sha256"]
                            else "sha256_mismatch"
                        )
        except OSError:
            status = "unreadable"
        finally:
            try:
                os.close(asset_fd)
            finally:
                os.close(parent_fd)
    return {
        "id": asset["id"],
        "relative_path": relative_path,
        "expected_size_bytes": asset["size_bytes"],
        "observed_size_bytes": observed_size,
        "expected_sha256": asset["sha256"],
        "observed_sha256": observed_sha256,
        "status": status,
    }


def verify_model_assets_snapshot(
    snapshot: AssetLockSnapshot, checkout: Path
) -> Dict[str, Any]:
    """Verify source/assets against one already captured lock snapshot."""

    lock = snapshot.parsed()
    checkout = checkout.resolve()
    source = _source_report(
        checkout,
        lock["source"]["commit"],
        {asset["relative_path"] for asset in lock["assets"]},
    )
    assets: List[Dict[str, Any]] = [
        _asset_report(checkout, asset) for asset in lock["assets"]
    ]
    ready = source["status"] == "verified" and all(
        asset["status"] == "verified" for asset in assets
    )
    return {
        "schema_version": 1,
        "stack_id": lock["stack_id"],
        "lock_sha256": snapshot.sha256,
        "checkout": str(checkout),
        "source": source,
        "assets": assets,
        "ready": ready,
    }


def verify_model_source_snapshot(
    snapshot: AssetLockSnapshot, checkout: Path
) -> Dict[str, Any]:
    """Verify only the pinned Git source, without opening large model assets."""

    lock = snapshot.parsed()
    checkout = checkout.resolve()
    source = _source_report(
        checkout,
        lock["source"]["commit"],
        {asset["relative_path"] for asset in lock["assets"]},
    )
    return {
        "schema_version": 1,
        "stack_id": lock["stack_id"],
        "lock_sha256": snapshot.sha256,
        "checkout": str(checkout),
        "source": source,
        "ready": source["status"] == "verified",
    }


def verify_model_assets(lock_path: Path, checkout: Path) -> Dict[str, Any]:
    """Return a deterministic readiness report without opening model code."""

    return verify_model_assets_snapshot(
        load_asset_lock_snapshot(lock_path), checkout
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        report = verify_model_assets(args.lock, args.checkout)
    except AssetLockError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
