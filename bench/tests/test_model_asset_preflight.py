from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO
from unittest import mock

from bench import model_asset_preflight
from bench.model_asset_preflight import (
    AssetLockSnapshot,
    AssetLockError,
    load_asset_lock_snapshot,
    verify_model_source_snapshot,
    verify_model_assets,
    verify_model_assets_snapshot,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ModelAssetPreflightTests(unittest.TestCase):
    def test_source_only_snapshot_verification_never_reads_model_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"model bytes that the source-only gate must not read"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock_path = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )
            snapshot = load_asset_lock_snapshot(lock_path)

            with mock.patch.object(
                model_asset_preflight,
                "_asset_report",
                side_effect=AssertionError("model asset bytes must not be read"),
            ):
                report = verify_model_source_snapshot(snapshot, checkout)

        self.assertTrue(report["ready"])
        self.assertEqual(report["lock_sha256"], snapshot.sha256)
        self.assertEqual(report["source"]["observed_commit"], commit)

    def test_lock_snapshot_rejects_a_digest_not_derived_from_its_bytes(self) -> None:
        with self.assertRaisesRegex(AssetLockError, "snapshot SHA-256"):
            AssetLockSnapshot(encoded=b"{}", sha256="0" * 64)

    def make_checkout(self, root: Path) -> tuple[Path, str]:
        checkout = root / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        (checkout / "tracked.txt").write_text("pinned source\n", encoding="utf-8")
        (checkout / ".gitignore").write_text(
            "weights/\n__pycache__/\n", encoding="utf-8"
        )
        subprocess.run(
            ["git", "-C", str(checkout), "add", "tracked.txt", ".gitignore"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Asset Preflight Test",
                "-c",
                "user.email=asset-preflight@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return checkout, commit

    def write_lock(
        self,
        root: Path,
        *,
        commit: str,
        asset_sha256: str,
        asset_size: int,
        relative_path: str = "weights/model.bin",
    ) -> Path:
        lock = {
            "schema_version": 1,
            "stack_id": "fixture-stack",
            "source": {
                "repository": "https://example.invalid/source.git",
                "commit": commit,
            },
            "assets": [
                {
                    "id": "model",
                    "relative_path": relative_path,
                    "size_bytes": asset_size,
                    "sha256": asset_sha256,
                    "source": "fixture@immutable:model.bin",
                }
            ],
        }
        path = root / "lock.json"
        path.write_text(json.dumps(lock), encoding="utf-8")
        return path

    def test_ready_requires_exact_clean_source_and_asset_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact model bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )

            report = verify_model_assets(lock, checkout)

            self.assertTrue(report["ready"])
            self.assertEqual(report["source"]["observed_commit"], commit)
            self.assertTrue(report["source"]["worktree_clean"])
            self.assertEqual(report["assets"][0]["status"], "verified")

    def test_snapshot_verification_does_not_reparse_a_rewritten_lock_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact model bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )
            original_bytes = lock.read_bytes()
            snapshot = load_asset_lock_snapshot(lock)

            rewritten = json.loads(lock.read_text(encoding="utf-8"))
            rewritten["stack_id"] = "rewritten-stack"
            lock.write_text(json.dumps(rewritten), encoding="utf-8")
            report = verify_model_assets_snapshot(snapshot, checkout)

            self.assertTrue(report["ready"])
            self.assertEqual(report["stack_id"], "fixture-stack")
            self.assertEqual(snapshot.sha256, sha256(original_bytes))
            self.assertEqual(report["lock_sha256"], snapshot.sha256)

    def test_missing_size_hash_and_dirty_source_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"wrong bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(b"expected bytes"),
                asset_size=len(payload) + 1,
            )
            (checkout / "tracked.txt").write_text("dirty\n", encoding="utf-8")

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertFalse(report["source"]["worktree_clean"])
            self.assertEqual(report["assets"][0]["status"], "size_mismatch")
            asset.unlink()
            self.assertEqual(
                verify_model_assets(lock, checkout)["assets"][0]["status"],
                "missing",
            )

    def test_hash_mismatch_is_distinct_from_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"same length"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256="0" * 64,
                asset_size=len(payload),
            )
            report = verify_model_assets(lock, checkout)
            self.assertEqual(report["assets"][0]["status"], "sha256_mismatch")
            self.assertFalse(report["ready"])

    def test_untracked_source_injection_blocks_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact model bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )
            (checkout / "pipeline.py").write_text(
                "raise RuntimeError('shadowed source')\n", encoding="utf-8"
            )

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertFalse(report["source"]["worktree_clean"])

    def test_ignored_bytecode_injection_blocks_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact model bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )
            cache = checkout / "__pycache__"
            cache.mkdir()
            (cache / "pipeline.cpython-39.pyc").write_bytes(b"forged bytecode")

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertFalse(report["source"]["worktree_clean"])

    def test_extra_ignored_file_next_to_pinned_asset_blocks_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact model bytes"
            weights = checkout / "weights"
            weights.mkdir()
            (weights / "model.bin").write_bytes(payload)
            (weights / "unregistered.bin").write_bytes(b"unregistered")
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertFalse(report["source"]["worktree_clean"])

    def test_git_environment_cannot_redirect_source_pin_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, _checkout_commit = self.make_checkout(root)
            other = root / "other"
            other.mkdir()
            subprocess.run(["git", "init", "-q", str(other)], check=True)
            (other / "other.txt").write_text("different source\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(other), "add", "other.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(other),
                    "-c",
                    "user.name=Asset Preflight Test",
                    "-c",
                    "user.email=asset-preflight@example.invalid",
                    "commit",
                    "-qm",
                    "other fixture",
                ],
                check=True,
            )
            other_commit = subprocess.run(
                ["git", "-C", str(other), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            payload = b"exact model bytes"
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=other_commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )

            with mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(other / ".git"),
                    "GIT_WORK_TREE": str(other),
                },
            ):
                report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertNotEqual(report["source"]["observed_commit"], other_commit)

    def test_in_tree_asset_symlink_is_not_a_verified_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"exact bytes behind an in-tree symlink"
            weights = checkout / "weights"
            weights.mkdir()
            (weights / "actual.bin").write_bytes(payload)
            (weights / "model.bin").symlink_to("actual.bin")
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertEqual(report["assets"][0]["status"], "not_regular_file")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO support")
    def test_non_regular_asset_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            os.mkfifo(asset)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256="0" * 64,
                asset_size=1,
            )

            report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertEqual(report["assets"][0]["status"], "not_regular_file")

    def test_size_change_inside_large_file_hash_window_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            payload = b"x" * (2 * 1024 * 1024 + 17)
            asset = checkout / "weights" / "model.bin"
            asset.parent.mkdir()
            asset.write_bytes(payload)
            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256=sha256(payload),
                asset_size=len(payload),
            )
            original_hash = model_asset_preflight._sha256_file

            def hash_then_grow(subject: BinaryIO) -> str:
                digest = original_hash(subject)
                with asset.open("ab") as handle:
                    handle.write(b"changed during verification")
                return digest

            with mock.patch.object(
                model_asset_preflight,
                "_sha256_file",
                side_effect=hash_then_grow,
            ):
                report = verify_model_assets(lock, checkout)

            self.assertFalse(report["ready"])
            self.assertEqual(
                report["assets"][0]["status"], "changed_during_verification"
            )

    def test_lock_rejects_duplicate_keys_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, commit = self.make_checkout(root)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}', encoding="utf-8"
            )
            with self.assertRaisesRegex(AssetLockError, "duplicate"):
                verify_model_assets(duplicate, checkout)

            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256="0" * 64,
                asset_size=1,
                relative_path="../escape.bin",
            )
            with self.assertRaisesRegex(AssetLockError, "relative_path"):
                verify_model_assets(lock, checkout)

            lock = self.write_lock(
                root,
                commit=commit,
                asset_sha256="0" * 64,
                asset_size=1,
                relative_path="weights/model\0.bin",
            )
            with self.assertRaisesRegex(AssetLockError, "relative_path"):
                verify_model_assets(lock, checkout)


if __name__ == "__main__":
    unittest.main()
