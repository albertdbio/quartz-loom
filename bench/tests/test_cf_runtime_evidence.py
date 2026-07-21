from __future__ import annotations

import base64
import builtins
import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bench.cf_runtime_evidence as runtime_evidence
from bench.cf_runtime_evidence import (
    MAX_RUNTIME_EVIDENCE_BYTES,
    RuntimeEvidenceError,
    RuntimeEvidenceSnapshot,
    RuntimeEvidenceStaticIdentity,
    capture_runtime_evidence,
    load_runtime_evidence_snapshot,
    main,
    runtime_evidence_capture_report,
    verify_runtime_evidence,
    verify_static_runtime_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.environment = root / "venv"
        self.stdlib = root / "stdlib"
        self.site = self.environment / "lib" / "python3.12" / "site-packages"
        self.wheelhouse = root / "wheelhouse"
        self.executable = self.environment / "bin" / "python"
        self.dist_info = self.site / "demo-1.0.dist-info"
        self.extra_record_paths: list[Path] = []
        self.environment.mkdir()
        self.stdlib.mkdir()
        self.site.mkdir(parents=True)
        self.wheelhouse.mkdir()
        self.executable.parent.mkdir()
        self.executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o755)
        (self.stdlib / "stdlib_demo.py").write_text("VALUE = 1\n", encoding="utf-8")
        self._build_wheel()
        self._install_distribution()

    @property
    def oci(self) -> dict[str, object]:
        return {
            "tag": "example/runtime:one",
            "index_digest": "sha256:" + "1" * 64,
            "platform": {"os": "linux", "architecture": "amd64"},
            "manifest_digest": "sha256:" + "2" * 64,
            "config_digest": "sha256:" + "3" * 64,
        }

    @property
    def loaded_facts(self) -> dict[str, object]:
        libraries = [
            {
                "scope": "environment",
                "identity": (self.site / "demo.py").relative_to(
                    self.environment
                ).as_posix(),
                "size": (self.site / "demo.py").stat().st_size,
                "sha256": _sha256(self.site / "demo.py"),
            }
        ]
        return {
            "loaded_library_identity_policy": "explicit-runtime-roots-v1",
            "torch_version": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "cxx11_abi": True,
            "cudnn_version": "9.10.2",
            "nccl_version": "2.27.3",
            "cuda_arch_list": ["sm_90"],
            "import_order": [
                "torch",
                "verified-attention-source",
                "flash-attention-2",
            ],
            "critical_modules": [
                {
                    "module": "demo._native",
                    "path": str(self.site / "demo.py"),
                    "elf_class": "ELF64",
                    "elf_machine": "Advanced Micro Devices X86-64",
                    "needed": ["libc.so.6"],
                    "rpath": [],
                    "runpath": ["$ORIGIN"],
                }
            ],
            "loaded_libraries": libraries,
            "loaded_libraries_manifest_sha256": runtime_evidence._canonical_sha256(
                libraries
            ),
            "loaded_library_count": len(libraries),
        }

    @contextlib.contextmanager
    def native_observation(self, facts: dict[str, object] | None = None):
        observed_facts = self.loaded_facts if facts is None else facts
        facts_by_path = {
            str(Path(row["path"]).resolve()): row
            for row in observed_facts["critical_modules"]
        }

        def observe(path: Path, label: str):
            del label
            resolved = Path(path).resolve(strict=True)
            row = facts_by_path[str(resolved)]
            metadata = resolved.stat()
            return {
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                "size": metadata.st_size,
                "mode": metadata.st_mode & 0o7777,
                "elf_class": row["elf_class"],
                "elf_machine": row["elf_machine"],
                "needed": list(row["needed"]),
                "rpath": list(row["rpath"]),
                "runpath": list(row["runpath"]),
            }

        with mock.patch.object(
            runtime_evidence, "_observe_elf_file", side_effect=observe
        ):
            yield

    def capture(self) -> dict[str, object]:
        with self.native_observation():
            return capture_runtime_evidence(
                runtime_id="cf1-h100-cu128-v1",
                oci=self.oci,
                python_executable=self.executable,
                python_implementation="CPython",
                python_version="3.12.3",
                python_build=("main", "Jul 20 2026"),
                stdlib_root=self.stdlib,
                environment_root=self.environment,
                distribution_paths=(self.site,),
                wheelhouse=self.wheelhouse,
                loaded_facts=self.loaded_facts,
            )

    def verify(self, snapshot: RuntimeEvidenceSnapshot):
        with self.native_observation():
            return verify_runtime_evidence(
                snapshot,
                observed_oci=self.oci,
                python_executable=self.executable,
                python_implementation="CPython",
                python_version="3.12.3",
                python_build=("main", "Jul 20 2026"),
                stdlib_root=self.stdlib,
                environment_root=self.environment,
                distribution_paths=(self.site,),
                wheelhouse=self.wheelhouse,
                loaded_facts=self.loaded_facts,
            )

    def verify_static(self, snapshot: RuntimeEvidenceSnapshot):
        with self.native_observation():
            return verify_static_runtime_evidence(
                snapshot,
                observed_oci=self.oci,
                python_executable=self.executable,
                python_implementation="CPython",
                python_version="3.12.3",
                python_build=("main", "Jul 20 2026"),
                stdlib_root=self.stdlib,
                environment_root=self.environment,
                distribution_paths=(self.site,),
                wheelhouse=self.wheelhouse,
            )

    def bootstrap_probe(self) -> dict[str, object]:
        loaded_facts = self.loaded_facts
        loaded_sha = runtime_evidence._canonical_sha256(loaded_facts)
        execution = {
            "gpu_execution_performed": True,
            "attention_backend": "flash-attention-2",
            "executed_callable": "flash_attn.flash_attn_varlen_func",
            "call_count": 1,
            "input_shape": [1, 16, 12, 128],
            "output_shape": [1, 16, 12, 128],
            "output_dtype": "torch.bfloat16",
            "output_finite": True,
        }
        identity_material = {
            "schema_version": 3,
            "probe_mode": "unbound-evidence-capture",
            "runtime_lock_sha256": "5" * 64,
            "runtime_evidence_sha256": None,
            "static_environment_sha256": None,
            "runtime_environment_sha256": None,
            "native_identity_sha256": None,
            "asset_lock_sha256": "6" * 64,
            "source_commit": "7" * 40,
            "attention_source_sha256": "8" * 64,
            "loaded_facts_sha256": loaded_sha,
            **execution,
        }
        return {
            "schema_version": 2,
            "kind": "cf1-executed-attention-probe",
            "probe_succeeded": True,
            "authorizes_boot": False,
            "ready": False,
            "probe_mode": "unbound-evidence-capture",
            "runtime_id": "cf1-h100-cu128-v1",
            "runtime_lock_sha256": "5" * 64,
            "runtime_evidence_sha256": None,
            "asset_lock_sha256": "6" * 64,
            "source_commit": "7" * 40,
            "attention_source_sha256": "8" * 64,
            "loaded_facts": loaded_facts,
            "loaded_facts_sha256": loaded_sha,
            "static_environment_sha256": None,
            "runtime_environment_sha256": None,
            "native_identity_sha256": None,
            **execution,
            "probe_identity_sha256": runtime_evidence._canonical_sha256(
                identity_material
            ),
        }

    def snapshot(self, evidence: dict[str, object] | None = None) -> RuntimeEvidenceSnapshot:
        encoded = json.dumps(
            evidence if evidence is not None else self.capture(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return RuntimeEvidenceSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def _build_wheel(self) -> None:
        wheel = self.wheelhouse / "demo-1.0-py3-none-any.whl"
        members = {
            "demo-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n"
            ),
            "demo-1.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
            ),
            "demo.py": b"VALUE = 1\n",
        }
        rows = []
        for name, encoded in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(encoded).digest()).rstrip(
                b"="
            ).decode("ascii")
            rows.append((name, "sha256=" + digest, str(len(encoded))))
        rows.append(("demo-1.0.dist-info/RECORD", "", ""))
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(rows)
        members["demo-1.0.dist-info/RECORD"] = record.getvalue().encode("utf-8")
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, encoded in members.items():
                archive.writestr(name, encoded)
        alias = self.wheelhouse / "demo-1.0-alias-py3-none-any.whl"
        os.link(wheel, alias)

    def _install_distribution(self) -> None:
        self.dist_info.mkdir()
        (self.site / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n",
            encoding="utf-8",
        )
        (self.dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
            encoding="utf-8",
        )
        (self.dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
        wheel = self.wheelhouse / "demo-1.0-py3-none-any.whl"
        direct_url = {
            "url": wheel.resolve().as_uri(),
            "archive_info": {"hashes": {"sha256": _sha256(wheel)}},
        }
        (self.dist_info / "direct_url.json").write_text(
            json.dumps(direct_url), encoding="utf-8"
        )
        self._write_record()

    def _write_record(self) -> None:
        record_paths = [
            self.site / "demo.py",
            self.dist_info / "METADATA",
            self.dist_info / "WHEEL",
            self.dist_info / "INSTALLER",
            self.dist_info / "direct_url.json",
            *self.extra_record_paths,
        ]
        rows = []
        for path in record_paths:
            relative = path.relative_to(self.site).as_posix()
            rows.append((relative, _record_digest(path), str(path.stat().st_size)))
        rows.append(("demo-1.0.dist-info/RECORD", "", ""))
        with (self.dist_info / "RECORD").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)


class RuntimeEvidenceTests(unittest.TestCase):
    def test_locked_identities_are_derived_from_exact_evidence_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            snapshot = fixture.snapshot()
            locked = snapshot.parsed()
            expected_runtime = runtime_evidence._canonical_sha256(
                {
                    "runtime_evidence_sha256": snapshot.sha256,
                    "oci": locked["oci"],
                    "python": locked["python"],
                    "environment": locked["environment"],
                    "native": locked["native"],
                }
            )
            expected_native = runtime_evidence._canonical_sha256(locked["native"])

            locked_identities = runtime_evidence.runtime_evidence_locked_identities(
                snapshot
            )
            self.assertEqual(
                locked_identities.runtime_environment_sha256,
                expected_runtime,
            )
            self.assertEqual(
                locked_identities.native_identity_sha256,
                expected_native,
            )

    def test_static_verify_rejects_wheel_and_tree_drift_without_loaded_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for label in ("wheel", "tree"):
                with self.subTest(label=label):
                    fixture = EvidenceFixture(Path(directory) / label)
                    snapshot = fixture.snapshot()
                    static_identity = fixture.verify_static(snapshot)
                    self.assertIsInstance(
                        static_identity, RuntimeEvidenceStaticIdentity
                    )
                    self.assertEqual(
                        static_identity.runtime_evidence_sha256, snapshot.sha256
                    )
                    if label == "wheel":
                        wheel = fixture.wheelhouse / "demo-1.0-py3-none-any.whl"
                        wheel.write_bytes(wheel.read_bytes() + b"drift")
                    else:
                        (fixture.site / "unrecorded.py").write_text(
                            "DRIFT = True\n", encoding="utf-8"
                        )
                    with mock.patch(
                        "builtins.__import__", wraps=builtins.__import__
                    ) as imported, self.assertRaisesRegex(
                        RuntimeEvidenceError, "does not match|unowned importable"
                    ):
                        fixture.verify_static(snapshot)
                    self.assertFalse(
                        any(
                            call.args
                            and isinstance(call.args[0], str)
                            and (
                                call.args[0] == "torch"
                                or call.args[0].startswith("torch.")
                            )
                            for call in imported.mock_calls
                        )
                    )

    def test_native_evidence_binds_elf_rpath_and_runpath(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            evidence = fixture.capture()

        module = evidence["native"]["critical_modules"][0]
        self.assertEqual(module["rpath"], [])
        self.assertEqual(module["runpath"], ["$ORIGIN"])
        self.assertEqual(
            evidence["native"]["import_order"],
            ["torch", "verified-attention-source", "flash-attention-2"],
        )

    def test_capture_is_bounded_strict_and_redacts_absolute_direct_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            evidence = fixture.capture()
            snapshot = fixture.snapshot(evidence)

            self.assertLessEqual(len(snapshot.encoded), MAX_RUNTIME_EVIDENCE_BYTES)
            self.assertNotIn(str(fixture.root), snapshot.encoded.decode("utf-8"))
            self.assertEqual(
                evidence["environment"]["packages"],
                [
                    {
                        "distribution": "demo",
                        "version": "1.0",
                        "wheel": {
                            "filenames": [
                                "demo-1.0-alias-py3-none-any.whl",
                                "demo-1.0-py3-none-any.whl",
                            ],
                            "sha256": _sha256(
                                fixture.wheelhouse / "demo-1.0-py3-none-any.whl"
                            ),
                        },
                        "installed_file_count": 6,
                        "installed_manifest_sha256": mock.ANY,
                        "install_metadata_sha256": mock.ANY,
                    }
                ],
            )
            self.assertEqual(
                evidence["native"]["loaded_libraries"],
                fixture.loaded_facts["loaded_libraries"],
            )
            self.assertEqual(
                evidence["native"]["loaded_library_count"],
                len(evidence["native"]["loaded_libraries"]),
            )
            self.assertEqual(
                evidence["native"]["critical_modules"][0]["origin_scope"],
                "environment",
            )
            self.assertNotIn(
                "path", evidence["native"]["critical_modules"][0]
            )

    def test_snapshot_rejects_oversize_digest_drift_and_duplicate_keys(self) -> None:
        oversized = b"{" + b" " * MAX_RUNTIME_EVIDENCE_BYTES + b"}"
        with self.assertRaisesRegex(RuntimeEvidenceError, "size"):
            RuntimeEvidenceSnapshot(
                encoded=oversized,
                sha256=hashlib.sha256(oversized).hexdigest(),
            )

        encoded = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaisesRegex(RuntimeEvidenceError, "duplicate"):
            RuntimeEvidenceSnapshot(
                encoded=encoded,
                sha256=hashlib.sha256(encoded).hexdigest(),
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeEvidenceError):
                load_runtime_evidence_snapshot(path)

        valid = b"{}"
        with self.assertRaisesRegex(RuntimeEvidenceError, "SHA-256"):
            RuntimeEvidenceSnapshot(encoded=valid, sha256="0" * 64)

    def test_verify_binds_wheel_record_tree_python_oci_and_loaded_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            snapshot = fixture.snapshot()
            identity = fixture.verify(snapshot)
            self.assertEqual(identity.runtime_evidence_sha256, snapshot.sha256)
            self.assertEqual(identity.runtime_id, "cf1-h100-cu128-v1")

            for label in (
                "wheel",
                "site tree",
                "Python executable",
                "OCI",
                "loaded facts",
            ):
                with self.subTest(label=label):
                    fresh = EvidenceFixture(Path(directory) / label.replace(" ", "-"))
                    fresh_snapshot = fresh.snapshot()
                    if label == "OCI":
                        observed = fresh.oci
                        observed["manifest_digest"] = "sha256:" + "9" * 64
                        with fresh.native_observation(), self.assertRaisesRegex(
                            RuntimeEvidenceError, "does not match"
                        ):
                            verify_runtime_evidence(
                                fresh_snapshot,
                                observed_oci=observed,
                                python_executable=fresh.executable,
                                python_implementation="CPython",
                                python_version="3.12.3",
                                python_build=("main", "Jul 20 2026"),
                                stdlib_root=fresh.stdlib,
                                environment_root=fresh.environment,
                                distribution_paths=(fresh.site,),
                                wheelhouse=fresh.wheelhouse,
                                loaded_facts=fresh.loaded_facts,
                            )
                        continue
                    if label == "loaded facts":
                        facts = fresh.loaded_facts
                        facts["cuda_runtime"] = "0.0"
                        with fresh.native_observation(facts), self.assertRaisesRegex(
                            RuntimeEvidenceError, "does not match"
                        ):
                            verify_runtime_evidence(
                                fresh_snapshot,
                                observed_oci=fresh.oci,
                                python_executable=fresh.executable,
                                python_implementation="CPython",
                                python_version="3.12.3",
                                python_build=("main", "Jul 20 2026"),
                                stdlib_root=fresh.stdlib,
                                environment_root=fresh.environment,
                                distribution_paths=(fresh.site,),
                                wheelhouse=fresh.wheelhouse,
                                loaded_facts=facts,
                            )
                        continue
                    if label == "wheel":
                        target = fresh.wheelhouse / "demo-1.0-py3-none-any.whl"
                        target.write_bytes(target.read_bytes() + b"changed")
                    elif label == "site tree":
                        (fresh.site / "unexpected.py").write_text(
                            "VALUE = 2\n", encoding="utf-8"
                        )
                    else:
                        fresh.executable.write_bytes(b"changed")
                    with self.assertRaisesRegex(
                        RuntimeEvidenceError, "does not match|unowned importable"
                    ):
                        fresh.verify(fresh_snapshot)

    def test_record_hash_and_unsafe_direct_url_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            record = fixture.dist_info / "RECORD"
            text = record.read_text(encoding="utf-8")
            record.write_text(text.replace("sha256=", "sha256=AAAA", 1), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeEvidenceError, "RECORD"):
                fixture.capture()

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            direct_url = fixture.dist_info / "direct_url.json"
            direct_url.write_text(
                json.dumps(
                    {
                        "url": "https://user:secret@example.invalid/demo.whl",
                        "archive_info": {"hashes": {"sha256": "a" * 64}},
                    }
                ),
                encoding="utf-8",
            )
            fixture._write_record()
            with self.assertRaisesRegex(RuntimeEvidenceError, "direct_url"):
                fixture.capture()

    def test_module_never_imports_torch_and_capture_report_never_authorizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("Torch import is forbidden")
                return original_import(name, *args, **kwargs)

            with mock.patch(
                "builtins.__import__", side_effect=guarded_import
            ), fixture.native_observation():
                report = runtime_evidence_capture_report(
                    runtime_id="cf1-h100-cu128-v1",
                    oci=fixture.oci,
                    python_executable=fixture.executable,
                    python_implementation="CPython",
                    python_version="3.12.3",
                    python_build=("main", "Jul 20 2026"),
                    stdlib_root=fixture.stdlib,
                    environment_root=fixture.environment,
                    distribution_paths=(fixture.site,),
                    wheelhouse=fixture.wheelhouse,
                    loaded_facts=fixture.loaded_facts,
                )

            self.assertTrue(report["capture_succeeded"])
            self.assertFalse(report["authorizes_boot"])
            self.assertFalse(report["ready"])
            self.assertEqual(len(report["runtime_evidence_sha256"]), 64)

    def test_tree_binds_pyc_bytes_and_rejects_unowned_importable_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "owned-pyc")
            cache = fixture.site / "__pycache__"
            cache.mkdir()
            pyc = cache / "demo.cpython-312.pyc"
            pyc.write_bytes(b"bound bytecode")
            fixture.extra_record_paths.append(pyc)
            fixture._write_record()
            snapshot = fixture.snapshot()
            pyc.write_bytes(b"mutated bytecode")
            fixture._write_record()
            with self.assertRaisesRegex(RuntimeEvidenceError, "does not match"):
                fixture.verify_static(snapshot)

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "unowned")
            (fixture.site / "rogue.py").write_text("ROGUE = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeEvidenceError, "unowned importable"):
                fixture.capture()

    def test_missing_unhashed_generated_bytecode_is_optional_only_inside_environment(
        self,
    ) -> None:
        def append_record_row(
            fixture: EvidenceFixture,
            path: str,
            digest: str = "",
            size: str = "",
        ) -> None:
            record = fixture.dist_info / "RECORD"
            with record.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    (path, digest, size)
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "optional-pyc")
            append_record_row(
                fixture,
                "__pycache__/demo.cpython-312.pyc",
            )
            report = fixture.capture()
            self.assertEqual(report["runtime_id"], "cf1-h100-cu128-v1")

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "hashed-pyc")
            append_record_row(
                fixture,
                "__pycache__/demo.cpython-312.pyc",
                "sha256=" + base64.urlsafe_b64encode(b"0" * 32).rstrip(b"=").decode(),
                "1",
            )
            with self.assertRaisesRegex(RuntimeEvidenceError, "RECORD path is missing"):
                fixture.capture()

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "missing-source")
            append_record_row(fixture, "missing.py")
            with self.assertRaisesRegex(RuntimeEvidenceError, "RECORD path is missing"):
                fixture.capture()

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "escaping-pyc")
            append_record_row(
                fixture,
                "../../../../outside/__pycache__/demo.cpython-312.pyc",
            )
            with self.assertRaisesRegex(
                RuntimeEvidenceError, "RECORD path escapes the environment"
            ):
                fixture.capture()

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "duplicate-pyc")
            missing_pyc = "__pycache__/demo.cpython-312.pyc"
            append_record_row(fixture, missing_pyc)
            append_record_row(fixture, missing_pyc)
            with self.assertRaisesRegex(RuntimeEvidenceError, "RECORD path is duplicated"):
                fixture.capture()

    def test_tree_rejects_escaping_symlink_and_binds_internal_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = EvidenceFixture(root / "escape")
            outside = root / "outside"
            outside.mkdir()
            (fixture.environment / "escape-link").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeEvidenceError, "symlink directory escapes"):
                fixture.capture()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = EvidenceFixture(root / "external-file")
            outside = root / "external-target"
            outside.write_text("one", encoding="utf-8")
            (fixture.environment / "external-link").symlink_to(outside)
            snapshot = fixture.snapshot()
            outside.write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeEvidenceError, "does not match"):
                fixture.verify_static(snapshot)

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "internal")
            targets = fixture.environment / "targets"
            targets.mkdir()
            (targets / "one").write_text("one", encoding="utf-8")
            (targets / "two").write_text("two", encoding="utf-8")
            link = fixture.environment / "bound-link"
            link.symlink_to(Path("targets") / "one")
            snapshot = fixture.snapshot()
            link.unlink()
            link.symlink_to(Path("targets") / "two")
            with self.assertRaisesRegex(RuntimeEvidenceError, "does not match"):
                fixture.verify_static(snapshot)

    def test_installed_payload_must_match_reviewed_wheel_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            installed = fixture.site / "demo.py"
            installed.write_text("VALUE = 999\n", encoding="utf-8")
            fixture._write_record()
            with self.assertRaisesRegex(RuntimeEvidenceError, "wheel payload"):
                fixture.capture()

    def test_native_metadata_preserves_dynamic_order_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            facts = fixture.loaded_facts
            module = facts["critical_modules"][0]
            module["needed"] = ["libz.so.1", "libc.so.6"]
            module["runpath"] = ["$ORIGIN/z", "$ORIGIN/a"]
            with fixture.native_observation(facts):
                evidence = capture_runtime_evidence(
                    runtime_id="cf1-h100-cu128-v1",
                    oci=fixture.oci,
                    python_executable=fixture.executable,
                    python_implementation="CPython",
                    python_version="3.12.3",
                    python_build=("main", "Jul 20 2026"),
                    stdlib_root=fixture.stdlib,
                    environment_root=fixture.environment,
                    distribution_paths=(fixture.site,),
                    wheelhouse=fixture.wheelhouse,
                    loaded_facts=facts,
                )
            locked = evidence["native"]["critical_modules"][0]
            self.assertEqual(locked["needed"], ["libz.so.1", "libc.so.6"])
            self.assertEqual(locked["runpath"], ["$ORIGIN/z", "$ORIGIN/a"])

            module["needed"] = ["libc.so.6", "libc.so.6"]
            with self.assertRaisesRegex(RuntimeEvidenceError, "duplicated"):
                with fixture.native_observation(facts):
                    capture_runtime_evidence(
                        runtime_id="cf1-h100-cu128-v1",
                        oci=fixture.oci,
                        python_executable=fixture.executable,
                        python_implementation="CPython",
                        python_version="3.12.3",
                        python_build=("main", "Jul 20 2026"),
                        stdlib_root=fixture.stdlib,
                        environment_root=fixture.environment,
                        distribution_paths=(fixture.site,),
                        wheelhouse=fixture.wheelhouse,
                        loaded_facts=facts,
                    )

            facts = fixture.loaded_facts
            path = fixture.site / "demo.py"
            mismatched_observation = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "mode": path.stat().st_mode & 0o7777,
                "elf_class": "ELF64",
                "elf_machine": "Advanced Micro Devices X86-64",
                "needed": ["libc.so.6"],
                "rpath": [],
                "runpath": ["/not-the-probe-value"],
            }
            with mock.patch.object(
                runtime_evidence,
                "_observe_elf_file",
                return_value=mismatched_observation,
            ), self.assertRaisesRegex(RuntimeEvidenceError, "independent readelf"):
                capture_runtime_evidence(
                    runtime_id="cf1-h100-cu128-v1",
                    oci=fixture.oci,
                    python_executable=fixture.executable,
                    python_implementation="CPython",
                    python_version="3.12.3",
                    python_build=("main", "Jul 20 2026"),
                    stdlib_root=fixture.stdlib,
                    environment_root=fixture.environment,
                    distribution_paths=(fixture.site,),
                    wheelhouse=fixture.wheelhouse,
                    loaded_facts=facts,
                )

    def test_native_metadata_is_derived_with_readelf_from_hashing_fd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.so"
            path.write_bytes(b"ELF bytes for mocked readelf")
            observed_fd: list[int] = []

            def run(command, **kwargs):
                fd = kwargs["pass_fds"][0]
                observed_fd.append(fd)
                self.assertEqual(command[-1], f"/proc/self/fd/{fd}")
                descriptor_stat = os.fstat(fd)
                path_stat = path.stat()
                self.assertEqual(
                    (descriptor_stat.st_dev, descriptor_stat.st_ino),
                    (path_stat.st_dev, path_stat.st_ino),
                )
                return SimpleNamespace(
                    stdout=(
                        "  Class:                             ELF64\n"
                        "  Machine:                           Advanced Micro Devices X86-64\n"
                        " 0x0 (NEEDED) Shared library: [libz.so.1]\n"
                        " 0x0 (NEEDED) Shared library: [libc.so.6]\n"
                        " 0x0 (RUNPATH) Library runpath: [$ORIGIN/z:$ORIGIN/a]\n"
                    )
                )

            with mock.patch.object(runtime_evidence.subprocess, "run", side_effect=run):
                observed = runtime_evidence._observe_elf_file(path, "test ELF")
            self.assertEqual(observed_fd and len(observed_fd), 1)
            self.assertEqual(observed["sha256"], _sha256(path))
            self.assertEqual(observed["needed"], ["libz.so.1", "libc.so.6"])
            self.assertEqual(observed["runpath"], ["$ORIGIN/z", "$ORIGIN/a"])

            def duplicate_run(command, **kwargs):
                del command, kwargs
                return SimpleNamespace(
                    stdout=(
                        "  Class: ELF64\n"
                        "  Machine: Advanced Micro Devices X86-64\n"
                        " 0x0 (NEEDED) Shared library: [libc.so.6]\n"
                        " 0x0 (NEEDED) Shared library: [libc.so.6]\n"
                    )
                )

            with mock.patch.object(
                runtime_evidence.subprocess, "run", side_effect=duplicate_run
            ), self.assertRaisesRegex(RuntimeEvidenceError, "duplicated"):
                runtime_evidence._observe_elf_file(path, "duplicate ELF")

    def test_capture_rechecks_static_bytes_after_native_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            original = runtime_evidence._capture_native

            def mutate_after_native(*args, **kwargs):
                result = original(*args, **kwargs)
                (fixture.stdlib / "stdlib_demo.py").write_text(
                    "VALUE = 2\n", encoding="utf-8"
                )
                return result

            with fixture.native_observation(), mock.patch.object(
                runtime_evidence, "_capture_native", side_effect=mutate_after_native
            ), self.assertRaisesRegex(RuntimeEvidenceError, "changed during native"):
                capture_runtime_evidence(
                    runtime_id="cf1-h100-cu128-v1",
                    oci=fixture.oci,
                    python_executable=fixture.executable,
                    python_implementation="CPython",
                    python_version="3.12.3",
                    python_build=("main", "Jul 20 2026"),
                    stdlib_root=fixture.stdlib,
                    environment_root=fixture.environment,
                    distribution_paths=(fixture.site,),
                    wheelhouse=fixture.wheelhouse,
                    loaded_facts=fixture.loaded_facts,
                )

    def test_legacy_probe_evidence_schema_and_identity_policy_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            evidence = fixture.capture()
            evidence["schema_version"] = 1
            with self.assertRaisesRegex(RuntimeEvidenceError, "schema version"):
                runtime_evidence.validate_runtime_evidence(evidence)

            evidence = fixture.capture()
            del evidence["native"]["loaded_library_identity_policy"]
            with self.assertRaisesRegex(RuntimeEvidenceError, "native evidence"):
                runtime_evidence.validate_runtime_evidence(evidence)

            probe = fixture.bootstrap_probe()
            probe["schema_version"] = 1
            with self.assertRaisesRegex(RuntimeEvidenceError, "complete success"):
                runtime_evidence._validate_bootstrap_probe(
                    probe,
                    runtime_id="cf1-h100-cu128-v1",
                )

            probe = fixture.bootstrap_probe()
            del probe["loaded_facts"]["loaded_library_identity_policy"]
            with self.assertRaisesRegex(RuntimeEvidenceError, "bootstrap loaded facts"):
                runtime_evidence._validate_bootstrap_probe(
                    probe,
                    runtime_id="cf1-h100-cu128-v1",
                )

    def test_capture_rehashes_loaded_library_scope_association(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            facts = fixture.loaded_facts
            facts["loaded_libraries"][0]["sha256"] = "a" * 64
            facts["loaded_libraries_manifest_sha256"] = (
                runtime_evidence._canonical_sha256(facts["loaded_libraries"])
            )
            with fixture.native_observation(facts), self.assertRaisesRegex(
                RuntimeEvidenceError, "loaded library does not match file"
            ):
                capture_runtime_evidence(
                    runtime_id="cf1-h100-cu128-v1",
                    oci=fixture.oci,
                    python_executable=fixture.executable,
                    python_implementation="CPython",
                    python_version="3.12.3",
                    python_build=("main", "Jul 20 2026"),
                    stdlib_root=fixture.stdlib,
                    environment_root=fixture.environment,
                    distribution_paths=(fixture.site,),
                    wheelhouse=fixture.wheelhouse,
                    loaded_facts=facts,
                )

    def test_cli_requires_complete_probe_and_atomically_writes_canonical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory) / "fixture")
            probe_path = Path(directory) / "probe.json"
            output = Path(directory) / "evidence.json"
            probe_path.write_text(json.dumps(fixture.bootstrap_probe()), encoding="utf-8")
            with fixture.native_observation(), contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--runtime-id", "cf1-h100-cu128-v1",
                        "--image-tag", "example/runtime:one",
                        "--image-index-digest", "sha256:" + "1" * 64,
                        "--image-manifest-digest", "sha256:" + "2" * 64,
                        "--image-config-digest", "sha256:" + "3" * 64,
                        "--wheelhouse", str(fixture.wheelhouse),
                        "--bootstrap-probe", str(probe_path),
                        "--output", str(output),
                        "--python-executable", str(fixture.executable),
                        "--stdlib-root", str(fixture.stdlib),
                        "--environment-root", str(fixture.environment),
                        "--distribution-path", str(fixture.site),
                    ]
                )
            self.assertEqual(status, 0)
            encoded = output.read_bytes()
            parsed = json.loads(encoded)
            self.assertEqual(
                encoded,
                json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
            self.assertLessEqual(len(encoded), MAX_RUNTIME_EVIDENCE_BYTES)

            probe_path.write_text(json.dumps(fixture.loaded_facts), encoding="utf-8")
            rejected = Path(directory) / "rejected.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--runtime-id", "cf1-h100-cu128-v1",
                        "--image-tag", "example/runtime:one",
                        "--image-index-digest", "sha256:" + "1" * 64,
                        "--image-manifest-digest", "sha256:" + "2" * 64,
                        "--image-config-digest", "sha256:" + "3" * 64,
                        "--wheelhouse", str(fixture.wheelhouse),
                        "--bootstrap-probe", str(probe_path),
                        "--output", str(rejected),
                        "--python-executable", str(fixture.executable),
                        "--stdlib-root", str(fixture.stdlib),
                        "--environment-root", str(fixture.environment),
                        "--distribution-path", str(fixture.site),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
