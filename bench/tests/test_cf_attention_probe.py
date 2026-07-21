from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from bench.cf_attention_probe import (
    AttentionProbeError,
    _execute_attention_kernel,
    _execute_attention_source,
    _loaded_library_identity,
    _probe_identity_roots,
    _require_isolated_import_context,
    attention_probe_report,
    main,
)
from bench.cf_runtime_evidence import (
    RuntimeEvidenceIdentity,
    RuntimeEvidenceStaticIdentity,
)
from bench.cf_runtime_preflight import (
    DEFAULT_RUNTIME_LOCK_PATH,
    RuntimeEvidenceContext,
    RuntimeLockSnapshot,
    load_runtime_lock_snapshot,
)
from bench.model_asset_preflight import (
    DEFAULT_LOCK_PATH,
    load_asset_lock_snapshot,
)


class _Scalar:
    def __init__(self, value: bool) -> None:
        self._value = value

    def item(self) -> bool:
        return self._value


class _Tensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (1, 16, 12, 128),
        dtype: object = "bfloat16",
        device: str = "cuda:0",
        finite: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="cuda", index=0)
        if device == "cpu":
            self.device = types.SimpleNamespace(type="cpu", index=None)
        self._finite = finite


class _FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self, output: _Tensor | None = None) -> None:
        self.output = output or _Tensor()
        self.load = mock.Mock(side_effect=AssertionError("torch.load must not run"))
        self.cuda = types.SimpleNamespace(synchronize=mock.Mock())

    def full(self, shape, _value, *, device, dtype):
        del device
        return _Tensor(shape=tuple(shape), dtype=dtype)

    def isfinite(self, value):
        return types.SimpleNamespace(all=lambda: _Scalar(value._finite))


def _attention_module(*, fa3: bool, fa2: bool, output: _Tensor | None = None):
    result = output or _Tensor()
    fa3_original = mock.Mock(return_value=(result, "ignored"))
    fa2_original = mock.Mock(return_value=result)
    module = types.SimpleNamespace(
        FLASH_ATTN_3_AVAILABLE=fa3,
        FLASH_ATTN_2_AVAILABLE=fa2,
        flash_attn_interface=types.SimpleNamespace(
            flash_attn_varlen_func=fa3_original
        ),
        flash_attn=types.SimpleNamespace(flash_attn_varlen_func=fa2_original),
    )

    def attention(q, k, v, **_kwargs):
        if module.FLASH_ATTN_3_AVAILABLE:
            return module.flash_attn_interface.flash_attn_varlen_func(
                q=q, k=k, v=v
            )[0]
        if module.FLASH_ATTN_2_AVAILABLE:
            return module.flash_attn.flash_attn_varlen_func(q=q, k=k, v=v)
        raise AssertionError("no backend")

    module.attention = attention
    return module, fa3_original, fa2_original


class CFAttentionProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_snapshot = load_runtime_lock_snapshot(DEFAULT_RUNTIME_LOCK_PATH)
        self.asset_snapshot = load_asset_lock_snapshot(DEFAULT_LOCK_PATH)

    def _successful_patches(self, module, torch, *, loaded_facts=None):
        source = b"# verified attention source\n"
        unbound_snapshot = self.unbound_candidate_snapshot()
        if loaded_facts is None:
            loaded_facts = {
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
                "critical_modules": [],
                "loaded_libraries_manifest_sha256": "a" * 64,
                "loaded_library_count": 1,
                "loaded_libraries": [
                    {
                        "scope": "system",
                        "identity": "libdemo.so",
                        "size": 1,
                        "sha256": "b" * 64,
                    }
                ],
            }
        return (
            mock.patch.multiple(
                "bench.cf_attention_probe",
                CF1_RUNTIME_LOCK_SHA256=unbound_snapshot.sha256,
                load_runtime_lock_snapshot=mock.Mock(
                    return_value=unbound_snapshot
                ),
            ),
            mock.patch(
                "bench.cf_attention_probe.CF1_ASSET_LOCK_SHA256",
                self.asset_snapshot.sha256,
            ),
            mock.patch(
                "bench.cf_attention_probe.verify_model_source_snapshot",
                return_value={
                    "ready": True,
                    "source": {
                        "status": "verified",
                        "observed_commit": "8db419e341e5fc52542c0b2c4542728420ddfb4a",
                    },
                },
            ),
            mock.patch(
                "bench.cf_attention_probe._read_verified_attention_source",
                return_value=(source, hashlib.sha256(source).hexdigest()),
            ),
            mock.patch("bench.cf_attention_probe._load_torch", return_value=torch),
            mock.patch(
                "bench.cf_attention_probe._validate_loaded_runtime",
                return_value=None,
            ),
            mock.patch(
                "bench.cf_attention_probe._execute_attention_source",
                return_value=module,
            ),
            mock.patch(
                "bench.cf_attention_probe._capture_loaded_facts",
                return_value=loaded_facts,
            ),
            mock.patch(
                "bench.cf_attention_probe._require_isolated_import_context",
            ),
            mock.patch(
                "bench.cf_attention_probe._validate_attention_origins",
            ),
            mock.patch(
                "bench.cf_attention_probe._probe_identity_roots",
                return_value=(
                    Path("/environment"),
                    Path("/stdlib"),
                    (Path("/environment/site-packages"),),
                ),
            ),
        )

    def unbound_candidate_snapshot(self) -> RuntimeLockSnapshot:
        value = self.runtime_snapshot.parsed()
        value["status"] = "candidate"
        value["evidence_lock_sha256"] = None
        value["unresolved"] = [
            "runtime evidence lock",
            "executed attention probe review",
        ]
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return RuntimeLockSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def bound_candidate_snapshot(self) -> RuntimeLockSnapshot:
        value = self.runtime_snapshot.parsed()
        value["evidence_lock_sha256"] = "e" * 64
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return RuntimeLockSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def test_lock_drift_refuses_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "runtime.json"
            lock.write_bytes(self.runtime_snapshot.encoded + b" ")
            with mock.patch(
                "bench.cf_attention_probe._load_torch",
                side_effect=AssertionError("Torch import must not run"),
            ) as load_torch:
                report = attention_probe_report(lock_path=lock)

        load_torch.assert_not_called()
        self.assertFalse(report["probe_succeeded"])
        self.assertFalse(report["authorizes_boot"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["failure"], "runtime lock digest changed")

    def test_default_candidate_probe_refuses_unbound_evidence_before_torch(self) -> None:
        unbound_snapshot = self.unbound_candidate_snapshot()
        with mock.patch(
            "bench.cf_attention_probe.CF1_RUNTIME_LOCK_SHA256",
            unbound_snapshot.sha256,
        ), mock.patch(
            "bench.cf_attention_probe.load_runtime_lock_snapshot",
            return_value=unbound_snapshot,
        ), mock.patch(
            "bench.cf_attention_probe._load_torch",
            side_effect=AssertionError("Torch import must not run"),
        ) as load_torch:
            report = attention_probe_report()

        load_torch.assert_not_called()
        self.assertFalse(report["probe_succeeded"])
        self.assertEqual(report["failure"], "runtime evidence lock is not bound")

    def test_source_mismatch_refuses_before_torch_import(self) -> None:
        unbound_snapshot = self.unbound_candidate_snapshot()
        with mock.patch(
            "bench.cf_attention_probe.CF1_RUNTIME_LOCK_SHA256",
            unbound_snapshot.sha256,
        ), mock.patch(
            "bench.cf_attention_probe.load_runtime_lock_snapshot",
            return_value=unbound_snapshot,
        ), mock.patch(
            "bench.cf_attention_probe.CF1_ASSET_LOCK_SHA256",
            self.asset_snapshot.sha256,
        ), mock.patch(
            "bench.cf_attention_probe.verify_model_source_snapshot",
            return_value={"ready": False, "source": {"status": "mismatch"}},
        ), mock.patch(
            "bench.cf_attention_probe._require_isolated_import_context",
        ), mock.patch(
            "bench.cf_attention_probe._load_torch",
            side_effect=AssertionError("Torch import must not run"),
        ) as load_torch:
            report = attention_probe_report(allow_unbound_evidence_capture=True)

        load_torch.assert_not_called()
        self.assertFalse(report["probe_succeeded"])
        self.assertEqual(report["failure"], "verified source checkout is required")

    def test_attention_loader_bypasses_wan_package_initializers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wan = root / "wan"
            modules = wan / "modules"
            modules.mkdir(parents=True)
            (wan / "__init__.py").write_text(
                "raise AssertionError('wan initializer ran')\n",
                encoding="utf-8",
            )
            (modules / "__init__.py").write_text(
                "raise AssertionError('modules initializer ran')\n",
                encoding="utf-8",
            )
            source = b"MARKER = 'loaded-directly'\n"

            loaded = _execute_attention_source(
                source,
                modules / "attention.py",
                module_name="cf1_probe_test_attention",
            )

        self.assertEqual(loaded.MARKER, "loaded-directly")

    def test_import_context_requires_isolated_no_site_and_no_preloads(self) -> None:
        with mock.patch(
            "bench.cf_attention_probe.sys.flags",
            types.SimpleNamespace(isolated=0, no_site=0),
        ), self.assertRaisesRegex(AttentionProbeError, "isolated no-site"):
            _require_isolated_import_context()

    def test_unbound_identity_roots_are_explicit_absolute_and_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            stdlib = root / "stdlib"
            distribution = environment / "site-packages"
            outside = root / "outside-site-packages"
            distribution.mkdir(parents=True)
            stdlib.mkdir()
            outside.mkdir()

            resolved = _probe_identity_roots(
                None,
                unbound_environment_root=environment,
                unbound_stdlib_root=stdlib,
                unbound_distribution_paths=(distribution,),
            )
            self.assertEqual(
                resolved,
                (environment.resolve(), stdlib.resolve(), (distribution.resolve(),)),
            )

            with self.assertRaisesRegex(AttentionProbeError, "environment root"):
                _probe_identity_roots(
                    None,
                    unbound_environment_root=None,
                    unbound_stdlib_root=stdlib,
                    unbound_distribution_paths=(distribution,),
                )
            with self.assertRaisesRegex(AttentionProbeError, "identity root"):
                _probe_identity_roots(
                    None,
                    unbound_environment_root=Path("relative-environment"),
                    unbound_stdlib_root=stdlib,
                    unbound_distribution_paths=(distribution,),
                )
            with self.assertRaisesRegex(AttentionProbeError, "escapes"):
                _probe_identity_roots(
                    None,
                    unbound_environment_root=environment,
                    unbound_stdlib_root=stdlib,
                    unbound_distribution_paths=(outside,),
                )

        with mock.patch(
            "bench.cf_attention_probe.sys.flags",
            types.SimpleNamespace(isolated=1, no_site=1),
        ), mock.patch.dict(
            "bench.cf_attention_probe.sys.modules",
            {"torch": object()},
        ), self.assertRaisesRegex(AttentionProbeError, "loaded before"):
            _require_isolated_import_context()

    def test_loaded_library_inventory_ignores_deleted_dev_zero_mappings_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            stdlib = root / "stdlib"
            environment.mkdir()
            stdlib.mkdir()
            mapped = root / "libdemo.so"
            mapped.write_bytes(b"ELF demo")
            maps = root / "maps"
            maps.write_text(
                "1000-2000 rw-s 00000000 00:01 1 /dev/zero (deleted)\n"
                f"2000-3000 r-xp 00000000 00:01 2 {mapped}\n",
                encoding="utf-8",
            )

            digest, count, rows = _loaded_library_identity(
                environment_root=environment,
                stdlib_root=stdlib,
                maps_path=maps,
            )

            self.assertEqual(len(digest), 64)
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]["identity"], "libdemo.so")

            maps.write_text(
                f"2000-3000 r-xp 00000000 00:01 2 {mapped} (deleted)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AttentionProbeError, "deleted"):
                _loaded_library_identity(
                    environment_root=environment,
                    stdlib_root=stdlib,
                    maps_path=maps,
                )

    def test_loaded_library_inventory_uses_explicit_environment_and_stdlib_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "reviewed-environment"
            stdlib = root / "reviewed-stdlib"
            environment.mkdir()
            stdlib.mkdir()
            environment_library = environment / "libtorch_cuda.so"
            stdlib_library = stdlib / "libpython3.12.so"
            environment_library.write_bytes(b"environment library")
            stdlib_library.write_bytes(b"stdlib library")
            maps = root / "maps"
            maps.write_text(
                f"1000-2000 r-xp 00000000 00:01 1 {environment_library}\n"
                f"2000-3000 r-xp 00000000 00:01 2 {stdlib_library}\n",
                encoding="utf-8",
            )

            with mock.patch("bench.cf_attention_probe.sys.prefix", "/wrong-prefix"):
                _digest, count, rows = _loaded_library_identity(
                    environment_root=environment,
                    stdlib_root=stdlib,
                    maps_path=maps,
                )

        self.assertEqual(count, 2)
        self.assertEqual(
            {(row["scope"], row["identity"]) for row in rows},
            {
                ("environment", "libtorch_cuda.so"),
                ("stdlib", "libpython3.12.so"),
            },
        )

    def test_fa3_has_priority_and_delegates_exactly_once(self) -> None:
        module, fa3, fa2 = _attention_module(fa3=True, fa2=True)
        report = _execute_attention_kernel(_FakeTorch(), module, device="cuda:0")

        self.assertEqual(report["attention_backend"], "flash-attention-3")
        self.assertEqual(
            report["executed_callable"],
            "flash_attn_interface.flash_attn_varlen_func",
        )
        self.assertEqual(report["call_count"], 1)
        fa3.assert_called_once()
        fa2.assert_not_called()
        self.assertIs(module.flash_attn_interface.flash_attn_varlen_func, fa3)
        self.assertIs(module.flash_attn.flash_attn_varlen_func, fa2)

    def test_fa2_executes_and_never_calls_torch_load(self) -> None:
        module, fa3, fa2 = _attention_module(fa3=False, fa2=True)
        torch = _FakeTorch()
        report = _execute_attention_kernel(torch, module, device="cuda:0")

        self.assertEqual(report["attention_backend"], "flash-attention-2")
        self.assertEqual(
            report["executed_callable"],
            "flash_attn.flash_attn_varlen_func",
        )
        fa3.assert_not_called()
        fa2.assert_called_once()
        torch.load.assert_not_called()

    def test_backend_flags_and_output_contract_fail_closed(self) -> None:
        invalid_modules = (
            _attention_module(fa3=False, fa2=False)[0],
            types.SimpleNamespace(
                FLASH_ATTN_3_AVAILABLE=1,
                FLASH_ATTN_2_AVAILABLE=True,
            ),
        )
        for module in invalid_modules:
            with self.subTest(module=module), self.assertRaises(AttentionProbeError):
                _execute_attention_kernel(_FakeTorch(), module, device="cuda:0")

        for output in (
            _Tensor(shape=(1, 1, 1, 1)),
            _Tensor(dtype="float16"),
            _Tensor(device="cpu"),
            _Tensor(finite=False),
        ):
            module, _fa3, _fa2 = _attention_module(
                fa3=False,
                fa2=True,
                output=output,
            )
            with self.subTest(output=output), self.assertRaises(AttentionProbeError):
                _execute_attention_kernel(
                    _FakeTorch(output=output), module, device="cuda:0"
                )

    def test_candidate_probe_executes_but_never_authorizes_boot(self) -> None:
        module, _fa3, _fa2 = _attention_module(fa3=False, fa2=True)
        torch = _FakeTorch()
        patches = self._successful_patches(module, torch)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
            report = attention_probe_report(allow_unbound_evidence_capture=True)

        self.assertTrue(report["probe_succeeded"])
        self.assertTrue(report["gpu_execution_performed"])
        self.assertFalse(report["authorizes_boot"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["attention_backend"], "flash-attention-2")
        self.assertEqual(len(report["probe_identity_sha256"]), 64)

    def test_success_binds_loaded_abi_and_torch_first_import_order(self) -> None:
        module, _fa3, _fa2 = _attention_module(fa3=False, fa2=True)
        torch = _FakeTorch()
        loaded_facts = {
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
            "critical_modules": [],
            "loaded_libraries_manifest_sha256": "a" * 64,
            "loaded_library_count": 1,
            "loaded_libraries": [
                {
                    "scope": "system",
                    "identity": "libdemo.so",
                    "size": 1,
                    "sha256": "b" * 64,
                }
            ],
        }
        patches = self._successful_patches(
            module,
            torch,
            loaded_facts=loaded_facts,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7] as capture, patches[8], patches[9], patches[10]:
            report = attention_probe_report(allow_unbound_evidence_capture=True)

        capture.assert_called_once_with(
            torch,
            module,
            attention_backend="flash-attention-2",
            environment_root=Path("/environment"),
            stdlib_root=Path("/stdlib"),
        )
        self.assertEqual(report["loaded_facts"], loaded_facts)

    def test_bound_probe_verifies_static_then_torch_then_loaded_evidence(self) -> None:
        runtime_snapshot = self.bound_candidate_snapshot()
        lock = runtime_snapshot.parsed()
        module, _fa3, _fa2 = _attention_module(fa3=False, fa2=True)
        torch = _FakeTorch()
        static_identity = RuntimeEvidenceStaticIdentity(
            runtime_id=lock["runtime_id"],
            runtime_evidence_sha256=lock["evidence_lock_sha256"],
            static_environment_sha256="a" * 64,
            image_manifest_digest=lock["image"]["digest"],
            python_executable_sha256="b" * 64,
            stdlib_manifest_sha256="c" * 64,
            environment_tree_sha256="d" * 64,
            package_inventory_sha256="f" * 64,
        )
        context = RuntimeEvidenceContext(
            snapshot=mock.Mock(sha256=lock["evidence_lock_sha256"]),
            static_identity=static_identity,
            observed_oci={"manifest_digest": lock["image"]["digest"]},
            python_executable=Path("/python"),
            stdlib_root=Path("/stdlib"),
            environment_root=Path("/environment"),
            distribution_paths=(Path("/environment/site-packages"),),
            wheelhouse=Path("/wheelhouse"),
        )
        full_identity = RuntimeEvidenceIdentity(
            runtime_id=lock["runtime_id"],
            runtime_evidence_sha256=lock["evidence_lock_sha256"],
            environment_sha256="1" * 64,
            image_manifest_digest=lock["image"]["digest"],
            python_executable_sha256="2" * 64,
            stdlib_manifest_sha256="3" * 64,
            environment_tree_sha256="4" * 64,
            native_identity_sha256="5" * 64,
        )
        events: list[str] = []
        source = b"# verified attention source\n"

        def static(*_args, **_kwargs):
            events.append("static")
            return context

        def load_torch(*_args, **_kwargs):
            events.append("torch")
            return torch

        def full(*_args, **_kwargs):
            events.append("full")
            return full_identity

        with mock.patch(
            "bench.cf_attention_probe.load_runtime_lock_snapshot",
            return_value=runtime_snapshot,
        ), mock.patch(
            "bench.cf_attention_probe.CF1_RUNTIME_LOCK_SHA256",
            runtime_snapshot.sha256,
        ), mock.patch(
            "bench.cf_attention_probe.CF1_ASSET_LOCK_SHA256",
            self.asset_snapshot.sha256,
        ), mock.patch(
            "bench.cf_attention_probe.verify_bound_static_runtime_evidence",
            side_effect=static,
        ), mock.patch(
            "bench.cf_attention_probe.verify_model_source_snapshot",
            return_value={
                "ready": True,
                "source": {
                    "status": "verified",
                    "observed_commit": "8db419e341e5fc52542c0b2c4542728420ddfb4a",
                },
            },
        ), mock.patch(
            "bench.cf_attention_probe._read_verified_attention_source",
            return_value=(source, hashlib.sha256(source).hexdigest()),
        ), mock.patch(
            "bench.cf_attention_probe._require_isolated_import_context",
        ), mock.patch(
            "bench.cf_attention_probe._load_torch",
            side_effect=load_torch,
        ), mock.patch(
            "bench.cf_attention_probe._validate_loaded_runtime",
            return_value=None,
        ), mock.patch(
            "bench.cf_attention_probe._execute_attention_source",
            return_value=module,
        ), mock.patch(
            "bench.cf_attention_probe._validate_attention_origins",
        ), mock.patch(
            "bench.cf_attention_probe._capture_loaded_facts",
            return_value={
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
                "critical_modules": [],
                "loaded_libraries_manifest_sha256": "6" * 64,
                "loaded_library_count": 1,
                "loaded_libraries": [
                    {
                        "scope": "system",
                        "identity": "libdemo.so",
                        "size": 1,
                        "sha256": "7" * 64,
                    }
                ],
            },
        ), mock.patch(
            "bench.cf_attention_probe.verify_runtime_evidence",
            side_effect=full,
        ):
            report = attention_probe_report()

        self.assertEqual(events, ["static", "torch", "full"])
        self.assertTrue(report["probe_succeeded"])
        self.assertEqual(report["runtime_evidence_sha256"], "e" * 64)
        self.assertEqual(report["runtime_environment_sha256"], "1" * 64)
        self.assertEqual(report["native_identity_sha256"], "5" * 64)

    def test_kernel_failure_is_sanitized_and_non_authorizing(self) -> None:
        module, _fa3, fa2 = _attention_module(fa3=False, fa2=True)
        fa2.side_effect = RuntimeError("sensitive kernel detail")
        patches = self._successful_patches(module, _FakeTorch())
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
            report = attention_probe_report(allow_unbound_evidence_capture=True)

        self.assertFalse(report["probe_succeeded"])
        self.assertFalse(report["authorizes_boot"])
        self.assertEqual(report["failure"], "unexpected probe error: RuntimeError")
        self.assertNotIn("sensitive", json.dumps(report))

    def test_cli_success_is_non_authorizing(self) -> None:
        payload = {
            "probe_succeeded": True,
            "authorizes_boot": False,
            "ready": False,
        }
        with mock.patch(
            "bench.cf_attention_probe.attention_probe_report",
            return_value=payload,
        ), mock.patch("builtins.print") as output:
            status = main([])
        self.assertEqual(status, 0)
        self.assertIn('"authorizes_boot": false', output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
