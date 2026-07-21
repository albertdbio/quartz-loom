from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.cf_runtime_evidence import RuntimeEvidenceStaticIdentity
from bench.cf_runtime_preflight import (
    CF1_TOKENIZER_SENTINEL_SHA256,
    DEFAULT_RUNTIME_LOCK_PATH,
    HostMemoryObservation,
    RuntimeLockSnapshot,
    RuntimeObservation,
    RuntimePreflightError,
    _collect_runtime_observation,
    _normalize_oci_sha256,
    _observe_gpu,
    capture_main,
    capture_runtime_environment,
    load_runtime_lock_snapshot,
    main,
    observe_host_memory,
    preflight_runtime_environment,
    preflight_current_runtime,
    runtime_preflight_report,
    runtime_capture_report,
    validate_current_host_capacity,
    validate_cf1_tokenizer_sentinel,
    validate_loaded_cuda_capacity,
    validate_runtime_lock,
)


_GIB = 1024**3


class FakeTensor:
    def __init__(
        self,
        values: list[list[int]],
        *,
        dtype: object,
        device_type: str = "cpu",
    ) -> None:
        self._values = values
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self.shape = (len(values), len(values[0]) if values else 0)

    def tolist(self) -> list[list[int]]:
        return copy.deepcopy(self._values)


class FakeInnerTokenizer:
    is_fast = True
    vocab_size = 256300
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 3

    def __len__(self) -> int:
        return 256300

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<extra_id_0>":
            return 256299
        return self.unk_token_id


class FakeTokenizerWrapper:
    def __init__(
        self,
        ids: list[list[int]],
        masks: list[list[int]],
        *,
        dtype: object,
    ) -> None:
        self.tokenizer = FakeInnerTokenizer()
        self.ids = FakeTensor(ids, dtype=dtype)
        self.masks = FakeTensor(masks, dtype=dtype)
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompts: object, **kwargs: object):
        self.calls.append((prompts, dict(kwargs)))
        return self.ids, self.masks


def sentinel_rows() -> tuple[list[list[int]], list[list[int]]]:
    prefixes = (
        (320, 4062, 273, 56209, 48150, 281, 274, 1),
        (25382, 273, 14985, 1),
        (256299, 1),
    )
    ids = [list(prefix) + [0] * (512 - len(prefix)) for prefix in prefixes]
    masks = [[1] * len(prefix) + [0] * (512 - len(prefix)) for prefix in prefixes]
    return ids, masks


class CFRuntimePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = load_runtime_lock_snapshot(DEFAULT_RUNTIME_LOCK_PATH)
        self.lock = self.snapshot.parsed()

    def frozen_snapshot(self) -> RuntimeLockSnapshot:
        value = copy.deepcopy(self.lock)
        value["status"] = "frozen"
        value["evidence_lock_sha256"] = "e" * 64
        value["image"]["digest"] = "sha256:" + "a" * 64
        value["target"]["python_version"] = "3.12.99"
        value["target"]["python_build"] = ["main", "Jul 20 2026"]
        value["target"]["nvidia_driver_version"] = "575.57.08"
        value["target"]["attention_backend"] = "flash-attention-2"
        for package in value["packages"]:
            package["evidence"] = "observed"
        value["unresolved"] = []
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return RuntimeLockSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def observation(self, snapshot: RuntimeLockSnapshot) -> RuntimeObservation:
        lock = snapshot.parsed()
        return RuntimeObservation(
            image_digest=lock["image"]["digest"],
            platform_system=lock["target"]["platform_system"],
            machine=lock["target"]["machine"],
            python_implementation=lock["target"]["python_implementation"],
            python_version=f"{lock['target']['python_major_minor']}.99",
            python_build=tuple(lock["target"]["python_build"]),
            package_versions={
                package["distribution"]: package["version"]
                for package in lock["packages"]
            },
            mem_available_bytes=128 * _GIB,
            cgroup_state="bounded",
            cgroup_headroom_bytes=96 * _GIB,
            swap_total_bytes=0,
            nvidia_driver_version=lock["target"]["nvidia_driver_version"],
            gpu_name=lock["target"]["gpu_name"],
            gpu_compute_capability=tuple(
                lock["target"]["compute_capability"]
            ),
            gpu_total_bytes=80_000_000_000,
            gpu_free_bytes=64 * _GIB,
        )

    def static_identity(
        self, snapshot: RuntimeLockSnapshot
    ) -> RuntimeEvidenceStaticIdentity:
        lock = snapshot.parsed()
        return RuntimeEvidenceStaticIdentity(
            runtime_id=lock["runtime_id"],
            runtime_evidence_sha256=lock["evidence_lock_sha256"],
            static_environment_sha256="d" * 64,
            image_manifest_digest=lock["image"]["digest"],
            python_executable_sha256="1" * 64,
            stdlib_manifest_sha256="2" * 64,
            environment_tree_sha256="3" * 64,
            package_inventory_sha256="4" * 64,
        )

    def capture_observation(self) -> RuntimeObservation:
        return RuntimeObservation(
            image_digest="sha256:" + "a" * 64,
            platform_system="Linux",
            machine="x86_64",
            python_implementation="CPython",
            python_version="3.12.9",
            python_build=("main", "Jul 20 2026"),
            package_versions={
                "z-last": "9.0",
                **{
                    package["distribution"]: package["version"]
                    for package in self.lock["packages"]
                },
                "a-first": "1.0",
            },
            mem_available_bytes=128 * _GIB,
            cgroup_state="bounded",
            cgroup_headroom_bytes=96 * _GIB,
            swap_total_bytes=0,
            nvidia_driver_version="575.57.08",
            gpu_name="NVIDIA H100 80GB HBM3",
            gpu_compute_capability=(9, 0),
            gpu_total_bytes=80_000_000_000,
            gpu_free_bytes=64 * _GIB,
        )

    def test_candidate_lock_is_valid_but_cannot_authorize_a_boot(self) -> None:
        candidate = copy.deepcopy(self.lock)
        candidate["status"] = "candidate"
        candidate["evidence_lock_sha256"] = None
        candidate["unresolved"] = [
            "runtime evidence lock",
            "executed attention probe review",
        ]
        validate_runtime_lock(candidate, require_frozen=False)
        self.assertEqual(candidate["schema_version"], 2)
        self.assertIsNone(candidate["evidence_lock_sha256"])
        self.assertEqual(candidate["status"], "candidate")
        with self.assertRaisesRegex(RuntimePreflightError, "not frozen"):
            validate_runtime_lock(candidate, require_frozen=True)

        missing_torch = copy.deepcopy(candidate)
        missing_torch["packages"] = [
            package
            for package in missing_torch["packages"]
            if package["distribution"] != "torch"
        ]
        with self.assertRaisesRegex(RuntimePreflightError, "package inventory"):
            validate_runtime_lock(missing_torch, require_frozen=False)

        malformed_digest = copy.deepcopy(self.lock)
        malformed_digest["image"]["digest"] = "a" * 64
        with self.assertRaisesRegex(RuntimePreflightError, "image identity"):
            validate_runtime_lock(malformed_digest, require_frozen=False)

        unsupported_backend = copy.deepcopy(self.lock)
        unsupported_backend["target"]["attention_backend"] = "torch-sdpa"
        with self.assertRaisesRegex(RuntimePreflightError, "attention backend"):
            validate_runtime_lock(unsupported_backend, require_frozen=False)
        self.assertEqual(
            _normalize_oci_sha256("a" * 64),
            "sha256:" + "a" * 64,
        )

        malformed_evidence = copy.deepcopy(self.lock)
        malformed_evidence["evidence_lock_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(RuntimePreflightError, "evidence lock"):
            validate_runtime_lock(malformed_evidence, require_frozen=False)

    def test_cli_reports_candidate_refusal_before_gpu_observation(self) -> None:
        candidate = copy.deepcopy(self.lock)
        candidate["status"] = "candidate"
        candidate["unresolved"] = ["executed attention probe review"]
        encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        candidate_sha256 = hashlib.sha256(encoded).hexdigest()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_bytes(encoded)
            with mock.patch(
                "bench.cf_runtime_preflight.CF1_RUNTIME_LOCK_SHA256",
                candidate_sha256,
            ), mock.patch(
                "bench.cf_runtime_preflight.observe_runtime_environment",
                side_effect=AssertionError("GPU observation must not run"),
            ), mock.patch(
                "bench.cf_runtime_preflight.verify_bound_static_runtime_evidence",
                side_effect=AssertionError("candidate must not read evidence"),
            ), contextlib.redirect_stdout(output):
                status = main(["--lock", str(path)])

        report = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertFalse(report["ready"])
        self.assertEqual(report["runtime_id"], "cf1-h100-cu128-v1")
        self.assertEqual(report["runtime_lock_sha256"], candidate_sha256)
        self.assertEqual(report["failure"], "runtime lock is not frozen")

    def test_frozen_capture_succeeds_but_never_authorizes_boot(self) -> None:
        observed = self.capture_observation()
        with mock.patch(
            "bench.cf_runtime_preflight._collect_runtime_observation",
            return_value=observed,
        ) as collect:
            report = runtime_capture_report()

        collect.assert_called_once_with(gpu_index=0)
        self.assertTrue(report["capture_succeeded"])
        self.assertFalse(report["authorizes_boot"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["kind"], "cf1-runtime-environment-capture")
        self.assertEqual(report["lock_status"], "frozen")
        self.assertEqual(report["lock_unresolved"], [])
        self.assertNotIn("environment_sha256", json.dumps(report))
        packages = report["observation"]["packages"]
        self.assertEqual(
            [package["distribution"] for package in packages],
            sorted(package["distribution"] for package in packages),
        )
        self.assertTrue(all(package["evidence"] == "observed" for package in packages))
        self.assertEqual(
            report["observation"]["host_memory"]["effective_headroom_bytes"],
            96 * _GIB,
        )
        self.assertEqual(
            report["observation"]["image"]["trusted_launcher_digest_assertion"],
            "sha256:" + "a" * 64,
        )

    def test_capture_reports_mismatches_and_low_capacity_without_authorizing(self) -> None:
        observed = self.capture_observation()
        observed.gpu_name = "NVIDIA A100"
        observed.gpu_compute_capability = (8, 0)
        observed.gpu_free_bytes = 1
        observed.cgroup_headroom_bytes = 1
        observed.package_versions = {"unrelated": "1.0"}
        with mock.patch(
            "bench.cf_runtime_preflight._collect_runtime_observation",
            return_value=observed,
        ):
            report = runtime_capture_report()

        self.assertTrue(report["capture_succeeded"])
        self.assertFalse(report["authorizes_boot"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["observation"]["gpu"]["name"], "NVIDIA A100")
        self.assertEqual(report["observation"]["gpu"]["free_bytes"], 1)
        self.assertEqual(
            report["observation"]["host_memory"]["effective_headroom_bytes"],
            1,
        )
        self.assertEqual(
            report["observation"]["packages"],
            [
                {
                    "distribution": "unrelated",
                    "version": "1.0",
                    "evidence": "observed",
                }
            ],
        )

    def test_capture_lock_drift_refuses_before_observation_and_sanitizes_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock.json"
            path.write_bytes(self.snapshot.encoded + b" ")
            with mock.patch(
                "bench.cf_runtime_preflight._collect_runtime_observation",
                side_effect=AssertionError("observation must not run"),
            ) as collect:
                report = runtime_capture_report(path)
        collect.assert_not_called()
        self.assertFalse(report["capture_succeeded"])
        self.assertFalse(report["authorizes_boot"])
        self.assertEqual(report["failure"], "runtime lock digest changed")

        with mock.patch(
            "bench.cf_runtime_preflight.capture_runtime_environment",
            side_effect=RuntimeError("sensitive detail"),
        ):
            report = runtime_capture_report()
        self.assertEqual(report["failure"], "unexpected capture error: RuntimeError")
        self.assertNotIn("sensitive", json.dumps(report))

    def test_capture_observer_reads_only_safe_identity_fields(self) -> None:
        distributions = (
            SimpleNamespace(metadata={"Name": "Z_Package"}, version="2.0"),
            SimpleNamespace(metadata={"Name": "a.package"}, version="1.0"),
        )
        host = HostMemoryObservation(
            mem_available_bytes=64 * _GIB,
            cgroup_state="bounded",
            cgroup_headroom_bytes=60 * _GIB,
            swap_total_bytes=8 * _GIB,
        )
        safe_digest = "a" * 64
        with mock.patch(
            "bench.cf_runtime_preflight.importlib.metadata.distributions",
            return_value=distributions,
        ), mock.patch(
            "bench.cf_runtime_preflight.observe_host_memory",
            return_value=host,
        ), mock.patch(
            "bench.cf_runtime_preflight._observe_gpu",
            return_value=(
                "NVIDIA H100 80GB HBM3",
                "575.57.08",
                (9, 0),
                80_000_000_000,
                50 * _GIB,
            ),
        ), mock.patch.dict(
            os.environ,
            {
                "CF1_RUNTIME_IMAGE_DIGEST": safe_digest,
                "TWELVELABS_API_KEY": "must-not-appear",
            },
            clear=True,
        ):
            observed = _collect_runtime_observation(gpu_index=0)

        self.assertEqual(observed.image_digest, "sha256:" + safe_digest)
        self.assertEqual(
            observed.package_versions,
            {"z-package": "2.0", "a-package": "1.0"},
        )
        self.assertNotIn("must-not-appear", repr(observed))

        with mock.patch.dict(
            os.environ,
            {"CF1_RUNTIME_IMAGE_DIGEST": "not-a-digest"},
            clear=True,
        ), self.assertRaisesRegex(RuntimePreflightError, "digest assertion"):
            _collect_runtime_observation(gpu_index=0)

    def test_observer_discovers_packages_from_explicit_path_without_site_imports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            distribution_path = Path(directory) / "site-packages"
            metadata = distribution_path / "demo_package-1.2.3.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: demo-package\nVersion: 1.2.3\n\n",
                encoding="utf-8",
            )
            host = HostMemoryObservation(
                mem_available_bytes=64 * _GIB,
                cgroup_state="bounded",
                cgroup_headroom_bytes=60 * _GIB,
                swap_total_bytes=0,
            )
            with mock.patch.object(sys, "path", []), mock.patch(
                "bench.cf_runtime_preflight.observe_host_memory",
                return_value=host,
            ), mock.patch(
                "bench.cf_runtime_preflight._observe_gpu",
                return_value=(
                    "NVIDIA H100 80GB HBM3",
                    "575.57.08",
                    (9, 0),
                    80_000_000_000,
                    50 * _GIB,
                ),
            ), mock.patch.dict(
                os.environ,
                {"CF1_RUNTIME_IMAGE_DIGEST": "a" * 64},
                clear=True,
            ):
                observed = _collect_runtime_observation(
                    gpu_index=0,
                    distribution_paths=(distribution_path,),
                )

        self.assertEqual(observed.package_versions, {"demo-package": "1.2.3"})

    def test_package_discovery_runs_in_actual_isolated_no_site_interpreter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            distribution_path = Path(directory) / "site-packages"
            metadata = distribution_path / "demo_package-1.2.3.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: demo-package\nVersion: 1.2.3\n\n",
                encoding="utf-8",
            )
            project_root = Path(__file__).resolve().parents[2]
            program = (
                "import json,sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(project_root)!r})\n"
                "from bench.cf_runtime_preflight import _collect_package_versions\n"
                "print(json.dumps(_collect_package_versions((Path("
                f"{str(distribution_path)!r}),)), sort_keys=True))\n"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", program],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {"demo-package": "1.2.3"},
        )

    def test_capture_cli_status_means_collection_only(self) -> None:
        success = {
            "capture_succeeded": True,
            "authorizes_boot": False,
            "ready": False,
        }
        output = io.StringIO()
        with mock.patch(
            "bench.cf_runtime_preflight.runtime_capture_report",
            return_value=success,
        ), contextlib.redirect_stdout(output):
            status = capture_main([])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), success)

        failure = {
            "capture_succeeded": False,
            "authorizes_boot": False,
            "ready": False,
        }
        output = io.StringIO()
        with mock.patch(
            "bench.cf_runtime_preflight.runtime_capture_report",
            return_value=failure,
        ), contextlib.redirect_stdout(output):
            status = capture_main([])
        self.assertEqual(status, 2)

    def test_lock_byte_drift_refuses_before_gpu_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock.json"
            path.write_bytes(self.snapshot.encoded + b" ")
            with mock.patch(
                "bench.cf_runtime_preflight.observe_runtime_environment",
                side_effect=AssertionError("GPU observation must not run"),
            ):
                report = runtime_preflight_report(path)
        self.assertFalse(report["ready"])
        self.assertEqual(report["failure"], "runtime lock digest changed")

    def test_unexpected_preflight_error_is_structured_without_message_leak(self) -> None:
        with mock.patch(
            "bench.cf_runtime_preflight.preflight_current_runtime",
            side_effect=RuntimeError("sensitive internal detail"),
        ):
            report = runtime_preflight_report()

        self.assertFalse(report["ready"])
        self.assertEqual(
            report["failure"],
            "unexpected preflight error: RuntimeError",
        )
        self.assertNotIn("sensitive", json.dumps(report))

    def test_real_cli_wrapper_returns_structured_environment_refusal(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "cf-runtime-preflight")],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(report["ready"])
        self.assertIsInstance(report["failure"], str)
        self.assertTrue(report["failure"])
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_exact_frozen_environment_passes_and_binds_lock_identity(self) -> None:
        snapshot = self.frozen_snapshot()

        identity = preflight_runtime_environment(
            snapshot,
            self.observation(snapshot),
            static_identity=self.static_identity(snapshot),
        )

        self.assertEqual(identity.runtime_lock_sha256, snapshot.sha256)
        self.assertEqual(
            identity.runtime_evidence_sha256,
            snapshot.parsed()["evidence_lock_sha256"],
        )
        self.assertEqual(identity.static_environment_sha256, "d" * 64)
        self.assertEqual(identity.effective_host_headroom_bytes, 96 * _GIB)
        self.assertEqual(identity.gpu_free_bytes, 64 * _GIB)
        self.assertEqual(len(identity.environment_sha256), 64)

    def test_environment_comparison_requires_static_evidence_identity(self) -> None:
        snapshot = self.frozen_snapshot()
        with self.assertRaisesRegex(RuntimePreflightError, "static runtime evidence"):
            preflight_runtime_environment(snapshot, self.observation(snapshot))

    def test_static_evidence_is_verified_before_gpu_observation(self) -> None:
        snapshot = self.frozen_snapshot()
        static = self.static_identity(snapshot)
        context = SimpleNamespace(
            static_identity=static,
            distribution_paths=(Path("/trusted/site-packages"),),
        )
        events: list[str] = []

        def verify(*_args, **_kwargs):
            events.append("static")
            return context

        def observe(*_args, **_kwargs):
            events.append("gpu")
            return self.observation(snapshot)

        with mock.patch(
            "bench.cf_runtime_preflight.verify_bound_static_runtime_evidence",
            side_effect=verify,
        ), mock.patch(
            "bench.cf_runtime_preflight.observe_runtime_environment",
            side_effect=observe,
        ):
            identity = preflight_current_runtime(snapshot, gpu_index=0)

        self.assertEqual(events, ["static", "gpu"])
        self.assertEqual(identity.runtime_evidence_sha256, static.runtime_evidence_sha256)

    def test_package_python_platform_and_gpu_identity_are_exact(self) -> None:
        snapshot = self.frozen_snapshot()
        mutations = (
            ("python", lambda value: setattr(value, "python_version", "3.11.99")),
            ("platform", lambda value: setattr(value, "platform_system", "Darwin")),
            ("machine", lambda value: setattr(value, "machine", "aarch64")),
            ("gpu", lambda value: setattr(value, "gpu_name", "NVIDIA A100")),
            (
                "capability",
                lambda value: setattr(value, "gpu_compute_capability", (8, 0)),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                observed = self.observation(snapshot)
                mutate(observed)
                with self.assertRaises(RuntimePreflightError):
                    preflight_runtime_environment(
                        snapshot,
                        observed,
                        static_identity=self.static_identity(snapshot),
                    )

        observed = self.observation(snapshot)
        package = next(iter(observed.package_versions))
        observed.package_versions[package] = "0.invalid"
        with self.assertRaisesRegex(RuntimePreflightError, package):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

        observed = self.observation(snapshot)
        del observed.package_versions[package]
        with self.assertRaisesRegex(RuntimePreflightError, "package inventory"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

        baseline = self.observation(snapshot)
        expected_digest = preflight_runtime_environment(
            snapshot,
            baseline,
            static_identity=self.static_identity(snapshot),
        ).environment_sha256
        reordered = self.observation(snapshot)
        reordered.package_versions = dict(
            reversed(tuple(reordered.package_versions.items()))
        )
        self.assertEqual(
            preflight_runtime_environment(
                snapshot,
                reordered,
                static_identity=self.static_identity(snapshot),
            ).environment_sha256,
            expected_digest,
        )
        with_extra = self.observation(snapshot)
        with_extra.package_versions["unrelated-package"] = "1.0"
        with self.assertRaisesRegex(RuntimePreflightError, "package inventory"):
            preflight_runtime_environment(
                snapshot,
                with_extra,
                static_identity=self.static_identity(snapshot),
            )

        for label, field, value in (
            ("image", "image_digest", "b" * 64),
            ("Python build", "python_build", ("other", "build")),
            ("driver", "nvidia_driver_version", "999.0"),
        ):
            with self.subTest(label=label):
                observed = self.observation(snapshot)
                setattr(observed, field, value)
                with self.assertRaises(RuntimePreflightError):
                    preflight_runtime_environment(
                        snapshot,
                        observed,
                        static_identity=self.static_identity(snapshot),
                    )

    def test_host_and_gpu_thresholds_fail_one_byte_below(self) -> None:
        snapshot = self.frozen_snapshot()
        lock = snapshot.parsed()
        host_minimum = lock["capacity"]["minimum_host_headroom_bytes"]
        gpu_minimum = lock["capacity"]["minimum_gpu_free_bytes"]

        observed = self.observation(snapshot)
        observed.cgroup_headroom_bytes = host_minimum - 1
        observed.swap_total_bytes = 10_000 * _GIB
        with self.assertRaisesRegex(RuntimePreflightError, "host headroom"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

        observed = self.observation(snapshot)
        observed.gpu_free_bytes = gpu_minimum - 1
        with self.assertRaisesRegex(RuntimePreflightError, "GPU free"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

        observed = self.observation(snapshot)
        observed.gpu_total_bytes = (
            lock["capacity"]["minimum_gpu_total_bytes"] - 1
        )
        with self.assertRaisesRegex(RuntimePreflightError, "GPU total"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

        observed = self.observation(snapshot)
        observed.gpu_free_bytes = observed.gpu_total_bytes + 1
        with self.assertRaisesRegex(RuntimePreflightError, "GPU memory accounting"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

    def test_post_import_host_capacity_is_rechecked(self) -> None:
        snapshot = self.frozen_snapshot()
        minimum = snapshot.parsed()["capacity"][
            "minimum_host_headroom_bytes"
        ]
        with mock.patch(
            "bench.cf_runtime_preflight.observe_host_memory",
            return_value=HostMemoryObservation(
                mem_available_bytes=128 * _GIB,
                cgroup_state="bounded",
                cgroup_headroom_bytes=minimum,
                swap_total_bytes=1000 * _GIB,
            ),
        ):
            self.assertEqual(
                validate_current_host_capacity(snapshot),
                minimum,
            )
        with mock.patch(
            "bench.cf_runtime_preflight.observe_host_memory",
            return_value=HostMemoryObservation(
                mem_available_bytes=128 * _GIB,
                cgroup_state="bounded",
                cgroup_headroom_bytes=minimum - 1,
                swap_total_bytes=1000 * _GIB,
            ),
        ), self.assertRaisesRegex(RuntimePreflightError, "host headroom"):
            validate_current_host_capacity(snapshot)

    def test_loaded_cuda_capacity_and_backend_are_exact(self) -> None:
        snapshot = self.frozen_snapshot()
        lock = snapshot.parsed()
        torch_version = next(
            package["version"]
            for package in lock["packages"]
            if package["distribution"] == "torch"
        )

        def loaded_torch(**changes):
            values = {
                "torch_version": torch_version,
                "cuda_runtime": lock["target"]["cuda_runtime"],
                "name": lock["target"]["gpu_name"],
                "capability": tuple(lock["target"]["compute_capability"]),
                "free": lock["capacity"]["minimum_gpu_free_bytes"],
                "total": lock["capacity"]["minimum_gpu_total_bytes"],
            }
            values.update(changes)
            cuda = SimpleNamespace(
                get_device_name=lambda _device: values["name"],
                get_device_capability=lambda _device: values["capability"],
                mem_get_info=lambda _device: (values["free"], values["total"]),
            )
            return SimpleNamespace(
                __version__=values["torch_version"],
                version=SimpleNamespace(cuda=values["cuda_runtime"]),
                cuda=cuda,
            )

        self.assertEqual(
            validate_loaded_cuda_capacity(
                loaded_torch(),
                snapshot,
                device="cuda:0",
                attention_backend=lock["target"]["attention_backend"],
            ),
            (
                lock["capacity"]["minimum_gpu_free_bytes"],
                lock["capacity"]["minimum_gpu_total_bytes"],
            ),
        )
        mutations = (
            (
                "Torch",
                {"torch_version": "0.invalid"},
                lock["target"]["attention_backend"],
            ),
            (
                "CUDA runtime",
                {"cuda_runtime": "0.0"},
                lock["target"]["attention_backend"],
            ),
            (
                "device name",
                {"name": "NVIDIA A100"},
                lock["target"]["attention_backend"],
            ),
            ("capability", {"capability": (8, 0)}, lock["target"]["attention_backend"]),
            (
                "total memory",
                {"total": lock["capacity"]["minimum_gpu_total_bytes"] - 1},
                lock["target"]["attention_backend"],
            ),
            (
                "free memory",
                {"free": lock["capacity"]["minimum_gpu_free_bytes"] - 1},
                lock["target"]["attention_backend"],
            ),
            (
                "memory accounting",
                {"free": lock["capacity"]["minimum_gpu_total_bytes"] + 1},
                lock["target"]["attention_backend"],
            ),
            ("attention backend", {}, "flash-attention-3"),
        )
        for label, changes, backend in mutations:
            with self.subTest(label=label), self.assertRaises(
                RuntimePreflightError
            ):
                validate_loaded_cuda_capacity(
                    loaded_torch(**changes),
                    snapshot,
                    device="cuda:0",
                    attention_backend=backend,
                )

    def test_nvidia_smi_parser_binds_driver_capability_and_bytes(self) -> None:
        completed = SimpleNamespace(
            stdout="NVIDIA H100 80GB HBM3, 575.57.08, 9.0, 81559, 79000\n"
        )
        with mock.patch(
            "bench.cf_runtime_preflight.subprocess.run",
            return_value=completed,
        ) as run:
            observed = _observe_gpu(0)

        self.assertEqual(
            observed,
            (
                "NVIDIA H100 80GB HBM3",
                "575.57.08",
                (9, 0),
                81559 * 1024**2,
                79000 * 1024**2,
            ),
        )
        command = run.call_args.args[0]
        self.assertFalse(any(item.startswith("--id=") for item in command))
        self.assertIn("driver_version", command[1])

        with mock.patch(
            "bench.cf_runtime_preflight.subprocess.run"
        ) as not_called, self.assertRaisesRegex(
            RuntimePreflightError, "exactly one GPU"
        ):
            _observe_gpu(1)
        not_called.assert_not_called()

        with mock.patch(
            "bench.cf_runtime_preflight.subprocess.run",
            return_value=SimpleNamespace(stdout="bad,row\n"),
        ), self.assertRaisesRegex(RuntimePreflightError, "ambiguous"):
            _observe_gpu(0)

        with mock.patch(
            "bench.cf_runtime_preflight.subprocess.run",
            return_value=SimpleNamespace(
                stdout=(
                    "NVIDIA H100 80GB HBM3, 575.57.08, 9.0, 81559, 99999\n"
                )
            ),
        ), self.assertRaisesRegex(RuntimePreflightError, "memory accounting"):
            _observe_gpu(0)

    def test_lower_cgroup_headroom_wins_and_ambiguous_cgroup_fails(self) -> None:
        snapshot = self.frozen_snapshot()
        observed = self.observation(snapshot)
        observed.mem_available_bytes = 256 * _GIB
        observed.cgroup_headroom_bytes = 60 * _GIB
        identity = preflight_runtime_environment(
            snapshot,
            observed,
            static_identity=self.static_identity(snapshot),
        )
        self.assertEqual(identity.effective_host_headroom_bytes, 60 * _GIB)

        observed = self.observation(snapshot)
        observed.cgroup_state = "ambiguous"
        observed.cgroup_headroom_bytes = None
        with self.assertRaisesRegex(RuntimePreflightError, "cgroup"):
            preflight_runtime_environment(
                snapshot,
                observed,
                static_identity=self.static_identity(snapshot),
            )

    def test_v2_host_observation_uses_limit_minus_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            cgroup = root / "cgroup"
            proc.mkdir()
            cgroup.mkdir()
            (proc / "self").mkdir()
            (proc / "self" / "cgroup").write_text(
                "0::/\n", encoding="utf-8"
            )
            (proc / "meminfo").write_text(
                "MemTotal:       134217728 kB\n"
                "MemAvailable:   100663296 kB\n"
                "SwapTotal:       8388608 kB\n",
                encoding="utf-8",
            )
            (cgroup / "cgroup.controllers").write_text(
                "cpu io memory\n", encoding="utf-8"
            )
            (cgroup / "memory.max").write_text(
                str(80 * _GIB), encoding="utf-8"
            )
            (cgroup / "memory.current").write_text(
                str(20 * _GIB), encoding="utf-8"
            )

            observed = observe_host_memory(proc_root=proc, cgroup_root=cgroup)

        self.assertEqual(observed.mem_available_bytes, 96 * _GIB)
        self.assertEqual(observed.swap_total_bytes, 8 * _GIB)
        self.assertEqual(observed.cgroup_state, "bounded")
        self.assertEqual(observed.cgroup_headroom_bytes, 60 * _GIB)

    def test_v2_uses_membership_and_tightest_ancestor_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            cgroup = root / "cgroup"
            worker = cgroup / "tenant" / "worker"
            (proc / "self").mkdir(parents=True)
            worker.mkdir(parents=True)
            (proc / "meminfo").write_text(
                "MemTotal:       268435456 kB\n"
                "MemAvailable:   201326592 kB\n"
                "SwapTotal:              0 kB\n",
                encoding="utf-8",
            )
            (proc / "self" / "cgroup").write_text(
                "0::/tenant/worker\n", encoding="utf-8"
            )
            (cgroup / "cgroup.controllers").write_text(
                "cpu io memory\n", encoding="utf-8"
            )
            for path, maximum, current in (
                (cgroup, 160 * _GIB, 20 * _GIB),
                (cgroup / "tenant", 80 * _GIB, 30 * _GIB),
                (worker, "max", 10 * _GIB),
            ):
                (path / "memory.max").write_text(
                    str(maximum), encoding="utf-8"
                )
                (path / "memory.current").write_text(
                    str(current), encoding="utf-8"
                )

            observed = observe_host_memory(proc_root=proc, cgroup_root=cgroup)

        self.assertEqual(observed.cgroup_state, "bounded")
        self.assertEqual(observed.cgroup_headroom_bytes, 50 * _GIB)

    def test_v1_uses_tightest_bounded_ancestor_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            hierarchy = root / "cgroup" / "memory"
            worker = hierarchy / "tenant" / "worker"
            (proc / "self").mkdir(parents=True)
            worker.mkdir(parents=True)
            (proc / "meminfo").write_text(
                "MemTotal:       268435456 kB\n"
                "MemAvailable:   201326592 kB\n"
                "SwapTotal:              0 kB\n",
                encoding="utf-8",
            )
            (proc / "self" / "cgroup").write_text(
                "7:cpu,cpuacct:/tenant/worker\n"
                "8:memory:/tenant/worker\n",
                encoding="utf-8",
            )
            unlimited = 1 << 62
            for path, maximum, current in (
                (hierarchy, unlimited, 20 * _GIB),
                (hierarchy / "tenant", 72 * _GIB, 24 * _GIB),
                (worker, 64 * _GIB, 8 * _GIB),
            ):
                (path / "memory.limit_in_bytes").write_text(
                    str(maximum), encoding="utf-8"
                )
                (path / "memory.usage_in_bytes").write_text(
                    str(current), encoding="utf-8"
                )

            observed = observe_host_memory(
                proc_root=proc,
                cgroup_root=root / "cgroup",
            )

        self.assertEqual(observed.cgroup_state, "bounded")
        self.assertEqual(observed.cgroup_headroom_bytes, 48 * _GIB)

    def test_tokenizer_sentinel_binds_exact_cleaning_ids_masks_and_specials(self) -> None:
        dtype = object()
        ids, masks = sentinel_rows()
        wrapper = FakeTokenizerWrapper(ids, masks, dtype=dtype)

        digest = validate_cf1_tokenizer_sentinel(
            wrapper,
            expected_dtype=dtype,
        )

        self.assertEqual(digest, CF1_TOKENIZER_SENTINEL_SHA256)
        self.assertEqual(len(wrapper.calls), 1)
        prompts, kwargs = wrapper.calls[0]
        self.assertEqual(
            prompts,
            (
                "  A  red\tfox\njumps.  ",
                "Caf\u00e9\u00a0\u732b",
                "<extra_id_0>",
            ),
        )
        self.assertEqual(
            kwargs,
            {"return_mask": True, "add_special_tokens": True},
        )

    def test_tokenizer_sentinel_rejects_every_structural_drift(self) -> None:
        dtype = object()
        mutations = []

        ids, masks = sentinel_rows()
        changed = copy.deepcopy(ids)
        changed[0][0] += 1
        mutations.append(("identifier", changed, masks, dtype, "cpu"))

        ids, masks = sentinel_rows()
        changed = copy.deepcopy(ids)
        changed[0][7] = 0
        mutations.append(("eos", changed, masks, dtype, "cpu"))

        ids, masks = sentinel_rows()
        changed = copy.deepcopy(ids)
        changed[1][-1] = 5
        mutations.append(("padding", changed, masks, dtype, "cpu"))

        ids, masks = sentinel_rows()
        changed = copy.deepcopy(masks)
        changed[0][2] = 0
        mutations.append(("mask", ids, changed, dtype, "cpu"))

        for label, mutated_ids, mutated_masks, tensor_dtype, device_type in mutations:
            with self.subTest(label=label):
                wrapper = FakeTokenizerWrapper(
                    mutated_ids,
                    mutated_masks,
                    dtype=tensor_dtype,
                )
                wrapper.ids.device.type = device_type
                with self.assertRaises(RuntimePreflightError):
                    validate_cf1_tokenizer_sentinel(
                        wrapper,
                        expected_dtype=dtype,
                    )

        ids, masks = sentinel_rows()
        wrapper = FakeTokenizerWrapper(ids, masks, dtype=object())
        with self.assertRaisesRegex(RuntimePreflightError, "dtype"):
            validate_cf1_tokenizer_sentinel(wrapper, expected_dtype=dtype)

        wrapper = FakeTokenizerWrapper(ids, masks, dtype=dtype)
        wrapper.ids.device.type = "cuda"
        with self.assertRaisesRegex(RuntimePreflightError, "CPU"):
            validate_cf1_tokenizer_sentinel(wrapper, expected_dtype=dtype)

        wrapper = FakeTokenizerWrapper(ids, masks, dtype=dtype)
        wrapper.tokenizer.is_fast = False
        with self.assertRaisesRegex(RuntimePreflightError, "fast"):
            validate_cf1_tokenizer_sentinel(wrapper, expected_dtype=dtype)

        wrapper = FakeTokenizerWrapper(ids, masks, dtype=dtype)
        wrapper.tokenizer.eos_token_id = 2
        with self.assertRaisesRegex(RuntimePreflightError, "special"):
            validate_cf1_tokenizer_sentinel(wrapper, expected_dtype=dtype)


if __name__ == "__main__":
    unittest.main()
    capture_main,
    capture_runtime_environment,
