from __future__ import annotations

import ast
import json
import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.cf_cuda_adapter import (
    CF1_RUNTIME_LOCK_SHA256,
    CF1_TOKENIZER_SENTINEL_SHA256,
    CF1Runtime,
    CF1RuntimePaths,
    RuntimeBootstrapError,
    RuntimeBindings,
    _active_attention_backend,
    _asset_paths,
    _build_checkpoint_only_generator,
    _build_verified_runtime,
    _bootstrap_provenance,
    _effective_config_sha256,
    _cf1_guard_bundle_sha256,
    _guard_bundle_sha256,
    _provenance_identity_sha256,
    _disable_upstream_bytecode_writes,
    _require_checkout_module,
    _validated_attention_probe,
    _validate_tokenizer_inventory,
    _validate_effective_config,
    build_cf1_runtime,
    validate_cf1_runtime_provenance,
)
from bench.cf_runtime_preflight import (
    RuntimeLockSnapshot,
    RuntimePreflightError,
    RuntimePreflightIdentity,
)
from bench.cf_runtime_evidence import runtime_evidence_locked_identities
from bench.model_asset_preflight import AssetLockSnapshot


_RESOLVED_CF1_CONFIG = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "model_assets"
        / "cf1-effective-config-v1.json"
    ).read_text(encoding="utf-8")
)


def _reachable_guard_modules(paths: dict[str, Path]) -> set[Path]:
    bench_root = Path(__file__).resolve().parents[1]
    pending = [bench_root / "__init__.py", *(path.resolve() for path in paths.values())]
    reached: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in reached:
            continue
        reached.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "bench":
                    module_names.update(alias.name for alias in node.names)
                elif node.module and node.module.startswith("bench."):
                    module_names.add(node.module.removeprefix("bench."))
            elif isinstance(node, ast.Import):
                module_names.update(
                    alias.name.removeprefix("bench.")
                    for alias in node.names
                    if alias.name.startswith("bench.")
                )
        for module_name in module_names:
            candidate = bench_root / (module_name.replace(".", "/") + ".py")
            if candidate.is_file() and candidate.resolve() not in reached:
                pending.append(candidate)
    return reached


class FakeModule:
    def __init__(self) -> None:
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class FakeModel(FakeModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.local_attn_size = kwargs["local_attn_size"]
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class FakeScheduler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.set_calls = []

    def set_timesteps(self, count, *, training):
        self.set_calls.append((count, training))


class FakeWanDiffusionWrapper(FakeModule):
    base_init_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).base_init_calls += 1
        raise AssertionError("from_pretrained wrapper constructor must not run")

    def post_init(self) -> None:
        self.post_init_called = True

    def state_dict(self):
        return {
            "model.patch_embedding.weight": object(),
            "model.patch_embedding.bias": object(),
        }

    def load_state_dict(self, state_dict, *, strict, assign=False):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        self.loaded_assign = assign


class FakeWanTextEncoder(FakeModule):
    base_init_calls = 0

    def __init__(self) -> None:
        type(self).base_init_calls += 1
        raise AssertionError("hard-coded unsafe text loader must not run")


class FakeTextModel(FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.eval_called = False
        self.requires_grad = None

    def eval(self):
        self.eval_called = True
        return self

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def state_dict(self):
        return {"encoder.weight": object()}

    def load_state_dict(self, state_dict, *, strict, assign=False):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        self.loaded_assign = assign


class FakeTokenizerTensor:
    def __init__(self, values):
        self._values = values
        self.shape = (len(values), len(values[0]))
        self.dtype = FakeTorch.int64
        self.device = SimpleNamespace(type="cpu")

    def tolist(self):
        return [list(row) for row in self._values]


class FakeInnerTokenizer:
    is_fast = True
    vocab_size = 256300
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 3

    def __len__(self):
        return 256300

    def convert_tokens_to_ids(self, token):
        return 256299 if token == "<extra_id_0>" else 3


class FakeTokenizer:
    calls = []
    sentinel_calls = []

    def __init__(self, **kwargs) -> None:
        type(self).calls.append(kwargs)
        self.tokenizer = FakeInnerTokenizer()

    def __call__(self, prompts, **kwargs):
        type(self).sentinel_calls.append((prompts, kwargs))
        prefixes = (
            (320, 4062, 273, 56209, 48150, 281, 274, 1),
            (25382, 273, 14985, 1),
            (256299, 1),
        )
        ids = [list(prefix) + [0] * (512 - len(prefix)) for prefix in prefixes]
        masks = [
            [1] * len(prefix) + [0] * (512 - len(prefix))
            for prefix in prefixes
        ]
        return FakeTokenizerTensor(ids), FakeTokenizerTensor(masks)


def fake_umt5_xxl(**kwargs):
    fake_umt5_xxl.calls.append(kwargs)
    return FakeTextModel()


fake_umt5_xxl.calls = []


class FakePipeline(FakeModule):
    def __init__(self, config, *, device, generator, text_encoder, vae) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class FakeTAEHV(FakeModule):
    def __init__(self, *, checkpoint_path) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.eval_called = False
        self.patch_calls = []

    def state_dict(self):
        return {"decoder.weight": object()}

    def patch_tgrow_layers(self, state_dict):
        self.patch_calls.append(state_dict)
        return state_dict

    def load_state_dict(self, state_dict, *, strict, assign=False):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        self.loaded_assign = assign

    def eval(self):
        self.eval_called = True
        return self


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    int64 = "int64"
    nn = SimpleNamespace(Module=FakeModule)
    checkpoint = None
    text_checkpoint = None
    taehv_checkpoint = None
    load_calls = []
    grad_calls = []

    @classmethod
    def load(cls, path, **kwargs):
        cls.load_calls.append((Path(path), kwargs))
        if Path(path).name == "models_t5_umt5-xxl-enc-bf16.pth":
            return cls.text_checkpoint
        if Path(path).name == "taew2_1.pth":
            return cls.taehv_checkpoint
        return cls.checkpoint

    @classmethod
    def device(cls, value):
        return value

    @classmethod
    def set_grad_enabled(cls, value):
        cls.grad_calls.append(value)


class FakeOmegaConf:
    load_calls = []

    @classmethod
    def load(cls, path):
        cls.load_calls.append(Path(path))
        return {"loaded": str(path)}

    @staticmethod
    def merge(default, candidate):
        return SimpleNamespace(
            default=default,
            candidate=candidate,
            model_kwargs={"timestep_shift": 5.0},
            denoising_step_list=[1000],
            denoising_step_list_first_chunk=[1000, 750, 500, 250],
            warp_denoising_step=True,
            independent_first_frame=False,
            num_frame_per_block=1,
            context_noise=0,
            resolved_config=json.loads(json.dumps(_RESOLVED_CF1_CONFIG)),
        )

    @staticmethod
    def to_container(config, *, resolve, enum_to_str):
        if resolve is not True or enum_to_str is not True:
            raise AssertionError("effective config must be fully resolved")
        return config.resolved_config


class CFCudaAdapterTests(unittest.TestCase):
    def test_guard_bundle_covers_recursive_local_import_closure(self) -> None:
        captured: dict[str, Path] = {}

        def capture(paths):
            captured.update(paths)
            return "a" * 64

        with mock.patch(
            "bench.cf_cuda_adapter._guard_bundle_sha256",
            side_effect=capture,
        ):
            self.assertEqual(_cf1_guard_bundle_sha256(), "a" * 64)

        covered = {path.resolve() for path in captured.values()}
        self.assertEqual(_reachable_guard_modules(captured) - covered, set())

    def test_guard_bundle_digest_changes_with_every_guard_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / name
                for name in (
                    "adapter.py",
                    "asset_preflight.py",
                    "generation_preflight.py",
                    "cuda_session.py",
                    "streaming_service.py",
                )
            }
            for name, path in paths.items():
                path.write_text(name, encoding="utf-8")
            baseline = _guard_bundle_sha256(paths)
            for name, path in paths.items():
                with self.subTest(name=name):
                    original = path.read_bytes()
                    path.write_bytes(original + b" changed")
                    self.assertNotEqual(_guard_bundle_sha256(paths), baseline)
                    path.write_bytes(original)

    def test_upstream_imports_disable_bytecode_writes_for_worker_lifetime(self) -> None:
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = False
            _disable_upstream_bytecode_writes()
            self.assertTrue(sys.dont_write_bytecode)
        finally:
            sys.dont_write_bytecode = previous

    def setUp(self) -> None:
        FakeWanDiffusionWrapper.base_init_calls = 0
        FakeWanTextEncoder.base_init_calls = 0
        FakeTokenizer.calls = []
        FakeTokenizer.sentinel_calls = []
        fake_umt5_xxl.calls = []
        FakeTorch.load_calls = []
        FakeTorch.grad_calls = []
        FakeOmegaConf.load_calls = []
        FakeTorch.checkpoint = {
            "generator_ema": {
                "model._fsdp_wrapped_module.patch_embedding.weight": "weight",
                "model._fsdp_wrapped_module.patch_embedding.bias": "bias",
            }
        }
        FakeTorch.text_checkpoint = {"encoder.weight": "text-weight"}
        FakeTorch.taehv_checkpoint = {"decoder.weight": "taehv-weight"}

    def bindings(self) -> RuntimeBindings:
        return RuntimeBindings(
            torch=FakeTorch,
            OmegaConf=FakeOmegaConf,
            CausalWanModel=FakeModel,
            WanDiffusionWrapper=FakeWanDiffusionWrapper,
            FlowMatchScheduler=FakeScheduler,
            WanTextEncoder=FakeWanTextEncoder,
            HuggingfaceTokenizer=FakeTokenizer,
            umt5_xxl=fake_umt5_xxl,
            CausalInferencePipeline=FakePipeline,
            TAEHV=FakeTAEHV,
            gpu="cuda:0",
            attention_backend="flash-attention-2",
        )

    def asset_snapshot(self) -> AssetLockSnapshot:
        path = (
            Path(__file__).resolve().parents[1]
            / "model_assets"
            / "cf1-rolling-taehv-v1.lock.json"
        )
        encoded = path.read_bytes()
        return AssetLockSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def runtime_snapshot(self) -> RuntimeLockSnapshot:
        path = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "cf1-h100-cu128-v1.lock.json"
        )
        encoded = path.read_bytes()
        return RuntimeLockSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def runtime_identity(
        self, snapshot: RuntimeLockSnapshot
    ) -> RuntimePreflightIdentity:
        return RuntimePreflightIdentity(
            runtime_id="cf1-h100-cu128-v1",
            runtime_lock_sha256=snapshot.sha256,
            runtime_evidence_sha256="b" * 64,
            static_environment_sha256="c" * 64,
            environment_sha256="e" * 64,
            effective_host_headroom_bytes=96 * 1024**3,
            gpu_total_bytes=80_000_000_000,
            gpu_free_bytes=64 * 1024**3,
        )

    def runtime_evidence_snapshot(self):
        locked = {
            "oci": {"manifest_digest": "sha256:" + "1" * 64},
            "python": {"executable_sha256": "2" * 64},
            "environment": {"tree_manifest": {"sha256": "3" * 64}},
            "native": {"loaded_libraries": [], "critical_modules": []},
        }
        return SimpleNamespace(
            sha256="b" * 64,
            encoded=b"evidence",
            parsed=lambda: locked,
        )

    def successful_attention_probe(
        self, runtime_identity: RuntimePreflightIdentity
    ) -> dict[str, object]:
        locked_identities = runtime_evidence_locked_identities(
            self.runtime_evidence_snapshot()
        )
        return {
            "probe_succeeded": True,
            "authorizes_boot": False,
            "ready": False,
            "probe_mode": "bound-verification",
            "gpu_execution_performed": True,
            "runtime_lock_sha256": runtime_identity.runtime_lock_sha256,
            "runtime_evidence_sha256": runtime_identity.runtime_evidence_sha256,
            "static_environment_sha256": (
                runtime_identity.static_environment_sha256
            ),
            "runtime_environment_sha256": (
                locked_identities.runtime_environment_sha256
            ),
            "native_identity_sha256": locked_identities.native_identity_sha256,
            "attention_backend": "flash-attention-2",
            "probe_identity_sha256": "3" * 64,
        }

    def test_attention_probe_requires_exact_locked_native_identities(self) -> None:
        runtime_identity = self.runtime_identity(self.runtime_snapshot())
        report = self.successful_attention_probe(runtime_identity)

        with self.assertRaisesRegex(RuntimeBootstrapError, "did not verify"):
            _validated_attention_probe(
                report,
                runtime_identity,
                expected_runtime_environment_sha256="4" * 64,
                expected_native_identity_sha256="5" * 64,
            )

        expected_runtime = report["runtime_environment_sha256"]
        expected_native = report["native_identity_sha256"]
        self.assertEqual(
            _validated_attention_probe(
                report,
                runtime_identity,
                expected_runtime_environment_sha256=expected_runtime,
                expected_native_identity_sha256=expected_native,
            ),
            ("3" * 64, expected_runtime, expected_native),
        )

    def ready_report(self, snapshot: AssetLockSnapshot) -> dict:
        lock = snapshot.parsed()
        return {
            "ready": True,
            "stack_id": lock["stack_id"],
            "lock_sha256": snapshot.sha256,
            "source": {
                "status": "verified",
                "observed_commit": lock["source"]["commit"],
            },
            "assets": [
                {
                    "id": asset["id"],
                    "relative_path": asset["relative_path"],
                    "status": "verified",
                    "expected_size_bytes": asset["size_bytes"],
                    "observed_size_bytes": asset["size_bytes"],
                    "expected_sha256": asset["sha256"],
                    "observed_sha256": asset["sha256"],
                }
                for asset in lock["assets"]
            ],
        }

    def make_paths(self, root: Path) -> CF1RuntimePaths:
        checkout = root / "checkout"
        (checkout / "configs").mkdir(parents=True)
        (checkout / "checkpoints" / "causal-forcing++").mkdir(parents=True)
        (checkout / "wan_models" / "Wan2.1-T2V-1.3B").mkdir(parents=True)
        default = checkout / "configs" / "default_config.yaml"
        candidate = checkout / "configs" / "causal_forcing_dmd_framewise_1step.yaml"
        model_config = checkout / "wan_models" / "Wan2.1-T2V-1.3B" / "config.json"
        checkpoint = checkout / "checkpoints" / "causal-forcing++" / "framewise-1step.pt"
        taehv = checkout / "checkpoints" / "taew2_1.pth"
        text_encoder = (
            checkout
            / "wan_models"
            / "Wan2.1-T2V-1.3B"
            / "models_t5_umt5-xxl-enc-bf16.pth"
        )
        tokenizer_directory = (
            checkout / "wan_models" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"
        )
        tokenizer_directory.mkdir(parents=True)
        tokenizer_files = tuple(
            tokenizer_directory / name
            for name in (
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        )
        for path in tokenizer_files:
            path.write_bytes(b"tokenizer fixture")
        for path in (default, candidate, checkpoint, taehv, text_encoder):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        model_config.write_text(
            json.dumps(
                {
                    "_class_name": "WanModel",
                    "_diffusers_version": "0.30.0",
                    "dim": 1536,
                    "eps": 1e-6,
                    "ffn_dim": 8960,
                    "freq_dim": 256,
                    "in_dim": 16,
                    "model_type": "t2v",
                    "num_heads": 12,
                    "num_layers": 30,
                    "out_dim": 16,
                    "text_len": 512,
                }
            ),
            encoding="utf-8",
        )
        return CF1RuntimePaths(
            checkout=checkout,
            default_config=default,
            candidate_config=candidate,
            model_config=model_config,
            generator_checkpoint=checkpoint,
            text_encoder_checkpoint=text_encoder,
            tokenizer_directory=tokenizer_directory,
            tokenizer_files=tokenizer_files,
            taehv_checkpoint=taehv,
        )

    def test_checkpoint_only_wrapper_bypasses_pretrained_constructor(self) -> None:
        model = FakeModel(local_attn_size=-1)

        wrapper = _build_checkpoint_only_generator(
            self.bindings(), model, timestep_shift=5.0
        )

        self.assertEqual(FakeWanDiffusionWrapper.base_init_calls, 0)
        self.assertIs(wrapper.model, model)
        self.assertFalse(wrapper.uniform_timestep)
        self.assertEqual(wrapper.seq_len, 32760)
        self.assertEqual(wrapper.scheduler.kwargs["shift"], 5.0)
        self.assertEqual(wrapper.scheduler.set_calls, [(1000, True)])
        self.assertTrue(wrapper.post_init_called)

    def test_cf1_adapter_rejects_a_different_stack_id(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        source_lock = (
            project_root
            / "bench"
            / "model_assets"
            / "cf1-rolling-taehv-v1.lock.json"
        )
        lock = json.loads(source_lock.read_text(encoding="utf-8"))
        lock["stack_id"] = "different-stack"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeBootstrapError, "stack_id"):
                _asset_paths(path, Path(directory))

    def test_cf1_adapter_requires_every_runtime_tokenizer_asset(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        source_lock = (
            project_root
            / "bench"
            / "model_assets"
            / "cf1-rolling-taehv-v1.lock.json"
        )
        lock = json.loads(source_lock.read_text(encoding="utf-8"))
        lock["assets"] = [
            asset for asset in lock["assets"] if asset["id"] != "wan-tokenizer"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeBootstrapError, "wan-tokenizer"):
                _asset_paths(path, Path(directory))

    def test_effective_config_requires_exact_runtime_fields(self) -> None:
        complete = FakeOmegaConf.merge({}, {})
        self.assertEqual(_validate_effective_config(complete), 5.0)
        complete.model_kwargs = {}
        with self.assertRaisesRegex(RuntimeBootstrapError, "timestep_shift"):
            _validate_effective_config(complete)
        incomplete = FakeOmegaConf.merge({}, {})
        del incomplete.independent_first_frame
        with self.assertRaisesRegex(RuntimeBootstrapError, "independent_first_frame"):
            _validate_effective_config(incomplete)

    def test_effective_config_rejects_every_cf1_value_drift(self) -> None:
        mutations = (
            ("denoising_step_list", [999, 1]),
            ("denoising_step_list_first_chunk", [1]),
            ("warp_denoising_step", False),
            ("independent_first_frame", True),
            ("num_frame_per_block", True),
            ("context_noise", 3.5),
            ("model_kwargs", {"timestep_shift": float("nan")}),
            ("model_kwargs", {"timestep_shift": 8.0}),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                config = FakeOmegaConf.merge({}, {})
                setattr(config, field, value)
                with self.assertRaises(RuntimeBootstrapError):
                    _validate_effective_config(config)

    def test_resolved_effective_config_is_bound_to_the_observed_digest(self) -> None:
        config = FakeOmegaConf.merge({}, {})
        self.assertEqual(
            _effective_config_sha256(self.bindings(), config),
            "54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06",
        )

        config.resolved_config["width"] = 831
        with self.assertRaisesRegex(RuntimeBootstrapError, "digest"):
            _effective_config_sha256(self.bindings(), config)

    def test_upstream_module_must_resolve_inside_the_verified_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            checkout.mkdir()
            inside = checkout / "pipeline" / "__init__.py"
            inside.parent.mkdir()
            inside.write_text("", encoding="utf-8")
            outside = Path(directory) / "pipeline.py"
            outside.write_text("", encoding="utf-8")

            self.assertEqual(
                _require_checkout_module(
                    SimpleNamespace(__file__=str(inside)), checkout, "pipeline"
                ),
                inside.resolve(),
            )
            with self.assertRaisesRegex(RuntimeBootstrapError, "pipeline"):
                _require_checkout_module(
                    SimpleNamespace(__file__=str(outside)), checkout, "pipeline"
                )

    def test_active_attention_backend_requires_fa3_or_fa2(self) -> None:
        for fa3, fa2, expected in (
            (True, True, "flash-attention-3"),
            (False, True, "flash-attention-2"),
        ):
            with self.subTest(fa3=fa3, fa2=fa2):
                self.assertEqual(
                    _active_attention_backend(
                        SimpleNamespace(
                            FLASH_ATTN_3_AVAILABLE=fa3,
                            FLASH_ATTN_2_AVAILABLE=fa2,
                        )
                    ),
                    expected,
                )
        with self.assertRaisesRegex(RuntimeBootstrapError, "FlashAttention"):
            _active_attention_backend(
                SimpleNamespace(
                    FLASH_ATTN_3_AVAILABLE=False,
                    FLASH_ATTN_2_AVAILABLE=False,
                )
            )
        with self.assertRaisesRegex(RuntimeBootstrapError, "backend flags"):
            _active_attention_backend(
                SimpleNamespace(
                    FLASH_ATTN_3_AVAILABLE=None,
                    FLASH_ATTN_2_AVAILABLE=True,
                )
            )

    def test_tokenizer_directory_must_contain_only_the_four_pinned_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            _validate_tokenizer_inventory(paths)
            (paths.tokenizer_directory / "added_tokens.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeBootstrapError, "tokenizer inventory"):
                _validate_tokenizer_inventory(paths)

    def test_verified_build_injects_model_text_and_passthrough_vae(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))

            runtime = _build_verified_runtime(paths, self.bindings())

        self.assertEqual(FakeWanDiffusionWrapper.base_init_calls, 0)
        self.assertEqual(FakeWanTextEncoder.base_init_calls, 0)
        self.assertEqual(
            runtime.pipeline.text_encoder.text_encoder.loaded_state_dict,
            {"encoder.weight": "text-weight"},
        )
        self.assertEqual(
            fake_umt5_xxl.calls,
            [
                {
                    "encoder_only": True,
                    "return_tokenizer": False,
                    "dtype": FakeTorch.float32,
                    "device": "cpu",
                }
            ],
        )
        self.assertTrue(runtime.pipeline.text_encoder.text_encoder.loaded_strict)
        self.assertTrue(runtime.pipeline.text_encoder.text_encoder.loaded_assign)
        self.assertEqual(
            FakeTokenizer.calls,
            [
                {
                    "name": str(paths.tokenizer_directory),
                    "seq_len": 512,
                    "clean": "whitespace",
                    "local_files_only": True,
                    "use_fast": True,
                }
            ],
        )
        self.assertEqual(runtime.pipeline.generator.model.kwargs["dim"], 1536)
        self.assertEqual(runtime.pipeline.generator.model.kwargs["local_attn_size"], -1)
        self.assertEqual(
            runtime.pipeline.generator.loaded_state_dict,
            {
                "model.patch_embedding.weight": "weight",
                "model.patch_embedding.bias": "bias",
            },
        )
        self.assertTrue(runtime.pipeline.generator.loaded_strict)
        self.assertTrue(runtime.pipeline.generator.loaded_assign)
        self.assertEqual(
            runtime.pipeline.vae.decode_to_pixel("latent", use_cache=True),
            "latent",
        )
        self.assertEqual(FakeTorch.load_calls[0][1], {
            "map_location": "cpu",
            "mmap": True,
            "weights_only": True,
        })
        self.assertEqual(FakeTorch.load_calls[1][1], {
            "map_location": "cpu",
            "mmap": True,
            "weights_only": True,
        })
        self.assertIsNone(runtime.taehv.checkpoint_path)
        self.assertEqual(
            runtime.taehv.loaded_state_dict,
            {"decoder.weight": "taehv-weight"},
        )
        self.assertTrue(runtime.taehv.loaded_strict)
        self.assertTrue(runtime.taehv.loaded_assign)
        self.assertEqual(
            FakeTorch.load_calls[2][1],
            {
                "map_location": "cpu",
                "mmap": True,
                "weights_only": True,
            },
        )
        self.assertEqual(
            runtime.taehv.to_calls,
            [((), {"device": "cuda:0", "dtype": FakeTorch.float16})],
        )
        self.assertTrue(runtime.pipeline.eval_called)
        self.assertTrue(runtime.taehv.eval_called)
        self.assertEqual(
            runtime.effective_config_sha256,
            "54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06",
        )
        self.assertEqual(FakeTorch.grad_calls, [False])
        self.assertEqual(
            runtime.tokenizer_sentinel_sha256,
            CF1_TOKENIZER_SENTINEL_SHA256,
        )
        self.assertIs(runtime.torch, FakeTorch)
        self.assertEqual(runtime.attention_backend, "flash-attention-2")
        self.assertEqual(len(FakeTokenizer.sentinel_calls), 1)

    def test_checkpoint_without_generator_ema_fails_before_state_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            FakeTorch.checkpoint = {"generator": {}}

            with self.assertRaisesRegex(ValueError, "generator_ema"):
                _build_verified_runtime(paths, self.bindings())

    def test_tokenizer_failure_prevents_every_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            with mock.patch(
                "bench.cf_cuda_adapter.validate_cf1_tokenizer_sentinel",
                side_effect=RuntimePreflightError("changed"),
            ), self.assertRaisesRegex(RuntimeBootstrapError, "tokenizer"):
                _build_verified_runtime(paths, self.bindings())
        self.assertEqual(FakeTorch.load_calls, [])

    def test_public_bootstrap_rechecks_assets_after_model_load(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        runtime_identity = self.runtime_identity(runtime_snapshot)
        ready = self.ready_report(snapshot)
        changed = {
            "ready": False,
            "source": {"status": "verified"},
            "assets": [{"id": "cf1-generator", "status": "sha256_mismatch"}],
        }
        bound_torch = object()
        sentinel = CF1Runtime(
            pipeline=object(),
            taehv=object(),
            effective_config=object(),
            effective_config_sha256="a" * 64,
            device="cuda:0",
            torch=bound_torch,
            attention_backend="flash-attention-2",
            tokenizer_sentinel_sha256=CF1_TOKENIZER_SENTINEL_SHA256,
        )
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=runtime_snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.preflight_current_runtime",
            return_value=runtime_identity,
        ), mock.patch(
            "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
            return_value=self.runtime_evidence_snapshot(),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_asset_lock_snapshot",
            return_value=snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.verify_model_assets_snapshot",
            side_effect=[ready, changed],
        ) as verify, mock.patch(
            "bench.cf_cuda_adapter._asset_paths",
            return_value=SimpleNamespace(checkout=Path("checkout")),
        ), mock.patch(
            "bench.cf_cuda_adapter._validate_tokenizer_inventory"
        ), mock.patch(
            "bench.cf_cuda_adapter.attention_probe_report",
            return_value=self.successful_attention_probe(runtime_identity),
        ), mock.patch(
            "bench.cf_cuda_adapter._load_runtime_bindings",
            return_value=SimpleNamespace(
                torch=bound_torch,
                gpu="cuda:0",
                attention_backend="flash-attention-2",
            ),
        ), mock.patch(
            "bench.cf_cuda_adapter.validate_current_host_capacity",
            return_value=96 * 1024**3,
        ), mock.patch(
            "bench.cf_cuda_adapter.validate_loaded_cuda_capacity",
            return_value=(64 * 1024**3, 80_000_000_000),
        ), mock.patch(
            "bench.cf_cuda_adapter._build_verified_runtime", return_value=sentinel
        ), mock.patch(
            "bench.cf_cuda_adapter._lock_snapshot_unchanged", return_value=True
        ), mock.patch(
            "bench.cf_cuda_adapter._runtime_lock_snapshot_unchanged",
            return_value=True,
        ), mock.patch(
            "bench.cf_cuda_adapter._runtime_evidence_snapshot_unchanged",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeBootstrapError, "changed during runtime bootstrap"
            ):
                build_cf1_runtime(lock_path=Path("lock"), checkout=Path("checkout"))
        self.assertEqual(verify.call_count, 2)

    def test_post_import_capacity_failure_prevents_model_construction(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        ready = self.ready_report(snapshot)
        for stage, message in (
            ("host", "host/CUDA"),
            ("gpu", "host/CUDA"),
        ):
            with self.subTest(stage=stage), mock.patch(
                "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
                return_value=runtime_snapshot,
            ), mock.patch(
                "bench.cf_cuda_adapter.preflight_current_runtime",
                return_value=self.runtime_identity(runtime_snapshot),
            ), mock.patch(
                "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
                return_value=self.runtime_evidence_snapshot(),
            ), mock.patch(
                "bench.cf_cuda_adapter.load_asset_lock_snapshot",
                return_value=snapshot,
            ), mock.patch(
                "bench.cf_cuda_adapter.verify_model_assets_snapshot",
                return_value=ready,
            ), mock.patch(
                "bench.cf_cuda_adapter._asset_paths",
                return_value=SimpleNamespace(checkout=Path("checkout")),
            ), mock.patch(
                "bench.cf_cuda_adapter._validate_tokenizer_inventory"
            ), mock.patch(
                "bench.cf_cuda_adapter.attention_probe_report",
                return_value=self.successful_attention_probe(
                    self.runtime_identity(runtime_snapshot)
                ),
            ), mock.patch(
                "bench.cf_cuda_adapter._load_runtime_bindings",
                return_value=SimpleNamespace(
                    torch=object(),
                    gpu="cuda:0",
                    attention_backend="flash-attention-2",
                ),
            ), mock.patch(
                "bench.cf_cuda_adapter.validate_current_host_capacity",
                return_value=96 * 1024**3,
                side_effect=(
                    RuntimePreflightError("low host")
                    if stage == "host"
                    else None
                ),
            ), mock.patch(
                "bench.cf_cuda_adapter.validate_loaded_cuda_capacity",
                return_value=(
                    64 * 1024**3,
                    80_000_000_000,
                ),
                side_effect=(
                    RuntimePreflightError("low gpu")
                    if stage == "gpu"
                    else None
                ),
            ) as loaded_gpu, mock.patch(
                "bench.cf_cuda_adapter._build_verified_runtime"
            ) as build_runtime, self.assertRaisesRegex(
                RuntimeBootstrapError, message
            ):
                build_cf1_runtime(
                    lock_path=Path("lock"),
                    checkout=Path("checkout"),
                )
            build_runtime.assert_not_called()
            if stage == "host":
                loaded_gpu.assert_not_called()

    def test_public_bootstrap_returns_runtime_bound_provenance(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        runtime_identity = self.runtime_identity(runtime_snapshot)
        ready = self.ready_report(snapshot)
        bound_torch = object()
        sentinel = CF1Runtime(
            pipeline=object(),
            taehv=object(),
            effective_config=object(),
            effective_config_sha256=(
                "54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06"
            ),
            device="cuda:0",
            torch=bound_torch,
            attention_backend="flash-attention-2",
            tokenizer_sentinel_sha256=CF1_TOKENIZER_SENTINEL_SHA256,
        )
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=runtime_snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.preflight_current_runtime",
            return_value=runtime_identity,
        ) as runtime_preflight, mock.patch(
            "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
            return_value=self.runtime_evidence_snapshot(),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_asset_lock_snapshot",
            return_value=snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.verify_model_assets_snapshot",
            side_effect=[ready, ready],
        ), mock.patch(
            "bench.cf_cuda_adapter._asset_paths",
            return_value=SimpleNamespace(checkout=Path("checkout")),
        ), mock.patch(
            "bench.cf_cuda_adapter._validate_tokenizer_inventory"
        ), mock.patch(
            "bench.cf_cuda_adapter.attention_probe_report",
            return_value=self.successful_attention_probe(runtime_identity),
        ), mock.patch(
            "bench.cf_cuda_adapter._load_runtime_bindings",
            return_value=SimpleNamespace(
                torch=bound_torch,
                gpu="cuda:0",
                attention_backend="flash-attention-2",
            ),
        ) as load_bindings, mock.patch(
            "bench.cf_cuda_adapter.validate_current_host_capacity",
            return_value=96 * 1024**3,
        ) as host_preflight, mock.patch(
            "bench.cf_cuda_adapter.validate_loaded_cuda_capacity",
            return_value=(48 * 1024**3, 81_000_000_000),
        ) as loaded_preflight, mock.patch(
            "bench.cf_cuda_adapter._build_verified_runtime", return_value=sentinel
        ), mock.patch(
            "bench.cf_cuda_adapter._lock_snapshot_unchanged", return_value=True
        ), mock.patch(
            "bench.cf_cuda_adapter._runtime_lock_snapshot_unchanged",
            return_value=True,
        ), mock.patch(
            "bench.cf_cuda_adapter._runtime_evidence_snapshot_unchanged",
            return_value=True,
        ):
            runtime = build_cf1_runtime(
                lock_path=Path("lock"),
                checkout=Path("checkout"),
                cuda_device_index=0,
            )

        runtime_preflight.assert_called_once_with(
            runtime_snapshot,
            gpu_index=0,
            evidence_path=mock.ANY,
        )
        load_bindings.assert_called_once_with(Path("checkout"), 0)
        host_preflight.assert_called_once_with(runtime_snapshot)
        loaded_preflight.assert_called_once_with(
            load_bindings.return_value.torch,
            runtime_snapshot,
            device="cuda:0",
            attention_backend="flash-attention-2",
        )
        self.assertEqual(runtime.runtime_identity.gpu_free_bytes, 48 * 1024**3)
        self.assertEqual(runtime.runtime_identity.gpu_total_bytes, 81_000_000_000)
        self.assertEqual(
            runtime.runtime_identity.effective_host_headroom_bytes,
            96 * 1024**3,
        )
        self.assertIsNotNone(runtime.provenance)
        self.assertEqual(runtime.provenance.asset_lock_sha256, snapshot.sha256)
        self.assertEqual(
            runtime.provenance.runtime_lock_sha256,
            CF1_RUNTIME_LOCK_SHA256,
        )
        self.assertEqual(
            runtime.provenance.runtime_environment_sha256,
            runtime_identity.environment_sha256,
        )
        self.assertEqual(runtime.provenance.runtime_evidence_sha256, "b" * 64)
        self.assertEqual(runtime.provenance.static_environment_sha256, "c" * 64)
        self.assertEqual(runtime.attention_probe_identity_sha256, "3" * 64)
        self.assertEqual(
            runtime.provenance.attention_probe_identity_sha256,
            runtime.attention_probe_identity_sha256,
        )
        self.assertEqual(
            runtime.provenance.tokenizer_sentinel_sha256,
            CF1_TOKENIZER_SENTINEL_SHA256,
        )
        self.assertEqual(
            runtime.provenance.effective_config_sha256,
            runtime.effective_config_sha256,
        )
        self.assertIs(runtime.torch, bound_torch)
        self.assertEqual(runtime.attention_backend, "flash-attention-2")
        self.assertEqual(
            runtime.provenance.attention_backend,
            runtime.attention_backend,
        )
        bound_lock = runtime_snapshot.parsed()
        bound_lock["evidence_lock_sha256"] = "b" * 64
        bound_runtime_snapshot = SimpleNamespace(
            sha256=runtime_snapshot.sha256,
            parsed=lambda: bound_lock,
        )
        with mock.patch(
            "bench.cf_cuda_adapter.load_asset_lock_snapshot",
            return_value=snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=bound_runtime_snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
            return_value=self.runtime_evidence_snapshot(),
        ), mock.patch(
            "bench.cf_cuda_adapter._cf1_guard_bundle_sha256",
            return_value=runtime.provenance.guard_bundle_sha256,
        ):
            self.assertIs(
                validate_cf1_runtime_provenance(runtime),
                runtime.provenance,
            )
            for field, runtime_field in (
                (
                    "runtime_native_environment_sha256",
                    "runtime_native_environment_sha256",
                ),
                ("native_identity_sha256", "native_identity_sha256"),
            ):
                with self.subTest(locked_identity=field):
                    drifted_provenance = replace(
                        runtime.provenance,
                        **{field: "4" * 64},
                        bootstrap_identity_sha256="0" * 64,
                    )
                    drifted_provenance = replace(
                        drifted_provenance,
                        bootstrap_identity_sha256=(
                            _provenance_identity_sha256(drifted_provenance)
                        ),
                    )
                    drifted_runtime = replace(
                        runtime,
                        **{runtime_field: "4" * 64},
                        provenance=drifted_provenance,
                    )
                    with self.assertRaisesRegex(
                        RuntimeBootstrapError,
                        "provenance identity",
                    ):
                        validate_cf1_runtime_provenance(drifted_runtime)

    def test_public_bootstrap_rejects_nonzero_cuda_index_before_lock_read(self) -> None:
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot"
        ) as load_lock, self.assertRaisesRegex(ValueError, "index zero"):
            build_cf1_runtime(cuda_device_index=1)
        load_lock.assert_not_called()

    def test_attention_probe_failure_prevents_bindings_and_model_construction(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=runtime_snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.preflight_current_runtime",
            return_value=self.runtime_identity(runtime_snapshot),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
            return_value=self.runtime_evidence_snapshot(),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_asset_lock_snapshot",
            return_value=snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.verify_model_assets_snapshot",
            return_value=self.ready_report(snapshot),
        ), mock.patch(
            "bench.cf_cuda_adapter._asset_paths",
            return_value=SimpleNamespace(checkout=Path("checkout")),
        ), mock.patch(
            "bench.cf_cuda_adapter._validate_tokenizer_inventory",
        ), mock.patch(
            "bench.cf_cuda_adapter.attention_probe_report",
            return_value={
                "probe_succeeded": False,
                "authorizes_boot": False,
                "failure": "loaded runtime evidence does not verify",
            },
        ), mock.patch(
            "bench.cf_cuda_adapter._load_runtime_bindings",
        ) as load_bindings, mock.patch(
            "bench.cf_cuda_adapter._build_verified_runtime",
        ) as build_runtime, self.assertRaisesRegex(
            RuntimeBootstrapError, "attention probe"
        ):
            build_cf1_runtime(lock_path=Path("lock"), checkout=Path("checkout"))

        load_bindings.assert_not_called()
        build_runtime.assert_not_called()

    def test_probe_identity_mismatch_prevents_bindings_and_constructors(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        runtime_identity = self.runtime_identity(runtime_snapshot)
        for field in (
            "runtime_environment_sha256",
            "native_identity_sha256",
        ):
            with self.subTest(field=field):
                probe_report = self.successful_attention_probe(runtime_identity)
                probe_report[field] = "4" * 64
                with mock.patch(
                    "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
                    return_value=runtime_snapshot,
                ), mock.patch(
                    "bench.cf_cuda_adapter.preflight_current_runtime",
                    return_value=runtime_identity,
                ), mock.patch(
                    "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
                    return_value=self.runtime_evidence_snapshot(),
                ), mock.patch(
                    "bench.cf_cuda_adapter.load_asset_lock_snapshot",
                    return_value=snapshot,
                ), mock.patch(
                    "bench.cf_cuda_adapter.verify_model_assets_snapshot",
                    return_value=self.ready_report(snapshot),
                ), mock.patch(
                    "bench.cf_cuda_adapter._asset_paths",
                    return_value=SimpleNamespace(checkout=Path("checkout")),
                ), mock.patch(
                    "bench.cf_cuda_adapter._validate_tokenizer_inventory",
                ), mock.patch(
                    "bench.cf_cuda_adapter.attention_probe_report",
                    return_value=probe_report,
                ), mock.patch(
                    "bench.cf_cuda_adapter._load_runtime_bindings",
                ) as load_bindings, mock.patch(
                    "bench.cf_cuda_adapter._build_verified_runtime",
                ) as build_runtime, self.assertRaisesRegex(
                    RuntimeBootstrapError, "attention probe"
                ):
                    build_cf1_runtime(
                        lock_path=Path("lock"),
                        checkout=Path("checkout"),
                    )

                load_bindings.assert_not_called()
                build_runtime.assert_not_called()

    def test_provenance_rejects_a_ready_report_with_unobserved_asset_bytes(self) -> None:
        snapshot = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        report = self.ready_report(snapshot)
        report["assets"][0]["observed_sha256"] = "0" * 64
        runtime = CF1Runtime(
            pipeline=object(),
            taehv=object(),
            effective_config=object(),
            effective_config_sha256=(
                "54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06"
            ),
            device="cuda:0",
            torch=object(),
            attention_backend="flash-attention-2",
            tokenizer_sentinel_sha256=CF1_TOKENIZER_SENTINEL_SHA256,
        )
        with self.assertRaisesRegex(RuntimeBootstrapError, "observed asset"):
            _bootstrap_provenance(
                snapshot,
                self.runtime_identity(runtime_snapshot),
                runtime,
                "d" * 64,
                report,
            )

    def test_public_bootstrap_rejects_semantically_equal_but_unpinned_lock_bytes(self) -> None:
        original = self.asset_snapshot()
        runtime_snapshot = self.runtime_snapshot()
        drifted_bytes = original.encoded + b" "
        drifted = AssetLockSnapshot(
            encoded=drifted_bytes,
            sha256=hashlib.sha256(drifted_bytes).hexdigest(),
        )
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=runtime_snapshot,
        ), mock.patch(
            "bench.cf_cuda_adapter.preflight_current_runtime",
            return_value=self.runtime_identity(runtime_snapshot),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_runtime_evidence_snapshot",
            return_value=self.runtime_evidence_snapshot(),
        ), mock.patch(
            "bench.cf_cuda_adapter.load_asset_lock_snapshot",
            return_value=drifted,
        ), self.assertRaisesRegex(RuntimeBootstrapError, "lock digest"):
            build_cf1_runtime(lock_path=Path("lock"), checkout=Path("checkout"))

    def test_public_bootstrap_rejects_runtime_lock_byte_drift_before_observation(
        self,
    ) -> None:
        original = self.runtime_snapshot()
        drifted_bytes = original.encoded + b" "
        drifted = RuntimeLockSnapshot(
            encoded=drifted_bytes,
            sha256=hashlib.sha256(drifted_bytes).hexdigest(),
        )
        with mock.patch(
            "bench.cf_cuda_adapter.load_runtime_lock_snapshot",
            return_value=drifted,
        ), mock.patch(
            "bench.cf_cuda_adapter.preflight_current_runtime"
        ) as observe, self.assertRaisesRegex(
            RuntimeBootstrapError, "runtime lock digest"
        ):
            build_cf1_runtime(lock_path=Path("lock"), checkout=Path("checkout"))
        observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
