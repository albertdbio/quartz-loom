from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from bench.cf_cuda_generator import (
    CF1LatentPullSession,
    CudaGenerationError,
)


class FakeDevice:
    def __init__(self, kind: str, index: int | None = None) -> None:
        self.type = kind
        self.index = index

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype,
        device,
        log,
        label="tensor",
        scalar=None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.log = log
        self.label = label
        self.scalar = scalar

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        shape = list(self.shape)
        output = []
        for axis, selector in enumerate(key):
            size = shape[axis]
            if isinstance(selector, int):
                continue
            if isinstance(selector, slice):
                start, stop, step = selector.indices(size)
                output.append(len(range(start, stop, step)))
                continue
            raise AssertionError(f"unsupported selector: {selector!r}")
        output.extend(shape[len(key) :])
        self.log.append(("slice", self.label, key, tuple(output)))
        return FakeTensor(
            output,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            label=f"{self.label}-slice",
        )

    def __mul__(self, value):
        self.log.append(("mul", self.label, value))
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            label=self.label,
            scalar=value,
        )

    __rmul__ = __mul__

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and args:
            dtype = args[0]
        self.log.append(("to", self.label, dtype))
        return FakeTensor(
            self.shape,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device,
            log=self.log,
            label=self.label,
            scalar=self.scalar,
        )

    def detach(self):
        self.log.append(("detach", self.label))
        return self

    def clone(self):
        self.log.append(("clone", self.label))
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            label=f"{self.label}-clone",
            scalar=self.scalar,
        )

    def flatten(self, start_dim, end_dim):
        self.log.append(("flatten", self.label, start_dim, end_dim))
        flattened = 1
        for size in self.shape[start_dim : end_dim + 1]:
            flattened *= size
        shape = (
            self.shape[:start_dim]
            + (flattened,)
            + self.shape[end_dim + 1 :]
        )
        return FakeTensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            label=f"{self.label}-flat",
        )

    def unflatten(self, dim, sizes):
        self.log.append(("unflatten", self.label, dim, tuple(sizes)))
        shape = self.shape[:dim] + tuple(sizes) + self.shape[dim + 1 :]
        return FakeTensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            log=self.log,
            label=f"{self.label}-unflat",
        )

    def zero_(self):
        self.log.append(("zero", self.label))
        self.scalar = 0
        return self

    def item(self):
        return self.scalar


class FakeGenerator:
    def __init__(self, device, log) -> None:
        self.device = device
        self.log = log
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        self.log.append(("generator-seed", str(self.device), seed))
        return self


class FakeEvent:
    def __init__(self, log) -> None:
        self.log = log

    def record(self, stream=None):
        self.log.append(("event-record", stream))


class FakeContext:
    def __init__(self, log, device, *, kind="device") -> None:
        self.log = log
        self.device = device
        self.kind = kind

    def __enter__(self):
        self.log.append((f"{self.kind}-enter", str(self.device)))

    def __exit__(self, exc_type, exc, traceback):
        self.log.append((f"{self.kind}-exit", str(self.device)))
        return False


class FakeCuda:
    def __init__(self, log) -> None:
        self.log = log

    def manual_seed_all(self, seed):
        self.log.append(("cuda-seed-all", seed))

    def Event(self, **kwargs):
        self.log.append(("event-create", kwargs))
        return FakeEvent(self.log)

    def default_stream(self, device=None):
        self.log.append(("default-stream", str(device)))
        return "generation-stream"

    def device(self, device):
        return FakeContext(self.log, device)

    def stream(self, stream):
        return FakeContext(self.log, stream, kind="stream")


class FakeTorch:
    bfloat16 = "bfloat16"
    int64 = "int64"
    long = "int64"

    def __init__(self, log) -> None:
        self.log = log
        self.cuda = FakeCuda(log)

    def manual_seed(self, seed):
        self.log.append(("torch-seed", seed))

    def Generator(self, device):
        self.log.append(("generator-create", str(device)))
        return FakeGenerator(device, self.log)

    def randn(self, shape, *, device, dtype, generator):
        self.log.append(
            (
                "randn",
                tuple(shape),
                str(device),
                dtype,
                generator.seed,
            )
        )
        return FakeTensor(
            shape,
            dtype=dtype,
            device=device,
            log=self.log,
            label="initial-noise",
        )

    def randn_like(self, value):
        self.log.append(("randn-like", value.label, value.shape))
        return FakeTensor(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            log=self.log,
            label="transition-noise",
        )

    def ones(self, shape, *, device, dtype):
        self.log.append(("ones", tuple(shape), str(device), dtype))
        return FakeTensor(
            shape,
            dtype=dtype,
            device=device,
            log=self.log,
            label="ones",
            scalar=1,
        )

    def ones_like(self, value):
        self.log.append(("ones-like", value.scalar))
        return FakeTensor(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            log=self.log,
            label="ones-like",
            scalar=1,
        )

    def tensor(self, value, *, dtype, device):
        self.log.append(("tensor", tuple(value), dtype, str(device)))
        return FakeTensor(
            (len(value),),
            dtype=dtype,
            device=device,
            log=self.log,
            label="cache-index",
            scalar=value[0],
        )


class FakeScheduler:
    def __init__(self, log) -> None:
        self.log = log

    def add_noise(self, denoised, transition_noise, next_timestep):
        self.log.append(
            (
                "add-noise",
                denoised.shape,
                transition_noise.shape,
                next_timestep.scalar,
            )
        )
        return FakeTensor(
            denoised.shape,
            dtype=denoised.dtype,
            device=denoised.device,
            log=self.log,
            label="scheduled-noise",
        )


class FakeTextEncoder:
    def __init__(self, log, torch, device) -> None:
        self.log = log
        self.torch = torch
        self.device = device

    def __call__(self, *, text_prompts):
        self.log.append(("text-encode", tuple(text_prompts)))
        return {
            "prompt_embeds": FakeTensor(
                (1, 512, 4096),
                dtype=self.torch.bfloat16,
                device=self.device,
                log=self.log,
                label="prompt-embeds",
            )
        }


class FakeModelGenerator:
    def __init__(self, log, torch, device) -> None:
        self.log = log
        self.torch = torch
        self.device = device
        self.calls = []
        self.fail_call = None
        self.bad_shape_call = None
        self.interrupt_after_mutation_call = None

    def __call__(self, **kwargs):
        call_number = len(self.calls)
        if self.fail_call == call_number:
            raise RuntimeError("synthetic generator failure")
        timestep = kwargs["timestep"].scalar
        self.calls.append(
            (
                timestep,
                kwargs["current_start"],
                kwargs["noisy_image_or_video"].shape,
                kwargs["kv_cache"],
                kwargs["crossattn_cache"],
            )
        )
        self.log.append(("model-forward", call_number, timestep, kwargs["current_start"]))
        cache_end = kwargs["current_start"] + 1560
        for cache in kwargs["kv_cache"]:
            cache["global_end_index"].scalar = cache_end
            cache["local_end_index"].scalar = cache_end
        if self.interrupt_after_mutation_call == call_number:
            raise KeyboardInterrupt
        shape = (
            (1, 2, 16, 60, 104)
            if self.bad_shape_call == call_number
            else (1, 1, 16, 60, 104)
        )
        denoised = FakeTensor(
            shape,
            dtype=self.torch.bfloat16,
            device=self.device,
            log=self.log,
            label=f"denoised-{call_number}",
        )
        flow = FakeTensor(
            shape,
            dtype=self.torch.bfloat16,
            device=self.device,
            log=self.log,
            label=f"flow-{call_number}",
        )
        return flow, denoised


class FakePipeline:
    def __init__(self, log, torch, device) -> None:
        self.log = log
        self.torch = torch
        self.device = device
        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.local_attn_size = -1
        self.args = SimpleNamespace(context_noise=0)
        self.denoising_step_list_first_chunk = (400, 300, 200, 100)
        self.denoising_step_list = (100,)
        self.scheduler = FakeScheduler(log)
        self.text_encoder = FakeTextEncoder(log, torch, device)
        self.generator = FakeModelGenerator(log, torch, device)
        self.kv_cache1 = None
        self.crossattn_cache = None

    def _initialize_kv_cache(self, *, batch_size, dtype, device):
        self.log.append(("init-kv", batch_size, dtype, str(device)))
        self.kv_cache1 = [
            {
                "global_end_index": FakeTensor(
                    (1,),
                    dtype=self.torch.long,
                    device=device,
                    log=self.log,
                    label=f"global-{index}",
                    scalar=0,
                ),
                "local_end_index": FakeTensor(
                    (1,),
                    dtype=self.torch.long,
                    device=device,
                    log=self.log,
                    label=f"local-{index}",
                    scalar=0,
                ),
            }
            for index in range(30)
        ]

    def _initialize_crossattn_cache(self, *, batch_size, dtype, device):
        self.log.append(("init-cross", batch_size, dtype, str(device)))
        self.crossattn_cache = [{"is_init": False} for _ in range(30)]


class CF1LatentPullSessionTests(unittest.TestCase):
    def make_runtime(self):
        log = []
        device = FakeDevice("cuda", 0)
        torch = FakeTorch(log)
        pipeline = FakePipeline(log, torch, device)
        runtime = SimpleNamespace(
            pipeline=pipeline,
            torch=torch,
            device=device,
        )
        return runtime, pipeline, log

    def make_session(self, runtime, *, prompt="A red fox runs.", seed=7):
        with mock.patch(
            "bench.cf_cuda_generator._require_verified_runtime",
            side_effect=lambda value: value,
        ):
            return CF1LatentPullSession(
                runtime=runtime,
                prompt=prompt,
                seed=seed,
            )

    def test_exact_recovered_21_block_schedule_rng_cache_and_event_order(self) -> None:
        runtime, pipeline, log = self.make_runtime()
        session = self.make_session(runtime)

        outputs = [session.pull() for _ in range(21)]

        self.assertEqual([item.block_index for item in outputs], list(range(21)))
        self.assertTrue(
            all(item.denoised_latent.shape == (1, 1, 16, 60, 104) for item in outputs)
        )
        expected_timesteps = [400, 300, 200, 100, 0] + [100, 0] * 20
        self.assertEqual([call[0] for call in pipeline.generator.calls], expected_timesteps)
        expected_starts = [0] * 5 + [index * 1560 for index in range(1, 21) for _ in range(2)]
        self.assertEqual([call[1] for call in pipeline.generator.calls], expected_starts)
        self.assertEqual(len(pipeline.generator.calls), 45)
        self.assertEqual(outputs[0].denoised_latent.label, "denoised-3-clone")
        clone_index = log.index(("clone", "denoised-3"))
        context_index = log.index(("model-forward", 4, 0, 0))
        self.assertLess(clone_index, context_index)
        self.assertEqual(
            [entry for entry in log if entry[:1] == ("randn-like",)],
            [
                ("randn-like", "denoised-0-flat", (1, 16, 60, 104)),
                ("randn-like", "denoised-1-flat", (1, 16, 60, 104)),
                ("randn-like", "denoised-2-flat", (1, 16, 60, 104)),
            ],
        )
        self.assertIn(("torch-seed", 7), log)
        self.assertIn(("cuda-seed-all", 7), log)
        self.assertIn(("generator-seed", "cuda:0", 7), log)
        self.assertIn(
            ("randn", (1, 21, 16, 60, 104), "cuda:0", "bfloat16", 7),
            log,
        )
        self.assertEqual(
            [entry for entry in log if entry[:1] == ("event-create",)],
            [("event-create", {})] * 21,
        )
        self.assertEqual(
            [entry for entry in log if entry[:1] == ("event-record",)],
            [("event-record", "generation-stream")] * 21,
        )
        for block_index, output in enumerate(outputs):
            context = ("model-forward", 4 if block_index == 0 else 5 + block_index * 2 - 1, 0, block_index * 1560)
            context_index = log.index(context)
            record_indices = [
                index for index, entry in enumerate(log) if entry == ("event-record", "generation-stream")
            ]
            self.assertLess(context_index, record_indices[block_index])
            self.assertIsNotNone(output.latent_ready_event)
        self.assertTrue(session.complete)
        session.finish()
        self.assertFalse(hasattr(pipeline, "_cf1_pull_session_owner"))
        with self.assertRaisesRegex(CudaGenerationError, "complete"):
            session.pull()

    def test_reuse_resets_existing_cache_in_place(self) -> None:
        runtime, pipeline, log = self.make_runtime()
        first = self.make_session(runtime)
        for _ in range(21):
            first.pull()
        first.finish()
        for cache in pipeline.kv_cache1:
            cache["global_end_index"].scalar = 99
            cache["local_end_index"].scalar = 99
        for cache in pipeline.crossattn_cache:
            cache["is_init"] = True

        second = self.make_session(runtime, seed=8)

        self.assertTrue(all(not cache["is_init"] for cache in pipeline.crossattn_cache))
        self.assertTrue(
            all(cache["global_end_index"].scalar == 0 for cache in pipeline.kv_cache1)
        )
        self.assertTrue(
            all(cache["local_end_index"].scalar == 0 for cache in pipeline.kv_cache1)
        )
        self.assertEqual(
            len([entry for entry in log if entry[:1] == ("init-kv",)]),
            1,
        )
        self.assertFalse(second.complete)

    def test_generation_failure_poisons_session_and_retains_runtime_ownership(self) -> None:
        runtime, pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        pipeline.generator.fail_call = 0

        with self.assertRaisesRegex(CudaGenerationError, "latent generation failed"):
            session.pull()

        self.assertTrue(session.poisoned)
        self.assertTrue(hasattr(pipeline, "_cf1_pull_session_owner"))
        with self.assertRaisesRegex(CudaGenerationError, "poisoned"):
            session.pull()
        with self.assertRaisesRegex(CudaGenerationError, "poisoned"):
            session.finish()

    def test_base_exception_after_cache_mutation_poisons_and_cannot_resume(self) -> None:
        runtime, pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        pipeline.generator.interrupt_after_mutation_call = 0

        with self.assertRaises(KeyboardInterrupt):
            session.pull()

        self.assertEqual(
            pipeline.kv_cache1[0]["global_end_index"].scalar,
            1560,
        )
        self.assertTrue(session.poisoned)
        self.assertIs(pipeline._cf1_pull_session_poisoned, True)
        self.assertTrue(hasattr(pipeline, "_cf1_pull_session_owner"))
        with self.assertRaisesRegex(CudaGenerationError, "poisoned"):
            session.pull()

    def test_base_exception_during_initialization_marks_pipeline_poisoned(self) -> None:
        runtime, pipeline, _log = self.make_runtime()

        def interrupting_text_encoder(*, text_prompts):
            raise KeyboardInterrupt

        pipeline.text_encoder = interrupting_text_encoder
        with self.assertRaises(KeyboardInterrupt):
            self.make_session(runtime)

        self.assertIs(pipeline._cf1_pull_session_poisoned, True)
        self.assertTrue(hasattr(pipeline, "_cf1_pull_session_owner"))
        with self.assertRaisesRegex(CudaGenerationError, "pipeline is poisoned"):
            self.make_session(runtime, seed=8)

    def test_base_exception_during_finalization_keeps_poison_monotonic(self) -> None:
        runtime, pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        for _ in range(21):
            session.pull()

        class InterruptingIndex:
            def item(self):
                raise KeyboardInterrupt

        pipeline.kv_cache1[0]["global_end_index"] = InterruptingIndex()
        with self.assertRaises(KeyboardInterrupt):
            session.finish()

        self.assertTrue(session.poisoned)
        self.assertIs(pipeline._cf1_pull_session_poisoned, True)
        self.assertTrue(hasattr(pipeline, "_cf1_pull_session_owner"))

    def test_bad_generated_latent_shape_poisons_before_event(self) -> None:
        runtime, pipeline, log = self.make_runtime()
        session = self.make_session(runtime)
        pipeline.generator.bad_shape_call = 0

        with self.assertRaisesRegex(CudaGenerationError, "latent generation failed"):
            session.pull()

        self.assertTrue(session.poisoned)
        self.assertFalse(any(entry[:1] == ("event-create",) for entry in log))

    def test_incomplete_finish_and_concurrent_session_fail_closed(self) -> None:
        runtime, _pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        with self.assertRaisesRegex(CudaGenerationError, "incomplete"):
            session.finish()
        with self.assertRaisesRegex(CudaGenerationError, "already owns"):
            self.make_session(runtime, seed=8)

    def test_ownership_loss_marks_pipeline_poisoned(self) -> None:
        runtime, pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        delattr(pipeline, "_cf1_pull_session_owner")

        with self.assertRaisesRegex(CudaGenerationError, "ownership changed"):
            session.pull()

        self.assertTrue(session.poisoned)
        self.assertIs(pipeline._cf1_pull_session_poisoned, True)
        with self.assertRaisesRegex(CudaGenerationError, "pipeline is poisoned"):
            self.make_session(runtime, seed=8)

    def test_pipeline_poison_is_monotonic_during_finalization(self) -> None:
        runtime, pipeline, _log = self.make_runtime()
        session = self.make_session(runtime)
        for _ in range(21):
            session.pull()
        pipeline._cf1_pull_session_poisoned = True

        with self.assertRaisesRegex(CudaGenerationError, "pipeline is poisoned"):
            session.finish()

        self.assertIs(pipeline._cf1_pull_session_poisoned, True)

    def test_invalid_request_and_pipeline_contract_fail_before_rng_or_text(self) -> None:
        cases = (
            ("prompt", {"prompt": "  "}, None),
            ("seed", {"seed": -1}, None),
            ("seed", {"seed": 2**32}, None),
            ("num_frame_per_block", {}, ("num_frame_per_block", 2)),
            ("independent_first_frame", {}, ("independent_first_frame", True)),
            ("num_transformer_blocks", {}, ("num_transformer_blocks", 29)),
            ("frame_seq_length", {}, ("frame_seq_length", 1)),
            ("local_attn_size", {}, ("local_attn_size", 0)),
            ("context_noise", {}, ("context_noise", 1)),
            ("first schedule", {}, ("denoising_step_list_first_chunk", (1,))),
            ("ordinary schedule", {}, ("denoising_step_list", (1, 2))),
        )
        for label, kwargs, mutation in cases:
            with self.subTest(label=label):
                runtime, pipeline, log = self.make_runtime()
                if mutation is not None:
                    target, value = mutation
                    if target == "context_noise":
                        pipeline.args.context_noise = value
                    else:
                        setattr(pipeline, target, value)
                with self.assertRaises(CudaGenerationError):
                    self.make_session(runtime, **kwargs)
                self.assertFalse(any(entry[:1] == ("randn",) for entry in log))
                self.assertFalse(any(entry[:1] == ("text-encode",) for entry in log))


if __name__ == "__main__":
    unittest.main()
