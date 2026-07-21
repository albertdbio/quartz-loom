import ast
import hashlib
import os
import py_compile
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench import cf_streaming_process_worker
from bench import cf_streaming_worker
from bench.cf_streaming_worker import (
    CF1ProcessStreamingBackend,
    CF1StreamingWorker,
    CF1StreamingWorkerError,
    build_cf1_process_streaming_backend,
    build_cf1_streaming_worker,
)
from bench.streaming_process import DEFAULT_WORKER_SCRIPT, WorkerProtocolError
from bench.streaming_process_worker import PNG_1X1, receive_packet, send_packet
from bench.streaming_process_protocol import worker_bundle_sha256
from bench.streaming_service import (
    BackendFatalError,
    DecodedChunk,
    StreamProtocolError,
    StreamRequest,
)


STACK_SHA256 = "7" * 64
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE_INDEX_DIGEST = "sha256:" + "b" * 64
IMAGE_CONFIG_DIGEST = "sha256:" + "c" * 64
RUNTIME_LAUNCH = {
    "runtime_image_index_digest": IMAGE_INDEX_DIGEST,
    "runtime_image_digest": IMAGE_DIGEST,
    "runtime_image_config_digest": IMAGE_CONFIG_DIGEST,
    "runtime_environment_root": "/runtime/venv",
    "runtime_distribution_path": "/runtime/venv/site-packages",
    "runtime_wheelhouse": "/runtime/wheelhouse",
}
WORKER_BUNDLE_SHA256 = worker_bundle_sha256(
    Path(cf_streaming_process_worker.__file__).resolve(),
    cf_streaming_worker.REAL_WORKER_BUNDLE_PATHS,
)
JPEG_832X480 = (
    b"\xff\xd8"
    b"\xff\xc0\x00\x11\x08\x01\xe0\x03\x40\x03"
    b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    b"\x00\xff\xd9"
)


def _reachable_bench_modules(entrypoint: Path) -> set[Path]:
    bench_root = entrypoint.resolve().parent
    pending = [(bench_root / "__init__.py").resolve(), entrypoint.resolve()]
    reached: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
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
            if candidate.is_file() and candidate not in reached:
                pending.append(candidate.resolve())
    return reached


class FakeSession:
    def __init__(self, *, runtime, prompt, seed, log):
        self.runtime = runtime
        self.prompt = prompt
        self.seed = seed
        self.log = log
        self.pulls = 0
        self.log.append(("session-start", prompt, seed))

    @property
    def complete(self):
        return self.pulls == 21

    def pull(self):
        index = self.pulls
        self.log.append(("pull", index))
        self.pulls += 1
        return SimpleNamespace(
            block_index=index,
            denoised_latent=f"latent-{index}",
            latent_ready_event=f"event-{index}",
        )

    def finish(self):
        self.log.append(("session-finish", self.pulls))


class FakeDecoder:
    def __init__(self, *, runtime, torch, encode_frames, frame_media_type, log):
        self.runtime = runtime
        self.torch = torch
        self.encode_frames = encode_frames
        self.frame_media_type = frame_media_type
        self.log = log
        self.decodes = 0
        self.log.append(("decoder-start", frame_media_type))

    @property
    def complete(self):
        return self.decodes == 21

    def decode(self, latent, *, latent_ready_event):
        index = self.decodes
        self.log.append(("decode", index, latent, latent_ready_event))
        self.decodes += 1
        count = 1 if index == 0 else 4
        return DecodedChunk(tuple(PNG_1X1 for _ in range(count)), "image/png")

    def finish(self):
        self.log.append(("decoder-finish", self.decodes))


class FakeJpegDecoder(FakeDecoder):
    def decode(self, latent, *, latent_ready_event):
        index = self.decodes
        self.log.append(("decode", index, latent, latent_ready_event))
        self.decodes += 1
        count = 1 if index == 0 else 4
        return DecodedChunk(
            tuple(JPEG_832X480 for _ in range(count)),
            "image/jpeg",
        )


class FakeProtocolEngine:
    def __init__(self, log, *, frame_media_type="image/png"):
        self.stack_sha256 = STACK_SHA256
        self.log = log
        self.next_index = 0
        self.frame_media_type = frame_media_type
        self.frame_encoding_profile = (
            "jpeg-q90-cpu-v1"
            if frame_media_type == "image/jpeg"
            else "png-c1-lossless-v1"
        )

    def start(self, *, prompt, seed, latent_frames):
        self.log.append(("start", prompt, seed, latent_frames))

    def pull(self, chunk_index):
        if chunk_index != self.next_index:
            raise AssertionError("noncontiguous test pull")
        self.log.append(("pull", chunk_index))
        self.next_index += 1
        count = 1 if chunk_index == 0 else 4
        payload = JPEG_832X480 if self.frame_media_type == "image/jpeg" else PNG_1X1
        return DecodedChunk(
            tuple(payload for _ in range(count)),
            self.frame_media_type,
        )

    def finish(self, chunk_index):
        self.log.append(("finish", chunk_index))


class CF1StreamingWorkerTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        return SimpleNamespace(
            torch=SimpleNamespace(),
            provenance=SimpleNamespace(bootstrap_identity_sha256=STACK_SHA256),
        )

    def build_engine(
        self,
        *,
        session_class=FakeSession,
        decoder_class=FakeDecoder,
        frame_encoding_profile="png-c1-lossless-v1",
    ):
        runtime = self.runtime()
        log = []

        def make_session(**kwargs):
            return session_class(**kwargs, log=log)

        def make_decoder(**kwargs):
            return decoder_class(**kwargs, log=log)

        patches = (
            mock.patch.object(
                cf_streaming_worker,
                "_require_verified_runtime",
                side_effect=lambda value: value,
            ),
            mock.patch.object(
                cf_streaming_worker,
                "build_png_encoder",
                return_value=lambda frames: frames,
            ),
            mock.patch.object(
                cf_streaming_worker,
                "build_cpu_jpeg_encoder",
                return_value=lambda frames: frames,
            ),
            mock.patch.object(
                cf_streaming_worker,
                "CF1LatentPullSession",
                side_effect=make_session,
            ),
            mock.patch.object(
                cf_streaming_worker,
                "RollingTaehvChunkDecoder",
                side_effect=make_decoder,
            ),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return CF1StreamingWorker(
            runtime,
            frame_encoding_profile=frame_encoding_profile,
        ), log

    def test_one_pull_credit_executes_exactly_one_generate_decode_pair(self):
        worker, log = self.build_engine()
        worker.start(prompt="a fox runs", seed=9, latent_frames=21)
        self.assertEqual([item for item in log if item[0] == "pull"], [])

        for index in range(21):
            prior_pulls = len([item for item in log if item[0] == "pull"])
            chunk = worker.pull(index)
            self.assertEqual(
                len([item for item in log if item[0] == "pull"]),
                prior_pulls + 1,
            )
            self.assertEqual(chunk.frame_count, 1 if index == 0 else 4)
            self.assertEqual(chunk.frame_media_type, "image/png")

        worker.finish(21)
        self.assertEqual(log[-2:], [("decoder-finish", 21), ("session-finish", 21)])
        self.assertFalse(worker.poisoned)
        self.assertFalse(worker.active)

        worker.start(prompt="a second job", seed=10, latent_frames=21)
        self.assertTrue(worker.active)

    def test_browser_worker_releases_exact_jpeg_frames(self):
        worker, log = self.build_engine(
            decoder_class=FakeJpegDecoder,
            frame_encoding_profile="jpeg-q90-cpu-v1",
        )
        worker.start(prompt="a fox runs", seed=9, latent_frames=21)

        chunk = worker.pull(0)

        self.assertEqual(chunk.frame_media_type, "image/jpeg")
        self.assertEqual(chunk.frame_payloads, (JPEG_832X480,))
        self.assertIn(("decoder-start", "image/jpeg"), log)

    def test_worker_refuses_unknown_frame_encoding_profile_before_encoder_build(self):
        runtime = self.runtime()
        with (
            mock.patch.object(
                cf_streaming_worker,
                "_require_verified_runtime",
                side_effect=lambda value: value,
            ),
            mock.patch.object(cf_streaming_worker, "build_png_encoder") as png,
            mock.patch.object(
                cf_streaming_worker,
                "build_cpu_jpeg_encoder",
            ) as jpeg,
        ):
            with self.assertRaisesRegex(
                CF1StreamingWorkerError,
                "frame encoding profile",
            ):
                CF1StreamingWorker(
                    runtime,
                    frame_encoding_profile="jpeg-q37-mystery-v9",
                )
        png.assert_not_called()
        jpeg.assert_not_called()

    def test_cpu_jpeg_profile_uses_fixed_q90_and_validates_exact_frame_shape(self):
        uint8 = object()
        cpu = SimpleNamespace(type="cpu")
        calls = []

        class Frame:
            def contiguous(self):
                calls.append(("contiguous", self))
                return self

        frame = Frame()

        class Batch:
            shape = (1, 3, 480, 832)
            dtype = uint8
            device = cpu

            def __getitem__(self, index):
                self_index = index
                calls.append(("frame", self_index))
                return frame

        class NumpyBytes:
            def tobytes(self):
                return JPEG_832X480

        encoded = SimpleNamespace(
            shape=(len(JPEG_832X480),),
            dtype=uint8,
            device=cpu,
            numpy=lambda: NumpyBytes(),
        )

        def encode_jpeg(observed_frame, *, quality):
            calls.append(("encode", observed_frame, quality))
            return encoded

        runtime = SimpleNamespace(torch=SimpleNamespace(uint8=uint8))
        with (
            mock.patch.object(cf_streaming_worker, "build_png_encoder") as authorize,
            mock.patch.object(
                cf_streaming_worker.importlib,
                "import_module",
                return_value=SimpleNamespace(encode_jpeg=encode_jpeg),
            ) as imported,
        ):
            encoder = cf_streaming_worker.build_cpu_jpeg_encoder(runtime)
            payloads = encoder(Batch())

        authorize.assert_called_once_with(runtime)
        imported.assert_called_once_with("torchvision.io")
        self.assertEqual(payloads, (JPEG_832X480,))
        self.assertIn(("encode", frame, 90), calls)

    def test_cf1_jpeg_parser_rejects_nearby_or_trailing_payloads(self):
        self.assertTrue(cf_streaming_worker._is_valid_cf1_jpeg(JPEG_832X480))
        wrong_height = JPEG_832X480.replace(b"\x01\xe0\x03\x40", b"\x01\xdf\x03\x40")
        progressive = JPEG_832X480.replace(b"\xff\xc0", b"\xff\xc2", 1)
        for payload in (
            wrong_height,
            progressive,
            JPEG_832X480 + b"trailing",
            JPEG_832X480[:-3] + b"\xff\xd9",
            b"\xff\xd8wrapped-but-not-a-jpeg\xff\xd9",
        ):
            with self.subTest(payload=payload[:20]):
                self.assertFalse(cf_streaming_worker._is_valid_cf1_jpeg(payload))

    def test_request_contract_rejects_before_session_construction(self):
        worker, log = self.build_engine()
        for kwargs, pattern in (
            ({"prompt": "", "seed": 0, "latent_frames": 21}, "prompt"),
            ({"prompt": "ok", "seed": -1, "latent_frames": 21}, "seed"),
            ({"prompt": "ok", "seed": 2**32, "latent_frames": 21}, "seed"),
            ({"prompt": "ok", "seed": 0, "latent_frames": 20}, "21"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(CF1StreamingWorkerError, pattern):
                    worker.start(**kwargs)
        self.assertFalse(any(item[0] == "session-start" for item in log))

    def test_noncontiguous_or_early_terminal_credit_is_fatal(self):
        worker, _log = self.build_engine()
        worker.start(prompt="ok", seed=0, latent_frames=21)
        with self.assertRaisesRegex(CF1StreamingWorkerError, "chunk index"):
            worker.pull(1)
        self.assertTrue(worker.poisoned)
        with self.assertRaisesRegex(CF1StreamingWorkerError, "poisoned"):
            worker.start(prompt="again", seed=0, latent_frames=21)

        other, _other_log = self.build_engine()
        other.start(prompt="ok", seed=0, latent_frames=21)
        with self.assertRaisesRegex(CF1StreamingWorkerError, "completion index"):
            other.finish(21)
        self.assertTrue(other.poisoned)

    def test_baseexception_from_generation_poison_rethrows_unchanged(self):
        class InterruptingSession(FakeSession):
            def pull(self):
                self.log.append(("pull-interrupt", self.pulls))
                raise KeyboardInterrupt

        worker, _log = self.build_engine(session_class=InterruptingSession)
        worker.start(prompt="ok", seed=0, latent_frames=21)
        with self.assertRaises(KeyboardInterrupt):
            worker.pull(0)
        self.assertTrue(worker.poisoned)
        with self.assertRaisesRegex(CF1StreamingWorkerError, "poisoned"):
            worker.pull(0)

    def test_builder_requires_observed_bootstrap_identity(self):
        built = SimpleNamespace(stack_sha256="8" * 64)
        with (
            mock.patch.object(
                cf_streaming_worker,
                "build_cf1_runtime",
                return_value=self.runtime(),
            ),
            mock.patch.object(
                cf_streaming_worker,
                "CF1StreamingWorker",
                return_value=built,
            ),
        ):
            with self.assertRaisesRegex(CF1StreamingWorkerError, "identity"):
                build_cf1_streaming_worker(expected_stack_sha256=STACK_SHA256)

    def test_builder_synchronizes_cuda_before_returning_verified_engine(self):
        log = []

        class FakeCuda:
            @staticmethod
            def synchronize(device):
                log.append(("synchronize", device))

        runtime = SimpleNamespace(
            torch=SimpleNamespace(cuda=FakeCuda()),
            device="cuda:0",
        )
        built = SimpleNamespace(stack_sha256=STACK_SHA256)

        def build_runtime():
            log.append(("build-runtime",))
            return runtime

        def build_engine(value, *, frame_encoding_profile):
            self.assertIs(value, runtime)
            self.assertEqual(frame_encoding_profile, "png-c1-lossless-v1")
            log.append(("build-engine",))
            return built

        with (
            mock.patch.object(
                cf_streaming_worker,
                "build_cf1_runtime",
                side_effect=build_runtime,
            ),
            mock.patch.object(
                cf_streaming_worker,
                "CF1StreamingWorker",
                side_effect=build_engine,
            ),
        ):
            result = build_cf1_streaming_worker(expected_stack_sha256=STACK_SHA256)
        self.assertIs(result, built)
        self.assertEqual(
            log,
            [
                ("build-runtime",),
                ("build-engine",),
                ("synchronize", "cuda:0"),
            ],
        )

    def test_real_backend_factory_is_exact_and_rejects_bad_job_shape_locally(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            frame_encoding_profile="jpeg-q90-cpu-v1",
            **RUNTIME_LAUNCH,
        )
        self.assertIsInstance(backend, CF1ProcessStreamingBackend)
        self.assertEqual(
            backend.worker_script,
            Path(cf_streaming_process_worker.__file__).resolve(),
        )
        self.assertEqual(
            backend.worker_args,
            (
                "--worker-code-sha256",
                WORKER_BUNDLE_SHA256,
                "--frame-encoding-profile",
                "jpeg-q90-cpu-v1",
            ),
        )
        self.assertEqual(backend.frame_encoding_profile, "jpeg-q90-cpu-v1")
        self.assertEqual(
            backend.expected_worker_code_sha256,
            WORKER_BUNDLE_SHA256,
        )
        self.assertEqual(
            backend.worker_environment,
            {
                "PYTHONUNBUFFERED": "1",
                "CF1_RUNTIME_IMAGE_INDEX_DIGEST": IMAGE_INDEX_DIGEST,
                "CF1_RUNTIME_IMAGE_DIGEST": IMAGE_DIGEST,
                "CF1_RUNTIME_IMAGE_CONFIG_DIGEST": IMAGE_CONFIG_DIGEST,
                "CF1_RUNTIME_ENVIRONMENT_ROOT": "/runtime/venv",
                "CF1_RUNTIME_DISTRIBUTION_PATH": ("/runtime/venv/site-packages"),
                "CF1_RUNTIME_WHEELHOUSE": "/runtime/wheelhouse",
            },
        )
        self.assertEqual(backend.max_latent_frames, 21)
        self.assertTrue(backend.require_warm_start)
        self.assertGreater(backend.startup_timeout_s, 30)
        self.assertGreater(
            backend.registry_chunk_timeout_s,
            backend.startup_timeout_s + backend.reap_timeout_s,
        )

        for request in (
            StreamRequest("c", "bad-latents", "prompt", 0, latent_frames=20),
            StreamRequest("c", "bad-seed", "prompt", -1, latent_frames=21),
            StreamRequest("c", "huge-seed", "prompt", 2**32, latent_frames=21),
        ):
            with self.subTest(job_id=request.job_id):
                with self.assertRaises(StreamProtocolError):
                    backend.stream(request)

    def test_real_backend_accepts_only_its_boot_bound_jpeg_payloads(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            frame_encoding_profile="jpeg-q90-cpu-v1",
            **RUNTIME_LAUNCH,
        )

        self.assertEqual(
            backend._validated_frame_media_type(
                "image/jpeg",
                (JPEG_832X480,),
            ),
            "image/jpeg",
        )
        for media_type, payloads in (
            ("image/png", (PNG_1X1,)),
            ("image/jpeg", (b"not-a-jpeg",)),
        ):
            with self.subTest(media_type=media_type, payloads=payloads):
                with self.assertRaises(WorkerProtocolError):
                    backend._validated_frame_media_type(media_type, payloads)

    def test_real_worker_bundle_binds_engine_and_protocol_helper_bytes(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        self.assertEqual(
            tuple(path.name for path in backend.worker_bundle_paths),
            (
                "__init__.py",
                "cf_streaming_worker.py",
                "streaming_process.py",
                "streaming_process_worker.py",
                "cf_cuda_adapter.py",
                "cf_cuda_generator.py",
                "cf_cuda_session.py",
                "cf_cuda_smoke.py",
                "cf_attention_probe.py",
                "cf_runtime_evidence.py",
                "cf_runtime_preflight.py",
                "generation_preflight.py",
                "model_asset_preflight.py",
                "png_validation.py",
                "streaming_service.py",
            ),
        )
        baseline = worker_bundle_sha256(
            backend.worker_script,
            backend.worker_bundle_paths,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker_copy = root / backend.worker_script.name
            worker_copy.write_bytes(backend.worker_script.read_bytes())
            copies = []
            for path in backend.worker_bundle_paths:
                copy = root / path.name
                copy.write_bytes(path.read_bytes())
                copies.append(copy)
            copied = worker_bundle_sha256(worker_copy, copies)
            self.assertEqual(copied, baseline)
            for candidate in (worker_copy, *copies):
                with self.subTest(path=candidate.name):
                    original = candidate.read_bytes()
                    candidate.write_bytes(original + b"\n# mutation\n")
                    self.assertNotEqual(
                        worker_bundle_sha256(worker_copy, copies),
                        baseline,
                    )
                    candidate.write_bytes(original)

    def test_real_worker_bundle_covers_recursive_local_import_closure(self):
        protocol = Path(__file__).resolve().parents[1] / "streaming_process_protocol.py"
        covered = {
            Path(cf_streaming_process_worker.__file__).resolve(),
            protocol.resolve(),
            *(path.resolve() for path in cf_streaming_worker.REAL_WORKER_BUNDLE_PATHS),
        }
        reached = _reachable_bench_modules(
            Path(cf_streaming_process_worker.__file__).resolve()
        )
        self.assertEqual(reached - covered, set())

    def test_parent_and_preimport_child_bundle_inventories_match(self):
        self.assertEqual(
            tuple(path.name for path in cf_streaming_worker.REAL_WORKER_BUNDLE_PATHS),
            cf_streaming_process_worker._PREIMPORT_COMPANION_NAMES,
        )


class CF1ProcessBackendWarmTests(unittest.IsolatedAsyncioTestCase):
    def ready_backend_for_acceptance_termination(
        self,
        *,
        pid: int = 43123,
        worker_instance_id: str = "acceptance-worker-instance",
    ):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        process = SimpleNamespace(pid=pid, poll=mock.Mock(return_value=None))
        socket_token = object()
        backend._state = "ready"
        backend._process = process
        backend._socket = socket_token
        backend._worker_instance_id = worker_instance_id
        return backend, process, socket_token

    async def abandon_acceptance_termination_backend(self, backend):
        backend._process = None
        backend._socket = None
        backend._state = "stopped"
        await backend.close()

    async def test_acceptance_termination_signals_exact_owned_worker_group_only(self):
        backend, process, socket_token = self.ready_backend_for_acceptance_termination()
        try:
            with (
                mock.patch.object(os, "getpgid", return_value=process.pid) as getpgid,
                mock.patch.object(os, "getsid", return_value=process.pid) as getsid,
                mock.patch.object(os, "killpg") as killpg,
            ):
                evidence = await backend.terminate_idle_worker_for_acceptance(
                    expected_pid=process.pid,
                    expected_worker_instance_id="acceptance-worker-instance",
                )

            process.poll.assert_called_once_with()
            getpgid.assert_called_once_with(process.pid)
            getsid.assert_called_once_with(process.pid)
            killpg.assert_called_once_with(process.pid, signal.SIGKILL)
            self.assertEqual(evidence.pid, process.pid)
            self.assertEqual(
                evidence.worker_instance_id,
                "acceptance-worker-instance",
            )
            self.assertEqual(evidence.process_group_id, process.pid)
            self.assertEqual(evidence.session_id, process.pid)
            self.assertEqual(evidence.signal_name, "SIGKILL")

            # Deliberately retain the stale live-worker state.  The next real job,
            # rather than this acceptance hook, must observe and poison the death.
            self.assertIs(backend._process, process)
            self.assertIs(backend._socket, socket_token)
            self.assertEqual(backend._state, "ready")
            self.assertIsNone(backend._active_job_id)
            self.assertFalse(backend.poisoned)
            process.poll.assert_called_once_with()
        finally:
            await self.abandon_acceptance_termination_backend(backend)

    async def test_acceptance_termination_refuses_identity_and_lifecycle_mismatch(self):
        refusal_cases = (
            ("pid-mismatch", {"expected_pid": 43124}),
            (
                "instance-mismatch",
                {"expected_worker_instance_id": "different-instance"},
            ),
            ("busy", {"state": "busy", "active_job_id": "active-job"}),
            ("stopped", {"state": "stopped"}),
            ("closed", {"closed": True}),
            ("poisoned", {"poisoned": True}),
            ("missing-socket", {"missing_socket": True}),
            ("dead", {"poll_result": 7}),
        )
        for label, mutation in refusal_cases:
            with self.subTest(case=label):
                backend, process, _socket_token = (
                    self.ready_backend_for_acceptance_termination()
                )
                backend._state = mutation.get("state", "ready")
                backend._active_job_id = mutation.get("active_job_id")
                backend._closed = mutation.get("closed", False)
                backend._poisoned = mutation.get("poisoned", False)
                if mutation.get("missing_socket", False):
                    backend._socket = None
                process.poll.return_value = mutation.get("poll_result")
                expected_pid = mutation.get("expected_pid", process.pid)
                expected_worker_instance_id = mutation.get(
                    "expected_worker_instance_id",
                    "acceptance-worker-instance",
                )
                try:
                    with (
                        mock.patch.object(os, "getpgid") as getpgid,
                        mock.patch.object(os, "getsid") as getsid,
                        mock.patch.object(os, "killpg") as killpg,
                    ):
                        with self.assertRaises(StreamProtocolError):
                            await backend.terminate_idle_worker_for_acceptance(
                                expected_pid=expected_pid,
                                expected_worker_instance_id=(
                                    expected_worker_instance_id
                                ),
                            )
                    getpgid.assert_not_called()
                    getsid.assert_not_called()
                    killpg.assert_not_called()
                finally:
                    backend._closed = False
                    backend._poisoned = False
                    backend._active_job_id = None
                    await self.abandon_acceptance_termination_backend(backend)

    async def test_acceptance_termination_refuses_unowned_or_vanished_group(self):
        ownership_cases = (
            ("wrong-process-group", 43124, 43123, None),
            ("wrong-session", 43123, 43124, None),
            ("vanished", None, None, ProcessLookupError()),
        )
        for label, process_group_id, session_id, lookup_error in ownership_cases:
            with self.subTest(case=label):
                backend, process, _socket_token = (
                    self.ready_backend_for_acceptance_termination()
                )
                try:
                    getpgid_effect = (
                        lookup_error if lookup_error is not None else process_group_id
                    )
                    with (
                        mock.patch.object(
                            os,
                            "getpgid",
                            side_effect=(
                                getpgid_effect
                                if isinstance(getpgid_effect, BaseException)
                                else None
                            ),
                            return_value=(
                                None
                                if isinstance(getpgid_effect, BaseException)
                                else getpgid_effect
                            ),
                        ),
                        mock.patch.object(os, "getsid", return_value=session_id),
                        mock.patch.object(os, "killpg") as killpg,
                    ):
                        with self.assertRaises(StreamProtocolError):
                            await backend.terminate_idle_worker_for_acceptance(
                                expected_pid=process.pid,
                                expected_worker_instance_id=(
                                    "acceptance-worker-instance"
                                ),
                            )
                    killpg.assert_not_called()
                finally:
                    await self.abandon_acceptance_termination_backend(backend)

    async def test_frozen_worker_bundle_drift_fails_before_process_creation(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        try:
            with (
                mock.patch(
                    "bench.cf_streaming_worker.worker_bundle_sha256",
                    return_value="8" * 64,
                ),
                mock.patch("bench.streaming_process.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(WorkerProtocolError, "expected digest"):
                    await backend.warm()
            popen.assert_not_called()
            self.assertTrue(backend.poisoned)
        finally:
            await backend.close()

    async def test_real_factory_freezes_every_validated_operational_bound(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        try:
            mutations = {
                "startup_timeout_s": float("inf"),
                "io_timeout_s": float("inf"),
                "reap_timeout_s": float("inf"),
                "registry_chunk_timeout_s": float("inf"),
                "registry_close_timeout_s": float("inf"),
                "max_header_bytes": 2**63,
                "max_frame_bytes": 2**63,
                "max_chunk_bytes": 2**63,
                "max_prompt_bytes": 2**63,
                "max_job_ids": 2**63,
                "stderr_tail_bytes": 0,
            }
            for name, value in mutations.items():
                with self.subTest(name=name):
                    with self.assertRaisesRegex(AttributeError, "immutable"):
                        setattr(backend, name, value)
        finally:
            await backend.close()

    async def test_explicit_warm_is_required_and_cold_state_never_spawns_in_job(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        request = StreamRequest("c", "job", "prompt", 0, latent_frames=21)
        with self.assertRaisesRegex(StreamProtocolError, "prewarmed"):
            backend.stream(request)

        async def fake_ensure_worker():
            backend._state = "ready"

        with (
            mock.patch.object(
                backend,
                "_ensure_worker",
                side_effect=fake_ensure_worker,
            ) as ensure,
            mock.patch.object(
                CF1ProcessStreamingBackend,
                "ready",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
        ):
            await backend.warm()
            ensure.assert_awaited_once()
            self.assertTrue(backend.ready)
            iterator = backend.stream(request)
            self.assertIsNotNone(iterator)
            await iterator.aclose()

            stale = backend.stream(
                StreamRequest("c", "stale-job", "prompt", 1, latent_frames=21)
            )

        backend._state = "stopped"
        with self.assertRaisesRegex(StreamProtocolError, "prewarmed"):
            backend.stream(StreamRequest("c", "job-two", "prompt", 1, latent_frames=21))
        with mock.patch.object(
            backend,
            "_spawn_worker",
            new_callable=mock.AsyncMock,
            side_effect=AssertionError("stale iterator attempted a cold spawn"),
        ) as spawn:
            with self.assertRaisesRegex(StreamProtocolError, "prewarmed"):
                await stale.__anext__()
        spawn.assert_not_awaited()

        for transitional_state in ("starting", "stopping"):
            backend._state = transitional_state
            with self.subTest(state=transitional_state):
                with self.assertRaisesRegex(StreamProtocolError, "not ready"):
                    backend.stream(
                        StreamRequest(
                            "c",
                            f"job-{transitional_state}",
                            "prompt",
                            2,
                            latent_frames=21,
                        )
                    )

    async def test_checked_in_candidate_lock_exits_before_hello_and_poisons(self):
        backend = CF1ProcessStreamingBackend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
            startup_timeout_s=0.75,
            io_timeout_s=0.25,
            reap_timeout_s=0.2,
            registry_chunk_timeout_s=1.0,
            registry_close_timeout_s=0.5,
        )
        try:
            with self.assertRaisesRegex(StreamProtocolError, "truncated"):
                await backend.warm()
            self.assertTrue(backend.poisoned)
            self.assertFalse(backend.ready)
            self.assertIsNone(backend.worker_pid)
            self.assertIn(b"RuntimeBootstrapError", backend.stderr_tail)
            with self.assertRaisesRegex(BackendFatalError, "poisoned"):
                backend.stream(StreamRequest("c", "after-refusal", "prompt", 0, 21))
        finally:
            await backend.close()

    async def test_dead_idle_worker_reaches_generic_poison_and_reap_path(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        backend._state = "ready"
        backend._process = SimpleNamespace(poll=lambda: 1)
        backend._socket = SimpleNamespace()
        request = StreamRequest("c", "dead-idle", "prompt", 0, 21)

        async def poison(_job_id=None):
            backend._poisoned = True
            backend._state = "poisoned"

        with (
            mock.patch.object(
                backend,
                "_ensure_worker",
                new_callable=mock.AsyncMock,
                side_effect=WorkerProtocolError("persistent worker exited"),
            ) as ensure,
            mock.patch.object(
                backend,
                "_poison_after_failure",
                new_callable=mock.AsyncMock,
                side_effect=poison,
            ) as poisoned,
        ):
            iterator = backend.stream(request)
            with self.assertRaisesRegex(WorkerProtocolError, "exited"):
                await iterator.__anext__()
        ensure.assert_awaited_once()
        poisoned.assert_awaited_once_with("dead-idle")
        self.assertTrue(backend.poisoned)

    async def test_real_factory_refuses_launch_configuration_downgrade(self):
        backend = build_cf1_process_streaming_backend(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_BUNDLE_SHA256,
            **RUNTIME_LAUNCH,
        )
        try:
            with mock.patch("bench.streaming_process.subprocess.Popen") as popen:
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    backend.worker_script = DEFAULT_WORKER_SCRIPT
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    backend.worker_bundle_paths = ()
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    backend.expected_worker_code_sha256 = "9" * 64
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    backend.frame_encoding_profile = "jpeg-q90-cpu-v1"
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    backend.require_warm_start = False
            popen.assert_not_called()
            self.assertIsNone(backend.worker_pid)
            self.assertFalse(backend.poisoned)
        finally:
            await backend.close()


class CF1StreamingProcessWorkerTests(unittest.TestCase):
    def test_worker_bundle_is_checked_before_project_imports(self):
        self.assertEqual(
            cf_streaming_process_worker._preimport_worker_bundle_sha256(),
            WORKER_BUNDLE_SHA256,
        )
        source = Path(cf_streaming_process_worker.__file__).read_text("utf-8")
        self.assertLess(
            source.index(
                "_authorize_preimport_worker_bundle("
                "_required_worker_code_sha256(sys.argv[1:]))"
            ),
            source.index("from bench.streaming_process_worker"),
        )

    def test_worker_redirects_pyc_lookup_before_importing_bench(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "probe_pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            module = package / "value.py"
            timestamp = int(time.time()) - 30
            module.write_text("VALUE = 'evil'\n", encoding="utf-8")
            os.utime(module, (timestamp, timestamp))
            py_compile.compile(
                str(module),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            module.write_text("VALUE = 'safe'\n", encoding="utf-8")
            os.utime(module, (timestamp, timestamp))

            ordinary = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        f"import sys; sys.path.insert(0, {str(root)!r}); "
                        "from probe_pkg.value import VALUE; print(VALUE)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(ordinary.stdout.strip(), "evil")

            cache_prefix = root / "guaranteed-empty-cache-prefix"
            hardened = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        f"import sys; sys.pycache_prefix={str(cache_prefix)!r}; "
                        "sys.dont_write_bytecode=True; "
                        f"sys.path.insert(0, {str(root)!r}); "
                        "from probe_pkg.value import VALUE; print(VALUE)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(hardened.stdout.strip(), "safe")
            self.assertFalse(cache_prefix.exists())

        source = Path(cf_streaming_process_worker.__file__).read_text("utf-8")
        self.assertLess(
            source.index("sys.pycache_prefix"),
            source.index("from bench.streaming_process_worker"),
        )

    def test_local_regular_bench_package_wins_a_later_shadow_package(self):
        project_root = Path(__file__).resolve().parents[2]
        local_init = project_root / "bench" / "__init__.py"
        self.assertTrue(local_init.is_file())
        with tempfile.TemporaryDirectory() as directory:
            shadow_root = Path(directory)
            shadow_bench = shadow_root / "bench"
            shadow_bench.mkdir()
            (shadow_bench / "__init__.py").write_text(
                "SOURCE = 'shadow'\n",
                encoding="utf-8",
            )
            observed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        "import pathlib, sys; "
                        f"sys.path.insert(0, {str(project_root)!r}); "
                        f"sys.path.insert(1, {str(shadow_root)!r}); "
                        "import bench; print(pathlib.Path(bench.__file__).resolve())"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(Path(observed.stdout.strip()), local_init.resolve())

    def test_protocol_boots_before_hello_and_spends_one_pull_per_next(self):
        parent, child = socket.socketpair()
        parent.settimeout(2)
        child.settimeout(2)
        log = []
        engine = FakeProtocolEngine(log, frame_media_type="image/jpeg")
        result = {}

        def build_engine(*, expected_stack_sha256, frame_encoding_profile):
            # Tokenizer libraries may create this non-secret runtime control while
            # the model boots. The HELLO attests what crossed exec, not trusted
            # library mutations made after the pre-import boundary.
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            log.append(("build", expected_stack_sha256, frame_encoding_profile))
            return engine

        def target():
            try:
                cf_streaming_process_worker.run(
                    child,
                    STACK_SHA256,
                    WORKER_BUNDLE_SHA256,
                    frame_encoding_profile="jpeg-q90-cpu-v1",
                    launch_sensitive_environment_names=(),
                )
            except EOFError:
                result["eof"] = True
            except BaseException as error:
                result["error"] = error
            finally:
                child.close()

        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(
                cf_streaming_process_worker,
                "build_cf1_streaming_worker",
                side_effect=build_engine,
            ),
        ):
            thread = threading.Thread(target=target)
            thread.start()
            try:
                hello, hello_payloads = receive_packet(parent)
                self.assertEqual(
                    log,
                    [("build", STACK_SHA256, "jpeg-q90-cpu-v1")],
                )
                self.assertEqual(hello["type"], "HELLO")
                self.assertEqual(hello["stack_sha256"], STACK_SHA256)
                self.assertEqual(hello["sensitive_environment_names"], [])
                self.assertEqual(hello_payloads, ())

                start = {
                    "type": "START",
                    "protocol_version": hello["protocol_version"],
                    "worker_instance_id": hello["worker_instance_id"],
                    "job_id": "job-real-protocol",
                    "chunk_index": -1,
                    "stack_sha256": STACK_SHA256,
                    "worker_code_sha256": hello["worker_code_sha256"],
                    "prompt": "a fox runs",
                    "prompt_sha256": hashlib.sha256(b"a fox runs").hexdigest(),
                    "seed": 11,
                    "latent_frames": 21,
                }
                send_packet(parent, start)
                started, payloads = receive_packet(parent)
                self.assertEqual(started["type"], "STARTED")
                self.assertEqual(payloads, ())
                self.assertEqual([item for item in log if item[0] == "pull"], [])

                first_frame_index = 0
                frame_hashes = []
                chunk_counts = []
                for index in range(21):
                    nonce = f"credit-{index:02d}-0123456789"
                    send_packet(
                        parent,
                        {
                            "type": "NEXT",
                            "protocol_version": hello["protocol_version"],
                            "worker_instance_id": hello["worker_instance_id"],
                            "job_id": "job-real-protocol",
                            "chunk_index": index,
                            "credit_nonce": nonce,
                        },
                    )
                    chunk, chunk_payloads = receive_packet(parent)
                    expected_count = 1 if index == 0 else 4
                    self.assertEqual(chunk["type"], "CHUNK")
                    self.assertEqual(chunk["chunk_index"], index)
                    self.assertEqual(chunk["credit_nonce"], nonce)
                    self.assertEqual(chunk["first_frame_index"], first_frame_index)
                    self.assertEqual(chunk["frame_media_type"], "image/jpeg")
                    self.assertEqual(len(chunk_payloads), expected_count)
                    self.assertTrue(
                        all(payload == JPEG_832X480 for payload in chunk_payloads)
                    )
                    self.assertEqual(
                        len([item for item in log if item[0] == "pull"]),
                        index + 1,
                    )
                    first_frame_index += expected_count
                    chunk_counts.append(expected_count)
                    frame_hashes.extend(
                        hashlib.sha256(payload).hexdigest()
                        for payload in chunk_payloads
                    )

                completion_nonce = "completion-credit-0123456789"
                send_packet(
                    parent,
                    {
                        "type": "NEXT",
                        "protocol_version": hello["protocol_version"],
                        "worker_instance_id": hello["worker_instance_id"],
                        "job_id": "job-real-protocol",
                        "chunk_index": 21,
                        "credit_nonce": completion_nonce,
                    },
                )
                complete, complete_payloads = receive_packet(parent)
                self.assertEqual(complete["type"], "COMPLETE")
                self.assertEqual(complete["credit_nonce"], completion_nonce)
                self.assertEqual(complete["chunk_frame_counts"], chunk_counts)
                self.assertEqual(complete["frame_payload_sha256"], frame_hashes)
                self.assertEqual(complete_payloads, ())
                self.assertEqual(log[-1], ("finish", 21))
            finally:
                parent.close()
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", result)
        self.assertTrue(result.get("eof"))

    def test_real_worker_captures_launch_environment_before_model_imports(self):
        source = Path(cf_streaming_process_worker.__file__).read_text("utf-8")
        self.assertLess(
            source.index(
                "_LAUNCH_SENSITIVE_ENVIRONMENT_NAMES = tuple("
                "sensitive_environment_names())"
            ),
            source.index("from bench.cf_streaming_worker"),
        )


if __name__ == "__main__":
    unittest.main()
