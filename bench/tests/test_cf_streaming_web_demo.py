from __future__ import annotations

import asyncio
import os
import signal
import unittest
from unittest import mock

from bench.cf_streaming_web_demo import (
    _build_prompt_resolver_from_environment,
    _parse_args,
    _run,
    _serve,
)
from bench.prompt_resolution import PromptResolutionError


STACK_SHA256 = "a" * 64
WORKER_SHA256 = "b" * 64
PROMPT_RESOLVER = object()
RUNTIME_LAUNCH = {
    "runtime_image_index_digest": "sha256:" + "c" * 64,
    "runtime_image_digest": "sha256:" + "d" * 64,
    "runtime_image_config_digest": "sha256:" + "e" * 64,
    "runtime_environment_root": "/runtime/env",
    "runtime_distribution_path": "/runtime/dist",
    "runtime_wheelhouse": "/runtime/wheels",
}


class FakeBackend:
    def __init__(
        self,
        lifecycle: list[str],
        *,
        warm_error: BaseException | None = None,
        hang_warm: bool = False,
    ) -> None:
        self.lifecycle = lifecycle
        self.warm_error = warm_error
        self.hang_warm = hang_warm
        self.warm_started = asyncio.Event()
        self.warm_cancelled = False
        self.stack_sha256 = STACK_SHA256
        self.expected_worker_code_sha256 = WORKER_SHA256
        self.registry_chunk_timeout_s = 930.0
        self.registry_close_timeout_s = 10.0
        self.max_chunk_bytes = 12_345_678
        self.max_prompt_bytes = 4096
        self.max_job_ids = 4

    async def warm(self) -> None:
        self.lifecycle.append("backend-warm")
        self.warm_started.set()
        if self.warm_error is not None:
            raise self.warm_error
        if self.hang_warm:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.warm_cancelled = True
                raise

    async def close(self) -> None:
        self.lifecycle.append("backend-close")


class FakeServer:
    def __init__(
        self,
        lifecycle: list[str],
        kwargs: dict[str, object],
        *,
        start_error: BaseException | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.kwargs = kwargs
        self.start_error = start_error
        self.started = asyncio.Event()
        self.origin = "http://127.0.0.1:8765"
        self.lifecycle.append("server-init")

    async def start(self) -> None:
        self.lifecycle.append("server-start")
        self.started.set()
        if self.start_error is not None:
            raise self.start_error

    async def close(self) -> None:
        self.lifecycle.append("server-close")


class ServerFactory:
    def __init__(
        self,
        lifecycle: list[str],
        *,
        start_error: BaseException | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.start_error = start_error
        self.calls: list[dict[str, object]] = []
        self.server: FakeServer | None = None

    def __call__(self, **kwargs: object) -> FakeServer:
        self.calls.append(kwargs)
        self.server = FakeServer(
            self.lifecycle,
            kwargs,
            start_error=self.start_error,
        )
        return self.server


def serve_kwargs(shutdown_event: asyncio.Event) -> dict[str, object]:
    return {
        "host": "127.0.0.1",
        "port": 8765,
        "expected_stack_sha256": STACK_SHA256,
        "expected_worker_code_sha256": WORKER_SHA256,
        "frame_encoding_profile": "jpeg-q90-cpu-v1",
        **RUNTIME_LAUNCH,
        "prompt_resolver": PROMPT_RESOLVER,
        "shutdown_event": shutdown_event,
    }


class CF1StreamingWebDemoTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_resolver_consumes_parent_key_without_disclosing_it(self) -> None:
        sentinel = object()
        with (
            mock.patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "parent-only-secret"},
                clear=True,
            ),
            mock.patch(
                "bench.cf_streaming_web_demo.GeminiFlashLitePromptResolver",
                return_value=sentinel,
            ) as build,
        ):
            self.assertIs(_build_prompt_resolver_from_environment(), sentinel)
            self.assertNotIn("GEMINI_API_KEY", os.environ)
        build.assert_called_once_with("parent-only-secret")

    def test_prompt_resolver_key_failure_is_sanitized(self) -> None:
        secret = "must-not-escape"
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=True),
            mock.patch(
                "bench.cf_streaming_web_demo.GeminiFlashLitePromptResolver",
                side_effect=PromptResolutionError(secret),
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                _build_prompt_resolver_from_environment()
            self.assertNotIn(secret, str(raised.exception))
            self.assertNotIn("GEMINI_API_KEY", os.environ)

    async def test_partial_signal_install_failure_removes_installed_handlers(
        self,
    ) -> None:
        class PartialSignalLoop:
            def __init__(self) -> None:
                self.installed: list[signal.Signals] = []
                self.removed: list[signal.Signals] = []

            def add_signal_handler(
                self,
                shutdown_signal: signal.Signals,
                _callback: object,
            ) -> None:
                if shutdown_signal == signal.SIGTERM:
                    raise RuntimeError("signal install failed")
                self.installed.append(shutdown_signal)

            def remove_signal_handler(self, shutdown_signal: signal.Signals) -> bool:
                self.removed.append(shutdown_signal)
                return True

        loop = PartialSignalLoop()
        with mock.patch(
            "bench.cf_streaming_web_demo.asyncio.get_running_loop",
            return_value=loop,
        ):
            with self.assertRaisesRegex(RuntimeError, "signal install failed"):
                await _run(mock.Mock())
        self.assertEqual(loop.installed, [signal.SIGINT])
        self.assertEqual(loop.removed, [signal.SIGINT])

    async def test_warms_before_listen_and_closes_server_before_backend(self) -> None:
        lifecycle: list[str] = []
        backend = FakeBackend(lifecycle)
        factory = ServerFactory(lifecycle)
        shutdown = asyncio.Event()
        with (
            mock.patch(
                "bench.cf_streaming_web_demo.build_cf1_process_streaming_backend",
                return_value=backend,
            ) as build,
            mock.patch(
                "bench.cf_streaming_web_demo.BrowserStreamingServer",
                side_effect=factory,
            ),
        ):
            task = asyncio.create_task(_serve(**serve_kwargs(shutdown)))
            while factory.server is None:
                await asyncio.sleep(0)
            await factory.server.started.wait()
            shutdown.set()
            await task

        build.assert_called_once_with(
            expected_stack_sha256=STACK_SHA256,
            expected_worker_code_sha256=WORKER_SHA256,
            frame_encoding_profile="jpeg-q90-cpu-v1",
            **RUNTIME_LAUNCH,
        )
        self.assertEqual(
            factory.calls,
            [
                {
                    "backend": backend,
                    "host": "127.0.0.1",
                    "port": 8765,
                    "max_prompt_bytes": 4096,
                    "max_chunk_bytes": 12_345_678,
                    "max_jobs_per_connection": 4,
                    "backend_chunk_timeout_s": 930.0,
                    "backend_close_timeout_s": 10.0,
                    "disconnect_timeout_s": 10.0,
                    "demo_backend_kind": "cf1",
                    "prompt_resolver": PROMPT_RESOLVER,
                    "prompt_resolution_timeout_s": 4.0,
                }
            ],
        )
        self.assertEqual(
            lifecycle,
            [
                "backend-warm",
                "server-init",
                "server-start",
                "server-close",
                "backend-close",
            ],
        )

    def test_browser_cli_defaults_to_the_boot_bound_jpeg_profile(self) -> None:
        arguments = _parse_args(
            [
                "--expected-stack-sha256",
                STACK_SHA256,
                "--expected-worker-code-sha256",
                WORKER_SHA256,
                "--runtime-image-index-digest",
                RUNTIME_LAUNCH["runtime_image_index_digest"],
                "--runtime-image-digest",
                RUNTIME_LAUNCH["runtime_image_digest"],
                "--runtime-image-config-digest",
                RUNTIME_LAUNCH["runtime_image_config_digest"],
                "--runtime-environment-root",
                RUNTIME_LAUNCH["runtime_environment_root"],
                "--runtime-distribution-path",
                RUNTIME_LAUNCH["runtime_distribution_path"],
                "--runtime-wheelhouse",
                RUNTIME_LAUNCH["runtime_wheelhouse"],
            ]
        )
        self.assertEqual(arguments.frame_encoding_profile, "jpeg-q90-cpu-v1")

    async def test_warm_failure_closes_backend_without_constructing_server(
        self,
    ) -> None:
        lifecycle: list[str] = []
        backend = FakeBackend(lifecycle, warm_error=RuntimeError("warm failed"))
        factory = ServerFactory(lifecycle)
        with (
            mock.patch(
                "bench.cf_streaming_web_demo.build_cf1_process_streaming_backend",
                return_value=backend,
            ),
            mock.patch(
                "bench.cf_streaming_web_demo.BrowserStreamingServer",
                side_effect=factory,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "warm failed"):
                await _serve(**serve_kwargs(asyncio.Event()))
        self.assertEqual(factory.calls, [])
        self.assertEqual(lifecycle, ["backend-warm", "backend-close"])

    async def test_shutdown_during_warm_cancels_worker_then_closes_backend(
        self,
    ) -> None:
        lifecycle: list[str] = []
        backend = FakeBackend(lifecycle, hang_warm=True)
        factory = ServerFactory(lifecycle)
        shutdown = asyncio.Event()
        with (
            mock.patch(
                "bench.cf_streaming_web_demo.build_cf1_process_streaming_backend",
                return_value=backend,
            ),
            mock.patch(
                "bench.cf_streaming_web_demo.BrowserStreamingServer",
                side_effect=factory,
            ),
        ):
            task = asyncio.create_task(_serve(**serve_kwargs(shutdown)))
            await backend.warm_started.wait()
            shutdown.set()
            await task
        self.assertTrue(backend.warm_cancelled)
        self.assertEqual(factory.calls, [])
        self.assertEqual(lifecycle, ["backend-warm", "backend-close"])

    async def test_server_start_failure_closes_both_owners_in_order(self) -> None:
        lifecycle: list[str] = []
        backend = FakeBackend(lifecycle)
        factory = ServerFactory(lifecycle, start_error=RuntimeError("listen failed"))
        with (
            mock.patch(
                "bench.cf_streaming_web_demo.build_cf1_process_streaming_backend",
                return_value=backend,
            ),
            mock.patch(
                "bench.cf_streaming_web_demo.BrowserStreamingServer",
                side_effect=factory,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "listen failed"):
                await _serve(**serve_kwargs(asyncio.Event()))
        self.assertEqual(
            lifecycle,
            [
                "backend-warm",
                "server-init",
                "server-start",
                "server-close",
                "backend-close",
            ],
        )


if __name__ == "__main__":
    unittest.main()
