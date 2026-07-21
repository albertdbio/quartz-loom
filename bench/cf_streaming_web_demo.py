"""Opt-in loopback browser server backed by one persistent CF++1 H100 worker."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
from typing import Any

from bench.cf_streaming_worker import (
    CF1_BROWSER_FRAME_ENCODING_PROFILE,
    CF1_FRAME_ENCODING_PROFILES,
    build_cf1_process_streaming_backend,
)
from bench.prompt_resolution import (
    GeminiFlashLitePromptResolver,
    PromptResolutionError,
    PromptResolver,
)
from bench.streaming_websocket import BrowserStreamingServer


def _build_prompt_resolver_from_environment() -> PromptResolver:
    """Consume the parent-only Gemini credential after the CUDA worker is warm."""

    api_key = os.environ.pop("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for browser prompt resolution")
    try:
        return GeminiFlashLitePromptResolver(api_key)
    except PromptResolutionError:
        raise RuntimeError("GEMINI_API_KEY is invalid for browser prompt resolution") from None


async def _warm_or_stop(backend: Any, shutdown_event: asyncio.Event) -> bool:
    """Warm once, cancelling and awaiting the worker if shutdown wins the race."""

    warm_task = asyncio.create_task(backend.warm())
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    tasks = (warm_task, shutdown_task)
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if warm_task.done():
            await warm_task
            return not shutdown_event.is_set()
        warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await warm_task
        return False
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _serve(
    *,
    host: str,
    port: int,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    frame_encoding_profile: str,
    runtime_image_index_digest: str,
    runtime_image_digest: str,
    runtime_image_config_digest: str,
    runtime_environment_root: str,
    runtime_distribution_path: str,
    runtime_wheelhouse: str,
    shutdown_event: asyncio.Event,
    prompt_resolver: PromptResolver | None = None,
) -> None:
    backend = build_cf1_process_streaming_backend(
        expected_stack_sha256=expected_stack_sha256,
        expected_worker_code_sha256=expected_worker_code_sha256,
        frame_encoding_profile=frame_encoding_profile,
        runtime_image_index_digest=runtime_image_index_digest,
        runtime_image_digest=runtime_image_digest,
        runtime_image_config_digest=runtime_image_config_digest,
        runtime_environment_root=runtime_environment_root,
        runtime_distribution_path=runtime_distribution_path,
        runtime_wheelhouse=runtime_wheelhouse,
    )
    server: BrowserStreamingServer | None = None
    try:
        if (
            backend.stack_sha256 != expected_stack_sha256
            or backend.expected_worker_code_sha256 != expected_worker_code_sha256
        ):
            raise RuntimeError("CF++1 browser backend launch identity changed")
        if not await _warm_or_stop(backend, shutdown_event):
            return
        if prompt_resolver is None:
            prompt_resolver = _build_prompt_resolver_from_environment()
        server = BrowserStreamingServer(
            backend=backend,
            host=host,
            port=port,
            max_prompt_bytes=backend.max_prompt_bytes,
            max_chunk_bytes=backend.max_chunk_bytes,
            max_jobs_per_connection=backend.max_job_ids,
            backend_chunk_timeout_s=backend.registry_chunk_timeout_s,
            backend_close_timeout_s=backend.registry_close_timeout_s,
            disconnect_timeout_s=backend.registry_close_timeout_s,
            demo_backend_kind="cf1",
            prompt_resolver=prompt_resolver,
            prompt_resolution_timeout_s=4.0,
        )
        await server.start()
        print(f"CF++1 browser streaming at {server.origin}/", flush=True)
        await shutdown_event.wait()
    finally:
        try:
            if server is not None:
                await server.close()
        finally:
            await backend.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--expected-stack-sha256", required=True)
    parser.add_argument("--expected-worker-code-sha256", required=True)
    parser.add_argument(
        "--frame-encoding-profile",
        choices=sorted(CF1_FRAME_ENCODING_PROFILES),
        default=CF1_BROWSER_FRAME_ENCODING_PROFILE,
    )
    parser.add_argument("--runtime-image-index-digest", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--runtime-image-config-digest", required=True)
    parser.add_argument("--runtime-environment-root", required=True)
    parser.add_argument("--runtime-distribution-path", required=True)
    parser.add_argument("--runtime-wheelhouse", required=True)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    try:
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(shutdown_signal, shutdown_event.set)
            installed.append(shutdown_signal)
        await _serve(
            host=args.host,
            port=args.port,
            expected_stack_sha256=args.expected_stack_sha256,
            expected_worker_code_sha256=args.expected_worker_code_sha256,
            frame_encoding_profile=args.frame_encoding_profile,
            runtime_image_index_digest=args.runtime_image_index_digest,
            runtime_image_digest=args.runtime_image_digest,
            runtime_image_config_digest=args.runtime_image_config_digest,
            runtime_environment_root=args.runtime_environment_root,
            runtime_distribution_path=args.runtime_distribution_path,
            runtime_wheelhouse=args.runtime_wheelhouse,
            shutdown_event=shutdown_event,
        )
    finally:
        for shutdown_signal in installed:
            loop.remove_signal_handler(shutdown_signal)


def main(argv: list[str] | None = None) -> int:
    asyncio.run(_run(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
