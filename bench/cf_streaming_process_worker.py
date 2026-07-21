"""Isolated real-CUDA child for the frozen persistent process protocol.

There are no fake, skip-preflight, or alternate-factory command-line paths in
this entrypoint.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PREIMPORT_COMPANION_NAMES = (
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
)
_LAUNCH_SENSITIVE_ENVIRONMENT_NAMES: tuple[str, ...] | None = None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_worker_code_sha256(argv: list[str]) -> str:
    indices = [
        index
        for index, argument in enumerate(argv)
        if argument == "--worker-code-sha256"
    ]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise RuntimeError("real worker expected bundle digest is missing")
    value = argv[indices[0] + 1]
    if not _is_sha256(value):
        raise RuntimeError("real worker expected bundle digest is invalid")
    return value


def _preimport_worker_bundle_sha256() -> str:
    """Reproduce the shared bundle digest without importing project code."""

    bench_root = _PROJECT_ROOT / "bench"
    paths = [
        (b"protocol", bench_root / "streaming_process_protocol.py"),
        (b"worker", Path(__file__).resolve()),
    ]
    paths.extend(
        (b"companion:" + name.encode("utf-8"), bench_root / name)
        for name in _PREIMPORT_COMPANION_NAMES
    )
    digest = hashlib.sha256()
    for label, path in paths:
        payload = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _authorize_preimport_worker_bundle(expected_worker_code_sha256: str) -> None:
    if _preimport_worker_bundle_sha256() != expected_worker_code_sha256:
        raise RuntimeError("real worker bundle does not match expected digest")


# The supervisor executes this file directly under ``python -I -S``.  Redirect
# bytecode lookup before importing project modules so a timestamp-valid local
# ``__pycache__`` entry cannot substitute bytes that the bundle hash did not
# authorize.  ``dont_write_bytecode`` keeps the random, nonexistent prefix
# read-only in practice.
if __name__ == "__main__":
    _pycache_candidate = Path("/tmp") / (
        f"cf1-worker-pycache-{os.getpid()}-{os.urandom(16).hex()}"
    )
    if _pycache_candidate.exists() or _pycache_candidate.is_symlink():
        raise RuntimeError("real worker bytecode cache prefix already exists")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = str(_pycache_candidate)
    sys.pycache_prefix = str(_pycache_candidate)
    sys.dont_write_bytecode = True
    _authorize_preimport_worker_bundle(_required_worker_code_sha256(sys.argv[1:]))


# ``python -I -S`` deliberately omits the script directory, working directory,
# and automatic site-package initialization.
# Install only this entrypoint's resolved project root; the runtime authorizer
# subsequently byte-verifies every executable/model boundary it trusts.
if not (_PROJECT_ROOT / "bench" / "cf_cuda_adapter.py").is_file():
    raise RuntimeError("real worker project root is invalid")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.dont_write_bytecode = True

from bench.streaming_process_worker import (  # noqa: E402
    MAX_LATENT_FRAMES,
    PROTOCOL_VERSION,
    ProtocolFailure,
    common_job_header,
    receive_packet,
    send_complete,
    send_packet,
    sensitive_environment_names,
    validate_next,
    validate_start,
    worker_bundle_sha256,
)

# Attest the exact environment that crossed ``exec`` before model/runtime imports
# may add benign controls such as TOKENIZERS_PARALLELISM. The parent separately
# validates the allowlisted environment immediately before Popen.
if __name__ == "__main__":
    _LAUNCH_SENSITIVE_ENVIRONMENT_NAMES = tuple(sensitive_environment_names())

from bench.cf_streaming_worker import (  # noqa: E402
    CF1_FRAME_ENCODING_PROFILES,
    CF1_LATENT_FRAMES,
    REAL_WORKER_BUNDLE_PATHS,
    build_cf1_streaming_worker,
)
from bench.streaming_service import DecodedChunk  # noqa: E402


if MAX_LATENT_FRAMES != CF1_LATENT_FRAMES:
    raise RuntimeError("real worker latent limit does not match protocol")


def _send_chunk(
    sock: socket.socket,
    active: Dict[str, Any],
    instance_id: str,
    index: int,
    credit_nonce: str,
    chunk: DecodedChunk,
    expected_frame_media_type: str,
) -> None:
    expected_frames = 1 if index == 0 else 4
    if (
        not isinstance(chunk, DecodedChunk)
        or chunk.frame_media_type != expected_frame_media_type
        or chunk.frame_count != expected_frames
    ):
        raise ProtocolFailure("real engine returned an invalid decoded chunk")
    header = common_job_header(
        "CHUNK",
        instance_id,
        active["job_id"],
        index,
        credit_nonce,
    )
    header.update(
        {
            "first_frame_index": len(active["frame_hashes"]),
            "frame_count": expected_frames,
            "frame_media_type": expected_frame_media_type,
        }
    )
    send_packet(sock, header, chunk.frame_payloads)
    active["chunk_frame_counts"].append(expected_frames)
    active["frame_hashes"].extend(
        hashlib.sha256(payload).hexdigest() for payload in chunk.frame_payloads
    )
    active["next_index"] = index + 1


def run(
    sock: socket.socket,
    expected_stack_sha256: str,
    expected_worker_code_sha256: str,
    *,
    frame_encoding_profile: str,
    launch_sensitive_environment_names: tuple[str, ...],
) -> None:
    """Boot once, then spend exactly one CUDA pull per earned NEXT credit."""

    if not isinstance(launch_sensitive_environment_names, tuple) or not all(
        isinstance(name, str) for name in launch_sensitive_environment_names
    ):
        raise ProtocolFailure("real worker launch environment report is invalid")

    worker_path = Path(__file__).resolve()
    worker_code_sha256 = worker_bundle_sha256(
        worker_path,
        REAL_WORKER_BUNDLE_PATHS,
    )
    if worker_code_sha256 != expected_worker_code_sha256:
        raise ProtocolFailure("real worker bundle does not match expected digest")
    engine = build_cf1_streaming_worker(
        expected_stack_sha256=expected_stack_sha256,
        frame_encoding_profile=frame_encoding_profile,
    )
    if engine.stack_sha256 != expected_stack_sha256:
        raise ProtocolFailure("real worker engine identity changed after bootstrap")
    if engine.frame_encoding_profile != frame_encoding_profile:
        raise ProtocolFailure("real worker frame encoding profile changed after bootstrap")
    if (
        worker_bundle_sha256(worker_path, REAL_WORKER_BUNDLE_PATHS)
        != expected_worker_code_sha256
    ):
        raise ProtocolFailure("real worker code changed during bootstrap")

    instance_id = hashlib.sha256(os.urandom(32)).hexdigest()
    send_packet(
        sock,
        {
            "type": "HELLO",
            "protocol_version": PROTOCOL_VERSION,
            "worker_instance_id": instance_id,
            "stack_sha256": engine.stack_sha256,
            "worker_code_sha256": worker_code_sha256,
            "max_latent_frames": MAX_LATENT_FRAMES,
            "isolated": bool(sys.flags.isolated),
            "sensitive_environment_names": list(launch_sensitive_environment_names),
        },
    )

    active: Optional[Dict[str, Any]] = None
    while True:
        header, payloads = receive_packet(sock)
        if payloads:
            raise ProtocolFailure("control message included payloads")
        if active is None:
            candidate = validate_start(
                header,
                instance_id,
                expected_stack_sha256,
                worker_code_sha256,
            )
            engine.start(
                prompt=header["prompt"],
                seed=header["seed"],
                latent_frames=header["latent_frames"],
            )
            active = candidate
            send_packet(
                sock,
                common_job_header(
                    "STARTED",
                    instance_id,
                    active["job_id"],
                    -1,
                ),
            )
            continue

        index, credit_nonce = validate_next(header, active, instance_id)
        if index < CF1_LATENT_FRAMES:
            chunk = engine.pull(index)
            _send_chunk(
                sock,
                active,
                instance_id,
                index,
                credit_nonce,
                chunk,
                engine.frame_media_type,
            )
            continue
        if index != CF1_LATENT_FRAMES:
            raise ProtocolFailure("completion NEXT index is invalid")
        engine.finish(index)
        send_complete(
            sock,
            active,
            instance_id,
            expected_stack_sha256,
            worker_code_sha256,
            credit_nonce,
            bad=False,
        )
        active = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--stack-sha256", required=True)
    parser.add_argument("--worker-code-sha256", required=True)
    parser.add_argument(
        "--frame-encoding-profile",
        required=True,
        choices=sorted(CF1_FRAME_ENCODING_PROFILES),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if _LAUNCH_SENSITIVE_ENVIRONMENT_NAMES is None:
        raise RuntimeError("real worker launch environment was not captured")
    os.set_inheritable(arguments.ipc_fd, False)
    sock = socket.socket(fileno=arguments.ipc_fd)
    try:
        run(
            sock,
            arguments.stack_sha256,
            arguments.worker_code_sha256,
            frame_encoding_profile=arguments.frame_encoding_profile,
            launch_sensitive_environment_names=(_LAUNCH_SENSITIVE_ENVIRONMENT_NAMES),
        )
    except EOFError:
        return 0
    except BaseException as error:
        diagnostic = f"real worker fatal: {type(error).__name__}\n".encode("ascii")
        with contextlib.suppress(OSError):
            os.write(2, diagnostic)
        return 2
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
