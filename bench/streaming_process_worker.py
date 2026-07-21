"""Dependency-free fake child for the persistent streaming process protocol."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import runpy
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_PROTOCOL_DEFINITION = runpy.run_path(
    str(Path(__file__).with_name("streaming_process_protocol.py"))
)
PROTOCOL_VERSION = _PROTOCOL_DEFINITION["WORKER_PROTOCOL_VERSION"]
MAX_LATENT_FRAMES = _PROTOCOL_DEFINITION["WORKER_PROTOCOL_MAX_LATENT_FRAMES"]
worker_bundle_sha256 = _PROTOCOL_DEFINITION["worker_bundle_sha256"]
if not isinstance(PROTOCOL_VERSION, str) or not PROTOCOL_VERSION:
    raise RuntimeError("worker protocol version definition is invalid")
if (
    isinstance(MAX_LATENT_FRAMES, bool)
    or not isinstance(MAX_LATENT_FRAMES, int)
    or MAX_LATENT_FRAMES <= 0
):
    raise RuntimeError("worker latent frame limit definition is invalid")

MAX_HEADER_BYTES = 64 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 16 * 1024 * 1024
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
FAULTS = (
    "bad-completion",
    "bad-frame-hash",
    "early-complete",
    "extra-chunk",
    "false-isolation",
    "fork-helper",
    "hang-handshake",
    "hang-next",
    "stderr-flood",
    "stray-after-complete",
    "sensitive-environment",
    "unsolicited",
    "wrong-digest",
    "wrong-index",
    "wrong-job",
    "wrong-latent-cap",
    "wrong-type",
    "wrong-worker",
)


class ProtocolFailure(Exception):
    pass


def canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ProtocolFailure("invalid JSON header") from error


def read_exact(sock: socket.socket, byte_count: int) -> bytes:
    parts: List[bytes] = []
    remaining = byte_count
    while remaining:
        part = sock.recv(min(remaining, 64 * 1024))
        if not part:
            raise EOFError
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def receive_packet(sock: socket.socket) -> Tuple[Dict[str, Any], Tuple[bytes, ...]]:
    prefix = read_exact(sock, 4)
    header_length = struct.unpack(">I", prefix)[0]
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise ProtocolFailure("header length outside bounds")
    encoded_header = read_exact(sock, header_length)
    try:
        header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolFailure("invalid JSON header") from error
    if not isinstance(header, dict) or canonical_json(header) != encoded_header:
        raise ProtocolFailure("non-canonical JSON header")
    lengths = header.get("payload_lengths")
    digests = header.get("payload_sha256")
    if not isinstance(lengths, list) or not isinstance(digests, list):
        raise ProtocolFailure("missing payload framing")
    if len(lengths) != len(digests):
        raise ProtocolFailure("payload framing count mismatch")
    total = 0
    for length in lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ProtocolFailure("invalid payload length")
        if length > MAX_FRAME_BYTES:
            raise ProtocolFailure("payload segment too large")
        total += length
        if total > MAX_CHUNK_BYTES:
            raise ProtocolFailure("packet payload too large")
    if not all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise ProtocolFailure("invalid payload digest")
    payloads: List[bytes] = []
    for length, digest in zip(lengths, digests):
        payload = read_exact(sock, length)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ProtocolFailure("payload digest mismatch")
        payloads.append(payload)
    return header, tuple(payloads)


def send_packet(
    sock: socket.socket,
    header: Mapping[str, Any],
    payloads: Sequence[bytes] = (),
    digest_override: Optional[Sequence[str]] = None,
) -> None:
    if "payload_lengths" in header or "payload_sha256" in header:
        raise ProtocolFailure("payload framing supplied twice")
    total = sum(len(payload) for payload in payloads)
    if any(len(payload) > MAX_FRAME_BYTES for payload in payloads):
        raise ProtocolFailure("payload segment too large")
    if total > MAX_CHUNK_BYTES:
        raise ProtocolFailure("packet payload too large")
    envelope = dict(header)
    envelope["payload_lengths"] = [len(payload) for payload in payloads]
    if digest_override is None:
        envelope["payload_sha256"] = [
            hashlib.sha256(payload).hexdigest() for payload in payloads
        ]
    else:
        envelope["payload_sha256"] = list(digest_override)
    encoded_header = canonical_json(envelope)
    if not encoded_header or len(encoded_header) > MAX_HEADER_BYTES:
        raise ProtocolFailure("header too large")
    sock.sendall(struct.pack(">I", len(encoded_header)) + encoded_header)
    for payload in payloads:
        sock.sendall(payload)


def require_exact_keys(header: Mapping[str, Any], keys: Sequence[str]) -> None:
    expected = set(keys) | {"payload_lengths", "payload_sha256"}
    if set(header) != expected:
        raise ProtocolFailure("message fields do not match protocol")


def require_string(value: object, expected: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ProtocolFailure("message string does not match active state")


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def hang_forever() -> None:
    while True:
        time.sleep(60)


def sensitive_environment_names() -> List[str]:
    markers = ("AUTH", "CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN")
    return sorted(
        name for name in os.environ if any(marker in name.upper() for marker in markers)
    )


def validate_start(
    header: Mapping[str, Any],
    instance_id: str,
    stack_sha256: str,
    worker_code_sha256: str,
) -> Dict[str, Any]:
    require_exact_keys(
        header,
        (
            "type",
            "protocol_version",
            "worker_instance_id",
            "job_id",
            "chunk_index",
            "stack_sha256",
            "worker_code_sha256",
            "prompt",
            "prompt_sha256",
            "seed",
            "latent_frames",
        ),
    )
    require_string(header["type"], "START")
    require_string(header["protocol_version"], PROTOCOL_VERSION)
    require_string(header["worker_instance_id"], instance_id)
    require_string(header["stack_sha256"], stack_sha256)
    require_string(header["worker_code_sha256"], worker_code_sha256)
    if header["chunk_index"] != -1 or not is_int(header["chunk_index"]):
        raise ProtocolFailure("START chunk index is invalid")
    job_id = header["job_id"]
    prompt = header["prompt"]
    if not isinstance(job_id, str) or not job_id:
        raise ProtocolFailure("job id is invalid")
    if not isinstance(prompt, str) or not prompt:
        raise ProtocolFailure("prompt is invalid")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    require_string(header["prompt_sha256"], prompt_sha256)
    seed = header["seed"]
    latent_frames = header["latent_frames"]
    if not is_int(seed):
        raise ProtocolFailure("seed is invalid")
    if (
        not is_int(latent_frames)
        or latent_frames <= 0
        or latent_frames > MAX_LATENT_FRAMES
    ):
        raise ProtocolFailure("latent frame count is invalid")
    return {
        "job_id": job_id,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "latent_frames": latent_frames,
        "next_index": 0,
        "frame_hashes": [],
        "chunk_frame_counts": [],
    }


def validate_next(
    header: Mapping[str, Any],
    active: Mapping[str, Any],
    instance_id: str,
) -> Tuple[int, str]:
    require_exact_keys(
        header,
        (
            "type",
            "protocol_version",
            "worker_instance_id",
            "job_id",
            "chunk_index",
            "credit_nonce",
        ),
    )
    require_string(header["type"], "NEXT")
    require_string(header["protocol_version"], PROTOCOL_VERSION)
    require_string(header["worker_instance_id"], instance_id)
    require_string(header["job_id"], active["job_id"])
    index = header["chunk_index"]
    if not is_int(index) or index != active["next_index"]:
        raise ProtocolFailure("NEXT index is invalid")
    credit_nonce = header["credit_nonce"]
    if not isinstance(credit_nonce, str) or len(credit_nonce) < 16:
        raise ProtocolFailure("NEXT credit is invalid")
    return index, credit_nonce


def common_job_header(
    message_type: str,
    instance_id: str,
    job_id: str,
    chunk_index: int,
    credit_nonce: Optional[str] = None,
) -> Dict[str, Any]:
    header: Dict[str, Any] = {
        "type": message_type,
        "protocol_version": PROTOCOL_VERSION,
        "worker_instance_id": instance_id,
        "job_id": job_id,
        "chunk_index": chunk_index,
    }
    if credit_nonce is not None:
        header["credit_nonce"] = credit_nonce
    return header


def send_chunk(
    sock: socket.socket,
    active: Dict[str, Any],
    instance_id: str,
    index: int,
    credit_nonce: str,
    fault: Optional[str],
) -> None:
    frame_count = 1 if index == 0 else 4
    payloads = tuple(PNG_1X1 for _ in range(frame_count))
    response_instance = (
        hashlib.sha256(b"wrong-worker").hexdigest()
        if fault == "wrong-worker" and index == 0
        else instance_id
    )
    response_job = (
        active["job_id"] + "-wrong"
        if fault == "wrong-job" and index == 0
        else active["job_id"]
    )
    response_index = index + 1 if fault == "wrong-index" and index == 0 else index
    header = common_job_header(
        "CHUNK",
        response_instance,
        response_job,
        response_index,
        credit_nonce,
    )
    header.update(
        {
            "first_frame_index": len(active["frame_hashes"]),
            "frame_count": frame_count,
            "frame_media_type": "image/png",
        }
    )
    digest_override = None
    if fault == "bad-frame-hash" and index == 0:
        digest_override = tuple("0" * 64 for _ in payloads)
    send_packet(sock, header, payloads, digest_override=digest_override)
    active["chunk_frame_counts"].append(frame_count)
    active["frame_hashes"].extend(
        hashlib.sha256(payload).hexdigest() for payload in payloads
    )
    active["next_index"] = index + 1


def send_complete(
    sock: socket.socket,
    active: Mapping[str, Any],
    instance_id: str,
    stack_sha256: str,
    worker_code_sha256: str,
    credit_nonce: str,
    bad: bool,
) -> None:
    header = common_job_header(
        "COMPLETE",
        instance_id,
        active["job_id"],
        active["latent_frames"],
        credit_nonce,
    )
    frame_count = len(active["frame_hashes"]) + (1 if bad else 0)
    header.update(
        {
            "stack_sha256": stack_sha256,
            "worker_code_sha256": worker_code_sha256,
            "prompt_sha256": active["prompt_sha256"],
            "seed": active["seed"],
            "chunk_count": active["latent_frames"],
            "chunk_frame_counts": list(active["chunk_frame_counts"]),
            "frame_count": frame_count,
            "frame_payload_sha256": list(active["frame_hashes"]),
        }
    )
    send_packet(sock, header)


def run(sock: socket.socket, stack_sha256: str, fault: Optional[str]) -> None:
    worker_path = Path(__file__).resolve()
    worker_code_sha256 = worker_bundle_sha256(worker_path)
    instance_id = hashlib.sha256(os.urandom(32)).hexdigest()

    if fault == "fork-helper":
        helper = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        os.write(2, ("helper-pid=" + str(helper.pid) + "\n").encode("ascii"))
    if fault == "stderr-flood":
        block = b"fake worker diagnostic\n" * 4096
        for _ in range(8):
            os.write(2, block)
    if fault == "hang-handshake":
        hang_forever()

    handshake_stack = "0" * 64 if fault == "wrong-digest" else stack_sha256
    send_packet(
        sock,
        {
            "type": "HELLO",
            "protocol_version": PROTOCOL_VERSION,
            "worker_instance_id": instance_id,
            "stack_sha256": handshake_stack,
            "worker_code_sha256": worker_code_sha256,
            "max_latent_frames": (
                MAX_LATENT_FRAMES + 1
                if fault == "wrong-latent-cap"
                else MAX_LATENT_FRAMES
            ),
            "isolated": (
                False if fault == "false-isolation" else bool(sys.flags.isolated)
            ),
            "sensitive_environment_names": (
                ["INJECTED_API_KEY"]
                if fault == "sensitive-environment"
                else sensitive_environment_names()
            ),
        },
    )

    active: Optional[Dict[str, Any]] = None
    while True:
        header, payloads = receive_packet(sock)
        if payloads:
            raise ProtocolFailure("control message included payloads")
        if active is None:
            active = validate_start(
                header, instance_id, stack_sha256, worker_code_sha256
            )
            send_packet(
                sock,
                common_job_header(
                    "STARTED", instance_id, active["job_id"], -1
                ),
            )
            if fault == "unsolicited":
                unsolicited = common_job_header(
                    "CHUNK", instance_id, active["job_id"], 0, "unearned-credit"
                )
                unsolicited.update(
                    {
                        "first_frame_index": 0,
                        "frame_count": 1,
                        "frame_media_type": "image/png",
                    }
                )
                send_packet(sock, unsolicited, (PNG_1X1,))
            continue

        index, credit_nonce = validate_next(header, active, instance_id)
        if fault == "hang-next":
            hang_forever()
        if fault == "wrong-type" and index == 0:
            wrong = common_job_header(
                "STARTED", instance_id, active["job_id"], index, credit_nonce
            )
            send_packet(sock, wrong)
            continue
        if fault == "early-complete" and index == 0:
            send_complete(
                sock,
                active,
                instance_id,
                stack_sha256,
                worker_code_sha256,
                credit_nonce,
                bad=False,
            )
            continue
        if index < active["latent_frames"]:
            send_chunk(sock, active, instance_id, index, credit_nonce, fault)
            continue
        if index != active["latent_frames"]:
            raise ProtocolFailure("completion NEXT index is invalid")
        if fault == "extra-chunk":
            send_chunk(sock, active, instance_id, index, credit_nonce, None)
            continue
        send_complete(
            sock,
            active,
            instance_id,
            stack_sha256,
            worker_code_sha256,
            credit_nonce,
            bad=fault == "bad-completion",
        )
        active = None
        if fault == "stray-after-complete":
            sock.sendall(b"x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--stack-sha256", required=True)
    parser.add_argument("--fault", choices=FAULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # ``pass_fds`` must cross the initial exec, but helpers launched by a future
    # model worker must never inherit the control channel.
    os.set_inheritable(args.ipc_fd, False)
    sock = socket.socket(fileno=args.ipc_fd)
    try:
        run(sock, args.stack_sha256, args.fault)
    except EOFError:
        return 0
    except (OSError, ProtocolFailure) as error:
        os.write(2, ("worker protocol failure: " + str(error) + "\n").encode("utf-8"))
        return 2
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
