"""Resumable, explicitly non-gating video understanding for CF++1 artifacts.

This module is the paid-call boundary for one development artifact.  It builds
and validates both provider requests before transport, persists an ``in_flight``
record before each request, and never retries an ambiguous outcome implicitly.
Credentials and inline media are deliberately absent from every persisted file.
"""

from __future__ import annotations

import copy
import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from bench.cf_video_artifact import validate_cf_video_artifact
from bench.gemini_video_judge import (
    GEMINI_ENDPOINT,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL_ID,
    GEMINI_RATING_SCHEMA,
    GEMINI_TEMPERATURE,
    GEMINI_VIDEO_FPS,
    build_gemini_request,
    parse_gemini_response,
    scrub_gemini_response,
)
from bench.quality_sweep import ProtocolError, canonical_sha256, load_json
from bench.video_judge import (
    PEGASUS_ENDPOINT,
    PEGASUS_MAX_TOKENS,
    PEGASUS_MODEL_ID,
    PEGASUS_RATING_SCHEMA,
    PEGASUS_TEMPERATURE,
    build_pegasus_request,
    build_rating_prompt,
    parse_pegasus_response,
    scrub_pegasus_response,
)


DEVELOPMENT_PURPOSE = "development-video-understanding-not-gate-evidence"
PROVIDERS = ("google", "twelvelabs")
BUILTIN_TRANSPORT_IDENTITIES = {
    "google": "builtin-urllib-post-gemini-v1",
    "twelvelabs": "builtin-urllib-post-pegasus-v1",
}
Transport = Callable[[dict[str, Any]], dict[str, Any]]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


def _local_output_lock(directory: Path) -> threading.Lock:
    key = str(directory.resolve())
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_output_transaction(directory: Path):
    """Serialize paid state transitions across threads and OS processes."""

    local_lock = _local_output_lock(directory)
    local_lock.acquire()
    directory_fd: int | None = None
    lock_fd: int | None = None
    locked = False
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(directory, directory_flags)
        lock_flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        lock_fd = os.open(
            ".development-video-understanding.lock",
            lock_flags,
            0o600,
            dir_fd=directory_fd,
        )
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ProtocolError("development understanding lock must be one regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        locked = True
        yield
    except OSError as exc:
        raise ProtocolError("cannot lock development understanding output") from exc
    finally:
        if locked and lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        if lock_fd is not None:
            os.close(lock_fd)
        if directory_fd is not None:
            os.close(directory_fd)
        local_lock.release()


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise ProtocolError("cannot durably synchronize development output") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_output_directory(path: str | Path) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise ProtocolError("development understanding output must not be a symlink")
    destination = requested.resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ProtocolError("development understanding output must be a directory")
        _fsync_directory(destination.parent)
        _fsync_directory(destination)
        return destination
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("development understanding output parent must exist")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_dir():
            raise ProtocolError("development understanding output race was not a directory")
    except OSError as exc:
        raise ProtocolError("cannot create development understanding output") from exc
    _fsync_directory(parent)
    _fsync_directory(destination)
    return destination


def _durable_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ProtocolError("development state path must not be a symlink")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_fields(value: Any, required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    missing = sorted(required - set(value))
    if missing:
        raise ProtocolError(f"{label} missing {missing[0]}")
    extra = sorted(set(value) - required)
    if extra:
        raise ProtocolError(f"{label} contains unsupported field {extra[0]}")
    return value


def _read_rubric(path: str | Path) -> tuple[str, str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ProtocolError("development rubric must be a regular file")
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolError("development rubric must be readable UTF-8") from exc
    if not text.strip():
        raise ProtocolError("development rubric must not be empty")
    return text, _sha256_bytes(raw)


def _scrub_gemini_request(
    request: dict[str, Any], media_sha256: str, media_bytes: bytes
) -> dict[str, Any]:
    request = _require_exact_fields(
        request, {"contents", "generationConfig"}, "Gemini wire request"
    )
    contents = request["contents"]
    if not isinstance(contents, list) or len(contents) != 1:
        raise ProtocolError("Gemini wire request must contain exactly one content")
    content = _require_exact_fields(contents[0], {"role", "parts"}, "Gemini content")
    if content["role"] != "user":
        raise ProtocolError("Gemini wire request role mismatch")
    parts = content["parts"]
    if not isinstance(parts, list) or len(parts) != 2:
        raise ProtocolError("Gemini wire request must contain exactly two parts")
    media_part = _require_exact_fields(
        parts[0], {"inlineData", "videoMetadata"}, "Gemini media part"
    )
    inline = _require_exact_fields(
        media_part["inlineData"], {"mimeType", "data"}, "Gemini inline media"
    )
    metadata = _require_exact_fields(
        media_part["videoMetadata"], {"fps"}, "Gemini video metadata"
    )
    text_part = _require_exact_fields(parts[1], {"text"}, "Gemini text part")
    encoded = inline["data"]
    if not isinstance(encoded, str):
        raise ProtocolError("Gemini inline media must be encoded text")
    expected_config = {
        "temperature": GEMINI_TEMPERATURE,
        "maxOutputTokens": GEMINI_MAX_TOKENS,
        "responseFormat": {
            "text": {
                "mimeType": "APPLICATION_JSON",
                "schema": GEMINI_RATING_SCHEMA,
            }
        },
    }
    if (
        inline["mimeType"] != "video/mp4"
        or metadata["fps"] != GEMINI_VIDEO_FPS
        or not isinstance(text_part["text"], str)
        or not text_part["text"]
        or request["generationConfig"] != expected_config
    ):
        raise ProtocolError("Gemini wire request identity mismatch")
    try:
        encoded_bytes = len(encoded.encode("ascii"))
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ProtocolError("Gemini inline media must be ASCII") from exc
    if decoded != media_bytes or _sha256_bytes(decoded) != media_sha256:
        raise ProtocolError("Gemini inline media does not match the artifact")
    scrubbed = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "media_sha256": media_sha256,
                            "decoded_bytes": len(media_bytes),
                            "encoded_bytes": encoded_bytes,
                        },
                        "videoMetadata": {"fps": GEMINI_VIDEO_FPS},
                    },
                    {"text": text_part["text"]},
                ],
            }
        ],
        "generationConfig": copy.deepcopy(expected_config),
    }
    if encoded in json.dumps(scrubbed, sort_keys=True, separators=(",", ":")):
        raise ProtocolError("Gemini scrubbed request retained inline media")
    return scrubbed


def _scrub_pegasus_request(
    request: dict[str, Any], media_sha256: str, media_bytes: bytes
) -> dict[str, Any]:
    request = _require_exact_fields(
        request,
        {
            "model_name",
            "video",
            "prompt",
            "temperature",
            "stream",
            "response_format",
            "max_tokens",
        },
        "TwelveLabs wire request",
    )
    video = _require_exact_fields(
        request["video"], {"type", "base64_string"}, "TwelveLabs inline media"
    )
    encoded = video["base64_string"]
    if not isinstance(encoded, str):
        raise ProtocolError("TwelveLabs inline media must be encoded text")
    expected_response_format = {
        "type": "json_schema",
        "json_schema": PEGASUS_RATING_SCHEMA,
    }
    if (
        request["model_name"] != PEGASUS_MODEL_ID
        or video["type"] != "base64_string"
        or not isinstance(request["prompt"], str)
        or not request["prompt"]
        or request["temperature"] != PEGASUS_TEMPERATURE
        or request["stream"] is not False
        or request["response_format"] != expected_response_format
        or request["max_tokens"] != PEGASUS_MAX_TOKENS
    ):
        raise ProtocolError("TwelveLabs wire request identity mismatch")
    try:
        encoded_bytes = len(encoded.encode("ascii"))
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ProtocolError("TwelveLabs inline media must be ASCII") from exc
    if decoded != media_bytes or _sha256_bytes(decoded) != media_sha256:
        raise ProtocolError("TwelveLabs inline media does not match the artifact")
    scrubbed = {
        "model_name": PEGASUS_MODEL_ID,
        "video": {
            "type": "inline-media-scrubbed",
            "media_sha256": media_sha256,
            "decoded_bytes": len(media_bytes),
            "encoded_bytes": encoded_bytes,
        },
        "prompt": request["prompt"],
        "temperature": PEGASUS_TEMPERATURE,
        "stream": False,
        "response_format": copy.deepcopy(expected_response_format),
        "max_tokens": PEGASUS_MAX_TOKENS,
    }
    if encoded in json.dumps(scrubbed, sort_keys=True, separators=(",", ":")):
        raise ProtocolError("TwelveLabs scrubbed request retained inline media")
    return scrubbed


def _provider_request(
    provider: str,
    *,
    request: dict[str, Any],
    scrubbed_request: dict[str, Any],
    artifact_manifest_sha256: str,
    media_sha256: str,
    media_byte_count: int,
    generation_prompt_sha256: str,
    rating_prompt_sha256: str,
    rubric_sha256: str,
    transport_identity: str,
) -> dict[str, Any]:
    if provider == "google":
        endpoint = GEMINI_ENDPOINT
        model_id = GEMINI_MODEL_ID
        schema = GEMINI_RATING_SCHEMA
        adapter = Path(__file__).with_name("gemini_video_judge.py")
    elif provider == "twelvelabs":
        endpoint = PEGASUS_ENDPOINT
        model_id = PEGASUS_MODEL_ID
        schema = PEGASUS_RATING_SCHEMA
        adapter = Path(__file__).with_name("video_judge.py")
    else:  # pragma: no cover - all callers use the closed provider set
        raise ProtocolError(f"unsupported development provider {provider}")
    identity = {
        "schema_version": 1,
        "purpose": DEVELOPMENT_PURPOSE,
        "authorizes_quality_claim": False,
        "authorizes_performance_claim": False,
        "provider": provider,
        "endpoint": endpoint,
        "model_id": model_id,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "media_sha256": media_sha256,
        "media_byte_count": media_byte_count,
        "generation_prompt_sha256": generation_prompt_sha256,
        "rating_prompt_sha256": rating_prompt_sha256,
        "rubric_sha256": rubric_sha256,
        "response_schema_sha256": canonical_sha256(schema),
        "adapter_sha256": _sha256_bytes(adapter.read_bytes()),
        "development_adapter_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "transport_identity": transport_identity,
        "wire_request_sha256": canonical_sha256(request),
        "scrubbed_request": scrubbed_request,
    }
    identity["request_identity_sha256"] = canonical_sha256(identity)
    return identity


def _prepare(
    artifact_manifest_path: str | Path,
    *,
    original_prompt: str,
    rubric_path: str | Path,
    transport_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        raise ProtocolError("original generation prompt is required")
    artifact = validate_cf_video_artifact(artifact_manifest_path)
    if not isinstance(artifact, dict):
        raise ProtocolError("CF video artifact validator returned an invalid result")
    required = {
        "artifact_manifest",
        "artifact_manifest_sha256",
        "media_path",
        "media_sha256",
        "media_bytes",
        "generation_prompt_sha256",
        "fps",
    }
    _require_exact_fields(artifact, required, "validated CF video artifact")
    artifact_manifest_sha256 = _require_sha256(
        artifact["artifact_manifest_sha256"], "artifact manifest SHA-256"
    )
    media_sha256 = _require_sha256(artifact["media_sha256"], "media SHA-256")
    generation_prompt_sha256 = _require_sha256(
        artifact["generation_prompt_sha256"], "generation prompt SHA-256"
    )
    media_bytes = artifact["media_bytes"]
    if not isinstance(media_bytes, bytes) or not media_bytes:
        raise ProtocolError("validated CF video artifact media must contain bytes")
    if _sha256_bytes(media_bytes) != media_sha256:
        raise ProtocolError("validated CF video artifact media SHA-256 mismatch")
    if _sha256_bytes(original_prompt.encode("utf-8")) != generation_prompt_sha256:
        raise ProtocolError("original generation prompt does not match the artifact")
    if artifact["fps"] != 16.0:
        raise ProtocolError("validated CF video artifact must use 16 fps")
    rubric_text, rubric_sha256 = _read_rubric(rubric_path)
    rating_prompt = build_rating_prompt(original_prompt, rubric_text)
    rating_prompt_sha256 = _sha256_bytes(rating_prompt.encode("utf-8"))
    identities = (
        {provider: "not-executed-preflight" for provider in PROVIDERS}
        if transport_identities is None
        else dict(transport_identities)
    )
    if set(identities) != set(PROVIDERS):
        raise ProtocolError("development transport identities must cover both providers")
    for provider, identity in identities.items():
        if not isinstance(identity, str) or re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{0,127}", identity
        ) is None:
            raise ProtocolError(f"{provider} transport identity is invalid")

    # Both full wire requests are built before either transport is callable.
    # Their builders enforce each provider's inline-size bound.
    google_wire = build_gemini_request(media_bytes, rating_prompt)
    twelvelabs_wire = build_pegasus_request(media_bytes, rating_prompt)
    provider_requests = {
        "google": _provider_request(
            "google",
            request=google_wire,
            scrubbed_request=_scrub_gemini_request(
                google_wire, media_sha256, media_bytes
            ),
            artifact_manifest_sha256=artifact_manifest_sha256,
            media_sha256=media_sha256,
            media_byte_count=len(media_bytes),
            generation_prompt_sha256=generation_prompt_sha256,
            rating_prompt_sha256=rating_prompt_sha256,
            rubric_sha256=rubric_sha256,
            transport_identity=identities["google"],
        ),
        "twelvelabs": _provider_request(
            "twelvelabs",
            request=twelvelabs_wire,
            scrubbed_request=_scrub_pegasus_request(
                twelvelabs_wire, media_sha256, media_bytes
            ),
            artifact_manifest_sha256=artifact_manifest_sha256,
            media_sha256=media_sha256,
            media_byte_count=len(media_bytes),
            generation_prompt_sha256=generation_prompt_sha256,
            rating_prompt_sha256=rating_prompt_sha256,
            rubric_sha256=rubric_sha256,
            transport_identity=identities["twelvelabs"],
        ),
    }
    return {
        "artifact": artifact,
        "media_bytes": media_bytes,
        "rating_prompt_sha256": rating_prompt_sha256,
        "rubric_sha256": rubric_sha256,
        "provider_requests": provider_requests,
        "wire_requests": {
            "google": google_wire,
            "twelvelabs": twelvelabs_wire,
        },
    }


def preflight_development_video_understanding(
    artifact_manifest_path: str | Path,
    *,
    original_prompt: str,
    rubric_path: str | Path,
) -> dict[str, Any]:
    """Validate one artifact and both bounded requests without loading credentials."""

    prepared = _prepare(
        artifact_manifest_path,
        original_prompt=original_prompt,
        rubric_path=rubric_path,
    )
    artifact = prepared["artifact"]
    providers = {
        provider: {
            key: value
            for key, value in request.items()
            if key not in {"scrubbed_request"}
        }
        for provider, request in prepared["provider_requests"].items()
    }
    return {
        "schema_version": 1,
        "purpose": DEVELOPMENT_PURPOSE,
        "authorizes_quality_claim": False,
        "authorizes_performance_claim": False,
        "status": "preflight-complete-no-provider-call",
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "media_sha256": artifact["media_sha256"],
        "media_byte_count": len(prepared["media_bytes"]),
        "generation_prompt_sha256": artifact["generation_prompt_sha256"],
        "rating_prompt_sha256": prepared["rating_prompt_sha256"],
        "rubric_sha256": prepared["rubric_sha256"],
        "providers": providers,
    }


_IDENTITY_FIELDS = {
    "schema_version",
    "purpose",
    "authorizes_quality_claim",
    "authorizes_performance_claim",
    "provider",
    "endpoint",
    "model_id",
    "artifact_manifest_sha256",
    "media_sha256",
    "media_byte_count",
    "generation_prompt_sha256",
    "rating_prompt_sha256",
    "rubric_sha256",
    "response_schema_sha256",
    "adapter_sha256",
    "development_adapter_sha256",
    "transport_identity",
    "wire_request_sha256",
    "scrubbed_request",
    "request_identity_sha256",
}


def _validate_provider_state(
    value: Any, expected: dict[str, Any], provider: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{provider} provider state must be an object")
    status = value.get("status")
    status_fields = {
        "in_flight": set(),
        "uncertain": {"failure_kind"},
        "complete": {"raw_response", "raw_response_sha256", "rating"},
    }
    if status not in status_fields:
        raise ProtocolError(f"{provider} provider state status is invalid")
    required = _IDENTITY_FIELDS | {"status", "attempt"} | status_fields[status]
    _require_exact_fields(value, required, f"{provider} provider state")
    for field in _IDENTITY_FIELDS:
        if value[field] != expected[field]:
            raise ProtocolError(f"{provider} provider state identity mismatch")
    attempt = value["attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ProtocolError(f"{provider} provider attempt must be a positive integer")
    if status == "uncertain":
        if value["failure_kind"] not in {
            "transport-or-response-validation",
            "interrupted",
        }:
            raise ProtocolError(f"{provider} provider failure kind is invalid")
    if status == "complete":
        raw = value["raw_response"]
        if provider == "google":
            canonical = scrub_gemini_response(raw)
            parsed = parse_gemini_response(canonical)
        else:
            canonical = scrub_pegasus_response(raw)
            parsed = parse_pegasus_response(canonical)
        if canonical != raw:
            raise ProtocolError(f"{provider} provider response is not canonical")
        if value["raw_response_sha256"] != canonical_sha256(raw):
            raise ProtocolError(f"{provider} provider response SHA-256 mismatch")
        if value["rating"] != parsed:
            raise ProtocolError(f"{provider} provider parsed rating mismatch")
    return value


def _read_state(path: Path, expected: dict[str, Any], provider: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"{provider} provider state must be a regular file")
    return _validate_provider_state(load_json(path), expected, provider)


def _state_from_identity(
    identity: dict[str, Any], *, status: str, attempt: int, **extra: Any
) -> dict[str, Any]:
    return {**copy.deepcopy(identity), "status": status, "attempt": attempt, **extra}


def _call_provider(
    provider: str,
    *,
    identity: dict[str, Any],
    wire_request: dict[str, Any],
    state_path: Path,
    transport: Transport,
    attempt: int,
) -> dict[str, Any]:
    _durable_write_json(
        state_path,
        _state_from_identity(identity, status="in_flight", attempt=attempt),
    )
    try:
        response = transport(wire_request)
        if provider == "google":
            raw = scrub_gemini_response(response)
            rating = parse_gemini_response(raw)
        else:
            raw = scrub_pegasus_response(response)
            rating = parse_pegasus_response(raw)
    except BaseException as exc:
        failure_kind = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "transport-or-response-validation"
        )
        _durable_write_json(
            state_path,
            _state_from_identity(
                identity,
                status="uncertain",
                attempt=attempt,
                failure_kind=failure_kind,
            ),
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ProtocolError(
            f"{provider} provider outcome is uncertain; explicit retry-uncertain is required"
        ) from exc
    complete = _state_from_identity(
        identity,
        status="complete",
        attempt=attempt,
        raw_response=raw,
        raw_response_sha256=canonical_sha256(raw),
        rating=rating,
    )
    _durable_write_json(state_path, complete)
    return complete


def _aggregate(prepared: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact = prepared["artifact"]
    return {
        "schema_version": 1,
        "purpose": DEVELOPMENT_PURPOSE,
        "authorizes_quality_claim": False,
        "authorizes_performance_claim": False,
        "status": "complete",
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "media_sha256": artifact["media_sha256"],
        "media_byte_count": len(prepared["media_bytes"]),
        "generation_prompt_sha256": artifact["generation_prompt_sha256"],
        "rating_prompt_sha256": prepared["rating_prompt_sha256"],
        "rubric_sha256": prepared["rubric_sha256"],
        "providers": copy.deepcopy(states),
    }


def _reject_artifact_output_overlap(prepared: dict[str, Any], destination: Path) -> None:
    media_path = prepared["artifact"].get("media_path")
    if not isinstance(media_path, Path) or not media_path.is_absolute():
        raise ProtocolError("validated CF video artifact media path is invalid")
    artifact_directory = media_path.parent.resolve()
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(artifact_directory)
    except ValueError:
        return
    raise ProtocolError("development understanding output cannot modify the artifact directory")


def _run_locked_development_video_understanding(
    prepared: dict[str, Any],
    *,
    destination: Path,
    google_transport: Transport,
    twelvelabs_transport: Transport,
    retry: set[str],
) -> dict[str, Any]:
    # Read and validate every cache before making either paid call.
    states: dict[str, dict[str, Any] | None] = {}
    for provider in PROVIDERS:
        states[provider] = _read_state(
            destination / f"{provider}.json",
            prepared["provider_requests"][provider],
            provider,
        )
    aggregate_path = destination / "manifest.json"
    existing_aggregate: dict[str, Any] | None = None
    if aggregate_path.exists():
        if aggregate_path.is_symlink() or not aggregate_path.is_file():
            raise ProtocolError("development understanding manifest must be a regular file")
        existing_aggregate = load_json(aggregate_path)

    for provider, state in states.items():
        if state is not None and state["status"] in {"in_flight", "uncertain"}:
            if provider not in retry:
                raise ProtocolError(
                    f"{provider} provider state requires --retry-uncertain {provider}"
                )
        elif provider in retry:
            raise ProtocolError(f"{provider} has no uncertain provider state to retry")
    if existing_aggregate is not None:
        if any(state is None or state["status"] != "complete" for state in states.values()):
            raise ProtocolError("development understanding manifest exists before complete states")
        expected_aggregate = _aggregate(
            prepared, {name: state for name, state in states.items() if state is not None}
        )
        if existing_aggregate != expected_aggregate:
            raise ProtocolError("development understanding manifest identity mismatch")

    transports = {
        "google": google_transport,
        "twelvelabs": twelvelabs_transport,
    }
    completed: dict[str, dict[str, Any]] = {}
    for provider in PROVIDERS:
        previous = states[provider]
        if previous is not None and previous["status"] == "complete":
            completed[provider] = previous
            continue
        attempt = 1 if previous is None else previous["attempt"] + 1
        completed[provider] = _call_provider(
            provider,
            identity=prepared["provider_requests"][provider],
            wire_request=prepared["wire_requests"][provider],
            state_path=destination / f"{provider}.json",
            transport=transports[provider],
            attempt=attempt,
        )

    result = _aggregate(prepared, completed)
    _durable_write_json(aggregate_path, result)
    return result


def run_development_video_understanding(
    artifact_manifest_path: str | Path,
    *,
    original_prompt: str,
    rubric_path: str | Path,
    output_dir: str | Path,
    google_transport: Transport,
    twelvelabs_transport: Transport,
    retry_uncertain: Iterable[str] = (),
    transport_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Upload once per provider, resuming only outcomes proven complete."""

    retry = set(retry_uncertain)
    unsupported = sorted(retry - set(PROVIDERS))
    if unsupported:
        raise ProtocolError(f"unsupported retry-uncertain provider {unsupported[0]}")
    effective_transport_identities = (
        {provider: "injected-unverified" for provider in PROVIDERS}
        if transport_identities is None
        else dict(transport_identities)
    )
    initial = _prepare(
        artifact_manifest_path,
        original_prompt=original_prompt,
        rubric_path=rubric_path,
        transport_identities=effective_transport_identities,
    )
    requested_destination = Path(output_dir)
    _reject_artifact_output_overlap(initial, requested_destination)
    destination = _ensure_output_directory(requested_destination)
    with _exclusive_output_transaction(destination):
        prepared = _prepare(
            artifact_manifest_path,
            original_prompt=original_prompt,
            rubric_path=rubric_path,
            transport_identities=effective_transport_identities,
        )
        _reject_artifact_output_overlap(prepared, destination)
        return _run_locked_development_video_understanding(
            prepared,
            destination=destination,
            google_transport=google_transport,
            twelvelabs_transport=twelvelabs_transport,
            retry=retry,
        )
