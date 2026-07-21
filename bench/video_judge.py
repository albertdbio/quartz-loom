"""Strict, resumable video-quality judging with TwelveLabs Pegasus 1.5.

The provider sees only opaque blind media plus the registered prompt and rubric.
API credentials and base64 media are deliberately excluded from persisted evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from bench.quality_sweep import ProtocolError, canonical_sha256, load_json


PEGASUS_MODEL_ID = "pegasus1.5"
PEGASUS_ENDPOINT = "https://api.twelvelabs.io/v1.3/analyze"
PEGASUS_MAX_INLINE_BYTES = 30_000_000
PEGASUS_MAX_TOKENS = 2048
PEGASUS_TEMPERATURE = 0

SCORE_DIMENSIONS = (
    "prompt_adherence",
    "spatial_fidelity",
    "identity_consistency",
    "motion_naturalness",
    "temporal_artifacts",
)

# TwelveLabs accepts a documented subset of JSON Schema Draft 2020-12. Keep the
# provider schema inside that subset, then enforce exact fields locally.
PEGASUS_RATING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                name: {"type": "integer", "minimum": 1, "maximum": 10}
                for name in SCORE_DIMENSIONS
            },
            "required": list(SCORE_DIMENSIONS),
        },
        "first_third_quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "final_third_quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "failure_tags": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "scores",
        "first_third_quality",
        "final_third_quality",
        "failure_tags",
        "rationale",
    ],
}

DEFAULT_RUBRIC_TEXT = """Score the complete video from 1 to 10 on exactly five dimensions:
- prompt_adherence: requested subject, action, setting, camera, and attributes.
- spatial_fidelity: plausible anatomy, rigid objects, contact, perspective, and geometry.
- identity_consistency: identities, counts, colors, shapes, and layout persist over time.
- motion_naturalness: motion and camera evolution are smooth and physically plausible.
- temporal_artifacts: score high when flicker, popping, tearing, jumps, texture crawl, and late degradation are absent.

Use 1 for unusable, 3 for poor, 5 for mixed, 7 for good, and 9 for excellent.
Inspect the entire motion sequence, including the first and final thirds. Static but clean
output must not receive a high motion score. Return only the requested structured fields.
Do not infer or mention the generating system."""


Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_json_object(value: str, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ProtocolError(f"{label} contains duplicate JSON key {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(value, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return parsed


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


def _require_integer_score(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 1 <= value <= 10
    ):
        raise ProtocolError(f"{label} must be an integer score in [1, 10]")
    return value


def build_rating_prompt(original_prompt: str, rubric_text: str = DEFAULT_RUBRIC_TEXT) -> str:
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        raise ProtocolError("original prompt is required")
    if not isinstance(rubric_text, str) or not rubric_text.strip():
        raise ProtocolError("rating rubric is required")
    return (
        "Evaluate this full video against the original generation prompt below.\n\n"
        f"ORIGINAL PROMPT:\n{original_prompt.strip()}\n\n"
        f"REGISTERED RUBRIC:\n{rubric_text.strip()}"
    )


def build_pegasus_request(video_bytes: bytes, rating_prompt: str) -> dict[str, Any]:
    if not isinstance(video_bytes, bytes) or not video_bytes:
        raise ProtocolError("Pegasus video must contain bytes")
    encoded_video = base64.b64encode(video_bytes).decode("ascii")
    if len(encoded_video.encode("ascii")) > PEGASUS_MAX_INLINE_BYTES:
        raise ProtocolError("Pegasus base64 video exceeds the 30 MB inline limit")
    if not isinstance(rating_prompt, str) or not rating_prompt.strip():
        raise ProtocolError("Pegasus rating prompt is required")
    return {
        "model_name": PEGASUS_MODEL_ID,
        "video": {
            "type": "base64_string",
            "base64_string": encoded_video,
        },
        "prompt": rating_prompt,
        "temperature": PEGASUS_TEMPERATURE,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": PEGASUS_RATING_SCHEMA,
        },
        "max_tokens": PEGASUS_MAX_TOKENS,
    }


def scrub_pegasus_response(response: dict[str, Any]) -> dict[str, Any]:
    """Allowlist the only provider response fields safe and needed for evidence."""

    if not isinstance(response, dict):
        raise ProtocolError("Pegasus response must be an object")
    request_id = response.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("Pegasus response id is required")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ProtocolError("Pegasus response usage is required")
    safe_usage = {
        field: usage[field]
        for field in ("input_tokens", "output_tokens")
        if field in usage
    }
    return {
        "id": request_id,
        "data": response.get("data"),
        "finish_reason": response.get("finish_reason"),
        "usage": safe_usage,
    }


def parse_pegasus_response(response: dict[str, Any]) -> dict[str, Any]:
    response = scrub_pegasus_response(response)
    if response.get("finish_reason") != "stop":
        raise ProtocolError("Pegasus response finish_reason must equal stop")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ProtocolError("Pegasus response usage is required")
    for field in ("input_tokens", "output_tokens"):
        if field in usage and (
            not isinstance(usage[field], int)
            or isinstance(usage[field], bool)
            or usage[field] < 0
        ):
            raise ProtocolError(f"Pegasus usage {field} must be a non-negative integer")
    if "output_tokens" not in usage:
        raise ProtocolError("Pegasus response usage.output_tokens is required")
    data = response.get("data")
    if not isinstance(data, str) or not data.strip():
        raise ProtocolError("Pegasus response data is required")
    rating = _parse_json_object(data, "Pegasus structured rating")
    required = {
        "scores",
        "first_third_quality",
        "final_third_quality",
        "failure_tags",
        "rationale",
    }
    rating = _require_exact_fields(rating, required, "Pegasus structured rating")
    scores = _require_exact_fields(
        rating["scores"], set(SCORE_DIMENSIONS), "Pegasus scores"
    )
    normalized_scores = {
        name: _require_integer_score(scores[name], f"Pegasus score {name}")
        for name in SCORE_DIMENSIONS
    }
    first = _require_integer_score(
        rating["first_third_quality"], "Pegasus first_third_quality"
    )
    final = _require_integer_score(
        rating["final_third_quality"], "Pegasus final_third_quality"
    )
    failure_tags = rating["failure_tags"]
    if not isinstance(failure_tags, list):
        raise ProtocolError("Pegasus failure_tags must be an array")
    if any(not isinstance(tag, str) or not tag.strip() for tag in failure_tags):
        raise ProtocolError("Pegasus failure_tags must contain non-empty strings")
    if len(set(failure_tags)) != len(failure_tags):
        raise ProtocolError("Pegasus failure_tags must be unique")
    rationale = rating["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProtocolError("Pegasus rationale is required")
    return {
        "scores": normalized_scores,
        "first_third_quality": first,
        "final_third_quality": final,
        "failure_tags": list(failure_tags),
        "rationale": rationale.strip(),
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def ratings_from_pegasus_evidence(
    protocol: dict[str, Any],
    public_plan: dict[str, Any],
    evidence_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-parse archived provider responses and reproduce their raw rating rows."""

    report = _require_exact_fields(
        evidence_report,
        {
            "schema_version",
            "provider",
            "model_id",
            "endpoint",
            "protocol_sha256",
            "manifest_sha256",
            "blind_plan_sha256",
            "family_id",
            "rater_id",
            "pass_id",
            "records",
        },
        "Pegasus evidence report",
    )
    expected_identity = {
        "schema_version": 1,
        "provider": "twelvelabs",
        "model_id": PEGASUS_MODEL_ID,
        "endpoint": PEGASUS_ENDPOINT,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": public_plan.get("manifest_sha256"),
        "blind_plan_sha256": canonical_sha256(public_plan),
        "family_id": public_plan.get("family_id"),
        "rater_id": public_plan.get("rater_id"),
        "pass_id": public_plan.get("pass_id"),
    }
    if any(report.get(field) != value for field, value in expected_identity.items()):
        raise ProtocolError("Pegasus evidence report identity mismatch")
    public_records = public_plan.get("records")
    records = report["records"]
    if not isinstance(public_records, list) or not isinstance(records, list) or not records:
        raise ProtocolError("Pegasus evidence records must be a non-empty array")
    public_by_id = {
        record.get("blind_id"): record
        for record in public_records
        if isinstance(record, dict)
    }
    if len(public_by_id) != len(public_records) or None in public_by_id:
        raise ProtocolError("Pegasus public-plan blind IDs must be unique")

    seen: set[str] = set()
    ratings: list[dict[str, Any]] = []
    for value in records:
        record = _require_exact_fields(
            value,
            {
                "schema_version",
                "endpoint",
                "model_id",
                "blind_id",
                "asset_id",
                "media_sha256",
                "request",
                "raw_response",
                "raw_response_sha256",
            },
            "Pegasus evidence record",
        )
        blind_id = record["blind_id"]
        public = public_by_id.get(blind_id)
        if public is None or blind_id in seen:
            raise ProtocolError("Pegasus evidence blind IDs must exactly cover the plan")
        seen.add(blind_id)
        expected_record_identity = {
            "schema_version": 1,
            "endpoint": PEGASUS_ENDPOINT,
            "model_id": PEGASUS_MODEL_ID,
            "asset_id": public.get("asset_id"),
            "media_sha256": public.get("media_sha256"),
        }
        if any(
            record.get(field) != expected
            for field, expected in expected_record_identity.items()
        ):
            raise ProtocolError("Pegasus evidence record identity mismatch")
        request = _require_exact_fields(
            record["request"],
            {
                "endpoint",
                "model_id",
                "adapter_sha256",
                "blind_id",
                "asset_id",
                "media_sha256",
                "prompt_sha256",
                "rubric_sha256",
                "response_schema_sha256",
                "temperature",
                "max_tokens",
                "stream",
                "wire_request",
            },
            "Pegasus evidence request",
        )
        request_identity = {
            "endpoint": PEGASUS_ENDPOINT,
            "model_id": PEGASUS_MODEL_ID,
            "blind_id": blind_id,
            "asset_id": public.get("asset_id"),
            "media_sha256": public.get("media_sha256"),
            "rubric_sha256": protocol.get("evaluation", {}).get("rubric_sha256"),
            "response_schema_sha256": canonical_sha256(PEGASUS_RATING_SCHEMA),
            "temperature": PEGASUS_TEMPERATURE,
            "max_tokens": PEGASUS_MAX_TOKENS,
            "stream": False,
        }
        if any(request.get(field) != expected for field, expected in request_identity.items()):
            raise ProtocolError("Pegasus evidence request identity mismatch")
        _require_sha256(request["adapter_sha256"], "Pegasus adapter_sha256")
        wire = _require_exact_fields(
            request["wire_request"],
            {
                "model_name",
                "video",
                "prompt",
                "temperature",
                "stream",
                "response_format",
                "max_tokens",
            },
            "Pegasus scrubbed wire request",
        )
        if (
            wire["model_name"] != PEGASUS_MODEL_ID
            or wire["temperature"] != PEGASUS_TEMPERATURE
            or wire["stream"] is not False
            or wire["max_tokens"] != PEGASUS_MAX_TOKENS
            or wire["response_format"]
            != {"type": "json_schema", "json_schema": PEGASUS_RATING_SCHEMA}
        ):
            raise ProtocolError("Pegasus scrubbed wire request mismatch")
        prompt = wire["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ProtocolError("Pegasus evidence prompt is required")
        if request["prompt_sha256"] != _sha256_bytes(prompt.encode("utf-8")):
            raise ProtocolError("Pegasus evidence prompt_sha256 mismatch")
        video = _require_exact_fields(
            wire["video"],
            {"type", "media_sha256", "decoded_bytes", "encoded_bytes"},
            "Pegasus scrubbed video request",
        )
        if (
            video["type"] != "base64_string"
            or video["media_sha256"] != public.get("media_sha256")
            or not isinstance(video["decoded_bytes"], int)
            or isinstance(video["decoded_bytes"], bool)
            or video["decoded_bytes"] < 1
            or not isinstance(video["encoded_bytes"], int)
            or isinstance(video["encoded_bytes"], bool)
            or not 1 <= video["encoded_bytes"] <= PEGASUS_MAX_INLINE_BYTES
        ):
            raise ProtocolError("Pegasus scrubbed video request mismatch")
        raw_response = _require_exact_fields(
            record["raw_response"],
            {"id", "data", "finish_reason", "usage"},
            "Pegasus raw response",
        )
        canonical_response = scrub_pegasus_response(raw_response)
        if raw_response != canonical_response:
            raise ProtocolError("Pegasus raw response is not canonical scrubbed evidence")
        if record["raw_response_sha256"] != canonical_sha256(raw_response):
            raise ProtocolError("Pegasus raw evidence response hash mismatch")
        ratings.append(
            {
                "blind_id": blind_id,
                "media_sha256": record["media_sha256"],
                **parse_pegasus_response(canonical_response),
            }
        )
    if seen != set(public_by_id):
        raise ProtocolError("Pegasus evidence blind IDs must exactly cover the plan")
    return ratings


def load_twelvelabs_api_key(
    dotenv_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    configured = environment.get("TWELVELABS_API_KEY", "").strip()
    if configured:
        return configured
    path = Path(dotenv_path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProtocolError("TWELVELABS_API_KEY is not configured") from exc
    matches: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        if name.strip() != "TWELVELABS_API_KEY":
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        matches.append(value.strip())
    if len(matches) != 1 or not matches[0]:
        raise ProtocolError("TWELVELABS_API_KEY is not configured exactly once")
    return matches[0]


def post_pegasus(
    request_body: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    if not isinstance(api_key, str) or not api_key:
        raise ProtocolError("TWELVELABS_API_KEY is required")
    encoded = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        PEGASUS_ENDPOINT,
        data=encoded,
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # Provider error bodies are untrusted and may echo submitted values. Do
        # not surface them through logs where a credential could be retained.
        exc.read(2048)
        raise ProtocolError(f"TwelveLabs HTTP {exc.code}; response body withheld") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProtocolError(f"TwelveLabs request failed: {type(exc).__name__}") from exc
    try:
        return _parse_json_object(raw.decode("utf-8"), "TwelveLabs HTTP response")
    except UnicodeDecodeError as exc:
        raise ProtocolError("TwelveLabs HTTP response is not UTF-8") from exc


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _validate_plan_binding(protocol: dict[str, Any], plan: dict[str, Any]) -> None:
    if not isinstance(protocol, dict) or not isinstance(plan, dict):
        raise ProtocolError("protocol and blind plan must be objects")
    if plan.get("schema_version") != 1:
        raise ProtocolError("blind plan must use schema_version 1")
    if plan.get("protocol_sha256") != canonical_sha256(protocol):
        raise ProtocolError("blind plan protocol_sha256 mismatch")
    evaluation = protocol.get("evaluation")
    families = evaluation.get("families") if isinstance(evaluation, dict) else None
    if not isinstance(families, list):
        raise ProtocolError("protocol evaluation families are required")
    if any(not isinstance(item, dict) for item in families):
        raise ProtocolError("protocol evaluation families must contain objects")
    matching = [item for item in families if item.get("id") == plan.get("family_id")]
    if len(matching) != 1:
        raise ProtocolError("blind plan family is not uniquely registered")
    family = matching[0]
    if family.get("kind") != "model" or family.get("model_id") != PEGASUS_MODEL_ID:
        raise ProtocolError("blind plan family is not registered to pegasus1.5")
    if family.get("evidence_provider") != "twelvelabs":
        raise ProtocolError("blind plan family is not registered to TwelveLabs evidence")
    if family.get("readiness") != "quality-qualified":
        raise ProtocolError("Pegasus blind-plan execution requires a quality-qualified family")
    if plan.get("rater_id") not in family.get("rater_ids", []):
        raise ProtocolError("blind plan rater is not registered to the Pegasus family")
    passes = family.get("passes")
    if not isinstance(passes, int) or isinstance(passes, bool) or passes < 1:
        raise ProtocolError("registered Pegasus family passes must be a positive integer")
    pass_id = plan.get("pass_id")
    if (
        not isinstance(pass_id, int)
        or isinstance(pass_id, bool)
        or not 1 <= pass_id <= passes
    ):
        raise ProtocolError("blind plan pass is not registered to the Pegasus family")


def _request_evidence(
    record: dict[str, Any],
    rating_prompt: str,
    rubric_text: str,
    request_body: dict[str, Any],
    video_bytes: bytes,
) -> dict[str, Any]:
    wire_request = {key: value for key, value in request_body.items() if key != "video"}
    wire_request["video"] = {
        "type": request_body["video"]["type"],
        "media_sha256": record["media_sha256"],
        "decoded_bytes": len(video_bytes),
        "encoded_bytes": len(request_body["video"]["base64_string"].encode("ascii")),
    }
    return {
        "endpoint": PEGASUS_ENDPOINT,
        "model_id": PEGASUS_MODEL_ID,
        "adapter_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "blind_id": record["blind_id"],
        "asset_id": record["asset_id"],
        "media_sha256": record["media_sha256"],
        "prompt_sha256": _sha256_bytes(rating_prompt.encode("utf-8")),
        "rubric_sha256": _sha256_bytes(rubric_text.encode("utf-8")),
        "response_schema_sha256": canonical_sha256(PEGASUS_RATING_SCHEMA),
        "temperature": PEGASUS_TEMPERATURE,
        "max_tokens": PEGASUS_MAX_TOKENS,
        "stream": False,
        "wire_request": wire_request,
    }


def run_pegasus_plan(
    protocol: dict[str, Any],
    public_plan: dict[str, Any],
    asset_dir: str | Path,
    evidence_dir: str | Path,
    *,
    transport: Transport,
    rubric_text: str = DEFAULT_RUBRIC_TEXT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge every blind asset, checkpointing scrubbed responses for safe resume."""

    _validate_plan_binding(protocol, public_plan)
    if not isinstance(rubric_text, str) or not rubric_text:
        raise ProtocolError("Pegasus plan rubric must be non-empty UTF-8 text")
    evaluation = protocol.get("evaluation")
    registered_rubric_sha256 = (
        evaluation.get("rubric_sha256") if isinstance(evaluation, dict) else None
    )
    if registered_rubric_sha256 != _sha256_bytes(rubric_text.encode("utf-8")):
        raise ProtocolError("Pegasus plan does not use the registered rubric SHA-256")
    records = public_plan.get("records")
    if not isinstance(records, list) or not records:
        raise ProtocolError("blind plan records must be a non-empty array")
    blind_ids: set[str] = set()
    prepared: list[
        tuple[dict[str, Any], bytes, str, dict[str, Any], dict[str, Any]]
    ] = []
    asset_root = Path(asset_dir)
    for value in records:
        required = {
            "blind_id",
            "asset_id",
            "case_id",
            "prompt_id",
            "prompt",
            "slot",
            "media_sha256",
        }
        record = _require_exact_fields(value, required, "blind plan record")
        blind_id = record["blind_id"]
        if (
            not isinstance(blind_id, str)
            or re.fullmatch(r"[0-9a-f]{20}", blind_id) is None
            or blind_id in blind_ids
        ):
            raise ProtocolError("blind plan blind IDs must be present and unique")
        blind_ids.add(blind_id)
        asset_id = record["asset_id"]
        if (
            not isinstance(asset_id, str)
            or re.fullmatch(r"asset-[0-9a-f]{20}", asset_id) is None
        ):
            raise ProtocolError("blind plan asset_id must be an opaque asset identifier")
        if not isinstance(record["case_id"], str) or re.fullmatch(
            r"[0-9a-f]{20}", record["case_id"]
        ) is None:
            raise ProtocolError("blind plan case_id must be an opaque case identifier")
        if not isinstance(record["prompt_id"], str) or not record["prompt_id"]:
            raise ProtocolError("blind plan prompt_id is required")
        if not isinstance(record["slot"], str) or re.fullmatch(
            r"[A-Z]", record["slot"]
        ) is None:
            raise ProtocolError("blind plan slot must be one uppercase letter")
        if not isinstance(record["media_sha256"], str) or re.fullmatch(
            r"[0-9a-f]{64}", record["media_sha256"]
        ) is None:
            raise ProtocolError("blind plan media_sha256 must be a lowercase SHA-256")
        path = asset_root / f"{asset_id}.mp4"
        try:
            video_bytes = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"cannot load blind asset {asset_id}") from exc
        if not video_bytes:
            raise ProtocolError(f"blind asset {asset_id} is empty")
        actual_sha256 = _sha256_bytes(video_bytes)
        if actual_sha256 != record.get("media_sha256"):
            raise ProtocolError(f"blind asset {asset_id} media SHA-256 mismatch")
        rating_prompt = build_rating_prompt(record["prompt"], rubric_text)
        request_body = build_pegasus_request(video_bytes, rating_prompt)
        expected_request = _request_evidence(
            record,
            rating_prompt,
            rubric_text,
            request_body,
            video_bytes,
        )
        prepared.append(
            (record, video_bytes, rating_prompt, request_body, expected_request)
        )

    evidence_root = Path(evidence_dir)
    evidence_records: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    for record, video_bytes, rating_prompt, request_body, expected_request in sorted(
        prepared, key=lambda item: item[0]["blind_id"]
    ):
        cache_path = evidence_root / f"{record['blind_id']}.json"
        if cache_path.exists():
            cached = _require_exact_fields(
                load_json(cache_path),
                {
                    "schema_version",
                    "endpoint",
                    "model_id",
                    "blind_id",
                    "asset_id",
                    "media_sha256",
                    "request",
                    "raw_response",
                    "raw_response_sha256",
                },
                "cached evidence",
            )
            expected_identity = {
                "schema_version": 1,
                "endpoint": PEGASUS_ENDPOINT,
                "model_id": PEGASUS_MODEL_ID,
                "blind_id": record["blind_id"],
                "asset_id": record["asset_id"],
                "media_sha256": record["media_sha256"],
            }
            if any(cached.get(field) != value for field, value in expected_identity.items()):
                raise ProtocolError(
                    f"cached evidence identity mismatch for {record['blind_id']}"
                )
            if cached.get("request") != expected_request:
                raise ProtocolError(
                    f"cached evidence request mismatch for {record['blind_id']}"
                )
            raw_response = cached.get("raw_response")
            if cached.get("raw_response_sha256") != canonical_sha256(raw_response):
                raise ProtocolError(
                    f"cached evidence response hash mismatch for {record['blind_id']}"
                )
            canonical_response = scrub_pegasus_response(raw_response)
            if raw_response != canonical_response:
                raise ProtocolError(
                    f"cached evidence response is not canonical scrubbed evidence for {record['blind_id']}"
                )
            rating = parse_pegasus_response(canonical_response)
            evidence_record = cached
        else:
            raw_response = scrub_pegasus_response(transport(request_body))
            rating = parse_pegasus_response(raw_response)
            evidence_record = {
                "schema_version": 1,
                "endpoint": PEGASUS_ENDPOINT,
                "model_id": PEGASUS_MODEL_ID,
                "blind_id": record["blind_id"],
                "asset_id": record["asset_id"],
                "media_sha256": record["media_sha256"],
                "request": expected_request,
                "raw_response": raw_response,
                "raw_response_sha256": canonical_sha256(raw_response),
            }
            _atomic_write_json(cache_path, evidence_record)
        evidence_records.append(evidence_record)
        ratings.append(
            {
                "blind_id": record["blind_id"],
                "media_sha256": record["media_sha256"],
                **rating,
            }
        )

    blind_plan_sha256 = canonical_sha256(public_plan)
    evidence_report = {
        "schema_version": 1,
        "provider": "twelvelabs",
        "model_id": PEGASUS_MODEL_ID,
        "endpoint": PEGASUS_ENDPOINT,
        "protocol_sha256": public_plan["protocol_sha256"],
        "manifest_sha256": public_plan["manifest_sha256"],
        "blind_plan_sha256": blind_plan_sha256,
        "family_id": public_plan["family_id"],
        "rater_id": public_plan["rater_id"],
        "pass_id": public_plan["pass_id"],
        "records": evidence_records,
    }
    raw_envelope = {
        "schema_version": 1,
        "protocol_sha256": public_plan["protocol_sha256"],
        "manifest_sha256": public_plan["manifest_sha256"],
        "blind_plan_sha256": blind_plan_sha256,
        "family_id": public_plan["family_id"],
        "rater_id": public_plan["rater_id"],
        "pass_id": public_plan["pass_id"],
        "raw_evidence_sha256": canonical_sha256(evidence_report),
        "ratings": ratings,
    }
    return raw_envelope, evidence_report
