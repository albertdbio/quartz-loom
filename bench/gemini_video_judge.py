"""Strict Gemini 3.1 Pro full-video rating transport and parser.

The adapter uses inline MP4 input with explicit 16 fps sampling so short-lived
temporal defects are not hidden by Gemini's one-frame-per-second default. API
credentials and base64 media are never part of persisted evidence.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from bench.quality_sweep import ProtocolError, canonical_sha256, load_json
from bench.video_judge import (
    SCORE_DIMENSIONS,
    _atomic_write_json,
    _parse_json_object,
    _require_exact_fields,
    _require_integer_score,
    _require_sha256,
    _sha256_bytes,
)


GEMINI_MODEL_ID = "gemini-3.1-pro-preview"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL_ID}:generateContent"
)
GEMINI_VIDEO_FPS = 16.0
# Google documents a 20 MB inline-video limit. Conservatively bound the entire
# serialized request so base64 expansion plus prompt/schema overhead cannot
# cross that boundary.
GEMINI_MAX_REQUEST_BYTES = 20_000_000
GEMINI_MAX_TOKENS = 2048
GEMINI_TEMPERATURE = 0

GEMINI_RATING_SCHEMA: dict[str, Any] = {
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

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def build_gemini_request(video_bytes: bytes, rating_prompt: str) -> dict[str, Any]:
    if not isinstance(video_bytes, bytes) or not video_bytes:
        raise ProtocolError("Gemini video must contain bytes")
    encoded_video = base64.b64encode(video_bytes).decode("ascii")
    if not isinstance(rating_prompt, str) or not rating_prompt.strip():
        raise ProtocolError("Gemini rating prompt is required")
    request = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": encoded_video,
                        },
                        "videoMetadata": {"fps": GEMINI_VIDEO_FPS},
                    },
                    {"text": rating_prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": GEMINI_TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_TOKENS,
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": GEMINI_RATING_SCHEMA,
                }
            },
        },
    }
    if (
        len(json.dumps(request, separators=(",", ":")).encode("utf-8"))
        > GEMINI_MAX_REQUEST_BYTES
    ):
        raise ProtocolError("Gemini serialized request exceeds the 20 MB inline limit")
    return request


def _scrub_safety_ratings(value: Any, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be an array")
    safe: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ProtocolError(f"{label} entries must be objects")
        rating: dict[str, Any] = {}
        for field in ("category", "probability", "severity"):
            if field in item:
                field_value = item[field]
                if not isinstance(field_value, str) or re.fullmatch(
                    r"[A-Z][A-Z0-9_]{0,63}", field_value
                ) is None:
                    raise ProtocolError(f"{label} {field} must be a non-empty string")
                rating[field] = field_value
        for field in ("probabilityScore", "severityScore"):
            if field in item:
                field_value = item[field]
                if (
                    not isinstance(field_value, (int, float))
                    or isinstance(field_value, bool)
                    or not math.isfinite(field_value)
                    or not 0 <= field_value <= 1
                ):
                    raise ProtocolError(f"{label} {field} must be in [0, 1]")
                rating[field] = field_value
        if "blocked" in item:
            if not isinstance(item["blocked"], bool):
                raise ProtocolError(f"{label} blocked must be boolean")
            rating["blocked"] = item["blocked"]
        if not rating:
            raise ProtocolError(f"{label} entry contains no supported safety fields")
        safe.append(rating)
    return safe


def _scrub_prompt_feedback(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("Gemini promptFeedback must be an object")
    block_reason = value.get("blockReason")
    if block_reason is not None and (
        not isinstance(block_reason, str)
        or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", block_reason) is None
    ):
        raise ProtocolError("Gemini promptFeedback blockReason is invalid")
    return {
        "blockReason": block_reason,
        "safetyRatings": _scrub_safety_ratings(
            value.get("safetyRatings"), "Gemini prompt safetyRatings"
        ),
    }


def scrub_gemini_response(response: dict[str, Any]) -> dict[str, Any]:
    """Allowlist provider fields needed to reproduce a structured rating."""

    if not isinstance(response, dict):
        raise ProtocolError("Gemini response must be an object")
    response_id = response.get("responseId")
    if not isinstance(response_id, str) or not response_id:
        raise ProtocolError("Gemini responseId is required")
    model_version = response.get("modelVersion")
    if model_version != GEMINI_MODEL_ID:
        raise ProtocolError("Gemini response modelVersion mismatch")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ProtocolError("Gemini response must contain exactly one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ProtocolError("Gemini candidate must be an object")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise ProtocolError("Gemini candidate content is required")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ProtocolError("Gemini candidate parts are required")
    safe_parts: list[dict[str, str]] = []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            raise ProtocolError("Gemini candidate parts must contain text only")
        safe_parts.append({"text": part["text"]})
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        raise ProtocolError("Gemini usageMetadata is required")
    safe_usage = {
        field: usage[field]
        for field in (
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            "thoughtsTokenCount",
            "cachedContentTokenCount",
        )
        if field in usage
    }
    return {
        "candidates": [
            {
                "content": {
                    "parts": safe_parts,
                    "role": content.get("role"),
                },
                "finishReason": candidate.get("finishReason"),
                "index": candidate.get("index"),
                "safetyRatings": _scrub_safety_ratings(
                    candidate.get("safetyRatings"),
                    "Gemini candidate safetyRatings",
                ),
            }
        ],
        "usageMetadata": safe_usage,
        "modelVersion": model_version,
        "promptFeedback": _scrub_prompt_feedback(response.get("promptFeedback")),
        "responseId": response_id,
    }


def parse_gemini_response(response: dict[str, Any]) -> dict[str, Any]:
    response = scrub_gemini_response(response)
    prompt_feedback = response.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        raise ProtocolError(
            f"Gemini prompt blocked with {prompt_feedback['blockReason']}"
        )
    if isinstance(prompt_feedback, dict) and any(
        rating.get("blocked") is True
        for rating in prompt_feedback.get("safetyRatings", [])
        if isinstance(rating, dict)
    ):
        raise ProtocolError("Gemini response contains a blocked prompt safety rating")
    candidate = response["candidates"][0]
    if candidate.get("index") != 0:
        raise ProtocolError("Gemini candidate index must equal zero")
    if candidate.get("finishReason") != "STOP":
        raise ProtocolError("Gemini response finishReason must equal STOP")
    if any(
        rating.get("blocked") is True
        for rating in candidate.get("safetyRatings", [])
        if isinstance(rating, dict)
    ):
        raise ProtocolError("Gemini response contains a blocked safety rating")
    content = candidate["content"]
    if content.get("role") != "model":
        raise ProtocolError("Gemini candidate role must equal model")
    parts = content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise ProtocolError("Gemini response must contain exactly one text part")
    text = parts[0].get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("Gemini structured rating is required")

    usage = response["usageMetadata"]
    for field in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProtocolError(f"Gemini usageMetadata.{field} is required")
    for field, value in usage.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 2**63 - 1
        ):
            raise ProtocolError(
                f"Gemini usageMetadata.{field} must be a non-negative integer"
            )

    rating = _parse_json_object(text, "Gemini structured rating")
    rating = _require_exact_fields(
        rating,
        {
            "scores",
            "first_third_quality",
            "final_third_quality",
            "failure_tags",
            "rationale",
        },
        "Gemini structured rating",
    )
    scores = _require_exact_fields(
        rating["scores"], set(SCORE_DIMENSIONS), "Gemini scores"
    )
    normalized_scores = {
        name: _require_integer_score(scores[name], f"Gemini score {name}")
        for name in SCORE_DIMENSIONS
    }
    first = _require_integer_score(
        rating["first_third_quality"], "Gemini first_third_quality"
    )
    final = _require_integer_score(
        rating["final_third_quality"], "Gemini final_third_quality"
    )
    failure_tags = rating["failure_tags"]
    if not isinstance(failure_tags, list):
        raise ProtocolError("Gemini failure_tags must be an array")
    if any(not isinstance(tag, str) or not tag.strip() for tag in failure_tags):
        raise ProtocolError("Gemini failure_tags must contain non-empty strings")
    if len(set(failure_tags)) != len(failure_tags):
        raise ProtocolError("Gemini failure_tags must be unique")
    rationale = rating["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProtocolError("Gemini rationale is required")
    return {
        "scores": normalized_scores,
        "first_third_quality": first,
        "final_third_quality": final,
        "failure_tags": list(failure_tags),
        "rationale": rationale.strip(),
    }


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
    if family.get("kind") != "model" or family.get("model_id") != GEMINI_MODEL_ID:
        raise ProtocolError(
            f"blind plan family is not registered to {GEMINI_MODEL_ID}"
        )
    if family.get("evidence_provider") != "google":
        raise ProtocolError("blind plan family is not registered to Google evidence")
    if family.get("readiness") != "quality-qualified":
        raise ProtocolError("Gemini blind-plan execution requires a quality-qualified family")
    if (
        family.get("rater_ids") != [GEMINI_MODEL_ID]
        or plan.get("rater_id") != GEMINI_MODEL_ID
    ):
        raise ProtocolError(
            "Gemini blind-plan execution requires the concrete model rater identity"
        )
    passes = family.get("passes")
    if not isinstance(passes, int) or isinstance(passes, bool) or passes < 1:
        raise ProtocolError("registered Gemini family passes must be a positive integer")
    pass_id = plan.get("pass_id")
    if (
        not isinstance(pass_id, int)
        or isinstance(pass_id, bool)
        or not 1 <= pass_id <= passes
    ):
        raise ProtocolError("blind plan pass is not registered to the Gemini family")


def _request_evidence(
    record: dict[str, Any],
    rating_prompt: str,
    rubric_text: str,
    request_body: dict[str, Any],
    video_bytes: bytes,
) -> dict[str, Any]:
    wire_request = json.loads(json.dumps(request_body))
    inline = wire_request["contents"][0]["parts"][0]["inlineData"]
    encoded_video = inline.pop("data")
    inline.update(
        {
            "media_sha256": record["media_sha256"],
            "decoded_bytes": len(video_bytes),
            "encoded_bytes": len(encoded_video.encode("ascii")),
        }
    )
    return {
        "endpoint": GEMINI_ENDPOINT,
        "model_id": GEMINI_MODEL_ID,
        "adapter_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "blind_id": record["blind_id"],
        "asset_id": record["asset_id"],
        "media_sha256": record["media_sha256"],
        "prompt_sha256": _sha256_bytes(rating_prompt.encode("utf-8")),
        "rubric_sha256": _sha256_bytes(rubric_text.encode("utf-8")),
        "response_schema_sha256": canonical_sha256(GEMINI_RATING_SCHEMA),
        "temperature": GEMINI_TEMPERATURE,
        "max_tokens": GEMINI_MAX_TOKENS,
        "video_fps": GEMINI_VIDEO_FPS,
        "stream": False,
        "wire_request": wire_request,
    }


def run_gemini_plan(
    protocol: dict[str, Any],
    public_plan: dict[str, Any],
    asset_dir: str | Path,
    evidence_dir: str | Path,
    *,
    transport: Transport,
    rubric_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge a complete blind plan with safe, hash-bound resumable evidence."""

    _validate_plan_binding(protocol, public_plan)
    if not isinstance(rubric_text, str) or not rubric_text:
        raise ProtocolError("Gemini plan rubric must be non-empty UTF-8 text")
    evaluation = protocol.get("evaluation")
    registered_rubric_sha256 = (
        evaluation.get("rubric_sha256") if isinstance(evaluation, dict) else None
    )
    if registered_rubric_sha256 != _sha256_bytes(rubric_text.encode("utf-8")):
        raise ProtocolError("Gemini plan does not use the registered rubric SHA-256")
    records = public_plan.get("records")
    if not isinstance(records, list) or not records:
        raise ProtocolError("blind plan records must be a non-empty array")
    blind_ids: set[str] = set()
    prepared: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    asset_root = Path(asset_dir)
    for value in records:
        record = _require_exact_fields(
            value,
            {
                "blind_id",
                "asset_id",
                "case_id",
                "prompt_id",
                "prompt",
                "slot",
                "media_sha256",
            },
            "blind plan record",
        )
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
        if not isinstance(record["prompt"], str) or not record["prompt"]:
            raise ProtocolError("blind plan prompt is required")
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
        if _sha256_bytes(video_bytes) != record["media_sha256"]:
            raise ProtocolError(f"blind asset {asset_id} media SHA-256 mismatch")
        # Import here to keep the rating prompt contract shared with Pegasus
        # without creating a dependency from the common module back to Google.
        from bench.video_judge import build_rating_prompt

        rating_prompt = build_rating_prompt(record["prompt"], rubric_text)
        request_body = build_gemini_request(video_bytes, rating_prompt)
        expected_request = _request_evidence(
            record, rating_prompt, rubric_text, request_body, video_bytes
        )
        prepared.append((record, request_body, expected_request))

    evidence_root = Path(evidence_dir)
    evidence_records: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    for record, request_body, expected_request in sorted(
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
                "cached Gemini evidence",
            )
            expected_identity = {
                "schema_version": 1,
                "endpoint": GEMINI_ENDPOINT,
                "model_id": GEMINI_MODEL_ID,
                "blind_id": record["blind_id"],
                "asset_id": record["asset_id"],
                "media_sha256": record["media_sha256"],
            }
            if any(cached.get(field) != value for field, value in expected_identity.items()):
                raise ProtocolError(
                    f"cached Gemini evidence identity mismatch for {record['blind_id']}"
                )
            if cached.get("request") != expected_request:
                raise ProtocolError(
                    f"cached Gemini evidence request mismatch for {record['blind_id']}"
                )
            raw_response = cached.get("raw_response")
            if cached.get("raw_response_sha256") != canonical_sha256(raw_response):
                raise ProtocolError(
                    f"cached Gemini response hash mismatch for {record['blind_id']}"
                )
            canonical_response = scrub_gemini_response(raw_response)
            if raw_response != canonical_response:
                raise ProtocolError(
                    f"cached Gemini response is not canonical scrubbed evidence for {record['blind_id']}"
                )
            rating = parse_gemini_response(canonical_response)
            evidence_record = cached
        else:
            raw_response = scrub_gemini_response(transport(request_body))
            rating = parse_gemini_response(raw_response)
            evidence_record = {
                "schema_version": 1,
                "endpoint": GEMINI_ENDPOINT,
                "model_id": GEMINI_MODEL_ID,
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
        "provider": "google",
        "model_id": GEMINI_MODEL_ID,
        "endpoint": GEMINI_ENDPOINT,
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


def ratings_from_gemini_evidence(
    protocol: dict[str, Any],
    public_plan: dict[str, Any],
    evidence_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-parse scrubbed Google responses and reproduce their rating rows."""

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
        "Gemini evidence report",
    )
    expected_identity = {
        "schema_version": 1,
        "provider": "google",
        "model_id": GEMINI_MODEL_ID,
        "endpoint": GEMINI_ENDPOINT,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": public_plan.get("manifest_sha256"),
        "blind_plan_sha256": canonical_sha256(public_plan),
        "family_id": public_plan.get("family_id"),
        "rater_id": public_plan.get("rater_id"),
        "pass_id": public_plan.get("pass_id"),
    }
    if any(report.get(field) != value for field, value in expected_identity.items()):
        raise ProtocolError("Gemini evidence report identity mismatch")
    public_records = public_plan.get("records")
    records = report["records"]
    if not isinstance(public_records, list) or not isinstance(records, list) or not records:
        raise ProtocolError("Gemini evidence records must be a non-empty array")
    public_by_id = {
        record.get("blind_id"): record
        for record in public_records
        if isinstance(record, dict)
    }
    if len(public_by_id) != len(public_records) or None in public_by_id:
        raise ProtocolError("Gemini public-plan blind IDs must be unique")

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
            "Gemini evidence record",
        )
        blind_id = record["blind_id"]
        public = public_by_id.get(blind_id)
        if public is None or blind_id in seen:
            raise ProtocolError("Gemini evidence blind IDs must exactly cover the plan")
        seen.add(blind_id)
        expected_record_identity = {
            "schema_version": 1,
            "endpoint": GEMINI_ENDPOINT,
            "model_id": GEMINI_MODEL_ID,
            "blind_id": blind_id,
            "asset_id": public.get("asset_id"),
            "media_sha256": public.get("media_sha256"),
        }
        if any(
            record.get(field) != expected
            for field, expected in expected_record_identity.items()
        ):
            raise ProtocolError("Gemini evidence record identity mismatch")
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
                "video_fps",
                "stream",
                "wire_request",
            },
            "Gemini evidence request",
        )
        request_identity = {
            "endpoint": GEMINI_ENDPOINT,
            "model_id": GEMINI_MODEL_ID,
            "blind_id": blind_id,
            "asset_id": public.get("asset_id"),
            "media_sha256": public.get("media_sha256"),
            "rubric_sha256": protocol.get("evaluation", {}).get("rubric_sha256"),
            "response_schema_sha256": canonical_sha256(GEMINI_RATING_SCHEMA),
            "temperature": GEMINI_TEMPERATURE,
            "max_tokens": GEMINI_MAX_TOKENS,
            "video_fps": GEMINI_VIDEO_FPS,
            "stream": False,
        }
        if any(request.get(field) != expected for field, expected in request_identity.items()):
            raise ProtocolError("Gemini evidence request identity mismatch")
        _require_sha256(request["adapter_sha256"], "Gemini adapter_sha256")
        if request["adapter_sha256"] != _sha256_bytes(Path(__file__).read_bytes()):
            raise ProtocolError("Gemini adapter_sha256 mismatch")
        wire = _require_exact_fields(
            request["wire_request"],
            {"contents", "generationConfig"},
            "Gemini scrubbed wire request",
        )
        contents = wire["contents"]
        if not isinstance(contents, list) or len(contents) != 1:
            raise ProtocolError("Gemini scrubbed contents mismatch")
        content = _require_exact_fields(
            contents[0], {"role", "parts"}, "Gemini scrubbed content"
        )
        if content["role"] != "user" or not isinstance(content["parts"], list) or len(
            content["parts"]
        ) != 2:
            raise ProtocolError("Gemini scrubbed content mismatch")
        video_part = _require_exact_fields(
            content["parts"][0],
            {"inlineData", "videoMetadata"},
            "Gemini scrubbed video part",
        )
        inline = _require_exact_fields(
            video_part["inlineData"],
            {
                "mimeType",
                "media_sha256",
                "decoded_bytes",
                "encoded_bytes",
            },
            "Gemini scrubbed inline video",
        )
        metadata = _require_exact_fields(
            video_part["videoMetadata"], {"fps"}, "Gemini video metadata"
        )
        if (
            inline["mimeType"] != "video/mp4"
            or inline["media_sha256"] != public.get("media_sha256")
            or not isinstance(inline["decoded_bytes"], int)
            or isinstance(inline["decoded_bytes"], bool)
            or inline["decoded_bytes"] < 1
            or not isinstance(inline["encoded_bytes"], int)
            or isinstance(inline["encoded_bytes"], bool)
            or not 1 <= inline["encoded_bytes"] <= GEMINI_MAX_REQUEST_BYTES
            or metadata["fps"] != GEMINI_VIDEO_FPS
        ):
            raise ProtocolError("Gemini scrubbed inline video mismatch")
        prompt_part = _require_exact_fields(
            content["parts"][1], {"text"}, "Gemini scrubbed prompt part"
        )
        prompt = prompt_part["text"]
        if not isinstance(prompt, str) or not prompt:
            raise ProtocolError("Gemini evidence prompt is required")
        if request["prompt_sha256"] != _sha256_bytes(prompt.encode("utf-8")):
            raise ProtocolError("Gemini evidence prompt_sha256 mismatch")
        expected_generation = {
            "temperature": GEMINI_TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_TOKENS,
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": GEMINI_RATING_SCHEMA,
                }
            },
        }
        if wire["generationConfig"] != expected_generation:
            raise ProtocolError("Gemini scrubbed generationConfig mismatch")
        raw_response = _require_exact_fields(
            record["raw_response"],
            {
                "candidates",
                "usageMetadata",
                "modelVersion",
                "promptFeedback",
                "responseId",
            },
            "Gemini raw response",
        )
        canonical_response = scrub_gemini_response(raw_response)
        if raw_response != canonical_response:
            raise ProtocolError("Gemini raw response is not canonical scrubbed evidence")
        if record["raw_response_sha256"] != canonical_sha256(raw_response):
            raise ProtocolError("Gemini raw evidence response hash mismatch")
        ratings.append(
            {
                "blind_id": blind_id,
                "media_sha256": record["media_sha256"],
                **parse_gemini_response(canonical_response),
            }
        )
    if seen != set(public_by_id):
        raise ProtocolError("Gemini evidence blind IDs must exactly cover the plan")
    return ratings


def load_gemini_api_key(
    dotenv_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    configured = environment.get("GEMINI_API_KEY", "").strip()
    if configured:
        return configured
    path = Path(dotenv_path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProtocolError("GEMINI_API_KEY is not configured") from exc
    matches: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        if name.strip() != "GEMINI_API_KEY":
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        matches.append(value.strip())
    if len(matches) != 1 or not matches[0]:
        raise ProtocolError("GEMINI_API_KEY is not configured exactly once")
    return matches[0]


def post_gemini(
    request_body: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    if not isinstance(api_key, str) or not api_key:
        raise ProtocolError("GEMINI_API_KEY is required")
    encoded = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        GEMINI_ENDPOINT,
        data=encoded,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw_error = exc.read(64_000)
        safe_status: str | None = None
        try:
            error_payload = _parse_json_object(
                raw_error.decode("utf-8"), "Gemini HTTP error"
            )
            error = error_payload.get("error")
            if isinstance(error, dict):
                status = error.get("status")
                allowed_statuses = {
                    "ABORTED",
                    "ALREADY_EXISTS",
                    "CANCELLED",
                    "DATA_LOSS",
                    "DEADLINE_EXCEEDED",
                    "FAILED_PRECONDITION",
                    "INTERNAL",
                    "INVALID_ARGUMENT",
                    "NOT_FOUND",
                    "OUT_OF_RANGE",
                    "PERMISSION_DENIED",
                    "RESOURCE_EXHAUSTED",
                    "UNAUTHENTICATED",
                    "UNAVAILABLE",
                    "UNIMPLEMENTED",
                    "UNKNOWN",
                }
                if status in allowed_statuses:
                    safe_status = status
        except (ProtocolError, UnicodeDecodeError):
            pass
        suffix = f" ({safe_status})" if safe_status else ""
        raise ProtocolError(
            f"Gemini HTTP {exc.code}{suffix}; response body withheld"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProtocolError(f"Gemini request failed: {type(exc).__name__}") from exc
    try:
        return _parse_json_object(raw.decode("utf-8"), "Gemini HTTP response")
    except UnicodeDecodeError as exc:
        raise ProtocolError("Gemini HTTP response is not UTF-8") from exc
