#!/usr/bin/env python3
"""Local quality-sweep integrity, blinding, aggregation, and gate helpers.

The module is deliberately stdlib-only. Paid GPU and judge calls happen only
after these local checks have frozen the protocol and bound every artifact to
its exact bytes and initial noise.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable


class ProtocolError(ValueError):
    """Raised when a protocol, artifact, rating, or gate input is invalid."""


_PLACEHOLDERS = {"", "tbd", "todo", "unknown", "unresolved", "placeholder"}
_BASELINE_HASH_FIELDS = (
    "source_diff_sha256",
    "checkpoint_sha256",
    "decoder_sha256",
    "runner_sha256",
    "config_sha256",
    "measured_artifact_sha256",
)
_REFERENCE_HASH_FIELDS = (
    "source_diff_sha256",
    "checkpoint_sha256",
    "decoder_sha256",
    "runner_sha256",
    "config_sha256",
)
SUPPORTED_GATE_KEYS = frozenset(
    {
        "absolute_quality",
        "family_floor",
        "stratum_floor",
        "sf4_noninferiority_margin",
        "maximum_low_quality_items",
        "low_quality_item_threshold",
        "final_third_temporal_floor",
        "maximum_early_to_late_drop",
        "warm_e2e_fps",
        "cold_e2e_fps",
        "sustained_seconds",
        "sustained_e2e_fps",
        "minimum_each_warm_trial_fps",
        "maximum_first_visible_rgb_s",
        "maximum_p95_effective_frame_interval_ms",
        "required_model_families",
        "require_human",
        "required_forwards",
        "required_rgb_frames",
    }
)
_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "protocol_sha256",
        "source_commit",
        "source_diff_sha256",
        "checkpoint_revision",
        "checkpoint_sha256",
        "decoder_revision",
        "decoder_sha256",
        "runner_sha256",
        "config_sha256",
        "rubric_sha256",
        "prompt_utf8_sha256",
        "effective_prompt_utf8_sha256",
        "seed",
        "rng_algorithm",
        "rng_device",
        "seed_application_point",
        "initial_noise_sha256",
        "input_noise_sha256",
        "latent_sha256",
        "media_sha256",
        "forwards",
        "decoded_frames",
        "torch_cuda_driver_hardware",
        "determinism_flags",
        "encoding_command",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def causal_forward_count(
    latent_frames: int,
    block_size: int,
    first_chunk_denoising_steps: int,
    steady_denoising_steps: int,
) -> int:
    """Count denoising plus clean-context forwards for a causal rollout."""

    values = (
        latent_frames,
        block_size,
        first_chunk_denoising_steps,
        steady_denoising_steps,
    )
    if any(not _is_int(value) or value <= 0 for value in values):
        raise ProtocolError("forward-count inputs must be positive integers")
    if latent_frames % block_size:
        raise ProtocolError("latent_frames must be divisible by block_size")
    chunks = latent_frames // block_size
    return (first_chunk_denoising_steps + 1) + (chunks - 1) * (
        steady_denoising_steps + 1
    )


def load_json(path: str | Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ProtocolError(f"duplicate JSON key {key}")
            value[key] = item
        return value

    try:
        return json.loads(
            Path(path).read_text(),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load JSON {path}: {exc}") from exc


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be an array")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ProtocolError(f"{label} must be a 40-character lowercase git commit")
    return value


def _verify_canonical_hash(value: dict, field: str, label: str) -> str:
    recorded = _require_sha256(value.get(field), f"{label} {field}")
    payload = dict(value)
    payload.pop(field, None)
    if recorded != canonical_sha256(payload):
        raise ProtocolError(f"{label} {field} mismatch")
    return recorded


def _placeholder_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_placeholder_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_placeholder_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
        found.append(path)
    return found


def validate_protocol(protocol: dict, require_frozen: bool = False) -> None:
    protocol = _require_mapping(protocol, "protocol")
    required = {
        "schema_version",
        "protocol_id",
        "status",
        "hypothesis",
        "baseline",
        "reference_systems",
        "repair_candidates",
        "development",
        "confirmatory",
        "prompts",
        "media_contract",
        "long_horizon_media_contract",
        "evaluation",
        "gates",
        "provenance_requirements",
        "stop_rules",
        "sources",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ProtocolError(f"protocol missing required fields: {', '.join(missing)}")
    if protocol["schema_version"] != 1:
        raise ProtocolError("schema_version must be 1")
    if protocol["status"] not in {"draft", "frozen", "superseded"}:
        raise ProtocolError("status must be draft, frozen, or superseded")
    if require_frozen and protocol["status"] != "frozen":
        raise ProtocolError("protocol must be frozen")
    if protocol["status"] == "frozen" or require_frozen:
        if not protocol.get("frozen_at"):
            raise ProtocolError("frozen protocol requires frozen_at")
        placeholders = _placeholder_paths(protocol)
        if placeholders:
            raise ProtocolError(f"frozen protocol contains placeholder at {placeholders[0]}")

    baseline = _require_mapping(protocol["baseline"], "baseline")
    baseline_required = {
        "system_id",
        "repository",
        "commit",
        "source_diff_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "weights",
        "decoder",
        "decoder_mode",
        "decoder_sha256",
        "runner",
        "runner_sha256",
        "runner_status",
        "config_sha256",
        "measured_artifact",
        "measured_artifact_sha256",
        "forwards",
    }
    baseline_missing = sorted(baseline_required - baseline.keys())
    if baseline_missing:
        raise ProtocolError(f"baseline missing {', '.join(baseline_missing)}")
    if baseline.get("forwards") != causal_forward_count(21, 1, 4, 1):
        raise ProtocolError("frozen CF1 baseline must preserve exactly 45 forwards")
    if protocol["status"] == "frozen" or require_frozen:
        _require_commit(baseline.get("commit"), "baseline.commit")
        for field in _BASELINE_HASH_FIELDS:
            _require_sha256(baseline.get(field), f"baseline.{field}")

    references = _require_mapping(protocol["reference_systems"], "reference_systems")
    sf4 = _require_mapping(references.get("sf4-reference"), "reference_systems.sf4-reference")
    reference_required = {
        "repository",
        "commit",
        "source_diff_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "decoder",
        "decoder_mode",
        "decoder_sha256",
        "runner_sha256",
        "runner_status",
        "config_sha256",
        "forwards",
    }
    reference_missing = sorted(reference_required - sf4.keys())
    if reference_missing:
        raise ProtocolError(
            "reference_systems.sf4-reference missing " + ", ".join(reference_missing)
        )
    if protocol["status"] == "frozen" or require_frozen:
        _require_commit(sf4.get("commit"), "reference_systems.sf4-reference.commit")
        for field in _REFERENCE_HASH_FIELDS:
            _require_sha256(sf4.get(field), f"reference_systems.sf4-reference.{field}")
        if sf4.get("forwards") != causal_forward_count(21, 3, 4, 4):
            raise ProtocolError("reference_systems.sf4-reference must preserve exactly 35 forwards")
    for label, system in (("baseline", baseline), ("sf4-reference", sf4)):
        if system.get("runner_status") not in {"historical-only", "confirmatory-ready"}:
            raise ProtocolError(f"{label}.runner_status is invalid")
        if (protocol["status"] == "frozen" or require_frozen) and system["runner_status"] != "confirmatory-ready":
            raise ProtocolError(f"{label} runner must be confirmatory-ready before freeze")
    if baseline.get("decoder_mode") != "rolling-three-latent":
        raise ProtocolError("baseline.decoder_mode must equal rolling-three-latent")
    if sf4.get("decoder_mode") != "stock-wan":
        raise ProtocolError("reference_systems.sf4-reference.decoder_mode must equal stock-wan")

    candidates = _require_list(protocol["repair_candidates"], "repair_candidates")
    if not candidates:
        raise ProtocolError("repair_candidates must not be empty")
    candidate_ids = [item.get("system_id") for item in candidates if isinstance(item, dict)]
    if len(candidate_ids) != len(candidates) or any(not item for item in candidate_ids):
        raise ProtocolError("every repair candidate requires system_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ProtocolError("duplicate repair candidate system_id")
    for item in candidates:
        system_id = item["system_id"]
        if "config_sha256" not in item:
            raise ProtocolError(f"repair candidate {system_id} missing config_sha256")
        if protocol["status"] == "frozen" or require_frozen:
            _require_sha256(
                item["config_sha256"],
                f"repair candidate {system_id} config_sha256",
            )

    prompts = _require_list(protocol["prompts"], "prompts")
    prompt_ids = [item.get("id") for item in prompts if isinstance(item, dict)]
    if len(prompt_ids) != len(prompts) or any(not item for item in prompt_ids):
        raise ProtocolError("every prompt requires id")
    duplicates = sorted(key for key, count in Counter(prompt_ids).items() if count > 1)
    if duplicates:
        raise ProtocolError(f"duplicate prompt id: {duplicates[0]}")
    prompt_by_id = {item["id"]: item for item in prompts}
    for item in prompts:
        for field in ("split", "stratum", "text", "seeds"):
            if field not in item:
                raise ProtocolError(f"prompt {item['id']} missing {field}")
        if item["split"] not in {"development", "confirmatory", "sentinel"}:
            raise ProtocolError(f"prompt {item['id']} has invalid split")
        seeds = _require_list(item["seeds"], f"prompt {item['id']} seeds")
        if not seeds or len(set(seeds)) != len(seeds) or any(not _is_int(seed) for seed in seeds):
            raise ProtocolError(f"prompt {item['id']} seeds must be unique integers")

    development = _require_mapping(protocol["development"], "development")
    confirmatory = _require_mapping(protocol["confirmatory"], "confirmatory")
    development_ids = _require_list(development.get("prompt_ids"), "development.prompt_ids")
    confirmatory_ids = _require_list(confirmatory.get("prompt_ids"), "confirmatory.prompt_ids")
    if len(set(development_ids)) != len(development_ids):
        raise ProtocolError("development prompt_ids contain duplicates")
    if len(development_ids) != 6:
        raise ProtocolError("development set must contain exactly six prompts")
    if len(confirmatory_ids) != 12:
        raise ProtocolError("confirmatory set must contain exactly 12 prompts")
    if len(set(confirmatory_ids)) != len(confirmatory_ids):
        raise ProtocolError("confirmatory prompt_ids contain duplicates")
    if set(development_ids) & set(confirmatory_ids):
        raise ProtocolError("development and confirmatory prompt sets must be disjoint")
    unknown = sorted((set(development_ids) | set(confirmatory_ids)) - prompt_by_id.keys())
    if unknown:
        raise ProtocolError(f"unknown prompt id: {unknown[0]}")
    for prompt_id in development_ids:
        if prompt_by_id[prompt_id]["split"] != "development":
            raise ProtocolError(f"development prompt {prompt_id} has wrong split")
    for prompt_id in confirmatory_ids:
        if prompt_by_id[prompt_id]["split"] != "confirmatory":
            raise ProtocolError(f"confirmatory prompt {prompt_id} has wrong split")
        if len(prompt_by_id[prompt_id]["seeds"]) != 2:
            raise ProtocolError(
                f"confirmatory prompt {prompt_id} must have exactly two seeds"
            )
    strata = Counter(prompt_by_id[prompt_id]["stratum"] for prompt_id in confirmatory_ids)
    if len(strata) != 6 or set(strata.values()) != {2}:
        raise ProtocolError("confirmatory prompts require exactly six strata with two prompts each")

    development_systems = _require_list(development.get("systems"), "development.systems")
    known_systems = {baseline["system_id"], *candidate_ids, *references.keys()}
    expected_development_systems = [baseline["system_id"], *candidate_ids]
    if development_systems != expected_development_systems:
        raise ProtocolError(
            "development.systems must contain baseline plus every registered candidate exactly once"
        )
    for field in ("round_a_seed_indexes", "round_b_seed_indexes"):
        indexes = _require_list(development.get(field), f"development.{field}")
        if not indexes or any(not _is_int(index) or index < 0 for index in indexes):
            raise ProtocolError(f"development.{field} must contain non-negative integer indexes")
        if len(set(indexes)) != len(indexes):
            raise ProtocolError(f"development.{field} contains duplicate indexes")
    if not set(development["round_a_seed_indexes"]).issubset(development["round_b_seed_indexes"]):
        raise ProtocolError("Round B seed indexes must include the Round A indexes")
    if development["round_a_seed_indexes"] != [0] or development["round_b_seed_indexes"] != [0, 1]:
        raise ProtocolError("development rounds must use exact seed indexes [0] and [0, 1]")
    for prompt_id in development_ids:
        seed_count = len(prompt_by_id[prompt_id]["seeds"])
        if max(development["round_b_seed_indexes"]) >= seed_count:
            raise ProtocolError(f"development prompt {prompt_id} lacks a registered Round B seed")

    confirmatory_systems = _require_list(confirmatory.get("systems"), "confirmatory.systems")
    expected_system_tokens = {
        baseline["system_id"],
        "$selection_lock.finalist_system_id",
        "sf4-reference",
    }
    if set(confirmatory_systems) != expected_system_tokens or len(confirmatory_systems) != 3:
        raise ProtocolError(
            "confirmatory.systems must bind baseline, selection-lock finalist, and sf4-reference"
        )
    if not confirmatory.get("selection_lock_schema"):
        raise ProtocolError("confirmatory.selection_lock_schema is required")
    sentinel_ids = _require_list(
        confirmatory.get("long_horizon_prompt_ids"),
        "confirmatory.long_horizon_prompt_ids",
    )
    if len(sentinel_ids) != 2 or len(set(sentinel_ids)) != len(sentinel_ids):
        raise ProtocolError("confirmatory requires exactly two unique long-horizon sentinels")
    for prompt_id in sentinel_ids:
        prompt = prompt_by_id.get(prompt_id)
        if not prompt or prompt.get("split") != "sentinel":
            raise ProtocolError(f"long-horizon prompt {prompt_id} must exist with sentinel split")
        if len(prompt["seeds"]) != 1:
            raise ProtocolError(f"long-horizon prompt {prompt_id} must have exactly one seed")
    if confirmatory.get("long_horizon_latent_frames") != 241:
        raise ProtocolError("confirmatory.long_horizon_latent_frames must equal 241")
    if confirmatory.get("long_horizon_rgb_frames") != 961:
        raise ProtocolError("confirmatory.long_horizon_rgb_frames must equal 961")

    contract = _require_mapping(protocol["media_contract"], "media_contract")
    long_contract = _require_mapping(
        protocol["long_horizon_media_contract"], "long_horizon_media_contract"
    )
    for label, actual_contract, expected_frames in (
        ("media_contract", contract, 81),
        ("long_horizon_media_contract", long_contract, 961),
    ):
        expected_contract = {
            "width": 832,
            "height": 480,
            "fps": 16,
            "decoded_frames": expected_frames,
            "codec": "h264",
            "pixel_format": "yuv420p",
        }
        for field, expected in expected_contract.items():
            if actual_contract.get(field) != expected:
                raise ProtocolError(f"{label}.{field} must equal {expected}")
        tolerance = actual_contract.get("duration_tolerance_frames")
        if not _is_int(tolerance) or tolerance < 0:
            raise ProtocolError(f"{label}.duration_tolerance_frames must be a non-negative integer")

    evaluation = _require_mapping(protocol["evaluation"], "evaluation")
    dimensions = _require_mapping(evaluation.get("dimensions"), "evaluation.dimensions")
    if not dimensions or any(not isinstance(weight, (int, float)) or weight <= 0 for weight in dimensions.values()):
        raise ProtocolError("evaluation dimensions need positive weights")
    if not math.isclose(sum(dimensions.values()), 1.0, abs_tol=1e-9):
        raise ProtocolError("evaluation dimension weights must sum to 1")
    families = _require_list(evaluation.get("families"), "evaluation.families")
    family_ids = [item.get("id") for item in families if isinstance(item, dict)]
    if len(family_ids) != len(families) or len(set(family_ids)) != len(family_ids):
        raise ProtocolError("evaluation family ids must be present and unique")
    model_families = [item for item in families if item.get("kind") == "model"]
    human_families = [item for item in families if item.get("kind") == "human"]
    if len(families) != 3 or len(model_families) != 2 or len(human_families) != 1:
        raise ProtocolError(
            "evaluation requires exactly two model families and one human family"
        )
    allowed_model_readiness = {
        "historical-only",
        "transport-verified",
        "calibration-failed",
        "quality-qualified",
    }
    for item in model_families:
        readiness = item.get("readiness")
        if readiness not in allowed_model_readiness:
            raise ProtocolError(
                f"evaluation model family {item.get('id')} has invalid readiness"
            )
        if (protocol["status"] == "frozen" or require_frozen) and readiness != "quality-qualified":
            raise ProtocolError(
                f"evaluation model family {item.get('id')} must be quality-qualified before freeze"
            )
    model_ids = [item.get("model_id") for item in model_families]
    if len(set(model_ids)) != 2:
        raise ProtocolError("evaluation model families require distinct model_id values")
    _require_sha256(evaluation.get("rubric_sha256"), "evaluation.rubric_sha256")
    if not evaluation.get("rubric"):
        raise ProtocolError("evaluation.rubric is required")
    for item in families:
        passes = item.get("passes")
        if not _is_int(passes) or passes < 1:
            raise ProtocolError(f"evaluation family {item.get('id')} passes must be positive")
        if item.get("kind") not in {"model", "human"}:
            raise ProtocolError(f"evaluation family {item.get('id')} has invalid kind")
        if not isinstance(item.get("evidence_provider"), str) or not item[
            "evidence_provider"
        ]:
            raise ProtocolError(
                f"evaluation family {item.get('id')} requires evidence_provider"
            )
        rater_ids = _require_list(
            item.get("rater_ids"),
            f"evaluation family {item.get('id')} rater_ids",
        )
        if (
            not rater_ids
            or any(not isinstance(rater_id, str) or not rater_id for rater_id in rater_ids)
            or len(set(rater_ids)) != len(rater_ids)
        ):
            raise ProtocolError(
                f"evaluation family {item.get('id')} rater_ids must be present and unique"
            )
        if item.get("kind") == "model" and len(rater_ids) != 1:
            raise ProtocolError(
                f"evaluation model family {item.get('id')} requires exactly one rater_id"
            )
        if item.get("kind") == "model" and passes != 2:
            raise ProtocolError(
                f"evaluation model family {item.get('id')} requires exactly two passes"
            )
        if item.get("kind") == "human":
            minimum_raters = item.get("minimum_raters")
            if not _is_int(minimum_raters) or minimum_raters < 3:
                raise ProtocolError("human evaluation requires at least three raters")
            if len(rater_ids) < minimum_raters:
                raise ProtocolError(
                    f"evaluation human family {item.get('id')} rater_ids is below minimum_raters"
                )
            if passes != 1 or minimum_raters != 3 or len(rater_ids) != 3:
                raise ProtocolError(
                    "evaluation human family requires exactly three raters and one pass"
                )
    if protocol["status"] == "frozen" or require_frozen:
        for item in families:
            if not item.get("model_id"):
                raise ProtocolError(f"evaluation family {item.get('id')} requires exact model_id")

    gates = _require_mapping(protocol["gates"], "gates")
    unsupported_gates = sorted(set(gates) - SUPPORTED_GATE_KEYS)
    missing_gates = sorted(SUPPORTED_GATE_KEYS - set(gates))
    if unsupported_gates:
        raise ProtocolError(f"unsupported gate {unsupported_gates[0]}")
    if missing_gates:
        raise ProtocolError(f"protocol gates missing {missing_gates[0]}")
    if gates["required_forwards"] != baseline["forwards"]:
        raise ProtocolError("gates.required_forwards must match baseline.forwards")
    if gates["required_rgb_frames"] != contract["decoded_frames"]:
        raise ProtocolError("gates.required_rgb_frames must match media_contract.decoded_frames")
    for key, value in gates.items():
        if key in {"require_human"}:
            if not isinstance(value, bool):
                raise ProtocolError(f"gates.{key} must be boolean")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ProtocolError(f"gates.{key} must be a finite number")

    provenance = _require_list(protocol["provenance_requirements"], "provenance_requirements")
    missing_provenance = sorted(_REQUIRED_PROVENANCE_FIELDS - set(provenance))
    if missing_provenance:
        raise ProtocolError(f"provenance_requirements missing {missing_provenance[0]}")


def validate_selection_lock(
    protocol: dict,
    selection_lock: dict,
    *,
    selection_report: dict | None = None,
) -> None:
    validate_protocol(protocol, require_frozen=True)
    selection_lock = _require_mapping(selection_lock, "selection_lock")
    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "finalist_system_id",
        "finalist_config_sha256",
        "selection_report_sha256",
        "locked_at",
    }
    missing = sorted(required - selection_lock.keys())
    if missing:
        raise ProtocolError(f"selection_lock missing {missing[0]}")
    if selection_lock["schema_version"] != 1 or selection_lock["status"] != "locked":
        raise ProtocolError("selection_lock must be schema v1 with status locked")
    if selection_lock["protocol_sha256"] != canonical_sha256(protocol):
        raise ProtocolError("selection_lock protocol_sha256 does not match protocol")
    candidates = {item["system_id"] for item in protocol["repair_candidates"]}
    if selection_lock["finalist_system_id"] not in candidates:
        raise ProtocolError("selection_lock finalist must be a registered repair candidate")
    _require_sha256(selection_lock["finalist_config_sha256"], "selection_lock.finalist_config_sha256")
    candidate = next(
        item
        for item in protocol["repair_candidates"]
        if item["system_id"] == selection_lock["finalist_system_id"]
    )
    if selection_lock["finalist_config_sha256"] != candidate["config_sha256"]:
        raise ProtocolError("selection_lock does not match registered candidate config")
    _require_sha256(
        selection_lock["selection_report_sha256"],
        "selection_lock.selection_report_sha256",
    )
    if not isinstance(selection_lock["locked_at"], str) or not selection_lock["locked_at"]:
        raise ProtocolError("selection_lock.locked_at is required")
    if selection_report is not None:
        selection_report = _require_mapping(
            selection_report, "selection_report"
        )
        _verify_canonical_hash(
            selection_report,
            "report_sha256",
            "selection report",
        )
        if (
            selection_lock["selection_report_sha256"]
            != selection_report["report_sha256"]
        ):
            raise ProtocolError(
                "selection_lock selection_report_sha256 does not match selection report"
            )
        for field in ("finalist_system_id", "finalist_config_sha256"):
            if selection_lock[field] != selection_report.get(field):
                raise ProtocolError(
                    f"selection_lock {field} does not match selection report"
                )


def _expected_scope(
    protocol: dict,
    phase: str,
    selection_lock: dict | None,
    round_b_systems: list[str] | None = None,
) -> tuple[list[tuple[str, int]], list[str], dict[str, dict]]:
    prompt_by_id = {item["id"]: item for item in protocol["prompts"]}
    if phase == "confirmatory":
        if selection_lock is None:
            raise ProtocolError("confirmatory manifest requires a selection lock")
        validate_selection_lock(protocol, selection_lock)
        prompt_ids = protocol["confirmatory"]["prompt_ids"]
        cases = [
            (prompt_id, seed)
            for prompt_id in prompt_ids
            for seed in prompt_by_id[prompt_id]["seeds"]
        ]
        systems = [
            protocol["baseline"]["system_id"],
            selection_lock["finalist_system_id"],
            "sf4-reference",
        ]
    elif phase in {"development-round-a", "development-round-b"}:
        prompt_ids = protocol["development"]["prompt_ids"]
        index_field = (
            "round_a_seed_indexes" if phase == "development-round-a" else "round_b_seed_indexes"
        )
        indexes = protocol["development"][index_field]
        cases = [
            (prompt_id, prompt_by_id[prompt_id]["seeds"][index])
            for prompt_id in prompt_ids
            for index in indexes
        ]
        if phase == "development-round-a":
            systems = list(protocol["development"]["systems"])
        else:
            systems = _require_list(
                round_b_systems,
                "manifest.scope.system_ids",
            )
            baseline_id = protocol["baseline"]["system_id"]
            registered_candidates = {
                item["system_id"] for item in protocol["repair_candidates"]
            }
            if (
                len(systems) < 2
                or systems[0] != baseline_id
                or len(systems) > protocol["development"]["max_finalists"] + 1
                or len(set(systems)) != len(systems)
                or any(system_id not in registered_candidates for system_id in systems[1:])
            ):
                raise ProtocolError(
                    "Round-B manifest systems must be baseline plus one to max_finalists registered candidates"
                )
    elif phase == "sentinel":
        if selection_lock is None:
            raise ProtocolError("sentinel manifest requires a selection lock")
        validate_selection_lock(protocol, selection_lock)
        prompt_ids = protocol["confirmatory"]["long_horizon_prompt_ids"]
        cases = [
            (prompt_id, seed)
            for prompt_id in prompt_ids
            for seed in prompt_by_id[prompt_id]["seeds"]
        ]
        systems = [selection_lock["finalist_system_id"]]
    else:
        raise ProtocolError(f"unsupported manifest phase {phase}")
    return cases, systems, prompt_by_id


def validate_run_manifest(
    protocol: dict,
    manifest: dict,
    *,
    selection_lock: dict | None = None,
) -> dict:
    """Validate an exact prompt/seed/system tensor and its per-system runs."""

    validate_protocol(protocol, require_frozen=True)
    manifest = _require_mapping(manifest, "manifest")
    if manifest.get("schema_version") != 1:
        raise ProtocolError("manifest.schema_version must be 1")
    protocol_hash = canonical_sha256(protocol)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ProtocolError("manifest protocol_sha256 does not match protocol")
    scope = _require_mapping(manifest.get("scope"), "manifest.scope")
    phase = scope.get("phase")
    declared_systems = _require_list(
        scope.get("system_ids"), "manifest.scope.system_ids"
    )
    expected_cases, expected_systems, prompt_by_id = _expected_scope(
        protocol,
        phase,
        selection_lock,
        declared_systems if phase == "development-round-b" else None,
    )
    declared_cases = _require_list(scope.get("cases"), "manifest.scope.cases")
    actual_declared_cases: list[tuple[str, int]] = []
    for item in declared_cases:
        item = _require_mapping(item, "manifest.scope case")
        prompt_id = item.get("prompt_id")
        seed = item.get("seed")
        if not _is_int(seed):
            raise ProtocolError("manifest scope seed must be an integer")
        actual_declared_cases.append((prompt_id, seed))
    if actual_declared_cases != expected_cases:
        raise ProtocolError("manifest scope cases do not match the complete registered case list")
    if declared_systems != expected_systems:
        raise ProtocolError("manifest scope systems do not match the registered system list")

    if selection_lock is not None:
        expected_lock_hash = canonical_sha256(selection_lock)
        if manifest.get("selection_lock_sha256") != expected_lock_hash:
            raise ProtocolError("manifest selection_lock_sha256 does not match selection lock")
    elif "selection_lock_sha256" in manifest:
        raise ProtocolError("development manifest must not carry a selection lock")

    runs = _require_list(manifest.get("runs"), "manifest.runs")
    run_by_id: dict[str, dict] = {}
    run_by_system: dict[str, dict] = {}
    run_required = {
        "run_id",
        "system_id",
        "source_commit",
        "source_diff_sha256",
        "checkpoint_revision",
        "checkpoint_sha256",
        "decoder_revision",
        "decoder_sha256",
        "runner_sha256",
        "config_sha256",
        "rubric_sha256",
        "hardware",
        "software",
        "determinism",
        "encoding",
    }
    for run in runs:
        run = _require_mapping(run, "manifest run")
        missing = sorted(run_required - run.keys())
        if missing:
            raise ProtocolError(f"manifest run missing {missing[0]}")
        run_id = run["run_id"]
        system_id = run["system_id"]
        if not isinstance(run_id, str) or not run_id or run_id in run_by_id:
            raise ProtocolError("manifest run_id must be present and unique")
        if system_id in run_by_system:
            raise ProtocolError(f"manifest has multiple runs for system {system_id}")
        if system_id not in expected_systems:
            raise ProtocolError(f"manifest run has unexpected system {system_id}")
        _require_commit(run["source_commit"], f"manifest run {run_id} source_commit")
        for field in (
            "source_diff_sha256",
            "checkpoint_sha256",
            "decoder_sha256",
            "runner_sha256",
            "config_sha256",
            "rubric_sha256",
        ):
            _require_sha256(run[field], f"manifest run {run_id} {field}")
        if run["rubric_sha256"] != protocol["evaluation"]["rubric_sha256"]:
            raise ProtocolError(f"manifest run {run_id} rubric_sha256 does not match protocol")
        hardware = _require_mapping(run["hardware"], f"manifest run {run_id} hardware")
        for field in ("gpu_model", "gpu_uuid", "driver_version", "cuda_version"):
            if not isinstance(hardware.get(field), str) or not hardware[field]:
                raise ProtocolError(f"manifest run {run_id} hardware.{field} is required")
        software = _require_mapping(run["software"], f"manifest run {run_id} software")
        for field in ("python_version", "torch_version"):
            if not isinstance(software.get(field), str) or not software[field]:
                raise ProtocolError(f"manifest run {run_id} software.{field} is required")
        _require_sha256(
            software.get("environment_lock_sha256"),
            f"manifest run {run_id} software.environment_lock_sha256",
        )
        determinism = _require_mapping(
            run["determinism"], f"manifest run {run_id} determinism"
        )
        for field in (
            "torch_deterministic_algorithms",
            "cudnn_benchmark",
            "allow_tf32",
        ):
            if not isinstance(determinism.get(field), bool):
                raise ProtocolError(
                    f"manifest run {run_id} determinism.{field} must be boolean"
                )
        encoding = _require_mapping(run["encoding"], f"manifest run {run_id} encoding")
        if not isinstance(encoding.get("command"), str) or not encoding["command"]:
            raise ProtocolError(f"manifest run {run_id} encoding.command is required")
        if encoding.get("codec") != protocol["media_contract"]["codec"]:
            raise ProtocolError(f"manifest run {run_id} encoding.codec mismatch")
        if encoding.get("pixel_format") != protocol["media_contract"]["pixel_format"]:
            raise ProtocolError(f"manifest run {run_id} encoding.pixel_format mismatch")
        run_by_id[run_id] = run
        run_by_system[system_id] = run
    if set(run_by_system) != set(expected_systems):
        raise ProtocolError("manifest runs do not cover the complete registered system list")

    baseline = protocol["baseline"]
    candidate_by_id = {
        item["system_id"]: item for item in protocol["repair_candidates"]
    }
    for system_id, run in run_by_system.items():
        if system_id == "sf4-reference":
            pinned = protocol["reference_systems"]["sf4-reference"]
        else:
            pinned = baseline
        for manifest_field, pin_field in (
            ("source_commit", "commit"),
            ("source_diff_sha256", "source_diff_sha256"),
            ("checkpoint_revision", "checkpoint"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("decoder_revision", "decoder"),
            ("decoder_sha256", "decoder_sha256"),
            ("runner_sha256", "runner_sha256"),
        ):
            if run[manifest_field] != pinned[pin_field]:
                raise ProtocolError(
                    f"manifest run {system_id} {manifest_field} does not match frozen system pin"
                )
        expected_config_sha256 = (
            pinned["config_sha256"]
            if system_id in {baseline["system_id"], "sf4-reference"}
            else candidate_by_id[system_id]["config_sha256"]
        )
        if run["config_sha256"] != expected_config_sha256:
            raise ProtocolError(
                f"manifest run {system_id} config_sha256 does not match frozen system pin"
            )

    records = _require_list(manifest.get("records"), "manifest.records")
    expected_grid = {
        (system_id, prompt_id, seed)
        for prompt_id, seed in expected_cases
        for system_id in expected_systems
    }
    observed_grid: set[tuple[str, str, int]] = set()
    artifact_ids: set[str] = set()
    source_files: set[str] = set()
    media_hashes: set[str] = set()
    records_by_case: dict[tuple[str, int], list[dict]] = defaultdict(list)
    record_required = {
        "artifact_id",
        "run_id",
        "system_id",
        "prompt_id",
        "split",
        "prompt_utf8_sha256",
        "effective_prompt_utf8_sha256",
        "seed",
        "rng_algorithm",
        "rng_device",
        "seed_application_point",
        "initial_noise_sha256",
        "input_noise_sha256",
        "latent_sha256",
        "source_file",
        "media_sha256",
        "runner_sha256",
        "decoder_mode",
        "forwards",
        "decoded_frames",
        "media_contract_id",
    }
    expected_contract_id = "long" if phase == "sentinel" else "short"
    expected_frames = 961 if phase == "sentinel" else 81
    for record in records:
        record = _require_mapping(record, "manifest record")
        missing = sorted(record_required - record.keys())
        if missing:
            raise ProtocolError(f"manifest record missing {missing[0]}")
        artifact_id = record["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in artifact_ids:
            raise ProtocolError("manifest artifact_id must be present and unique")
        artifact_ids.add(artifact_id)
        if record["run_id"] not in run_by_id:
            raise ProtocolError(f"manifest record {artifact_id} has unknown run_id")
        run = run_by_id[record["run_id"]]
        if run["system_id"] != record["system_id"]:
            raise ProtocolError(f"manifest record {artifact_id} run/system mismatch")
        if not _is_int(record["seed"]):
            raise ProtocolError("manifest record seed must be an integer")
        key = (record["system_id"], record["prompt_id"], record["seed"])
        if key in observed_grid:
            raise ProtocolError(f"duplicate manifest semantic record {key}")
        observed_grid.add(key)
        if record["split"] != phase:
            raise ProtocolError(f"manifest record {artifact_id} split does not match scope")
        prompt = prompt_by_id.get(record["prompt_id"])
        if not prompt:
            raise ProtocolError(f"manifest record {artifact_id} has unknown prompt")
        prompt_hash = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        if record["prompt_utf8_sha256"] != prompt_hash:
            raise ProtocolError(f"manifest record {artifact_id} prompt hash mismatch")
        prompt_suffix = (
            candidate_by_id.get(record["system_id"], {})
            .get("changes", {})
            .get("prompt_suffix", "")
        )
        effective_prompt_hash = hashlib.sha256(
            (prompt["text"] + prompt_suffix).encode("utf-8")
        ).hexdigest()
        if record["effective_prompt_utf8_sha256"] != effective_prompt_hash:
            raise ProtocolError(
                f"manifest record {artifact_id} effective prompt hash mismatch"
            )
        for field in (
            "prompt_utf8_sha256",
            "effective_prompt_utf8_sha256",
            "initial_noise_sha256",
            "input_noise_sha256",
            "latent_sha256",
            "media_sha256",
            "runner_sha256",
        ):
            _require_sha256(record[field], f"manifest record {artifact_id} {field}")
        if record["runner_sha256"] != run["runner_sha256"]:
            raise ProtocolError(f"manifest record {artifact_id} runner_sha256 mismatch")
        expected_decoder_mode = (
            protocol["reference_systems"]["sf4-reference"]["decoder_mode"]
            if record["system_id"] == "sf4-reference"
            else baseline["decoder_mode"]
        )
        if record["decoder_mode"] != expected_decoder_mode:
            raise ProtocolError(f"manifest record {artifact_id} decoder_mode mismatch")
        expected_forwards = (
            causal_forward_count(241, 1, 4, 1)
            if phase == "sentinel"
            else (
                protocol["reference_systems"]["sf4-reference"]["forwards"]
                if record["system_id"] == "sf4-reference"
                else protocol["gates"]["required_forwards"]
            )
        )
        if record["forwards"] != expected_forwards:
            raise ProtocolError(f"manifest record {artifact_id} has wrong forwards")
        if record["decoded_frames"] != expected_frames:
            raise ProtocolError(f"manifest record {artifact_id} has wrong decoded_frames")
        if record["media_contract_id"] != expected_contract_id:
            raise ProtocolError(f"manifest record {artifact_id} has wrong media_contract_id")
        if record["source_file"] in source_files:
            raise ProtocolError(f"duplicate source_file {record['source_file']}")
        source_files.add(record["source_file"])
        if record["media_sha256"] in media_hashes:
            raise ProtocolError(f"duplicate media_sha256 {record['media_sha256']}")
        media_hashes.add(record["media_sha256"])
        records_by_case[(record["prompt_id"], record["seed"])].append(record)
    if observed_grid != expected_grid:
        raise ProtocolError("manifest records do not form the complete grid")
    for case, group in records_by_case.items():
        initial_hashes = {record["initial_noise_sha256"] for record in group}
        if len(initial_hashes) != 1:
            raise ProtocolError(f"paired systems do not share initial noise for {case}")

    return {
        "phase": phase,
        "protocol_sha256": protocol_hash,
        "manifest_sha256": canonical_sha256(manifest),
        "cases": expected_cases,
        "systems": expected_systems,
        "runs": run_by_id,
        "records": records,
        "records_by_artifact": {record["artifact_id"]: record for record in records},
    }


def build_blind_plan(
    protocol: dict,
    manifest: dict,
    family_id: str,
    pass_id: int,
    blind_secret: bytes,
    *,
    selection_lock: dict | None = None,
    rater_id: str | None = None,
) -> tuple[dict, dict]:
    manifest_index = validate_run_manifest(
        protocol, manifest, selection_lock=selection_lock
    )
    if not isinstance(blind_secret, bytes) or len(blind_secret) != 32:
        raise ProtocolError("blind_secret must contain exactly 32 bytes")
    family_ids = {item["id"] for item in protocol["evaluation"]["families"]}
    if family_id not in family_ids:
        raise ProtocolError(f"unknown blind family {family_id}")
    family = next(item for item in protocol["evaluation"]["families"] if item["id"] == family_id)
    if not _is_int(pass_id) or not 1 <= pass_id <= family["passes"]:
        raise ProtocolError("pass_id must be a positive integer")
    registered_raters = family.get("rater_ids", [])
    effective_rater_id = (
        rater_id
        if rater_id is not None
        else registered_raters[0]
        if len(registered_raters) == 1
        else None
    )
    if family["kind"] == "human" and rater_id is None:
        raise ProtocolError("human blind plans require an explicit rater_id")
    if not isinstance(effective_rater_id, str) or not effective_rater_id:
        raise ProtocolError("rater_id must be a non-empty string")
    if effective_rater_id not in registered_raters:
        raise ProtocolError("rater_id is not registered for the blind family")
    records = manifest_index["records"]
    systems = list(manifest_index["systems"])
    if len(systems) > 26:
        raise ProtocolError("blind plans support at most 26 systems")
    by_case: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_case[(record["prompt_id"], record["seed"])][record["system_id"]] = record
    prompt_by_id = {item["id"]: item for item in protocol["prompts"]}
    context = (
        f"{canonical_sha256(protocol)}:{protocol['evaluation']['blind_seed']}:"
        f"{manifest_index['manifest_sha256']}:{family_id}:{effective_rater_id}:{pass_id}"
    ).encode()
    seed_material = hmac.new(blind_secret, context, hashlib.sha256).digest()
    rng = random.Random(int.from_bytes(seed_material, "big"))
    groups = list(manifest_index["cases"])
    rng.shuffle(groups)
    permutations = list(itertools.permutations(systems))
    slot_counts = {system_id: [0] * len(systems) for system_id in systems}

    def choose_permutation(case: tuple[str, int]) -> tuple[str, ...]:
        ranked = []
        for permutation in permutations:
            projected = {system_id: counts[:] for system_id, counts in slot_counts.items()}
            for slot_index, system_id in enumerate(permutation):
                projected[system_id][slot_index] += 1
            imbalance = max(max(values) - min(values) for values in projected.values())
            square_penalty = sum(value * value for values in projected.values() for value in values)
            tie = hmac.new(
                blind_secret,
                seed_material + repr((case, permutation)).encode(),
                hashlib.sha256,
            ).digest()
            ranked.append(((imbalance, square_penalty, tie), permutation))
        chosen = min(ranked, key=lambda item: item[0])[1]
        for slot_index, system_id in enumerate(chosen):
            slot_counts[system_id][slot_index] += 1
        return chosen

    public_records: list[dict] = []
    key_records: list[dict] = []
    for prompt_id, seed in groups:
        case_id = hmac.new(
            blind_secret,
            seed_material + f":case:{prompt_id}:{seed}".encode(),
            hashlib.sha256,
        ).hexdigest()[:20]
        slot_order = choose_permutation((prompt_id, seed))
        for slot_index, system_id in enumerate(slot_order):
            source = by_case[(prompt_id, seed)][system_id]
            blind_id = hmac.new(
                blind_secret,
                seed_material
                + f":{prompt_id}:{seed}:{system_id}".encode(),
                hashlib.sha256,
            ).hexdigest()[:20]
            public_records.append(
                {
                    "blind_id": blind_id,
                    "asset_id": f"asset-{blind_id}",
                    "media_sha256": source["media_sha256"],
                    "case_id": case_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt_by_id[prompt_id]["text"],
                    "slot": chr(ord("A") + slot_index),
                }
            )
            key_records.append(
                {
                    "blind_id": blind_id,
                    "artifact_id": source["artifact_id"],
                }
            )
    rng.shuffle(public_records)
    public = {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": manifest_index["manifest_sha256"],
        "family_id": family_id,
        "rater_id": effective_rater_id,
        "pass_id": pass_id,
        "records": public_records,
    }
    key = {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": manifest_index["manifest_sha256"],
        "blind_plan_sha256": canonical_sha256(public),
        "family_id": family_id,
        "rater_id": effective_rater_id,
        "pass_id": pass_id,
        "blind_secret_sha256": hashlib.sha256(blind_secret).hexdigest(),
        "records": sorted(key_records, key=lambda item: item["blind_id"]),
    }
    return public, key


def _validate_score(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 1 <= value <= 10
    ):
        raise ProtocolError(f"{label} must be a finite score in [1, 10]")
    return float(value)


_RAW_RATING_FIELDS = (
    "blind_id",
    "media_sha256",
    "scores",
    "first_third_quality",
    "final_third_quality",
    "failure_tags",
    "rationale",
)


def _raw_rating_from_normalized(row: dict) -> dict:
    return {field: row[field] for field in _RAW_RATING_FIELDS}


def _evidence_family(protocol: dict, family_id: str) -> dict:
    family = next(
        (
            item
            for item in protocol["evaluation"]["families"]
            if item.get("id") == family_id
        ),
        None,
    )
    if family is None:
        raise ProtocolError(f"raw evidence references unknown family {family_id}")
    return family


def validate_raw_evidence_report(
    protocol: dict,
    public_plan: dict,
    raw_ratings: list[dict],
    raw_evidence_report: dict,
) -> str:
    """Validate one provider/human report and prove it produced the raw ratings."""

    public_plan = _require_mapping(public_plan, "public plan")
    report = _require_mapping(raw_evidence_report, "raw evidence report")
    required = {
        "schema_version",
        "provider",
        "model_id",
        "protocol_sha256",
        "manifest_sha256",
        "blind_plan_sha256",
        "family_id",
        "rater_id",
        "pass_id",
        "records",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ProtocolError(f"raw evidence report missing {missing[0]}")
    if report["schema_version"] != 1:
        raise ProtocolError("raw evidence report must use schema_version 1")
    for field in (
        "protocol_sha256",
        "manifest_sha256",
        "blind_plan_sha256",
        "family_id",
        "rater_id",
        "pass_id",
    ):
        expected = (
            canonical_sha256(public_plan)
            if field == "blind_plan_sha256"
            else public_plan.get(field)
        )
        if report[field] != expected:
            raise ProtocolError(f"raw evidence report {field} mismatch")
    family = _evidence_family(protocol, report["family_id"])
    if report["provider"] != family["evidence_provider"]:
        raise ProtocolError("raw evidence report provider mismatch")
    if report["model_id"] != family["model_id"]:
        raise ProtocolError("raw evidence report model_id mismatch")

    public_records = _require_list(public_plan.get("records"), "public plan records")
    evidence_records = _require_list(report["records"], "raw evidence records")
    raw_ratings = _require_list(raw_ratings, "raw ratings")

    def unique_index(rows: list[dict], label: str) -> dict[str, dict]:
        index: dict[str, dict] = {}
        for value in rows:
            row = _require_mapping(value, f"{label} row")
            blind_id = row.get("blind_id")
            if not isinstance(blind_id, str) or not blind_id or blind_id in index:
                raise ProtocolError(f"{label} blind IDs must be present and unique")
            index[blind_id] = row
        return index

    public_by_id = unique_index(public_records, "public plan")
    evidence_by_id = unique_index(evidence_records, "raw evidence")
    raw_by_id = unique_index(raw_ratings, "raw ratings")
    if not (set(public_by_id) == set(evidence_by_id) == set(raw_by_id)):
        raise ProtocolError("raw evidence must cover the complete blind-id set")
    for blind_id, record in evidence_by_id.items():
        if record.get("media_sha256") != public_by_id[blind_id].get("media_sha256"):
            raise ProtocolError("raw evidence media_sha256 mismatch")
        _require_sha256(
            record.get("raw_response_sha256"),
            "raw evidence raw_response_sha256",
        )

    provider = report["provider"]
    if provider == "twelvelabs":
        from bench.video_judge import ratings_from_pegasus_evidence

        evidenced_ratings = ratings_from_pegasus_evidence(
            protocol,
            public_plan,
            report,
        )
    elif provider == "google":
        from bench.gemini_video_judge import ratings_from_gemini_evidence

        evidenced_ratings = ratings_from_gemini_evidence(
            protocol,
            public_plan,
            report,
        )
    elif provider in {"human", "test-fixture"}:
        evidenced_ratings = []
        for record in evidence_records:
            raw = raw_by_id[record["blind_id"]]
            if record["raw_response_sha256"] != canonical_sha256(raw):
                raise ProtocolError("raw evidence response hash does not match rating")
            evidenced_ratings.append(raw)
    else:
        raise ProtocolError(f"unsupported raw evidence provider {provider}")
    if raw_ratings != evidenced_ratings:
        raise ProtocolError("raw evidence ratings do not match submitted ratings")
    return canonical_sha256(report)


def unblind_ratings(
    protocol: dict,
    manifest: dict,
    public_plan: dict,
    key: dict,
    raw_ratings: list[dict],
    *,
    blind_secret: bytes,
    family_id: str,
    pass_id: int,
    rater_id: str,
    raw_evidence_report: dict,
    selection_lock: dict | None = None,
) -> list[dict]:
    """Join judge-visible blind IDs to trusted manifest artifacts, fail-closed."""

    expected_public, expected_key = build_blind_plan(
        protocol,
        manifest,
        family_id,
        pass_id,
        blind_secret,
        selection_lock=selection_lock,
        rater_id=rater_id,
    )
    if public_plan != expected_public or key != expected_key:
        raise ProtocolError("unblinding key integrity check failed")
    manifest_index = validate_run_manifest(
        protocol, manifest, selection_lock=selection_lock
    )
    protocol_hash = canonical_sha256(protocol)
    for label, value in (("public plan", public_plan), ("unblinding key", key)):
        value = _require_mapping(value, label)
        if value.get("protocol_sha256") != protocol_hash:
            raise ProtocolError(f"{label} protocol_sha256 mismatch")
        if value.get("manifest_sha256") != manifest_index["manifest_sha256"]:
            raise ProtocolError(f"{label} manifest_sha256 mismatch")
        if value.get("family_id") != family_id or value.get("pass_id") != pass_id:
            raise ProtocolError(f"{label} family/pass mismatch")
        if value.get("rater_id") != rater_id:
            raise ProtocolError(f"{label} rater_id mismatch")
    if key.get("blind_plan_sha256") != canonical_sha256(public_plan):
        raise ProtocolError("unblinding key blind_plan_sha256 mismatch")
    raw_evidence_sha256 = canonical_sha256(
        _require_mapping(raw_evidence_report, "raw evidence report")
    )

    public_rows = _require_list(public_plan.get("records"), "public_plan.records")
    key_rows = _require_list(key.get("records"), "key.records")
    raw_ratings = _require_list(raw_ratings, "raw_ratings")

    def unique_index(rows: list[dict], label: str) -> dict[str, dict]:
        index: dict[str, dict] = {}
        for row in rows:
            row = _require_mapping(row, f"{label} row")
            blind_id = row.get("blind_id")
            if not isinstance(blind_id, str) or not blind_id or blind_id in index:
                raise ProtocolError(f"{label} blind IDs must be present and unique")
            index[blind_id] = row
        return index

    public_by_id = unique_index(public_rows, "public plan")
    key_by_id = unique_index(key_rows, "unblinding key")
    ratings_by_id = unique_index(raw_ratings, "raw ratings")
    if not (set(public_by_id) == set(key_by_id) == set(ratings_by_id)):
        raise ProtocolError("raw ratings must cover the complete blind-id set")

    dimensions = protocol["evaluation"]["dimensions"]
    records_by_artifact = manifest_index["records_by_artifact"]
    blind_plan_sha256 = canonical_sha256(public_plan)
    unblinding_key_sha256 = canonical_sha256(key)
    raw_ratings_sha256 = canonical_sha256(raw_ratings)
    normalized: list[dict] = []
    forbidden = {"artifact_id", "system_id", "source_file", "prompt_id", "seed"}
    for blind_id in sorted(ratings_by_id):
        raw = ratings_by_id[blind_id]
        leaked = sorted(forbidden & raw.keys())
        if leaked:
            raise ProtocolError(f"blind rating must not contain {leaked[0]}")
        artifact_id = key_by_id[blind_id].get("artifact_id")
        record = records_by_artifact.get(artifact_id)
        if record is None:
            raise ProtocolError(f"unblinding key references unknown artifact {artifact_id}")
        visible_media_sha256 = raw.get("media_sha256")
        expected_media_sha256 = public_by_id[blind_id].get("media_sha256")
        if (
            visible_media_sha256 != expected_media_sha256
            or visible_media_sha256 != record["media_sha256"]
        ):
            raise ProtocolError("rating judge-visible media SHA-256 mismatch")
        scores = _require_mapping(raw.get("scores"), "raw rating scores")
        if set(scores) != set(dimensions):
            raise ProtocolError("rating scores do not exactly match protocol dimensions")
        for name, value in scores.items():
            _validate_score(value, f"rating {name}")
        _validate_score(raw.get("first_third_quality"), "first_third_quality")
        _validate_score(raw.get("final_third_quality"), "final_third_quality")
        failure_tags = _require_list(raw.get("failure_tags"), "failure_tags")
        if any(not isinstance(tag, str) for tag in failure_tags):
            raise ProtocolError("failure_tags must contain strings")
        rationale = raw.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProtocolError("rating rationale is required")
        normalized.append(
            {
                "family_id": family_id,
                "rater_id": rater_id,
                "pass_id": pass_id,
                "blind_id": blind_id,
                "artifact_id": artifact_id,
                "media_sha256": visible_media_sha256,
                "protocol_sha256": protocol_hash,
                "manifest_sha256": manifest_index["manifest_sha256"],
                "blind_plan_sha256": blind_plan_sha256,
                "unblinding_key_sha256": unblinding_key_sha256,
                "raw_ratings_sha256": raw_ratings_sha256,
                "raw_evidence_sha256": raw_evidence_sha256,
                "system_id": record["system_id"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "scores": dict(scores),
                "first_third_quality": raw["first_third_quality"],
                "final_third_quality": raw["final_third_quality"],
                "failure_tags": list(failure_tags),
                "rationale": rationale,
            }
        )
    validate_raw_evidence_report(
        protocol,
        public_plan,
        raw_ratings,
        raw_evidence_report,
    )
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if len(videos) != 1:
            raise ValueError(f"expected exactly one video stream, found {len(videos)}")
        stream = videos[0]
        rate = Fraction(stream["r_frame_rate"])
        frames = stream.get("nb_read_frames") or stream.get("nb_frames")
        duration = stream.get("duration") or payload.get("format", {}).get("duration")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(rate),
            "decoded_frames": int(frames),
            "duration_s": float(duration),
            "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"],
            "video_streams": len(videos),
            "audio_streams": len(audios),
        }
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe failed for {path}: {exc}") from exc


def audit_media(
    records: Iterable[dict],
    media_contract: dict,
    probe: Callable[[Path], dict] | None = None,
    *,
    protocol_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict:
    probe = probe or _ffprobe
    expected = _require_mapping(media_contract, "media_contract")
    report_records: list[dict] = []
    seen_paths: set[Path] = set()
    for record in records:
        logical_source_file = record.get("source_file", "")
        path = Path(record.get("physical_source_file", logical_source_file))
        if not path.is_file():
            raise ProtocolError(f"media file does not exist: {path}")
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ProtocolError(f"duplicate media file: {path}")
        seen_paths.add(resolved)
        expected_hash = record.get("media_sha256")
        actual_hash = _sha256_file(path)
        if not _is_sha256(expected_hash) or actual_hash != expected_hash:
            raise ProtocolError(f"media sha256 mismatch for {path}")
        actual = probe(path)
        for field in ("width", "height", "decoded_frames"):
            if int(actual[field]) != int(expected[field]):
                raise ProtocolError(
                    f"{path} {field}={actual[field]} does not match {expected[field]}"
                )
        if not math.isclose(float(actual["fps"]), float(expected["fps"]), abs_tol=1e-6):
            raise ProtocolError(f"{path} fps={actual['fps']} does not match {expected['fps']}")
        if actual.get("codec") != expected.get("codec"):
            raise ProtocolError(f"{path} codec={actual.get('codec')} does not match {expected.get('codec')}")
        if actual.get("pixel_format") != expected.get("pixel_format"):
            raise ProtocolError(
                f"{path} pixel_format={actual.get('pixel_format')} does not match {expected.get('pixel_format')}"
            )
        if actual.get("video_streams") != 1 or actual.get("audio_streams") != 0:
            raise ProtocolError(f"{path} must contain exactly one video stream and no audio")
        expected_duration = expected["decoded_frames"] / expected["fps"]
        tolerance_s = expected.get("duration_tolerance_frames", 1) / expected["fps"]
        if abs(float(actual["duration_s"]) - expected_duration) > tolerance_s:
            raise ProtocolError(
                f"{path} duration_s={actual['duration_s']} is outside media contract"
            )
        report_records.append(
            {
                **({"artifact_id": record["artifact_id"]} if record.get("artifact_id") else {}),
                "source_file": str(logical_source_file),
                "media_sha256": actual_hash,
                **actual,
            }
        )
    report = {
        "ok": True,
        "media_contract": expected,
        "records": report_records,
    }
    if protocol_sha256 is not None:
        _require_sha256(protocol_sha256, "media report protocol_sha256")
        report["protocol_sha256"] = protocol_sha256
    if manifest_sha256 is not None:
        _require_sha256(manifest_sha256, "media report manifest_sha256")
        report["manifest_sha256"] = manifest_sha256
    report["report_sha256"] = canonical_sha256(report)
    return report


def _validate_bound_media_report(
    protocol: dict,
    manifest_index: dict,
    media_report: dict,
) -> dict:
    media_report = _require_mapping(media_report, "media report")
    _verify_canonical_hash(media_report, "report_sha256", "media report")
    if not media_report.get("ok"):
        raise ProtocolError("media report is not successful")
    if media_report.get("protocol_sha256") != canonical_sha256(protocol):
        raise ProtocolError("media report protocol_sha256 mismatch")
    if media_report.get("manifest_sha256") != manifest_index["manifest_sha256"]:
        raise ProtocolError("media report manifest_sha256 mismatch")
    expected_contract = (
        protocol["long_horizon_media_contract"]
        if manifest_index["phase"] == "sentinel"
        else protocol["media_contract"]
    )
    if media_report.get("media_contract") != expected_contract:
        raise ProtocolError("media report media_contract mismatch")
    report_records = _require_list(media_report.get("records"), "media report records")
    by_artifact: dict[str, dict] = {}
    for row in report_records:
        row = _require_mapping(row, "media report record")
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in by_artifact:
            raise ProtocolError("media report artifact IDs must be present and unique")
        by_artifact[artifact_id] = row
    expected_by_artifact = manifest_index["records_by_artifact"]
    if set(by_artifact) != set(expected_by_artifact):
        raise ProtocolError("media report does not cover the complete artifact set")
    for artifact_id, expected_record in expected_by_artifact.items():
        actual = by_artifact[artifact_id]
        if actual.get("source_file") != expected_record["source_file"]:
            raise ProtocolError(f"media report source_file mismatch for {artifact_id}")
        if actual.get("media_sha256") != expected_record["media_sha256"]:
            raise ProtocolError(f"media report media_sha256 mismatch for {artifact_id}")
        for field in ("width", "height", "decoded_frames"):
            if actual.get(field) != expected_contract[field]:
                raise ProtocolError(f"media report {field} mismatch for {artifact_id}")
        if not math.isclose(
            float(actual.get("fps", float("nan"))),
            float(expected_contract["fps"]),
            abs_tol=1e-6,
        ):
            raise ProtocolError(f"media report fps mismatch for {artifact_id}")
        if actual.get("codec") != expected_contract["codec"]:
            raise ProtocolError(f"media report codec mismatch for {artifact_id}")
        if actual.get("pixel_format") != expected_contract["pixel_format"]:
            raise ProtocolError(f"media report pixel_format mismatch for {artifact_id}")
        if actual.get("video_streams") != 1 or actual.get("audio_streams") != 0:
            raise ProtocolError(
                f"media report stream contract mismatch for {artifact_id}"
            )
        duration = actual.get("duration_s")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ProtocolError(f"media report duration_s invalid for {artifact_id}")
        expected_duration = expected_contract["decoded_frames"] / expected_contract["fps"]
        tolerance = (
            expected_contract.get("duration_tolerance_frames", 1)
            / expected_contract["fps"]
        )
        if abs(duration - expected_duration) > tolerance:
            raise ProtocolError(f"media report duration_s mismatch for {artifact_id}")
    return media_report


def ratings_from_legacy_eval(legacy: dict, manifests: dict[str, list], family_id: str) -> list[dict]:
    file_lookup: dict[str, tuple[str, str, int]] = {}
    for system_id, manifest in manifests.items():
        for item in manifest:
            file_lookup[item["file"]] = (system_id, f"p{item['prompt_index']}", int(item["seed"]))
    mapping = legacy["model_ab"]["mapping"]
    rows: list[dict] = []
    for pass_number in (1, 2):
        pass_data = legacy["model_ab"][f"pass_{pass_number}"]
        for prompt_label, prompt_result in pass_data.items():
            prompt_index = int(prompt_label[1:])
            for system_id, score in prompt_result["overall_scores"].items():
                mapped_files = [
                    filename
                    for blind_label, filename in mapping.items()
                    if blind_label.startswith(f"P{prompt_index}-") and file_lookup[filename][0] == system_id
                ]
                if len(mapped_files) != 1:
                    raise ProtocolError(f"legacy mapping cannot resolve {prompt_label}/{system_id}")
                resolved_system, prompt_id, seed = file_lookup[mapped_files[0]]
                rows.append(
                    {
                        "family_id": family_id,
                        "rater_id": legacy["evaluator"],
                        "pass_id": pass_number,
                        "system_id": resolved_system,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "scores": {"overall": score},
                    }
                )
    return rows


def aggregate_ratings(
    protocol: dict,
    ratings: list[dict],
    *,
    manifest: dict | None = None,
    selection_lock: dict | None = None,
    blind_secret: bytes | None = None,
    raw_evidence_reports: list[dict] | None = None,
    allow_unbound_legacy: bool = False,
    require_complete: bool = False,
) -> dict:
    dimensions = _require_mapping(
        protocol["evaluation"]["dimensions"], "evaluation.dimensions"
    )
    family_defs = {
        item["id"]: item for item in protocol["evaluation"]["families"]
    }
    manifest_index = None
    if manifest is not None:
        manifest_index = validate_run_manifest(
            protocol, manifest, selection_lock=selection_lock
        )
        if not isinstance(blind_secret, bytes) or len(blind_secret) != 32:
            raise ProtocolError(
                "manifest-bound aggregation requires the 32-byte blinding secret"
            )
        if not isinstance(raw_evidence_reports, list) or not raw_evidence_reports:
            raise ProtocolError(
                "manifest-bound aggregation requires raw evidence reports"
            )
    elif require_complete:
        raise ProtocolError("complete aggregation requires a validated manifest")
    elif allow_unbound_legacy is not True:
        raise ProtocolError(
            "unbound aggregation requires explicit legacy opt-in"
        )

    reports_by_group: dict[tuple[str, str, int], dict] = {}
    report_hashes: set[str] = set()
    for value in raw_evidence_reports or []:
        evidence_report = _require_mapping(value, "raw evidence report")
        group_key = (
            evidence_report.get("family_id"),
            evidence_report.get("rater_id"),
            evidence_report.get("pass_id"),
        )
        if (
            not isinstance(group_key[0], str)
            or not isinstance(group_key[1], str)
            or not _is_int(group_key[2])
        ):
            raise ProtocolError("raw evidence report group identity is invalid")
        report_hash = canonical_sha256(evidence_report)
        if group_key in reports_by_group or report_hash in report_hashes:
            raise ProtocolError("duplicate raw evidence report")
        reports_by_group[group_key] = evidence_report
        report_hashes.add(report_hash)

    seen: set[tuple] = set()
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    observed_coordinates: set[tuple] = set()
    evidence_groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    required = (
        "family_id",
        "rater_id",
        "pass_id",
        "system_id",
        "prompt_id",
        "seed",
        "scores",
    )
    for row in ratings:
        row = _require_mapping(row, "rating")
        missing = [field for field in required if field not in row]
        if missing:
            raise ProtocolError(f"rating missing {missing[0]}")
        if row["family_id"] not in family_defs:
            raise ProtocolError(f"unknown rating family {row['family_id']}")
        if not isinstance(row["rater_id"], str) or not row["rater_id"]:
            raise ProtocolError("rating rater_id must be a non-empty string")
        if not _is_int(row["pass_id"]):
            raise ProtocolError("rating pass_id must be an integer")
        if not _is_int(row["seed"]):
            raise ProtocolError("rating seed must be an integer")
        if manifest_index is not None:
            strict_required = {
                "artifact_id",
                "media_sha256",
                "blind_id",
                "protocol_sha256",
                "manifest_sha256",
                "blind_plan_sha256",
                "unblinding_key_sha256",
                "raw_ratings_sha256",
                "raw_evidence_sha256",
                "first_third_quality",
                "final_third_quality",
                "failure_tags",
                "rationale",
            }
            strict_missing = sorted(strict_required - row.keys())
            if strict_missing:
                raise ProtocolError(
                    f"manifest-bound rating missing {strict_missing[0]}"
                )
            if row["protocol_sha256"] != canonical_sha256(protocol):
                raise ProtocolError("rating protocol_sha256 mismatch")
            if row["manifest_sha256"] != manifest_index["manifest_sha256"]:
                raise ProtocolError("rating manifest_sha256 mismatch")
            for field in (
                "blind_plan_sha256",
                "unblinding_key_sha256",
                "raw_ratings_sha256",
                "raw_evidence_sha256",
            ):
                _require_sha256(row[field], f"rating {field}")
            blind_id = row["blind_id"]
            if not isinstance(blind_id, str) or not blind_id:
                raise ProtocolError("rating blind_id must be a non-empty string")
            artifact_id = row["artifact_id"]
            artifact = manifest_index["records_by_artifact"].get(artifact_id)
            if artifact is None:
                raise ProtocolError("rating artifact_id is not in the manifest")
            coordinate = (row["system_id"], row["prompt_id"], row["seed"])
            artifact_coordinate = (
                artifact["system_id"],
                artifact["prompt_id"],
                artifact["seed"],
            )
            if coordinate != artifact_coordinate:
                raise ProtocolError("rating coordinate does not match artifact_id")
            if row["media_sha256"] != artifact["media_sha256"]:
                raise ProtocolError("rating media_sha256 does not match artifact_id")
            failure_tags = _require_list(row["failure_tags"], "rating failure_tags")
            if any(not isinstance(tag, str) for tag in failure_tags):
                raise ProtocolError("rating failure_tags must contain strings")
            if not isinstance(row["rationale"], str) or not row["rationale"].strip():
                raise ProtocolError("rating rationale is required")
            group_key = (row["family_id"], row["rater_id"], row["pass_id"])
            evidence = evidence_groups.setdefault(
                group_key,
                {
                    "blind_plan_sha256": set(),
                    "unblinding_key_sha256": set(),
                    "raw_ratings_sha256": set(),
                    "raw_evidence_sha256": set(),
                    "blind_ids": set(),
                    "artifact_ids": set(),
                    "mapping": set(),
                    "rows_by_blind": {},
                },
            )
            evidence["blind_plan_sha256"].add(row["blind_plan_sha256"])
            evidence["unblinding_key_sha256"].add(row["unblinding_key_sha256"])
            evidence["raw_ratings_sha256"].add(row["raw_ratings_sha256"])
            evidence["raw_evidence_sha256"].add(row["raw_evidence_sha256"])
            if blind_id in evidence["blind_ids"]:
                raise ProtocolError("duplicate blind_id in rating evidence group")
            evidence["blind_ids"].add(blind_id)
            evidence["artifact_ids"].add(artifact_id)
            evidence["mapping"].add((blind_id, artifact_id))
            evidence["rows_by_blind"][blind_id] = row
        identity = tuple(row[field] for field in required[:-1])
        if identity in seen:
            raise ProtocolError(f"duplicate rating {identity}")
        seen.add(identity)
        observed_coordinates.add(identity)
        scores = _require_mapping(row["scores"], "rating.scores")
        if set(scores) != set(dimensions):
            raise ProtocolError("rating scores do not exactly match protocol dimensions")
        clean_scores = {
            name: _validate_score(value, f"rating {name}")
            for name, value in scores.items()
        }
        composite = sum(
            clean_scores[name] * float(weight)
            for name, weight in dimensions.items()
        )
        first = (
            _validate_score(row["first_third_quality"], "first_third_quality")
            if "first_third_quality" in row
            else composite
        )
        final = (
            _validate_score(row["final_third_quality"], "final_third_quality")
            if "final_third_quality" in row
            else composite
        )
        grouped[
            (
                row["family_id"],
                row["system_id"],
                row["prompt_id"],
                row["seed"],
            )
        ].append(
            {
                "composite": composite,
                "first": first,
                "final": final,
                "dimensions": clean_scores,
            }
        )

    complete = False
    expected_coordinates: set[tuple] = set()
    if manifest_index is not None:
        cases = manifest_index["cases"]
        systems = manifest_index["systems"]
        for family_id, family in family_defs.items():
            if family["kind"] == "human":
                rater_ids = family.get("rater_ids")
                if not isinstance(rater_ids, list) or len(rater_ids) < family["minimum_raters"]:
                    raise ProtocolError(
                        f"human family {family_id} requires a frozen rater_ids roster"
                    )
            else:
                rater_ids = family.get("rater_ids") or [family["model_id"]]
            for rater_id in rater_ids:
                for pass_id in range(1, family["passes"] + 1):
                    for system_id in systems:
                        for prompt_id, seed in cases:
                            expected_coordinates.add(
                                (
                                    family_id,
                                    rater_id,
                                    pass_id,
                                    system_id,
                                    prompt_id,
                                    seed,
                                )
                            )
        if observed_coordinates != expected_coordinates:
            raise ProtocolError(
                "ratings do not form the complete rating tensor "
                f"(expected {len(expected_coordinates)}, observed {len(observed_coordinates)})"
            )
        expected_artifact_ids = set(manifest_index["records_by_artifact"])
        expected_group_keys: set[tuple[str, str, int]] = set()
        for family_id, family in family_defs.items():
            for rater_id in family["rater_ids"]:
                for pass_id in range(1, family["passes"] + 1):
                    group_key = (family_id, rater_id, pass_id)
                    expected_group_keys.add(group_key)
                    evidence = evidence_groups.get(group_key)
                    if evidence is None:
                        raise ProtocolError("rating evidence group is missing")
                    if evidence["artifact_ids"] != expected_artifact_ids:
                        raise ProtocolError(
                            "rating evidence group does not cover the complete artifact set"
                        )
                    for field in (
                        "blind_plan_sha256",
                        "unblinding_key_sha256",
                        "raw_ratings_sha256",
                        "raw_evidence_sha256",
                    ):
                        if len(evidence[field]) != 1:
                            raise ProtocolError(
                                f"rating evidence group has inconsistent {field}"
                            )
                    expected_public, expected_key = build_blind_plan(
                        protocol,
                        manifest,
                        family_id,
                        pass_id,
                        blind_secret,
                        selection_lock=selection_lock,
                        rater_id=rater_id,
                    )
                    expected_mapping = {
                        (row["blind_id"], row["artifact_id"])
                        for row in expected_key["records"]
                    }
                    if evidence["mapping"] != expected_mapping:
                        raise ProtocolError(
                            "rating evidence mapping does not match keyed blind plan"
                        )
                    if evidence["blind_plan_sha256"] != {
                        canonical_sha256(expected_public)
                    }:
                        raise ProtocolError("rating blind_plan_sha256 mismatch")
                    if evidence["unblinding_key_sha256"] != {
                        canonical_sha256(expected_key)
                    }:
                        raise ProtocolError("rating unblinding_key_sha256 mismatch")
                    raw_evidence_report = reports_by_group.get(group_key)
                    if raw_evidence_report is None:
                        raise ProtocolError("raw evidence report is missing")
                    report_records = _require_list(
                        raw_evidence_report.get("records"),
                        "raw evidence records",
                    )
                    raw_ratings = []
                    for record in report_records:
                        record = _require_mapping(record, "raw evidence record")
                        blind_id = record.get("blind_id")
                        normalized = evidence["rows_by_blind"].get(blind_id)
                        if normalized is None:
                            raise ProtocolError(
                                "raw evidence record is not represented in normalized ratings"
                            )
                        raw_ratings.append(_raw_rating_from_normalized(normalized))
                    raw_evidence_sha256 = validate_raw_evidence_report(
                        protocol,
                        expected_public,
                        raw_ratings,
                        raw_evidence_report,
                    )
                    if evidence["raw_evidence_sha256"] != {raw_evidence_sha256}:
                        raise ProtocolError("rating raw_evidence_sha256 mismatch")
                    if evidence["raw_ratings_sha256"] != {
                        canonical_sha256(raw_ratings)
                    }:
                        raise ProtocolError("rating raw_ratings_sha256 mismatch")
        if set(evidence_groups) != expected_group_keys:
            raise ProtocolError("unexpected rating evidence group")
        if set(reports_by_group) != expected_group_keys:
            raise ProtocolError("raw evidence reports do not exactly cover rating groups")
        complete = True

    family_item: dict[tuple[str, str, str, int], dict] = {}
    for key, values in grouped.items():
        family_item[key] = {
            "mean": sum(value["composite"] for value in values) / len(values),
            "first": sum(value["first"] for value in values) / len(values),
            "final": sum(value["final"] for value in values) / len(values),
            "dimensions": {
                name: sum(value["dimensions"][name] for value in values) / len(values)
                for name in dimensions
            },
            "n_ratings": len(values),
        }

    prompt_by_id = {item["id"]: item for item in protocol["prompts"]}
    systems = sorted({system_id for _family_id, system_id, _prompt_id, _seed in family_item})
    system_report: dict[str, dict] = {}
    low_threshold = protocol.get("gates", {}).get("low_quality_item_threshold", 4.0)
    for system_id in systems:
        families = {
            family_id: sum(
                value["mean"]
                for (fid, sid, _prompt_id, _seed), value in family_item.items()
                if fid == family_id and sid == system_id
            )
            / sum(
                1
                for fid, sid, _prompt_id, _seed in family_item
                if fid == family_id and sid == system_id
            )
            for family_id in family_defs
            if any(fid == family_id and sid == system_id for fid, sid, _p, _s in family_item)
        }
        cases = sorted(
            {
                (prompt_id, seed)
                for _family_id, sid, prompt_id, seed in family_item
                if sid == system_id
            }
        )
        item_rows = []
        for prompt_id, seed in cases:
            by_family = {
                family_id: family_item[(family_id, system_id, prompt_id, seed)]
                for family_id in family_defs
                if (family_id, system_id, prompt_id, seed) in family_item
            }
            item_rows.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "stratum": prompt_by_id.get(prompt_id, {}).get("stratum", "legacy"),
                    "mean": sum(value["mean"] for value in by_family.values()) / len(by_family),
                    "first_third": sum(value["first"] for value in by_family.values()) / len(by_family),
                    "final_third": sum(value["final"] for value in by_family.values()) / len(by_family),
                    "families": {key: value["mean"] for key, value in by_family.items()},
                }
            )
        for item in item_rows:
            item["early_to_late_drop"] = item["first_third"] - item["final_third"]
        strata_values: dict[str, list[float]] = defaultdict(list)
        for item in item_rows:
            strata_values[item["stratum"]].append(item["mean"])
        dimension_means = {}
        for name in dimensions:
            values = [
                value["dimensions"][name]
                for (family_id, sid, _prompt_id, _seed), value in family_item.items()
                if sid == system_id and family_id in family_defs
            ]
            dimension_means[name] = sum(values) / len(values)
        first_mean = sum(item["first_third"] for item in item_rows) / len(item_rows)
        final_mean = sum(item["final_third"] for item in item_rows) / len(item_rows)
        system_report[system_id] = {
            "mean": sum(families.values()) / len(families),
            "families": families,
            "n_families": len(families),
            "dimensions": dimension_means,
            "items": item_rows,
            "strata": {
                stratum: sum(values) / len(values)
                for stratum, values in sorted(strata_values.items())
            },
            "first_third_mean": first_mean,
            "final_third_mean": final_mean,
            "early_to_late_drop": first_mean - final_mean,
            "low_quality_item_threshold": low_threshold,
            "low_quality_items": sum(1 for item in item_rows if item["mean"] < low_threshold),
        }

    paired_deltas: dict[str, float] = {}
    for left in systems:
        for right in systems:
            if left == right:
                continue
            family_deltas = []
            for family_id in family_defs:
                left_items = {
                    (prompt_id, seed): value["mean"]
                    for (fid, sid, prompt_id, seed), value in family_item.items()
                    if fid == family_id and sid == left
                }
                right_items = {
                    (prompt_id, seed): value["mean"]
                    for (fid, sid, prompt_id, seed), value in family_item.items()
                    if fid == family_id and sid == right
                }
                if complete and set(left_items) != set(right_items):
                    raise ProtocolError("complete paired comparison has mismatched case sets")
                matches = sorted(set(left_items) & set(right_items))
                if matches:
                    family_deltas.append(
                        sum(left_items[key] - right_items[key] for key in matches)
                        / len(matches)
                    )
            if family_deltas:
                paired_deltas[f"{left}-vs-{right}"] = sum(family_deltas) / len(family_deltas)

    evidence_summary = [
        {
            "family_id": family_id,
            "rater_id": rater_id,
            "pass_id": pass_id,
            "blind_plan_sha256": next(iter(evidence["blind_plan_sha256"])),
            "unblinding_key_sha256": next(
                iter(evidence["unblinding_key_sha256"])
            ),
            "raw_ratings_sha256": next(iter(evidence["raw_ratings_sha256"])),
            "raw_evidence_sha256": next(
                iter(evidence["raw_evidence_sha256"])
            ),
        }
        for (family_id, rater_id, pass_id), evidence in sorted(
            evidence_groups.items()
        )
    ]
    report = {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        **(
            {"manifest_sha256": manifest_index["manifest_sha256"]}
            if manifest_index is not None
            else {}
        ),
        "complete": complete,
        **(
            {
                "manifest_phase": manifest_index["phase"],
                "selection_lock_sha256": manifest.get("selection_lock_sha256"),
                "rating_evidence": evidence_summary,
                "rating_evidence_sha256": canonical_sha256(evidence_summary),
            }
            if manifest_index is not None
            else {}
        ),
        "coverage": {
            "expected": len(expected_coordinates),
            "observed": len(observed_coordinates),
        },
        "systems": system_report,
        "paired_deltas": paired_deltas,
        "n_ratings": len(ratings),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _validate_raw_timing_trial(
    trial: dict,
    label: str,
    trial_ids: set[str],
) -> dict:
    trial = _require_mapping(trial, label)
    required = {
        "trial_id",
        "media_sha256",
        "e2e_fps",
        "first_visible_rgb_s",
        "p95_effective_frame_interval_ms",
        "forwards",
        "decoded_frames",
        "wall_started_ns",
        "wall_finished_ns",
        "rgb_ready_ns",
    }
    missing = sorted(required - trial.keys())
    if missing:
        raise ProtocolError(f"{label} missing {missing[0]}")
    trial_id = trial["trial_id"]
    if not isinstance(trial_id, str) or not trial_id or trial_id in trial_ids:
        raise ProtocolError("performance trial_id must be present and unique")
    trial_ids.add(trial_id)
    _require_sha256(trial["media_sha256"], f"{label}.media_sha256")
    for field in (
        "e2e_fps",
        "first_visible_rgb_s",
        "p95_effective_frame_interval_ms",
    ):
        value = trial[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ProtocolError(f"{label}.{field} must be positive and finite")
    if not _is_int(trial["forwards"]) or trial["forwards"] <= 0:
        raise ProtocolError(f"{label}.forwards must be a positive integer")
    if not _is_int(trial["decoded_frames"]) or trial["decoded_frames"] < 2:
        raise ProtocolError(
            f"{label}.decoded_frames must contain at least two frames for cadence"
        )
    started = trial["wall_started_ns"]
    finished = trial["wall_finished_ns"]
    if not _is_int(started) or not _is_int(finished) or started < 0 or finished <= started:
        raise ProtocolError(
            f"{label} wall timestamps must be non-negative increasing integers"
        )
    ready = _require_list(trial["rgb_ready_ns"], f"{label}.rgb_ready_ns")
    if len(ready) != trial["decoded_frames"]:
        raise ProtocolError(
            f"{label}.rgb_ready_ns must contain exactly decoded_frames timestamps"
        )
    if any(not _is_int(timestamp) for timestamp in ready):
        raise ProtocolError(f"{label}.rgb_ready_ns must contain integer timestamps")
    if not ready or ready[0] <= started or ready[-1] > finished:
        raise ProtocolError(f"{label}.rgb_ready_ns is outside the wall interval")
    ready_pairs = list(zip(ready, ready[1:]))
    if any(left >= right for left, right in ready_pairs):
        raise ProtocolError(f"{label}.rgb_ready_ns must be strictly increasing")

    elapsed_s = (finished - started) / 1_000_000_000
    intervals_ms = sorted(
        (right - left) / 1_000_000 for left, right in ready_pairs
    )
    p95_index = max(0, math.ceil(0.95 * len(intervals_ms)) - 1)
    derived = {
        "e2e_fps": trial["decoded_frames"] / elapsed_s,
        "first_visible_rgb_s": (ready[0] - started) / 1_000_000_000,
        "p95_effective_frame_interval_ms": intervals_ms[p95_index],
    }
    for field, value in derived.items():
        if not math.isclose(
            float(trial[field]),
            value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProtocolError(f"{label}.{field} does not match raw timestamps")
    normalized = dict(trial)
    normalized.update(derived)
    return normalized


_DEVELOPMENT_TRIAL_FIELDS = frozenset(
    {
        "trial_id",
        "artifact_id",
        "run_id",
        "system_id",
        "prompt_id",
        "seed",
        "media_sha256",
        "config_sha256",
        "wall_started_ns",
        "wall_finished_ns",
        "rgb_ready_ns",
        "e2e_fps",
        "first_visible_rgb_s",
        "p95_effective_frame_interval_ms",
        "forwards",
        "decoded_frames",
    }
)


def _validate_development_performance(
    protocol: dict,
    evidence: dict,
    round_a_index: dict,
    round_b_index: dict,
) -> dict[str, dict[str, list[dict]]]:
    evidence = _require_mapping(evidence, "development performance evidence")
    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "round_a_manifest_sha256",
        "round_b_manifest_sha256",
        "round_a_trials",
        "round_b_trials",
        "evidence_sha256",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ProtocolError(
            f"development performance evidence missing {missing[0]}"
        )
    extra = sorted(evidence.keys() - required)
    if extra:
        raise ProtocolError(
            f"development performance evidence contains unsupported field {extra[0]}"
        )
    if evidence["schema_version"] != 1 or evidence["status"] != "complete":
        raise ProtocolError(
            "development performance evidence must be schema v1 with status complete"
        )
    _verify_canonical_hash(
        evidence,
        "evidence_sha256",
        "development performance evidence",
    )
    expected_bindings = {
        "protocol_sha256": canonical_sha256(protocol),
        "round_a_manifest_sha256": round_a_index["manifest_sha256"],
        "round_b_manifest_sha256": round_b_index["manifest_sha256"],
    }
    for field, expected in expected_bindings.items():
        if evidence[field] != expected:
            raise ProtocolError(
                f"development performance evidence {field} mismatch"
            )

    trial_ids: set[str] = set()

    def validate_round(
        field: str,
        index: dict,
        expected_per_system: int,
    ) -> dict[str, list[dict]]:
        trials = _require_list(evidence[field], f"development performance {field}")
        by_system: dict[str, list[dict]] = defaultdict(list)
        seen_artifacts: set[str] = set()
        for position, raw_trial in enumerate(trials):
            label = f"development performance {field}[{position}]"
            raw_trial = _require_mapping(raw_trial, label)
            missing_trial = sorted(_DEVELOPMENT_TRIAL_FIELDS - raw_trial.keys())
            if missing_trial:
                raise ProtocolError(f"{label} missing {missing_trial[0]}")
            extra_trial = sorted(raw_trial.keys() - _DEVELOPMENT_TRIAL_FIELDS)
            if extra_trial:
                raise ProtocolError(
                    f"{label} contains unsupported field {extra_trial[0]}"
                )
            trial = _validate_raw_timing_trial(raw_trial, label, trial_ids)
            artifact_id = trial["artifact_id"]
            artifact = index["records_by_artifact"].get(artifact_id)
            if artifact is None:
                raise ProtocolError(f"{label} artifact_id is not in the manifest")
            if artifact_id in seen_artifacts:
                raise ProtocolError(
                    f"development performance {field} requires distinct artifacts"
                )
            seen_artifacts.add(artifact_id)
            run = index["runs"].get(artifact["run_id"])
            if run is None:
                raise ProtocolError(f"{label} run_id is not in the manifest")
            for binding in (
                "run_id",
                "system_id",
                "prompt_id",
                "seed",
                "media_sha256",
                "forwards",
                "decoded_frames",
            ):
                if trial[binding] != artifact[binding]:
                    raise ProtocolError(f"{label} {binding} does not match manifest")
            if trial["config_sha256"] != run["config_sha256"]:
                raise ProtocolError(
                    f"{label} config_sha256 does not match manifest run"
                )
            by_system[trial["system_id"]].append(trial)
        if set(by_system) != set(index["systems"]):
            raise ProtocolError(
                f"development performance {field} does not cover the manifest systems"
            )
        for system_id in index["systems"]:
            if len(by_system[system_id]) != expected_per_system:
                raise ProtocolError(
                    f"development performance {field} requires exactly "
                    f"{expected_per_system} trial(s) for {system_id}"
                )
        return dict(by_system)

    return {
        "round_a": validate_round("round_a_trials", round_a_index, 1),
        "round_b": validate_round("round_b_trials", round_b_index, 3),
    }


def _selection_development_rules(protocol: dict) -> tuple[dict, dict]:
    development = _require_mapping(protocol["development"], "development")
    required = {
        "selection_min_delta",
        "temporal_motion_min_delta",
        "max_single_prompt_drop",
        "minimum_screening_fps",
        "max_finalists",
        "round_b",
    }
    missing = sorted(required - development.keys())
    if missing:
        raise ProtocolError(f"development selection rules missing {missing[0]}")
    for field in (
        "selection_min_delta",
        "temporal_motion_min_delta",
        "max_single_prompt_drop",
        "minimum_screening_fps",
    ):
        value = development[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ProtocolError(f"development.{field} must be finite")
    if not _is_int(development["max_finalists"]) or development["max_finalists"] < 1:
        raise ProtocolError("development.max_finalists must be a positive integer")
    round_b = _require_mapping(development["round_b"], "development.round_b")
    round_b_required = {
        "required_channel_mean",
        "minimum_delta_vs_baseline",
        "minimum_item_score",
        "minimum_each_warm_trial_fps",
        "maximum_first_visible_rgb_s",
        "maximum_p95_effective_frame_interval_ms",
    }
    missing = sorted(round_b_required - round_b.keys())
    if missing:
        raise ProtocolError(f"development.round_b missing {missing[0]}")
    for field in round_b_required:
        value = round_b[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ProtocolError(f"development.round_b.{field} must be finite")
    return development, round_b


def build_selection_report(
    protocol: dict,
    *,
    round_a_manifest: dict,
    round_a_ratings: list[dict],
    round_a_raw_evidence_reports: list[dict],
    round_b_manifest: dict,
    round_b_ratings: list[dict],
    round_b_raw_evidence_reports: list[dict],
    development_performance: dict,
    blind_secret: bytes,
) -> dict:
    """Recompute both development rounds and choose one deterministic finalist."""

    validate_protocol(protocol, require_frozen=True)
    development, round_b_rules = _selection_development_rules(protocol)
    round_a_index = validate_run_manifest(protocol, round_a_manifest)
    round_b_index = validate_run_manifest(protocol, round_b_manifest)
    if round_a_index["phase"] != "development-round-a":
        raise ProtocolError("selection Round-A manifest has the wrong phase")
    if round_b_index["phase"] != "development-round-b":
        raise ProtocolError("selection Round-B manifest has the wrong phase")
    round_a_report = aggregate_ratings(
        protocol,
        round_a_ratings,
        manifest=round_a_manifest,
        blind_secret=blind_secret,
        raw_evidence_reports=round_a_raw_evidence_reports,
        require_complete=True,
    )
    round_b_report = aggregate_ratings(
        protocol,
        round_b_ratings,
        manifest=round_b_manifest,
        blind_secret=blind_secret,
        raw_evidence_reports=round_b_raw_evidence_reports,
        require_complete=True,
    )
    performance = _validate_development_performance(
        protocol,
        development_performance,
        round_a_index,
        round_b_index,
    )

    baseline_id = protocol["baseline"]["system_id"]
    baseline_a = round_a_report["systems"].get(baseline_id)
    baseline_b = round_b_report["systems"].get(baseline_id)
    if baseline_a is None or baseline_b is None:
        raise ProtocolError("development selection evidence is missing the baseline")
    temporal_dimensions = ("motion_naturalness", "temporal_artifacts")
    if any(name not in baseline_a["dimensions"] for name in temporal_dimensions):
        raise ProtocolError(
            "development selection requires motion_naturalness and temporal_artifacts dimensions"
        )
    baseline_a_items = {
        (item["prompt_id"], item["seed"]): item["mean"]
        for item in baseline_a["items"]
    }
    round_a_candidates: dict[str, dict] = {}
    ranked_survivors: list[tuple[tuple, str]] = []
    candidate_ids = [item["system_id"] for item in protocol["repair_candidates"]]
    for system_id in candidate_ids:
        candidate = round_a_report["systems"].get(system_id)
        if candidate is None:
            raise ProtocolError(f"Round-A report is missing candidate {system_id}")
        candidate_items = {
            (item["prompt_id"], item["seed"]): item["mean"]
            for item in candidate["items"]
        }
        if set(candidate_items) != set(baseline_a_items):
            raise ProtocolError(f"Round-A candidate {system_id} has mismatched cases")
        mean_delta = candidate["mean"] - baseline_a["mean"]
        temporal_motion_delta = sum(
            candidate["dimensions"][name] - baseline_a["dimensions"][name]
            for name in temporal_dimensions
        ) / len(temporal_dimensions)
        worst_item_delta = min(
            candidate_items[key] - baseline_a_items[key]
            for key in baseline_a_items
        )
        screening_fps = performance["round_a"][system_id][0]["e2e_fps"]
        criteria = {
            "selection_min_delta": mean_delta >= development["selection_min_delta"],
            "temporal_motion_min_delta": temporal_motion_delta
            >= development["temporal_motion_min_delta"],
            "max_single_prompt_drop": worst_item_delta
            >= -development["max_single_prompt_drop"],
            "minimum_screening_fps": screening_fps
            >= development["minimum_screening_fps"],
        }
        passed = all(criteria.values())
        round_a_candidates[system_id] = {
            "mean": candidate["mean"],
            "mean_delta_vs_baseline": mean_delta,
            "temporal_motion_delta_vs_baseline": temporal_motion_delta,
            "worst_item_delta_vs_baseline": worst_item_delta,
            "screening_fps": screening_fps,
            "criteria": criteria,
            "pass": passed,
        }
        if passed:
            ranked_survivors.append(
                (
                    (
                        -mean_delta,
                        -temporal_motion_delta,
                        -candidate["mean"],
                        system_id,
                    ),
                    system_id,
                )
            )
    ranked_survivors.sort()
    survivors = [
        system_id
        for _rank, system_id in ranked_survivors[: development["max_finalists"]]
    ]
    if not survivors:
        raise ProtocolError("selection has no Round-A survivor")
    expected_round_b_systems = [baseline_id, *survivors]
    if round_b_index["systems"] != expected_round_b_systems:
        raise ProtocolError(
            "Round-B manifest systems do not equal the deterministic Round-A survivors"
        )

    round_b_candidates: dict[str, dict] = {}
    ranked_finalists: list[tuple[tuple, str]] = []
    for system_id in survivors:
        candidate = round_b_report["systems"].get(system_id)
        if candidate is None:
            raise ProtocolError(f"Round-B report is missing survivor {system_id}")
        # A selection "channel" is one independent evaluator family. Rubric
        # dimensions are handled separately by the registered Round-A rule.
        minimum_channel_mean = min(candidate["families"].values())
        mean_delta = candidate["mean"] - baseline_b["mean"]
        minimum_item_score = min(item["mean"] for item in candidate["items"])
        warm_trials = performance["round_b"][system_id]
        minimum_warm_fps = min(trial["e2e_fps"] for trial in warm_trials)
        maximum_first_visible = max(
            trial["first_visible_rgb_s"] for trial in warm_trials
        )
        maximum_p95 = max(
            trial["p95_effective_frame_interval_ms"] for trial in warm_trials
        )
        criteria = {
            "required_channel_mean": minimum_channel_mean
            >= round_b_rules["required_channel_mean"],
            "minimum_delta_vs_baseline": mean_delta
            >= round_b_rules["minimum_delta_vs_baseline"],
            "minimum_item_score": minimum_item_score
            >= round_b_rules["minimum_item_score"],
            "minimum_each_warm_trial_fps": minimum_warm_fps
            >= round_b_rules["minimum_each_warm_trial_fps"],
            "maximum_first_visible_rgb_s": maximum_first_visible
            <= round_b_rules["maximum_first_visible_rgb_s"],
            "maximum_p95_effective_frame_interval_ms": maximum_p95
            <= round_b_rules["maximum_p95_effective_frame_interval_ms"],
        }
        passed = all(criteria.values())
        round_b_candidates[system_id] = {
            "mean": candidate["mean"],
            "mean_delta_vs_baseline": mean_delta,
            "minimum_channel_mean": minimum_channel_mean,
            "minimum_item_score": minimum_item_score,
            "minimum_warm_fps": minimum_warm_fps,
            "maximum_first_visible_rgb_s": maximum_first_visible,
            "maximum_p95_effective_frame_interval_ms": maximum_p95,
            "criteria": criteria,
            "pass": passed,
        }
        if passed:
            ranked_finalists.append(
                (
                    (
                        -mean_delta,
                        -minimum_channel_mean,
                        -candidate["mean"],
                        system_id,
                    ),
                    system_id,
                )
            )
    ranked_finalists.sort()
    if not ranked_finalists:
        raise ProtocolError("selection has no Round-B finalist")
    finalist_system_id = ranked_finalists[0][1]
    finalist = next(
        item
        for item in protocol["repair_candidates"]
        if item["system_id"] == finalist_system_id
    )
    report = {
        "schema_version": 1,
        "status": "selected",
        "protocol_sha256": canonical_sha256(protocol),
        "development_performance_sha256": development_performance[
            "evidence_sha256"
        ],
        "round_a": {
            "manifest_sha256": round_a_index["manifest_sha256"],
            "aggregate_report_sha256": round_a_report["report_sha256"],
            "candidates": round_a_candidates,
            "survivors": survivors,
        },
        "round_b": {
            "manifest_sha256": round_b_index["manifest_sha256"],
            "aggregate_report_sha256": round_b_report["report_sha256"],
            "candidates": round_b_candidates,
        },
        "finalist_system_id": finalist_system_id,
        "finalist_config_sha256": finalist["config_sha256"],
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_selection_report(
    protocol: dict,
    selection_report: dict,
    *,
    round_a_manifest: dict,
    round_a_ratings: list[dict],
    round_a_raw_evidence_reports: list[dict],
    round_b_manifest: dict,
    round_b_ratings: list[dict],
    round_b_raw_evidence_reports: list[dict],
    development_performance: dict,
    blind_secret: bytes,
) -> dict:
    selection_report = _require_mapping(selection_report, "selection report")
    _verify_canonical_hash(
        selection_report,
        "report_sha256",
        "selection report",
    )
    recomputed = build_selection_report(
        protocol,
        round_a_manifest=round_a_manifest,
        round_a_ratings=round_a_ratings,
        round_a_raw_evidence_reports=round_a_raw_evidence_reports,
        round_b_manifest=round_b_manifest,
        round_b_ratings=round_b_ratings,
        round_b_raw_evidence_reports=round_b_raw_evidence_reports,
        development_performance=development_performance,
        blind_secret=blind_secret,
    )
    if selection_report != recomputed:
        raise ProtocolError("selection report does not match recomputed evidence")
    return recomputed


def build_selection_lock(
    protocol: dict,
    selection_report: dict,
    *,
    locked_at: str,
) -> dict:
    validate_protocol(protocol, require_frozen=True)
    selection_report = _require_mapping(selection_report, "selection report")
    _verify_canonical_hash(
        selection_report,
        "report_sha256",
        "selection report",
    )
    if selection_report.get("protocol_sha256") != canonical_sha256(protocol):
        raise ProtocolError("selection report protocol_sha256 mismatch")
    if not isinstance(locked_at, str) or not locked_at:
        raise ProtocolError("selection lock locked_at is required")
    selection_lock = {
        "schema_version": 1,
        "status": "locked",
        "protocol_sha256": canonical_sha256(protocol),
        "finalist_system_id": selection_report["finalist_system_id"],
        "finalist_config_sha256": selection_report["finalist_config_sha256"],
        "selection_report_sha256": selection_report["report_sha256"],
        "locked_at": locked_at,
    }
    validate_selection_lock(
        protocol,
        selection_lock,
        selection_report=selection_report,
    )
    return selection_lock


def _validate_performance_evidence(
    protocol: dict,
    performance: dict,
    selected_system: str,
    selection_lock: dict,
    confirmatory_manifest_index: dict,
    confirmatory_media_report: dict,
    sentinel_manifest_index: dict,
    sentinel_media_report: dict,
) -> dict:
    performance = _require_mapping(performance, "performance evidence")
    if performance.get("schema_version") != 1 or performance.get("status") != "complete":
        raise ProtocolError("performance evidence must be schema v1 with status complete")
    _verify_canonical_hash(
        performance,
        "evidence_sha256",
        "performance evidence",
    )
    if performance.get("protocol_sha256") != canonical_sha256(protocol):
        raise ProtocolError("performance evidence protocol_sha256 mismatch")
    if performance.get("selected_system") != selected_system:
        raise ProtocolError("performance evidence selected_system mismatch")
    expected_bindings = {
        "selection_lock_sha256": canonical_sha256(selection_lock),
        "confirmatory_manifest_sha256": confirmatory_manifest_index[
            "manifest_sha256"
        ],
        "confirmatory_media_report_sha256": confirmatory_media_report[
            "report_sha256"
        ],
        "sentinel_manifest_sha256": sentinel_manifest_index["manifest_sha256"],
        "sentinel_media_report_sha256": sentinel_media_report["report_sha256"],
    }
    for field, expected in expected_bindings.items():
        if performance.get(field) != expected:
            raise ProtocolError(f"performance evidence {field} mismatch")

    selected_run = next(
        run
        for run in confirmatory_manifest_index["runs"].values()
        if run["system_id"] == selected_system
    )
    sentinel_run = next(iter(sentinel_manifest_index["runs"].values()))
    if performance.get("selected_run_sha256") != canonical_sha256(selected_run):
        raise ProtocolError("performance evidence selected_run_sha256 mismatch")
    if performance.get("sentinel_run_sha256") != canonical_sha256(sentinel_run):
        raise ProtocolError("performance evidence sentinel_run_sha256 mismatch")

    trial_ids: set[str] = set()

    def validate_trial(trial: dict, label: str) -> dict:
        return _validate_raw_timing_trial(trial, label, trial_ids)

    cold = validate_trial(
        performance.get("cold_trial"),
        "performance cold_trial",
    )
    warm_trials = _require_list(performance.get("warm_trials"), "performance warm_trials")
    if len(warm_trials) != 3:
        raise ProtocolError("performance evidence requires exactly three warm trials")
    warm = [
        validate_trial(
            trial,
            f"performance warm_trial[{index}]",
        )
        for index, trial in enumerate(warm_trials)
    ]
    short_trials = [cold, *warm]
    selected_records = {
        record["artifact_id"]: record
        for record in confirmatory_manifest_index["records"]
        if record["system_id"] == selected_system
    }
    confirmatory_media_by_artifact = {
        row["artifact_id"]: row for row in confirmatory_media_report["records"]
    }
    short_artifact_ids: set[str] = set()
    for index, trial in enumerate(short_trials):
        label = "performance cold_trial" if index == 0 else f"performance warm_trial[{index - 1}]"
        for field in ("artifact_id", "run_id", "prompt_id", "seed"):
            if field not in trial:
                raise ProtocolError(f"{label} missing {field}")
        artifact_id = trial["artifact_id"]
        expected_record = selected_records.get(artifact_id)
        if expected_record is None:
            raise ProtocolError(
                f"{label} artifact_id is not a selected-system confirmatory artifact"
            )
        if artifact_id in short_artifact_ids:
            raise ProtocolError("performance short trials require four distinct artifacts")
        short_artifact_ids.add(artifact_id)
        for field in ("run_id", "prompt_id", "seed"):
            if trial[field] != expected_record[field]:
                raise ProtocolError(f"{label} {field} does not match manifest")
        for field in ("media_sha256", "forwards", "decoded_frames"):
            if trial[field] != expected_record[field]:
                raise ProtocolError(f"{label} {field} does not match manifest")
        if (
            confirmatory_media_by_artifact.get(artifact_id, {}).get("media_sha256")
            != trial["media_sha256"]
        ):
            raise ProtocolError(f"{label} media_sha256 does not match media report")
    if len(short_artifact_ids) != 4:
        raise ProtocolError("performance short trials require four distinct artifacts")

    sentinel_trials = _require_list(
        performance.get("sentinel_trials"), "performance sentinel_trials"
    )
    expected_sentinel_records = sentinel_manifest_index["records"]
    if len(sentinel_trials) != len(expected_sentinel_records):
        raise ProtocolError("performance sentinel trials do not cover sentinel manifest")
    media_by_artifact = {
        row["artifact_id"]: row for row in sentinel_media_report["records"]
    }
    sentinels = []
    for index, (trial, expected_record) in enumerate(
        zip(sentinel_trials, expected_sentinel_records)
    ):
        trial = validate_trial(
            trial,
            f"performance sentinel_trial[{index}]",
        )
        for field in ("artifact_id", "prompt_id", "seed", "run_id"):
            if trial.get(field) != expected_record[field]:
                raise ProtocolError(
                    f"performance sentinel trial {field} does not match manifest"
                )
        if trial["media_sha256"] != expected_record["media_sha256"]:
            raise ProtocolError(
                "performance sentinel trial media_sha256 does not match manifest"
            )
        if media_by_artifact[trial["artifact_id"]]["media_sha256"] != trial["media_sha256"]:
            raise ProtocolError(
                "performance sentinel trial media_sha256 does not match media report"
            )
        media_duration = trial.get("media_duration_s")
        if (
            not isinstance(media_duration, (int, float))
            or isinstance(media_duration, bool)
            or not math.isfinite(media_duration)
            or media_duration <= 0
        ):
            raise ProtocolError("sentinel media_duration_s must be positive and finite")
        contract = protocol["long_horizon_media_contract"]
        expected_duration = contract["decoded_frames"] / contract["fps"]
        tolerance = contract["duration_tolerance_frames"] / contract["fps"]
        if abs(media_duration - expected_duration) > tolerance:
            raise ProtocolError("sentinel media_duration_s does not match media contract")
        audited_duration = media_by_artifact[trial["artifact_id"]]["duration_s"]
        if not math.isclose(media_duration, audited_duration, abs_tol=1e-9):
            raise ProtocolError(
                "sentinel media_duration_s does not match media report duration"
            )
        normalized_trial = dict(trial)
        normalized_trial["media_duration_s"] = audited_duration
        sentinels.append(normalized_trial)
    return {"cold": cold, "warm": warm, "sentinels": sentinels}


def evaluate_gate(
    protocol: dict,
    report: dict,
    selected_system: str,
    reference_system: str,
    performance: dict | None = None,
    media_report: dict | None = None,
    *,
    manifest: dict | None = None,
    selection_lock: dict | None = None,
    sentinel_manifest: dict | None = None,
    sentinel_media_report: dict | None = None,
    ratings: list[dict] | None = None,
    raw_evidence_reports: list[dict] | None = None,
    blind_secret: bytes | None = None,
    media_records: Iterable[dict] | None = None,
    sentinel_media_records: Iterable[dict] | None = None,
    media_probe: Callable[[Path], dict] | None = None,
    selection_report: dict | None = None,
    round_a_manifest: dict | None = None,
    round_a_ratings: list[dict] | None = None,
    round_a_raw_evidence_reports: list[dict] | None = None,
    round_b_manifest: dict | None = None,
    round_b_ratings: list[dict] | None = None,
    round_b_raw_evidence_reports: list[dict] | None = None,
    development_performance: dict | None = None,
) -> dict:
    validate_protocol(protocol, require_frozen=True)
    if selection_lock is None:
        raise ProtocolError("a locked finalist selection is required")
    if selection_report is None:
        raise ProtocolError("the recomputable selection report is required")
    selection_inputs = {
        "round_a_manifest": round_a_manifest,
        "round_a_ratings": round_a_ratings,
        "round_a_raw_evidence_reports": round_a_raw_evidence_reports,
        "round_b_manifest": round_b_manifest,
        "round_b_ratings": round_b_ratings,
        "round_b_raw_evidence_reports": round_b_raw_evidence_reports,
        "development_performance": development_performance,
    }
    missing_selection = [
        name for name, value in selection_inputs.items() if value is None
    ]
    if missing_selection:
        raise ProtocolError(
            f"selection report recomputation requires {missing_selection[0]}"
        )
    validated_selection_report = validate_selection_report(
        protocol,
        selection_report,
        round_a_manifest=round_a_manifest,
        round_a_ratings=round_a_ratings,
        round_a_raw_evidence_reports=round_a_raw_evidence_reports,
        round_b_manifest=round_b_manifest,
        round_b_ratings=round_b_ratings,
        round_b_raw_evidence_reports=round_b_raw_evidence_reports,
        development_performance=development_performance,
        blind_secret=blind_secret,
    )
    validate_selection_lock(
        protocol,
        selection_lock,
        selection_report=validated_selection_report,
    )
    if selected_system != selection_lock["finalist_system_id"]:
        raise ProtocolError("selected_system must equal the locked finalist")
    if reference_system != "sf4-reference":
        raise ProtocolError("reference_system must equal sf4-reference")
    if manifest is None:
        raise ProtocolError("a confirmatory manifest is required")
    if _require_mapping(manifest.get("scope"), "manifest.scope").get("phase") != "confirmatory":
        raise ProtocolError("headline gate requires a confirmatory manifest")
    manifest_index = validate_run_manifest(
        protocol,
        manifest,
        selection_lock=selection_lock,
    )
    if manifest_index["phase"] != "confirmatory":
        raise ProtocolError("headline gate requires a confirmatory manifest")
    if sentinel_manifest is None:
        raise ProtocolError("a sentinel manifest is required")
    if (
        _require_mapping(sentinel_manifest.get("scope"), "sentinel manifest.scope").get("phase")
        != "sentinel"
    ):
        raise ProtocolError("long-horizon evidence requires a sentinel manifest")
    sentinel_manifest_index = validate_run_manifest(
        protocol,
        sentinel_manifest,
        selection_lock=selection_lock,
    )
    if sentinel_manifest_index["phase"] != "sentinel":
        raise ProtocolError("long-horizon evidence requires a sentinel manifest")
    gates = protocol["gates"]
    report = _require_mapping(report, "aggregate report")
    _verify_canonical_hash(report, "report_sha256", "aggregate report")
    if ratings is None:
        raise ProtocolError("normalized ratings are required to recompute the aggregate")
    if not isinstance(blind_secret, bytes) or len(blind_secret) != 32:
        raise ProtocolError("the 32-byte blinding secret is required at gate time")
    recomputed_report = aggregate_ratings(
        protocol,
        ratings,
        manifest=manifest,
        selection_lock=selection_lock,
        blind_secret=blind_secret,
        raw_evidence_reports=raw_evidence_reports,
        require_complete=True,
    )
    if report != recomputed_report:
        raise ProtocolError("aggregate report does not match recomputed ratings")
    report = recomputed_report
    if report.get("protocol_sha256") != canonical_sha256(protocol):
        raise ProtocolError("aggregate report protocol_sha256 mismatch")
    if report.get("manifest_sha256") != manifest_index["manifest_sha256"]:
        raise ProtocolError("aggregate report manifest_sha256 mismatch")
    if report.get("manifest_phase") != "confirmatory":
        raise ProtocolError("aggregate report is not confirmatory")
    if report.get("selection_lock_sha256") != canonical_sha256(selection_lock):
        raise ProtocolError("aggregate report selection_lock_sha256 mismatch")
    evidence = _require_list(report.get("rating_evidence"), "rating_evidence")
    if report.get("rating_evidence_sha256") != canonical_sha256(evidence):
        raise ProtocolError("aggregate report rating_evidence_sha256 mismatch")
    if not report.get("complete"):
        raise ProtocolError("aggregate report is not a complete rating tensor")
    selected = report.get("systems", {}).get(selected_system)
    if not selected:
        raise ProtocolError(f"aggregate report has no selected system {selected_system}")
    if reference_system not in report.get("systems", {}):
        raise ProtocolError(f"aggregate report has no reference system {reference_system}")
    if media_report is None:
        raise ProtocolError("a bound media report is required")
    submitted_media_report = _validate_bound_media_report(
        protocol,
        manifest_index,
        media_report,
    )
    if media_records is None:
        raise ProtocolError("confirmatory physical media records are required at gate time")
    fresh_media_report = audit_media(
        media_records,
        protocol["media_contract"],
        media_probe,
        protocol_sha256=manifest_index["protocol_sha256"],
        manifest_sha256=manifest_index["manifest_sha256"],
    )
    media_report = _validate_bound_media_report(
        protocol,
        manifest_index,
        fresh_media_report,
    )
    if submitted_media_report != media_report:
        raise ProtocolError("submitted media report does not match gate-time media audit")
    if sentinel_media_report is None:
        raise ProtocolError("a bound sentinel media report is required")
    submitted_sentinel_media_report = _validate_bound_media_report(
        protocol,
        sentinel_manifest_index,
        sentinel_media_report,
    )
    if sentinel_media_records is None:
        raise ProtocolError("sentinel physical media records are required at gate time")
    fresh_sentinel_media_report = audit_media(
        sentinel_media_records,
        protocol["long_horizon_media_contract"],
        media_probe,
        protocol_sha256=sentinel_manifest_index["protocol_sha256"],
        manifest_sha256=sentinel_manifest_index["manifest_sha256"],
    )
    sentinel_media_report = _validate_bound_media_report(
        protocol,
        sentinel_manifest_index,
        fresh_sentinel_media_report,
    )
    if submitted_sentinel_media_report != sentinel_media_report:
        raise ProtocolError(
            "submitted sentinel media report does not match gate-time media audit"
        )
    if performance is None:
        raise ProtocolError("performance evidence is required")
    perf = _validate_performance_evidence(
        protocol,
        performance,
        selected_system,
        selection_lock,
        manifest_index,
        media_report,
        sentinel_manifest_index,
        sentinel_media_report,
    )

    family_defs = {
        item["id"]: item for item in protocol["evaluation"]["families"]
    }
    present = set(selected["families"])
    model_count = sum(
        1
        for family_id in present
        if family_defs.get(family_id, {}).get("kind") == "model"
    )
    human_present = any(
        family_defs.get(family_id, {}).get("kind") == "human"
        for family_id in present
    )
    delta = report.get("paired_deltas", {}).get(
        f"{selected_system}-vs-{reference_system}"
    )
    warm_fps = [trial["e2e_fps"] for trial in perf["warm"]]
    first_rgb = [trial["first_visible_rgb_s"] for trial in perf["warm"]]
    p95 = [
        trial["p95_effective_frame_interval_ms"]
        for trial in [*perf["warm"], *perf["sentinels"]]
    ]
    all_short = [perf["cold"], *perf["warm"]]
    all_trials = [*all_short, *perf["sentinels"]]
    expected_sentinel_forwards = causal_forward_count(241, 1, 4, 1)

    checks: dict[str, dict] = {}

    def add(name: str, actual: Any, operator: str, threshold: Any, passed: bool) -> None:
        checks[name] = {
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "pass": bool(passed),
        }

    add(
        "absolute_quality",
        selected["mean"],
        ">=",
        gates["absolute_quality"],
        selected["mean"] >= gates["absolute_quality"],
    )
    family_min = min(selected["families"].values()) if selected["families"] else None
    add(
        "family_floor",
        family_min,
        ">=",
        gates["family_floor"],
        family_min is not None and family_min >= gates["family_floor"],
    )
    stratum_min = min(selected["strata"].values()) if selected["strata"] else None
    add(
        "stratum_floor",
        stratum_min,
        ">=",
        gates["stratum_floor"],
        stratum_min is not None and stratum_min >= gates["stratum_floor"],
    )
    add(
        "sf4_noninferiority_margin",
        delta,
        ">=",
        gates["sf4_noninferiority_margin"],
        delta is not None and delta >= gates["sf4_noninferiority_margin"],
    )
    add(
        "maximum_low_quality_items",
        selected["low_quality_items"],
        "<=",
        gates["maximum_low_quality_items"],
        selected["low_quality_items"] <= gates["maximum_low_quality_items"],
    )
    add(
        "low_quality_item_threshold",
        selected["low_quality_item_threshold"],
        "==",
        gates["low_quality_item_threshold"],
        math.isclose(
            selected["low_quality_item_threshold"],
            gates["low_quality_item_threshold"],
            abs_tol=1e-12,
        ),
    )
    add(
        "final_third_temporal_floor",
        selected["final_third_mean"],
        ">=",
        gates["final_third_temporal_floor"],
        selected["final_third_mean"] >= gates["final_third_temporal_floor"],
    )
    add(
        "maximum_early_to_late_drop",
        selected["early_to_late_drop"],
        "<=",
        gates["maximum_early_to_late_drop"],
        selected["early_to_late_drop"] <= gates["maximum_early_to_late_drop"],
    )
    add(
        "warm_e2e_fps",
        sum(warm_fps) / len(warm_fps),
        ">=",
        gates["warm_e2e_fps"],
        sum(warm_fps) / len(warm_fps) >= gates["warm_e2e_fps"],
    )
    add(
        "cold_e2e_fps",
        perf["cold"]["e2e_fps"],
        ">=",
        gates["cold_e2e_fps"],
        perf["cold"]["e2e_fps"] >= gates["cold_e2e_fps"],
    )
    min_sustained = min(trial["media_duration_s"] for trial in perf["sentinels"])
    add(
        "sustained_seconds",
        min_sustained,
        ">=",
        gates["sustained_seconds"],
        min_sustained >= gates["sustained_seconds"],
    )
    min_sustained_fps = min(trial["e2e_fps"] for trial in perf["sentinels"])
    add(
        "sustained_e2e_fps",
        min_sustained_fps,
        ">=",
        gates["sustained_e2e_fps"],
        min_sustained_fps >= gates["sustained_e2e_fps"],
    )
    add(
        "minimum_each_warm_trial_fps",
        min(warm_fps),
        ">=",
        gates["minimum_each_warm_trial_fps"],
        min(warm_fps) >= gates["minimum_each_warm_trial_fps"],
    )
    add(
        "maximum_first_visible_rgb_s",
        max(first_rgb),
        "<=",
        gates["maximum_first_visible_rgb_s"],
        max(first_rgb) <= gates["maximum_first_visible_rgb_s"],
    )
    add(
        "maximum_p95_effective_frame_interval_ms",
        max(p95),
        "<=",
        gates["maximum_p95_effective_frame_interval_ms"],
        max(p95) <= gates["maximum_p95_effective_frame_interval_ms"],
    )
    add(
        "required_model_families",
        model_count,
        ">=",
        gates["required_model_families"],
        model_count >= gates["required_model_families"],
    )
    add(
        "require_human",
        human_present,
        "==",
        gates["require_human"],
        human_present if gates["require_human"] else True,
    )
    short_forwards_ok = all(
        trial["forwards"] == gates["required_forwards"] for trial in all_short
    )
    sentinel_forwards_ok = all(
        trial["forwards"] == expected_sentinel_forwards
        for trial in perf["sentinels"]
    )
    add(
        "required_forwards",
        [trial["forwards"] for trial in all_trials],
        "== short/long",
        {"short": gates["required_forwards"], "long": expected_sentinel_forwards},
        short_forwards_ok and sentinel_forwards_ok,
    )
    short_frames_ok = all(
        trial["decoded_frames"] == gates["required_rgb_frames"]
        for trial in all_short
    )
    sentinel_frames_ok = all(
        trial["decoded_frames"] == 961 for trial in perf["sentinels"]
    )
    add(
        "required_rgb_frames",
        [trial["decoded_frames"] for trial in all_trials],
        "== short/long",
        {"short": gates["required_rgb_frames"], "long": 961},
        short_frames_ok and sentinel_frames_ok,
    )
    if set(checks) != set(gates):
        raise ProtocolError("gate registry does not consume every protocol gate exactly once")
    failed = [name for name, check in checks.items() if not check["pass"]]
    result = {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": report["manifest_sha256"],
        "aggregate_report_sha256": report["report_sha256"],
        "media_report_sha256": media_report.get("report_sha256"),
        "sentinel_manifest_sha256": sentinel_manifest_index["manifest_sha256"],
        "sentinel_media_report_sha256": sentinel_media_report["report_sha256"],
        "selection_lock_sha256": canonical_sha256(selection_lock),
        "selection_report_sha256": validated_selection_report["report_sha256"],
        "performance_evidence_sha256": performance["evidence_sha256"],
        "pass": not failed,
        "selected_system": selected_system,
        "reference_system": reference_system,
        "failed": failed,
        "checks": checks,
        "quality_mean": selected["mean"],
        "paired_delta": delta,
    }
    result["report_sha256"] = canonical_sha256(result)
    return result
