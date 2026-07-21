"""Batch comparison wrapper for displacement and optional coherence reports.

Each scorer remains the single source of metric behavior.  This module only
discovers clips, reuses backend instances, projects comparison fields, and
writes deterministic JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.coherence_metrics import (
    CoherenceConfig,
    CoherenceMetricError,
    FlowBackend,
    OpenCVFarnebackFlowBackend,
    RaftSmallFlowBackend,
    score_clip as score_coherence_clip,
)
from bench.displacement_metrics import (
    CoTracker3Backend,
    DINOv2AppearanceBackend,
    DisplacementMetricError,
    HistogramAppearanceBackend,
    MetricConfig,
    OpenCVLKTracker,
    score_clip,
)


class BatchComparisonError(ValueError):
    """The batch layout or a scorer report violated the batch contract."""


ScoreClip = Callable[..., dict[str, Any]]
CoherenceScoreClip = Callable[..., dict[str, Any]]
InvalidConditionSpec = Mapping[str, str]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _canonical_batch_json(value: Any, *, level: int = 0) -> str:
    indentation = "  " * level
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BatchComparisonError("batch report contains a non-finite float")
        return f"{value:.6f}"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        child_indent = "  " * (level + 1)
        children = [
            child_indent + _canonical_batch_json(item, level=level + 1)
            for item in value
        ]
        return "[\n" + ",\n".join(children) + "\n" + indentation + "]"
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        if any(not isinstance(key, str) for key in value):
            raise BatchComparisonError("batch report JSON keys must be strings")
        child_indent = "  " * (level + 1)
        children = [
            child_indent
            + json.dumps(key)
            + ": "
            + _canonical_batch_json(value[key], level=level + 1)
            for key in sorted(value)
        ]
        return "{\n" + ",\n".join(children) + "\n" + indentation + "}"
    raise BatchComparisonError(
        f"batch report contains unsupported JSON type {type(value).__name__}"
    )


def _six_places(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchComparisonError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise BatchComparisonError(f"{label} must be finite")
    rounded = round(numeric, 6)
    return 0.0 if rounded == 0.0 else rounded


def _read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise BatchComparisonError("prompts file must contain at least one prompt name")
    prompts: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        prompt = line.strip()
        if not prompt:
            raise BatchComparisonError(
                f"prompts file contains a blank name on line {line_number}"
            )
        prompts.append(prompt)
    if len(set(prompts)) != len(prompts):
        raise BatchComparisonError("prompt names must be unique")
    return prompts


def _discover_conditions(
    root: Path,
    *,
    output_dir: Path,
) -> tuple[list[str], dict[str, list[Path]]]:
    if not root.is_dir():
        raise BatchComparisonError(f"batch root is not a directory: {root}")
    conditions: list[Path] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        if candidate.resolve() == output_dir:
            continue
        conditions.append(candidate)
    conditions.sort(key=lambda path: path.name)
    if not conditions:
        raise BatchComparisonError("batch root contains no condition directories")

    clips_by_condition: dict[str, list[Path]] = {}
    for condition in conditions:
        direct_clips = sorted(
            (
                path
                for path in condition.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            ),
            key=lambda path: path.name,
        )
        nested_clips = sorted(
            path
            for path in condition.rglob("*")
            if path.is_file()
            and path.parent != condition
            and path.suffix.lower() == ".mp4"
        )
        if nested_clips:
            first = nested_clips[0].relative_to(root).as_posix()
            raise BatchComparisonError(
                "clips must use the direct <condition>/<clip>.mp4 layout; "
                f"found nested clip {first}"
            )
        clips_by_condition[condition.name] = direct_clips
    if not any(clips_by_condition.values()):
        raise BatchComparisonError("condition directories contain no MP4 clips")
    return [path.name for path in conditions], clips_by_condition


def _slot_assignments(
    conditions: list[str],
    clips_by_condition: Mapping[str, list[Path]],
    prompts: list[str],
    prompt_substrings: Mapping[str, str] | None = None,
) -> tuple[str, list[str | None], dict[str, list[Path | None]]]:
    largest_condition = max(len(clips_by_condition[name]) for name in conditions)
    if largest_condition > len(prompts):
        raise BatchComparisonError(
            "prompt count is smaller than at least one condition's clip count "
            f"({len(prompts)} prompts, {largest_condition} clips)"
        )

    if prompt_substrings is not None:
        if set(prompt_substrings) != set(prompts):
            raise BatchComparisonError(
                "prompt substring names must exactly match the prompts file"
            )
        normalized_substrings: dict[str, str] = {}
        for prompt in prompts:
            substring = prompt_substrings[prompt]
            if not isinstance(substring, str) or not substring.strip():
                raise BatchComparisonError(
                    f"prompt substring for {prompt!r} must be a non-empty string"
                )
            normalized_substrings[prompt] = substring.casefold()
        if len(set(normalized_substrings.values())) != len(normalized_substrings):
            raise BatchComparisonError("prompt substrings must be unique")

        assignments: dict[str, list[Path | None]] = {}
        for condition in conditions:
            by_prompt: dict[str, Path] = {}
            for path in clips_by_condition[condition]:
                folded_name = path.name.casefold()
                matches = [
                    prompt
                    for prompt in prompts
                    if normalized_substrings[prompt] in folded_name
                ]
                if len(matches) != 1:
                    raise BatchComparisonError(
                        f"clip {condition}/{path.name} matched {len(matches)} prompt substrings"
                    )
                prompt = matches[0]
                if prompt in by_prompt:
                    raise BatchComparisonError(
                        f"condition {condition} has multiple clips for prompt {prompt}"
                    )
                by_prompt[prompt] = path
            assignments[condition] = [by_prompt.get(prompt) for prompt in prompts]
        slot_filenames = []
        for index in range(len(prompts)):
            names = {
                paths[index].name
                for paths in assignments.values()
                if paths[index] is not None
            }
            slot_filenames.append(next(iter(names)) if len(names) == 1 else None)
        return "substring", slot_filenames, assignments

    filenames = sorted(
        {
            path.name
            for condition in conditions
            for path in clips_by_condition[condition]
        }
    )
    has_complete_filename_template = any(
        len(clips_by_condition[condition]) == len(prompts)
        for condition in conditions
    )
    if len(filenames) == len(prompts) and has_complete_filename_template:
        slot_mode = "shared-filename"
        slot_filenames: list[str | None] = list(filenames)
        assignments = {
            condition: [
                {path.name: path for path in clips_by_condition[condition]}.get(filename)
                for filename in filenames
            ]
            for condition in conditions
        }
    else:
        slot_mode = "positional"
        slot_filenames = [None] * len(prompts)
        assignments = {
            condition: [
                clips_by_condition[condition][index]
                if index < len(clips_by_condition[condition])
                else None
                for index in range(len(prompts))
            ]
            for condition in conditions
        }
    return slot_mode, slot_filenames, assignments


def _backends(config: MetricConfig) -> tuple[Any, Any]:
    track_backend = (
        OpenCVLKTracker()
        if config.tracker == "opencv-lk"
        else CoTracker3Backend(config.device)
    )
    appearance_backend = (
        HistogramAppearanceBackend()
        if config.appearance == "histogram"
        else DINOv2AppearanceBackend(config.device)
    )
    return track_backend, appearance_backend


def _project_report(report: Mapping[str, Any]) -> dict[str, Any]:
    foreground = report.get("foreground")
    if not isinstance(foreground, Mapping):
        raise BatchComparisonError("scorer report is missing foreground metrics")
    displaced = report.get("displaced_vs_collapsed")
    if not isinstance(displaced, bool):
        raise BatchComparisonError(
            "scorer report displaced_vs_collapsed must be a boolean"
        )
    has_degraded = "coherence_degraded" in report
    has_compensated = "camera_compensated" in report
    if has_degraded != has_compensated:
        raise BatchComparisonError(
            "scorer report degraded markers must be present together"
        )
    if has_degraded:
        coherence_degraded = report["coherence_degraded"]
        camera_compensated = report["camera_compensated"]
        if not isinstance(coherence_degraded, bool) or not isinstance(
            camera_compensated,
            bool,
        ):
            raise BatchComparisonError("scorer report degraded markers must be boolean")
        if coherence_degraded == camera_compensated:
            raise BatchComparisonError(
                "scorer report degraded markers must describe one measurement mode"
            )
    else:
        coherence_degraded = False
        camera_compensated = True
    return {
        "camera_compensated": camera_compensated,
        "coherence_degraded": coherence_degraded,
        "displacement_score": _six_places(
            report.get("displacement_score"),
            label="displacement_score",
        ),
        "span_fraction": _six_places(
            foreground.get("trajectory_span_fraction_width"),
            label="trajectory_span_fraction_width",
        ),
        "survival": _six_places(
            foreground.get("track_survival"),
            label="track_survival",
        ),
        "displaced": displaced,
    }


def _project_coherence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    degrades_over_time = report.get("degrades_over_time")
    if not isinstance(degrades_over_time, bool):
        raise BatchComparisonError(
            "coherence report degrades_over_time must be a boolean"
        )
    return {
        "coherence_score": _six_places(
            report.get("coherence_score"),
            label="coherence_score",
        ),
        "degrades_over_time": degrades_over_time,
        "spatial_integrity_score": _six_places(
            report.get("spatial_integrity_score"),
            label="spatial_integrity_score",
        ),
        "temporal_coherence_score": _six_places(
            report.get("temporal_coherence_score"),
            label="temporal_coherence_score",
        ),
    }


def _coherence_error_projection(error: CoherenceMetricError) -> dict[str, Any]:
    return {
        "coherence_error_message": str(error),
        "coherence_error_type": type(error).__name__,
        "coherence_result_json": None,
        "coherence_score": None,
        "coherence_status": "error",
        "degrades_over_time": None,
        "spatial_integrity_score": None,
        "temporal_coherence_score": None,
    }


def _missing_cell(
    condition: str,
    prompt: str,
    filename: str | None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "displaced": None,
        "displacement_score": None,
        "filename": filename,
        "prompt": prompt,
        "result_json": None,
        "span_fraction": None,
        "status": "missing",
        "survival": None,
    }


def _invalid_cell(
    condition: str,
    prompt: str,
    filename: str | None,
    *,
    artifact_status: str,
    spec: InvalidConditionSpec,
) -> dict[str, Any]:
    return {
        "artifact_status": artifact_status,
        "condition": condition,
        "displaced": None,
        "displacement_score": None,
        "filename": filename,
        "prompt": prompt,
        "reason": spec["reason"],
        "reason_code": spec["reason_code"],
        "remediation": spec["remediation"],
        "result_json": None,
        "span_fraction": None,
        "status": "invalid",
        "survival": None,
    }


def _error_cell(
    condition: str,
    prompt: str,
    filename: str,
    error: DisplacementMetricError,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "displaced": None,
        "displacement_score": None,
        "error_message": str(error),
        "error_type": type(error).__name__,
        "filename": filename,
        "prompt": prompt,
        "result_json": None,
        "span_fraction": None,
        "status": "error",
        "survival": None,
    }


def _normalize_invalid_conditions(
    invalid_conditions: Mapping[str, InvalidConditionSpec] | None,
    conditions: Sequence[str],
) -> dict[str, dict[str, str]]:
    if invalid_conditions is None:
        return {}
    normalized: dict[str, dict[str, str]] = {}
    required = ("reason_code", "reason", "remediation")
    for condition in sorted(invalid_conditions):
        if condition not in conditions:
            raise BatchComparisonError(f"invalid condition not found: {condition}")
        spec = invalid_conditions[condition]
        if not isinstance(spec, Mapping):
            raise BatchComparisonError(
                f"invalid condition {condition} must have a reason object"
            )
        values: dict[str, str] = {}
        for key in required:
            value = spec.get(key)
            if not isinstance(value, str) or not value.strip():
                raise BatchComparisonError(
                    f"invalid condition {condition} requires non-empty {key}"
                )
            values[key] = value
        normalized[condition] = values
    return normalized


def _normalize_comparisons(
    compare: tuple[str, str] | None,
    comparisons: Sequence[tuple[str, str]] | None,
    conditions: Sequence[str],
    invalid_conditions: Mapping[str, InvalidConditionSpec],
) -> list[tuple[str, str]]:
    if compare is not None and comparisons is not None:
        raise BatchComparisonError("use compare or comparisons, not both")
    pairs = [compare] if compare is not None else list(comparisons or ())
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise BatchComparisonError("each comparison must contain exactly two conditions")
        condition_a, condition_b = pair
        if condition_a == condition_b:
            raise BatchComparisonError("comparison conditions must be different")
        missing_conditions = [
            name for name in (condition_a, condition_b) if name not in conditions
        ]
        if missing_conditions:
            raise BatchComparisonError(
                "comparison condition not found: " + ", ".join(missing_conditions)
            )
        invalid = [
            name for name in (condition_a, condition_b) if name in invalid_conditions
        ]
        if invalid:
            raise BatchComparisonError(
                "invalid conditions cannot be compared: " + ", ".join(invalid)
            )
        normalized.append((condition_a, condition_b))
    if len(set(normalized)) != len(normalized):
        raise BatchComparisonError("comparison pairs must be unique")
    return normalized


def _comparison(
    matrix: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    *,
    include_degraded: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    for matrix_row in matrix:
        prompt = matrix_row["prompt"]
        cell_a = matrix_row["cells"][condition_a]
        cell_b = matrix_row["cells"][condition_b]
        degraded_conditions = [
            condition
            for condition, cell in (
                (condition_a, cell_a),
                (condition_b, cell_b),
            )
            if cell.get("coherence_degraded") is True
        ]

        def comparable(cell: Mapping[str, Any]) -> bool:
            return cell["status"] == "scored" and (
                include_degraded or cell.get("coherence_degraded") is not True
            )

        if not comparable(cell_a) or not comparable(cell_b):
            unavailable = [
                condition
                for condition, cell in (
                    (condition_a, cell_a),
                    (condition_b, cell_b),
                )
                if not comparable(cell)
            ]
            statuses = {cell_a["status"], cell_b["status"]}
            if "invalid" in statuses:
                status = "invalid"
            elif "error" in statuses:
                status = "error"
            elif "missing" in statuses:
                status = "missing"
            else:
                status = "degraded"
            rows.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "degraded_conditions": degraded_conditions,
                    "delta_b_minus_a": None,
                    "displaced_transition": None,
                    "prompt": prompt,
                    "score_a": cell_a["displacement_score"],
                    "score_b": cell_b["displacement_score"],
                    "span_fraction_delta_b_minus_a": None,
                    "status": status,
                    "survival_delta_b_minus_a": None,
                    "verdict": f"{status}: {', '.join(unavailable)}",
                }
            )
            continue

        score_delta = _six_places(
            cell_b["displacement_score"] - cell_a["displacement_score"],
            label="score delta",
        )
        score_deltas.append(score_delta)
        if score_delta > 0.0:
            verdict = f"{condition_b} higher by {score_delta:.6f}"
        elif score_delta < 0.0:
            verdict = f"{condition_a} higher by {abs(score_delta):.6f}"
        else:
            verdict = "tie at 0.000000"
        if degraded_conditions:
            verdict = "DEGRADED — " + verdict
        rows.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "degraded_conditions": degraded_conditions,
                "delta_b_minus_a": score_delta,
                "displaced_transition": (
                    f"{str(cell_a['displaced']).lower()}"
                    f"->{str(cell_b['displaced']).lower()}"
                ),
                "prompt": prompt,
                "score_a": cell_a["displacement_score"],
                "score_b": cell_b["displacement_score"],
                "span_fraction_delta_b_minus_a": _six_places(
                    cell_b["span_fraction"] - cell_a["span_fraction"],
                    label="span fraction delta",
                ),
                "status": "compared",
                "survival_delta_b_minus_a": _six_places(
                    cell_b["survival"] - cell_a["survival"],
                    label="survival delta",
                ),
                "verdict": verdict,
            }
        )

    mean_delta = (
        _six_places(sum(score_deltas) / len(score_deltas), label="mean score delta")
        if score_deltas
        else None
    )
    if mean_delta is None:
        mean_verdict = "no comparable prompts"
    elif mean_delta > 0.0:
        mean_verdict = f"{condition_b} higher on mean by {mean_delta:.6f}"
    elif mean_delta < 0.0:
        mean_verdict = f"{condition_a} higher on mean by {abs(mean_delta):.6f}"
    else:
        mean_verdict = "mean tie at 0.000000"
    return {
        "comparable_prompt_count": len(score_deltas),
        "condition_a": condition_a,
        "condition_b": condition_b,
        "degraded_prompt_count": sum(
            bool(row["degraded_conditions"]) for row in rows
        ),
        "delta_definition": f"{condition_b} - {condition_a}",
        "error_prompt_count": sum(row["status"] == "error" for row in rows),
        "hole_prompt_count": sum(row["status"] == "missing" for row in rows),
        "invalid_prompt_count": sum(row["status"] == "invalid" for row in rows),
        "include_degraded": include_degraded,
        "mean_delta_b_minus_a": mean_delta,
        "mean_verdict": mean_verdict,
        "per_prompt": rows,
    }


def _escape_markdown(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("_", "\\_")
        .replace("*", "\\*")
    )


def _metric_text(cell: Mapping[str, Any], key: str) -> str:
    if cell["status"] == "missing":
        return "MISSING"
    if cell["status"] == "invalid":
        return "INVALID"
    if cell["status"] == "error":
        return "ERROR"
    if key == "displaced":
        return str(cell[key]).lower()
    suffix = (
        "*"
        if key == "displacement_score"
        and cell.get("coherence_degraded") is True
        else ""
    )
    return f"{cell[key]:.6f}{suffix}"


def _coherence_metric_text(cell: Mapping[str, Any]) -> str:
    if cell["status"] == "missing":
        return "MISSING"
    if cell["status"] == "invalid":
        return "INVALID"
    coherence_status = cell.get("coherence_status")
    if coherence_status == "error":
        return "ERROR"
    if coherence_status != "scored":
        return "UNAVAILABLE"
    return f"{cell['coherence_score']:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    conditions = report["conditions"]
    with_coherence = report.get("with_coherence") is True
    header = ["Prompt"]
    for condition in conditions:
        escaped = _escape_markdown(condition)
        if with_coherence:
            header.extend(
                [
                    f"{escaped} motion",
                    f"{escaped} coherence",
                    f"{escaped} span",
                    f"{escaped} survival",
                    f"{escaped} displaced",
                ]
            )
        else:
            header.extend(
                [
                    f"{escaped} score",
                    f"{escaped} span",
                    f"{escaped} survival",
                    f"{escaped} displaced",
                ]
            )
    lines = [
        (
            "# Motion × coherence batch matrix"
            if with_coherence
            else "# Displacement batch matrix"
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in report["matrix"]:
        values = [_escape_markdown(row["prompt"])]
        for condition in conditions:
            cell = row["cells"][condition]
            if with_coherence:
                values.extend(
                    [
                        _metric_text(cell, "displacement_score"),
                        _coherence_metric_text(cell),
                        _metric_text(cell, "span_fraction"),
                        _metric_text(cell, "survival"),
                        _metric_text(cell, "displaced"),
                    ]
                )
            else:
                values.extend(
                    _metric_text(cell, key)
                    for key in (
                        "displacement_score",
                        "span_fraction",
                        "survival",
                        "displaced",
                    )
                )
        lines.append("| " + " | ".join(values) + " |")

    has_degraded = any(
        cell.get("coherence_degraded") is True
        for row in report["matrix"]
        for cell in row["cells"].values()
    )
    if has_degraded:
        lines.extend(
            [
                "",
                "\\* Screen-space-only fallback; camera compensation unavailable. "
                "Excluded from comparisons unless `--include-degraded` is set.",
            ]
        )

    comparisons = report.get("comparisons")
    if comparisons is None:
        comparison = report.get("comparison")
        comparisons = [] if comparison is None else [comparison]
    for comparison in comparisons:
        condition_a = comparison["condition_a"]
        condition_b = comparison["condition_b"]
        lines.extend(
            [
                "",
                f"## Comparison: {_escape_markdown(condition_a)} vs "
                f"{_escape_markdown(condition_b)}",
                "",
                "| Prompt | A score | B score | B - A | Verdict |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparison["per_prompt"]:
            degraded_conditions = set(row.get("degraded_conditions", ()))

            def score_text(key: str, condition: str, unavailable: str) -> str:
                value = row[key]
                if value is None:
                    return unavailable
                suffix = "*" if condition in degraded_conditions else ""
                return f"{value:.6f}{suffix}"

            if row["status"] != "compared":
                unavailable = row["status"].upper()
                score_a = score_text("score_a", condition_a, unavailable)
                score_b = score_text("score_b", condition_b, unavailable)
                delta = unavailable
            else:
                score_a = score_text("score_a", condition_a, "UNAVAILABLE")
                score_b = score_text("score_b", condition_b, "UNAVAILABLE")
                delta = f"{row['delta_b_minus_a']:.6f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(row["prompt"]),
                        score_a,
                        score_b,
                        delta,
                        _escape_markdown(row["verdict"]),
                    ]
                )
                + " |"
            )
        mean = comparison["mean_delta_b_minus_a"]
        mean_text = "UNAVAILABLE" if mean is None else f"{mean:.6f}"
        lines.extend(
            [
                "",
                f"Mean B - A ({comparison['comparable_prompt_count']} comparable): "
                f"{mean_text} — {_escape_markdown(comparison['mean_verdict'])}",
                f"Degraded prompts: {comparison['degraded_prompt_count']} "
                f"({'included' if comparison['include_degraded'] else 'excluded'}).",
            ]
        )
    return "\n".join(lines) + "\n"


def run_batch(
    root: str | Path,
    *,
    prompts_path: str | Path,
    output_dir: str | Path,
    compare: tuple[str, str] | None = None,
    comparisons: Sequence[tuple[str, str]] | None = None,
    prompt_substrings: Mapping[str, str] | None = None,
    invalid_conditions: Mapping[str, InvalidConditionSpec] | None = None,
    include_degraded: bool = False,
    scorer: ScoreClip | None = None,
    config: MetricConfig | None = None,
    with_coherence: bool = False,
    coherence_scorer: CoherenceScoreClip | None = None,
    coherence_config: CoherenceConfig | None = None,
) -> dict[str, Any]:
    input_root = Path(root).expanduser().resolve()
    prompt_file = Path(prompts_path).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    if resolved_output == input_root:
        raise BatchComparisonError("output directory cannot be the batch input root")
    prompts = _read_prompts(prompt_file)
    conditions, clips_by_condition = _discover_conditions(
        input_root,
        output_dir=resolved_output,
    )
    normalized_invalid = _normalize_invalid_conditions(invalid_conditions, conditions)
    comparison_pairs = _normalize_comparisons(
        compare,
        comparisons,
        conditions,
        normalized_invalid,
    )

    slot_mode, slot_filenames, assignments = _slot_assignments(
        conditions,
        clips_by_condition,
        prompts,
        prompt_substrings,
    )
    resolved_config = MetricConfig() if config is None else config
    resolved_scorer = score_clip if scorer is None else scorer
    if scorer is None:
        track_backend, appearance_backend = _backends(resolved_config)
    else:
        track_backend, appearance_backend = None, None
    resolved_coherence_config: CoherenceConfig | None = None
    resolved_coherence_scorer: CoherenceScoreClip | None = None
    coherence_appearance_backend: Any | None = None
    coherence_flow_backend: FlowBackend | None = None
    if with_coherence:
        resolved_coherence_config = (
            CoherenceConfig(
                expected_frames=resolved_config.expected_frames,
                expected_height=resolved_config.expected_height,
                expected_width=resolved_config.expected_width,
                device=resolved_config.device,
            )
            if coherence_config is None
            else coherence_config
        )
        resolved_coherence_scorer = (
            score_coherence_clip
            if coherence_scorer is None
            else coherence_scorer
        )
        if coherence_scorer is None:
            if (
                appearance_backend is not None
                and resolved_config.appearance == resolved_coherence_config.appearance
            ):
                coherence_appearance_backend = appearance_backend
            elif resolved_coherence_config.appearance == "histogram":
                coherence_appearance_backend = HistogramAppearanceBackend()
            else:
                coherence_appearance_backend = DINOv2AppearanceBackend(
                    resolved_coherence_config.device
                )
            coherence_flow_backend = (
                RaftSmallFlowBackend(
                    resolved_coherence_config.device,
                    batch_size=resolved_coherence_config.flow_batch_size,
                )
                if resolved_coherence_config.flow == "raft-small"
                else OpenCVFarnebackFlowBackend()
            )

    clips: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    holes: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    coherence_errors: list[dict[str, Any]] = []
    matrix = [
        {"cells": {}, "filename_slot": slot_filenames[index], "prompt": prompt}
        for index, prompt in enumerate(prompts)
    ]
    full_reports: list[tuple[Path, dict[str, Any]]] = []
    full_coherence_reports: list[tuple[Path, dict[str, Any]]] = []
    for condition in conditions:
        for index, prompt in enumerate(prompts):
            source = assignments[condition][index]
            slot_filename = slot_filenames[index]
            if condition in normalized_invalid:
                filename = slot_filename if source is None else source.name
                artifact_status = "missing" if source is None else "present"
                cell = _invalid_cell(
                    condition,
                    prompt,
                    filename,
                    artifact_status=artifact_status,
                    spec=normalized_invalid[condition],
                )
                matrix[index]["cells"][condition] = cell
                invalid.append(
                    {
                        **cell,
                        "source": None if source is None else str(source.resolve()),
                    }
                )
                continue
            if source is None:
                hole = {
                    "condition": condition,
                    "filename": slot_filename,
                    "prompt": prompt,
                }
                holes.append(hole)
                matrix[index]["cells"][condition] = _missing_cell(
                    condition,
                    prompt,
                    slot_filename,
                )
                continue

            coherence_fields: dict[str, Any] = {}
            if with_coherence:
                if resolved_coherence_config is None or resolved_coherence_scorer is None:
                    raise BatchComparisonError("coherence scorer setup is incomplete")
                coherence_relative = (
                    Path("coherence") / condition / f"{source.name}.json"
                )
                try:
                    coherence_report = resolved_coherence_scorer(
                        source,
                        config=resolved_coherence_config,
                        appearance_backend=coherence_appearance_backend,
                        flow_backend=coherence_flow_backend,
                    )
                except CoherenceMetricError as error:
                    coherence_fields = _coherence_error_projection(error)
                    coherence_errors.append(
                        {
                            "condition": condition,
                            "error_message": str(error),
                            "error_type": type(error).__name__,
                            "filename": source.name,
                            "prompt": prompt,
                            "source": str(source.resolve()),
                        }
                    )
                else:
                    coherence_fields = {
                        "coherence_result_json": coherence_relative.as_posix(),
                        "coherence_status": "scored",
                        **_project_coherence_report(coherence_report),
                    }
                    full_coherence_reports.append(
                        (coherence_relative, coherence_report)
                    )

            try:
                report = resolved_scorer(
                    source,
                    config=resolved_config,
                    track_backend=track_backend,
                    appearance_backend=appearance_backend,
                )
            except DisplacementMetricError as error:
                cell = _error_cell(
                    condition,
                    prompt,
                    source.name,
                    error,
                )
                if with_coherence:
                    cell.update(coherence_fields)
                matrix[index]["cells"][condition] = cell
                errors.append({**cell, "source": str(source.resolve())})
                continue
            projected = _project_report(report)
            result_relative = Path("clips") / condition / f"{source.name}.json"
            cell = {
                "condition": condition,
                "filename": source.name,
                "prompt": prompt,
                "result_json": result_relative.as_posix(),
                "status": "scored",
                **projected,
            }
            if with_coherence:
                cell.update(coherence_fields)
            matrix[index]["cells"][condition] = cell
            clip_record = {
                "condition": condition,
                "filename": source.name,
                "prompt": prompt,
                "result_json": result_relative.as_posix(),
                "source": str(source.resolve()),
                **projected,
            }
            if with_coherence:
                clip_record.update(coherence_fields)
            clips.append(clip_record)
            full_reports.append((result_relative, report))

    batch_report: dict[str, Any] = {
        "clips": clips,
        "conditions": conditions,
        "errors": errors,
        "holes": holes,
        "invalid": invalid,
        "invalid_conditions": normalized_invalid,
        "include_degraded_comparisons": include_degraded,
        "input_root": str(input_root),
        "kind": "coherent-displacement-batch-comparison",
        "matrix": matrix,
        "metric_kind": "coherent-displacement-metrics",
        "prompt_order": prompts,
        "prompt_substrings": (
            None
            if prompt_substrings is None
            else {prompt: prompt_substrings[prompt] for prompt in prompts}
        ),
        "schema_version": 3,
        "slot_mode": slot_mode,
    }
    if with_coherence:
        batch_report.update(
            {
                "coherence_errors": coherence_errors,
                "coherence_metric_kind": "video-coherence-metrics",
                "schema_version": 4,
                "with_coherence": True,
            }
        )
    if comparison_pairs:
        comparison_reports = [
            _comparison(matrix, *pair, include_degraded=include_degraded)
            for pair in comparison_pairs
        ]
        batch_report["comparisons"] = comparison_reports
        if compare is not None:
            batch_report["comparison"] = comparison_reports[0]

    owned_clips = resolved_output / "clips"
    owned_coherence = resolved_output / "coherence"
    for owned_path, label in (
        (owned_clips, "clips"),
        (owned_coherence, "coherence"),
    ):
        if owned_path.is_symlink():
            raise BatchComparisonError(
                f"output {label} directory cannot be a symlink"
            )
        if owned_path.exists() and not owned_path.is_dir():
            raise BatchComparisonError(f"output {label} path must be a directory")
    if owned_clips.exists():
        shutil.rmtree(owned_clips)
    if owned_coherence.exists():
        shutil.rmtree(owned_coherence)
    for relative_path, full_report in full_reports:
        destination = resolved_output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_canonical_json(full_report), encoding="utf-8")
    for relative_path, full_report in full_coherence_reports:
        destination = resolved_output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_canonical_json(full_report), encoding="utf-8")
    resolved_output.mkdir(parents=True, exist_ok=True)
    (resolved_output / "report.json").write_text(
        _canonical_batch_json(batch_report) + "\n",
        encoding="utf-8",
    )
    (resolved_output / "matrix.md").write_text(
        render_markdown(batch_report),
        encoding="utf-8",
    )
    return batch_report


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height") from error
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height")
    return values


def _read_json_mapping(path: Path | None, *, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BatchComparisonError(f"{label} file is not valid JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BatchComparisonError(f"{label} file must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root containing <condition>/<clip>.mp4")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        help="repeatable pair; every delta is B minus A",
    )
    parser.add_argument(
        "--prompt-substrings",
        type=Path,
        help="JSON object mapping each prompt name to a filename substring",
    )
    parser.add_argument(
        "--invalid-conditions",
        type=Path,
        help="JSON object mapping invalid condition names to reason objects",
    )
    parser.add_argument(
        "--include-degraded",
        action="store_true",
        help="include screen-space-only degraded scores in comparison deltas",
    )
    parser.add_argument(
        "--with-coherence",
        action="store_true",
        help="add the independent DINOv2 + RAFT coherence axis",
    )
    parser.add_argument(
        "--tracker",
        choices=("cotracker3", "opencv-lk"),
        default="cotracker3",
    )
    parser.add_argument(
        "--appearance",
        choices=("dinov2", "histogram"),
        default="dinov2",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--bbox", type=_parse_bbox)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        prompt_substrings = _read_json_mapping(
            arguments.prompt_substrings,
            label="prompt substrings",
        )
        invalid_conditions = _read_json_mapping(
            arguments.invalid_conditions,
            label="invalid conditions",
        )
        config = MetricConfig(
            tracker=arguments.tracker,
            appearance=arguments.appearance,
            device=arguments.device,
            grid_size=arguments.grid_size,
            sample_stride=arguments.sample_stride,
            primary_bbox=arguments.bbox,
        )
        report = run_batch(
            arguments.root,
            prompts_path=arguments.prompts,
            output_dir=arguments.output_dir,
            comparisons=(
                None
                if arguments.compare is None
                else tuple(tuple(pair) for pair in arguments.compare)
            ),
            prompt_substrings=prompt_substrings,
            invalid_conditions=invalid_conditions,
            include_degraded=arguments.include_degraded,
            config=config,
            with_coherence=arguments.with_coherence,
        )
        sys.stdout.write(render_markdown(report))
    except (
        BatchComparisonError,
        CoherenceMetricError,
        DisplacementMetricError,
        OSError,
    ) as error:
        print(f"displacement batch failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
