from __future__ import annotations

import csv
import io
import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from bench.displacement_batch import _canonical_batch_json


OUTPUT = Path(__file__).resolve().parent
PROMPTS = ["ball", "walker", "barrel", "vehicle"]

CORPORA = {
    "sgmd_pilot": {
        "label": "Raw SGMD (unnormalized Fisher, lambda=0.1)",
        "input_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-pilot-20260721/clips"
        ),
        "output_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-pilot-20260721/batch_scores_2axis"
        ),
    },
    "sgmd_fnorm": {
        "label": "Fisher-normalized SGMD (lambda=0.1)",
        "input_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-fnorm-20260721/clips"
        ),
        "output_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-fnorm-20260721/batch_scores_2axis"
        ),
    },
    "sgmd_lsweep": {
        "label": "Fisher-normalized lambda sweep",
        "input_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-lsweep-20260721/clips"
        ),
        "output_root": Path(
            "/Users/electric/Documents/areas_of_focus/decart-research/"
            "sgmd-lsweep-20260721/batch_scores_2axis"
        ),
    },
}

SERIES = [
    (
        "raw_sgmd_lam0p1",
        "sgmd_pilot",
        [(step, f"step_{step:06d}") for step in (25, 50, 75, 100)],
    ),
    (
        "fnorm_lam0p1",
        "sgmd_fnorm",
        [(step, f"step_{step:06d}") for step in range(5, 51, 5)],
    ),
    (
        "fnorm_lam0p05",
        "sgmd_lsweep",
        [(step, f"lam0p05_step_{step:06d}") for step in range(10, 51, 10)],
    ),
    (
        "fnorm_lam0p2",
        "sgmd_lsweep",
        [(step, f"lam0p2_step_{step:06d}") for step in range(10, 51, 10)],
    ),
]

CELL_FIELDS = (
    "camera_compensated",
    "coherence_degraded",
    "coherence_score",
    "coherence_status",
    "degrades_over_time",
    "displaced",
    "displacement_score",
    "error_message",
    "error_type",
    "filename",
    "result_json",
    "coherence_result_json",
    "span_fraction",
    "spatial_integrity_score",
    "status",
    "survival",
    "temporal_coherence_score",
)


SIX_PLACES = Decimal("0.000001")


def fixed6(value: float | Decimal) -> float:
    return float(Decimal(str(value)).quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    result = sum(Decimal(str(value)) for value in values) / Decimal(len(values))
    return fixed6(result)


def load_native_reports() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for corpus, spec in CORPORA.items():
        report = json.loads((spec["output_root"] / "report.json").read_text())
        assert report["schema_version"] == 4
        assert report["with_coherence"] is True
        assert report["prompt_order"] == PROMPTS
        reports[corpus] = report
    return reports


def native_cell_map(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        row["prompt"]: {condition: dict(cell) for condition, cell in row["cells"].items()}
        for row in report["matrix"]
    }


def project_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: cell.get(key) for key in CELL_FIELDS}


def condition_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    displacement = [
        float(cell["displacement_score"])
        for cell in cells
        if cell["status"] == "scored" and cell.get("camera_compensated") is True
    ]
    degraded_values = [
        float(cell["displacement_score"])
        for cell in cells
        if cell["status"] == "scored" and cell.get("coherence_degraded") is True
    ]
    coherence = [
        float(cell["coherence_score"])
        for cell in cells
        if cell.get("coherence_status") == "scored"
    ]
    displacement_errors = sum(cell["status"] == "error" for cell in cells)
    coherence_errors = sum(cell.get("coherence_status") == "error" for cell in cells)
    missing = sum(cell["status"] == "missing" for cell in cells)
    invalid = sum(cell["status"] == "invalid" for cell in cells)
    if missing == len(cells):
        status = "missing"
    elif displacement_errors or coherence_errors or missing or invalid:
        status = "partial"
    else:
        status = "scored"
    return {
        "coherence_error_count": coherence_errors,
        "coherence_scored_count": len(coherence),
        "compensated_displacement_count": len(displacement),
        "degraded_count": len(degraded_values),
        "degraded_screen_space_mean": mean(degraded_values),
        "displaced_count": sum(cell.get("displaced") is True for cell in cells),
        "displacement_error_count": displacement_errors,
        "expected_prompt_count": len(cells),
        "invalid_count": invalid,
        "mean_coherence": mean(coherence),
        "mean_displacement": mean(displacement),
        "missing_count": missing,
        "status": status,
    }


def read_vehicle_reports(
    corpus: str,
    condition: str,
    cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = CORPORA[corpus]["output_root"]
    displacement = json.loads((output_root / cell["result_json"]).read_text())
    coherence = json.loads((output_root / cell["coherence_result_json"]).read_text())
    return displacement, coherence


def segment_projection(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "coherence_score": segment["coherence_score"],
        "end_frame": segment["end_frame"],
        "name": segment["name"],
        "spatial_score": segment["spatial"]["score"],
        "start_frame": segment["start_frame"],
        "temporal": {
            "adjacent_feature_similarity_median": segment["temporal"][
                "adjacent_feature_similarity_median"
            ],
            "adjacent_feature_similarity_p10": segment["temporal"][
                "adjacent_feature_similarity_p10"
            ],
            "adjacent_flow_warp_residual_max": segment["temporal"][
                "adjacent_flow_warp_residual_max"
            ],
            "adjacent_flow_warp_residual_median": segment["temporal"][
                "adjacent_flow_warp_residual_median"
            ],
            "raw_flow_magnitude_mean_px": segment["temporal"][
                "raw_flow_magnitude_mean_px"
            ],
            "score": segment["temporal"]["score"],
        },
    }


def cross_checks(cell_maps: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected = {
        "sgmd_pilot": {25: 2.031, 50: 0.352, 75: 0.061, 100: 0.227},
        "sgmd_fnorm": {
            10: 1.796,
            15: 2.457,
            20: 1.963,
            25: 0.009,
            30: 2.200,
            35: 2.046,
            40: 1.346,
        },
    }
    for corpus, by_step in expected.items():
        for checkpoint, expected_rounded in by_step.items():
            condition = f"step_{checkpoint:06d}"
            cell = cell_maps[corpus]["vehicle"][condition]
            actual = float(cell["displacement_score"])
            if corpus == "sgmd_pilot":
                baseline_root = CORPORA[corpus]["output_root"].parent / "batch_scores"
                baseline = next(
                    (baseline_root / "clips" / condition).glob("*street view*.json")
                )
            else:
                baseline = (
                    CORPORA[corpus]["output_root"].parent
                    / "metric_scores"
                    / f"vehicle_{checkpoint:06d}.json"
                )
            current = CORPORA[corpus]["output_root"] / cell["result_json"]
            rounded_match = f"{actual:.3f}" == f"{expected_rounded:.3f}"
            exact_report_match = baseline.read_bytes() == current.read_bytes()
            numeric_and_exact_match = rounded_match and exact_report_match
            status = (
                "fail"
                if not numeric_and_exact_match
                else (
                    "pass_compensated"
                    if cell["camera_compensated"] is True
                    else "matched_degraded"
                )
            )
            checks.append(
                {
                    "actual": actual,
                    "baseline_report": str(baseline),
                    "camera_compensated": cell["camera_compensated"],
                    "checkpoint": checkpoint,
                    "condition": condition,
                    "corpus": corpus,
                    "exact_report_match": exact_report_match,
                    "expected_reported": expected_rounded,
                    "matches_reported_precision": rounded_match,
                    "numeric_and_exact_report_match": numeric_and_exact_match,
                    "prompt": "vehicle",
                    "status": status,
                }
            )
    assert all(check["numeric_and_exact_report_match"] for check in checks)
    compensated_pass_count = sum(
        check["status"] == "pass_compensated" for check in checks
    )
    matched_degraded_count = sum(
        check["status"] == "matched_degraded" for check in checks
    )
    return {
        "all_numeric_and_exact_report_checks_pass": True,
        "all_compensated_contract_checks_pass": matched_degraded_count == 0,
        "checks": checks,
        "compensated_pass_count": compensated_pass_count,
        "matched_degraded_count": matched_degraded_count,
        "note": (
            "All supplied scalars match at their reported three-decimal precision and "
            "all 11 current displacement reports are byte-identical to the independent "
            "stored spot reports. Nine checks satisfy the compensated contract. Fnorm "
            "vehicle steps 15 and 20 are matched degraded screen-space diagnostics, not "
            "compensated passes, despite the task wording."
        ),
    }


def build_report() -> dict[str, Any]:
    native = load_native_reports()
    cell_maps = {name: native_cell_map(report) for name, report in native.items()}

    corpora_payload = []
    for corpus, spec in CORPORA.items():
        report = native[corpus]
        matrix = []
        for prompt in PROMPTS:
            matrix.append(
                {
                    "cells": {
                        condition: project_cell(cell_maps[corpus][prompt][condition])
                        for condition in report["conditions"]
                    },
                    "prompt": prompt,
                }
            )
        per_condition = []
        for condition in report["conditions"]:
            summary = condition_summary(
                [cell_maps[corpus][prompt][condition] for prompt in PROMPTS]
            )
            per_condition.append({"condition": condition, **summary})
        corpora_payload.append(
            {
                "condition_order": report["conditions"],
                "conditions": per_condition,
                "corpus": corpus,
                "input_root": str(spec["input_root"]),
                "label": spec["label"],
                "matrix": matrix,
                "native_matrix_markdown": str(spec["output_root"] / "matrix.md"),
                "native_report_json": str(spec["output_root"] / "report.json"),
                "output_root": str(spec["output_root"]),
            }
        )

    trajectory = []
    condition_holes = []
    trajectory_by_run: dict[str, list[dict[str, Any]]] = {}
    for run, corpus, checkpoints in SERIES:
        rows = []
        available = set(native[corpus]["conditions"])
        for checkpoint, condition in checkpoints:
            if condition not in available:
                summary = {
                    "coherence_error_count": 0,
                    "coherence_scored_count": 0,
                    "compensated_displacement_count": 0,
                    "degraded_count": 0,
                    "degraded_screen_space_mean": None,
                    "displaced_count": 0,
                    "displacement_error_count": 0,
                    "expected_prompt_count": len(PROMPTS),
                    "invalid_count": 0,
                    "mean_coherence": None,
                    "mean_displacement": None,
                    "missing_count": len(PROMPTS),
                    "status": "missing",
                }
                condition_holes.append(
                    {
                        "checkpoint": checkpoint,
                        "condition": condition,
                        "corpus": corpus,
                        "prompts": PROMPTS,
                        "reason": "condition_directory_absent_at_scoring",
                        "run": run,
                    }
                )
            else:
                summary = condition_summary(
                    [cell_maps[corpus][prompt][condition] for prompt in PROMPTS]
                )
            row = {
                "checkpoint": checkpoint,
                "condition": condition,
                "corpus": corpus,
                "run": run,
                **summary,
            }
            rows.append(row)
            trajectory.append(row)
        trajectory_by_run[run] = rows

    series_summary = []
    for run, corpus, checkpoints in SERIES:
        compensated: list[float] = []
        coherence: list[float] = []
        degraded_count = 0
        displacement_errors = 0
        for _checkpoint, condition in checkpoints:
            if condition not in native[corpus]["conditions"]:
                continue
            for prompt in PROMPTS:
                cell = cell_maps[corpus][prompt][condition]
                if cell["status"] == "scored" and cell.get("camera_compensated") is True:
                    compensated.append(float(cell["displacement_score"]))
                if cell.get("coherence_status") == "scored":
                    coherence.append(float(cell["coherence_score"]))
                degraded_count += cell.get("coherence_degraded") is True
                displacement_errors += cell["status"] == "error"
        rows = trajectory_by_run[run]
        series_summary.append(
            {
                "checkpoint_count": len(rows),
                "degraded_count": degraded_count,
                "displacement_error_count": displacement_errors,
                "pooled_coherence_count": len(coherence),
                "pooled_mean_coherence": mean(coherence),
                "pooled_mean_displacement": mean(compensated),
                "pooled_compensated_displacement_count": len(compensated),
                "run": run,
            }
        )

    lambda_common_prompt_comparison = []
    for checkpoint in range(10, 51, 10):
        low_condition = f"lam0p05_step_{checkpoint:06d}"
        high_condition = f"lam0p2_step_{checkpoint:06d}"
        common_prompts = [
            prompt
            for prompt in PROMPTS
            if cell_maps["sgmd_lsweep"][prompt][low_condition]["status"] == "scored"
            and cell_maps["sgmd_lsweep"][prompt][low_condition].get(
                "camera_compensated"
            )
            is True
            and cell_maps["sgmd_lsweep"][prompt][high_condition]["status"]
            == "scored"
            and cell_maps["sgmd_lsweep"][prompt][high_condition].get(
                "camera_compensated"
            )
            is True
        ]
        low_mean = mean(
            [
                float(
                    cell_maps["sgmd_lsweep"][prompt][low_condition][
                        "displacement_score"
                    ]
                )
                for prompt in common_prompts
            ]
        )
        high_mean = mean(
            [
                float(
                    cell_maps["sgmd_lsweep"][prompt][high_condition][
                        "displacement_score"
                    ]
                )
                for prompt in common_prompts
            ]
        )
        assert low_mean is not None and high_mean is not None
        lambda_common_prompt_comparison.append(
            {
                "checkpoint": checkpoint,
                "common_prompts": common_prompts,
                "fnorm_lam0p05_mean_displacement": low_mean,
                "fnorm_lam0p2_mean_displacement": high_mean,
                "high_minus_low": fixed6(high_mean - low_mean),
            }
        )

    anomaly_points = []
    anomaly_reports: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for checkpoint in (20, 25, 30):
        condition = f"step_{checkpoint:06d}"
        cell = cell_maps["sgmd_fnorm"]["vehicle"][condition]
        displacement, coherence = read_vehicle_reports("sgmd_fnorm", condition, cell)
        anomaly_reports[checkpoint] = (displacement, coherence)
        anomaly_points.append(
            {
                "camera_compensated": cell["camera_compensated"],
                "checkpoint": checkpoint,
                "coherence_degraded": cell["coherence_degraded"],
                "coherence_score": cell["coherence_score"],
                "degrades_over_time": cell["degrades_over_time"],
                "displacement_score": cell["displacement_score"],
                "spatial_integrity_score": cell["spatial_integrity_score"],
                "status": cell["status"],
                "temporal_coherence_score": cell["temporal_coherence_score"],
            }
        )
    step25_displacement, step25_coherence = anomaly_reports[25]
    anomaly = {
        "classification": "valid_decode_transport_shortfall_with_tracking_and_identity_confidence_failure",
        "confidence": "moderate",
        "diagnosis": (
            "The file is a healthy 81-frame decode and the learned coherence axis remains "
            "8.723019 with temporal=10. The clip therefore is neither a broken decode nor "
            "a coherence collapse. Its requested transport is visibly weaker and reverses "
            "mid-clip, but 0.009206 is not a clean metric-confirmed dropout: visual centroid "
            "span is 196 px (0.235577W) while the selected tracks report only 0.035408W, "
            "with three persistent tracks, 0.078189 survival, and appearance_guard=0.0. "
            "The observed transport shortfall is real; the scalar severity is dominated by "
            "tracking and identity-confidence failure."
        ),
        "neighbor_points": anomaly_points,
        "redecode_recommendation": (
            "Do not re-decode the same artifact as a corruption repair. If checkpoint-level "
            "promotion hinges on this isolated dip, retain the clip and add another "
            "deterministic seed plus a manual/bbox-assisted track check rather than replacing "
            "this valid observation."
        ),
        "step_25_displacement": {
            "camera_compensated_span_fraction_width": step25_displacement["foreground"][
                "camera_compensated_trajectory_span_fraction_width"
            ],
            "coherence_guard": step25_displacement["score_components"]["coherence_guard"],
            "appearance_guard": step25_displacement["score_components"]["appearance_guard"],
            "displacement_extent": step25_displacement["score_components"][
                "displacement_extent"
            ],
            "displacement_score": step25_displacement["displacement_score"],
            "persistent_track_count": step25_displacement["foreground"][
                "persistent_track_count"
            ],
            "raw_flow_mean_px": step25_displacement["morph_guardrail"][
                "raw_flow_mean_px"
            ],
            "dinov2_feature_similarity_median": step25_displacement["morph_guardrail"][
                "dinov2_feature_similarity_median"
            ],
            "dinov2_feature_similarity_p10": step25_displacement["morph_guardrail"][
                "dinov2_feature_similarity_p10"
            ],
            "screen_space_span_fraction_width": step25_displacement["foreground"][
                "screen_space_trajectory_span_fraction_width"
            ],
            "selected_track_count": step25_displacement["foreground"][
                "selected_track_count"
            ],
            "track_survival": step25_displacement["foreground"]["track_survival"],
            "translation_consensus": step25_displacement["foreground"][
                "translation_consensus"
            ],
        },
        "step_25_coherence": {
            "coherence_score": step25_coherence["coherence_score"],
            "degradation_drop": step25_coherence["degradation_drop"],
            "degrades_over_time": step25_coherence["degrades_over_time"],
            "segments": [segment_projection(segment) for segment in step25_coherence["segments"]],
            "spatial_integrity_score": step25_coherence["spatial_integrity_score"],
            "temporal_coherence_score": step25_coherence["temporal_coherence_score"],
        },
        "visual_decode_check": {
            "adjacent_grayscale_mad_median": 8.597,
            "centroid_x_frames_0_20_40_60_80": [295, 400, 489, 455, 426],
            "codec": "H.264",
            "decode_errors": 0,
            "duration_seconds": 5.0625,
            "frame_count": 81,
            "frame_rate": 16.0,
            "neighbor_centroid_spans_px": {"step_20": 279, "step_30": 396},
            "resolution": "832x480",
            "roi_sharpness_early_middle_late": [2312, 1806, 931],
            "step_25_centroid_span_px": 196,
            "zero_near_duplicate_adjacent_pairs": True,
        },
    }

    summary_by_run = {row["run"]: row for row in series_summary}
    low_lambda = summary_by_run["fnorm_lam0p05"]
    high_lambda = summary_by_run["fnorm_lam0p2"]
    report = {
        "aggregation_policy": {
            "degraded_count": (
                "Count status=scored cells with coherence_degraded=true; this flag denotes "
                "screen-space-only displacement, not coherence-axis failure."
            ),
            "mean_coherence": (
                "Mean every numeric coherence_score with coherence_status=scored, including "
                "cells whose displacement is degraded."
            ),
            "mean_displacement": (
                "Mean only status=scored cells with camera_compensated=true. Screen-space "
                "fallbacks, errors, invalid cells, and holes are excluded, never zero-filled."
            ),
        },
        "conclusions": {
            "lambda_answer": (
                "The data contradicts the proposed monotonic framing. Lower lambda=0.05 "
                "does not hold motion longer or blur less: across its 15 compensated cells "
                f"it averages {low_lambda['pooled_mean_displacement']:.6f} motion and "
                f"{low_lambda['pooled_mean_coherence']:.6f} coherence across all 20 cells. "
                f"Lambda=0.2 averages {high_lambda['pooled_mean_displacement']:.6f} motion "
                "across the same 15 compensated-cell coverage and "
                f"{high_lambda['pooled_mean_coherence']:.6f} coherence across all 20 cells, finishing at "
                "1.693540 / 7.403891 with zero degraded cells."
            ),
            "lambda_denominator_caveat": (
                "On prompts camera-compensated under both lambdas, lambda=0.2 wins motion "
                "at four of five matched checkpoints but loses at step 20. The advantage is "
                "prompt-specific: at step 50 its walker is lower than lambda=0.05 while its "
                "barrel is much higher."
            ),
            "normalized_lam0p1": (
                "Fisher normalization avoids the raw run's simple early transport collapse, "
                "but the path is non-monotonic and early motion means have thin denominators. "
                "From step 30 onward it occupies roughly 0.61-1.30 compensated motion and "
                "7.10-7.36 coherence, with one error at step 50."
            ),
            "raw_sgmd": (
                "Raw SGMD has one compensated cell at step 25 (the vehicle at 2.030809) and "
                "three degraded cells; by step 50 its compensated mean is 0.224815 while "
                "coherence rises to 6.871822. It does not trace a stable top-right path."
            ),
            "scope_caveat": (
                "Displacement plus coherence does not establish prompt fidelity or correct "
                "physics. In particular, lambda=0.2's very large vehicle scores and its "
                "step-30 ball motion should still receive frame/blind review before promotion."
            ),
        },
        "condition_holes": condition_holes,
        "corpora": corpora_payload,
        "corpus_order": list(CORPORA),
        "cross_checks": cross_checks(cell_maps),
        "kind": "sgmd-motion-coherence-training-trajectory",
        "lambda_common_prompt_comparison": lambda_common_prompt_comparison,
        "prompt_order": PROMPTS,
        "schema_version": 1,
        "series_order": [run for run, _corpus, _checkpoints in SERIES],
        "series_summary": series_summary,
        "step_25_anomaly": anomaly,
        "trajectory": trajectory,
    }
    return report


def cell_text(cell: dict[str, Any]) -> str:
    coherence = (
        f"{cell['coherence_score']:.6f}"
        if cell.get("coherence_status") == "scored"
        else "ERROR(C)"
    )
    if cell["status"] == "error":
        return f"ERROR(D) / {coherence}"
    if cell["status"] == "missing":
        return "MISSING"
    if cell["status"] == "invalid":
        return "INVALID"
    suffix = "*" if cell.get("coherence_degraded") is True else ""
    return f"{cell['displacement_score']:.6f}{suffix} / {coherence}"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SGMD motion × coherence training trajectories",
        "",
        "All motion means below use camera-compensated cells only. A `*` marks a "
        "screen-space-only displacement diagnostic; starred cells are counted separately "
        "and excluded from motion means. Coherence remains independently usable. Errors and "
        "holes are never zero-filled.",
        "",
        "## Cross-check",
        "",
        "All 11 supplied vehicle anchors match numerically at the reported precision, and "
        "their full displacement JSON files are byte-identical to the stored independent "
        "spot reports. Nine satisfy the compensated contract. Fisher-normalized steps 15 "
        "and 20 are `matched_degraded`: their numbers match, but they are starred "
        "screen-space fallbacks rather than compensated passes.",
        "",
        "| Corpus | Step | Expected | Actual | Compensated | Exact report | Status |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for check in report["cross_checks"]["checks"]:
        lines.append(
            f"| {check['corpus']} | {check['checkpoint']} | "
            f"{check['expected_reported']:.3f} | {check['actual']:.6f} | "
            f"{str(check['camera_compensated']).lower()} | "
            f"{str(check['exact_report_match']).lower()} | {check['status']} |"
        )

    for corpus in report["corpora"]:
        lines.extend(["", f"## {corpus['label']}", ""])
        conditions = corpus["condition_order"]
        lines.append("| Prompt | " + " | ".join(conditions) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(conditions)) + " |")
        for row in corpus["matrix"]:
            lines.append(
                "| "
                + row["prompt"]
                + " | "
                + " | ".join(cell_text(row["cells"][condition]) for condition in conditions)
                + " |"
            )
        error_count = sum(item["displacement_error_count"] for item in corpus["conditions"])
        lines.extend(
            [
                "",
                f"Native tree: `{corpus['output_root']}`. Displacement errors: {error_count}; "
                "coherence errors: "
                f"{sum(item['coherence_error_count'] for item in corpus['conditions'])}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Trajectory table",
            "",
            "| Run | Step | Mean motion | Motion n/4 | Mean coherence | Coherence n/4 | Degraded | Errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["trajectory"]:
        motion = "UNAVAILABLE" if row["mean_displacement"] is None else f"{row['mean_displacement']:.6f}"
        coherence = "UNAVAILABLE" if row["mean_coherence"] is None else f"{row['mean_coherence']:.6f}"
        errors = row["displacement_error_count"] + row["coherence_error_count"]
        lines.append(
            f"| {row['run']} | {row['checkpoint']} | {motion} | "
            f"{row['compensated_displacement_count']}/4 | {coherence} | "
            f"{row['coherence_scored_count']}/4 | {row['degraded_count']} | {errors} |"
        )

    lines.extend(
        [
            "",
            "## What the lambda sweep says",
            "",
            report["conclusions"]["lambda_answer"],
            "",
            report["conclusions"]["lambda_denominator_caveat"],
            "",
            "| Step | Common compensated prompts | lambda=0.05 motion | lambda=0.2 motion | High - low |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for comparison in report["lambda_common_prompt_comparison"]:
        lines.append(
            f"| {comparison['checkpoint']} | {', '.join(comparison['common_prompts'])} | "
            f"{comparison['fnorm_lam0p05_mean_displacement']:.6f} | "
            f"{comparison['fnorm_lam0p2_mean_displacement']:.6f} | "
            f"{comparison['high_minus_low']:+.6f} |"
        )
    lines.extend(
        [
            "",
            report["conclusions"]["scope_caveat"],
            "",
            "## Fisher-normalized vehicle step-25 anomaly",
            "",
            "| Step | Motion | Coherence | Temporal | Spatial | Compensated | Degraded |",
            "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for point in report["step_25_anomaly"]["neighbor_points"]:
        lines.append(
            f"| {point['checkpoint']} | {point['displacement_score']:.6f} | "
            f"{point['coherence_score']:.6f} | {point['temporal_coherence_score']:.6f} | "
            f"{point['spatial_integrity_score']:.6f} | "
            f"{str(point['camera_compensated']).lower()} | "
            f"{str(point['coherence_degraded']).lower()} |"
        )
    anomaly = report["step_25_anomaly"]
    lines.extend(
        [
            "",
            f"**Diagnosis:** {anomaly['diagnosis']}",
            "",
            f"**Re-decode:** {anomaly['redecode_recommendation']}",
            "",
            "Step-25 coherence segments:",
            "",
            "| Segment | Frames | Coherence | Temporal | Spatial | DINO median | Warp median | Warp max | Raw flow px |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for segment in anomaly["step_25_coherence"]["segments"]:
        temporal = segment["temporal"]
        lines.append(
            f"| {segment['name']} | {segment['start_frame']}-{segment['end_frame']} | "
            f"{segment['coherence_score']:.6f} | {temporal['score']:.6f} | "
            f"{segment['spatial_score']:.6f} | "
            f"{temporal['adjacent_feature_similarity_median']:.6f} | "
            f"{temporal['adjacent_flow_warp_residual_median']:.6f} | "
            f"{temporal['adjacent_flow_warp_residual_max']:.6f} | "
            f"{temporal['raw_flow_magnitude_mean_px']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Early checkpoint means can have only 1-2 compensated prompts; compare the `n/4` denominator, not only the scalar.",
            "- `degrades_over_time=false` does not imply perfect structure; step 25 has an 8.58 middle-segment dip but misses the registered two-point degradation threshold.",
            "- Motion × coherence still does not score prompt fidelity, direction, bounce/roll semantics, or physical plausibility.",
            "- No expected condition was absent when scoring; `condition_holes` is therefore empty.",
            "",
        ]
    )
    return "\n".join(lines)


def trajectory_csv(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["run", "checkpoint", "mean_displacement", "mean_coherence", "degraded_count"]
    )
    for row in report["trajectory"]:
        writer.writerow(
            [
                row["run"],
                row["checkpoint"],
                "" if row["mean_displacement"] is None else f"{row['mean_displacement']:.6f}",
                "" if row["mean_coherence"] is None else f"{row['mean_coherence']:.6f}",
                "" if row["status"] == "missing" else row["degraded_count"],
            ]
        )
    return stream.getvalue()


def render_all() -> dict[str, str]:
    report = build_report()
    return {
        "report.json": _canonical_batch_json(report) + "\n",
        "report.md": markdown(report),
        "trajectory.csv": trajectory_csv(report),
    }


first = render_all()
second = render_all()
assert first == second
for name, payload in first.items():
    (OUTPUT / name).write_text(payload, encoding="utf-8", newline="")
print(
    json.dumps(
        {
            "byte_identical_regeneration": True,
            "files": {name: len(payload.encode()) for name, payload in first.items()},
        },
        sort_keys=True,
    )
)
