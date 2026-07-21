from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from bench.quality_sweep import (
    ProtocolError,
    aggregate_ratings,
    audit_media,
    build_blind_plan,
    build_selection_lock,
    build_selection_report,
    causal_forward_count,
    canonical_sha256,
    evaluate_gate,
    unblind_ratings,
    validate_protocol,
    validate_run_manifest,
    validate_selection_report,
    validate_selection_lock,
)
from bench.tests.test_quality_sweep import BLIND_SECRET, make_manifest, make_protocol


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def contract_protocol() -> dict:
    protocol = make_protocol()
    protocol["baseline"].update(
        {
            "commit": "8" * 40,
            "source_diff_sha256": SHA_A,
            "runner": "bench/reference/runner.py",
            "runner_sha256": SHA_B,
            "measured_artifact": "bench/results/metrics.json",
            "measured_artifact_sha256": SHA_C,
            "config_sha256": "d" * 64,
        }
    )
    protocol["reference_systems"] = {
        "sf4-reference": {
            "repository": "example/self-forcing",
            "commit": "7" * 40,
            "source_diff_sha256": "6" * 64,
            "checkpoint": "sf4.pt",
            "checkpoint_sha256": "e" * 64,
            "decoder": "Wan",
            "decoder_mode": "stock-wan",
            "runner_sha256": "f" * 64,
            "runner_status": "confirmatory-ready",
            "config_sha256": "1" * 64,
            "decoder_sha256": protocol["baseline"]["decoder_sha256"],
            "forwards": 35,
        }
    }
    protocol["development"].update(
        {
            "systems": ["cf1-baseline", "candidate-a"],
            "round_a_seed_indexes": [0],
            "round_b_seed_indexes": [0, 1],
        }
    )
    protocol["confirmatory"].update(
        {
            "systems": [
                "cf1-baseline",
                "$selection_lock.finalist_system_id",
                "sf4-reference",
            ],
            "selection_lock_schema": "bench/schemas/quality-selection-lock-v1.schema.json",
            "long_horizon_prompt_ids": ["sentinel-a", "sentinel-b"],
            "long_horizon_latent_frames": 241,
            "long_horizon_rgb_frames": 961,
        }
    )
    for prompt in protocol["prompts"]:
        if prompt["id"] == "sentinel-a":
            prompt.update(stratum="long_horizon_motion", text="Sentinel A", seeds=[9001])
        if prompt["id"] == "sentinel-b":
            prompt.update(stratum="long_horizon_geometry", text="Sentinel B", seeds=[9002])
    protocol["long_horizon_media_contract"] = {
        "width": 832,
        "height": 480,
        "fps": 16,
        "decoded_frames": 961,
        "duration_tolerance_frames": 1,
        "codec": "h264",
        "pixel_format": "yuv420p",
    }
    protocol["evaluation"].update(
        {
            "rubric": "bench/quality/rubric-v1.md",
            "rubric_sha256": "2" * 64,
        }
    )
    protocol["gates"].update(
        {
            "stratum_floor": 6.0,
            "maximum_low_quality_items": 1,
            "low_quality_item_threshold": 4.0,
            "final_third_temporal_floor": 6.5,
            "maximum_early_to_late_drop": 1.0,
            "minimum_each_warm_trial_fps": 29.0,
            "maximum_first_visible_rgb_s": 0.267,
            "maximum_p95_effective_frame_interval_ms": 39.5,
            "required_forwards": 45,
            "required_rgb_frames": 81,
        }
    )
    protocol["provenance_requirements"] = [
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
    ]
    return protocol


def selection_lock(protocol: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "locked",
        "protocol_sha256": canonical_sha256(protocol),
        "finalist_system_id": "candidate-a",
        "finalist_config_sha256": "3" * 64,
        "selection_report_sha256": "4" * 64,
        "locked_at": "2026-07-19T01:00:00Z",
    }


def confirmatory_manifest(protocol: dict, lock: dict) -> dict:
    systems = ["cf1-baseline", lock["finalist_system_id"], "sf4-reference"]
    runs = []
    for system_id in systems:
        pin = (
            protocol["reference_systems"]["sf4-reference"]
            if system_id == "sf4-reference"
            else protocol["baseline"]
        )
        runs.append(
            {
                "run_id": f"run-{system_id}",
                "system_id": system_id,
                "source_commit": pin["commit"],
                "source_diff_sha256": pin.get("source_diff_sha256", "6" * 64),
                "checkpoint_revision": pin["checkpoint"],
                "checkpoint_sha256": pin["checkpoint_sha256"],
                "decoder_revision": pin.get("decoder", "Wan"),
                "decoder_sha256": pin["decoder_sha256"],
                "runner_sha256": pin["runner_sha256"],
                "config_sha256": (
                    lock["finalist_config_sha256"]
                    if system_id == lock["finalist_system_id"]
                    else pin["config_sha256"]
                ),
                "rubric_sha256": protocol["evaluation"]["rubric_sha256"],
                "hardware": {
                    "gpu_model": "H100",
                    "gpu_uuid": f"GPU-{system_id}",
                    "driver_version": "570.00",
                    "cuda_version": "12.8",
                },
                "software": {
                    "python_version": "3.12",
                    "torch_version": "2.8.0",
                    "environment_lock_sha256": "7" * 64,
                },
                "determinism": {
                    "torch_deterministic_algorithms": True,
                    "cudnn_benchmark": False,
                    "allow_tf32": False,
                },
                "encoding": {
                    "command": "ffmpeg -c:v libx264 -pix_fmt yuv420p",
                    "codec": "h264",
                    "pixel_format": "yuv420p",
                },
            }
        )
    records = []
    prompt_by_id = {item["id"]: item for item in protocol["prompts"]}
    for prompt_id in protocol["confirmatory"]["prompt_ids"]:
        prompt = prompt_by_id[prompt_id]
        for seed in prompt["seeds"]:
            initial_hash = hashlib.sha256(f"noise:{prompt_id}:{seed}".encode()).hexdigest()
            for system_id in systems:
                artifact_id = f"{system_id}:{prompt_id}:{seed}"
                records.append(
                    {
                        "artifact_id": artifact_id,
                        "run_id": f"run-{system_id}",
                        "system_id": system_id,
                        "prompt_id": prompt_id,
                        "split": "confirmatory",
                        "prompt_utf8_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                        "effective_prompt_utf8_sha256": hashlib.sha256(
                            (
                                prompt["text"]
                                + next(
                                    (
                                        item["changes"].get("prompt_suffix", "")
                                        for item in protocol["repair_candidates"]
                                        if item["system_id"] == system_id
                                    ),
                                    "",
                                )
                            ).encode()
                        ).hexdigest(),
                        "seed": seed,
                        "rng_algorithm": "torch.Philox",
                        "rng_device": "cuda",
                        "seed_application_point": "pre-generation torch.Generator",
                        "initial_noise_sha256": initial_hash,
                        "input_noise_sha256": hashlib.sha256(
                            f"input:{artifact_id}".encode()
                        ).hexdigest(),
                        "latent_sha256": hashlib.sha256(f"latent:{artifact_id}".encode()).hexdigest(),
                        "source_file": f"media/{artifact_id}.mp4",
                        "media_sha256": hashlib.sha256(f"media:{artifact_id}".encode()).hexdigest(),
                        "runner_sha256": next(
                            run["runner_sha256"]
                            for run in runs
                            if run["system_id"] == system_id
                        ),
                        "decoder_mode": (
                            "stock-wan"
                            if system_id == "sf4-reference"
                            else "rolling-three-latent"
                        ),
                        "forwards": 45 if system_id != "sf4-reference" else 35,
                        "decoded_frames": 81,
                        "media_contract_id": "short",
                    }
                )
    return {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "selection_lock_sha256": canonical_sha256(lock),
        "scope": {
            "phase": "confirmatory",
            "cases": [
                {"prompt_id": prompt_id, "seed": seed}
                for prompt_id in protocol["confirmatory"]["prompt_ids"]
                for seed in prompt_by_id[prompt_id]["seeds"]
            ],
            "system_ids": systems,
        },
        "runs": runs,
        "records": records,
    }


def complete_ratings(
    protocol: dict,
    manifest: dict,
    lock: dict | None,
    *,
    scores_by_system: dict[str, float] | None = None,
    scores_by_family_system: dict[tuple[str, str], float] | None = None,
    raw_evidence_reports: list[dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    system_by_artifact = {
        record["artifact_id"]: record["system_id"]
        for record in manifest["records"]
    }
    for family in protocol["evaluation"]["families"]:
        for rater_id in family["rater_ids"]:
            for pass_id in range(1, family["passes"] + 1):
                public, key = build_blind_plan(
                    protocol,
                    manifest,
                    family["id"],
                    pass_id,
                    BLIND_SECRET,
                    selection_lock=lock,
                    rater_id=rater_id,
                )
                artifact_by_blind = {
                    record["blind_id"]: record["artifact_id"]
                    for record in key["records"]
                }
                raw = []
                for record in public["records"]:
                    system_id = system_by_artifact[
                        artifact_by_blind[record["blind_id"]]
                    ]
                    score = (scores_by_family_system or {}).get(
                        (family["id"], system_id),
                        (scores_by_system or {}).get(system_id, 8),
                    )
                    raw.append(
                        {
                            "blind_id": record["blind_id"],
                            "media_sha256": record["media_sha256"],
                            "scores": {
                                name: score
                                for name in protocol["evaluation"]["dimensions"]
                            },
                            "first_third_quality": score,
                            "final_third_quality": score,
                            "failure_tags": [],
                            "rationale": "coherent",
                        }
                    )
                raw_evidence_report = {
                    "schema_version": 1,
                    "provider": family["evidence_provider"],
                    "model_id": family["model_id"],
                    "protocol_sha256": canonical_sha256(protocol),
                    "manifest_sha256": canonical_sha256(manifest),
                    "blind_plan_sha256": canonical_sha256(public),
                    "family_id": family["id"],
                    "rater_id": rater_id,
                    "pass_id": pass_id,
                    "records": [
                        {
                            "blind_id": row["blind_id"],
                            "media_sha256": row["media_sha256"],
                            "raw_response_sha256": canonical_sha256(row),
                        }
                        for row in raw
                    ],
                }
                if raw_evidence_reports is not None:
                    raw_evidence_reports.append(raw_evidence_report)
                rows.extend(
                    unblind_ratings(
                        protocol,
                        manifest,
                        public,
                        key,
                        raw,
                        blind_secret=BLIND_SECRET,
                        family_id=family["id"],
                        pass_id=pass_id,
                        rater_id=rater_id,
                        raw_evidence_report=raw_evidence_report,
                        selection_lock=lock,
                    )
                )
    return rows


def selection_protocol() -> dict:
    protocol = contract_protocol()
    protocol["development"].update(
        {
            "selection_min_delta": 0.75,
            "temporal_motion_min_delta": 0.75,
            "max_single_prompt_drop": 1.0,
            "minimum_screening_fps": 29.5,
            "max_finalists": 1,
            "round_b": {
                "required_channel_mean": 7.0,
                "minimum_delta_vs_baseline": 1.0,
                "minimum_item_score": 5.0,
                "minimum_each_warm_trial_fps": 29.0,
                "maximum_first_visible_rgb_s": 0.267,
                "maximum_p95_effective_frame_interval_ms": 39.5,
            },
        }
    )
    return protocol


def _raw_timing(decoded_frames: int, e2e_fps: float) -> dict:
    wall_started_ns = 1_000_000_000
    wall_finished_ns = wall_started_ns + round(
        decoded_frames / e2e_fps * 1_000_000_000
    )
    interval_count = decoded_frames - 1
    first_ready_ns = wall_started_ns + 250_000_000
    p95_index = math.ceil(0.95 * interval_count) - 1
    intervals_ns = [30_000_000] * p95_index + [38_000_000] * (
        interval_count - p95_index
    )
    rgb_ready_ns = [first_ready_ns]
    for interval_ns in intervals_ns:
        rgb_ready_ns.append(rgb_ready_ns[-1] + interval_ns)
    if rgb_ready_ns[-1] > wall_finished_ns:
        raise AssertionError("test timestamps exceed the measured wall interval")
    return {
        "wall_started_ns": wall_started_ns,
        "wall_finished_ns": wall_finished_ns,
        "rgb_ready_ns": rgb_ready_ns,
        "e2e_fps": e2e_fps,
        "first_visible_rgb_s": 0.25,
        "p95_effective_frame_interval_ms": 38.0,
    }


def development_performance(
    protocol: dict,
    round_a_manifest: dict,
    round_b_manifest: dict,
) -> dict:
    def trial(record: dict, run: dict, trial_id: str) -> dict:
        return {
            "trial_id": trial_id,
            "artifact_id": record["artifact_id"],
            "run_id": record["run_id"],
            "system_id": record["system_id"],
            "prompt_id": record["prompt_id"],
            "seed": record["seed"],
            "media_sha256": record["media_sha256"],
            "config_sha256": run["config_sha256"],
            **_raw_timing(81, 30.0),
            "forwards": 45,
            "decoded_frames": 81,
        }

    round_a_runs = {run["system_id"]: run for run in round_a_manifest["runs"]}
    round_b_runs = {run["system_id"]: run for run in round_b_manifest["runs"]}
    round_a_by_system: dict[str, list[dict]] = {}
    round_b_by_system: dict[str, list[dict]] = {}
    for record in round_a_manifest["records"]:
        round_a_by_system.setdefault(record["system_id"], []).append(record)
    for record in round_b_manifest["records"]:
        round_b_by_system.setdefault(record["system_id"], []).append(record)
    evidence = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": canonical_sha256(protocol),
        "round_a_manifest_sha256": canonical_sha256(round_a_manifest),
        "round_b_manifest_sha256": canonical_sha256(round_b_manifest),
        "round_a_trials": [
            trial(records[0], round_a_runs[system_id], f"round-a-{system_id}")
            for system_id, records in round_a_by_system.items()
        ],
        "round_b_trials": [
            trial(record, round_b_runs[system_id], f"round-b-{system_id}-{index}")
            for system_id, records in round_b_by_system.items()
            for index, record in enumerate(records[:3], start=1)
        ],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def complete_selection_evidence() -> dict:
    protocol = selection_protocol()
    round_a_manifest = make_manifest(protocol, phase="development-round-a")
    round_b_manifest = make_manifest(protocol, phase="development-round-b")
    round_a_raw_evidence_reports: list[dict] = []
    round_a_ratings = complete_ratings(
        protocol,
        round_a_manifest,
        None,
        scores_by_system={"cf1-baseline": 7.0, "candidate-a": 8.0},
        raw_evidence_reports=round_a_raw_evidence_reports,
    )
    round_b_raw_evidence_reports: list[dict] = []
    round_b_ratings = complete_ratings(
        protocol,
        round_b_manifest,
        None,
        scores_by_system={"cf1-baseline": 7.0, "candidate-a": 8.25},
        raw_evidence_reports=round_b_raw_evidence_reports,
    )
    performance = development_performance(
        protocol, round_a_manifest, round_b_manifest
    )
    return {
        "protocol": protocol,
        "round_a_manifest": round_a_manifest,
        "round_a_ratings": round_a_ratings,
        "round_a_raw_evidence_reports": round_a_raw_evidence_reports,
        "round_b_manifest": round_b_manifest,
        "round_b_ratings": round_b_ratings,
        "round_b_raw_evidence_reports": round_b_raw_evidence_reports,
        "development_performance": performance,
    }


def recompute_selection_report(evidence: dict) -> dict:
    return build_selection_report(
        evidence["protocol"],
        round_a_manifest=evidence["round_a_manifest"],
        round_a_ratings=evidence["round_a_ratings"],
        round_a_raw_evidence_reports=evidence["round_a_raw_evidence_reports"],
        round_b_manifest=evidence["round_b_manifest"],
        round_b_ratings=evidence["round_b_ratings"],
        round_b_raw_evidence_reports=evidence["round_b_raw_evidence_reports"],
        development_performance=evidence["development_performance"],
        blind_secret=BLIND_SECRET,
    )


def sentinel_manifest(protocol: dict, lock: dict) -> dict:
    source = confirmatory_manifest(protocol, lock)
    selected_run = copy.deepcopy(
        next(run for run in source["runs"] if run["system_id"] == lock["finalist_system_id"])
    )
    prompt_by_id = {item["id"]: item for item in protocol["prompts"]}
    records = []
    for prompt_id in protocol["confirmatory"]["long_horizon_prompt_ids"]:
        prompt = prompt_by_id[prompt_id]
        for seed in prompt["seeds"]:
            artifact_id = f"sentinel-{prompt_id}-{seed}"
            records.append(
                {
                    "artifact_id": artifact_id,
                    "run_id": selected_run["run_id"],
                    "system_id": lock["finalist_system_id"],
                    "prompt_id": prompt_id,
                    "split": "sentinel",
                    "prompt_utf8_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                    "effective_prompt_utf8_sha256": hashlib.sha256(
                        (
                            prompt["text"]
                            + next(
                                (
                                    item["changes"].get("prompt_suffix", "")
                                    for item in protocol["repair_candidates"]
                                    if item["system_id"] == lock["finalist_system_id"]
                                ),
                                "",
                            )
                        ).encode()
                    ).hexdigest(),
                    "seed": seed,
                    "rng_algorithm": "torch.Philox",
                    "rng_device": "cuda",
                    "seed_application_point": "pre-generation",
                    "initial_noise_sha256": hashlib.sha256(
                        f"noise:{prompt_id}:{seed}".encode()
                    ).hexdigest(),
                    "input_noise_sha256": hashlib.sha256(
                        f"input:{artifact_id}".encode()
                    ).hexdigest(),
                    "latent_sha256": hashlib.sha256(
                        f"latent:{artifact_id}".encode()
                    ).hexdigest(),
                    "source_file": f"{artifact_id}.mp4",
                    "media_sha256": hashlib.sha256(
                        f"media:{artifact_id}".encode()
                    ).hexdigest(),
                    "runner_sha256": selected_run["runner_sha256"],
                    "decoder_mode": "rolling-three-latent",
                    "forwards": 485,
                    "decoded_frames": 961,
                    "media_contract_id": "long",
                }
            )
    return {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "selection_lock_sha256": canonical_sha256(lock),
        "scope": {
            "phase": "sentinel",
            "cases": [
                {"prompt_id": record["prompt_id"], "seed": record["seed"]}
                for record in records
            ],
            "system_ids": [lock["finalist_system_id"]],
        },
        "runs": [selected_run],
        "records": records,
    }


def bound_media_report(protocol: dict, manifest: dict) -> dict:
    """Build the expected report; gate tests separately re-audit physical bytes."""

    contract = (
        protocol["long_horizon_media_contract"]
        if manifest["scope"]["phase"] == "sentinel"
        else protocol["media_contract"]
    )
    report = {
        "ok": True,
        "protocol_sha256": canonical_sha256(protocol),
        "manifest_sha256": canonical_sha256(manifest),
        "media_contract": contract,
        "records": [
            {
                "artifact_id": record["artifact_id"],
                "source_file": record["source_file"],
                "media_sha256": record["media_sha256"],
                "width": contract["width"],
                "height": contract["height"],
                "fps": float(contract["fps"]),
                "decoded_frames": contract["decoded_frames"],
                "duration_s": contract["decoded_frames"] / contract["fps"],
                "codec": contract["codec"],
                "pixel_format": contract["pixel_format"],
                "video_streams": 1,
                "audio_streams": 0,
            }
            for record in manifest["records"]
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def fake_media_probe(protocol: dict):
    def probe(path: Path) -> dict:
        contract = (
            protocol["long_horizon_media_contract"]
            if path.parent.name == "sentinel"
            else protocol["media_contract"]
        )
        return {
            "width": contract["width"],
            "height": contract["height"],
            "fps": float(contract["fps"]),
            "decoded_frames": contract["decoded_frames"],
            "duration_s": contract["decoded_frames"] / contract["fps"],
            "codec": contract["codec"],
            "pixel_format": contract["pixel_format"],
            "video_streams": 1,
            "audio_streams": 0,
        }

    return probe


def materialize_manifest_media(
    testcase: unittest.TestCase,
    protocol: dict,
    manifest: dict,
    probe,
) -> tuple[list[dict], dict]:
    temporary = tempfile.TemporaryDirectory()
    testcase.addCleanup(temporary.cleanup)
    phase = manifest["scope"]["phase"]
    directory = Path(temporary.name) / phase
    directory.mkdir()
    records = []
    for index, record in enumerate(manifest["records"]):
        physical_path = directory / f"{index:03d}.mp4"
        payload = next(
            (
                candidate
                for candidate in (
                    f"media:{record['artifact_id']}".encode(),
                    record["source_file"].encode(),
                )
                if hashlib.sha256(candidate).hexdigest() == record["media_sha256"]
            ),
            None,
        )
        if payload is None:
            raise AssertionError(
                f"test manifest media hash is not materializable for {record['artifact_id']}"
            )
        physical_path.write_bytes(payload)
        records.append({**record, "physical_source_file": str(physical_path)})
    contract = (
        protocol["long_horizon_media_contract"]
        if phase == "sentinel"
        else protocol["media_contract"]
    )
    report = audit_media(
        records,
        contract,
        probe=probe,
        protocol_sha256=canonical_sha256(protocol),
        manifest_sha256=canonical_sha256(manifest),
    )
    return records, report


def rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = canonical_sha256(value)


def passing_performance(
    protocol: dict,
    lock: dict,
    manifest: dict,
    sentinels: dict,
    confirmatory_media: dict,
    sentinel_media: dict,
) -> dict:
    def timing(decoded_frames: int, e2e_fps: float) -> dict:
        wall_started_ns = 1_000_000_000
        wall_finished_ns = wall_started_ns + round(
            decoded_frames / e2e_fps * 1_000_000_000
        )
        interval_count = decoded_frames - 1
        first_ready_ns = wall_started_ns + 250_000_000
        p95_index = math.ceil(0.95 * interval_count) - 1
        intervals_ns = [30_000_000] * p95_index + [38_000_000] * (
            interval_count - p95_index
        )
        rgb_ready_ns = [first_ready_ns]
        for interval_ns in intervals_ns:
            rgb_ready_ns.append(rgb_ready_ns[-1] + interval_ns)
        if rgb_ready_ns[-1] > wall_finished_ns:
            raise AssertionError("test timestamps exceed the measured wall interval")
        return {
            "wall_started_ns": wall_started_ns,
            "wall_finished_ns": wall_finished_ns,
            "rgb_ready_ns": rgb_ready_ns,
            "e2e_fps": e2e_fps,
            "first_visible_rgb_s": 0.25,
            "p95_effective_frame_interval_ms": 38.0,
        }

    selected_run = next(
        run for run in manifest["runs"] if run["system_id"] == lock["finalist_system_id"]
    )
    selected_records = [
        record
        for record in manifest["records"]
        if record["system_id"] == lock["finalist_system_id"]
    ]
    if len(selected_records) < 4:
        raise AssertionError("performance fixture requires four selected short artifacts")

    def short_trial(record: dict, trial_id: str, e2e_fps: float) -> dict:
        return {
            "trial_id": trial_id,
            "artifact_id": record["artifact_id"],
            "run_id": record["run_id"],
            "prompt_id": record["prompt_id"],
            "seed": record["seed"],
            "media_sha256": record["media_sha256"],
            **timing(81, e2e_fps),
            "forwards": 45,
            "decoded_frames": 81,
        }

    sentinel_run = sentinels["runs"][0]
    performance = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": canonical_sha256(protocol),
        "selected_system": "candidate-a",
        "selection_lock_sha256": canonical_sha256(lock),
        "confirmatory_manifest_sha256": canonical_sha256(manifest),
        "confirmatory_media_report_sha256": confirmatory_media["report_sha256"],
        "sentinel_manifest_sha256": canonical_sha256(sentinels),
        "sentinel_media_report_sha256": sentinel_media["report_sha256"],
        "selected_run_sha256": canonical_sha256(selected_run),
        "sentinel_run_sha256": canonical_sha256(sentinel_run),
        "cold_trial": short_trial(selected_records[0], "cold-1", 25.0),
        "warm_trials": [
            short_trial(record, f"warm-{index}", 30.0)
            for index, record in enumerate(selected_records[1:4], start=1)
        ],
        "sentinel_trials": [
            {
                "trial_id": f"long-{index}",
                "artifact_id": record["artifact_id"],
                "run_id": record["run_id"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "media_sha256": record["media_sha256"],
                "media_duration_s": 60.0625,
                **timing(961, 25.0),
                "forwards": 485,
                "decoded_frames": 961,
            }
            for index, record in enumerate(sentinels["records"], start=1)
        ],
    }
    performance["evidence_sha256"] = canonical_sha256(performance)
    return performance


def passing_gate_fixture(testcase: unittest.TestCase) -> dict:
    selection_evidence = complete_selection_evidence()
    protocol = selection_evidence["protocol"]
    selection_report = recompute_selection_report(selection_evidence)
    lock = build_selection_lock(
        protocol,
        selection_report,
        locked_at="2026-07-19T02:00:00Z",
    )
    manifest = confirmatory_manifest(protocol, lock)
    raw_evidence_reports: list[dict] = []
    ratings = complete_ratings(
        protocol,
        manifest,
        lock,
        raw_evidence_reports=raw_evidence_reports,
    )
    report = aggregate_ratings(
        protocol,
        ratings,
        manifest=manifest,
        selection_lock=lock,
        blind_secret=BLIND_SECRET,
        raw_evidence_reports=raw_evidence_reports,
        require_complete=True,
    )
    sentinels = sentinel_manifest(protocol, lock)
    probe = fake_media_probe(protocol)
    media_records, media_report = materialize_manifest_media(
        testcase, protocol, manifest, probe
    )
    sentinel_media_records, sentinel_media_report = materialize_manifest_media(
        testcase, protocol, sentinels, probe
    )
    performance = passing_performance(
        protocol,
        lock,
        manifest,
        sentinels,
        media_report,
        sentinel_media_report,
    )
    return {
        "protocol": protocol,
        "selection_lock": lock,
        "selection_report": selection_report,
        "round_a_manifest": selection_evidence["round_a_manifest"],
        "round_a_ratings": selection_evidence["round_a_ratings"],
        "round_a_raw_evidence_reports": selection_evidence[
            "round_a_raw_evidence_reports"
        ],
        "round_b_manifest": selection_evidence["round_b_manifest"],
        "round_b_ratings": selection_evidence["round_b_ratings"],
        "round_b_raw_evidence_reports": selection_evidence[
            "round_b_raw_evidence_reports"
        ],
        "development_performance": selection_evidence[
            "development_performance"
        ],
        "manifest": manifest,
        "ratings": ratings,
        "raw_evidence_reports": raw_evidence_reports,
        "report": report,
        "sentinel_manifest": sentinels,
        "media_records": media_records,
        "media_report": media_report,
        "sentinel_media_records": sentinel_media_records,
        "sentinel_media_report": sentinel_media_report,
        "media_probe": probe,
        "performance": performance,
    }


def evaluate_fixture(fixture: dict, **overrides) -> dict:
    arguments = {
        "report": fixture["report"],
        "selected_system": "candidate-a",
        "reference_system": "sf4-reference",
        "performance": fixture["performance"],
        "media_report": fixture["media_report"],
        "manifest": fixture["manifest"],
        "selection_lock": fixture["selection_lock"],
        "sentinel_manifest": fixture["sentinel_manifest"],
        "sentinel_media_report": fixture["sentinel_media_report"],
        "ratings": fixture["ratings"],
        "raw_evidence_reports": fixture["raw_evidence_reports"],
        "blind_secret": BLIND_SECRET,
        "media_records": fixture["media_records"],
        "sentinel_media_records": fixture["sentinel_media_records"],
        "media_probe": fixture["media_probe"],
        "selection_report": fixture["selection_report"],
        "round_a_manifest": fixture["round_a_manifest"],
        "round_a_ratings": fixture["round_a_ratings"],
        "round_a_raw_evidence_reports": fixture[
            "round_a_raw_evidence_reports"
        ],
        "round_b_manifest": fixture["round_b_manifest"],
        "round_b_ratings": fixture["round_b_ratings"],
        "round_b_raw_evidence_reports": fixture[
            "round_b_raw_evidence_reports"
        ],
        "development_performance": fixture["development_performance"],
    }
    arguments.update(overrides)
    report = arguments.pop("report")
    return evaluate_gate(fixture["protocol"], report, **arguments)


class ProtocolContractTests(unittest.TestCase):
    def test_forward_count_formula_covers_short_reference_and_sentinel(self) -> None:
        self.assertEqual(causal_forward_count(21, 1, 4, 1), 45)
        self.assertEqual(causal_forward_count(21, 3, 4, 4), 35)
        self.assertEqual(causal_forward_count(241, 1, 4, 1), 485)

    def test_frozen_protocol_pins_all_artifacts_and_declares_only_supported_gates(self) -> None:
        validate_protocol(contract_protocol(), require_frozen=True)

        protocol = contract_protocol()
        protocol["gates"]["silently_ignored_gate"] = 1
        with self.assertRaisesRegex(ProtocolError, "unsupported gate"):
            validate_protocol(protocol, require_frozen=True)

        protocol = contract_protocol()
        protocol["baseline"]["runner_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ProtocolError, "runner_sha256"):
            validate_protocol(protocol, require_frozen=True)

        protocol = contract_protocol()
        protocol["reference_systems"]["sf4-reference"]["forwards"] = 1
        with self.assertRaisesRegex(ProtocolError, "exactly 35 forwards"):
            validate_protocol(protocol, require_frozen=True)

    def test_selection_lock_is_protocol_bound_and_uses_registered_candidate(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        validate_selection_lock(protocol, lock)
        lock["finalist_system_id"] = "post-hoc-candidate"
        with self.assertRaisesRegex(ProtocolError, "registered repair candidate"):
            validate_selection_lock(protocol, lock)

    def test_selection_lock_must_match_registered_candidate_config(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        lock["finalist_config_sha256"] = "9" * 64
        with self.assertRaisesRegex(ProtocolError, "registered candidate config"):
            validate_selection_lock(protocol, lock)


class SelectionReportContractTests(unittest.TestCase):
    def test_selection_report_recomputes_rounds_and_builds_bound_lock(self) -> None:
        evidence = complete_selection_evidence()
        report = recompute_selection_report(evidence)
        repeated = recompute_selection_report(evidence)

        self.assertEqual(report, repeated)
        self.assertEqual(report["round_a"]["survivors"], ["candidate-a"])
        self.assertEqual(report["finalist_system_id"], "candidate-a")
        self.assertEqual(
            report["finalist_config_sha256"],
            evidence["protocol"]["repair_candidates"][0]["config_sha256"],
        )
        lock = build_selection_lock(
            evidence["protocol"],
            report,
            locked_at="2026-07-19T02:00:00Z",
        )
        self.assertEqual(lock["selection_report_sha256"], report["report_sha256"])
        self.assertEqual(lock["finalist_system_id"], report["finalist_system_id"])
        validate_selection_lock(
            evidence["protocol"], lock, selection_report=report
        )

    def test_selection_ties_use_registered_deterministic_system_id_order(self) -> None:
        protocol = selection_protocol()
        protocol["repair_candidates"].append(
            {
                "system_id": "candidate-b",
                "config_sha256": "9" * 64,
                "changes": {"axis": "second-control", "value": 2},
            }
        )
        protocol["development"]["systems"].append("candidate-b")
        protocol["development"]["max_finalists"] = 2
        round_a_manifest = make_manifest(protocol, phase="development-round-a")
        round_b_manifest = make_manifest(
            protocol,
            systems=["cf1-baseline", "candidate-a", "candidate-b"],
            phase="development-round-b",
        )
        round_a_raw_evidence_reports: list[dict] = []
        round_a_ratings = complete_ratings(
            protocol,
            round_a_manifest,
            None,
            scores_by_system={
                "cf1-baseline": 7.0,
                "candidate-a": 8.0,
                "candidate-b": 8.0,
            },
            raw_evidence_reports=round_a_raw_evidence_reports,
        )
        round_b_raw_evidence_reports: list[dict] = []
        round_b_ratings = complete_ratings(
            protocol,
            round_b_manifest,
            None,
            scores_by_system={
                "cf1-baseline": 7.0,
                "candidate-a": 8.25,
                "candidate-b": 8.25,
            },
            raw_evidence_reports=round_b_raw_evidence_reports,
        )
        evidence = {
            "protocol": protocol,
            "round_a_manifest": round_a_manifest,
            "round_a_ratings": round_a_ratings,
            "round_a_raw_evidence_reports": round_a_raw_evidence_reports,
            "round_b_manifest": round_b_manifest,
            "round_b_ratings": round_b_ratings,
            "round_b_raw_evidence_reports": round_b_raw_evidence_reports,
            "development_performance": development_performance(
                protocol, round_a_manifest, round_b_manifest
            ),
        }
        report = recompute_selection_report(evidence)
        self.assertEqual(report["round_a"]["survivors"], ["candidate-a", "candidate-b"])
        self.assertEqual(report["finalist_system_id"], "candidate-a")

    def test_selection_report_rejects_incomplete_or_forged_development_evidence(self) -> None:
        evidence = complete_selection_evidence()
        incomplete = copy.deepcopy(evidence)
        incomplete["round_a_ratings"].pop()
        with self.assertRaisesRegex(ProtocolError, "complete rating tensor"):
            recompute_selection_report(incomplete)

        report = recompute_selection_report(evidence)
        forged = copy.deepcopy(report)
        forged["finalist_system_id"] = "cf1-baseline"
        rehash(forged, "report_sha256")
        with self.assertRaisesRegex(ProtocolError, "selection report.*recomputed"):
            validate_selection_report(
                evidence["protocol"],
                forged,
                round_a_manifest=evidence["round_a_manifest"],
                round_a_ratings=evidence["round_a_ratings"],
                round_a_raw_evidence_reports=evidence[
                    "round_a_raw_evidence_reports"
                ],
                round_b_manifest=evidence["round_b_manifest"],
                round_b_ratings=evidence["round_b_ratings"],
                round_b_raw_evidence_reports=evidence[
                    "round_b_raw_evidence_reports"
                ],
                development_performance=evidence["development_performance"],
                blind_secret=BLIND_SECRET,
            )

    def test_selection_report_requires_exact_round_evidence_documents(self) -> None:
        evidence = complete_selection_evidence()
        report = recompute_selection_report(evidence)

        missing_round_a = copy.deepcopy(evidence["round_a_raw_evidence_reports"])
        missing_round_a.pop()
        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            build_selection_report(
                evidence["protocol"],
                round_a_manifest=evidence["round_a_manifest"],
                round_a_ratings=evidence["round_a_ratings"],
                round_a_raw_evidence_reports=missing_round_a,
                round_b_manifest=evidence["round_b_manifest"],
                round_b_ratings=evidence["round_b_ratings"],
                round_b_raw_evidence_reports=evidence[
                    "round_b_raw_evidence_reports"
                ],
                development_performance=evidence["development_performance"],
                blind_secret=BLIND_SECRET,
            )

        forged_round_b = copy.deepcopy(evidence["round_b_raw_evidence_reports"])
        forged_round_b[0]["records"][0]["raw_response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            validate_selection_report(
                evidence["protocol"],
                report,
                round_a_manifest=evidence["round_a_manifest"],
                round_a_ratings=evidence["round_a_ratings"],
                round_a_raw_evidence_reports=evidence[
                    "round_a_raw_evidence_reports"
                ],
                round_b_manifest=evidence["round_b_manifest"],
                round_b_ratings=evidence["round_b_ratings"],
                round_b_raw_evidence_reports=forged_round_b,
                development_performance=evidence["development_performance"],
                blind_secret=BLIND_SECRET,
            )

        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            build_selection_report(
                evidence["protocol"],
                round_a_manifest=evidence["round_a_manifest"],
                round_a_ratings=evidence["round_a_ratings"],
                round_a_raw_evidence_reports=evidence[
                    "round_b_raw_evidence_reports"
                ],
                round_b_manifest=evidence["round_b_manifest"],
                round_b_ratings=evidence["round_b_ratings"],
                round_b_raw_evidence_reports=evidence[
                    "round_a_raw_evidence_reports"
                ],
                development_performance=evidence["development_performance"],
                blind_secret=BLIND_SECRET,
            )

    def test_selection_report_enforces_performance_and_artifact_bindings(self) -> None:
        evidence = complete_selection_evidence()
        slow = copy.deepcopy(evidence)
        candidate_trial = next(
            trial
            for trial in slow["development_performance"]["round_a_trials"]
            if trial["system_id"] == "candidate-a"
        )
        candidate_trial.update(_raw_timing(81, 29.0))
        rehash(slow["development_performance"], "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "no Round-A survivor"):
            recompute_selection_report(slow)

        tampered = copy.deepcopy(evidence)
        tampered["development_performance"]["round_b_trials"][0][
            "config_sha256"
        ] = "0" * 64
        rehash(tampered["development_performance"], "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "config_sha256.*manifest"):
            recompute_selection_report(tampered)

        inconsistent = copy.deepcopy(evidence)
        inconsistent["development_performance"]["round_b_trials"][0][
            "e2e_fps"
        ] = 29.5
        rehash(inconsistent["development_performance"], "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "e2e_fps.*raw timestamps"):
            recompute_selection_report(inconsistent)

    def test_round_b_channel_floor_means_each_independent_evaluator_family(self) -> None:
        evidence = complete_selection_evidence()
        evidence["round_b_raw_evidence_reports"] = []
        evidence["round_b_ratings"] = complete_ratings(
            evidence["protocol"],
            evidence["round_b_manifest"],
            None,
            scores_by_system={"cf1-baseline": 7.0, "candidate-a": 10.0},
            scores_by_family_system={("judge-b", "candidate-a"): 6.0},
            raw_evidence_reports=evidence["round_b_raw_evidence_reports"],
        )
        with self.assertRaisesRegex(ProtocolError, "no Round-B finalist"):
            recompute_selection_report(evidence)


class ManifestAndBlindContractTests(unittest.TestCase):
    def test_confirmatory_manifest_must_be_exact_and_noise_paired(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        validate_run_manifest(protocol, manifest, selection_lock=lock)

        missing = copy.deepcopy(manifest)
        missing["records"].pop()
        with self.assertRaisesRegex(ProtocolError, "complete grid"):
            validate_run_manifest(protocol, missing, selection_lock=lock)

        mismatch = copy.deepcopy(manifest)
        mismatch["records"][1]["initial_noise_sha256"] = "9" * 64
        with self.assertRaisesRegex(ProtocolError, "initial noise"):
            validate_run_manifest(protocol, mismatch, selection_lock=lock)

        config_mismatch = copy.deepcopy(manifest)
        config_mismatch["runs"][1]["config_sha256"] = "9" * 64
        with self.assertRaisesRegex(ProtocolError, "config_sha256"):
            validate_run_manifest(protocol, config_mismatch, selection_lock=lock)

        diff_mismatch = copy.deepcopy(manifest)
        diff_mismatch["runs"][1]["source_diff_sha256"] = "9" * 64
        with self.assertRaisesRegex(ProtocolError, "source_diff_sha256"):
            validate_run_manifest(protocol, diff_mismatch, selection_lock=lock)

        incomplete_runtime = copy.deepcopy(manifest)
        del incomplete_runtime["runs"][1]["hardware"]["gpu_uuid"]
        with self.assertRaisesRegex(ProtocolError, "hardware.gpu_uuid"):
            validate_run_manifest(protocol, incomplete_runtime, selection_lock=lock)

        prompt_mismatch = copy.deepcopy(manifest)
        prompt_mismatch["records"][1]["effective_prompt_utf8_sha256"] = "9" * 64
        with self.assertRaisesRegex(ProtocolError, "effective prompt hash"):
            validate_run_manifest(protocol, prompt_mismatch, selection_lock=lock)

        decoder_mismatch = copy.deepcopy(manifest)
        sf_record = next(
            row for row in decoder_mismatch["records"] if row["system_id"] == "sf4-reference"
        )
        sf_record["decoder_mode"] = "rolling-three-latent"
        with self.assertRaisesRegex(ProtocolError, "decoder_mode"):
            validate_run_manifest(protocol, decoder_mismatch, selection_lock=lock)

    def test_blinding_requires_frozen_protocol_and_is_not_cyclic(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        public, key = build_blind_plan(
            protocol,
            manifest,
            "judge-a",
            1,
            BLIND_SECRET,
            selection_lock=lock,
        )
        serialized = json.dumps(public, sort_keys=True)
        for forbidden in ("cf1-baseline", "candidate-a", "sf4-reference", ".mp4", '"seed"'):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(len({row["blind_id"] for row in key["records"]}), len(manifest["records"]))

        draft = copy.deepcopy(protocol)
        draft["status"] = "draft"
        draft.pop("frozen_at")
        draft_manifest = confirmatory_manifest(draft, selection_lock(draft))
        with self.assertRaisesRegex(ProtocolError, "frozen"):
            build_blind_plan(
                draft,
                draft_manifest,
                "judge-a",
                1,
                BLIND_SECRET,
                selection_lock=selection_lock(draft),
            )

    def test_unblind_is_key_joined_and_rejects_missing_or_tampered_ids(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        public, key = build_blind_plan(
            protocol, manifest, "judge-b", 1, BLIND_SECRET, selection_lock=lock
        )
        raw = [
            {
                "blind_id": row["blind_id"],
                "media_sha256": row["media_sha256"],
                "scores": {name: 8 for name in protocol["evaluation"]["dimensions"]},
                "first_third_quality": 8,
                "final_third_quality": 8,
                "failure_tags": [],
                "rationale": "coherent",
            }
            for row in public["records"]
        ]

        def evidence_for(rows: list[dict]) -> dict:
            return {
                "schema_version": 1,
                "provider": "test-fixture",
                "model_id": "model-b",
                "protocol_sha256": canonical_sha256(protocol),
                "manifest_sha256": canonical_sha256(manifest),
                "blind_plan_sha256": canonical_sha256(public),
                "family_id": "judge-b",
                "rater_id": "model-b",
                "pass_id": 1,
                "records": [
                    {
                        "blind_id": row["blind_id"],
                        "media_sha256": row["media_sha256"],
                        "raw_response_sha256": canonical_sha256(row),
                    }
                    for row in rows
                ],
            }

        normalized = unblind_ratings(
            protocol,
            manifest,
            public,
            key,
            raw,
            blind_secret=BLIND_SECRET,
            family_id="judge-b",
            pass_id=1,
            rater_id="model-b",
            raw_evidence_report=evidence_for(raw),
            selection_lock=lock,
        )
        self.assertEqual(len(normalized), len(manifest["records"]))
        self.assertTrue(all("system_id" in row for row in normalized))
        byte_mismatch = copy.deepcopy(raw)
        byte_mismatch[0]["media_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "judge-visible media"):
            unblind_ratings(
                protocol,
                manifest,
                public,
                key,
                byte_mismatch,
                blind_secret=BLIND_SECRET,
                family_id="judge-b",
                pass_id=1,
                rater_id="model-b",
                raw_evidence_report=evidence_for(byte_mismatch),
                selection_lock=lock,
            )
        with self.assertRaisesRegex(ProtocolError, "complete blind-id set"):
            unblind_ratings(
                protocol,
                manifest,
                public,
                key,
                raw[:-1],
                blind_secret=BLIND_SECRET,
                family_id="judge-b",
                pass_id=1,
                rater_id="model-b",
                raw_evidence_report=evidence_for(raw[:-1]),
                selection_lock=lock,
            )

        tampered_key = copy.deepcopy(key)
        tampered_key["records"][0]["artifact_id"], tampered_key["records"][1]["artifact_id"] = (
            tampered_key["records"][1]["artifact_id"],
            tampered_key["records"][0]["artifact_id"],
        )
        with self.assertRaisesRegex(ProtocolError, "unblinding key integrity"):
            unblind_ratings(
                protocol,
                manifest,
                public,
                tampered_key,
                raw,
                blind_secret=BLIND_SECRET,
                family_id="judge-b",
                pass_id=1,
                rater_id="model-b",
                raw_evidence_report=evidence_for(raw),
                selection_lock=lock,
            )


class AggregationAndGateContractTests(unittest.TestCase):
    def test_manifest_bound_aggregation_requires_exact_raw_evidence_documents(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        raw_evidence_reports: list[dict] = []
        ratings = complete_ratings(
            protocol,
            manifest,
            lock,
            raw_evidence_reports=raw_evidence_reports,
        )

        report = aggregate_ratings(
            protocol,
            ratings,
            manifest=manifest,
            selection_lock=lock,
            blind_secret=BLIND_SECRET,
            raw_evidence_reports=raw_evidence_reports,
            require_complete=True,
        )
        self.assertTrue(report["complete"])

        altered_ratings = copy.deepcopy(ratings)
        altered_ratings[0]["scores"]["prompt_adherence"] = 1
        with self.assertRaisesRegex(ProtocolError, "raw evidence|ratings"):
            aggregate_ratings(
                protocol,
                altered_ratings,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=raw_evidence_reports,
                require_complete=True,
            )

        reordered = copy.deepcopy(raw_evidence_reports)
        reordered[0]["records"].reverse()
        with self.assertRaisesRegex(ProtocolError, "raw[_ ]evidence|raw_ratings"):
            aggregate_ratings(
                protocol,
                ratings,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=reordered,
                require_complete=True,
            )

        with self.assertRaisesRegex(ProtocolError, "duplicate raw evidence"):
            aggregate_ratings(
                protocol,
                ratings,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=[*raw_evidence_reports, raw_evidence_reports[0]],
                require_complete=True,
            )

        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            aggregate_ratings(
                protocol,
                ratings,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=raw_evidence_reports[:-1],
                require_complete=True,
            )

        forged = copy.deepcopy(raw_evidence_reports)
        forged[0]["records"][0]["raw_response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            aggregate_ratings(
                protocol,
                ratings,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=forged,
                require_complete=True,
            )

    def test_complete_aggregation_rejects_coordinate_only_rows(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        raw_evidence_reports: list[dict] = []
        coordinate_only = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "artifact_id",
                    "blind_id",
                    "protocol_sha256",
                    "manifest_sha256",
                    "blind_plan_sha256",
                    "unblinding_key_sha256",
                    "raw_ratings_sha256",
                }
            }
            for row in complete_ratings(
                protocol,
                manifest,
                lock,
                raw_evidence_reports=raw_evidence_reports,
            )
        ]
        with self.assertRaisesRegex(ProtocolError, "artifact_id"):
            aggregate_ratings(
                protocol,
                coordinate_only,
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=raw_evidence_reports,
                require_complete=True,
            )

    def test_complete_tensor_is_required_and_preserves_item_stratum_and_segment_scores(self) -> None:
        protocol = contract_protocol()
        lock = selection_lock(protocol)
        manifest = confirmatory_manifest(protocol, lock)
        raw_evidence_reports: list[dict] = []
        ratings = complete_ratings(
            protocol,
            manifest,
            lock,
            raw_evidence_reports=raw_evidence_reports,
        )
        report = aggregate_ratings(
            protocol,
            ratings,
            manifest=manifest,
            selection_lock=lock,
            blind_secret=BLIND_SECRET,
            raw_evidence_reports=raw_evidence_reports,
            require_complete=True,
        )
        selected = report["systems"]["candidate-a"]
        self.assertTrue(report["complete"])
        self.assertEqual(selected["mean"], 8.0)
        self.assertEqual(selected["final_third_mean"], 8.0)
        self.assertEqual(selected["early_to_late_drop"], 0.0)
        self.assertEqual(len(selected["strata"]), 6)
        self.assertEqual(len(selected["items"]), 24)
        self.assertEqual(report["n_ratings"], 504)

        with self.assertRaisesRegex(ProtocolError, "complete rating tensor"):
            aggregate_ratings(
                protocol,
                ratings[:-1],
                manifest=manifest,
                selection_lock=lock,
                blind_secret=BLIND_SECRET,
                raw_evidence_reports=raw_evidence_reports,
                require_complete=True,
            )

        duplicate = copy.deepcopy(ratings[0])
        duplicate["seed"] = str(duplicate["seed"])
        with self.assertRaisesRegex(ProtocolError, "seed must be an integer"):
            aggregate_ratings(
                protocol,
                ratings + [duplicate],
                allow_unbound_legacy=True,
            )

    def test_every_declared_gate_has_a_fail_closed_check(self) -> None:
        fixture = passing_gate_fixture(self)
        passed = evaluate_fixture(fixture)
        self.assertTrue(passed["pass"], passed)
        self.assertEqual(
            passed["selection_report_sha256"],
            fixture["selection_report"]["report_sha256"],
        )
        self.assertEqual(
            set(passed["checks"]), set(fixture["protocol"]["gates"])
        )

        invalid_duration = copy.deepcopy(fixture["performance"])
        invalid_duration["sentinel_trials"][0]["media_duration_s"] = 59.0
        rehash(invalid_duration, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "media_duration_s"):
            evaluate_fixture(fixture, performance=invalid_duration)

    def test_gate_recomputes_and_rejects_forged_rehashed_aggregate(self) -> None:
        fixture = passing_gate_fixture(self)
        forged = copy.deepcopy(fixture["report"])
        forged["systems"]["candidate-a"]["mean"] = 10.0
        forged["n_ratings"] -= 1
        rehash(forged, "report_sha256")

        with self.assertRaisesRegex(
            ProtocolError, "(recomputed aggregate|aggregate report.*match)"
        ):
            evaluate_fixture(fixture, report=forged)

    def test_gate_requires_and_recomputes_actual_selection_report(self) -> None:
        fixture = passing_gate_fixture(self)
        with self.assertRaisesRegex(ProtocolError, "selection report is required"):
            evaluate_fixture(fixture, selection_report=None)

        forged = copy.deepcopy(fixture["selection_report"])
        forged["round_b"]["candidates"]["candidate-a"][
            "mean_delta_vs_baseline"
        ] = 9.0
        rehash(forged, "report_sha256")
        forged_lock = copy.deepcopy(fixture["selection_lock"])
        forged_lock["selection_report_sha256"] = forged["report_sha256"]
        with self.assertRaisesRegex(ProtocolError, "selection report.*recomputed"):
            evaluate_fixture(
                fixture,
                selection_report=forged,
                selection_lock=forged_lock,
            )

    def test_gate_requires_confirmatory_and_both_rounds_raw_evidence(self) -> None:
        fixture = passing_gate_fixture(self)

        forged_confirmatory = copy.deepcopy(fixture["raw_evidence_reports"])
        forged_confirmatory[0]["records"][0]["raw_response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            evaluate_fixture(
                fixture,
                raw_evidence_reports=forged_confirmatory,
            )

        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            evaluate_fixture(
                fixture,
                round_a_raw_evidence_reports=fixture[
                    "round_a_raw_evidence_reports"
                ][:-1],
            )

        forged_round_b = copy.deepcopy(fixture["round_b_raw_evidence_reports"])
        forged_round_b[0]["records"][0]["raw_response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "raw evidence"):
            evaluate_fixture(
                fixture,
                round_b_raw_evidence_reports=forged_round_b,
            )

    def test_gate_reaudits_media_and_rejects_forged_report_or_corrupt_bytes(self) -> None:
        fixture = passing_gate_fixture(self)
        forged_media = copy.deepcopy(fixture["media_report"])
        forged_media["records"][0]["duration_s"] += 0.01
        rehash(forged_media, "report_sha256")
        with self.assertRaisesRegex(
            ProtocolError, "submitted media report.*gate-time media audit"
        ):
            evaluate_fixture(fixture, media_report=forged_media)

        first_path = Path(fixture["media_records"][0]["physical_source_file"])
        first_path.write_bytes(b"corrupted after the report was created")
        with self.assertRaisesRegex(ProtocolError, "media sha256 mismatch"):
            evaluate_fixture(fixture)

    def test_gate_rejects_summary_metrics_inconsistent_with_raw_timestamps(self) -> None:
        fixture = passing_gate_fixture(self)
        performance = copy.deepcopy(fixture["performance"])
        performance["warm_trials"][0]["e2e_fps"] = 29.5
        rehash(performance, "evidence_sha256")
        with self.assertRaisesRegex(
            ProtocolError, "e2e_fps.*(timestamps|derived|raw)"
        ):
            evaluate_fixture(fixture, performance=performance)

    def test_gate_rejects_duplicate_or_wrong_short_artifact_bindings(self) -> None:
        fixture = passing_gate_fixture(self)
        binding_fields = (
            "artifact_id",
            "run_id",
            "prompt_id",
            "seed",
            "media_sha256",
        )

        duplicate = copy.deepcopy(fixture["performance"])
        for field in binding_fields:
            duplicate["warm_trials"][0][field] = duplicate["cold_trial"][field]
        rehash(duplicate, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "four distinct artifacts"):
            evaluate_fixture(fixture, performance=duplicate)

        wrong = copy.deepcopy(fixture["performance"])
        baseline_record = next(
            record
            for record in fixture["manifest"]["records"]
            if record["system_id"] == "cf1-baseline"
        )
        for field in binding_fields:
            wrong["cold_trial"][field] = baseline_record[field]
        rehash(wrong, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "selected-system confirmatory artifact"):
            evaluate_fixture(fixture, performance=wrong)

    def test_gate_rejects_malformed_raw_timestamps(self) -> None:
        fixture = passing_gate_fixture(self)
        malformed_cases = [
            (
                "non-integer start",
                lambda trial: trial.update(wall_started_ns=True),
                "wall timestamps.*integers",
            ),
            (
                "non-increasing ready times",
                lambda trial: trial["rgb_ready_ns"].__setitem__(
                    1, trial["rgb_ready_ns"][0]
                ),
                "rgb_ready_ns.*strictly increasing",
            ),
            (
                "wrong ready count",
                lambda trial: trial["rgb_ready_ns"].pop(),
                "rgb_ready_ns.*decoded_frames",
            ),
            (
                "single-frame trial has no cadence interval",
                lambda trial: trial.update(
                    decoded_frames=1,
                    rgb_ready_ns=[trial["rgb_ready_ns"][0]],
                ),
                "decoded_frames.*at least two",
            ),
        ]
        for label, mutate, message in malformed_cases:
            with self.subTest(case=label):
                performance = copy.deepcopy(fixture["performance"])
                mutate(performance["warm_trials"][0])
                rehash(performance, "evidence_sha256")
                with self.assertRaisesRegex(ProtocolError, message):
                    evaluate_fixture(fixture, performance=performance)

    def test_gate_rejects_missing_media_or_unbound_performance(self) -> None:
        fixture = passing_gate_fixture(self)
        with self.assertRaisesRegex(ProtocolError, "media report"):
            evaluate_fixture(fixture, media_report=None)
        performance = copy.deepcopy(fixture["performance"])
        performance["selected_run_sha256"] = "0" * 64
        rehash(performance, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "selected_run_sha256"):
            evaluate_fixture(fixture, performance=performance)

    def test_gate_rejects_identity_hash_and_development_bypasses(self) -> None:
        fixture = passing_gate_fixture(self)

        baseline_performance = copy.deepcopy(fixture["performance"])
        baseline_performance["selected_system"] = "cf1-baseline"
        rehash(baseline_performance, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "locked finalist"):
            evaluate_fixture(
                fixture,
                selected_system="cf1-baseline",
                performance=baseline_performance,
            )

        with self.assertRaisesRegex(ProtocolError, "sf4-reference"):
            evaluate_fixture(
                fixture,
                reference_system="cf1-baseline",
            )

        stale_report = copy.deepcopy(fixture["report"])
        stale_report["n_ratings"] += 1
        with self.assertRaisesRegex(ProtocolError, "aggregate report.*report_sha256"):
            evaluate_fixture(fixture, report=stale_report)

        bad_performance = copy.deepcopy(fixture["performance"])
        bad_performance["evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "evidence_sha256"):
            evaluate_fixture(fixture, performance=bad_performance)

        duration_mismatch = copy.deepcopy(fixture["performance"])
        duration_mismatch["sentinel_trials"][0]["media_duration_s"] = 60.0
        rehash(duration_mismatch, "evidence_sha256")
        with self.assertRaisesRegex(ProtocolError, "media report duration"):
            evaluate_fixture(fixture, performance=duration_mismatch)

        protocol = fixture["protocol"]
        development_manifest = make_manifest(protocol)
        development_raw_evidence_reports: list[dict] = []
        development_ratings = complete_ratings(
            protocol,
            development_manifest,
            None,
            raw_evidence_reports=development_raw_evidence_reports,
        )
        development_report = aggregate_ratings(
            protocol,
            development_ratings,
            manifest=development_manifest,
            blind_secret=BLIND_SECRET,
            raw_evidence_reports=development_raw_evidence_reports,
            require_complete=True,
        )
        development_records, development_media = materialize_manifest_media(
            self, protocol, development_manifest, fixture["media_probe"]
        )
        with self.assertRaisesRegex(ProtocolError, "confirmatory"):
            evaluate_fixture(
                fixture,
                report=development_report,
                media_report=development_media,
                manifest=development_manifest,
                ratings=development_ratings,
                media_records=development_records,
            )


if __name__ == "__main__":
    unittest.main()
