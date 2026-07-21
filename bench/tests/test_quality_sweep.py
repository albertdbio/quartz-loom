from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bench.quality_sweep import (
    ProtocolError,
    aggregate_ratings,
    audit_media,
    build_blind_plan,
    canonical_sha256,
    evaluate_gate,
    load_json,
    ratings_from_legacy_eval,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
BLIND_SECRET = bytes.fromhex("42" * 32)


def _prompt(prompt_id: str, split: str, stratum: str, seed: int) -> dict:
    return {
        "id": prompt_id,
        "split": split,
        "stratum": stratum,
        "text": f"Prompt {prompt_id}",
        "seeds": [seed, seed + 1],
    }


def make_protocol() -> dict:
    development = [
        _prompt(f"dev-{index:02d}", "development", f"s{index}", 1000 + index * 10)
        for index in range(6)
    ]
    confirmatory = [
        _prompt(f"hold-{index:02d}", "confirmatory", f"s{index // 2}", 2000 + index * 10)
        for index in range(12)
    ]
    protocol = {
        "schema_version": 1,
        "protocol_id": "test-quality-v1",
        "status": "frozen",
        "frozen_at": "2026-07-19T00:00:00Z",
        "hypothesis": "A registered repair can improve quality without changing the performance contract.",
        "baseline": {
            "system_id": "cf1-baseline",
            "repository": "thu-ml/Causal-Forcing",
            "commit": "8" * 40,
            "source_diff_sha256": "c" * 64,
            "checkpoint": "framewise-1step.pt",
            "checkpoint_sha256": "a" * 64,
            "weights": "generator_ema",
            "decoder": "TAEHV taew2_1 rolling",
            "decoder_mode": "rolling-three-latent",
            "decoder_sha256": "b" * 64,
            "runner": "runner.py",
            "runner_sha256": "d" * 64,
            "runner_status": "confirmatory-ready",
            "config_sha256": "e" * 64,
            "measured_artifact": "metrics.json",
            "measured_artifact_sha256": "f" * 64,
            "forwards": 45,
        },
        "reference_systems": {
            "sf4-reference": {
                "repository": "gdhe17/Self-Forcing",
                "commit": "7" * 40,
                "source_diff_sha256": "6" * 64,
                "checkpoint": "sf4.pt",
                "checkpoint_sha256": "1" * 64,
                "decoder": "Wan",
                "decoder_mode": "stock-wan",
                "decoder_sha256": "2" * 64,
                "runner_sha256": "3" * 64,
                "runner_status": "confirmatory-ready",
                "config_sha256": "4" * 64,
                "forwards": 35,
            }
        },
        "repair_candidates": [
            {
                "system_id": "candidate-a",
                "config_sha256": "3" * 64,
                "changes": {"axis": "supported-control", "value": 1},
            }
        ],
        "development": {
            "prompt_ids": [item["id"] for item in development],
            "systems": ["cf1-baseline", "candidate-a"],
            "round_a_seed_indexes": [0],
            "round_b_seed_indexes": [0, 1],
            "selection_min_delta": 0.75,
            "max_finalists": 1,
        },
        "confirmatory": {
            "prompt_ids": [item["id"] for item in confirmatory],
            "systems": [
                "cf1-baseline",
                "$selection_lock.finalist_system_id",
                "sf4-reference",
            ],
            "selection_lock_schema": "selection-lock.schema.json",
            "long_horizon_prompt_ids": ["sentinel-a", "sentinel-b"],
            "long_horizon_latent_frames": 241,
            "long_horizon_rgb_frames": 961,
        },
        "prompts": development
        + confirmatory
        + [
            {**_prompt("sentinel-a", "sentinel", "long-a", 9000), "seeds": [9000]},
            {**_prompt("sentinel-b", "sentinel", "long-b", 9010), "seeds": [9010]},
        ],
        "media_contract": {
            "width": 832,
            "height": 480,
            "fps": 16,
            "decoded_frames": 81,
            "duration_tolerance_frames": 1,
            "codec": "h264",
            "pixel_format": "yuv420p",
        },
        "long_horizon_media_contract": {
            "width": 832,
            "height": 480,
            "fps": 16,
            "decoded_frames": 961,
            "duration_tolerance_frames": 1,
            "codec": "h264",
            "pixel_format": "yuv420p",
        },
        "evaluation": {
            "blind_seed": 20260719,
            "rubric": "rubric.md",
            "rubric_sha256": "5" * 64,
            "dimensions": {
                "prompt_adherence": 0.2,
                "spatial_fidelity": 0.2,
                "identity_consistency": 0.2,
                "motion_naturalness": 0.2,
                "temporal_artifacts": 0.2,
            },
            "families": [
                {
                    "id": "judge-a",
                    "kind": "model",
                    "model_id": "model-a",
                    "evidence_provider": "test-fixture",
                    "readiness": "quality-qualified",
                    "passes": 2,
                    "rater_ids": ["model-a"],
                },
                {
                    "id": "judge-b",
                    "kind": "model",
                    "model_id": "model-b",
                    "evidence_provider": "test-fixture",
                    "readiness": "quality-qualified",
                    "passes": 2,
                    "rater_ids": ["model-b"],
                },
                {
                    "id": "human",
                    "kind": "human",
                    "model_id": "three-blind-raters",
                    "evidence_provider": "test-fixture",
                    "passes": 1,
                    "minimum_raters": 3,
                    "rater_ids": ["human-1", "human-2", "human-3"],
                },
            ],
        },
        "gates": {
            "absolute_quality": 7.0,
            "family_floor": 6.5,
            "stratum_floor": 6.0,
            "sf4_noninferiority_margin": -0.25,
            "maximum_low_quality_items": 1,
            "low_quality_item_threshold": 4.0,
            "final_third_temporal_floor": 6.5,
            "maximum_early_to_late_drop": 1.0,
            "warm_e2e_fps": 29.0,
            "cold_e2e_fps": 24.0,
            "sustained_seconds": 60.0,
            "sustained_e2e_fps": 24.0,
            "minimum_each_warm_trial_fps": 29.0,
            "maximum_first_visible_rgb_s": 0.267,
            "maximum_p95_effective_frame_interval_ms": 39.5,
            "required_model_families": 2,
            "require_human": True,
            "required_forwards": 45,
            "required_rgb_frames": 81,
        },
        "provenance_requirements": [
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
        ],
        "stop_rules": ["Stop on baseline drift or provenance ambiguity."],
        "sources": ["PLAN.md", "brain/current/State.md"],
    }
    return protocol


def make_manifest(
    protocol: dict,
    systems: list[str] | None = None,
    *,
    phase: str = "development-round-a",
) -> dict:
    if phase not in {"development-round-a", "development-round-b"}:
        raise ValueError(f"unsupported test development phase {phase}")
    systems = systems or list(protocol["development"]["systems"])
    runs = []
    for system in systems:
        pin = (
            protocol["reference_systems"]["sf4-reference"]
            if system == "sf4-reference"
            else (
                protocol["baseline"]
                if system == protocol["baseline"]["system_id"]
                else {
                    **protocol["baseline"],
                    "config_sha256": next(
                        item["config_sha256"]
                        for item in protocol["repair_candidates"]
                        if item["system_id"] == system
                    ),
                }
            )
        )
        runs.append(
            {
                "run_id": f"run-{system}",
                "system_id": system,
                "source_commit": pin["commit"],
                "source_diff_sha256": pin.get("source_diff_sha256", "6" * 64),
                "checkpoint_revision": pin["checkpoint"],
                "checkpoint_sha256": pin["checkpoint_sha256"],
                "decoder_revision": pin.get("decoder", "Wan"),
                "decoder_sha256": pin["decoder_sha256"],
                "runner_sha256": pin["runner_sha256"],
                "config_sha256": pin["config_sha256"],
                "rubric_sha256": protocol["evaluation"]["rubric_sha256"],
                "hardware": {
                    "gpu_model": "H100",
                    "gpu_uuid": f"GPU-{system}",
                    "driver_version": "test-driver",
                    "cuda_version": "test-cuda",
                },
                "software": {
                    "python_version": "3.12",
                    "torch_version": "test",
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
    index_field = (
        "round_a_seed_indexes"
        if phase == "development-round-a"
        else "round_b_seed_indexes"
    )
    for prompt_id in protocol["development"]["prompt_ids"]:
        prompt = prompt_by_id[prompt_id]
        for seed in [prompt["seeds"][index] for index in protocol["development"][index_field]]:
            initial_hash = hashlib.sha256(f"noise-{prompt['id']}-{seed}".encode()).hexdigest()
            for system in systems:
                name = f"{system}-{prompt['id']}-{seed}.mp4"
                records.append(
                    {
                        "artifact_id": name[:-4],
                        "run_id": f"run-{system}",
                        "system_id": system,
                        "prompt_id": prompt["id"],
                        "split": phase,
                        "prompt_utf8_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                        "effective_prompt_utf8_sha256": hashlib.sha256(
                            (
                                prompt["text"]
                                + next(
                                    (
                                        item["changes"].get("prompt_suffix", "")
                                        for item in protocol["repair_candidates"]
                                        if item["system_id"] == system
                                    ),
                                    "",
                                )
                            ).encode()
                        ).hexdigest(),
                        "seed": seed,
                        "rng_algorithm": "torch.Philox",
                        "rng_device": "cuda",
                        "seed_application_point": "pre-generation",
                        "input_noise_sha256": hashlib.sha256(f"input-{name}".encode()).hexdigest(),
                        "latent_sha256": hashlib.sha256(f"latent-{name}".encode()).hexdigest(),
                        "source_file": name,
                        "media_sha256": hashlib.sha256(name.encode()).hexdigest(),
                        "initial_noise_sha256": initial_hash,
                        "runner_sha256": next(run for run in runs if run["system_id"] == system)["runner_sha256"],
                        "decoder_mode": (
                            "stock-wan"
                            if system == "sf4-reference"
                            else "rolling-three-latent"
                        ),
                        "forwards": 35 if system == "sf4-reference" else 45,
                        "decoded_frames": 81,
                        "media_contract_id": "short",
                    }
                )
    return {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "scope": {
            "phase": phase,
            "cases": [
                {"prompt_id": prompt_id, "seed": prompt_by_id[prompt_id]["seeds"][index]}
                for prompt_id in protocol["development"]["prompt_ids"]
                for index in protocol["development"][index_field]
            ],
            "system_ids": systems,
        },
        "runs": runs,
        "records": records,
    }


class ProtocolValidationTests(unittest.TestCase):
    def test_json_loader_rejects_duplicate_object_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"stratum":"one","stratum":"two"}\n')
            with self.assertRaisesRegex(ProtocolError, "duplicate JSON key stratum"):
                load_json(path)

    def test_frozen_protocol_requires_unique_registered_raters(self) -> None:
        protocol = make_protocol()
        protocol["evaluation"]["families"][0]["rater_ids"] = ["model-a", "model-a"]
        with self.assertRaisesRegex(ProtocolError, "rater_ids"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_requires_distinct_model_ids_and_exact_roster(self) -> None:
        protocol = make_protocol()
        protocol["evaluation"]["families"][1]["model_id"] = "model-a"
        with self.assertRaisesRegex(ProtocolError, "distinct model_id"):
            validate_protocol(protocol, require_frozen=True)

        protocol = make_protocol()
        protocol["evaluation"]["families"][1]["passes"] = 1
        with self.assertRaisesRegex(ProtocolError, "exactly two passes"):
            validate_protocol(protocol, require_frozen=True)

        protocol = make_protocol()
        for prompt in protocol["prompts"]:
            if prompt["split"] == "confirmatory":
                prompt["seeds"] = prompt["seeds"][:1]
        with self.assertRaisesRegex(ProtocolError, "exactly two seeds"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_requires_quality_qualified_model_families(self) -> None:
        protocol = make_protocol()
        protocol["evaluation"]["families"][1]["readiness"] = "calibration-failed"
        with self.assertRaisesRegex(ProtocolError, "quality-qualified"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_requires_candidate_config_pin(self) -> None:
        protocol = make_protocol()
        del protocol["repair_candidates"][0]["config_sha256"]
        with self.assertRaisesRegex(ProtocolError, "candidate-a.*config_sha256"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_rejects_historical_only_runner(self) -> None:
        protocol = make_protocol()
        protocol["baseline"]["runner_status"] = "historical-only"
        with self.assertRaisesRegex(ProtocolError, "confirmatory-ready"):
            validate_protocol(protocol, require_frozen=True)

    def test_project_protocol_is_a_valid_draft(self) -> None:
        path = ROOT / "bench/quality/quality-repair-v1.protocol.json"
        protocol = load_json(path)
        validate_protocol(protocol, require_frozen=False)
        self.assertEqual(protocol["status"], "draft")
        decoder = protocol["baseline"]["decoder"]
        self.assertIn("block 0 trims 3 RGB frames", decoder)
        self.assertIn("later blocks trim prior_context_latents*4", decoder)
        self.assertIn(
            "not CPU payload readiness or browser presentation",
            protocol["baseline"]["historical_latency_caveat"],
        )

    def test_valid_protocol(self) -> None:
        validate_protocol(make_protocol(), require_frozen=True)

    def test_rejects_non_exact_confirmatory_prompt_count(self) -> None:
        protocol = make_protocol()
        protocol["confirmatory"]["prompt_ids"] = protocol["confirmatory"]["prompt_ids"][:9]
        with self.assertRaisesRegex(ProtocolError, "exactly 12"):
            validate_protocol(protocol, require_frozen=True)

    def test_rejects_duplicate_prompt_ids(self) -> None:
        protocol = make_protocol()
        protocol["prompts"][1]["id"] = protocol["prompts"][0]["id"]
        with self.assertRaisesRegex(ProtocolError, "duplicate prompt"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_rejects_placeholders(self) -> None:
        protocol = make_protocol()
        protocol["baseline"]["checkpoint_sha256"] = "TBD"
        with self.assertRaisesRegex(ProtocolError, "placeholder"):
            validate_protocol(protocol, require_frozen=True)

    def test_frozen_protocol_requires_exact_provenance(self) -> None:
        protocol = make_protocol()
        del protocol["baseline"]["checkpoint_sha256"]
        with self.assertRaisesRegex(ProtocolError, "checkpoint_sha256"):
            validate_protocol(protocol, require_frozen=True)


class BlindPlanTests(unittest.TestCase):
    def test_blinding_is_deterministic_balanced_and_publicly_opaque(self) -> None:
        protocol = make_protocol()
        manifest = make_manifest(protocol)
        public_a, key_a = build_blind_plan(
            protocol, manifest, "judge-a", 1, BLIND_SECRET
        )
        public_b, key_b = build_blind_plan(
            protocol, manifest, "judge-a", 1, BLIND_SECRET
        )
        self.assertEqual(public_a, public_b)
        self.assertEqual(key_a, key_b)

        public_text = json.dumps(public_a, sort_keys=True)
        for forbidden in ("cf1-baseline", "candidate-a", "sf4-reference", ".mp4"):
            self.assertNotIn(forbidden, public_text)

        artifact_system = {
            row["artifact_id"]: row["system_id"] for row in manifest["records"]
        }
        by_blind = {row["blind_id"]: row for row in key_a["records"]}
        slot_counts: dict[str, dict[str, int]] = {}
        for row in public_a["records"]:
            system_id = artifact_system[by_blind[row["blind_id"]]["artifact_id"]]
            slot_counts.setdefault(system_id, {}).setdefault(row["slot"], 0)
            slot_counts[system_id][row["slot"]] += 1
        slots = tuple(
            chr(ord("A") + index)
            for index in range(len(manifest["scope"]["system_ids"]))
        )
        for counts in slot_counts.values():
            values = [counts.get(slot, 0) for slot in slots]
            self.assertLessEqual(max(values) - min(values), 1)

    def test_different_pass_changes_assignment(self) -> None:
        protocol = make_protocol()
        manifest = make_manifest(protocol)
        public_a, _ = build_blind_plan(
            protocol, manifest, "judge-a", 1, BLIND_SECRET
        )
        public_b, _ = build_blind_plan(
            protocol, manifest, "judge-a", 2, BLIND_SECRET
        )
        self.assertNotEqual(public_a, public_b)

    def test_secret_is_required_and_changes_public_assignment(self) -> None:
        protocol = make_protocol()
        manifest = make_manifest(protocol)
        public_a, key_a = build_blind_plan(
            protocol, manifest, "judge-a", 1, BLIND_SECRET
        )
        public_b, key_b = build_blind_plan(
            protocol, manifest, "judge-a", 1, bytes.fromhex("43" * 32)
        )
        self.assertNotEqual(public_a["records"], public_b["records"])
        self.assertNotEqual(key_a["blind_secret_sha256"], key_b["blind_secret_sha256"])
        self.assertNotIn(BLIND_SECRET.hex(), json.dumps(public_a))
        with self.assertRaisesRegex(ProtocolError, "32 bytes"):
            build_blind_plan(protocol, manifest, "judge-a", 1, b"too short")


class MediaAuditTests(unittest.TestCase):
    def test_media_audit_reads_physical_path_but_preserves_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            physical = Path(directory) / "artifact.mp4"
            physical.write_bytes(b"captured-media")
            record = {
                "artifact_id": "artifact",
                "source_file": "media/artifact.mp4",
                "physical_source_file": str(physical),
                "media_sha256": hashlib.sha256(physical.read_bytes()).hexdigest(),
            }
            report = audit_media(
                [record],
                make_protocol()["media_contract"],
                probe=lambda _path: {
                    "width": 832,
                    "height": 480,
                    "fps": 16.0,
                    "decoded_frames": 81,
                    "duration_s": 81 / 16,
                    "codec": "h264",
                    "pixel_format": "yuv420p",
                    "video_streams": 1,
                    "audio_streams": 0,
                },
            )
            self.assertEqual(report["records"][0]["source_file"], "media/artifact.mp4")

    def test_all_current_video_artifacts_match_the_media_contract(self) -> None:
        paths = sorted((ROOT / "bench/results").glob("*.mp4"))
        self.assertEqual(len(paths), 14)
        records = [
            {"source_file": str(path), "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
        report = audit_media(records, make_protocol()["media_contract"])
        self.assertEqual(len(report["records"]), 14)

    def test_real_captured_video_passes(self) -> None:
        path = ROOT / "bench/results/cf1_p0.mp4"
        record = {
            "source_file": str(path),
            "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        report = audit_media([record], make_protocol()["media_contract"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["records"][0]["decoded_frames"], 81)

    def test_73_frame_probe_fails(self) -> None:
        path = ROOT / "bench/results/cf1_p0.mp4"
        record = {
            "source_file": str(path),
            "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with self.assertRaisesRegex(ProtocolError, "decoded_frames"):
            audit_media(
                [record],
                make_protocol()["media_contract"],
                probe=lambda _path: {
                    "width": 832,
                    "height": 480,
                    "fps": 16.0,
                    "decoded_frames": 73,
                    "duration_s": 73 / 16,
                },
            )

    def test_hash_mismatch_fails(self) -> None:
        path = ROOT / "bench/results/cf1_p0.mp4"
        with self.assertRaisesRegex(ProtocolError, "sha256"):
            audit_media(
                [{"source_file": str(path), "media_sha256": "0" * 64}],
                make_protocol()["media_contract"],
            )


class AggregationTests(unittest.TestCase):
    def test_historical_fixture_reproduces_recorded_means_and_fails_gate(self) -> None:
        legacy = json.loads((ROOT / "bench/results/h100_quality_eval.json").read_text())
        manifests = {
            name: json.loads((ROOT / f"bench/results/{name}_manifest.json").read_text())
            for name in ("cf1", "cf2", "sf4")
        }
        protocol = make_protocol()
        protocol["evaluation"]["dimensions"] = {"overall": 1.0}
        protocol["evaluation"]["families"] = [
            {"id": "gemini", "kind": "model", "model_id": legacy["evaluator"], "passes": 2}
        ]
        rows = ratings_from_legacy_eval(legacy, manifests, family_id="gemini")
        with self.assertRaisesRegex(ProtocolError, "explicit legacy opt-in"):
            aggregate_ratings(protocol, rows)
        report = aggregate_ratings(protocol, rows, allow_unbound_legacy=True)
        self.assertAlmostEqual(report["systems"]["cf1"]["mean"], 5.6666666667)
        self.assertAlmostEqual(report["systems"]["sf4"]["mean"], 4.6666666667)
        self.assertAlmostEqual(report["systems"]["cf2"]["mean"], 4.3333333333)
        self.assertAlmostEqual(report["paired_deltas"]["cf1-vs-sf4"], 1.0)
        self.assertFalse(report["complete"])
        with self.assertRaisesRegex(ProtocolError, "exactly two model families"):
            evaluate_gate(protocol, report, selected_system="cf1", reference_system="sf4")

    def test_extra_passes_do_not_overweight_one_family(self) -> None:
        protocol = make_protocol()
        protocol["evaluation"]["dimensions"] = {"overall": 1.0}
        rows = []
        for pass_id in (1, 2):
            rows.append({
                "family_id": "judge-a", "rater_id": "a", "pass_id": pass_id,
                "system_id": "candidate-a", "prompt_id": "p", "seed": 1,
                "scores": {"overall": 10},
            })
        rows.append({
            "family_id": "judge-b", "rater_id": "b", "pass_id": 1,
            "system_id": "candidate-a", "prompt_id": "p", "seed": 1,
            "scores": {"overall": 2},
        })
        report = aggregate_ratings(protocol, rows, allow_unbound_legacy=True)
        self.assertEqual(report["systems"]["candidate-a"]["mean"], 6.0)

    def test_duplicate_rating_fails(self) -> None:
        protocol = make_protocol()
        protocol["evaluation"]["dimensions"] = {"overall": 1.0}
        row = {
            "family_id": "judge-a", "rater_id": "a", "pass_id": 1,
            "system_id": "candidate-a", "prompt_id": "p", "seed": 1,
            "scores": {"overall": 8},
        }
        with self.assertRaisesRegex(ProtocolError, "duplicate rating"):
            aggregate_ratings(
                protocol,
                [row, copy.deepcopy(row)],
                allow_unbound_legacy=True,
            )


if __name__ == "__main__":
    unittest.main()
