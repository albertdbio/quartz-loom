from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from bench.quality_sweep import (
    ProtocolError,
    aggregate_ratings,
    build_blind_plan,
    build_selection_lock,
    canonical_sha256,
    evaluate_gate,
    unblind_ratings,
)
from bench.tests.test_quality_contract import (
    BLIND_SECRET,
    bound_media_report,
    complete_ratings,
    complete_selection_evidence,
    confirmatory_manifest,
    contract_protocol,
    passing_performance,
    recompute_selection_report,
    selection_lock,
    sentinel_manifest,
)
from bench.video_judge import (
    DEFAULT_RUBRIC_TEXT,
    PEGASUS_MODEL_ID,
    run_pegasus_plan,
)
from bench.tests.test_quality_sweep import make_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI = PROJECT_ROOT / "scripts" / "quality-sweep"


class QualitySweepCliTests(unittest.TestCase):
    def _write_json(self, root: Path, name: str, value: object) -> Path:
        path = root / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path

    def _run(
        self,
        *arguments: object,
        expected: int = 0,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(CLI), *(str(argument) for argument in arguments)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            completed.returncode,
            expected,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return completed

    def _materialize_media(self, root: Path, manifest: dict) -> list[dict]:
        records = []
        for item in manifest["records"]:
            path = root / item["source_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"media:{item['artifact_id']}".encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["media_sha256"])
            path.write_bytes(payload)
            records.append({**item, "physical_source_file": str(path)})
        return records

    @staticmethod
    def _probe_media(path: Path) -> dict:
        decoded_frames = 81 if path.parent.name == "media" else 961
        return {
            "width": 832,
            "height": 480,
            "fps": 16.0,
            "decoded_frames": decoded_frames,
            "duration_s": decoded_frames / 16.0,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "video_streams": 1,
            "audio_streams": 0,
        }

    def _write_fake_ffprobe(self, root: Path) -> Path:
        path = root / "ffprobe"
        path.write_text(
            """#!/usr/bin/env python3
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[-1])
decoded_frames = 81 if source.parent.name == "media" else 961
print(json.dumps({
    "streams": [{
        "codec_type": "video",
        "width": 832,
        "height": 480,
        "r_frame_rate": "16/1",
        "nb_read_frames": str(decoded_frames),
        "duration": str(decoded_frames / 16.0),
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
    }],
    "format": {"duration": str(decoded_frames / 16.0)},
}))
"""
        )
        path.chmod(0o755)
        return path

    def _schema(self, name: str) -> dict:
        return json.loads((PROJECT_ROOT / "bench" / "schemas" / name).read_text())

    def _write_normalized_envelopes(
        self,
        root: Path,
        prefix: str,
        rows: list[dict],
        raw_evidence_reports: list[dict],
    ) -> tuple[list[Path], list[Path]]:
        grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["family_id"], row["rater_id"], row["pass_id"])].append(row)
        evidence_by_sha256 = {
            canonical_sha256(report): report for report in raw_evidence_reports
        }
        rating_paths = []
        evidence_paths = []
        for index, group in enumerate(grouped.values(), start=1):
            first = group[0]
            raw_evidence_sha256 = first["raw_evidence_sha256"]
            self.assertTrue(
                all(
                    row["raw_evidence_sha256"] == raw_evidence_sha256
                    for row in group
                )
            )
            self.assertIn(raw_evidence_sha256, evidence_by_sha256)
            envelope = {
                "schema_version": 1,
                "protocol_sha256": first["protocol_sha256"],
                "manifest_sha256": first["manifest_sha256"],
                "blind_plan_sha256": first["blind_plan_sha256"],
                "unblinding_key_sha256": first["unblinding_key_sha256"],
                "raw_ratings_sha256": first["raw_ratings_sha256"],
                "raw_evidence_sha256": raw_evidence_sha256,
                "family_id": first["family_id"],
                "rater_id": first["rater_id"],
                "pass_id": first["pass_id"],
                "ratings": group,
            }
            rating_paths.append(
                self._write_json(root, f"{prefix}-ratings-{index}.json", envelope)
            )
            evidence_paths.append(
                self._write_json(
                    root,
                    f"{prefix}-evidence-{index}.json",
                    evidence_by_sha256[raw_evidence_sha256],
                )
            )
        self.assertEqual(len(evidence_paths), len(raw_evidence_reports))
        return rating_paths, evidence_paths

    def _assert_allowed_shape(self, schema: dict, value: dict) -> None:
        self.assertIs(schema.get("additionalProperties"), False)
        self.assertLessEqual(set(schema.get("required", [])), set(value))
        self.assertLessEqual(set(value), set(schema["properties"]))

    def _assert_exact_required_shape(self, schema: dict, value: dict) -> None:
        self._assert_allowed_shape(schema, value)
        self.assertEqual(set(schema["required"]), set(value))

    def test_select_cli_recomputes_report_and_bound_lock(self) -> None:
        evidence = complete_selection_evidence()
        expected_report = recompute_selection_report(evidence)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = self._write_json(root, "protocol.json", evidence["protocol"])
            round_a_manifest_path = self._write_json(
                root, "round-a-manifest.json", evidence["round_a_manifest"]
            )
            round_b_manifest_path = self._write_json(
                root, "round-b-manifest.json", evidence["round_b_manifest"]
            )
            performance_path = self._write_json(
                root,
                "development-performance.json",
                evidence["development_performance"],
            )
            round_a_rating_paths, round_a_evidence_paths = self._write_normalized_envelopes(
                root,
                "round-a",
                evidence["round_a_ratings"],
                evidence["round_a_raw_evidence_reports"],
            )
            round_b_rating_paths, round_b_evidence_paths = self._write_normalized_envelopes(
                root,
                "round-b",
                evidence["round_b_ratings"],
                evidence["round_b_raw_evidence_reports"],
            )
            secret_path = root / "blind-secret.bin"
            secret_path.write_bytes(BLIND_SECRET)
            report_path = root / "selection-report.json"
            lock_path = root / "selection-lock.json"

            self._run(
                "select",
                protocol_path,
                "--round-a-manifest",
                round_a_manifest_path,
                "--round-a-ratings",
                *round_a_rating_paths,
                "--round-a-evidence-reports",
                *round_a_evidence_paths,
                "--round-b-manifest",
                round_b_manifest_path,
                "--round-b-ratings",
                *round_b_rating_paths,
                "--round-b-evidence-reports",
                *round_b_evidence_paths,
                "--development-performance",
                performance_path,
                "--secret-file",
                secret_path,
                "--locked-at",
                "2026-07-19T02:00:00Z",
                "--report-output",
                report_path,
                "--lock-output",
                lock_path,
            )
            report = json.loads(report_path.read_text())
            lock = json.loads(lock_path.read_text())
            self.assertEqual(report, expected_report)
            self.assertEqual(lock["finalist_system_id"], "candidate-a")
            self.assertEqual(lock["selection_report_sha256"], report["report_sha256"])

            rejected = self._run(
                "select",
                protocol_path,
                "--round-a-manifest",
                round_a_manifest_path,
                "--round-a-ratings",
                *round_a_rating_paths[:-1],
                "--round-a-evidence-reports",
                *round_a_evidence_paths,
                "--round-b-manifest",
                round_b_manifest_path,
                "--round-b-ratings",
                *round_b_rating_paths,
                "--round-b-evidence-reports",
                *round_b_evidence_paths,
                "--development-performance",
                performance_path,
                "--secret-file",
                secret_path,
                "--locked-at",
                "2026-07-19T02:00:00Z",
                "--report-output",
                root / "must-not-exist-report.json",
                "--lock-output",
                root / "must-not-exist-lock.json",
                expected=1,
            )
            self.assertIn("complete rating tensor", rejected.stderr)

            missing_round_evidence = self._run(
                "select",
                protocol_path,
                "--round-a-manifest",
                round_a_manifest_path,
                "--round-a-ratings",
                *round_a_rating_paths,
                "--round-b-manifest",
                round_b_manifest_path,
                "--round-b-ratings",
                *round_b_rating_paths,
                "--round-b-evidence-reports",
                *round_b_evidence_paths,
                "--development-performance",
                performance_path,
                "--secret-file",
                secret_path,
                "--locked-at",
                "2026-07-19T02:00:00Z",
                "--report-output",
                root / "must-not-exist-missing-evidence-report.json",
                "--lock-output",
                root / "must-not-exist-missing-evidence-lock.json",
                expected=2,
            )
            self.assertIn("--round-a-evidence-reports", missing_round_evidence.stderr)

    def test_normalized_loader_rejects_row_evidence_hash_mismatching_envelope(self) -> None:
        evidence = complete_selection_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rating_paths, _ = self._write_normalized_envelopes(
                root,
                "round-a",
                evidence["round_a_ratings"],
                evidence["round_a_raw_evidence_reports"],
            )
            envelope = json.loads(rating_paths[0].read_text())
            self.assertNotEqual(
                envelope["ratings"][0]["raw_evidence_sha256"], "0" * 64
            )
            envelope["ratings"][0]["raw_evidence_sha256"] = "0" * 64
            rating_paths[0].write_text(json.dumps(envelope))

            load_normalized = runpy.run_path(str(CLI))["_load_normalized_ratings"]
            with self.assertRaisesRegex(
                ProtocolError,
                "raw_evidence_sha256.*envelope",
            ):
                load_normalized(
                    rating_paths[0],
                    protocol_sha256=canonical_sha256(evidence["protocol"]),
                    manifest_sha256=canonical_sha256(evidence["round_a_manifest"]),
                )

    def test_unblind_rejects_scores_or_identity_not_supported_by_pegasus_evidence(self) -> None:
        protocol = contract_protocol()
        protocol["evaluation"]["rubric_sha256"] = hashlib.sha256(
            DEFAULT_RUBRIC_TEXT.encode()
        ).hexdigest()
        family = protocol["evaluation"]["families"][0]
        family.update(
            {
                "id": "pegasus-video",
                "model_id": PEGASUS_MODEL_ID,
                "evidence_provider": "twelvelabs",
                "rater_ids": [PEGASUS_MODEL_ID],
            }
        )
        manifest = make_manifest(protocol, phase="development-round-a")
        public, key = build_blind_plan(
            protocol,
            manifest,
            family["id"],
            1,
            BLIND_SECRET,
            rater_id=PEGASUS_MODEL_ID,
        )
        captured = json.loads(
            (
                PROJECT_ROOT
                / "bench/tests/fixtures/twelvelabs-pegasus15-sync-response.json"
            ).read_text()
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = root / "assets"
            asset_root.mkdir()
            payload_by_sha256 = {
                record["media_sha256"]: record["source_file"].encode()
                for record in manifest["records"]
            }
            for record in public["records"]:
                payload = payload_by_sha256[record["media_sha256"]]
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), record["media_sha256"]
                )
                (asset_root / f"{record['asset_id']}.mp4").write_bytes(payload)
            raw, evidence_report = run_pegasus_plan(
                protocol,
                public,
                asset_root,
                root / "evidence-cache",
                transport=lambda _request: copy.deepcopy(captured),
            )

            protocol_path = self._write_json(root, "protocol.json", protocol)
            manifest_path = self._write_json(root, "manifest.json", manifest)
            public_path = self._write_json(root, "public.json", public)
            key_path = self._write_json(root, "key.json", key)
            evidence_path = self._write_json(root, "evidence.json", evidence_report)
            secret_path = root / "secret.bin"
            secret_path.write_bytes(BLIND_SECRET)

            altered_scores = copy.deepcopy(raw)
            altered_scores["ratings"][0]["scores"]["prompt_adherence"] = 1
            altered_path = self._write_json(root, "altered-ratings.json", altered_scores)
            rejected = self._run(
                "unblind",
                protocol_path,
                manifest_path,
                public_path,
                key_path,
                altered_path,
                "--secret-file",
                secret_path,
                "--evidence-report",
                evidence_path,
                "--output",
                root / "must-not-exist-altered.json",
                expected=1,
            )
            self.assertRegex(rejected.stderr, "evidence|rating")

            identity_mutations = {
                "protocol_sha256": "f" * 64,
                "manifest_sha256": "e" * 64,
                "blind_plan_sha256": "d" * 64,
                "family_id": "wrong-family",
                "rater_id": "wrong-rater",
                "pass_id": 2,
            }
            for field, value in identity_mutations.items():
                with self.subTest(field=field):
                    mismatched_evidence = copy.deepcopy(evidence_report)
                    mismatched_evidence[field] = value
                    mismatched_raw = copy.deepcopy(raw)
                    mismatched_raw["raw_evidence_sha256"] = canonical_sha256(
                        mismatched_evidence
                    )
                    mismatched_evidence_path = self._write_json(
                        root, f"mismatched-{field}-evidence.json", mismatched_evidence
                    )
                    mismatched_raw_path = self._write_json(
                        root, f"mismatched-{field}-ratings.json", mismatched_raw
                    )
                    rejected = self._run(
                        "unblind",
                        protocol_path,
                        manifest_path,
                        public_path,
                        key_path,
                        mismatched_raw_path,
                        "--secret-file",
                        secret_path,
                        "--evidence-report",
                        mismatched_evidence_path,
                        "--output",
                        root / f"must-not-exist-{field}.json",
                        expected=1,
                    )
                    self.assertIn(field, rejected.stderr)

    def test_schema_shapes_match_every_pipeline_artifact(self) -> None:
        protocol_schema = self._schema("quality-protocol-v1.schema.json")
        draft_protocol = json.loads(
            (PROJECT_ROOT / "bench" / "quality" / "quality-repair-v1.protocol.json").read_text()
        )
        self._assert_allowed_shape(protocol_schema, draft_protocol)
        for field, definition in (
            ("baseline", "baseline"),
            ("development", "development"),
            ("confirmatory", "confirmatory"),
            ("evaluation", "evaluation"),
            ("gates", "gates"),
        ):
            self._assert_exact_required_shape(
                protocol_schema["$defs"][definition], draft_protocol[field]
            )
        self._assert_exact_required_shape(
            protocol_schema["$defs"]["referenceSystem"],
            draft_protocol["reference_systems"]["sf4-reference"],
        )

        selection_evidence = complete_selection_evidence()
        protocol = selection_evidence["protocol"]
        selection_report = recompute_selection_report(selection_evidence)
        lock = build_selection_lock(
            protocol,
            selection_report,
            locked_at="2026-07-19T02:00:00Z",
        )
        manifest = confirmatory_manifest(protocol, lock)
        sentinels = sentinel_manifest(protocol, lock)
        media = bound_media_report(protocol, manifest)
        sentinel_media = bound_media_report(protocol, sentinels)
        performance = passing_performance(
            protocol, lock, manifest, sentinels, media, sentinel_media
        )
        family = protocol["evaluation"]["families"][0]
        rater_id = family["rater_ids"][0]
        public, key = build_blind_plan(
            protocol,
            manifest,
            family["id"],
            1,
            BLIND_SECRET,
            selection_lock=lock,
            rater_id=rater_id,
        )
        raw_rows = [
            {
                "blind_id": row["blind_id"],
                "media_sha256": row["media_sha256"],
                "scores": {name: 8 for name in protocol["evaluation"]["dimensions"]},
                "first_third_quality": 8,
                "final_third_quality": 8,
                "failure_tags": [],
                "rationale": "No registered failure observed.",
            }
            for row in public["records"]
        ]
        raw_evidence_report = {
            "schema_version": 1,
            "provider": family["evidence_provider"],
            "model_id": family["model_id"],
            "protocol_sha256": canonical_sha256(protocol),
            "manifest_sha256": canonical_sha256(manifest),
            "blind_plan_sha256": canonical_sha256(public),
            "family_id": family["id"],
            "rater_id": rater_id,
            "pass_id": 1,
            "records": [
                {
                    "blind_id": row["blind_id"],
                    "media_sha256": row["media_sha256"],
                    "raw_response_sha256": canonical_sha256(row),
                }
                for row in raw_rows
            ],
        }
        raw_evidence_sha256 = canonical_sha256(raw_evidence_report)
        normalized_rows = unblind_ratings(
            protocol,
            manifest,
            public,
            key,
            raw_rows,
            blind_secret=BLIND_SECRET,
            family_id=family["id"],
            pass_id=1,
            rater_id=rater_id,
            raw_evidence_report=raw_evidence_report,
            selection_lock=lock,
        )
        raw_envelope = {
            "schema_version": 1,
            "protocol_sha256": canonical_sha256(protocol),
            "manifest_sha256": canonical_sha256(manifest),
            "blind_plan_sha256": canonical_sha256(public),
            "family_id": family["id"],
            "rater_id": rater_id,
            "pass_id": 1,
            "raw_evidence_sha256": raw_evidence_sha256,
            "ratings": raw_rows,
        }
        normalized_envelope = {
            "schema_version": 1,
            "protocol_sha256": canonical_sha256(protocol),
            "manifest_sha256": canonical_sha256(manifest),
            "blind_plan_sha256": canonical_sha256(public),
            "unblinding_key_sha256": canonical_sha256(key),
            "raw_ratings_sha256": canonical_sha256(raw_rows),
            "raw_evidence_sha256": raw_evidence_sha256,
            "family_id": family["id"],
            "rater_id": rater_id,
            "pass_id": 1,
            "ratings": normalized_rows,
        }
        raw_evidence_reports: list[dict] = []
        complete = complete_ratings(
            protocol,
            manifest,
            lock,
            raw_evidence_reports=raw_evidence_reports,
        )
        aggregate = aggregate_ratings(
            protocol,
            complete,
            manifest=manifest,
            selection_lock=lock,
            blind_secret=BLIND_SECRET,
            raw_evidence_reports=raw_evidence_reports,
            require_complete=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            media_root = Path(temporary)
            gate = evaluate_gate(
                protocol,
                aggregate,
                selected_system=lock["finalist_system_id"],
                reference_system="sf4-reference",
                performance=performance,
                media_report=media,
                manifest=manifest,
                selection_lock=lock,
                sentinel_manifest=sentinels,
                sentinel_media_report=sentinel_media,
                ratings=complete,
                raw_evidence_reports=raw_evidence_reports,
                blind_secret=BLIND_SECRET,
                media_records=self._materialize_media(media_root, manifest),
                sentinel_media_records=self._materialize_media(media_root, sentinels),
                media_probe=self._probe_media,
                selection_report=selection_report,
                round_a_manifest=selection_evidence["round_a_manifest"],
                round_a_ratings=selection_evidence["round_a_ratings"],
                round_a_raw_evidence_reports=selection_evidence[
                    "round_a_raw_evidence_reports"
                ],
                round_b_manifest=selection_evidence["round_b_manifest"],
                round_b_ratings=selection_evidence["round_b_ratings"],
                round_b_raw_evidence_reports=selection_evidence[
                    "round_b_raw_evidence_reports"
                ],
                development_performance=selection_evidence[
                    "development_performance"
                ],
            )

        artifacts = (
            ("quality-selection-lock-v1.schema.json", lock),
            ("quality-selection-report-v1.schema.json", selection_report),
            (
                "quality-development-performance-v1.schema.json",
                selection_evidence["development_performance"],
            ),
            ("quality-run-manifest-v1.schema.json", manifest),
            ("quality-blind-plan-v1.schema.json", public),
            ("quality-unblinding-key-v1.schema.json", key),
            ("quality-raw-ratings-v1.schema.json", raw_envelope),
            ("quality-ratings-v1.schema.json", normalized_envelope),
            ("quality-media-report-v1.schema.json", media),
            ("quality-performance-evidence-v1.schema.json", performance),
            ("quality-aggregate-report-v1.schema.json", aggregate),
            ("quality-gate-report-v1.schema.json", gate),
        )
        for schema_name, artifact in artifacts:
            with self.subTest(schema=schema_name):
                schema = self._schema(schema_name)
                if schema_name == "quality-run-manifest-v1.schema.json":
                    self._assert_allowed_shape(schema, artifact)
                else:
                    self._assert_exact_required_shape(schema, artifact)

        manifest_schema = self._schema("quality-run-manifest-v1.schema.json")
        self._assert_exact_required_shape(manifest_schema["$defs"]["run"], manifest["runs"][0])
        self._assert_exact_required_shape(
            manifest_schema["$defs"]["record"], manifest["records"][0]
        )
        self._assert_exact_required_shape(
            self._schema("quality-blind-plan-v1.schema.json")["properties"]["records"]["items"],
            public["records"][0],
        )
        self._assert_exact_required_shape(
            self._schema("quality-unblinding-key-v1.schema.json")["properties"]["records"]["items"],
            key["records"][0],
        )
        self._assert_exact_required_shape(
            self._schema("quality-raw-ratings-v1.schema.json")["properties"]["ratings"]["items"],
            raw_rows[0],
        )
        self._assert_exact_required_shape(
            self._schema("quality-ratings-v1.schema.json")["$defs"]["rating"],
            normalized_rows[0],
        )
        development_performance_schema = self._schema(
            "quality-development-performance-v1.schema.json"
        )
        self._assert_exact_required_shape(
            development_performance_schema["$defs"]["trial"],
            selection_evidence["development_performance"]["round_a_trials"][0],
        )
        selection_report_schema = self._schema(
            "quality-selection-report-v1.schema.json"
        )
        self._assert_exact_required_shape(
            selection_report_schema["$defs"]["roundA"],
            selection_report["round_a"],
        )
        self._assert_exact_required_shape(
            selection_report_schema["$defs"]["roundB"],
            selection_report["round_b"],
        )
        self._assert_exact_required_shape(
            selection_report_schema["$defs"]["roundACandidate"],
            selection_report["round_a"]["candidates"]["candidate-a"],
        )
        self._assert_exact_required_shape(
            selection_report_schema["$defs"]["roundBCandidate"],
            selection_report["round_b"]["candidates"]["candidate-a"],
        )
        performance_schema = self._schema("quality-performance-evidence-v1.schema.json")
        short_schema = performance_schema["$defs"]["shortTrial"]["allOf"][1]
        sentinel_schema = performance_schema["$defs"]["sentinelTrial"]["allOf"][1]
        self._assert_exact_required_shape(
            {
                **short_schema,
                "required": [
                    *performance_schema["$defs"]["trialBase"]["required"],
                    *short_schema["required"],
                ],
            },
            performance["cold_trial"],
        )
        self._assert_exact_required_shape(
            {
                **sentinel_schema,
                "required": [
                    *performance_schema["$defs"]["trialBase"]["required"],
                    *sentinel_schema["required"],
                ],
            },
            performance["sentinel_trials"][0],
        )
        self._assert_exact_required_shape(
            self._schema("quality-media-report-v1.schema.json")["$defs"]["mediaRecord"],
            media["records"][0],
        )
        aggregate_schema = self._schema("quality-aggregate-report-v1.schema.json")
        self._assert_exact_required_shape(
            aggregate_schema["$defs"]["ratingEvidence"], aggregate["rating_evidence"][0]
        )
        system = next(iter(aggregate["systems"].values()))
        self._assert_exact_required_shape(aggregate_schema["$defs"]["systemReport"], system)
        self._assert_exact_required_shape(
            aggregate_schema["$defs"]["itemReport"], system["items"][0]
        )
        gate_schema = self._schema("quality-gate-report-v1.schema.json")
        self.assertEqual(set(gate_schema["properties"]["checks"]["required"]), set(gate["checks"]))
        for check in gate["checks"].values():
            self._assert_exact_required_shape(gate_schema["$defs"]["check"], check)

    def test_strict_blind_unblind_aggregate_and_gate_pipeline(self) -> None:
        selection_evidence = complete_selection_evidence()
        protocol = selection_evidence["protocol"]
        selection_report = recompute_selection_report(selection_evidence)
        lock = build_selection_lock(
            protocol,
            selection_report,
            locked_at="2026-07-19T02:00:00Z",
        )
        manifest = confirmatory_manifest(protocol, lock)
        sentinels = sentinel_manifest(protocol, lock)
        media = bound_media_report(protocol, manifest)
        sentinel_media = bound_media_report(protocol, sentinels)
        performance = passing_performance(
            protocol,
            lock,
            manifest,
            sentinels,
            media,
            sentinel_media,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._materialize_media(root, manifest)
            self._materialize_media(root, sentinels)
            self._write_fake_ffprobe(root)
            protocol_path = self._write_json(root, "protocol.json", protocol)
            lock_path = self._write_json(root, "selection-lock.json", lock)
            selection_report_path = self._write_json(
                root, "selection-report.json", selection_report
            )
            round_a_manifest_path = self._write_json(
                root,
                "round-a-manifest.json",
                selection_evidence["round_a_manifest"],
            )
            round_b_manifest_path = self._write_json(
                root,
                "round-b-manifest.json",
                selection_evidence["round_b_manifest"],
            )
            development_performance_path = self._write_json(
                root,
                "development-performance.json",
                selection_evidence["development_performance"],
            )
            round_a_rating_paths, round_a_evidence_paths = self._write_normalized_envelopes(
                root,
                "round-a",
                selection_evidence["round_a_ratings"],
                selection_evidence["round_a_raw_evidence_reports"],
            )
            round_b_rating_paths, round_b_evidence_paths = self._write_normalized_envelopes(
                root,
                "round-b",
                selection_evidence["round_b_ratings"],
                selection_evidence["round_b_raw_evidence_reports"],
            )
            manifest_path = self._write_json(root, "manifest.json", manifest)
            sentinel_manifest_path = self._write_json(
                root, "sentinel-manifest.json", sentinels
            )
            media_path = self._write_json(root, "media-report.json", media)
            sentinel_media_path = self._write_json(
                root, "sentinel-media-report.json", sentinel_media
            )
            performance_path = self._write_json(
                root, "performance.json", performance
            )
            secret_path = root / "blind-secret.bin"
            secret_path.write_bytes(BLIND_SECRET)

            family = protocol["evaluation"]["families"][0]
            rater_id = family["rater_ids"][0]
            public_path = root / "public-plan.json"
            key_path = root / "unblinding-key.json"
            self._run(
                "blind",
                protocol_path,
                manifest_path,
                "--family",
                family["id"],
                "--rater-id",
                rater_id,
                "--pass-id",
                1,
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--public-output",
                public_path,
                "--key-output",
                key_path,
            )
            public = json.loads(public_path.read_text())
            raw_rows = [
                {
                    "blind_id": row["blind_id"],
                    "media_sha256": row["media_sha256"],
                    "scores": {
                        dimension: 8
                        for dimension in protocol["evaluation"]["dimensions"]
                    },
                    "first_third_quality": 8,
                    "final_third_quality": 8,
                    "failure_tags": [],
                    "rationale": "No registered failure observed.",
                }
                for row in public["records"]
            ]
            raw_envelope = {
                "schema_version": 1,
                "protocol_sha256": canonical_sha256(protocol),
                "manifest_sha256": canonical_sha256(manifest),
                "blind_plan_sha256": canonical_sha256(public),
                "family_id": family["id"],
                "rater_id": rater_id,
                "pass_id": 1,
                "raw_evidence_sha256": "0" * 64,
                "ratings": raw_rows,
            }
            raw_evidence_report = {
                "schema_version": 1,
                "provider": family["evidence_provider"],
                "model_id": family["model_id"],
                "protocol_sha256": raw_envelope["protocol_sha256"],
                "manifest_sha256": raw_envelope["manifest_sha256"],
                "blind_plan_sha256": raw_envelope["blind_plan_sha256"],
                "family_id": raw_envelope["family_id"],
                "rater_id": raw_envelope["rater_id"],
                "pass_id": raw_envelope["pass_id"],
                "records": [
                    {
                        "blind_id": row["blind_id"],
                        "media_sha256": row["media_sha256"],
                        "raw_response_sha256": canonical_sha256(row),
                    }
                    for row in raw_rows
                ],
            }
            raw_envelope["raw_evidence_sha256"] = canonical_sha256(
                raw_evidence_report
            )
            evidence_path = self._write_json(
                root,
                "raw-evidence.json",
                raw_evidence_report,
            )
            raw_path = self._write_json(root, "raw-ratings.json", raw_envelope)
            normalized_path = root / "normalized-ratings.json"
            self._run(
                "unblind",
                protocol_path,
                manifest_path,
                public_path,
                key_path,
                raw_path,
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--evidence-report",
                evidence_path,
                "--output",
                normalized_path,
            )
            normalized = json.loads(normalized_path.read_text())
            self.assertEqual(normalized["raw_ratings_sha256"], canonical_sha256(raw_rows))
            self.assertEqual(len(normalized["ratings"]), len(manifest["records"]))

            wrong_secret = root / "wrong-secret.bin"
            wrong_secret.write_bytes(b"x" * 32)
            rejected = self._run(
                "unblind",
                protocol_path,
                manifest_path,
                public_path,
                key_path,
                raw_path,
                "--selection-lock",
                lock_path,
                "--secret-file",
                wrong_secret,
                "--evidence-report",
                evidence_path,
                "--output",
                root / "must-not-exist.json",
                expected=1,
            )
            self.assertIn("integrity check failed", rejected.stderr)

            mismatched_evidence = self._write_json(
                root,
                "mismatched-evidence.json",
                {**raw_evidence_report, "pass_id": 2},
            )
            rejected = self._run(
                "unblind",
                protocol_path,
                manifest_path,
                public_path,
                key_path,
                raw_path,
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--evidence-report",
                mismatched_evidence,
                "--output",
                root / "must-not-exist-evidence.json",
                expected=1,
            )
            self.assertIn("raw_evidence_sha256 mismatch", rejected.stderr)

            raw_evidence_reports: list[dict] = []
            complete = complete_ratings(
                protocol,
                manifest,
                lock,
                raw_evidence_reports=raw_evidence_reports,
            )
            rating_paths, evidence_paths = self._write_normalized_envelopes(
                root,
                "confirmatory",
                complete,
                raw_evidence_reports,
            )

            aggregate_path = root / "aggregate.json"
            self._run(
                "aggregate",
                protocol_path,
                manifest_path,
                *rating_paths,
                "--evidence-reports",
                *evidence_paths,
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--output",
                aggregate_path,
            )
            aggregate = json.loads(aggregate_path.read_text())
            self.assertTrue(aggregate["complete"])

            missing_aggregate_evidence = self._run(
                "aggregate",
                protocol_path,
                manifest_path,
                *rating_paths,
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--output",
                root / "must-not-exist-missing-aggregate-evidence.json",
                expected=2,
            )
            self.assertIn("--evidence-reports", missing_aggregate_evidence.stderr)

            mismatched_report = copy.deepcopy(raw_evidence_reports[0])
            mismatched_report["records"][0]["raw_response_sha256"] = "0" * 64
            mismatched_report_path = self._write_json(
                root, "mismatched-confirmatory-evidence.json", mismatched_report
            )
            rejected = self._run(
                "aggregate",
                protocol_path,
                manifest_path,
                *rating_paths,
                "--evidence-reports",
                mismatched_report_path,
                *evidence_paths[1:],
                "--selection-lock",
                lock_path,
                "--secret-file",
                secret_path,
                "--output",
                root / "must-not-exist-mismatched-aggregate-evidence.json",
                expected=1,
            )
            self.assertIn("raw evidence", rejected.stderr)

            gate_path = root / "gate.json"
            self._run(
                "gate",
                protocol_path,
                aggregate_path,
                "--manifest",
                manifest_path,
                "--selection-lock",
                lock_path,
                "--selection-report",
                selection_report_path,
                "--round-a-manifest",
                round_a_manifest_path,
                "--round-a-ratings",
                *round_a_rating_paths,
                "--round-a-evidence-reports",
                *round_a_evidence_paths,
                "--round-b-manifest",
                round_b_manifest_path,
                "--round-b-ratings",
                *round_b_rating_paths,
                "--round-b-evidence-reports",
                *round_b_evidence_paths,
                "--development-performance",
                development_performance_path,
                "--sentinel-manifest",
                sentinel_manifest_path,
                "--performance",
                performance_path,
                "--media-report",
                media_path,
                "--sentinel-media-report",
                sentinel_media_path,
                "--ratings",
                *rating_paths,
                "--evidence-reports",
                *evidence_paths,
                "--secret-file",
                secret_path,
                "--output",
                gate_path,
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ.get('PATH', '')}",
                },
            )
            gate = json.loads(gate_path.read_text())
            self.assertTrue(gate["pass"])
            self.assertEqual(set(gate["checks"]), set(protocol["gates"]))


if __name__ == "__main__":
    unittest.main()
