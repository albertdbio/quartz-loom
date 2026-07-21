from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from bench.quality_sweep import ProtocolError, canonical_sha256
from bench.video_judge import (
    DEFAULT_RUBRIC_TEXT,
    PEGASUS_ENDPOINT,
    PEGASUS_MODEL_ID,
    PEGASUS_RATING_SCHEMA,
    build_pegasus_request,
    load_twelvelabs_api_key,
    parse_pegasus_response,
    post_pegasus,
    ratings_from_pegasus_evidence,
    run_pegasus_plan,
)


FIXTURES = Path(__file__).parent / "fixtures"


def rating_payload() -> dict:
    return {
        "scores": {
            "prompt_adherence": 7,
            "spatial_fidelity": 6,
            "identity_consistency": 7,
            "motion_naturalness": 6,
            "temporal_artifacts": 8,
        },
        "first_third_quality": 7,
        "final_third_quality": 6,
        "failure_tags": ["minor-late-drift"],
        "rationale": "The subject remains recognizable; slight drift appears near the end.",
    }


def pegasus_response(rating: dict | None = None) -> dict:
    return {
        "id": "request-123",
        "data": json.dumps(rating or rating_payload()),
        "finish_reason": "stop",
        "usage": {"input_tokens": 42, "output_tokens": 77},
    }


def protocol_for(plan: dict) -> dict:
    return {
        "schema_version": 1,
        "evaluation": {
            "rubric_sha256": hashlib.sha256(DEFAULT_RUBRIC_TEXT.encode()).hexdigest(),
            "families": [
                {
                    "id": plan["family_id"],
                    "kind": "model",
                    "model_id": PEGASUS_MODEL_ID,
                    "evidence_provider": "twelvelabs",
                    "readiness": "quality-qualified",
                    "passes": 2,
                    "rater_ids": [plan["rater_id"]],
                }
            ]
        },
    }


def blind_plan(media_sha256: str) -> dict:
    plan = {
        "schema_version": 1,
        "protocol_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "family_id": "pegasus-video",
        "rater_id": PEGASUS_MODEL_ID,
        "pass_id": 1,
        "records": [
            {
                "blind_id": "a" * 20,
                "asset_id": "asset-" + "a" * 20,
                "case_id": "b" * 20,
                "prompt_id": "hold-test",
                "prompt": "A red fox runs through snow.",
                "slot": "A",
                "media_sha256": media_sha256,
            }
        ],
    }
    protocol = protocol_for(plan)
    plan["protocol_sha256"] = canonical_sha256(protocol)
    return plan


class PegasusRequestTests(unittest.TestCase):
    def test_captured_sync_response_fixture_matches_the_parser_contract(self) -> None:
        captured = json.loads(
            (FIXTURES / "twelvelabs-pegasus15-sync-response.json").read_text()
        )
        rating = parse_pegasus_response(captured)
        self.assertEqual(rating["scores"]["prompt_adherence"], 9)
        self.assertEqual(captured["usage"]["output_tokens"], 182)

    def test_request_is_non_streaming_structured_pegasus_15(self) -> None:
        request = build_pegasus_request(b"video-bytes", "Judge this full video.")

        self.assertEqual(request["model_name"], PEGASUS_MODEL_ID)
        self.assertEqual(request["stream"], False)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(
            request["response_format"]["json_schema"], PEGASUS_RATING_SCHEMA
        )
        self.assertEqual(
            request["video"],
            {"type": "base64_string", "base64_string": "dmlkZW8tYnl0ZXM="},
        )
        self.assertNotIn("api_key", json.dumps(request).lower())

    def test_response_must_finish_cleanly_and_match_exact_integer_contract(self) -> None:
        self.assertEqual(parse_pegasus_response(pegasus_response()), rating_payload())

        truncated = pegasus_response()
        truncated["finish_reason"] = "length"
        with self.assertRaisesRegex(ProtocolError, "finish_reason"):
            parse_pegasus_response(truncated)

        fractional = rating_payload()
        fractional["scores"]["motion_naturalness"] = 6.5
        with self.assertRaisesRegex(ProtocolError, "integer"):
            parse_pegasus_response(pegasus_response(fractional))

        extra = rating_payload()
        extra["overall"] = 7
        with self.assertRaisesRegex(ProtocolError, "unsupported field"):
            parse_pegasus_response(pegasus_response(extra))

    def test_inline_limit_applies_after_base64_expansion(self) -> None:
        with patch("bench.video_judge.PEGASUS_MAX_INLINE_BYTES", 8):
            with self.assertRaisesRegex(ProtocolError, "base64 video"):
                build_pegasus_request(b"1234567", "Judge this video")

    def test_dotenv_loader_never_returns_or_reports_an_empty_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("TWELVELABS_API_KEY='local-secret'\n")
            self.assertEqual(load_twelvelabs_api_key(path, environ={}), "local-secret")

            path.write_text("TWELVELABS_API_KEY=\n")
            with self.assertRaisesRegex(ProtocolError, "TWELVELABS_API_KEY") as raised:
                load_twelvelabs_api_key(path, environ={})
            self.assertNotIn("local-secret", str(raised.exception))

    def test_http_error_never_echoes_the_api_key(self) -> None:
        secret = "secret-that-must-not-escape"
        error = urllib.error.HTTPError(
            PEGASUS_ENDPOINT,
            400,
            "bad request",
            {},
            io.BytesIO(f"provider echoed {secret}".encode()),
        )
        with patch("bench.video_judge.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProtocolError) as raised:
                post_pegasus({}, api_key=secret)
        self.assertNotIn(secret, str(raised.exception))


class PegasusPlanTests(unittest.TestCase):
    def test_plan_hashes_media_archives_scrubbed_response_and_resumes(self) -> None:
        video = b"not-a-real-video-but-the-transport-is-mocked"
        digest = hashlib.sha256(video).hexdigest()
        plan = blind_plan(digest)
        protocol = protocol_for(plan)
        calls: list[dict] = []

        def transport(request: dict) -> dict:
            calls.append(copy.deepcopy(request))
            return pegasus_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            evidence_dir = root / "evidence"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(video)

            raw, evidence = run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                evidence_dir,
                transport=transport,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(raw["raw_evidence_sha256"], canonical_sha256(evidence))
            self.assertEqual(raw["ratings"][0]["blind_id"], "a" * 20)
            self.assertEqual(raw["ratings"][0]["media_sha256"], digest)
            record = evidence["records"][0]
            self.assertEqual(record["raw_response"], pegasus_response())
            self.assertNotIn('"base64_string":', json.dumps(evidence))
            self.assertNotIn("x-api-key", json.dumps(evidence).lower())
            self.assertEqual(record["endpoint"], PEGASUS_ENDPOINT)
            request_evidence = record["request"]
            self.assertEqual(
                request_evidence["adapter_sha256"],
                hashlib.sha256(
                    (Path(__file__).parents[1] / "video_judge.py").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                request_evidence["wire_request"]["video"],
                {
                    "type": "base64_string",
                    "media_sha256": digest,
                    "decoded_bytes": len(video),
                    "encoded_bytes": len("bm90LWEtcmVhbC12aWRlby1idXQtdGhlLXRyYW5zcG9ydC1pcy1tb2NrZWQ="),
                },
            )

            resumed_raw, resumed_evidence = run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                evidence_dir,
                transport=lambda _: self.fail("resume must not call the provider"),
            )
            self.assertEqual(resumed_raw, raw)
            self.assertEqual(resumed_evidence, evidence)

            poisoned_report = copy.deepcopy(evidence)
            poisoned_raw = poisoned_report["records"][0]["raw_response"]
            poisoned_raw["usage"]["api_key"] = "must-not-escape"
            poisoned_report["records"][0]["raw_response_sha256"] = canonical_sha256(
                poisoned_raw
            )
            with self.assertRaisesRegex(ProtocolError, "canonical scrubbed") as context:
                ratings_from_pegasus_evidence(protocol, plan, poisoned_report)
            self.assertNotIn("must-not-escape", str(context.exception))

    def test_media_hash_mismatch_fails_before_provider_call(self) -> None:
        plan = blind_plan("f" * 64)
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(
                b"wrong-video"
            )

            with self.assertRaisesRegex(ProtocolError, "media SHA-256"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    asset_dir,
                    root / "evidence",
                    transport=lambda _: self.fail("must fail before provider call"),
                )

    def test_blind_asset_identifiers_cannot_escape_evidence_or_asset_roots(self) -> None:
        plan = blind_plan("f" * 64)
        plan["records"][0]["asset_id"] = "../../private"
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProtocolError, "asset_id"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    Path(tmp) / "assets",
                    Path(tmp) / "evidence",
                    transport=lambda _: self.fail("invalid ID must fail before provider call"),
                )

    def test_cached_response_is_bound_to_prompt_and_model_request(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            evidence_dir = root / "evidence"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(video)
            run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                evidence_dir,
                transport=lambda _: pegasus_response(),
            )

            changed = copy.deepcopy(plan)
            changed["records"][0]["prompt"] = "A different prompt."
            with self.assertRaisesRegex(ProtocolError, "cached evidence request"):
                run_pegasus_plan(
                    protocol,
                    changed,
                    asset_dir,
                    evidence_dir,
                    transport=lambda _: self.fail("mismatched cache must fail closed"),
                )

    def test_cached_evidence_rejects_unexpected_secret_bearing_fields(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            evidence_dir = root / "evidence"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(video)
            run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                evidence_dir,
                transport=lambda _: pegasus_response(),
            )
            cache_path = evidence_dir / (plan["records"][0]["blind_id"] + ".json")
            cached = json.loads(cache_path.read_text())
            cached["api_key"] = "must-not-be-propagated"
            cache_path.write_text(json.dumps(cached))

            with self.assertRaisesRegex(ProtocolError, "unsupported field api_key"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    asset_dir,
                    evidence_dir,
                    transport=lambda _: self.fail("invalid cache must fail closed"),
                )

    def test_cached_raw_response_must_already_be_canonical_scrubbed_evidence(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            evidence_dir = root / "evidence"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(video)
            run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                evidence_dir,
                transport=lambda _: pegasus_response(),
            )
            cache_path = evidence_dir / (plan["records"][0]["blind_id"] + ".json")
            cached = json.loads(cache_path.read_text())
            cached["raw_response"]["api_key"] = "must-not-be-propagated"
            cached["raw_response_sha256"] = canonical_sha256(cached["raw_response"])
            cache_path.write_text(json.dumps(cached))

            with self.assertRaisesRegex(ProtocolError, "canonical scrubbed") as context:
                run_pegasus_plan(
                    protocol,
                    plan,
                    asset_dir,
                    evidence_dir,
                    transport=lambda _: self.fail("poisoned cache must fail closed"),
                )
            self.assertNotIn("must-not-be-propagated", str(context.exception))

    def test_provider_echo_fields_are_not_persisted_to_evidence(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)
        echoed = pegasus_response()
        echoed["request_params"] = {
            "video": {"base64_string": "sensitive-media-must-not-be-persisted"},
            "x-api-key": "sensitive-key-must-not-be-persisted",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_dir = root / "assets"
            asset_dir.mkdir()
            (asset_dir / (plan["records"][0]["asset_id"] + ".mp4")).write_bytes(video)
            _, evidence = run_pegasus_plan(
                protocol,
                plan,
                asset_dir,
                root / "evidence",
                transport=lambda _: echoed,
            )
            serialized = json.dumps(evidence)
            self.assertNotIn('"base64_string":', serialized)
            self.assertNotIn("sensitive-key", serialized)
            self.assertNotIn("request_params", serialized)

    def test_malformed_registered_pass_count_fails_as_protocol_error(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)
        protocol["evaluation"]["families"][0]["passes"] = "two"
        plan["protocol_sha256"] = canonical_sha256(protocol)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProtocolError, "passes"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    Path(tmp) / "assets",
                    Path(tmp) / "evidence",
                    transport=lambda _: self.fail("malformed protocol must fail first"),
                )

    def test_full_plan_runner_rejects_a_non_qualified_model_family(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)
        protocol["evaluation"]["families"][0]["readiness"] = "calibration-failed"
        plan["protocol_sha256"] = canonical_sha256(protocol)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProtocolError, "quality-qualified"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    Path(tmp) / "assets",
                    Path(tmp) / "evidence",
                    transport=lambda _: self.fail("unqualified family must not fan out"),
                )

    def test_full_plan_runner_requires_the_exact_registered_rubric(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProtocolError, "registered rubric SHA-256"):
                run_pegasus_plan(
                    protocol,
                    plan,
                    Path(tmp) / "assets",
                    Path(tmp) / "evidence",
                    rubric_text="a different unregistered rubric",
                    transport=lambda _: self.fail("rubric mismatch must fail first"),
                )


if __name__ == "__main__":
    unittest.main()
