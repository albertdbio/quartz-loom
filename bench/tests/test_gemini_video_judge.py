from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from bench.quality_sweep import (
    ProtocolError,
    canonical_sha256,
    validate_raw_evidence_report,
)
from bench.gemini_video_judge import (
    GEMINI_ENDPOINT,
    GEMINI_MODEL_ID,
    GEMINI_RATING_SCHEMA,
    GEMINI_VIDEO_FPS,
    build_gemini_request,
    load_gemini_api_key,
    parse_gemini_response,
    post_gemini,
    ratings_from_gemini_evidence,
    run_gemini_plan,
    scrub_gemini_response,
)
from bench.video_judge import DEFAULT_RUBRIC_TEXT


FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMINI_CLI = PROJECT_ROOT / "scripts" / "quality-gemini-video-judge"


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


def gemini_response(rating: dict | None = None) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(rating or rating_payload())}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1234,
            "candidatesTokenCount": 78,
            "totalTokenCount": 1512,
            "thoughtsTokenCount": 200,
        },
        "modelVersion": GEMINI_MODEL_ID,
        "responseId": "response-123",
    }


def blind_plan(media_sha256: str) -> dict:
    plan = {
        "schema_version": 1,
        "protocol_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "family_id": "gemini-video",
        "rater_id": GEMINI_MODEL_ID,
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
    plan["protocol_sha256"] = canonical_sha256(protocol_for(plan))
    return plan


def protocol_for(plan: dict, *, readiness: str = "quality-qualified") -> dict:
    return {
        "schema_version": 1,
        "evaluation": {
            "rubric_sha256": hashlib.sha256(DEFAULT_RUBRIC_TEXT.encode()).hexdigest(),
            "families": [
                {
                    "id": plan["family_id"],
                    "kind": "model",
                    "model_id": GEMINI_MODEL_ID,
                    "evidence_provider": "google",
                    "readiness": readiness,
                    "passes": 2,
                    "rater_ids": [plan["rater_id"]],
                }
            ],
        },
    }


class GeminiRequestTests(unittest.TestCase):
    def test_captured_gemini31_video_response_matches_the_parser_contract(self) -> None:
        captured = json.loads(
            (FIXTURES / "google-gemini31-video-response.json").read_text()
        )
        rating = parse_gemini_response(captured)
        self.assertEqual(rating["scores"]["motion_naturalness"], 4)
        self.assertEqual(captured["usageMetadata"]["promptTokenCount"], 6114)

    def test_request_uploads_the_video_inline_at_the_registered_source_fps(self) -> None:
        request = build_gemini_request(b"video-bytes", "Judge this full video.")

        self.assertEqual(GEMINI_VIDEO_FPS, 16.0)
        self.assertEqual(request["contents"][0]["role"], "user")
        parts = request["contents"][0]["parts"]
        self.assertEqual(
            parts[0],
            {
                "inlineData": {
                    "mimeType": "video/mp4",
                    "data": base64.b64encode(b"video-bytes").decode("ascii"),
                },
                "videoMetadata": {"fps": 16.0},
            },
        )
        self.assertEqual(parts[1], {"text": "Judge this full video."})
        config = request["generationConfig"]
        self.assertEqual(config["temperature"], 0)
        self.assertEqual(config["maxOutputTokens"], 2048)
        self.assertEqual(
            config["responseFormat"],
            {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": GEMINI_RATING_SCHEMA,
                }
            },
        )
        self.assertNotIn("api_key", json.dumps(request).lower())

    def test_inline_limit_covers_the_complete_serialized_request(self) -> None:
        with patch("bench.gemini_video_judge.GEMINI_MAX_REQUEST_BYTES", 8):
            with self.assertRaisesRegex(ProtocolError, "serialized request"):
                build_gemini_request(b"1234567", "Judge this video.")

    def test_response_must_finish_cleanly_and_match_the_exact_integer_contract(self) -> None:
        self.assertEqual(parse_gemini_response(gemini_response()), rating_payload())

        truncated = gemini_response()
        truncated["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with self.assertRaisesRegex(ProtocolError, "finishReason"):
            parse_gemini_response(truncated)

        fractional = rating_payload()
        fractional["scores"]["motion_naturalness"] = 6.5
        with self.assertRaisesRegex(ProtocolError, "integer"):
            parse_gemini_response(gemini_response(fractional))

        extra = rating_payload()
        extra["overall"] = 7
        with self.assertRaisesRegex(ProtocolError, "unsupported field"):
            parse_gemini_response(gemini_response(extra))

        huge_usage = gemini_response()
        huge_usage["usageMetadata"]["totalTokenCount"] = 10**10000
        with self.assertRaisesRegex(ProtocolError, "non-negative integer"):
            parse_gemini_response(huge_usage)

    def test_response_rejects_ambiguous_candidates_and_provider_version_drift(self) -> None:
        ambiguous = gemini_response()
        ambiguous["candidates"].append(ambiguous["candidates"][0].copy())
        with self.assertRaisesRegex(ProtocolError, "exactly one candidate"):
            parse_gemini_response(ambiguous)

        drifted = gemini_response()
        drifted["modelVersion"] = "gemini-other"
        with self.assertRaisesRegex(ProtocolError, "modelVersion"):
            parse_gemini_response(drifted)

    def test_response_rejects_duplicate_json_keys(self) -> None:
        response = gemini_response()
        response["candidates"][0]["content"]["parts"][0]["text"] = (
            '{"scores":{"prompt_adherence":7,"prompt_adherence":8}}'
        )
        with self.assertRaisesRegex(ProtocolError, "duplicate JSON key"):
            parse_gemini_response(response)

    def test_scrubber_preserves_safety_codes_but_drops_provider_messages(self) -> None:
        response = gemini_response()
        response["candidates"][0]["safetyRatings"] = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "probability": "NEGLIGIBLE",
                "probabilityScore": 0.01,
                "blocked": False,
            }
        ]
        response["promptFeedback"] = {
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "NEGLIGIBLE",
                    "blocked": False,
                }
            ],
            "blockReasonMessage": "provider echoed private prompt",
        }

        scrubbed = scrub_gemini_response(response)

        self.assertEqual(
            scrubbed["candidates"][0]["safetyRatings"][0]["probability"],
            "NEGLIGIBLE",
        )
        self.assertIsNone(scrubbed["promptFeedback"]["blockReason"])
        self.assertNotIn("private prompt", json.dumps(scrubbed))

        unsafe_code = gemini_response()
        unsafe_code["candidates"][0]["safetyRatings"] = [
            {"category": "private prompt contents", "blocked": False}
        ]
        with self.assertRaisesRegex(ProtocolError, "category") as context:
            scrub_gemini_response(unsafe_code)
        self.assertNotIn("private prompt contents", str(context.exception))

        prompt_blocked = gemini_response()
        prompt_blocked["promptFeedback"] = {
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "HIGH",
                    "blocked": True,
                }
            ]
        }
        with self.assertRaisesRegex(ProtocolError, "prompt safety rating"):
            parse_gemini_response(prompt_blocked)

    def test_dotenv_loader_requires_exactly_one_nonempty_key_without_logging_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY='secret-value'\n")
            self.assertEqual(
                load_gemini_api_key(path, environ={}),
                "secret-value",
            )
            path.write_text("GEMINI_API_KEY=one\nGEMINI_API_KEY=two\n")
            with self.assertRaisesRegex(ProtocolError, "exactly once") as context:
                load_gemini_api_key(path, environ={})
            self.assertNotIn("one", str(context.exception))
            self.assertNotIn("two", str(context.exception))

    def test_http_transport_uses_header_auth_and_withholds_error_bodies(self) -> None:
        request = build_gemini_request(b"video", "Judge this video.")
        captured = gemini_response()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(captured).encode()

        with patch("bench.gemini_video_judge.urllib.request.urlopen", return_value=FakeResponse()) as opened:
            self.assertEqual(post_gemini(request, api_key="secret"), captured)
        sent = opened.call_args.args[0]
        self.assertEqual(sent.full_url, GEMINI_ENDPOINT)
        self.assertEqual(sent.get_header("X-goog-api-key"), "secret")
        self.assertNotIn("secret", sent.data.decode())

        error = urllib.error.HTTPError(
            GEMINI_ENDPOINT,
            400,
            "bad",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": {
                            "code": 400,
                            "message": "provider echoed secret-value",
                            "status": "INVALID_ARGUMENT",
                            "details": [
                                {
                                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                    "reason": "API_KEY_INVALID",
                                    "domain": "googleapis.com",
                                },
                                {
                                    "@type": "type.googleapis.com/google.rpc.BadRequest",
                                    "fieldViolations": [
                                        {
                                            "field": "generation_config.response_format",
                                            "description": "provider echoed secret-value",
                                        }
                                    ],
                                },
                            ],
                        }
                    }
                ).encode()
            ),
        )
        with patch("bench.gemini_video_judge.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(ProtocolError, "INVALID_ARGUMENT.*body withheld") as context:
                post_gemini(request, api_key="secret-value")
        self.assertNotIn("secret-value", str(context.exception))
        self.assertNotIn("API_KEY_INVALID", str(context.exception))
        self.assertNotIn("field=", str(context.exception))


class GeminiEvidenceTests(unittest.TestCase):
    def test_google_evidence_schema_is_provider_specific_and_covers_safe_shapes(self) -> None:
        schema = json.loads(
            (
                Path(__file__).parents[1]
                / "schemas"
                / "quality-google-model-evidence-v1.schema.json"
            ).read_text()
        )
        self.assertEqual(schema["properties"]["provider"]["const"], "google")
        self.assertEqual(
            schema["properties"]["model_id"]["const"], GEMINI_MODEL_ID
        )
        self.assertEqual(
            schema["properties"]["endpoint"]["const"], GEMINI_ENDPOINT
        )
        self.assertIn("promptFeedback", schema["$defs"]["rawResponse"]["required"])
        self.assertIn("safetyRatings", schema["$defs"]["candidate"]["required"])
        self.assertEqual(
            schema["$defs"]["wireRequest"]["properties"]["generationConfig"]
            ["properties"]["responseFormat"]["properties"]["text"]
            ["properties"]["schema"],
            {"const": GEMINI_RATING_SCHEMA},
        )

    def test_full_plan_is_resumable_hash_bound_and_replayable(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)
        calls = 0

        def transport(_request: dict) -> dict:
            nonlocal calls
            calls += 1
            return gemini_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            evidence_dir = root / "evidence"
            assets.mkdir()
            (assets / f"{plan['records'][0]['asset_id']}.mp4").write_bytes(video)

            raw, evidence = run_gemini_plan(
                protocol,
                plan,
                assets,
                evidence_dir,
                transport=transport,
                rubric_text=DEFAULT_RUBRIC_TEXT,
            )
            self.assertEqual(calls, 1)
            self.assertEqual(evidence["provider"], "google")
            self.assertEqual(evidence["model_id"], GEMINI_MODEL_ID)
            self.assertNotIn(base64.b64encode(video).decode(), json.dumps(evidence))
            self.assertEqual(raw["ratings"], ratings_from_gemini_evidence(protocol, plan, evidence))
            self.assertEqual(
                validate_raw_evidence_report(
                    protocol, plan, raw["ratings"], evidence
                ),
                canonical_sha256(evidence),
            )

            resumed_raw, resumed_evidence = run_gemini_plan(
                protocol,
                plan,
                assets,
                evidence_dir,
                transport=lambda _request: self.fail("resume must use cached evidence"),
                rubric_text=DEFAULT_RUBRIC_TEXT,
            )
            self.assertEqual(resumed_raw, raw)
            self.assertEqual(resumed_evidence, evidence)

            cache_path = evidence_dir / f"{plan['records'][0]['blind_id']}.json"
            poisoned = json.loads(cache_path.read_text())
            poisoned["raw_response"]["api_key"] = "must-not-escape"
            poisoned["raw_response_sha256"] = canonical_sha256(
                poisoned["raw_response"]
            )
            cache_path.write_text(json.dumps(poisoned))
            with self.assertRaisesRegex(ProtocolError, "canonical scrubbed") as context:
                run_gemini_plan(
                    protocol,
                    plan,
                    assets,
                    evidence_dir,
                    transport=lambda _request: self.fail(
                        "poisoned cache must fail before transport"
                    ),
                    rubric_text=DEFAULT_RUBRIC_TEXT,
                )
            self.assertNotIn("must-not-escape", str(context.exception))

            drifted_adapter = json.loads(json.dumps(evidence))
            drifted_adapter["records"][0]["request"]["adapter_sha256"] = "f" * 64
            with self.assertRaisesRegex(ProtocolError, "adapter_sha256 mismatch"):
                ratings_from_gemini_evidence(protocol, plan, drifted_adapter)

            tampered = json.loads(json.dumps(raw["ratings"]))
            tampered[0]["scores"]["motion_naturalness"] = 1
            with self.assertRaisesRegex(ProtocolError, "do not match"):
                validate_raw_evidence_report(protocol, plan, tampered, evidence)

            poisoned_report = json.loads(json.dumps(evidence))
            poisoned_raw = poisoned_report["records"][0]["raw_response"]
            poisoned_raw["candidates"][0]["requestEcho"] = {
                "api_key": "must-not-escape"
            }
            poisoned_report["records"][0]["raw_response_sha256"] = canonical_sha256(
                poisoned_raw
            )
            with self.assertRaisesRegex(ProtocolError, "canonical scrubbed") as context:
                validate_raw_evidence_report(
                    protocol, plan, raw["ratings"], poisoned_report
                )
            self.assertNotIn("must-not-escape", str(context.exception))

    def test_full_plan_rejects_unqualified_family_before_transport(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan, readiness="calibration-pending")
        plan["protocol_sha256"] = canonical_sha256(protocol)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProtocolError, "quality-qualified"):
                run_gemini_plan(
                    protocol,
                    plan,
                    Path(tmp) / "assets",
                    Path(tmp) / "evidence",
                    transport=lambda _request: self.fail(
                        "unqualified family must not call provider"
                    ),
                    rubric_text=DEFAULT_RUBRIC_TEXT,
                )

    def test_full_plan_requires_the_concrete_model_as_rater_identity(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        plan["rater_id"] = "gemini-review-seat"
        protocol = protocol_for(plan)
        plan["protocol_sha256"] = canonical_sha256(protocol)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            (assets / f"{plan['records'][0]['asset_id']}.mp4").write_bytes(video)
            with self.assertRaisesRegex(ProtocolError, "concrete model rater"):
                run_gemini_plan(
                    protocol,
                    plan,
                    assets,
                    root / "evidence",
                    transport=lambda _request: self.fail(
                        "rater drift must fail before provider transport"
                    ),
                    rubric_text=DEFAULT_RUBRIC_TEXT,
                )

    def test_media_and_cached_prompt_drift_fail_before_provider_transport(self) -> None:
        video = b"video"
        plan = blind_plan(hashlib.sha256(video).hexdigest())
        protocol = protocol_for(plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            evidence_dir = root / "evidence"
            assets.mkdir()
            asset_path = assets / f"{plan['records'][0]['asset_id']}.mp4"
            asset_path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ProtocolError, "media SHA-256"):
                run_gemini_plan(
                    protocol,
                    plan,
                    assets,
                    evidence_dir,
                    transport=lambda _request: self.fail(
                        "media mismatch must fail before transport"
                    ),
                    rubric_text=DEFAULT_RUBRIC_TEXT,
                )

            asset_path.write_bytes(video)
            run_gemini_plan(
                protocol,
                plan,
                assets,
                evidence_dir,
                transport=lambda _request: gemini_response(),
                rubric_text=DEFAULT_RUBRIC_TEXT,
            )
            changed = json.loads(json.dumps(plan))
            changed["records"][0]["prompt"] = "A materially different prompt."
            with self.assertRaisesRegex(ProtocolError, "cached Gemini evidence request"):
                run_gemini_plan(
                    protocol,
                    changed,
                    assets,
                    evidence_dir,
                    transport=lambda _request: self.fail(
                        "prompt-drifted cache must fail before transport"
                    ),
                    rubric_text=DEFAULT_RUBRIC_TEXT,
                )

    def test_plan_cli_refuses_unqualified_family_before_any_provider_call(self) -> None:
        plan = blind_plan(hashlib.sha256(b"video").hexdigest())
        protocol = protocol_for(plan, readiness="transport-verified")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rubric = root / "rubric.md"
            rubric.write_text(DEFAULT_RUBRIC_TEXT)
            protocol["evaluation"]["rubric"] = str(rubric)
            plan["protocol_sha256"] = canonical_sha256(protocol)
            protocol_path = root / "protocol.json"
            plan_path = root / "plan.json"
            protocol_path.write_text(json.dumps(protocol))
            plan_path.write_text(json.dumps(plan))
            environment = dict(os.environ)
            environment["GEMINI_API_KEY"] = "not-a-real-key"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GEMINI_CLI),
                    "plan",
                    str(protocol_path),
                    str(plan_path),
                    "--asset-dir",
                    str(root / "assets"),
                    "--evidence-dir",
                    str(root / "evidence"),
                    "--raw-output",
                    str(root / "raw.json"),
                    "--evidence-output",
                    str(root / "report.json"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("quality-qualified", completed.stderr)
            self.assertFalse((root / "raw.json").exists())
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
