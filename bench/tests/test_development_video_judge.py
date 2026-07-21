from __future__ import annotations

import copy
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.development_video_judge import (
    DEVELOPMENT_PURPOSE,
    _durable_write_json,
    _ensure_output_directory,
    preflight_development_video_understanding,
    run_development_video_understanding,
)
from bench.gemini_video_judge import GEMINI_MODEL_ID
from bench.quality_sweep import ProtocolError
from bench.video_judge import PEGASUS_MODEL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNDERSTAND_CLI = PROJECT_ROOT / "scripts" / "cf-video-understand"


def _rating() -> dict:
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
        "rationale": "The subject remains recognizable throughout the clip.",
    }


def _gemini_response() -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": json.dumps(_rating())}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 40,
            "totalTokenCount": 140,
        },
        "modelVersion": GEMINI_MODEL_ID,
        "responseId": "gemini-response-1",
    }


def _pegasus_response() -> dict:
    return {
        "id": "pegasus-response-1",
        "data": json.dumps(_rating()),
        "finish_reason": "stop",
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }


class DevelopmentVideoUnderstandingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifact_dir = self.root / "artifact"
        self.artifact_dir.mkdir()
        self.video = self.artifact_dir / "video.mp4"
        self.video.write_bytes(b"small-development-video")
        self.prompt = "A red fox runs through clean snow."
        self.rubric = self.root / "rubric.md"
        self.rubric.write_text("Inspect the entire video and score every registered dimension.")
        self.artifact_manifest = self.artifact_dir / "manifest.json"
        self.artifact_manifest.write_text("{}\n")
        self.artifact = {
            "artifact_manifest": {},
            "artifact_manifest_sha256": hashlib.sha256(self.artifact_manifest.read_bytes()).hexdigest(),
            "media_path": self.video.resolve(),
            "media_sha256": hashlib.sha256(self.video.read_bytes()).hexdigest(),
            "media_bytes": self.video.read_bytes(),
            "generation_prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "fps": 16.0,
        }
        self.output = self.root / "understanding"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _patch_artifact(self):
        return patch(
            "bench.development_video_judge.validate_cf_video_artifact",
            return_value=copy.deepcopy(self.artifact),
        )

    def _run(self, google, twelvelabs, *, retry_uncertain=()):
        with self._patch_artifact():
            return run_development_video_understanding(
                self.artifact_manifest,
                original_prompt=self.prompt,
                rubric_path=self.rubric,
                output_dir=self.output,
                google_transport=google,
                twelvelabs_transport=twelvelabs,
                retry_uncertain=set(retry_uncertain),
            )

    def test_preflight_builds_and_size_checks_both_requests_without_transport_or_attempts(self) -> None:
        calls: list[str] = []
        with self._patch_artifact():
            report = preflight_development_video_understanding(
                self.artifact_manifest,
                original_prompt=self.prompt,
                rubric_path=self.rubric,
            )

        self.assertEqual(calls, [])
        self.assertEqual(report["purpose"], DEVELOPMENT_PURPOSE)
        self.assertFalse(report["authorizes_quality_claim"])
        self.assertFalse(report["authorizes_performance_claim"])
        self.assertEqual(set(report["providers"]), {"google", "twelvelabs"})
        self.assertEqual(
            report["providers"]["google"]["transport_identity"],
            "not-executed-preflight",
        )
        self.assertEqual(report["media_sha256"], self.artifact["media_sha256"])
        self.assertFalse(self.output.exists())
        self.assertNotIn("base64", json.dumps(report).lower())

    def test_upload_calls_each_provider_once_and_persists_only_scrubbed_evidence(self) -> None:
        google_calls: list[dict] = []
        twelve_calls: list[dict] = []

        result = self._run(
            lambda request: google_calls.append(copy.deepcopy(request)) or _gemini_response(),
            lambda request: twelve_calls.append(copy.deepcopy(request)) or _pegasus_response(),
        )

        self.assertEqual(len(google_calls), 1)
        self.assertEqual(len(twelve_calls), 1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(
            result["providers"]["google"]["transport_identity"],
            "injected-unverified",
        )
        self.assertEqual(result["media_sha256"], self.artifact["media_sha256"])
        self.assertEqual(result["rating_prompt_sha256"], result["providers"]["google"]["rating_prompt_sha256"])
        self.assertEqual(result["rating_prompt_sha256"], result["providers"]["twelvelabs"]["rating_prompt_sha256"])
        self.assertEqual(result["rubric_sha256"], result["providers"]["google"]["rubric_sha256"])
        self.assertEqual(result["rubric_sha256"], result["providers"]["twelvelabs"]["rubric_sha256"])
        persisted = "\n".join(path.read_text() for path in self.output.glob("*.json"))
        encoded_media = base64.b64encode(self.video.read_bytes()).decode("ascii")
        self.assertNotIn(encoded_media, persisted)
        self.assertNotIn("base64_string", persisted)
        self.assertNotIn('"data": "c21', persisted)
        self.assertNotIn("small-development-video", persisted)
        self.assertNotIn("api_key", persisted.lower())
        self.assertIn(DEVELOPMENT_PURPOSE, persisted)

    def test_future_request_media_copy_is_rejected_before_output_or_transport(self) -> None:
        calls: list[str] = []
        for builder_name in ("build_gemini_request", "build_pegasus_request"):
            with self.subTest(builder=builder_name), self._patch_artifact():
                target = f"bench.development_video_judge.{builder_name}"
                if builder_name == "build_gemini_request":
                    from bench.gemini_video_judge import build_gemini_request as original
                else:
                    from bench.video_judge import build_pegasus_request as original

                def future_builder(video: bytes, prompt: str, *, _original=original) -> dict:
                    request = _original(video, prompt)
                    request["future_media_copy"] = base64.b64encode(video).decode("ascii")
                    return request

                with patch(target, side_effect=future_builder), self.assertRaisesRegex(
                    ProtocolError, "unsupported field"
                ):
                    run_development_video_understanding(
                        self.artifact_manifest,
                        original_prompt=self.prompt,
                        rubric_path=self.rubric,
                        output_dir=self.output,
                        google_transport=lambda _: calls.append("google") or _gemini_response(),
                        twelvelabs_transport=lambda _: calls.append("twelvelabs") or _pegasus_response(),
                    )
                self.assertFalse(self.output.exists())
        self.assertEqual(calls, [])

    def test_primary_inline_media_must_decode_to_the_validated_artifact(self) -> None:
        calls: list[str] = []
        for builder_name in ("build_gemini_request", "build_pegasus_request"):
            with self.subTest(builder=builder_name), self._patch_artifact():
                target = f"bench.development_video_judge.{builder_name}"
                if builder_name == "build_gemini_request":
                    from bench.gemini_video_judge import build_gemini_request as original
                else:
                    from bench.video_judge import build_pegasus_request as original

                def corrupt_builder(video: bytes, prompt: str, *, _original=original) -> dict:
                    request = _original(video, prompt)
                    corrupt = base64.b64encode(b"different-video-payload").decode("ascii")
                    if builder_name == "build_gemini_request":
                        request["contents"][0]["parts"][0]["inlineData"]["data"] = corrupt
                    else:
                        request["video"]["base64_string"] = corrupt
                    return request

                with patch(target, side_effect=corrupt_builder), self.assertRaisesRegex(
                    ProtocolError, "does not match the artifact"
                ):
                    run_development_video_understanding(
                        self.artifact_manifest,
                        original_prompt=self.prompt,
                        rubric_path=self.rubric,
                        output_dir=self.output,
                        google_transport=lambda _: calls.append("google") or _gemini_response(),
                        twelvelabs_transport=lambda _: calls.append("twelvelabs") or _pegasus_response(),
                    )
                self.assertFalse(self.output.exists())
        self.assertEqual(calls, [])

    def test_durable_state_writer_fsyncs_data_then_rename_then_directory(self) -> None:
        state = self.root / "durable.json"
        events: list[str] = []
        real_fsync = __import__("os").fsync
        real_replace = __import__("os").replace

        def recording_fsync(descriptor: int) -> None:
            import os
            import stat

            kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
            events.append(f"fsync-{kind}")
            real_fsync(descriptor)

        def recording_replace(source, destination) -> None:
            events.append("replace")
            real_replace(source, destination)

        with patch(
            "bench.development_video_judge.os.fsync", side_effect=recording_fsync
        ), patch(
            "bench.development_video_judge.os.replace", side_effect=recording_replace
        ):
            _durable_write_json(state, {"status": "in_flight"})

        self.assertEqual(events, ["fsync-file", "replace", "fsync-directory"])
        self.assertEqual(json.loads(state.read_text()), {"status": "in_flight"})

    def test_first_use_directory_creation_race_revalidates_the_winner(self) -> None:
        destination = self.root / "raced-understanding"
        import os

        real_mkdir = os.mkdir

        def concurrent_winner(path, mode=0o777) -> None:
            real_mkdir(path, mode)
            raise FileExistsError(path)

        with patch(
            "bench.development_video_judge.os.mkdir", side_effect=concurrent_winner
        ):
            resolved = _ensure_output_directory(destination)

        self.assertEqual(resolved, destination.resolve())
        self.assertTrue(resolved.is_dir())

    def test_existing_output_is_parent_and_child_fsynced_before_paid_work(self) -> None:
        destination = self.root / "existing-understanding"
        destination.mkdir()
        with patch("bench.development_video_judge._fsync_directory") as synchronize:
            resolved = _ensure_output_directory(destination)

        self.assertEqual(resolved, destination.resolve())
        self.assertEqual(
            synchronize.call_args_list,
            [
                unittest.mock.call(destination.resolve().parent),
                unittest.mock.call(destination.resolve()),
            ],
        )

    def test_cli_rejects_nonpositive_or_nonfinite_http_timeouts(self) -> None:
        for timeout in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(UNDERSTAND_CLI),
                        "upload",
                        "--artifact",
                        "missing.json",
                        "--prompt",
                        "prompt",
                        "--output",
                        "output",
                        "--timeout-seconds",
                        timeout,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("timeout", completed.stderr)

    def test_output_cannot_modify_the_validated_artifact_directory(self) -> None:
        calls: list[str] = []
        with self._patch_artifact(), self.assertRaisesRegex(
            ProtocolError, "artifact directory"
        ):
            run_development_video_understanding(
                self.artifact_manifest,
                original_prompt=self.prompt,
                rubric_path=self.rubric,
                output_dir=self.artifact_dir / "understanding",
                google_transport=lambda _: calls.append("google") or _gemini_response(),
                twelvelabs_transport=lambda _: calls.append("twelvelabs") or _pegasus_response(),
            )
        self.assertEqual(calls, [])
        self.assertFalse((self.artifact_dir / "understanding").exists())

    def test_complete_resume_makes_zero_provider_calls(self) -> None:
        self._run(lambda _: _gemini_response(), lambda _: _pegasus_response())

        calls: list[str] = []
        result = self._run(
            lambda _: calls.append("google") or _gemini_response(),
            lambda _: calls.append("twelvelabs") or _pegasus_response(),
        )

        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "complete")

    def test_timeout_is_uncertain_and_normal_resume_refuses_before_either_call(self) -> None:
        def timeout(_: dict) -> dict:
            raise TimeoutError("simulated timeout")

        with self.assertRaisesRegex(ProtocolError, "uncertain"):
            self._run(timeout, lambda _: _pegasus_response())
        state = json.loads((self.output / "google.json").read_text())
        self.assertEqual(state["status"], "uncertain")

        calls: list[str] = []
        with self.assertRaisesRegex(ProtocolError, "retry-uncertain"):
            self._run(
                lambda _: calls.append("google") or _gemini_response(),
                lambda _: calls.append("twelvelabs") or _pegasus_response(),
            )
        self.assertEqual(calls, [])

    def test_explicit_retry_calls_only_the_uncertain_provider(self) -> None:
        self._run(lambda _: _gemini_response(), lambda _: _pegasus_response())
        google_state = json.loads((self.output / "google.json").read_text())
        google_state["status"] = "uncertain"
        google_state["failure_kind"] = "transport-or-response-validation"
        google_state.pop("raw_response")
        google_state.pop("raw_response_sha256")
        google_state.pop("rating")
        (self.output / "google.json").write_text(json.dumps(google_state, sort_keys=True) + "\n")
        (self.output / "manifest.json").unlink()

        calls: list[str] = []
        result = self._run(
            lambda _: calls.append("google") or _gemini_response(),
            lambda _: calls.append("twelvelabs") or _pegasus_response(),
            retry_uncertain={"google"},
        )

        self.assertEqual(calls, ["google"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(json.loads((self.output / "google.json").read_text())["attempt"], 2)

    def test_second_provider_timeout_never_repeats_the_completed_first_provider(self) -> None:
        calls: list[str] = []

        def twelve_timeout(_: dict) -> dict:
            calls.append("twelvelabs-timeout")
            raise TimeoutError("simulated timeout")

        with self.assertRaisesRegex(ProtocolError, "uncertain"):
            self._run(
                lambda _: calls.append("google") or _gemini_response(),
                twelve_timeout,
            )
        self.assertEqual(calls, ["google", "twelvelabs-timeout"])
        self.assertEqual(
            json.loads((self.output / "google.json").read_text())["status"],
            "complete",
        )

        calls.clear()
        result = self._run(
            lambda _: calls.append("google") or _gemini_response(),
            lambda _: calls.append("twelvelabs") or _pegasus_response(),
            retry_uncertain={"twelvelabs"},
        )
        self.assertEqual(calls, ["twelvelabs"])
        self.assertEqual(result["status"], "complete")

    def test_response_parse_failure_is_uncertain_and_is_not_auto_retried(self) -> None:
        malformed = _gemini_response()
        malformed["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with self.assertRaisesRegex(ProtocolError, "uncertain"):
            self._run(lambda _: malformed, lambda _: _pegasus_response())
        state = json.loads((self.output / "google.json").read_text())
        self.assertEqual(state["status"], "uncertain")
        self.assertEqual(state["failure_kind"], "transport-or-response-validation")

    def test_concurrent_invocations_share_one_paid_transaction(self) -> None:
        calls: list[str] = []
        calls_lock = threading.Lock()
        first_google_started = threading.Event()
        release_google = threading.Event()
        results: list[dict] = []
        errors: list[BaseException] = []

        def google(_: dict) -> dict:
            with calls_lock:
                calls.append("google")
            first_google_started.set()
            if not release_google.wait(timeout=5):
                raise AssertionError("concurrency test did not release Google transport")
            return _gemini_response()

        def twelvelabs(_: dict) -> dict:
            with calls_lock:
                calls.append("twelvelabs")
            return _pegasus_response()

        def invoke() -> None:
            try:
                results.append(
                    run_development_video_understanding(
                        self.artifact_manifest,
                        original_prompt=self.prompt,
                        rubric_path=self.rubric,
                        output_dir=self.output,
                        google_transport=google,
                        twelvelabs_transport=twelvelabs,
                    )
                )
            except BaseException as exc:  # test captures thread failures for assertion
                errors.append(exc)

        with self._patch_artifact():
            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(first_google_started.wait(timeout=2))
            second.start()
            time.sleep(0.05)
            release_google.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(calls.count("google"), 1)
        self.assertEqual(calls.count("twelvelabs"), 1)

    def test_prompt_artifact_or_cache_drift_fails_before_either_call(self) -> None:
        calls: list[str] = []
        drifted = dict(self.artifact)
        drifted["generation_prompt_sha256"] = "0" * 64
        with patch(
            "bench.development_video_judge.validate_cf_video_artifact",
            return_value=drifted,
        ):
            with self.assertRaisesRegex(ProtocolError, "generation prompt"):
                run_development_video_understanding(
                    self.artifact_manifest,
                    original_prompt=self.prompt,
                    rubric_path=self.rubric,
                    output_dir=self.output,
                    google_transport=lambda _: calls.append("google") or _gemini_response(),
                    twelvelabs_transport=lambda _: calls.append("twelvelabs") or _pegasus_response(),
                )
        self.assertEqual(calls, [])

        self._run(lambda _: _gemini_response(), lambda _: _pegasus_response())
        state_path = self.output / "twelvelabs.json"
        state = json.loads(state_path.read_text())
        state["media_sha256"] = "f" * 64
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ProtocolError, "identity"):
            self._run(
                lambda _: calls.append("google") or _gemini_response(),
                lambda _: calls.append("twelvelabs") or _pegasus_response(),
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
