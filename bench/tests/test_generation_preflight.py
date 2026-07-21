from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bench.generation_preflight import (
    PreflightError,
    build_chunk_release_events,
    normalize_fsdp_generator_state_dict,
    rolling_taehv_trim_frames,
    validate_cache_plan,
    validate_confirmatory_artifact_coordinates,
    validate_preflight_plan,
    validate_strict_checkpoint_keys,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _coordinate(index: int) -> dict:
    return {
        "artifact_id": f"candidate-a__hold-{index:02d}__{2000 + index}",
        "run_id": "confirmatory-candidate-a",
        "prompt_id": f"hold-{index:02d}",
        "seed": 2000 + index,
        "split": "confirmatory",
    }


class ChunkReleaseEventTests(unittest.TestCase):
    def test_records_one_timestamp_per_released_chunk(self) -> None:
        events = build_chunk_release_events(
            ready_ns=[250_000_000, 310_000_000, 365_000_000],
            frame_counts=[1, 4, 4],
        )

        self.assertEqual(
            [event.to_dict() for event in events],
            [
                {
                    "chunk_index": 0,
                    "first_frame_index": 0,
                    "frame_count": 1,
                    "ready_ns": 250_000_000,
                },
                {
                    "chunk_index": 1,
                    "first_frame_index": 1,
                    "frame_count": 4,
                    "ready_ns": 310_000_000,
                },
                {
                    "chunk_index": 2,
                    "first_frame_index": 5,
                    "frame_count": 4,
                    "ready_ns": 365_000_000,
                },
            ],
        )
        self.assertTrue(
            all("frame_ready_ns" not in event.to_dict() for event in events)
        )

    def test_rejects_non_monotonic_chunk_release_timestamps(self) -> None:
        with self.assertRaisesRegex(PreflightError, "strictly increasing"):
            build_chunk_release_events(
                ready_ns=[250_000_000, 250_000_000],
                frame_counts=[1, 4],
            )


class RollingTaehvTrimTests(unittest.TestCase):
    def test_first_block_uses_causal_startup_trim_then_context_scaled_trim(self) -> None:
        self.assertEqual(rolling_taehv_trim_frames(0, 0), 3)
        self.assertEqual(
            [rolling_taehv_trim_frames(index, min(index, 3)) for index in range(1, 5)],
            [4, 8, 12, 12],
        )

    def test_trim_formula_reproduces_one_then_four_frame_releases(self) -> None:
        emitted = []
        for block_index in range(21):
            prior_context_latents = min(block_index, 3)
            decoded_frames = 4 * (prior_context_latents + 1)
            emitted.append(
                decoded_frames
                - rolling_taehv_trim_frames(
                    block_index,
                    prior_context_latents,
                )
            )
        self.assertEqual(emitted, [1, *([4] * 20)])

    def test_rejects_impossible_startup_or_context_state(self) -> None:
        for block_index, prior_context_latents in ((0, 1), (1, 0), (1, 4)):
            with self.subTest(
                block_index=block_index,
                prior_context_latents=prior_context_latents,
            ):
                with self.assertRaises(PreflightError):
                    rolling_taehv_trim_frames(
                        block_index,
                        prior_context_latents,
                    )


class CachePlanTests(unittest.TestCase):
    def test_rejects_241_global_attention_latents_in_fixed_21_cache(self) -> None:
        with self.assertRaisesRegex(
            PreflightError,
            r"241.*local_attn_size=-1.*21",
        ):
            validate_cache_plan(
                latent_frames=241,
                local_attn_size=-1,
                cache_latent_frames=21,
            )

    def test_accepts_short_global_attention_and_bounded_long_attention(self) -> None:
        validate_cache_plan(
            latent_frames=21,
            local_attn_size=-1,
            cache_latent_frames=21,
        )
        validate_cache_plan(
            latent_frames=241,
            local_attn_size=21,
            cache_latent_frames=21,
        )

    def test_rejects_local_attention_window_larger_than_the_fixed_cache(self) -> None:
        with self.assertRaisesRegex(PreflightError, "local attention window.*cache"):
            validate_cache_plan(
                latent_frames=241,
                local_attn_size=22,
                cache_latent_frames=21,
            )


class ReleaseSchedulePlanTests(unittest.TestCase):
    def test_plan_rejects_chunk_releases_that_do_not_cover_the_rollout(self) -> None:
        plan = {
            "latent_frames": 21,
            "local_attn_size": -1,
            "cache_latent_frames": 21,
            "expected_checkpoint_keys": ["model.weight"],
            "checkpoint_keys": ["model.weight"],
            "chunk_ready_ns": [250_000_000, 310_000_000],
            "chunk_frame_counts": [1, 4],
            "confirmatory_artifacts": [_coordinate(index) for index in range(4)],
        }

        with self.assertRaisesRegex(PreflightError, "5.*81.*RGB frames"):
            validate_preflight_plan(plan)


class CheckpointKeyTests(unittest.TestCase):
    def test_rejects_empty_checkpoint_key_sets(self) -> None:
        with self.assertRaisesRegex(PreflightError, "expected checkpoint keys.*non-empty"):
            validate_strict_checkpoint_keys(
                expected_keys=[],
                checkpoint_keys=[],
            )

        with self.assertRaisesRegex(PreflightError, "expected checkpoint keys.*non-empty"):
            normalize_fsdp_generator_state_dict({}, expected_keys=[])

    def test_accepts_exact_strict_checkpoint_key_set(self) -> None:
        validate_strict_checkpoint_keys(
            expected_keys=["model.layer.weight", "model.layer.bias"],
            checkpoint_keys=["model.layer.bias", "model.layer.weight"],
        )

    def test_rejects_missing_and_unexpected_checkpoint_keys(self) -> None:
        with self.assertRaisesRegex(
            PreflightError,
            r"missing=.*model\.layer\.weight.*unexpected=.*wrapped\.weight",
        ):
            validate_strict_checkpoint_keys(
                expected_keys=["model.layer.weight", "model.layer.bias"],
                checkpoint_keys=["model.layer.bias", "wrapped.weight"],
            )

    def test_normalizes_only_the_known_fsdp_prefix_then_requires_exact_keys(self) -> None:
        weight = object()
        bias = object()
        normalized = normalize_fsdp_generator_state_dict(
            {
                "model._fsdp_wrapped_module.layer.weight": weight,
                "model._fsdp_wrapped_module.layer.bias": bias,
            },
            expected_keys=["model.layer.weight", "model.layer.bias"],
        )

        self.assertEqual(set(normalized), {"model.layer.weight", "model.layer.bias"})
        self.assertIs(normalized["model.layer.weight"], weight)
        self.assertIs(normalized["model.layer.bias"], bias)

    def test_normalized_checkpoint_rejects_collisions_or_key_drift(self) -> None:
        with self.assertRaisesRegex(PreflightError, "collision"):
            normalize_fsdp_generator_state_dict(
                {
                    "model.layer.weight": object(),
                    "model._fsdp_wrapped_module.layer.weight": object(),
                },
                expected_keys=["model.layer.weight"],
            )

        with self.assertRaisesRegex(
            PreflightError,
            r"missing=.*model\.layer\.bias.*unexpected=.*model\.extra",
        ):
            normalize_fsdp_generator_state_dict(
                {
                    "model._fsdp_wrapped_module.layer.weight": object(),
                    "model._fsdp_wrapped_module.extra": object(),
                },
                expected_keys=["model.layer.weight", "model.layer.bias"],
            )


class ConfirmatoryCoordinateTests(unittest.TestCase):
    def test_accepts_exactly_four_distinct_confirmatory_coordinates(self) -> None:
        coordinates = [_coordinate(index) for index in range(4)]

        validated = validate_confirmatory_artifact_coordinates(coordinates)

        self.assertEqual([item.to_dict() for item in validated], coordinates)

    def test_rejects_duplicate_artifact_across_four_trials(self) -> None:
        coordinates = [_coordinate(index) for index in range(4)]
        coordinates[3]["artifact_id"] = coordinates[0]["artifact_id"]

        with self.assertRaisesRegex(PreflightError, "distinct artifact_id"):
            validate_confirmatory_artifact_coordinates(coordinates)

    def test_rejects_duplicate_run_prompt_seed_coordinate(self) -> None:
        coordinates = [_coordinate(index) for index in range(4)]
        coordinates[3].update(
            {
                "run_id": coordinates[0]["run_id"],
                "prompt_id": coordinates[0]["prompt_id"],
                "seed": coordinates[0]["seed"],
            }
        )

        with self.assertRaisesRegex(PreflightError, "run_id,prompt_id,seed"):
            validate_confirmatory_artifact_coordinates(coordinates)

    def test_rejects_non_confirmatory_coordinate(self) -> None:
        coordinates = [_coordinate(index) for index in range(4)]
        coordinates[2]["split"] = "development"

        with self.assertRaisesRegex(PreflightError, "split must be confirmatory"):
            validate_confirmatory_artifact_coordinates(coordinates)


class GenerationPreflightCliTests(unittest.TestCase):
    def test_cli_emits_chunk_events_not_per_frame_timestamps(self) -> None:
        plan = {
            "latent_frames": 2,
            "local_attn_size": -1,
            "cache_latent_frames": 21,
            "expected_checkpoint_keys": ["model.weight"],
            "checkpoint_keys": ["model.weight"],
            "chunk_ready_ns": [250_000_000, 310_000_000],
            "chunk_frame_counts": [1, 4],
            "confirmatory_artifacts": [_coordinate(index) for index in range(4)],
        }
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_text(json.dumps(plan))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "generation-preflight"),
                    str(plan_path),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            report["checkpoint"]["key_set_sha256"],
            hashlib.sha256(b'["model.weight"]').hexdigest(),
        )
        self.assertEqual(
            [event["frame_count"] for event in report["chunk_release_events"]],
            [1, 4],
        )
        self.assertNotIn("rgb_ready_ns", report)


if __name__ == "__main__":
    unittest.main()
