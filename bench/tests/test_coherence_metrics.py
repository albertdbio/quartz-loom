from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench.coherence_metrics import (
    CoherenceConfig,
    FlowObservations,
    RaftSmallFlowBackend,
    score_clip,
    score_frames,
)
from bench.displacement_metrics import (
    DINOv2AppearanceBackend,
    MetricConfig as DisplacementMetricConfig,
    load_clip,
)


FRAME_COUNT = 81
HEIGHT = 96
WIDTH = 160


def _structured_frame() -> np.ndarray:
    rng = np.random.default_rng(20260721)
    frame = rng.integers(45, 185, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    for x in range(0, WIDTH, 16):
        cv2.line(frame, (x, 0), (x, HEIGHT - 1), (215, 215, 215), 1)
    for y in range(0, HEIGHT, 16):
        cv2.line(frame, (0, y), (WIDTH - 1, y), (30, 30, 30), 1)
    return frame


def _draw_square(frame: np.ndarray, center_x: int) -> None:
    center_y = HEIGHT // 2
    half = 11
    cv2.rectangle(
        frame,
        (center_x - half, center_y - half),
        (center_x + half, center_y + half),
        (25, 35, 240),
        -1,
    )
    cv2.rectangle(
        frame,
        (center_x - half, center_y - half),
        (center_x + half, center_y + half),
        (255, 255, 255),
        2,
    )
    cv2.line(
        frame,
        (center_x - 7, center_y),
        (center_x + 7, center_y),
        (20, 20, 20),
        2,
    )


def clean_static_clip() -> np.ndarray:
    frame = _structured_frame()
    return np.repeat(frame[None], FRAME_COUNT, axis=0)


def fast_translating_square_clip() -> np.ndarray:
    background = _structured_frame()
    frames = []
    for index in range(FRAME_COUNT):
        phase = (index * 4) % 224
        offset = phase if phase <= 112 else 224 - phase
        frame = background.copy()
        _draw_square(frame, 24 + offset)
        frames.append(frame)
    return np.stack(frames)


def full_frame_translation_clip(
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    speed: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(1729)
    panorama = rng.integers(
        35,
        220,
        size=(height, width + speed * (FRAME_COUNT - 1), 3),
        dtype=np.uint8,
    )
    panorama = cv2.GaussianBlur(panorama, (5, 5), 0)
    return np.stack(
        [
            panorama[:, index * speed : index * speed + width]
            for index in range(FRAME_COUNT)
        ]
    )


def entering_square_translation_clip(*, speed: int = 4) -> np.ndarray:
    panorama_width = WIDTH + speed * (FRAME_COUNT - 1)
    rng = np.random.default_rng(5)
    texture = cv2.GaussianBlur(
        rng.normal(0, 65, size=(HEIGHT, panorama_width)).astype(np.float32),
        (0, 0),
        1.5,
    )
    panorama = np.repeat(
        np.clip(120 + texture[..., None], 0, 255).astype(np.uint8),
        3,
        axis=2,
    )
    left = 210
    top = 12
    size = 72
    cv2.rectangle(
        panorama,
        (left, top),
        (left + size - 1, top + size - 1),
        (20, 30, 235),
        -1,
    )
    cv2.rectangle(
        panorama,
        (left, top),
        (left + size - 1, top + size - 1),
        (255, 255, 255),
        3,
    )
    cv2.line(
        panorama,
        (left + 15, top + size // 2),
        (left + size - 16, top + size // 2),
        (10, 10, 10),
        3,
    )
    return np.stack(
        [
            panorama[:, index * speed : index * speed + WIDTH]
            for index in range(FRAME_COUNT)
        ]
    )


class ExactTranslationFlowBackend:
    def __init__(self, speed: int) -> None:
        self.speed = speed

    def estimate(self, frames: np.ndarray) -> FlowObservations:
        flows = np.zeros(
            (len(frames) - 1, frames.shape[1], frames.shape[2], 2),
            dtype=np.float32,
        )
        flows[..., 0] = -float(self.speed)
        return FlowObservations(
            flows=flows,
            backend="synthetic:exact-translation",
            device="cpu",
        )


def progressively_noisy_clip() -> np.ndarray:
    base = _structured_frame()
    rng = np.random.default_rng(991)
    frames = []
    for index in range(FRAME_COUNT):
        amount = max(0.0, (index - 39) / 41.0)
        noise = rng.integers(0, 256, size=base.shape, dtype=np.uint8)
        frame = np.rint(
            (1.0 - amount) * base.astype(np.float32)
            + amount * noise.astype(np.float32)
        ).astype(np.uint8)
        frames.append(frame)
    return np.stack(frames)


def progressively_blurred_clip() -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    checker = ((xx // 4 + yy // 4) % 2).astype(np.uint8)
    base = np.stack(
        (
            35 + checker * 190,
            45 + checker * 170,
            55 + checker * 150,
        ),
        axis=-1,
    ).astype(np.uint8)
    frames = []
    for index in range(FRAME_COUNT):
        amount = max(0.0, (index - 39) / 41.0)
        if amount == 0.0:
            frames.append(base.copy())
            continue
        frames.append(
            cv2.GaussianBlur(
                base,
                (0, 0),
                sigmaX=0.5 + 5.5 * amount,
                sigmaY=0.5 + 5.5 * amount,
            )
        )
    return np.stack(frames)


def localized_pointillist_clip() -> np.ndarray:
    base = _structured_frame()
    rng = np.random.default_rng(8128)
    pointillist_band = rng.choice(
        np.asarray([0, 255], dtype=np.uint8),
        size=(HEIGHT, 32, 1),
    )
    pointillist_band = np.repeat(pointillist_band, 3, axis=2)
    frames = []
    for index in range(FRAME_COUNT):
        frame = base.copy()
        if index >= 40:
            frame[:, 64:96] = pointillist_band
        frames.append(frame)
    return np.stack(frames)


def abrupt_pointillist_replacement_clip() -> np.ndarray:
    base = _structured_frame()
    stripe = ((np.arange(WIDTH) // 3) % 2).astype(np.uint8)
    texture = np.repeat((20 + stripe * 220)[None, :, None], HEIGHT, axis=0)
    texture = np.repeat(texture, 3, axis=2)
    return np.stack([base] * 40 + [texture] * 41)


def uniform_flicker_clip() -> np.ndarray:
    base = _structured_frame().astype(np.int16)
    frames = []
    for index in range(FRAME_COUNT):
        offset = -45 if index % 2 == 0 else 65
        frames.append(np.clip(base + offset, 0, 255).astype(np.uint8))
    return np.stack(frames)


def synthetic_config() -> CoherenceConfig:
    return CoherenceConfig(
        expected_frames=FRAME_COUNT,
        expected_height=HEIGHT,
        expected_width=WIDTH,
        appearance="histogram",
        flow="farneback",
        device="cpu",
    )


class SyntheticCoherenceTests(unittest.TestCase):
    def test_clean_static_clip_scores_high(self) -> None:
        report = score_frames(clean_static_clip(), config=synthetic_config())

        self.assertGreaterEqual(report["coherence_score"], 8.5)
        self.assertGreaterEqual(report["temporal"]["score"], 8.5)
        self.assertGreaterEqual(report["spatial"]["score"], 8.5)
        self.assertFalse(report["degrades_over_time"])

    def test_fast_translation_is_not_incoherence(self) -> None:
        report = score_frames(
            fast_translating_square_clip(),
            config=synthetic_config(),
        )

        self.assertGreaterEqual(report["coherence_score"], 8.0)
        self.assertGreaterEqual(report["temporal"]["score"], 8.0)
        self.assertFalse(report["degrades_over_time"])

    def test_full_frame_translation_is_warped_before_scoring(self) -> None:
        speed = 4
        report = score_frames(
            full_frame_translation_clip(speed=speed),
            config=synthetic_config(),
            flow_backend=ExactTranslationFlowBackend(speed),
        )

        self.assertGreaterEqual(report["coherence_score"], 8.0)
        self.assertGreaterEqual(report["temporal_coherence_score"], 8.0)
        self.assertGreaterEqual(report["spatial_integrity_score"], 8.0)
        self.assertGreaterEqual(
            report["temporal"]["raw_flow_magnitude_mean_px"],
            float(speed),
        )
        self.assertLess(
            report["temporal"]["adjacent_flow_warp_residual_mean"],
            0.01,
        )
        self.assertFalse(report["degrades_over_time"])

    def test_entering_sharp_object_is_not_pointillist_incoherence(self) -> None:
        speed = 4
        report = score_frames(
            entering_square_translation_clip(speed=speed),
            config=synthetic_config(),
            flow_backend=ExactTranslationFlowBackend(speed),
        )

        self.assertGreaterEqual(report["temporal_coherence_score"], 9.0)
        self.assertGreaterEqual(report["spatial_integrity_score"], 8.0)
        self.assertGreaterEqual(report["coherence_score"], 8.0)
        self.assertFalse(report["degrades_over_time"])

    def test_progressive_noise_scores_low_and_exposes_decay(self) -> None:
        report = score_frames(progressively_noisy_clip(), config=synthetic_config())
        segments = {segment["name"]: segment for segment in report["segments"]}

        self.assertLess(report["coherence_score"], 5.0)
        self.assertGreaterEqual(segments["early"]["coherence_score"], 8.0)
        self.assertGreater(
            segments["early"]["coherence_score"]
            - segments["late"]["coherence_score"],
            4.0,
        )
        self.assertTrue(report["degrades_over_time"])

    def test_uniform_flicker_is_temporally_low_but_spatially_intact(self) -> None:
        report = score_frames(uniform_flicker_clip(), config=synthetic_config())

        self.assertLess(report["coherence_score"], 5.0)
        self.assertLess(report["temporal"]["score"], 4.0)
        self.assertGreaterEqual(report["spatial"]["score"], 7.0)

    def test_progressive_blur_loses_spatial_integrity(self) -> None:
        report = score_frames(progressively_blurred_clip(), config=synthetic_config())
        segments = {segment["name"]: segment for segment in report["segments"]}

        self.assertLess(report["coherence_score"], 5.0)
        self.assertGreater(
            segments["early"]["coherence_score"]
            - segments["late"]["coherence_score"],
            4.0,
        )
        self.assertTrue(report["degrades_over_time"])

    def test_localized_pointillist_tail_is_detected(self) -> None:
        report = score_frames(localized_pointillist_clip(), config=synthetic_config())
        segments = {segment["name"]: segment for segment in report["segments"]}

        self.assertGreaterEqual(segments["early"]["coherence_score"], 8.0)
        self.assertLess(segments["late"]["spatial"]["score"], 4.0)
        self.assertGreater(
            segments["early"]["coherence_score"]
            - segments["late"]["coherence_score"],
            4.0,
        )
        self.assertTrue(report["degrades_over_time"])

    def test_brief_replacement_into_stable_texture_soup_is_detected(self) -> None:
        report = score_frames(
            abrupt_pointillist_replacement_clip(),
            config=synthetic_config(),
            flow_backend=ExactTranslationFlowBackend(speed=0),
        )
        segments = {segment["name"]: segment for segment in report["segments"]}

        self.assertEqual(report["temporal"]["flow_warp_guard"], 1.0)
        self.assertGreater(
            report["temporal"]["adjacent_flow_warp_residual_max"],
            0.04,
        )
        self.assertLess(report["spatial_integrity_score"], 4.0)
        self.assertLess(segments["late"]["spatial"]["score"], 4.0)
        self.assertTrue(report["degrades_over_time"])

    def test_late_failure_does_not_retroactively_poison_clean_segments(self) -> None:
        speed = 4
        frames = entering_square_translation_clip(speed=speed)
        stripe = ((np.arange(WIDTH) // 3) % 2).astype(np.uint8)
        texture = np.repeat((20 + stripe * 220)[None, :, None], HEIGHT, axis=0)
        frames[70:] = np.repeat(texture, 3, axis=2)
        report = score_frames(
            frames,
            config=synthetic_config(),
            flow_backend=ExactTranslationFlowBackend(speed=speed),
        )
        segments = {segment["name"]: segment for segment in report["segments"]}

        self.assertGreaterEqual(segments["early"]["coherence_score"], 8.0)
        self.assertGreaterEqual(segments["middle"]["coherence_score"], 8.0)
        self.assertLess(segments["late"]["coherence_score"], 4.0)
        self.assertTrue(report["degrades_over_time"])


class CoherenceCalibrationFixtureTests(unittest.TestCase):
    def test_real_calibration_is_pinned_with_expected_ordering(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "coherence_calibration.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual(fixture["metric"], "video-coherence-metrics-v1")
        self.assertEqual(
            fixture["models"]["flow"],
            "torchvision:raft_small:C_T_V2:half-resolution",
        )
        self.assertEqual(
            fixture["models"]["spatial"],
            "patch-pointillist-and-structure-reference-v2",
        )
        self.assertEqual(len(fixture["rows"]), 26)
        self.assertTrue(
            fixture["expected_ordering_checks"]["group_mean_order"]["passed"]
        )
        self.assertTrue(
            fixture["expected_ordering_checks"]["condition_tier_separation"][
                "passed"
            ]
        )
        self.assertEqual(fixture["ordering_violations"], [])
        condition_means = {
            row["condition"]: row["mean_score"]
            for row in fixture["condition_summary"]
        }
        self.assertEqual(
            condition_means,
            {
                "cd_1step": 1.938434,
                "cd_4step": 7.30236,
                "final_1step": 9.631239,
                "final_2step": 9.961025,
                "ode_1step": 0.0,
                "ode_4step": 4.044224,
            },
        )
        anchors = {
            row["prompt"]: row
            for row in fixture["rows"]
            if row["source_group"] == "sgmd_pilot"
        }
        self.assertEqual(anchors["ball"]["coherence_score"], 0.0)
        self.assertEqual(
            anchors["ball"]["segment_scores"],
            {"early": 0.0, "late": 0.0, "middle": 0.0},
        )
        self.assertEqual(anchors["vehicle"]["coherence_score"], 7.553927)
        self.assertEqual(
            anchors["vehicle"]["segment_scores"],
            {"early": 10.0, "late": 6.935005, "middle": 8.763047},
        )
        self.assertTrue(anchors["vehicle"]["degrades_over_time"])


@unittest.skipUnless(
    os.environ.get("COHERENCE_FORK_GRID_ROOT")
    and os.environ.get("COHERENCE_SGMD_ROOT"),
    "set both coherence roots for the learned-backend calibration replay",
)
class LearnedCalibrationReplayTests(unittest.TestCase):
    def test_all_twenty_six_clips_reproduce_the_fixture_exactly(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "coherence_calibration.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        roots = {
            "fork_grid": Path(os.environ["COHERENCE_FORK_GRID_ROOT"]).resolve(),
            "sgmd_pilot": Path(os.environ["COHERENCE_SGMD_ROOT"]).resolve(),
        }
        config = CoherenceConfig()
        appearance = DINOv2AppearanceBackend(config.device)
        flow = RaftSmallFlowBackend(
            config.device,
            batch_size=config.flow_batch_size,
        )

        for expected in fixture["rows"]:
            with self.subTest(
                source_group=expected["source_group"],
                condition=expected["condition"],
                prompt=expected["prompt"],
            ):
                report = score_clip(
                    roots[expected["source_group"]] / expected["relative_path"],
                    config=config,
                    appearance_backend=appearance,
                    flow_backend=flow,
                )
                self.assertEqual(
                    report["coherence_score"],
                    expected["coherence_score"],
                )
                self.assertEqual(
                    report["temporal_coherence_score"],
                    expected["temporal_score"],
                )
                self.assertEqual(
                    report["spatial_integrity_score"],
                    expected["spatial_score"],
                )
                self.assertEqual(
                    report["degrades_over_time"],
                    expected["degrades_over_time"],
                )
                self.assertEqual(
                    {
                        segment["name"]: segment["coherence_score"]
                        for segment in report["segments"]
                    },
                    expected["segment_scores"],
                )
                self.assertEqual(report["models"], {
                    "appearance": fixture["models"]["appearance"],
                    "appearance_device": report["models"]["appearance_device"],
                    "flow": fixture["models"]["flow"],
                    "flow_device": report["models"]["flow_device"],
                    "spatial": fixture["models"]["spatial"],
                })


@unittest.skipUnless(
    os.environ.get("RUN_COHERENCE_LEARNED_REGRESSION") == "1",
    "set RUN_COHERENCE_LEARNED_REGRESSION=1 for DINOv2/RAFT motion replay",
)
class LearnedMotionRegressionTests(unittest.TestCase):
    def test_full_frame_translation_stays_coherent_with_default_backends(self) -> None:
        config = CoherenceConfig()
        report = score_frames(
            full_frame_translation_clip(
                height=config.expected_height,
                width=config.expected_width,
            ),
            config=config,
        )

        self.assertGreaterEqual(report["coherence_score"], 8.0)
        self.assertGreaterEqual(report["temporal_coherence_score"], 8.0)
        self.assertGreaterEqual(report["spatial_integrity_score"], 8.0)
        self.assertGreaterEqual(
            report["temporal"]["raw_flow_magnitude_mean_px"],
            1.0,
        )
        self.assertLess(
            report["temporal"]["adjacent_flow_warp_residual_mean"],
            0.04,
        )


@unittest.skipUnless(
    os.environ.get("COHERENCE_SGMD_ROOT"),
    "set COHERENCE_SGMD_ROOT for the learned pointillist regression",
)
class LearnedPointillistRegressionTests(unittest.TestCase):
    def test_brief_transition_into_stable_pointillism_is_not_averaged_away(
        self,
    ) -> None:
        root = Path(os.environ["COHERENCE_SGMD_ROOT"]).resolve()
        matches = sorted((root / "step_000025").glob("*wooden barrel*.mp4"))
        self.assertEqual(len(matches), 1)
        frames = load_clip(matches[0], config=DisplacementMetricConfig())
        replacement = np.stack([frames[0]] * 40 + [frames[-1]] * 41)
        report = score_frames(
            replacement,
            config=CoherenceConfig(appearance="histogram", flow="farneback"),
            flow_backend=ExactTranslationFlowBackend(speed=0),
        )
        segments = {segment["name"]: segment for segment in report["segments"]}

        # Median/p90 legitimately leave temporal coherence high because only
        # one of 80 transitions is bad.  The early-reference spatial axis must
        # still identify the stable pointillist replacement and its late tail.
        self.assertEqual(report["temporal"]["flow_warp_guard"], 1.0)
        self.assertLess(report["spatial_integrity_score"], 4.0)
        self.assertLess(segments["late"]["spatial"]["score"], 4.0)
        self.assertTrue(report["degrades_over_time"])


if __name__ == "__main__":
    unittest.main()
