from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import bench.displacement_metrics as displacement_metrics
from bench.displacement_metrics import (
    AppearanceObservations,
    DisplacementMetricError,
    MetricConfig,
    TrackObservations,
    load_clip,
    score_clip,
    score_frames,
)


FRAME_COUNT = 81
HEIGHT = 96
WIDTH = 160


def _textured_world(width: int = WIDTH, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    world = rng.integers(25, 180, size=(HEIGHT, width, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (5, 5), 0)
    for x in range(0, width, 16):
        cv2.line(world, (x, 0), (x, HEIGHT - 1), (210, 210, 210), 1)
    for y in range(0, HEIGHT, 16):
        cv2.line(world, (0, y), (width - 1, y), (35, 35, 35), 1)
    return world


def _draw_square(frame: np.ndarray, center_x: int, size: int) -> None:
    center_y = HEIGHT // 2
    half = size // 2
    top_left = (center_x - half, center_y - half)
    bottom_right = (center_x + half, center_y + half)
    cv2.rectangle(frame, top_left, bottom_right, (20, 35, 245), -1)
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
    cv2.line(
        frame,
        (center_x - half + 3, center_y),
        (center_x + half - 3, center_y),
        (20, 20, 20),
        1,
    )
    cv2.line(
        frame,
        (center_x, center_y - half + 3),
        (center_x, center_y + half - 3),
        (20, 20, 20),
        1,
    )


def translating_square() -> np.ndarray:
    background = _textured_world()
    frames = []
    for index in range(FRAME_COUNT):
        frame = background.copy()
        center_x = round(24 + index * 108 / (FRAME_COUNT - 1))
        _draw_square(frame, center_x, 22)
        frames.append(frame)
    return np.stack(frames)


def flickering_pulsing_square() -> np.ndarray:
    background = _textured_world(seed=11)
    frames = []
    for index in range(FRAME_COUNT):
        frame = background.copy()
        size = 16 if index % 2 == 0 else 38
        _draw_square(frame, WIDTH // 2, size)
        half = size // 2
        x0 = WIDTH // 2 - half + 2
        y0 = HEIGHT // 2 - half + 2
        for y in range(y0, y0 + size - 3, 4):
            for x in range(x0, x0 + size - 3, 4):
                if ((x - x0) // 4 + (y - y0) // 4 + index) % 2 == 0:
                    frame[y : y + 3, x : x + 3] = (245, 245, 245)
        if index % 3 == 0:
            frame[HEIGHT // 2 - 5 : HEIGHT // 2 + 5, WIDTH // 2 - 5 : WIDTH // 2 + 5] = 255
        frames.append(frame)
    return np.stack(frames)


def camera_pan_over_static_square() -> np.ndarray:
    pan_distance = 48
    world = _textured_world(WIDTH + pan_distance, seed=19)
    _draw_square(world[:, :WIDTH], 104, 22)
    # Redraw into the full world so the object is genuinely fixed in world space.
    center_y = HEIGHT // 2
    cv2.rectangle(world, (93, center_y - 11), (115, center_y + 11), (20, 35, 245), -1)
    cv2.rectangle(world, (93, center_y - 11), (115, center_y + 11), (255, 255, 255), 2)
    frames = []
    for index in range(FRAME_COUNT):
        offset = round(index * pan_distance / (FRAME_COUNT - 1))
        frames.append(world[:, offset : offset + WIDTH].copy())
    return np.stack(frames)


def camera_pan_with_screen_locked_square() -> np.ndarray:
    pan_distance = 48
    world = _textured_world(WIDTH + pan_distance, seed=23)
    frames = []
    for index in range(FRAME_COUNT):
        offset = round(index * pan_distance / (FRAME_COUNT - 1))
        frame = world[:, offset : offset + WIDTH].copy()
        _draw_square(frame, WIDTH // 2, 22)
        frames.append(frame)
    return np.stack(frames)


def synthetic_config() -> MetricConfig:
    return MetricConfig(
        expected_frames=FRAME_COUNT,
        expected_height=HEIGHT,
        expected_width=WIDTH,
        tracker="opencv-lk",
        appearance="histogram",
        sample_stride=5,
        grid_size=12,
        displaced_threshold=6.0,
    )


class DenseSyntheticTracker:
    """CoTracker-shaped grid where only points seeded on the square translate."""

    def __init__(
        self,
        *,
        start_box: tuple[float, float, float, float] = (13, 37, 35, 59),
        horizontal_displacement: float = 108.0,
    ) -> None:
        self.start_box = start_box
        self.horizontal_displacement = horizontal_displacement

    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        del frames
        x_coordinates = np.linspace(4, WIDTH - 5, grid_size)
        y_coordinates = np.linspace(4, HEIGHT - 5, max(8, round(grid_size * HEIGHT / WIDTH)))
        grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
        first = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1).astype(np.float32)
        x0, y0, x1, y1 = self.start_box
        object_tracks = (
            (first[:, 0] >= x0)
            & (first[:, 0] <= x1)
            & (first[:, 1] >= y0)
            & (first[:, 1] <= y1)
        )
        tracks = np.repeat(first[None], FRAME_COUNT, axis=0)
        for frame_index in range(FRAME_COUNT):
            tracks[frame_index, object_tracks, 0] += (
                frame_index * self.horizontal_displacement / (FRAME_COUNT - 1)
            )
        return TrackObservations(
            tracks=tracks,
            visibility=np.ones(tracks.shape[:2], dtype=bool),
            backend="dense-synthetic-cotracker-shape",
            device="cpu",
        )


class SparseFrameBackgroundTracker(DenseSyntheticTracker):
    """Dense tracks whose background visibility collapses on one frame."""

    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        observations = super().track(frames, grid_size=grid_size)
        visibility = observations.visibility.copy()
        visibility[40] = False
        visibility[40, :3] = True
        return TrackObservations(
            tracks=observations.tracks,
            visibility=visibility,
            backend="sparse-frame-background-cotracker-shape",
            device="cpu",
        )


class LowSurvivalChurnTracker(DenseSyntheticTracker):
    """All tracks disappear early, matching a clip that dissolves into noise."""

    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        observations = super().track(frames, grid_size=grid_size)
        visibility = np.zeros_like(observations.visibility)
        visibility[:21] = True
        return TrackObservations(
            tracks=observations.tracks,
            visibility=visibility,
            backend="low-survival-churn-cotracker-shape",
            device="cpu",
        )


class LowTailAppearanceBackend:
    """Mostly stable embeddings with recurring severe identity failures."""

    def encode(self, crops: list[np.ndarray]) -> AppearanceObservations:
        features = np.repeat(np.asarray([[1.0, 0.0]]), len(crops), axis=0)
        features[3::4] = (-1.0, 0.0)
        return AppearanceObservations(
            features=features,
            backend="synthetic-low-tail-appearance",
            device="cpu",
        )


class SyntheticDisplacementTests(unittest.TestCase):
    def test_motion_proposal_skips_unbounded_camera_warp(self) -> None:
        frames = np.zeros((2, HEIGHT, WIDTH, 3), dtype=np.uint8)
        homographies = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
        homographies[1] = np.diag([1.0e20, 1.0e20, 1.0])

        with patch.object(
            displacement_metrics.cv2,
            "warpPerspective",
            side_effect=AssertionError("unsafe perspective warp was attempted"),
        ) as warp:
            proposal = displacement_metrics._motion_proposal_mask(
                frames,
                homographies,
                np.asarray([0, 1]),
            )

        self.assertIsNone(proposal)
        warp.assert_not_called()

    def test_motion_proposal_skips_unbounded_inverse_camera_warp(self) -> None:
        frames = np.zeros((2, HEIGHT, WIDTH, 3), dtype=np.uint8)
        homographies = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
        homographies[1] = np.diag([1.0e-20, 1.0e-20, 1.0])

        with patch.object(
            displacement_metrics.cv2,
            "warpPerspective",
            side_effect=AssertionError("unsafe inverse perspective warp was attempted"),
        ) as warp:
            proposal = displacement_metrics._motion_proposal_mask(
                frames,
                homographies,
                np.asarray([0, 1]),
            )

        self.assertIsNone(proposal)
        warp.assert_not_called()

    def test_unsafe_late_warp_discards_partial_motion_proposal(self) -> None:
        frames = np.zeros((3, HEIGHT, WIDTH, 3), dtype=np.uint8)
        homographies = np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0)
        homographies[2] = np.diag([1.0e20, 1.0e20, 1.0])
        stabilized = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        stabilized[HEIGHT // 2 - 10 : HEIGHT // 2 + 10, WIDTH // 2 - 10 : WIDTH // 2 + 10] = 255

        with patch.object(
            displacement_metrics.cv2,
            "warpPerspective",
            return_value=stabilized,
        ) as warp:
            proposal = displacement_metrics._motion_proposal_mask(
                frames,
                homographies,
                np.asarray([0, 1, 2]),
            )

        self.assertIsNone(proposal)
        warp.assert_called_once()

    def test_singular_late_warp_discards_partial_motion_proposal(self) -> None:
        frames = np.zeros((3, HEIGHT, WIDTH, 3), dtype=np.uint8)
        homographies = np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0)
        homographies[2] = np.diag([0.0, 0.0, 1.0])
        stabilized = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        stabilized[HEIGHT // 2 - 10 : HEIGHT // 2 + 10, WIDTH // 2 - 10 : WIDTH // 2 + 10] = 255

        with patch.object(
            displacement_metrics.cv2,
            "warpPerspective",
            return_value=stabilized,
        ) as warp:
            proposal = displacement_metrics._motion_proposal_mask(
                frames,
                homographies,
                np.asarray([0, 1, 2]),
            )

        self.assertIsNone(proposal)
        warp.assert_called_once()

    def test_translating_square_scores_high(self) -> None:
        report = score_frames(translating_square(), config=synthetic_config())

        self.assertGreaterEqual(report["displacement_score"], 7.0)
        self.assertTrue(report["displaced_vs_collapsed"])
        self.assertGreater(
            report["foreground"]["max_centroid_excursion_fraction_width"],
            0.45,
        )
        self.assertGreaterEqual(report["foreground"]["track_survival"], 0.70)

    def test_dense_tracks_do_not_select_the_static_motion_trail(self) -> None:
        config = synthetic_config()
        config = MetricConfig(
            expected_frames=config.expected_frames,
            expected_height=config.expected_height,
            expected_width=config.expected_width,
            tracker=config.tracker,
            appearance=config.appearance,
            sample_stride=config.sample_stride,
            grid_size=20,
            displaced_threshold=config.displaced_threshold,
        )
        report = score_frames(
            translating_square(),
            config=config,
            track_backend=DenseSyntheticTracker(),
        )

        self.assertGreaterEqual(report["displacement_score"], 7.0)
        self.assertTrue(report["displaced_vs_collapsed"])
        self.assertGreater(
            report["foreground"]["trajectory_span_fraction_width"],
            0.60,
        )
        self.assertNotIn("coherence_degraded", report)
        self.assertNotIn("camera_compensated", report)

    def test_camera_failure_emits_explicit_screen_space_fallback(self) -> None:
        report = score_frames(
            translating_square(),
            config=MetricConfig(
                expected_frames=FRAME_COUNT,
                expected_height=HEIGHT,
                expected_width=WIDTH,
                tracker="opencv-lk",
                appearance="histogram",
                sample_stride=5,
                grid_size=20,
                displaced_threshold=6.0,
            ),
            track_backend=SparseFrameBackgroundTracker(),
        )

        json.dumps(report, allow_nan=False)
        self.assertIs(report["coherence_degraded"], True)
        self.assertIs(report["camera_compensated"], False)
        self.assertEqual(
            report["degradation_reason"],
            "too few persistent background tracks for camera compensation",
        )
        self.assertEqual(
            report["models"]["camera_compensation"],
            "screen-space-only-v1",
        )
        self.assertGreater(report["displacement_score"], 0.0)
        self.assertGreater(
            report["foreground"]["trajectory_span_fraction_width"],
            0.60,
        )
        self.assertEqual(
            report["foreground"]["trajectory_span_fraction_width"],
            report["foreground"]["screen_space_trajectory_span_fraction_width"],
        )
        self.assertIsNone(
            report["foreground"][
                "camera_compensated_trajectory_span_fraction_width"
            ]
        )
        self.assertIsNone(report["camera"]["median_global_motion_px"])
        self.assertIsNone(report["camera"]["median_homography_inlier_fraction"])
        self.assertIsNone(
            report["camera"]["median_homography_reprojection_error_px"]
        )

    def test_non_camera_metric_error_does_not_use_degraded_fallback(self) -> None:
        config = MetricConfig(
            expected_frames=FRAME_COUNT,
            expected_height=HEIGHT,
            expected_width=WIDTH,
            tracker="opencv-lk",
            appearance="histogram",
            sample_stride=5,
            grid_size=20,
            displaced_threshold=6.0,
            primary_bbox=(0, 0, 1, 1),
        )

        with self.assertRaisesRegex(
            DisplacementMetricError,
            "primary_bbox contains too few persistent tracks",
        ):
            score_frames(
                translating_square(),
                config=config,
                track_backend=DenseSyntheticTracker(),
            )

    def test_camera_fallback_reports_low_survival_dissolve(self) -> None:
        report = score_frames(
            translating_square(),
            config=MetricConfig(
                expected_frames=FRAME_COUNT,
                expected_height=HEIGHT,
                expected_width=WIDTH,
                tracker="opencv-lk",
                appearance="histogram",
                sample_stride=5,
                grid_size=20,
                displaced_threshold=6.0,
            ),
            track_backend=LowSurvivalChurnTracker(),
        )

        json.dumps(report, allow_nan=False)
        self.assertIs(report["coherence_degraded"], True)
        self.assertIs(report["camera_compensated"], False)
        self.assertIn(
            "no tracks survive long enough to identify an object",
            report["degradation_reason"],
        )
        self.assertLess(report["foreground"]["track_survival"], 0.5)
        self.assertGreater(report["displacement_score"], 0.0)
        self.assertFalse(report["displaced_vs_collapsed"])

    def test_flicker_and_pulse_do_not_score_as_displacement(self) -> None:
        report = score_frames(flickering_pulsing_square(), config=synthetic_config())

        self.assertLessEqual(report["displacement_score"], 2.5)
        self.assertFalse(report["displaced_vs_collapsed"])
        self.assertGreater(report["morph_guardrail"]["raw_flow_mean_px"], 0.5)
        self.assertLess(
            report["foreground"]["max_centroid_excursion_fraction_width"],
            0.08,
        )

    def test_default_stride_does_not_alias_adjacent_frame_flicker(self) -> None:
        config = synthetic_config()
        config = MetricConfig(
            expected_frames=config.expected_frames,
            expected_height=config.expected_height,
            expected_width=config.expected_width,
            tracker=config.tracker,
            appearance=config.appearance,
            sample_stride=10,
            grid_size=config.grid_size,
            displaced_threshold=config.displaced_threshold,
        )

        report = score_frames(flickering_pulsing_square(), config=config)

        self.assertGreater(report["morph_guardrail"]["raw_flow_mean_px"], 0.5)

    def test_low_tail_appearance_blocks_false_coherent_displacement(self) -> None:
        report = score_frames(
            flickering_pulsing_square(),
            config=MetricConfig(
                expected_frames=FRAME_COUNT,
                expected_height=HEIGHT,
                expected_width=WIDTH,
                tracker="opencv-lk",
                appearance="histogram",
                sample_stride=10,
                grid_size=20,
                displaced_threshold=6.0,
            ),
            track_backend=DenseSyntheticTracker(
                start_box=(WIDTH / 2 - 19, HEIGHT / 2 - 19, WIDTH / 2 + 19, HEIGHT / 2 + 19),
                horizontal_displacement=64,
            ),
            appearance_backend=LowTailAppearanceBackend(),
        )

        self.assertGreater(report["score_components"]["displacement_extent"], 0.8)
        self.assertGreater(report["score_components"]["coherence_guard"], 0.8)
        self.assertLessEqual(report["morph_guardrail"]["dino_guard"], 0.25)
        self.assertLessEqual(report["displacement_score"], 3.0)
        self.assertFalse(report["displaced_vs_collapsed"])

    def test_camera_pan_is_removed_from_object_displacement(self) -> None:
        report = score_frames(camera_pan_over_static_square(), config=synthetic_config())

        self.assertLessEqual(report["displacement_score"], 2.5)
        self.assertFalse(report["displaced_vs_collapsed"])
        self.assertGreater(report["camera"]["median_global_motion_px"], 20.0)
        self.assertLess(
            report["foreground"]["max_centroid_excursion_fraction_width"],
            0.08,
        )

    def test_screen_locked_subject_does_not_inherit_background_pan(self) -> None:
        report = score_frames(
            camera_pan_with_screen_locked_square(),
            config=synthetic_config(),
        )

        self.assertLessEqual(report["displacement_score"], 2.5)
        self.assertFalse(report["displaced_vs_collapsed"])
        self.assertGreater(
            report["foreground"]["camera_compensated_trajectory_span_fraction_width"],
            0.20,
        )
        self.assertLess(
            report["foreground"]["screen_space_trajectory_span_fraction_width"],
            0.08,
        )

    def test_png_directory_loader_and_json_shape(self) -> None:
        frames = translating_square()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, frame in enumerate(frames):
                self.assertTrue(
                    cv2.imwrite(str(root / f"frame-{index:06d}.png"), frame)
                )
            report = score_clip(root, config=synthetic_config())

        json.dumps(report, allow_nan=False)
        self.assertEqual(report["source"]["frame_count"], FRAME_COUNT)
        self.assertEqual(report["source"]["height"], HEIGHT)
        self.assertEqual(report["source"]["width"], WIDTH)
        self.assertEqual(report["models"]["tracker"], "opencv-lk")

    def test_unpadded_png_names_are_loaded_in_numeric_order(self) -> None:
        frames = np.stack(
            [np.full((HEIGHT, WIDTH, 3), index, dtype=np.uint8) for index in range(FRAME_COUNT)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, frame in enumerate(frames):
                self.assertTrue(cv2.imwrite(str(root / f"{index}.png"), frame))
            loaded = load_clip(root, config=synthetic_config())

        np.testing.assert_array_equal(loaded[:, 0, 0, 0], np.arange(FRAME_COUNT))


class P0CalibrationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "p0_displacement_calibration.json"
        )
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        cls.by_id = {row["blind_id"]: row for row in cls.fixture["rows"]}

    def test_same_latent_rolling_and_wan_decoders_are_equal_within_tolerance(self) -> None:
        regression = self.fixture["regressions"]["same_latent_decoder"]
        rolling = self.by_id[regression["rolling_blind_id"]]
        wan = self.by_id[regression["wan_blind_id"]]

        self.assertLessEqual(
            abs(rolling["metric_score"] - wan["metric_score"]),
            regression["score_tolerance"],
        )
        self.assertLessEqual(
            abs(rolling["span_fraction"] - wan["span_fraction"]),
            regression["span_fraction_tolerance"],
        )
        self.assertEqual(rolling["displaced"], wan["displaced"])

    def test_ring_on_does_not_outscore_off_where_blind_review_worsened(self) -> None:
        regression = self.fixture["regressions"]["ring"]
        for pair in regression["blind_worse_pairs"]:
            with self.subTest(pair=pair):
                self.assertLessEqual(
                    self.by_id[pair["on_blind_id"]]["metric_score"],
                    self.by_id[pair["off_blind_id"]]["metric_score"],
                )
        self.assertLessEqual(regression["on_mean"], regression["off_mean"])


if __name__ == "__main__":
    unittest.main()
