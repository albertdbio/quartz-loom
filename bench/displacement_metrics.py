"""Measure coherent foreground displacement without rewarding raw pixel motion.

The default path uses Meta's CoTracker3 offline model for quasi-dense point
tracks and DINOv2 ViT-S/14 for object-crop identity.  Camera motion is removed
with background-only RANSAC homographies.  Optical flow is diagnostic and
penalizing only: it can never add displacement credit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np


class DisplacementMetricError(ValueError):
    """The clip or a measurement backend violated the metric contract."""


class _CameraCompensationUnavailable(DisplacementMetricError):
    """The validated tracks cannot support a background camera transform."""


@dataclass(frozen=True)
class MetricConfig:
    expected_frames: int = 81
    expected_height: int = 480
    expected_width: int = 832
    tracker: str = "cotracker3"
    appearance: str = "dinov2"
    sample_stride: int = 10
    grid_size: int = 20
    displaced_threshold: float = 5.0
    device: str = "auto"
    primary_bbox: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.expected_frames, "expected_frames"),
            (self.expected_height, "expected_height"),
            (self.expected_width, "expected_width"),
            (self.sample_stride, "sample_stride"),
            (self.grid_size, "grid_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DisplacementMetricError(f"{label} must be a positive integer")
        if self.tracker not in {"cotracker3", "opencv-lk"}:
            raise DisplacementMetricError("tracker must be cotracker3 or opencv-lk")
        if self.appearance not in {"dinov2", "histogram"}:
            raise DisplacementMetricError("appearance must be dinov2 or histogram")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise DisplacementMetricError("device must be auto, cuda, mps, or cpu")
        if (
            isinstance(self.displaced_threshold, bool)
            or not isinstance(self.displaced_threshold, (int, float))
            or not math.isfinite(float(self.displaced_threshold))
            or not 0.0 <= float(self.displaced_threshold) <= 10.0
        ):
            raise DisplacementMetricError("displaced_threshold must be in [0, 10]")
        if self.primary_bbox is not None:
            if (
                len(self.primary_bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in self.primary_bbox)
                or self.primary_bbox[2] <= 0
                or self.primary_bbox[3] <= 0
            ):
                raise DisplacementMetricError("primary_bbox must be integer x,y,width,height")


@dataclass(frozen=True)
class TrackObservations:
    tracks: np.ndarray
    visibility: np.ndarray
    backend: str
    device: str


@dataclass(frozen=True)
class AppearanceObservations:
    features: np.ndarray
    backend: str
    device: str


class TrackBackend(Protocol):
    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        ...


class AppearanceBackend(Protocol):
    def encode(self, crops: Sequence[np.ndarray]) -> AppearanceObservations:
        ...


def _finite_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DisplacementMetricError("metric produced a non-finite value")
    return result


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _smoothstep(low: float, high: float, value: float) -> float:
    if not low < high:
        raise DisplacementMetricError("smoothstep bounds are invalid")
    position = _clip01((float(value) - low) / (high - low))
    return position * position * (3.0 - 2.0 * position)


def _resolve_torch_device(requested: str) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by CLI install guard
        raise DisplacementMetricError("torch is required for learned metric backends") from error
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise DisplacementMetricError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise DisplacementMetricError("MPS was requested but is unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenCVLKTracker:
    """Small deterministic fallback used by unit tests and explicit CLI opt-in."""

    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        height, width = gray[0].shape
        minimum_distance = max(3, round(min(height, width) / max(grid_size, 4)))
        corners = cv2.goodFeaturesToTrack(
            gray[0],
            maxCorners=max(grid_size * grid_size, 96),
            qualityLevel=0.005,
            minDistance=minimum_distance,
            blockSize=5,
            useHarrisDetector=False,
        )
        if corners is None or len(corners) < 12:
            raise DisplacementMetricError("OpenCV could not seed enough point tracks")
        points = corners.reshape(-1, 2).astype(np.float32)
        track_count = len(points)
        tracks = np.repeat(points[None, :, :], len(frames), axis=0)
        visibility = np.zeros((len(frames), track_count), dtype=bool)
        visibility[0] = True
        current = points.reshape(-1, 1, 2)
        active = np.ones(track_count, dtype=bool)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        for frame_index in range(1, len(frames)):
            following, status, _error = cv2.calcOpticalFlowPyrLK(
                gray[frame_index - 1],
                gray[frame_index],
                current,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=criteria,
            )
            if following is None or status is None:
                active[:] = False
                tracks[frame_index] = tracks[frame_index - 1]
                continue
            backward, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
                gray[frame_index],
                gray[frame_index - 1],
                following,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=criteria,
            )
            next_points = following.reshape(-1, 2)
            if backward is None or backward_status is None:
                consistent = np.zeros(track_count, dtype=bool)
            else:
                backward_points = backward.reshape(-1, 2)
                round_trip = np.linalg.norm(backward_points - current.reshape(-1, 2), axis=1)
                consistent = round_trip <= 2.0
                consistent &= backward_status.reshape(-1).astype(bool)
            inside = (
                (next_points[:, 0] >= 0)
                & (next_points[:, 0] < width)
                & (next_points[:, 1] >= 0)
                & (next_points[:, 1] < height)
            )
            active &= status.reshape(-1).astype(bool) & consistent & inside
            tracks[frame_index] = np.where(
                active[:, None],
                next_points,
                tracks[frame_index - 1],
            )
            visibility[frame_index] = active
            current = tracks[frame_index].reshape(-1, 1, 2).astype(np.float32)
        return TrackObservations(
            tracks=tracks,
            visibility=visibility,
            backend="opencv-lk",
            device="cpu",
        )


class CoTracker3Backend:
    def __init__(self, device: str) -> None:
        self.device = _resolve_torch_device(device)
        self._model: Any | None = None
        self._model_device: str | None = None

    def _model_for(self, device: str) -> Any:
        import torch

        if self._model is None:
            self._model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                trust_repo=True,
            ).eval()
        if self._model_device != device:
            self._model = self._model.to(device)
            self._model_device = device
        return self._model

    def _run(self, frames: np.ndarray, grid_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
        import torch

        model = self._model_for(device)
        video = (
            torch.from_numpy(np.ascontiguousarray(frames))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(device)
        )
        with torch.inference_mode():
            predicted_tracks, predicted_visibility = model(video, grid_size=grid_size)
        tracks = predicted_tracks[0].detach().float().cpu().numpy()
        visibility = predicted_visibility[0].detach().float().cpu().numpy()
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        return tracks, visibility >= 0.5

    def track(self, frames: np.ndarray, *, grid_size: int) -> TrackObservations:
        actual_device = self.device
        try:
            tracks, visibility = self._run(frames, grid_size, actual_device)
        except RuntimeError as error:
            if actual_device != "mps":
                raise DisplacementMetricError("CoTracker3 inference failed") from error
            tracks, visibility = self._run(frames, grid_size, "cpu")
            actual_device = "cpu-fallback-after-mps"
        return TrackObservations(
            tracks=tracks,
            visibility=visibility,
            backend="facebookresearch/co-tracker:cotracker3_offline",
            device=actual_device,
        )


class HistogramAppearanceBackend:
    def encode(self, crops: Sequence[np.ndarray]) -> AppearanceObservations:
        features = []
        for crop in crops:
            channels = []
            for channel in range(3):
                histogram, _edges = np.histogram(
                    crop[..., channel],
                    bins=24,
                    range=(0, 256),
                    density=False,
                )
                channels.append(histogram.astype(np.float64))
            feature = np.concatenate(channels)
            norm = np.linalg.norm(feature)
            features.append(feature / max(norm, 1e-12))
        return AppearanceObservations(
            features=np.stack(features),
            backend="rgb-histogram-24x3",
            device="cpu",
        )


class DINOv2AppearanceBackend:
    def __init__(self, device: str) -> None:
        self.device = _resolve_torch_device(device)
        self._model: Any | None = None
        self._model_device: str | None = None

    def _model_for(self, device: str) -> Any:
        import torch

        if self._model is None:
            self._model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                trust_repo=True,
            ).eval()
        if self._model_device != device:
            self._model = self._model.to(device)
            self._model_device = device
        return self._model

    def _run(self, crops: Sequence[np.ndarray], device: str) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        model = self._model_for(device)
        batch = np.stack(
            [
                cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                for crop in crops
            ]
        )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(batch))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        standard_deviation = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / standard_deviation).to(device)
        with torch.inference_mode():
            features = model(tensor)
            features = functional.normalize(features.float(), dim=-1)
        return features.cpu().numpy()

    def encode(self, crops: Sequence[np.ndarray]) -> AppearanceObservations:
        actual_device = self.device
        try:
            features = self._run(crops, actual_device)
        except RuntimeError as error:
            if actual_device != "mps":
                raise DisplacementMetricError("DINOv2 inference failed") from error
            features = self._run(crops, "cpu")
            actual_device = "cpu-fallback-after-mps"
        return AppearanceObservations(
            features=features,
            backend="facebookresearch/dinov2:dinov2_vits14",
            device=actual_device,
        )


def _validate_frames(frames: np.ndarray, config: MetricConfig) -> np.ndarray:
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise DisplacementMetricError("frames must have shape [time,height,width,3]")
    if value.shape[:3] != (
        config.expected_frames,
        config.expected_height,
        config.expected_width,
    ):
        raise DisplacementMetricError(
            "clip geometry must be exactly "
            f"{config.expected_frames} frames at "
            f"{config.expected_height}x{config.expected_width}"
        )
    if value.dtype != np.uint8:
        if not np.issubdtype(value.dtype, np.number):
            raise DisplacementMetricError("frames must contain numeric pixels")
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 255:
            raise DisplacementMetricError("frame pixels must be finite values in [0,255]")
        value = np.rint(value).astype(np.uint8)
    return np.ascontiguousarray(value)


def _load_png_directory(source: Path) -> np.ndarray:
    paths = [
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    ]
    if not paths:
        raise DisplacementMetricError("PNG directory contains no PNG frames")
    numeric_stems = [path.stem.isdecimal() for path in paths]
    if any(numeric_stems) and not all(numeric_stems):
        raise DisplacementMetricError("PNG frame names cannot mix numeric and non-numeric stems")
    if all(numeric_stems):
        numeric_indices = [int(path.stem) for path in paths]
        if len(set(numeric_indices)) != len(numeric_indices):
            raise DisplacementMetricError("PNG frame names contain duplicate numeric indices")
        paths.sort(key=lambda path: int(path.stem))
    else:
        paths.sort(key=lambda path: path.name)
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise DisplacementMetricError(f"could not decode PNG frame {path.name}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        return np.stack(frames)
    except ValueError as error:
        raise DisplacementMetricError("PNG frames do not share one geometry") from error


def _load_mp4(source: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise DisplacementMetricError("could not open video clip")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise DisplacementMetricError("video clip decoded no frames")
    try:
        return np.stack(frames)
    except ValueError as error:
        raise DisplacementMetricError("decoded video frames changed geometry") from error


def load_clip(source: str | Path, *, config: MetricConfig) -> np.ndarray:
    path = Path(source).expanduser()
    if path.is_dir():
        frames = _load_png_directory(path)
    elif path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        frames = _load_mp4(path)
    else:
        raise DisplacementMetricError("source must be an MP4/MOV file or PNG directory")
    return _validate_frames(frames, config)


def _sample_indices(frame_count: int, stride: int) -> np.ndarray:
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return np.asarray(indices, dtype=np.int64)


def _translation_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    delta = np.median(destination - source, axis=0)
    return np.asarray(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _fit_homographies(
    tracks: np.ndarray,
    visibility: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count, track_count, _coordinates = tracks.shape
    if candidate_mask.shape != (track_count,):
        raise DisplacementMetricError("camera candidate mask shape is invalid")
    homographies = np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)
    inlier_fractions = np.ones(frame_count, dtype=np.float64)
    reprojection_errors = np.zeros(frame_count, dtype=np.float64)
    reference = tracks[0]
    for frame_index in range(1, frame_count):
        usable = candidate_mask & visibility[0] & visibility[frame_index]
        source = reference[usable].astype(np.float32)
        destination = tracks[frame_index, usable].astype(np.float32)
        if len(source) < 4:
            raise _CameraCompensationUnavailable(
                "too few persistent background tracks for camera compensation"
            )
        homography, inliers = cv2.findHomography(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if homography is None or not np.isfinite(homography).all():
            homography = _translation_homography(source, destination)
            inlier_mask = np.ones(len(source), dtype=bool)
        else:
            inlier_mask = (
                np.ones(len(source), dtype=bool)
                if inliers is None
                else inliers.reshape(-1).astype(bool)
            )
        predicted = cv2.perspectiveTransform(source[None], homography)[0]
        errors = np.linalg.norm(predicted - destination, axis=1)
        homographies[frame_index] = homography / homography[2, 2]
        inlier_fractions[frame_index] = float(np.mean(inlier_mask))
        reprojection_errors[frame_index] = float(np.median(errors[inlier_mask]))
    return homographies, inlier_fractions, reprojection_errors


def _predicted_global_tracks(reference: np.ndarray, homographies: np.ndarray) -> np.ndarray:
    predicted = []
    source = reference.astype(np.float32)[None]
    for homography in homographies:
        predicted.append(cv2.perspectiveTransform(source, homography)[0])
    return np.stack(predicted)


def _spatial_component(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    energy: np.ndarray,
    *,
    grid_size: int,
    width: int,
    height: int,
) -> np.ndarray:
    if len(candidate_indices) <= 3:
        return candidate_indices
    candidate_points = points[candidate_indices]
    radius = 1.8 * min(width, height) / max(grid_size, 4)
    distances = np.linalg.norm(
        candidate_points[:, None, :] - candidate_points[None, :, :],
        axis=-1,
    )
    unseen = set(range(len(candidate_indices)))
    components: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = set(np.flatnonzero(distances[current] <= radius).tolist()) & unseen
            unseen -= neighbors
            frontier.extend(neighbors)
            component.extend(neighbors)
        components.append(component)

    def component_score(component: list[int]) -> float:
        indices = candidate_indices[np.asarray(component)]
        center = np.median(points[indices], axis=0)
        normalized_center_distance = np.linalg.norm(
            center - np.asarray([width / 2.0, height / 2.0])
        ) / max(width, height)
        center_prior = max(0.5, 1.0 - normalized_center_distance)
        return len(indices) * float(np.median(energy[indices])) * center_prior

    best = max(components, key=component_score)
    return candidate_indices[np.asarray(best)]


def _select_object_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    compensated: np.ndarray,
    config: MetricConfig,
    proposal_mask: np.ndarray | None,
    *,
    minimum_survival: float = 0.5,
) -> np.ndarray:
    height = config.expected_height
    width = config.expected_width
    first = tracks[0]
    survival = visibility.mean(axis=0)
    excursion = np.quantile(np.linalg.norm(compensated, axis=-1), 0.95, axis=0)
    eligible = survival >= minimum_survival
    if config.primary_bbox is not None:
        x, y, box_width, box_height = config.primary_bbox
        selected = (
            (first[:, 0] >= x)
            & (first[:, 0] <= x + box_width)
            & (first[:, 1] >= y)
            & (first[:, 1] <= y + box_height)
            & (survival >= minimum_survival)
        )
        if selected.sum() < 3:
            raise DisplacementMetricError("primary_bbox contains too few persistent tracks")
        return selected
    if proposal_mask is not None:
        x_coordinates = np.clip(np.rint(first[:, 0]).astype(int), 0, width - 1)
        y_coordinates = np.clip(np.rint(first[:, 1]).astype(int), 0, height - 1)
        proposed = proposal_mask[y_coordinates, x_coordinates] & eligible
        if proposed.sum() >= 3:
            eligible_values = excursion[eligible]
            median = float(np.median(eligible_values))
            mad = float(np.median(np.abs(eligible_values - median)))
            motion_threshold = median + max(3.0, 2.5 * mad)
            motion_candidates = np.flatnonzero(proposed & (excursion > motion_threshold))
            if len(motion_candidates) < 3:
                proposed_indices = np.flatnonzero(proposed)
                ranked = proposed_indices[np.argsort(excursion[proposed_indices])]
                motion_candidates = ranked[-min(8, len(ranked)) :]
            component = _spatial_component(
                first,
                motion_candidates,
                excursion,
                grid_size=config.grid_size,
                width=width,
                height=height,
            )
            selected = np.zeros(len(first), dtype=bool)
            selected[component] = True
            return selected
        mask_points = np.argwhere(proposal_mask)
        if len(mask_points):
            mask_center_yx = np.median(mask_points, axis=0)
            mask_center_xy = mask_center_yx[::-1]
            distance = np.linalg.norm(first - mask_center_xy, axis=1)
            distance += (1.0 - survival) * min(width, height) * 0.1
            nearest = np.argsort(distance)[: min(6, len(first))]
            proposed = np.zeros(len(first), dtype=bool)
            proposed[nearest] = True
            return proposed
    values = excursion[eligible]
    if not len(values):
        raise DisplacementMetricError("no tracks survive long enough to identify an object")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + max(3.0, 2.5 * mad)
    candidates = np.flatnonzero(eligible & (excursion > threshold))
    if len(candidates) < 3:
        ranked = np.flatnonzero(eligible)[np.argsort(excursion[eligible])]
        candidates = ranked[-min(8, len(ranked)) :]
    component = _spatial_component(
        first,
        candidates,
        excursion,
        grid_size=config.grid_size,
        width=width,
        height=height,
    )
    selected = np.zeros(len(first), dtype=bool)
    selected[component] = True
    return selected


def _proposal_warp_is_bounded(
    homography: np.ndarray,
    *,
    width: int,
    height: int,
) -> bool:
    corners = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [float(width - 1), 0.0, 1.0],
            [0.0, float(height - 1), 1.0],
            [float(width - 1), float(height - 1), 1.0],
        ],
        dtype=np.float64,
    )
    mapped = corners @ homography.T
    denominators = mapped[:, 2]
    denominator_scale = float(np.max(np.abs(denominators)))
    if (
        not np.isfinite(mapped).all()
        or denominator_scale == 0.0
        or float(np.min(np.abs(denominators))) <= denominator_scale * 1.0e-8
        or float(np.min(denominators)) < 0.0 < float(np.max(denominators))
    ):
        return False
    projected = mapped[:, :2] / denominators[:, None]
    coordinate_limit = 8.0 * max(width, height)
    return bool(
        np.isfinite(projected).all()
        and float(np.max(np.abs(projected))) <= coordinate_limit
    )


def _motion_proposal_mask(
    frames: np.ndarray,
    homographies: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray | None:
    height, width = frames.shape[1:3]
    reference = frames[0].astype(np.float32)
    changed = []
    amplitudes = []
    for frame_index in indices[1:]:
        homography = homographies[frame_index]
        if not _proposal_warp_is_bounded(
            homography,
            width=width,
            height=height,
        ):
            return None
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return None
        if not _proposal_warp_is_bounded(
            inverse,
            width=width,
            height=height,
        ):
            return None
        stabilized = cv2.warpPerspective(
            frames[frame_index],
            inverse,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ).astype(np.float32)
        difference = np.mean(np.abs(reference - stabilized), axis=2)
        amplitudes.append(difference)
        changed.append(difference >= 18.0)
    if not changed:
        return None
    frequency = np.mean(np.stack(changed), axis=0)
    amplitude = np.mean(np.stack(amplitudes), axis=0)
    active = ((frequency >= 0.25) & (amplitude >= 14.0)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, kernel)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, kernel, iterations=2)
    component_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        active,
        connectivity=8,
    )
    candidates = []
    frame_area = height * width
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if not max(12, round(frame_area * 0.001)) <= area <= round(frame_area * 0.45):
            continue
        center = centroids[label]
        center_distance = np.linalg.norm(
            center - np.asarray([width / 2.0, height / 2.0])
        ) / max(width, height)
        center_prior = max(0.5, 1.0 - float(center_distance))
        component_amplitude = float(np.mean(amplitude[labels == label]))
        candidates.append((area * component_amplitude * center_prior, label))
    if not candidates:
        return None
    _score, best_label = max(candidates)
    mask = (labels == best_label).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return mask.astype(bool)


def _camera_and_object_tracks(
    frames: np.ndarray,
    observations: TrackObservations,
    config: MetricConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks = observations.tracks
    visibility = observations.visibility
    if (
        tracks.ndim != 3
        or tracks.shape[0] != config.expected_frames
        or tracks.shape[2] != 2
        or visibility.shape != tracks.shape[:2]
        or tracks.shape[1] < 12
        or not np.isfinite(tracks).all()
    ):
        raise DisplacementMetricError("tracking backend returned an invalid tensor")
    first = tracks[0]
    width = config.expected_width
    height = config.expected_height
    border = (
        (first[:, 0] <= 0.18 * width)
        | (first[:, 0] >= 0.82 * width)
        | (first[:, 1] <= 0.18 * height)
        | (first[:, 1] >= 0.82 * height)
    )
    persistent = visibility.mean(axis=0) >= 0.5
    initial_background = border & persistent
    if initial_background.sum() < 8:
        initial_background = persistent
    initial_h, _initial_inliers, _initial_errors = _fit_homographies(
        tracks,
        visibility,
        initial_background,
    )
    initial_global = _predicted_global_tracks(first, initial_h)
    initial_compensated = tracks - initial_global
    proposal_mask = _motion_proposal_mask(
        frames,
        initial_h,
        _sample_indices(config.expected_frames, config.sample_stride),
    )
    selected = _select_object_tracks(
        tracks,
        visibility,
        initial_compensated,
        config,
        proposal_mask,
    )
    background = (~selected) & persistent
    if background.sum() < 8:
        background = initial_background & (~selected)
    if background.sum() < 4:
        raise _CameraCompensationUnavailable(
            "object selection leaves too few background tracks"
        )
    homographies, inlier_fractions, reprojection_errors = _fit_homographies(
        tracks,
        visibility,
        background,
    )
    global_tracks = _predicted_global_tracks(first, homographies)
    compensated = tracks - global_tracks
    return selected, background, compensated, inlier_fractions, reprojection_errors


def _centroid_trajectory(
    compensated: np.ndarray,
    visibility: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    survival = visibility.mean(axis=0)
    core = selected & (survival >= 0.8)
    if core.sum() < 3:
        core = selected & (survival >= 0.5)
    if core.sum() < 3:
        selected_indices = np.flatnonzero(selected)
        ranked = selected_indices[np.argsort(survival[selected_indices])]
        core = np.zeros_like(selected)
        core[ranked[-min(3, len(ranked)) :]] = True
    if core.sum() < 2:
        raise DisplacementMetricError("primary object has too few surviving tracks")
    centroids = []
    for frame_index in range(len(compensated)):
        usable = core & visibility[frame_index]
        if usable.sum() < 2:
            centroids.append(np.asarray([np.nan, np.nan], dtype=np.float64))
        else:
            centroids.append(np.median(compensated[frame_index, usable], axis=0))
    trajectory = np.stack(centroids)
    valid = np.isfinite(trajectory).all(axis=1)
    if valid.sum() < 2:
        trajectory = np.zeros((len(compensated), 2), dtype=np.float64)
        return trajectory, core
    for coordinate in range(2):
        trajectory[:, coordinate] = np.interp(
            np.arange(len(trajectory)),
            np.flatnonzero(valid),
            trajectory[valid, coordinate],
        )
    return trajectory, core


def _translation_consensus(
    compensated: np.ndarray,
    visibility: np.ndarray,
    core: np.ndarray,
    centroid: np.ndarray,
) -> float:
    values = []
    for frame_index in range(1, len(compensated)):
        usable = core & visibility[frame_index]
        if usable.sum() < 2:
            continue
        vectors = compensated[frame_index, usable]
        vector_magnitudes = np.linalg.norm(vectors, axis=1)
        deviations = np.linalg.norm(vectors - centroid[frame_index], axis=1)
        scale = float(np.median(vector_magnitudes))
        if scale < 1.0:
            continue
        directional = _clip01(float(np.linalg.norm(centroid[frame_index])) / (scale + 1.0))
        concentration = _clip01(
            1.0 - float(np.median(deviations)) / (scale + 1.0)
        )
        values.append(math.sqrt(directional * concentration))
    return 0.0 if not values else float(np.median(values))


def _trajectory_span(trajectory: np.ndarray) -> float:
    differences = trajectory[:, None, :] - trajectory[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=-1)))


def _fixed_object_crops(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    core: np.ndarray,
    indices: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    height, width = frames.shape[1:3]
    centers = []
    extents = []
    for frame_index in indices:
        usable = core & visibility[frame_index]
        points = tracks[frame_index, usable]
        if len(points) < 2:
            points = tracks[frame_index, core]
        lower = np.quantile(points, 0.05, axis=0)
        upper = np.quantile(points, 0.95, axis=0)
        centers.append(np.median(points, axis=0))
        extents.append(upper - lower)
    maximum_extent = np.max(np.stack(extents), axis=0)
    crop_width = int(np.clip(math.ceil(maximum_extent[0] * 1.6), 24, width))
    crop_height = int(np.clip(math.ceil(maximum_extent[1] * 1.6), 24, height))
    crops = []
    boxes = []
    for frame_index, center in zip(indices, centers, strict=True):
        x0 = int(np.clip(round(center[0] - crop_width / 2), 0, width - crop_width))
        y0 = int(np.clip(round(center[1] - crop_height / 2), 0, height - crop_height))
        x1 = x0 + crop_width
        y1 = y0 + crop_height
        crops.append(frames[frame_index, y0:y1, x0:x1])
        boxes.append((x0, y0, x1, y1))
    return crops, boxes


def _flow_guardrail(
    frames: np.ndarray,
    indices: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[float, float, float]:
    raw_magnitudes = []
    warp_residuals = []
    translation_coherences = []
    for pair_index, first_index in enumerate(indices[:-1]):
        second_index = first_index + 1
        first = cv2.cvtColor(frames[first_index], cv2.COLOR_RGB2GRAY)
        second = cv2.cvtColor(frames[second_index], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        x0, y0, x1, y1 = boxes[pair_index]
        pad_x = max(1, round((x1 - x0) * 0.1))
        pad_y = max(1, round((y1 - y0) * 0.1))
        x0_inner = min(x1 - 1, x0 + pad_x)
        x1_inner = max(x0_inner + 1, x1 - pad_x)
        y0_inner = min(y1 - 1, y0 + pad_y)
        y1_inner = max(y0_inner + 1, y1 - pad_y)
        object_flow = flow[y0_inner:y1_inner, x0_inner:x1_inner]
        magnitudes = np.linalg.norm(object_flow, axis=-1)
        raw_magnitude = float(np.mean(magnitudes))
        median_vector = np.median(object_flow.reshape(-1, 2), axis=0)
        translation_coherence = _clip01(
            float(np.linalg.norm(median_vector)) / (raw_magnitude + 1e-6)
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(first.shape[1], dtype=np.float32),
            np.arange(first.shape[0], dtype=np.float32),
        )
        warped_second = cv2.remap(
            second,
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        residual = np.abs(first.astype(np.float32) - warped_second.astype(np.float32))
        warp_residual = float(
            np.mean(residual[y0_inner:y1_inner, x0_inner:x1_inner]) / 255.0
        )
        raw_magnitudes.append(raw_magnitude)
        warp_residuals.append(warp_residual)
        translation_coherences.append(translation_coherence)
    return (
        float(np.mean(raw_magnitudes)),
        float(np.mean(warp_residuals)),
        float(np.median(translation_coherences)),
    )


def _appearance_similarity(features: np.ndarray) -> tuple[float, float]:
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise DisplacementMetricError("appearance backend returned invalid features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    similarities = np.sum(normalized[0] * normalized[1:], axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return float(np.median(similarities)), float(np.quantile(similarities, 0.1))


def _track_backend(config: MetricConfig) -> TrackBackend:
    if config.tracker == "opencv-lk":
        return OpenCVLKTracker()
    return CoTracker3Backend(config.device)


def _appearance_backend(config: MetricConfig) -> AppearanceBackend:
    if config.appearance == "histogram":
        return HistogramAppearanceBackend()
    return DINOv2AppearanceBackend(config.device)


def _score_frames(
    frames: np.ndarray,
    *,
    config: MetricConfig,
    source: str,
    track_backend: TrackBackend,
    appearance_backend: AppearanceBackend,
) -> dict[str, Any]:
    observations = track_backend.track(frames, grid_size=config.grid_size)
    screen_space_displacement = observations.tracks - observations.tracks[0:1]
    camera_compensation_succeeded = True
    degradation_reason: str | None = None
    try:
        selected, background, compensated, inlier_fractions, reprojection_errors = (
            _camera_and_object_tracks(frames, observations, config)
        )
    except _CameraCompensationUnavailable as error:
        camera_compensation_succeeded = False
        degradation_reason = str(error)
        fallback_minimum_survival = 0.5
        try:
            selected = _select_object_tracks(
                observations.tracks,
                observations.visibility,
                screen_space_displacement,
                config,
                proposal_mask=None,
            )
        except DisplacementMetricError as selection_error:
            if str(selection_error) != (
                "no tracks survive long enough to identify an object"
            ):
                raise
            degradation_reason += f"; {selection_error}"
            fallback_minimum_survival = 2.0 / config.expected_frames
            selected = _select_object_tracks(
                observations.tracks,
                observations.visibility,
                screen_space_displacement,
                config,
                proposal_mask=None,
                minimum_survival=fallback_minimum_survival,
            )
        if selected.sum() < 3:
            survival = observations.visibility.mean(axis=0)
            excursion = np.quantile(
                np.linalg.norm(screen_space_displacement, axis=-1),
                0.95,
                axis=0,
            )
            eligible = np.flatnonzero(survival >= fallback_minimum_survival)
            ranked = eligible[np.argsort(excursion[eligible])]
            selected = np.zeros(len(survival), dtype=bool)
            selected[ranked[-min(8, len(ranked)) :]] = True
        persistent = observations.visibility.mean(axis=0) >= 0.5
        background = (~selected) & persistent
        compensated = screen_space_displacement
        inlier_fractions = None
        reprojection_errors = None
    centroid, core = _centroid_trajectory(
        compensated,
        observations.visibility,
        selected,
    )
    screen_space_centroid, _screen_core = _centroid_trajectory(
        screen_space_displacement,
        observations.visibility,
        core,
    )
    indices = _sample_indices(config.expected_frames, config.sample_stride)
    appearance_indices = np.unique(
        np.concatenate(
            (
                indices,
                np.minimum(indices + 1, config.expected_frames - 1),
            )
        )
    )
    crops, _appearance_boxes = _fixed_object_crops(
        frames,
        observations.tracks,
        observations.visibility,
        core,
        appearance_indices,
    )
    _flow_crops, flow_boxes = _fixed_object_crops(
        frames,
        observations.tracks,
        observations.visibility,
        core,
        indices,
    )
    appearance = appearance_backend.encode(crops)
    dino_median, dino_p10 = _appearance_similarity(appearance.features)
    raw_flow, warp_residual, flow_translation_coherence = _flow_guardrail(
        frames,
        indices,
        flow_boxes,
    )

    width = float(config.expected_width)
    core_indices = np.flatnonzero(core)
    endpoint_usable = core & observations.visibility[-1]
    if endpoint_usable.sum() < 2:
        endpoint_usable = core
    endpoint_vectors = compensated[-1, endpoint_usable]
    median_per_track_endpoint = float(np.median(np.linalg.norm(endpoint_vectors, axis=1)))
    centroid_endpoint = float(np.linalg.norm(centroid[-1] - centroid[0]))
    centroid_excursion = float(np.max(np.linalg.norm(centroid - centroid[0], axis=1)))
    measured_span = _trajectory_span(centroid)
    screen_space_span = _trajectory_span(screen_space_centroid)
    camera_compensated_span = (
        measured_span if camera_compensation_succeeded else None
    )
    trajectory_span = (
        min(measured_span, screen_space_span)
        if camera_compensation_succeeded
        else screen_space_span
    )
    steps = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    path_length = float(np.sum(steps))
    straightness = 0.0 if path_length <= 1e-9 else _clip01(centroid_endpoint / path_length)
    track_survival = float(np.mean(observations.visibility[:, core_indices]))
    consensus = _translation_consensus(
        compensated,
        observations.visibility,
        core,
        centroid,
    )

    if camera_compensation_succeeded:
        reference = observations.tracks[0]
        global_tracks = observations.tracks - compensated
        global_motion = np.linalg.norm(global_tracks - reference[None], axis=-1)
        background_global_motion = global_motion[:, background]
        median_global_motion = float(np.median(background_global_motion))
        camera_report = {
            "background_track_count": int(background.sum()),
            "median_global_motion_px": _finite_float(median_global_motion),
            "median_homography_inlier_fraction": _finite_float(
                np.median(inlier_fractions)
            ),
            "median_homography_reprojection_error_px": _finite_float(
                np.median(reprojection_errors)
            ),
        }
    else:
        camera_report = {
            "background_track_count": int(background.sum()),
            "median_global_motion_px": None,
            "median_homography_inlier_fraction": None,
            "median_homography_reprojection_error_px": None,
        }

    span_fraction = trajectory_span / width
    displacement_extent = _smoothstep(0.015, 0.40, span_fraction)
    coherence_guard = math.sqrt(_clip01(track_survival) * _clip01(consensus))
    warp_guard = _clip01(1.0 - warp_residual / 0.16)
    dino_median_guard = _smoothstep(0.65, 0.95, dino_median)
    dino_p10_guard = _smoothstep(0.55, 0.92, dino_p10)
    dino_guard = math.sqrt(dino_median_guard * dino_p10_guard)
    flow_guard = 0.5 + 0.5 * _clip01(flow_translation_coherence)
    appearance_guard = (warp_guard * dino_guard * flow_guard) ** (1.0 / 3.0)
    score = 10.0 * displacement_extent
    score *= 0.25 + 0.75 * coherence_guard
    score *= 0.25 + 0.75 * appearance_guard
    score = min(10.0, max(0.0, score))
    displaced = bool(
        span_fraction >= 0.06
        and consensus >= 0.50
        and track_survival >= 0.50
        and score >= config.displaced_threshold
    )

    report = {
        "schema_version": 1,
        "kind": "coherent-displacement-metrics",
        "purpose": "development-displacement-measurement",
        "authorizes_quality_claim": False,
        "authorizes_performance_claim": False,
        "source": {
            "path": source,
            "frame_count": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "sampled_frame_indices": indices.tolist(),
            "guardrail_frame_indices": appearance_indices.tolist(),
        },
        "models": {
            "tracker": observations.backend,
            "tracker_device": observations.device,
            "appearance": appearance.backend,
            "appearance_device": appearance.device,
            "camera_compensation": (
                "background-ransac-homography-v1"
                if camera_compensation_succeeded
                else "screen-space-only-v1"
            ),
            "flow": "opencv-farneback-warp-v1",
        },
        "foreground": {
            "selected_track_count": int(selected.sum()),
            "persistent_track_count": int(core.sum()),
            "median_net_displacement_px": _finite_float(median_per_track_endpoint),
            "median_net_displacement_fraction_width": _finite_float(
                median_per_track_endpoint / width
            ),
            "centroid_endpoint_displacement_px": _finite_float(centroid_endpoint),
            "centroid_endpoint_displacement_fraction_width": _finite_float(
                centroid_endpoint / width
            ),
            "max_centroid_excursion_px": _finite_float(centroid_excursion),
            "max_centroid_excursion_fraction_width": _finite_float(
                centroid_excursion / width
            ),
            "trajectory_span_px": _finite_float(trajectory_span),
            "trajectory_span_fraction_width": _finite_float(span_fraction),
            "camera_compensated_trajectory_span_px": _finite_float(
                camera_compensated_span
            ) if camera_compensated_span is not None else None,
            "camera_compensated_trajectory_span_fraction_width": (
                _finite_float(camera_compensated_span / width)
                if camera_compensated_span is not None
                else None
            ),
            "screen_space_trajectory_span_px": _finite_float(screen_space_span),
            "screen_space_trajectory_span_fraction_width": _finite_float(
                screen_space_span / width
            ),
            "trajectory_path_length_px": _finite_float(path_length),
            "trajectory_straightness": _finite_float(straightness),
            "translation_consensus": _finite_float(consensus),
            "track_survival": _finite_float(track_survival),
        },
        "camera": camera_report,
        "morph_guardrail": {
            "raw_flow_mean_px": _finite_float(raw_flow),
            "flow_translation_coherence": _finite_float(flow_translation_coherence),
            "flow_warp_residual": _finite_float(warp_residual),
            "dinov2_feature_similarity_median": _finite_float(dino_median),
            "dinov2_feature_similarity_p10": _finite_float(dino_p10),
            "warp_guard": _finite_float(warp_guard),
            "dino_guard": _finite_float(dino_guard),
            "appearance_guard": _finite_float(appearance_guard),
        },
        "score_components": {
            "displacement_extent": _finite_float(displacement_extent),
            "coherence_guard": _finite_float(coherence_guard),
            "appearance_guard": _finite_float(appearance_guard),
        },
        "displacement_score": round(_finite_float(score), 6),
        "displaced_vs_collapsed": displaced,
        "decision_threshold": _finite_float(config.displaced_threshold),
    }
    if not camera_compensation_succeeded:
        report["camera_compensated"] = False
        report["coherence_degraded"] = True
        report["degradation_reason"] = degradation_reason
    return report


def score_frames(
    frames: np.ndarray,
    *,
    config: MetricConfig | None = None,
    track_backend: TrackBackend | None = None,
    appearance_backend: AppearanceBackend | None = None,
) -> dict[str, Any]:
    resolved_config = MetricConfig() if config is None else config
    validated = _validate_frames(frames, resolved_config)
    return _score_frames(
        validated,
        config=resolved_config,
        source="<array>",
        track_backend=track_backend or _track_backend(resolved_config),
        appearance_backend=appearance_backend or _appearance_backend(resolved_config),
    )


def score_clip(
    source: str | Path,
    *,
    config: MetricConfig | None = None,
    track_backend: TrackBackend | None = None,
    appearance_backend: AppearanceBackend | None = None,
) -> dict[str, Any]:
    resolved_config = MetricConfig() if config is None else config
    frames = load_clip(source, config=resolved_config)
    return _score_frames(
        frames,
        config=resolved_config,
        source=str(Path(source).expanduser().resolve()),
        track_backend=track_backend or _track_backend(resolved_config),
        appearance_backend=appearance_backend or _appearance_backend(resolved_config),
    )


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height") from error
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="81-frame MP4/MOV or PNG directory")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--tracker",
        choices=("cotracker3", "opencv-lk"),
        default="cotracker3",
    )
    parser.add_argument(
        "--appearance",
        choices=("dinov2", "histogram"),
        default="dinov2",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--bbox", type=_parse_bbox, help="optional x,y,width,height override")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = MetricConfig(
            tracker=arguments.tracker,
            appearance=arguments.appearance,
            device=arguments.device,
            grid_size=arguments.grid_size,
            sample_stride=arguments.sample_stride,
            primary_bbox=arguments.bbox,
        )
        report = score_clip(arguments.source, config=config)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output is None:
            sys.stdout.write(encoded)
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(encoded, encoding="utf-8")
    except (DisplacementMetricError, OSError) as error:
        print(f"displacement metric failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
