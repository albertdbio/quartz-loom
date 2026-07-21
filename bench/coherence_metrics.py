"""Measure temporal coherence and spatial integrity independently of motion.

The temporal axis combines adjacent-frame appearance similarity with an
optical-flow warp residual.  Raw flow magnitude is diagnostic only: coherent
motion is aligned before residual measurement and can never reduce the score by
being large.  The spatial axis compares patch high-frequency structure with an
early-clip reference so both pointillist noise and structure loss are visible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from bench.displacement_metrics import (
    AppearanceBackend,
    DINOv2AppearanceBackend,
    DisplacementMetricError,
    HistogramAppearanceBackend,
    MetricConfig as DisplacementMetricConfig,
    load_clip,
)


class CoherenceMetricError(ValueError):
    """The clip or a measurement backend violated the coherence contract."""


@dataclass(frozen=True)
class CoherenceConfig:
    expected_frames: int = 81
    expected_height: int = 480
    expected_width: int = 832
    appearance: str = "dinov2"
    flow: str = "raft-small"
    device: str = "auto"
    feature_batch_size: int = 8
    flow_batch_size: int = 4
    patch_size: int = 32
    degradation_drop: float = 2.0

    def __post_init__(self) -> None:
        for value, label in (
            (self.expected_frames, "expected_frames"),
            (self.expected_height, "expected_height"),
            (self.expected_width, "expected_width"),
            (self.feature_batch_size, "feature_batch_size"),
            (self.flow_batch_size, "flow_batch_size"),
            (self.patch_size, "patch_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CoherenceMetricError(f"{label} must be a positive integer")
        if self.expected_frames < 3:
            raise CoherenceMetricError("expected_frames must be at least three")
        if self.patch_size < 8:
            raise CoherenceMetricError("patch_size must be at least eight pixels")
        if self.patch_size > min(self.expected_height, self.expected_width):
            raise CoherenceMetricError("patch_size exceeds the expected frame geometry")
        if self.appearance not in {"dinov2", "histogram"}:
            raise CoherenceMetricError("appearance must be dinov2 or histogram")
        if self.flow not in {"raft-small", "farneback"}:
            raise CoherenceMetricError("flow must be raft-small or farneback")
        if self.flow == "raft-small" and min(
            self.expected_height,
            self.expected_width,
        ) < 256:
            raise CoherenceMetricError(
                "raft-small requires expected frame dimensions of at least 256 pixels"
            )
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise CoherenceMetricError("device must be auto, cuda, mps, or cpu")
        if (
            isinstance(self.degradation_drop, bool)
            or not isinstance(self.degradation_drop, (int, float))
            or not math.isfinite(float(self.degradation_drop))
            or float(self.degradation_drop) <= 0.0
        ):
            raise CoherenceMetricError("degradation_drop must be positive and finite")


@dataclass(frozen=True)
class FlowObservations:
    flows: np.ndarray
    backend: str
    device: str


class FlowBackend(Protocol):
    def estimate(self, frames: np.ndarray) -> FlowObservations:
        ...


def _finite_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CoherenceMetricError("coherence metric produced a non-finite value")
    return result


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _smoothstep(low: float, high: float, value: float) -> float:
    if not low < high:
        raise CoherenceMetricError("smoothstep bounds are invalid")
    position = _clip01((float(value) - low) / (high - low))
    return position * position * (3.0 - 2.0 * position)


def _validate_frames(frames: np.ndarray, config: CoherenceConfig) -> np.ndarray:
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise CoherenceMetricError("frames must have shape [time,height,width,3]")
    if value.shape[:3] != (
        config.expected_frames,
        config.expected_height,
        config.expected_width,
    ):
        raise CoherenceMetricError(
            "clip geometry must be exactly "
            f"{config.expected_frames} frames at "
            f"{config.expected_height}x{config.expected_width}"
        )
    if value.dtype != np.uint8:
        if not np.issubdtype(value.dtype, np.number):
            raise CoherenceMetricError("frames must contain numeric pixels")
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 255:
            raise CoherenceMetricError("frame pixels must be finite values in [0,255]")
        value = np.rint(value).astype(np.uint8)
    return np.ascontiguousarray(value)


def _resolve_torch_device(requested: str) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - CLI dependency guard
        raise CoherenceMetricError("torch is required for learned coherence backends") from error
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise CoherenceMetricError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise CoherenceMetricError("MPS was requested but is unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _appearance_backend(config: CoherenceConfig) -> AppearanceBackend:
    if config.appearance == "histogram":
        return HistogramAppearanceBackend()
    return DINOv2AppearanceBackend(config.device)


class OpenCVFarnebackFlowBackend:
    """Deterministic lightweight fallback used by the synthetic regressions."""

    def estimate(self, frames: np.ndarray) -> FlowObservations:
        grayscale = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        flows = []
        for first, second in zip(grayscale[:-1], grayscale[1:], strict=True):
            flows.append(
                cv2.calcOpticalFlowFarneback(
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
            )
        return FlowObservations(
            flows=np.stack(flows).astype(np.float32, copy=False),
            backend="opencv-farneback-adjacent-v1",
            device="cpu",
        )


class RaftSmallFlowBackend:
    """TorchVision RAFT-Small flow shared across clips in learned batch runs."""

    def __init__(self, device: str, *, batch_size: int = 4) -> None:
        self.device = _resolve_torch_device(device)
        self.batch_size = batch_size
        self._model: Any | None = None
        self._model_device: str | None = None
        self._transforms: Any | None = None

    def _model_for(self, device: str) -> tuple[Any, Any]:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        if self._model is None:
            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights, progress=True).eval()
            self._transforms = weights.transforms()
        if self._model_device != device:
            self._model = self._model.to(device)
            self._model_device = device
        return self._model, self._transforms

    def _run(self, frames: np.ndarray, device: str) -> np.ndarray:
        import torch

        model, transforms = self._model_for(device)
        inference_width = frames.shape[2] // 2
        inference_height = frames.shape[1] // 2
        resized = np.stack(
            [
                cv2.resize(
                    frame,
                    (inference_width, inference_height),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames
            ]
        )
        results: list[np.ndarray] = []
        for start in range(0, len(resized) - 1, self.batch_size):
            end = min(len(resized) - 1, start + self.batch_size)
            first = (
                torch.from_numpy(np.ascontiguousarray(resized[start:end]))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            second = (
                torch.from_numpy(np.ascontiguousarray(resized[start + 1 : end + 1]))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            first, second = transforms(first, second)
            with torch.inference_mode():
                flow = model(first.to(device), second.to(device))[-1]
            results.append(flow.permute(0, 2, 3, 1).float().cpu().numpy())
        return np.concatenate(results, axis=0)

    def estimate(self, frames: np.ndarray) -> FlowObservations:
        actual_device = self.device
        try:
            flows = self._run(frames, actual_device)
        except RuntimeError as error:
            if actual_device != "mps":
                raise CoherenceMetricError("RAFT-Small inference failed") from error
            flows = self._run(frames, "cpu")
            actual_device = "cpu-fallback-after-mps"
        return FlowObservations(
            flows=flows,
            backend="torchvision:raft_small:C_T_V2:half-resolution",
            device=actual_device,
        )


def _flow_backend(config: CoherenceConfig) -> FlowBackend:
    if config.flow == "farneback":
        return OpenCVFarnebackFlowBackend()
    return RaftSmallFlowBackend(config.device, batch_size=config.flow_batch_size)


def _encode_frames(
    frames: np.ndarray,
    *,
    backend: AppearanceBackend,
    batch_size: int,
) -> tuple[np.ndarray, str, str]:
    batches: list[np.ndarray] = []
    backend_name: str | None = None
    device_name: str | None = None
    for start in range(0, len(frames), batch_size):
        try:
            observations = backend.encode(list(frames[start : start + batch_size]))
        except DisplacementMetricError as error:
            raise CoherenceMetricError(str(error)) from error
        features = np.asarray(observations.features)
        expected = min(batch_size, len(frames) - start)
        if (
            features.ndim != 2
            or len(features) != expected
            or not np.isfinite(features).all()
        ):
            raise CoherenceMetricError("appearance backend returned invalid features")
        if backend_name is None:
            backend_name = observations.backend
            device_name = observations.device
        elif observations.backend != backend_name or observations.device != device_name:
            raise CoherenceMetricError("appearance backend identity changed during one clip")
        batches.append(features.astype(np.float64, copy=False))
    if backend_name is None or device_name is None:
        raise CoherenceMetricError("appearance backend returned no observations")
    concatenated = np.concatenate(batches, axis=0)
    norms = np.linalg.norm(concatenated, axis=1, keepdims=True)
    normalized = concatenated / np.maximum(norms, 1e-12)
    return normalized, backend_name, device_name


def _adjacent_feature_similarities(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or len(features) < 2:
        raise CoherenceMetricError("at least two appearance features are required")
    similarities = np.sum(features[:-1] * features[1:], axis=1)
    return np.clip(similarities, -1.0, 1.0)


def _adjacent_flow_observations(
    frames: np.ndarray,
    *,
    backend: FlowBackend,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    observations = backend.estimate(frames)
    flows = np.asarray(observations.flows)
    if (
        flows.ndim != 4
        or flows.shape[0] != len(frames) - 1
        or flows.shape[-1] != 2
        or min(flows.shape[1:3]) <= 0
        or not np.isfinite(flows).all()
    ):
        raise CoherenceMetricError("flow backend returned invalid observations")
    raw_magnitudes: list[float] = []
    warp_residuals: list[float] = []
    height, width = flows.shape[1:3]
    resized_frames = [
        frame
        if frame.shape[:2] == (height, width)
        else cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    for first, second, flow in zip(
        resized_frames[:-1],
        resized_frames[1:],
        flows,
        strict=True,
    ):
        raw_magnitudes.append(float(np.mean(np.linalg.norm(flow, axis=-1))))
        sample_x = grid_x + flow[..., 0]
        sample_y = grid_y + flow[..., 1]
        in_bounds = (
            (sample_x >= 0.0)
            & (sample_x <= width - 1)
            & (sample_y >= 0.0)
            & (sample_y <= height - 1)
        )
        if not np.any(in_bounds):
            raise CoherenceMetricError("flow warp has no in-bounds pixels")
        warped_second = cv2.remap(
            second,
            sample_x,
            sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        residual = np.mean(
            np.abs(first.astype(np.float32) - warped_second.astype(np.float32)),
            axis=-1,
        )
        warp_residuals.append(float(np.mean(residual[in_bounds]) / 255.0))
    return (
        np.asarray(raw_magnitudes, dtype=np.float64),
        np.asarray(warp_residuals, dtype=np.float64),
        observations.backend,
        observations.device,
    )


def _patch_spatial_signature(
    frame: np.ndarray,
    *,
    patch_size: int,
) -> np.ndarray:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height, width = grayscale.shape
    pointillist_values: list[float] = []
    structure_values: list[float] = []
    frequencies_y = np.fft.fftfreq(patch_size)
    frequencies_x = np.fft.fftfreq(patch_size)
    frequency_x, frequency_y = np.meshgrid(frequencies_x, frequencies_y)
    normalized_radius = np.sqrt(
        np.square(frequency_x / 0.5) + np.square(frequency_y / 0.5)
    )
    high_band = normalized_radius >= 0.55
    non_dc = normalized_radius > 0.0
    for y0 in range(0, height - patch_size + 1, patch_size):
        for x0 in range(0, width - patch_size + 1, patch_size):
            patch = grayscale[y0 : y0 + patch_size, x0 : x0 + patch_size]
            centered = patch - float(np.mean(patch))
            variance = float(np.mean(np.square(centered)))
            blurred = cv2.GaussianBlur(
                patch,
                (0, 0),
                sigmaX=1.0,
                sigmaY=1.0,
                borderType=cv2.BORDER_REFLECT101,
            )
            structure_values.append(float(np.mean(np.square(patch - blurred))))
            if variance < (3.0 / 255.0) ** 2:
                pointillist_values.append(0.0)
                continue
            power = np.square(np.abs(np.fft.fft2(centered))).astype(np.float64)
            high_power = power[high_band]
            high_frequency_ratio = float(np.sum(high_power)) / (
                float(np.sum(power[non_dc])) + 1e-12
            )
            spectral_flatness = math.exp(float(np.mean(np.log(high_power + 1e-12)))) / (
                float(np.mean(high_power)) + 1e-12
            )
            # Texture soup has both broad high-frequency energy and a flat
            # spectrum.  Multiplying the ratio by sqrt(flatness) prevents a
            # single clean edge from looking like pointillist noise.
            pointillist_values.append(
                _clip01(high_frequency_ratio)
                * math.sqrt(_clip01(spectral_flatness))
            )
    if not pointillist_values or not structure_values:
        raise CoherenceMetricError("frame geometry produced no spatial patches")
    pointillist = np.asarray(pointillist_values, dtype=np.float64)
    structure = np.asarray(structure_values, dtype=np.float64)
    return np.asarray(
        [
            np.quantile(pointillist, 0.75),
            np.quantile(pointillist, 0.95),
            np.quantile(structure, 0.75),
            np.quantile(structure, 0.99),
        ],
        dtype=np.float64,
    )


def _spatial_observations(
    frames: np.ndarray,
    *,
    patch_size: int,
    reference_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    signatures = np.stack(
        [
            _patch_spatial_signature(frame, patch_size=patch_size)
            for frame in frames
        ]
    )
    reference = np.median(signatures[: reference_end + 1], axis=0)
    return signatures, reference


def _temporal_summary(
    similarities: np.ndarray,
    raw_flow: np.ndarray,
    warp_residuals: np.ndarray,
) -> dict[str, float]:
    if not (len(similarities) == len(raw_flow) == len(warp_residuals)):
        raise CoherenceMetricError("adjacent temporal observations changed length")
    if len(similarities) == 0:
        raise CoherenceMetricError("temporal segment contains no adjacent frame pair")
    similarity_median = float(np.median(similarities))
    similarity_p10 = float(np.quantile(similarities, 0.1))
    residual_median = float(np.median(warp_residuals))
    residual_mean = float(np.mean(warp_residuals))
    residual_p90 = float(np.quantile(warp_residuals, 0.9))
    residual_max = float(np.max(warp_residuals))
    feature_guard = math.sqrt(
        _smoothstep(0.65, 0.95, similarity_median)
        * _smoothstep(0.55, 0.92, similarity_p10)
    )
    warp_guard = math.sqrt(
        (1.0 - _smoothstep(0.04, 0.18, residual_median))
        * (1.0 - _smoothstep(0.07, 0.24, residual_p90))
    )
    score = 10.0 * math.sqrt(feature_guard * warp_guard)
    return {
        "adjacent_feature_similarity_median": round(
            _finite_float(similarity_median), 6
        ),
        "adjacent_feature_similarity_p10": round(_finite_float(similarity_p10), 6),
        "adjacent_flow_warp_residual_mean": round(_finite_float(residual_mean), 6),
        "adjacent_flow_warp_residual_median": round(
            _finite_float(residual_median), 6
        ),
        "adjacent_flow_warp_residual_p90": round(_finite_float(residual_p90), 6),
        "adjacent_flow_warp_residual_max": round(_finite_float(residual_max), 6),
        "raw_flow_magnitude_mean_px": round(
            _finite_float(float(np.mean(raw_flow))), 6
        ),
        "feature_guard": round(_finite_float(feature_guard), 6),
        "flow_warp_guard": round(_finite_float(warp_guard), 6),
        "score": round(_finite_float(score), 6),
    }


def _spatial_summary(
    signatures: np.ndarray,
    *,
    reference: np.ndarray,
    alignment_warp_residual_max: float,
) -> dict[str, float | bool]:
    if signatures.ndim != 2 or signatures.shape[1] != 4:
        raise CoherenceMetricError("spatial signature shape is invalid")
    pointillist_bulk = signatures[:, 0]
    pointillist_tail = signatures[:, 1]
    structure_bulk = signatures[:, 2]
    structure_tail = signatures[:, 3]
    pointillist_bulk_reference = float(reference[0])
    pointillist_tail_reference = float(reference[1])
    structure_bulk_reference = float(reference[2])
    structure_tail_reference = float(reference[3])

    pointillist_bulk_median = float(np.median(pointillist_bulk))
    pointillist_tail_p90 = float(np.quantile(pointillist_tail, 0.9))
    # Bulk catches widespread texture soup; the 5% patch tail catches narrow
    # bands and small regions that a frame q75 necessarily discards.  Both are
    # relative to the clip's own clean opening, so stable detailed scenes and
    # translated sharp objects are not penalized for having texture.
    pointillist_bulk_guard = 1.0 - _smoothstep(
        0.08,
        0.25,
        max(0.0, pointillist_bulk_median - pointillist_bulk_reference),
    )
    pointillist_tail_guard = 1.0 - _smoothstep(
        0.04,
        0.12,
        max(0.0, pointillist_tail_p90 - pointillist_tail_reference),
    )
    pointillist_guard = math.sqrt(
        pointillist_bulk_guard * pointillist_tail_guard
    )

    structure_bulk_median = float(np.median(structure_bulk))
    structure_bulk_p10 = float(np.quantile(structure_bulk, 0.1))
    structure_tail_median = float(np.median(structure_tail))
    structure_tail_p10 = float(np.quantile(structure_tail, 0.1))

    def retention(value: float, reference_value: float) -> float:
        if reference_value <= 1e-10:
            return 1.0 if value <= 1e-10 else value / 1e-10
        return value / reference_value

    structure_bulk_median_retention = retention(
        structure_bulk_median,
        structure_bulk_reference,
    )
    structure_bulk_p10_retention = retention(
        structure_bulk_p10,
        structure_bulk_reference,
    )
    structure_tail_median_retention = retention(
        structure_tail_median,
        structure_tail_reference,
    )
    structure_tail_p10_retention = retention(
        structure_tail_p10,
        structure_tail_reference,
    )
    # Structure can fail in both directions: blur removes the early structure,
    # while pointillist shards create far too much edge energy.  Spatial
    # quantiles make this comparison location-invariant, so coherent transport
    # does not matter, and the 1% tail keeps small foregrounds observable.
    structure_bulk_low_guard = _smoothstep(
        0.25,
        0.75,
        structure_bulk_p10_retention,
    )
    # A normal sharp object entering can legitimately raise structure energy
    # by orders of magnitude.  Suppress only the excess-energy guards when
    # every adjacent transition in the relevant causal prefix is flow-
    # consistent.  A maximum is intentional: median/p90 can erase a single
    # catastrophic transition into stable texture soup, including from a late
    # segment whose own adjacent pairs are subsequently static.
    flow_aligned_structure_entry = alignment_warp_residual_max <= 0.04
    structure_bulk_high_guard = (
        1.0
        if flow_aligned_structure_entry
        else 1.0
        - _smoothstep(
            2.0,
            4.0,
            structure_bulk_median_retention,
        )
    )
    structure_tail_low_guard = _smoothstep(
        0.25,
        0.65,
        structure_tail_p10_retention,
    )
    structure_tail_high_guard = (
        1.0
        if flow_aligned_structure_entry
        else 1.0
        - _smoothstep(
            2.0,
            4.0,
            structure_tail_median_retention,
        )
    )
    structure_bulk_guard = math.sqrt(
        structure_bulk_low_guard * structure_bulk_high_guard
    )
    structure_tail_guard = math.sqrt(
        structure_tail_low_guard * structure_tail_high_guard
    )
    structure_guard = math.sqrt(structure_bulk_guard * structure_tail_guard)
    score = 10.0 * math.sqrt(pointillist_guard * structure_guard)
    return {
        "patch_pointillist_index_bulk_median": round(
            _finite_float(pointillist_bulk_median), 6
        ),
        "patch_pointillist_index_tail_p90": round(
            _finite_float(pointillist_tail_p90), 6
        ),
        "pointillist_bulk_guard": round(
            _finite_float(pointillist_bulk_guard), 6
        ),
        "pointillist_tail_guard": round(
            _finite_float(pointillist_tail_guard), 6
        ),
        "pointillist_guard": round(_finite_float(pointillist_guard), 6),
        "patch_structure_bulk_energy_median": round(
            _finite_float(structure_bulk_median), 8
        ),
        "patch_structure_bulk_energy_p10": round(
            _finite_float(structure_bulk_p10), 8
        ),
        "patch_structure_tail_energy_median": round(
            _finite_float(structure_tail_median), 8
        ),
        "patch_structure_tail_energy_p10": round(
            _finite_float(structure_tail_p10), 8
        ),
        "structure_bulk_median_retention": round(
            _finite_float(structure_bulk_median_retention), 6
        ),
        "structure_bulk_p10_retention": round(
            _finite_float(structure_bulk_p10_retention), 6
        ),
        "structure_tail_median_retention": round(
            _finite_float(structure_tail_median_retention), 6
        ),
        "structure_tail_p10_retention": round(
            _finite_float(structure_tail_p10_retention), 6
        ),
        "structure_bulk_guard": round(_finite_float(structure_bulk_guard), 6),
        "structure_tail_guard": round(_finite_float(structure_tail_guard), 6),
        "structure_bulk_low_guard": round(
            _finite_float(structure_bulk_low_guard), 6
        ),
        "structure_bulk_high_guard": round(
            _finite_float(structure_bulk_high_guard), 6
        ),
        "structure_tail_low_guard": round(
            _finite_float(structure_tail_low_guard), 6
        ),
        "structure_tail_high_guard": round(
            _finite_float(structure_tail_high_guard), 6
        ),
        "flow_aligned_structure_entry": flow_aligned_structure_entry,
        "alignment_warp_residual_max": round(
            _finite_float(alignment_warp_residual_max), 6
        ),
        "structure_guard": round(_finite_float(structure_guard), 6),
        "score": round(_finite_float(score), 6),
    }


def _segment_definitions(frame_count: int) -> tuple[tuple[str, int, int], ...]:
    final_index = frame_count - 1
    early_end = round(final_index * 0.25)
    middle_end = round(final_index * 0.625)
    if not 0 < early_end < middle_end < final_index:
        raise CoherenceMetricError("clip is too short for three degradation segments")
    return (
        ("early", 0, early_end),
        ("middle", early_end, middle_end),
        ("late", middle_end, final_index),
    )


def _score_validated_frames(
    frames: np.ndarray,
    *,
    config: CoherenceConfig,
    source: str,
    appearance_backend: AppearanceBackend,
    flow_backend: FlowBackend,
) -> dict[str, Any]:
    features, backend_name, device_name = _encode_frames(
        frames,
        backend=appearance_backend,
        batch_size=config.feature_batch_size,
    )
    similarities = _adjacent_feature_similarities(features)
    raw_flow, warp_residuals, flow_name, flow_device = _adjacent_flow_observations(
        frames,
        backend=flow_backend,
    )
    segment_definitions = _segment_definitions(len(frames))
    reference_end = min(4, segment_definitions[0][2])
    spatial_signatures, spatial_reference = _spatial_observations(
        frames,
        patch_size=config.patch_size,
        reference_end=reference_end,
    )
    overall_temporal = _temporal_summary(similarities, raw_flow, warp_residuals)
    clip_flow_warp_residual_max = float(np.max(warp_residuals))

    segments: list[dict[str, Any]] = []
    segment_coherence_scores: list[float] = []
    for name, start_frame, end_frame in segment_definitions:
        temporal = _temporal_summary(
            similarities[start_frame:end_frame],
            raw_flow[start_frame:end_frame],
            warp_residuals[start_frame:end_frame],
        )
        # Segment trajectories are causal: a late failure cannot change an
        # earlier segment, while a prior bad transition must remain visible in
        # every later segment even if the damaged frames become mutually static.
        alignment_warp_residual_max = float(np.max(warp_residuals[:end_frame]))
        spatial = _spatial_summary(
            spatial_signatures[start_frame : end_frame + 1],
            reference=spatial_reference,
            alignment_warp_residual_max=alignment_warp_residual_max,
        )
        coherence_score = math.sqrt(temporal["score"] * spatial["score"])
        segment_coherence_scores.append(coherence_score)
        segments.append(
            {
                "coherence_score": round(_finite_float(coherence_score), 6),
                "end_frame": end_frame,
                "name": name,
                "spatial": spatial,
                "start_frame": start_frame,
                "temporal": temporal,
            }
        )

    spatial = _spatial_summary(
        spatial_signatures,
        reference=spatial_reference,
        alignment_warp_residual_max=clip_flow_warp_residual_max,
    )
    temporal_score = overall_temporal["score"]
    spatial_score = spatial["score"]
    coherence_score = math.sqrt(temporal_score * spatial_score)
    early_score = segment_coherence_scores[0]
    late_score = segment_coherence_scores[-1]
    degrades_over_time = bool(
        early_score - late_score >= float(config.degradation_drop)
        and late_score <= 0.75 * early_score
    )
    spatial.update(
        {
            "reference_end_frame": reference_end,
            "reference_patch_pointillist_bulk_index": round(
                _finite_float(spatial_reference[0]), 6
            ),
            "reference_patch_pointillist_tail_index": round(
                _finite_float(spatial_reference[1]), 6
            ),
            "reference_patch_structure_bulk_energy": round(
                _finite_float(spatial_reference[2]), 8
            ),
            "reference_patch_structure_tail_energy": round(
                _finite_float(spatial_reference[3]), 8
            ),
        }
    )
    return {
        "schema_version": 1,
        "kind": "video-coherence-metrics",
        "source": {
            "frame_count": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "path": source,
            "width": int(frames.shape[2]),
        },
        "models": {
            "appearance": backend_name,
            "appearance_device": device_name,
            "flow": flow_name,
            "flow_device": flow_device,
            "spatial": "patch-pointillist-and-structure-reference-v2",
        },
        "temporal": overall_temporal,
        "spatial": spatial,
        "segments": segments,
        "coherence_score": round(_finite_float(coherence_score), 6),
        "temporal_coherence_score": round(_finite_float(temporal_score), 6),
        "spatial_integrity_score": round(_finite_float(spatial_score), 6),
        "degrades_over_time": degrades_over_time,
        "degradation_drop": round(
            _finite_float(early_score - late_score), 6
        ),
        "degradation_rule": {
            "absolute_drop_at_least": _finite_float(config.degradation_drop),
            "late_over_early_at_most": 0.75,
        },
    }


def score_frames(
    frames: np.ndarray,
    *,
    config: CoherenceConfig | None = None,
    appearance_backend: AppearanceBackend | None = None,
    flow_backend: FlowBackend | None = None,
) -> dict[str, Any]:
    resolved_config = CoherenceConfig() if config is None else config
    validated = _validate_frames(frames, resolved_config)
    return _score_validated_frames(
        validated,
        config=resolved_config,
        source="<array>",
        appearance_backend=appearance_backend or _appearance_backend(resolved_config),
        flow_backend=flow_backend or _flow_backend(resolved_config),
    )


def score_clip(
    source: str | Path,
    *,
    config: CoherenceConfig | None = None,
    appearance_backend: AppearanceBackend | None = None,
    flow_backend: FlowBackend | None = None,
) -> dict[str, Any]:
    resolved_config = CoherenceConfig() if config is None else config
    displacement_config = DisplacementMetricConfig(
        expected_frames=resolved_config.expected_frames,
        expected_height=resolved_config.expected_height,
        expected_width=resolved_config.expected_width,
        appearance=resolved_config.appearance,
        device=resolved_config.device,
    )
    try:
        frames = load_clip(source, config=displacement_config)
    except DisplacementMetricError as error:
        raise CoherenceMetricError(str(error)) from error
    return _score_validated_frames(
        frames,
        config=resolved_config,
        source=str(Path(source).expanduser().resolve()),
        appearance_backend=appearance_backend or _appearance_backend(resolved_config),
        flow_backend=flow_backend or _flow_backend(resolved_config),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="81-frame MP4/MOV or PNG directory")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
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
    parser.add_argument(
        "--flow",
        choices=("raft-small", "farneback"),
        default="raft-small",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--flow-batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = CoherenceConfig(
            appearance=arguments.appearance,
            flow=arguments.flow,
            device=arguments.device,
            feature_batch_size=arguments.feature_batch_size,
            flow_batch_size=arguments.flow_batch_size,
            patch_size=arguments.patch_size,
        )
        report = score_clip(arguments.source, config=config)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output is None:
            sys.stdout.write(encoded)
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(encoded, encoding="utf-8")
    except (CoherenceMetricError, OSError) as error:
        print(f"coherence metric failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
