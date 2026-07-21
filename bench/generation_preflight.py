"""CPU-only validation helpers for the recovered generation runner.

This module validates plans before any model, checkpoint, or GPU is opened.  It
deliberately records one availability timestamp per decoded chunk.  A chunk's
frames become observable together, so inventing evenly spaced per-frame times
would overstate the serving behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class PreflightError(ValueError):
    """Raised when a generation plan cannot be executed or evidenced safely."""


@dataclass(frozen=True)
class ChunkReleaseEvent:
    """One measured release boundary for a batch of simultaneously ready frames."""

    chunk_index: int
    first_frame_index: int
    frame_count: int
    ready_ns: int

    def to_dict(self) -> dict[str, int]:
        return {
            "chunk_index": self.chunk_index,
            "first_frame_index": self.first_frame_index,
            "frame_count": self.frame_count,
            "ready_ns": self.ready_ns,
        }


@dataclass(frozen=True)
class ConfirmatoryArtifactCoordinate:
    artifact_id: str
    run_id: str
    prompt_id: str
    seed: int
    split: str = "confirmatory"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "split": self.split,
        }


def _require_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreflightError(f"{label} must be an integer")
    if positive and value <= 0:
        raise PreflightError(f"{label} must be positive")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"{label} must be a non-empty string")
    return value


def rolling_taehv_trim_frames(
    block_index: int,
    prior_context_latents: int,
) -> int:
    """Return the exact recovered causal-TAEHV trim for one latent block.

    The first one-latent decode yields four RGB frames but only its final frame
    belongs to the rollout, so startup trims three. Later decodes prepend up to
    three retained context latents and trim four RGB frames per retained latent.
    """

    block_index = _require_int(block_index, "block_index")
    prior_context_latents = _require_int(
        prior_context_latents,
        "prior_context_latents",
    )
    if block_index < 0:
        raise PreflightError("block_index must be non-negative")
    expected_context_latents = min(block_index, 3)
    if prior_context_latents != expected_context_latents:
        raise PreflightError(
            f"block {block_index} requires {expected_context_latents} prior "
            f"context latents, got {prior_context_latents}"
        )
    if block_index == 0:
        return 3
    return prior_context_latents * 4


def build_chunk_release_events(
    *,
    ready_ns: Sequence[int],
    frame_counts: Sequence[int],
) -> list[ChunkReleaseEvent]:
    """Build honest chunk availability events from measured chunk timestamps."""

    if isinstance(ready_ns, (str, bytes)) or not isinstance(ready_ns, Sequence):
        raise PreflightError("ready_ns must be an array")
    if isinstance(frame_counts, (str, bytes)) or not isinstance(frame_counts, Sequence):
        raise PreflightError("frame_counts must be an array")
    if not ready_ns:
        raise PreflightError("at least one chunk release event is required")
    if len(ready_ns) != len(frame_counts):
        raise PreflightError("ready_ns and frame_counts must have equal length")

    events: list[ChunkReleaseEvent] = []
    first_frame_index = 0
    previous_ready_ns: int | None = None
    for chunk_index, (timestamp, frame_count) in enumerate(
        zip(ready_ns, frame_counts)
    ):
        timestamp = _require_int(timestamp, f"ready_ns[{chunk_index}]")
        if timestamp < 0:
            raise PreflightError(f"ready_ns[{chunk_index}] must be non-negative")
        frame_count = _require_int(
            frame_count,
            f"frame_counts[{chunk_index}]",
            positive=True,
        )
        if previous_ready_ns is not None and timestamp <= previous_ready_ns:
            raise PreflightError("chunk release timestamps must be strictly increasing")
        events.append(
            ChunkReleaseEvent(
                chunk_index=chunk_index,
                first_frame_index=first_frame_index,
                frame_count=frame_count,
                ready_ns=timestamp,
            )
        )
        first_frame_index += frame_count
        previous_ready_ns = timestamp
    return events


def validate_cache_plan(
    *,
    latent_frames: int,
    local_attn_size: int,
    cache_latent_frames: int,
) -> None:
    """Reject rollouts whose global-attention history cannot fit the fixed cache."""

    latent_frames = _require_int(latent_frames, "latent_frames", positive=True)
    cache_latent_frames = _require_int(
        cache_latent_frames,
        "cache_latent_frames",
        positive=True,
    )
    local_attn_size = _require_int(local_attn_size, "local_attn_size")
    if local_attn_size != -1 and local_attn_size <= 0:
        raise PreflightError("local_attn_size must be -1 or a positive integer")
    if local_attn_size == -1 and latent_frames > cache_latent_frames:
        raise PreflightError(
            f"{latent_frames}-latent rollout with local_attn_size=-1 requires a "
            f"cache of at least {latent_frames} latents; fixed cache has "
            f"{cache_latent_frames}"
        )
    if local_attn_size > cache_latent_frames:
        raise PreflightError(
            f"local attention window {local_attn_size} exceeds fixed cache "
            f"capacity {cache_latent_frames}"
        )


def validate_strict_checkpoint_keys(
    *,
    expected_keys: Iterable[str],
    checkpoint_keys: Iterable[str],
) -> None:
    """Prove that strict state-dict loading has no missing or unexpected keys."""

    expected = _validated_key_set(expected_keys, "expected checkpoint keys")
    actual = _validated_key_set(checkpoint_keys, "checkpoint keys")
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise PreflightError(
            "strict checkpoint key mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )


def normalize_fsdp_generator_state_dict(
    state_dict: Mapping[str, Any],
    *,
    expected_keys: Iterable[str],
) -> dict[str, Any]:
    """Normalize the one observed FSDP prefix, then require an exact key set.

    This replaces the recovered runner's unsafe ``strict=False`` retry. Values
    are returned by identity; checkpoint bytes must be hashed separately before
    this structural normalization.
    """

    if not isinstance(state_dict, Mapping):
        raise PreflightError("generator state_dict must be an object")
    prefix = "model._fsdp_wrapped_module."
    normalized: dict[str, Any] = {}
    for index, (raw_key, value) in enumerate(state_dict.items()):
        key = _require_nonempty_string(raw_key, f"generator state_dict key[{index}]")
        if key.startswith(prefix):
            key = "model." + key[len(prefix) :]
        if key in normalized:
            raise PreflightError(f"generator state_dict normalization collision at {key}")
        normalized[key] = value
    validate_strict_checkpoint_keys(
        expected_keys=expected_keys,
        checkpoint_keys=normalized,
    )
    return normalized


def _validated_key_set(values: Iterable[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise PreflightError(f"{label} must be an array")
    try:
        keys = list(values)
    except TypeError as exc:
        raise PreflightError(f"{label} must be an array") from exc
    if not keys:
        raise PreflightError(f"{label} must be non-empty")
    for index, key in enumerate(keys):
        _require_nonempty_string(key, f"{label}[{index}]")
    if len(set(keys)) != len(keys):
        raise PreflightError(f"{label} must not contain duplicates")
    return set(keys)


def validate_confirmatory_artifact_coordinates(
    coordinates: Sequence[Mapping[str, Any]],
) -> list[ConfirmatoryArtifactCoordinate]:
    """Validate the four physical short-trial artifacts required by the gate."""

    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        raise PreflightError("confirmatory_artifacts must be an array")
    if len(coordinates) != 4:
        raise PreflightError(
            "confirmatory_artifacts must contain exactly four coordinates"
        )

    validated: list[ConfirmatoryArtifactCoordinate] = []
    artifact_ids: set[str] = set()
    identities: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(coordinates):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"confirmatory_artifacts[{index}] must be an object")
        artifact_id = _require_nonempty_string(
            raw.get("artifact_id"),
            f"confirmatory_artifacts[{index}].artifact_id",
        )
        run_id = _require_nonempty_string(
            raw.get("run_id"),
            f"confirmatory_artifacts[{index}].run_id",
        )
        prompt_id = _require_nonempty_string(
            raw.get("prompt_id"),
            f"confirmatory_artifacts[{index}].prompt_id",
        )
        seed = _require_int(raw.get("seed"), f"confirmatory_artifacts[{index}].seed")
        if raw.get("split") != "confirmatory":
            raise PreflightError(
                f"confirmatory_artifacts[{index}].split must be confirmatory"
            )
        if artifact_id in artifact_ids:
            raise PreflightError(
                "confirmatory_artifacts require four distinct artifact_id values"
            )
        identity = (run_id, prompt_id, seed)
        if identity in identities:
            raise PreflightError(
                "confirmatory_artifacts require four distinct "
                "(run_id,prompt_id,seed) coordinates"
            )
        artifact_ids.add(artifact_id)
        identities.add(identity)
        validated.append(
            ConfirmatoryArtifactCoordinate(
                artifact_id=artifact_id,
                run_id=run_id,
                prompt_id=prompt_id,
                seed=seed,
            )
        )
    return validated


def validate_preflight_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one JSON-friendly CPU-only preflight plan."""

    if not isinstance(plan, Mapping):
        raise PreflightError("preflight plan must be an object")
    validate_cache_plan(
        latent_frames=plan.get("latent_frames"),
        local_attn_size=plan.get("local_attn_size"),
        cache_latent_frames=plan.get("cache_latent_frames"),
    )
    validate_strict_checkpoint_keys(
        expected_keys=plan.get("expected_checkpoint_keys"),
        checkpoint_keys=plan.get("checkpoint_keys"),
    )
    events = build_chunk_release_events(
        ready_ns=plan.get("chunk_ready_ns"),
        frame_counts=plan.get("chunk_frame_counts"),
    )
    expected_rgb_frames = 1 + 4 * (plan["latent_frames"] - 1)
    released_rgb_frames = sum(event.frame_count for event in events)
    if released_rgb_frames != expected_rgb_frames:
        raise PreflightError(
            f"chunk releases cover {released_rgb_frames} RGB frames; "
            f"the {plan['latent_frames']}-latent rollout requires "
            f"{expected_rgb_frames} RGB frames"
        )
    coordinates = validate_confirmatory_artifact_coordinates(
        plan.get("confirmatory_artifacts")
    )
    checkpoint_keys = sorted(set(plan["checkpoint_keys"]))
    key_set_sha256 = hashlib.sha256(
        json.dumps(checkpoint_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "pass",
        "rollout": {
            "latent_frames": plan["latent_frames"],
            "local_attn_size": plan["local_attn_size"],
            "cache_latent_frames": plan["cache_latent_frames"],
        },
        "checkpoint": {
            "strict": True,
            "key_count": len(checkpoint_keys),
            "key_set_sha256": key_set_sha256,
        },
        "chunk_release_events": [event.to_dict() for event in events],
        "confirmatory_artifact_coordinates": [
            coordinate.to_dict() for coordinate in coordinates
        ],
    }
