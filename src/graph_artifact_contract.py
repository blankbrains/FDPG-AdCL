"""Validate graph-score provenance before a new search or fusion uses it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_results import CalibratedScoreArtifact, load_calibrated_score_artifact
from data.splitting import UnifiedTemporalSplit, split_full_series
from frequency_patch_graph import FrequencyPatchGraphState


GRAPH_PROTOCOL = "frequency_patch_dynamic_sparse_graph_normal_source_v3"
GRAPH_FIT_SCOPE = "training_candidate_intersect_original_normal_source"
GRAPH_STATE_PROTOCOL = "frequency_patch_graph_state_v1"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot identify research inputs")
    digest = hashlib.sha256()
    digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_graph_state(state: FrequencyPatchGraphState) -> None:
    integers = (state.patch_size, state.hop_size, state.frequency_bins)
    if any(type(value) is not int for value in integers) or not (
        state.patch_size >= 4
        and 1 <= state.hop_size <= state.patch_size
        and 0 <= state.frequency_bins <= state.patch_size // 2
        and np.isfinite(state.attention_temperature)
        and state.attention_temperature > 0
        and np.isfinite(state.topk_ratio)
        and 0 < state.topk_ratio <= 1
        and state.attention_mode in {"dynamic", "prior"}
    ):
        raise ValueError("Invalid frozen graph configuration")
    neighbors = np.asarray(state.neighbor_indices)
    if neighbors.ndim != 2 or neighbors.dtype.kind not in "iu":
        raise ValueError("Frozen graph neighbors must be an integer matrix")
    channels, count = neighbors.shape
    dim = 4 + state.frequency_bins
    if not (channels >= 2 and 1 <= count < channels) or (
        np.any(neighbors < 0)
        or np.any(neighbors >= channels)
        or np.any(neighbors == np.arange(channels)[:, None])
        or any(len(np.unique(row)) != count for row in neighbors)
    ):
        raise ValueError("Frozen graph must contain distinct non-self neighbors")
    shapes = {
        "neighbor_priors": (channels, count),
        "token_location": (channels, dim), "token_scale": (channels, dim),
        "coefficients": (channels, 2 * dim, dim), "intercepts": (channels, dim),
        "residual_location": (channels,), "residual_scale": (channels,),
        "reliability": (channels,),
    }
    for name, shape in shapes.items():
        value = np.asarray(getattr(state, name))
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid frozen graph array: {name}")
    if (
        np.any(state.token_scale <= 0) or np.any(state.residual_scale <= 0)
        or np.any(state.neighbor_priors <= 0)
        or not np.allclose(state.neighbor_priors.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
        or np.any(state.reliability < 0) or np.any(state.reliability > 1)
    ):
        raise ValueError("Invalid frozen graph scales, priors or reliability")


def save_graph_state(path: str | Path, state: FrequencyPatchGraphState) -> Path:
    """Persist the fitted state once; this function never fits or updates it."""
    validate_graph_state(state)
    output = Path(path)
    payload = {field.name: getattr(state, field.name) for field in fields(state)}
    with output.open("xb") as handle:
        np.savez_compressed(handle, protocol=GRAPH_STATE_PROTOCOL, **payload)
    return output


def load_graph_state(path: str | Path) -> FrequencyPatchGraphState:
    scalar_types = {
        "patch_size": int, "hop_size": int, "frequency_bins": int,
        "attention_temperature": float, "topk_ratio": float, "attention_mode": str,
    }
    names = {field.name for field in fields(FrequencyPatchGraphState)}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != names | {"protocol"}:
            raise ValueError("Frozen graph archive has missing or unexpected fields")
        if archive["protocol"].shape != () or archive["protocol"].item() != GRAPH_STATE_PROTOCOL:
            raise ValueError("Unsupported frozen graph state protocol")
        payload = {}
        for name in names:
            value = archive[name]
            if name in scalar_types:
                kind = "iu" if scalar_types[name] is int else "fiu" if scalar_types[name] is float else "U"
                if value.shape != () or value.dtype.kind not in kind:
                    raise ValueError(f"Invalid frozen graph scalar: {name}")
                payload[name] = scalar_types[name](value.item())
            else:
                payload[name] = value.copy()
    state = FrequencyPatchGraphState(**payload)
    validate_graph_state(state)
    for name in names - scalar_types.keys():
        getattr(state, name).setflags(write=False)
    return state


def load_bound_graph_state(artifact_path: str | Path, metadata: dict[str, Any]) -> FrequencyPatchGraphState:
    path = Path(artifact_path)
    state_path = path.with_suffix(".state")
    if metadata.get("state_file") != state_path.name or metadata.get("state_protocol") != GRAPH_STATE_PROTOCOL:
        raise ValueError("Graph artifact lacks a bound frozen state; preserve it and derive a new run")
    if metadata.get("state_sha256") != file_sha256(state_path):
        raise ValueError("Frozen graph state hash mismatch")
    state = load_graph_state(state_path)
    for name in ("patch_size", "hop_size", "frequency_bins", "attention_temperature", "topk_ratio", "attention_mode"):
        if metadata.get(name) != getattr(state, name):
            raise ValueError(f"Graph metadata/state configuration mismatch: {name}")
    count = metadata.get("neighbor_count")
    if type(count) is not int or count < 1 or state.neighbor_indices.shape[1] != min(count, len(state.reliability) - 1):
        raise ValueError("Graph metadata/state neighbor-count mismatch")
    return state


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph_metadata(
    artifact_path: str | Path, *, dataset: str, entity: str | None
) -> dict[str, Any]:
    path = Path(artifact_path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete graph artifact pair: {path}, {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid graph metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Graph metadata must be a mapping: {metadata_path}")
    expected = {
        "protocol": GRAPH_PROTOCOL,
        "fit_scope": GRAPH_FIT_SCOPE,
        "dataset": dataset,
        "entity": entity,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Graph identity mismatch for {path}: {key}={metadata.get(key)!r}, "
                f"expected {value!r}. Preserve legacy scores and derive a new isolated artifact."
            )
    integer_fields = ("seed", "train_end", "train_source_length", "normal_fit_end")
    if any(type(metadata.get(key)) is not int for key in integer_fields):
        raise ValueError(f"Graph fit range/seed is missing or invalid: {metadata_path}")
    raw_split = metadata.get("split_protocol")
    if not isinstance(raw_split, dict):
        raise ValueError(f"Graph split metadata is missing: {metadata_path}")
    try:
        split = UnifiedTemporalSplit.from_dict(raw_split)
        reconstructed = split_full_series(
            train_length=split.train_source_length,
            test_length=split.full_length - split.train_source_length,
            seed=split.seed,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid graph split metadata: {metadata_path}") from error
    if split != reconstructed or any(
        metadata[key] != getattr(split, key) for key in integer_fields
    ):
        raise ValueError(f"Graph fit range conflicts with its normal-source split: {metadata_path}")
    if metadata.get("score_sha256") != file_sha256(path):
        raise ValueError(f"Graph score hash mismatch: {path}")
    return metadata


def load_normal_source_graph_artifact(
    artifact_path: str | Path, *, dataset: str, entity: str | None
) -> CalibratedScoreArtifact:
    metadata = load_graph_metadata(artifact_path, dataset=dataset, entity=entity)
    load_bound_graph_state(artifact_path, metadata)
    artifact = load_calibrated_score_artifact(artifact_path)
    percentile = metadata.get("threshold_percentile")
    if (
        type(percentile) not in (int, float)
        or not np.isfinite(percentile)
        or float(percentile) != artifact.threshold_percentile
    ):
        raise ValueError(f"Graph percentile identity mismatch: {artifact_path}")
    expected_threshold = float(np.percentile(
        artifact.validation_point_scores, artifact.threshold_percentile
    ))
    if not np.isclose(artifact.threshold, expected_threshold, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Graph threshold does not match validation calibration: {artifact_path}")
    recorded_threshold = metadata.get("threshold")
    if (
        type(recorded_threshold) not in (int, float)
        or not np.isfinite(recorded_threshold)
        or not np.isclose(recorded_threshold, artifact.threshold, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(f"Graph threshold identity mismatch: {artifact_path}")
    return artifact
