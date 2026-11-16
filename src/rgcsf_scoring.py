"""Regime-gated channel-sparse scoring for multivariate AdCL-FD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from enhanced_scoring import (
    _raw_relation_window_scores,
    robust_location_scale,
    validate_regime_memory,
)
from scoring import aggregate_window_scores


CHANNEL_METHOD = "regime_channel_prototype_v1"
RELATION_METHOD = "regime_ledoit_wolf_relation_v1"
CALIBRATION_METHOD = "rgcsf_validation_ecdf_gate_v1"


def assign_regimes(
    embeddings: torch.Tensor,
    regime_state: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2 or len(embeddings) < 1:
        raise ValueError("Regime assignment requires non-empty 2D embeddings")
    validate_regime_memory(
        regime_state,
        features_dim=int(embeddings.shape[1]),
    )
    work = torch.nn.functional.normalize(embeddings.float(), dim=1)
    centers = torch.as_tensor(
        regime_state["centers"],
        device=work.device,
        dtype=torch.float32,
    )
    distances = torch.cdist(work, centers)
    assignments = distances.argmin(dim=1)
    if distances.shape[1] == 1:
        confidence = torch.ones(len(work), device=work.device)
    else:
        nearest = distances.topk(2, dim=1, largest=False).values
        confidence = (
            1.0 - nearest[:, 0] / nearest[:, 1].clamp_min(1e-6)
        ).clamp(0.0, 1.0)
    return (
        assignments.detach().cpu().numpy().astype(np.int64),
        confidence.detach().cpu().numpy().astype(np.float64),
    )


def fit_channel_regime_state(
    channel_reference: torch.Tensor,
    regime_state: Mapping[str, object],
    *,
    top_fraction: float,
) -> dict[str, object]:
    reference = channel_reference.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if (
        reference.ndim != 3
        or len(reference) < 2
        or reference.shape[1] < 1
        or reference.shape[2] < 2
        or not torch.isfinite(reference).all()
    ):
        raise ValueError(
            "Channel references must be finite [windows, channels, features]"
        )
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    assignments = np.asarray(
        regime_state.get("assignments"),
        dtype=np.int64,
    )
    regime_count = int(regime_state.get("effective_cluster_count", -1))
    if assignments.shape != (len(reference),) or regime_count < 1:
        raise ValueError("Channel references do not align with regime memory")
    normalized = torch.nn.functional.normalize(reference, dim=-1)
    regimes: list[dict[str, object]] = []
    for regime_index in range(regime_count):
        mask = torch.as_tensor(
            assignments == regime_index,
            dtype=torch.bool,
        )
        local = normalized[mask]
        if len(local) < 2:
            raise ValueError("Each channel regime requires at least two windows")
        prototypes = torch.nn.functional.normalize(
            local.mean(dim=0),
            dim=-1,
        )
        distances = 1.0 - (local * prototypes.unsqueeze(0)).sum(dim=-1)
        locations = torch.empty(reference.shape[1], dtype=torch.float32)
        scales = torch.empty(reference.shape[1], dtype=torch.float32)
        for channel in range(reference.shape[1]):
            location, scale = robust_location_scale(distances[:, channel])
            locations[channel] = location
            scales[channel] = scale
        regimes.append(
            {
                "prototypes": prototypes.numpy().astype(np.float32),
                "locations": locations.numpy().astype(np.float32),
                "scales": scales.numpy().astype(np.float32),
            }
        )
    state: dict[str, object] = {
        "method": CHANNEL_METHOD,
        "channels": int(reference.shape[1]),
        "features_dim": int(reference.shape[2]),
        "regime_count": regime_count,
        "top_fraction": float(top_fraction),
        "top_count": max(
            1,
            int(np.ceil(reference.shape[1] * float(top_fraction))),
        ),
        "regimes": regimes,
    }
    validate_channel_regime_state(state)
    return state


def validate_channel_regime_state(
    state: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(state, Mapping) or state.get("method") != CHANNEL_METHOD:
        raise ValueError("Unsupported channel regime state")
    channels = int(state.get("channels", -1))
    features_dim = int(state.get("features_dim", -1))
    regime_count = int(state.get("regime_count", -1))
    top_count = int(state.get("top_count", -1))
    regimes = state.get("regimes")
    if (
        channels < 1
        or features_dim < 2
        or regime_count < 1
        or not 1 <= top_count <= channels
        or not isinstance(regimes, list)
        or len(regimes) != regime_count
    ):
        raise ValueError("Channel regime state shapes are invalid")
    for regime in regimes:
        if not isinstance(regime, Mapping):
            raise ValueError("Channel regime entry must be a mapping")
        prototypes = np.asarray(regime.get("prototypes"), dtype=np.float32)
        locations = np.asarray(regime.get("locations"), dtype=np.float32)
        scales = np.asarray(regime.get("scales"), dtype=np.float32)
        if (
            prototypes.shape != (channels, features_dim)
            or locations.shape != (channels,)
            or scales.shape != (channels,)
            or not np.isfinite(prototypes).all()
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or np.any(scales <= 0)
        ):
            raise ValueError("Channel regime assets are invalid")
    return dict(state)


def score_channel_regimes(
    channel_embeddings: torch.Tensor,
    global_embeddings: torch.Tensor,
    regime_state: Mapping[str, object],
    channel_state: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validated = validate_channel_regime_state(channel_state)
    if (
        channel_embeddings.ndim != 3
        or len(channel_embeddings) != len(global_embeddings)
        or channel_embeddings.shape[1] != int(validated["channels"])
        or channel_embeddings.shape[2] != int(validated["features_dim"])
        or not torch.isfinite(channel_embeddings).all()
    ):
        raise ValueError("Channel embeddings are incompatible with channel state")
    assignments, confidence = assign_regimes(global_embeddings, regime_state)
    normalized = torch.nn.functional.normalize(
        channel_embeddings.float(),
        dim=-1,
    )
    result = np.empty(len(normalized), dtype=np.float64)
    regimes = validated["regimes"]
    assert isinstance(regimes, list)
    for regime_index, regime in enumerate(regimes):
        positions = np.flatnonzero(assignments == regime_index)
        if positions.size == 0:
            continue
        assert isinstance(regime, Mapping)
        local = normalized[
            torch.as_tensor(
                positions,
                device=normalized.device,
                dtype=torch.long,
            )
        ]
        prototypes = torch.as_tensor(
            regime["prototypes"],
            device=normalized.device,
            dtype=torch.float32,
        )
        locations = torch.as_tensor(
            regime["locations"],
            device=normalized.device,
            dtype=torch.float32,
        )
        scales = torch.as_tensor(
            regime["scales"],
            device=normalized.device,
            dtype=torch.float32,
        )
        distances = 1.0 - (local * prototypes.unsqueeze(0)).sum(dim=-1)
        robust = 1.0 + torch.relu(
            (distances - locations.unsqueeze(0)) / scales.unsqueeze(0)
        )
        top_scores = robust.topk(
            int(validated["top_count"]),
            dim=1,
            largest=True,
        ).values.mean(dim=1)
        result[positions] = top_scores.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Channel regime scoring produced NaN/Inf")
    return result, assignments, confidence


def fit_regime_relation_state(
    standardized_training: np.ndarray,
    reference_starts: Sequence[int],
    regime_state: Mapping[str, object],
    *,
    window_size: int,
    aggregate_percentile: float,
) -> dict[str, object]:
    training = np.asarray(standardized_training, dtype=np.float32)
    starts = np.asarray(reference_starts, dtype=np.int64)
    assignments = np.asarray(
        regime_state.get("assignments"),
        dtype=np.int64,
    )
    regime_count = int(regime_state.get("effective_cluster_count", -1))
    if (
        training.ndim != 2
        or len(training) < window_size
        or training.shape[1] < 1
        or not np.isfinite(training).all()
        or starts.shape != assignments.shape
        or regime_count < 1
        or not np.isfinite(aggregate_percentile)
        or not 0.0 < aggregate_percentile < 100.0
    ):
        raise ValueError("Regime relation fitting inputs are invalid")
    if np.any(starts < 0) or np.any(starts + window_size > len(training)):
        raise ValueError("Reference window starts are out of training range")
    channels = int(training.shape[1])
    regimes: list[dict[str, object]] = []
    for regime_index in range(regime_count):
        local_starts = starts[assignments == regime_index]
        if local_starts.size < 2:
            raise ValueError("Each relation regime requires at least two windows")
        point_mask = np.zeros(len(training), dtype=bool)
        for start in local_starts:
            point_mask[int(start):int(start) + window_size] = True
        local_points = training[point_mask]
        if channels == 1:
            regimes.append(
                {
                    "location": np.zeros(1, dtype=np.float32),
                    "precision": np.ones((1, 1), dtype=np.float32),
                    "score_location": 0.0,
                    "score_scale": 1.0,
                }
            )
            continue
        estimator = LedoitWolf(assume_centered=False).fit(local_points)
        location = np.asarray(estimator.location_, dtype=np.float64)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        precision = 0.5 * (precision + precision.T)
        raw_scores = _raw_relation_window_scores(
            training,
            local_starts.tolist(),
            location=location,
            precision=precision,
            window_size=int(window_size),
            aggregate_percentile=float(aggregate_percentile),
        )
        score_location, score_scale = robust_location_scale(
            torch.as_tensor(raw_scores, dtype=torch.float32)
        )
        regimes.append(
            {
                "location": location.astype(np.float32),
                "precision": precision.astype(np.float32),
                "score_location": score_location,
                "score_scale": score_scale,
            }
        )
    state: dict[str, object] = {
        "method": RELATION_METHOD,
        "channels": channels,
        "regime_count": regime_count,
        "window_size": int(window_size),
        "aggregate_percentile": float(aggregate_percentile),
        "regimes": regimes,
    }
    validate_regime_relation_state(state)
    return state


def validate_regime_relation_state(
    state: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(state, Mapping) or state.get("method") != RELATION_METHOD:
        raise ValueError("Unsupported regime relation state")
    channels = int(state.get("channels", -1))
    regime_count = int(state.get("regime_count", -1))
    window_size = int(state.get("window_size", -1))
    percentile = float(state.get("aggregate_percentile", float("nan")))
    regimes = state.get("regimes")
    if (
        channels < 1
        or regime_count < 1
        or window_size < 2
        or not np.isfinite(percentile)
        or not 0.0 < percentile < 100.0
        or not isinstance(regimes, list)
        or len(regimes) != regime_count
    ):
        raise ValueError("Regime relation state shape is invalid")
    for regime in regimes:
        if not isinstance(regime, Mapping):
            raise ValueError("Relation regime entry must be a mapping")
        location = np.asarray(regime.get("location"), dtype=np.float32)
        precision = np.asarray(regime.get("precision"), dtype=np.float32)
        values = np.asarray(
            [regime.get("score_location"), regime.get("score_scale")],
            dtype=np.float64,
        )
        if (
            location.shape != (channels,)
            or precision.shape != (channels, channels)
            or not np.isfinite(location).all()
            or not np.isfinite(precision).all()
            or not np.allclose(precision, precision.T, atol=1e-5)
            or not np.isfinite(values).all()
            or values[1] <= 0
        ):
            raise ValueError("Relation regime assets are invalid")
    return dict(state)


def score_regime_relations(
    standardized_series: np.ndarray,
    starts: Sequence[int],
    assignments: np.ndarray,
    relation_state: Mapping[str, object],
) -> np.ndarray:
    validated = validate_regime_relation_state(relation_state)
    series = np.asarray(standardized_series, dtype=np.float32)
    start_array = np.asarray(starts, dtype=np.int64)
    regime_assignments = np.asarray(assignments, dtype=np.int64)
    if (
        series.ndim != 2
        or series.shape[1] != int(validated["channels"])
        or start_array.shape != regime_assignments.shape
        or not np.isfinite(series).all()
    ):
        raise ValueError("Relation scoring inputs are invalid")
    result = np.empty(len(start_array), dtype=np.float64)
    regimes = validated["regimes"]
    assert isinstance(regimes, list)
    for regime_index, regime in enumerate(regimes):
        positions = np.flatnonzero(regime_assignments == regime_index)
        if positions.size == 0:
            continue
        assert isinstance(regime, Mapping)
        raw = _raw_relation_window_scores(
            series,
            start_array[positions].tolist(),
            location=np.asarray(regime["location"], dtype=np.float64),
            precision=np.asarray(regime["precision"], dtype=np.float64),
            window_size=int(validated["window_size"]),
            aggregate_percentile=float(validated["aggregate_percentile"]),
        )
        result[positions] = 1.0 + np.maximum(
            (raw - float(regime["score_location"]))
            / float(regime["score_scale"]),
            0.0,
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("Regime relation scoring produced NaN/Inf")
    return result


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.asarray(reference, dtype=np.float64).ravel()
    targets = np.asarray(values, dtype=np.float64).ravel()
    if (
        ordered.size < 2
        or targets.size < 1
        or not np.isfinite(ordered).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("ECDF inputs must be finite and non-empty")
    ordered = np.sort(ordered)
    return (
        np.searchsorted(ordered, targets, side="right").astype(np.float64)
        / float(len(ordered) + 1)
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_rgcsf_calibration(
    global_scores: np.ndarray,
    channel_scores: np.ndarray,
    relation_scores: np.ndarray,
    regime_confidence: np.ndarray,
    *,
    starts: Sequence[int],
    series_length: int,
    window_size: int,
    global_weight: float,
    relation_weight: float,
    channel_gate_quantile: float,
    relation_gate_quantile: float,
    gate_temperature: float,
    threshold_percentile: float,
) -> dict[str, object]:
    arrays = [
        np.asarray(values, dtype=np.float64).ravel()
        for values in (
            global_scores,
            channel_scores,
            relation_scores,
            regime_confidence,
        )
    ]
    if (
        any(values.size < 2 for values in arrays)
        or len({values.shape for values in arrays}) != 1
        or any(not np.isfinite(values).all() for values in arrays)
        or not 0.0 <= global_weight <= 1.0
        or not 0.0 <= relation_weight <= 1.0
        or not 0.0 < channel_gate_quantile < 1.0
        or not 0.0 < relation_gate_quantile < 1.0
        or not np.isfinite(gate_temperature)
        or gate_temperature <= 0
        or not 0.0 < threshold_percentile < 100.0
    ):
        raise ValueError("RG-CSF calibration inputs are invalid")
    state: dict[str, object] = {
        "method": CALIBRATION_METHOD,
        "global_reference": np.sort(arrays[0]),
        "channel_reference": np.sort(arrays[1]),
        "relation_reference": np.sort(arrays[2]),
        "global_weight": float(global_weight),
        "relation_weight": float(relation_weight),
        "channel_gate_quantile": float(channel_gate_quantile),
        "relation_gate_quantile": float(relation_gate_quantile),
        "gate_temperature": float(gate_temperature),
        "threshold_percentile": float(threshold_percentile),
    }
    fused = score_rgcsf_fusion(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        state,
    )
    point_scores = aggregate_window_scores(
        fused,
        [int(value) for value in starts],
        series_length=int(series_length),
        window_size=int(window_size),
    )
    state["validation_point_scores"] = point_scores
    state["threshold"] = float(
        np.percentile(point_scores, threshold_percentile)
    )
    validate_rgcsf_calibration(state)
    return state


def validate_rgcsf_calibration(
    state: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(state, Mapping) or state.get("method") != CALIBRATION_METHOD:
        raise ValueError("Unsupported RG-CSF calibration state")
    references = [
        np.asarray(state.get(key), dtype=np.float64).ravel()
        for key in (
            "global_reference",
            "channel_reference",
            "relation_reference",
        )
    ]
    validation_scores = np.asarray(
        state.get("validation_point_scores"),
        dtype=np.float64,
    ).ravel()
    values = np.asarray(
        [
            state.get("global_weight"),
            state.get("relation_weight"),
            state.get("channel_gate_quantile"),
            state.get("relation_gate_quantile"),
            state.get("gate_temperature"),
            state.get("threshold_percentile"),
            state.get("threshold"),
        ],
        dtype=np.float64,
    )
    if (
        any(reference.size < 2 for reference in references)
        or any(not np.isfinite(reference).all() for reference in references)
        or validation_scores.size < 2
        or not np.isfinite(validation_scores).all()
        or not np.isfinite(values).all()
        or not 0.0 <= values[0] <= 1.0
        or not 0.0 <= values[1] <= 1.0
        or not 0.0 < values[2] < 1.0
        or not 0.0 < values[3] < 1.0
        or values[4] <= 0
        or not 0.0 < values[5] < 100.0
        or values[6] <= 0
    ):
        raise ValueError("RG-CSF calibration assets are invalid")
    return dict(state)


def score_rgcsf_fusion(
    global_scores: np.ndarray,
    channel_scores: np.ndarray,
    relation_scores: np.ndarray,
    regime_confidence: np.ndarray,
    calibration_state: Mapping[str, object],
) -> np.ndarray:
    state = dict(calibration_state)
    if state.get("method") != CALIBRATION_METHOD:
        raise ValueError("Unsupported RG-CSF calibration method")
    arrays = [
        np.asarray(values, dtype=np.float64).ravel()
        for values in (
            global_scores,
            channel_scores,
            relation_scores,
            regime_confidence,
        )
    ]
    if (
        any(values.size < 1 for values in arrays)
        or len({values.shape for values in arrays}) != 1
        or any(not np.isfinite(values).all() for values in arrays)
    ):
        raise ValueError("RG-CSF fusion inputs are invalid")
    global_u = _ecdf(
        np.asarray(state["global_reference"], dtype=np.float64),
        arrays[0],
    )
    channel_u = _ecdf(
        np.asarray(state["channel_reference"], dtype=np.float64),
        arrays[1],
    )
    relation_u = _ecdf(
        np.asarray(state["relation_reference"], dtype=np.float64),
        arrays[2],
    )
    global_weight = float(state["global_weight"])
    relation_weight = float(state["relation_weight"])
    channel_quantile = float(state["channel_gate_quantile"])
    relation_quantile = float(state["relation_gate_quantile"])
    temperature = float(state["gate_temperature"])
    base = (
        (1.0 - global_weight) * channel_u
        + global_weight * global_u
    )
    channel_gate = _sigmoid(
        (channel_u - channel_quantile) / temperature
    )
    relation_excess = np.maximum(
        (relation_u - relation_quantile)
        / max(1.0 - relation_quantile, 1e-6),
        0.0,
    )
    fused = (
        base
        + relation_weight
        * np.clip(arrays[3], 0.0, 1.0)
        * channel_gate
        * relation_excess
    )
    if not np.isfinite(fused).all():
        raise FloatingPointError("RG-CSF fusion produced NaN/Inf")
    return fused
