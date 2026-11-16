"""Training-only regime memory and cross-channel scoring assets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

from src.scoring.scoring import fit_score_statistics, robust_location_scale, score_embeddings


REGIME_METHOD = "training_only_regime_memory_v1"
RELATION_METHOD = "ledoit_wolf_relation_v1"
SINGLE_CHANNEL_RELATION_METHOD = "single_channel_zero_relation_v1"


def _normalized_reference(reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim != 2 or len(reference) < 2 or reference.shape[1] < 1:
        raise ValueError("Regime memory requires at least two reference embeddings")
    work = reference.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(work).all():
        raise FloatingPointError("Reference embeddings contain NaN/Inf")
    return torch.nn.functional.normalize(work, dim=1)


def _global_assignments(length: int) -> np.ndarray:
    return np.zeros(length, dtype=np.int64)


def _merge_small_clusters(
    reference: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    *,
    minimum_size: int,
) -> tuple[np.ndarray, bool]:
    counts = np.bincount(assignments, minlength=len(centers))
    valid = np.flatnonzero(counts >= minimum_size)
    if valid.size == 0:
        return _global_assignments(len(reference)), True
    merged = assignments.copy()
    invalid_mask = ~np.isin(assignments, valid)
    if invalid_mask.any():
        distances = np.linalg.norm(
            reference[invalid_mask, None, :] - centers[valid][None, :, :],
            axis=2,
        )
        merged[invalid_mask] = valid[distances.argmin(axis=1)]
    remap = {int(old): new for new, old in enumerate(sorted(np.unique(merged)))}
    return (
        np.asarray([remap[int(value)] for value in merged], dtype=np.int64),
        False,
    )


def fit_regime_memory(
    reference: torch.Tensor,
    *,
    k: int,
    cluster_count: int,
    min_cluster_size: int,
    seed: int,
    mix_weight: float = 0.5,
    kmeans_n_init: int = 10,
) -> dict[str, object]:
    normalized = _normalized_reference(reference)
    if k < 1 or k >= len(normalized):
        raise ValueError("Regime memory requires 1 <= k < reference count")
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")
    if not np.isfinite(mix_weight) or not 0.0 <= mix_weight <= 1.0:
        raise ValueError("mix_weight must be finite and in [0, 1]")
    if kmeans_n_init < 1:
        raise ValueError("kmeans_n_init must be positive")
    array = normalized.numpy()
    requested = min(int(cluster_count), len(array))
    used_global_fallback = False
    if requested == 1:
        assignments = _global_assignments(len(array))
    else:
        assignments = KMeans(
            n_clusters=requested,
            n_init=int(kmeans_n_init),
            random_state=int(seed),
        ).fit_predict(array).astype(np.int64)
        raw_centers = np.stack(
            [array[assignments == index].mean(axis=0) for index in range(requested)]
        )
        assignments, used_global_fallback = _merge_small_clusters(
            array,
            assignments,
            raw_centers,
            minimum_size=max(int(min_cluster_size), int(k) + 1),
        )

    cluster_ids = sorted(np.unique(assignments).tolist())
    regimes: list[dict[str, object]] = []
    centers: list[np.ndarray] = []
    cluster_sizes: list[int] = []
    for cluster_id in cluster_ids:
        local = normalized[
            torch.as_tensor(assignments == cluster_id, dtype=torch.bool)
        ]
        if len(local) <= k:
            assignments = _global_assignments(len(array))
            return fit_regime_memory(
                normalized,
                k=k,
                cluster_count=1,
                min_cluster_size=min_cluster_size,
                seed=seed,
                mix_weight=mix_weight,
                kmeans_n_init=kmeans_n_init,
            )
        prototype = torch.nn.functional.normalize(local.mean(dim=0), dim=0)
        center = prototype.numpy().astype(np.float32, copy=True)
        centers.append(center)
        cluster_sizes.append(len(local))
        regimes.append(
            {
                "reference": local.numpy().astype(np.float32, copy=True),
                "prototype": center.copy(),
                **fit_score_statistics(
                    local,
                    prototype,
                    k=k,
                    mix_weight=float(mix_weight),
                ),
                "fallback": bool(used_global_fallback),
            }
        )

    state: dict[str, object] = {
        "method": REGIME_METHOD,
        "features_dim": int(normalized.shape[1]),
        "k": int(k),
        "seed": int(seed),
        "requested_cluster_count": int(cluster_count),
        "effective_cluster_count": len(regimes),
        "min_cluster_size": int(min_cluster_size),
        "mix_weight": float(mix_weight),
        "kmeans_n_init": int(kmeans_n_init),
        "centers": np.stack(centers).astype(np.float32),
        "assignments": assignments.astype(np.int64),
        "cluster_sizes": cluster_sizes,
        "regimes": regimes,
    }
    validate_regime_memory(state, features_dim=int(normalized.shape[1]))
    return state


def validate_regime_memory(
    state: Mapping[str, object],
    *,
    features_dim: int,
) -> dict[str, object]:
    if not isinstance(state, Mapping):
        raise ValueError("regime_memory_state must be a mapping")
    required = {
        "method",
        "features_dim",
        "k",
        "centers",
        "assignments",
        "cluster_sizes",
        "regimes",
        "effective_cluster_count",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError("regime_memory_state is missing: " + ", ".join(missing))
    if state["method"] != REGIME_METHOD:
        raise ValueError("Unsupported regime memory method")
    if int(state["features_dim"]) != int(features_dim):
        raise ValueError("Regime memory embedding dimension mismatch")
    centers = np.asarray(state["centers"], dtype=np.float32)
    assignments = np.asarray(state["assignments"], dtype=np.int64)
    sizes = [int(value) for value in state["cluster_sizes"]]
    regimes = state["regimes"]
    count = int(state["effective_cluster_count"])
    if (
        centers.shape != (count, features_dim)
        or assignments.ndim != 1
        or len(sizes) != count
        or not isinstance(regimes, list)
        or len(regimes) != count
        or sum(sizes) != len(assignments)
    ):
        raise ValueError("Regime memory shapes are inconsistent")
    if not np.isfinite(centers).all() or np.any(assignments < 0):
        raise ValueError("Regime memory contains invalid centers or assignments")
    k = int(state["k"])
    mix_weight = float(state.get("mix_weight", 0.5))
    kmeans_n_init = int(state.get("kmeans_n_init", 10))
    if (
        not np.isfinite(mix_weight)
        or not 0.0 <= mix_weight <= 1.0
        or kmeans_n_init < 1
    ):
        raise ValueError("Regime memory estimator parameters are invalid")
    for index, regime in enumerate(regimes):
        if not isinstance(regime, Mapping):
            raise ValueError("Each regime must be a mapping")
        reference = np.asarray(regime.get("reference"), dtype=np.float32)
        prototype = np.asarray(regime.get("prototype"), dtype=np.float32)
        if (
            reference.shape != (sizes[index], features_dim)
            or prototype.shape != (features_dim,)
            or sizes[index] <= k
            or not np.isfinite(reference).all()
            or not np.isfinite(prototype).all()
        ):
            raise ValueError("Regime reference assets are invalid")
        values = np.asarray(
            [
                regime.get("knn_location"),
                regime.get("knn_scale"),
                regime.get("cosine_location"),
                regime.get("cosine_scale"),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values[1::2] <= 0):
            raise ValueError("Regime score scale values must be finite and positive")
    return dict(state)


def score_regime_memory(
    embeddings: torch.Tensor,
    state: Mapping[str, object],
    *,
    distance_chunk_size: int,
) -> np.ndarray:
    if embeddings.ndim != 2 or len(embeddings) < 1:
        raise ValueError("Regime scoring requires non-empty 2D embeddings")
    validate_regime_memory(state, features_dim=int(embeddings.shape[1]))
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive")
    device = embeddings.device
    work = torch.nn.functional.normalize(embeddings.float(), dim=1)
    centers = torch.as_tensor(
        state["centers"],
        device=device,
        dtype=torch.float32,
    )
    assignments = torch.cdist(work, centers).argmin(dim=1)
    result = np.empty(len(work), dtype=np.float64)
    regimes = state["regimes"]
    assert isinstance(regimes, list)
    for regime_index, regime in enumerate(regimes):
        mask = assignments == regime_index
        if not bool(mask.any()):
            continue
        assert isinstance(regime, Mapping)
        local_reference = torch.as_tensor(
            regime["reference"],
            device=device,
            dtype=torch.float32,
        )
        local_prototype = torch.as_tensor(
            regime["prototype"],
            device=device,
            dtype=torch.float32,
        )
        local_scores = score_embeddings(
            work[mask],
            local_reference,
            local_prototype,
            scoring="hybrid",
            k=int(state["k"]),
            mix_weight=float(state.get("mix_weight", 0.5)),
            score_stats=regime,
            distance_chunk_size=distance_chunk_size,
        )
        result[mask.detach().cpu().numpy()] = local_scores
    if not np.isfinite(result).all():
        raise FloatingPointError("Regime scoring produced NaN/Inf")
    return result


def _validate_series(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("Relation scoring expects non-empty [time, channels] data")
    if not np.isfinite(array).all():
        raise FloatingPointError("Relation scoring data contains NaN/Inf")
    return array


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if window_size < 2 or window_size > length:
        raise ValueError("window_size must be in [2, series length]")
    if stride < 1:
        raise ValueError("stride must be positive")
    starts = list(range(0, length - window_size + 1, stride))
    tail = length - window_size
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def _raw_relation_window_scores(
    series: np.ndarray,
    starts: list[int],
    *,
    location: np.ndarray,
    precision: np.ndarray,
    window_size: int,
    aggregate_percentile: float,
) -> np.ndarray:
    if not starts:
        raise ValueError("Relation scoring requires at least one window start")
    centered = np.asarray(series, dtype=np.float64) - location
    point_scores = np.einsum(
        "tc,cd,td->t",
        centered,
        precision,
        centered,
        optimize=True,
    )
    point_scores = np.maximum(point_scores, 0.0)
    scores = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        start_value = int(start)
        end = start_value + window_size
        if start_value < 0 or end > len(series):
            raise ValueError("Relation window location is out of range")
        scores[index] = float(
            np.percentile(point_scores[start_value:end], aggregate_percentile)
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError("Relation window scoring produced NaN/Inf")
    return scores


def fit_relation_state(
    standardized_training: np.ndarray,
    *,
    window_size: int,
    stride: int,
    aggregate_percentile: float,
) -> dict[str, object]:
    training = _validate_series(standardized_training)
    if (
        not np.isfinite(aggregate_percentile)
        or not 0.0 < aggregate_percentile < 100.0
    ):
        raise ValueError("aggregate_percentile must be finite and in (0, 100)")
    starts = _window_starts(len(training), int(window_size), int(stride))
    channels = int(training.shape[1])
    if channels == 1:
        state: dict[str, object] = {
            "method": SINGLE_CHANNEL_RELATION_METHOD,
            "channels": 1,
            "window_size": int(window_size),
            "aggregate_percentile": float(aggregate_percentile),
        }
        validate_relation_state(state, channels=1, window_size=int(window_size))
        return state

    estimator = LedoitWolf(assume_centered=False).fit(training)
    location = np.asarray(estimator.location_, dtype=np.float64)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    precision = 0.5 * (precision + precision.T)
    raw_scores = _raw_relation_window_scores(
        training,
        starts,
        location=location,
        precision=precision,
        window_size=int(window_size),
        aggregate_percentile=float(aggregate_percentile),
    )
    score_location, score_scale = robust_location_scale(
        torch.as_tensor(raw_scores, dtype=torch.float32)
    )
    state = {
        "method": RELATION_METHOD,
        "channels": channels,
        "window_size": int(window_size),
        "aggregate_percentile": float(aggregate_percentile),
        "location": location.astype(np.float32),
        "precision": precision.astype(np.float32),
        "score_location": score_location,
        "score_scale": score_scale,
    }
    validate_relation_state(
        state,
        channels=channels,
        window_size=int(window_size),
    )
    return state


def validate_relation_state(
    state: Mapping[str, object],
    *,
    channels: int,
    window_size: int,
) -> dict[str, object]:
    if not isinstance(state, Mapping):
        raise ValueError("relation_state must be a mapping")
    method = state.get("method")
    if method not in {RELATION_METHOD, SINGLE_CHANNEL_RELATION_METHOD}:
        raise ValueError("Unsupported relation scoring method")
    if int(state.get("channels", -1)) != int(channels):
        raise ValueError("Relation state channel count mismatch")
    if int(state.get("window_size", -1)) != int(window_size):
        raise ValueError("Relation state window_size mismatch")
    percentile = float(state.get("aggregate_percentile", float("nan")))
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("Relation state aggregate_percentile is invalid")
    if method == SINGLE_CHANNEL_RELATION_METHOD:
        if channels != 1:
            raise ValueError("Single-channel relation state requires one channel")
        return dict(state)

    location = np.asarray(state.get("location"), dtype=np.float32)
    precision = np.asarray(state.get("precision"), dtype=np.float32)
    values = np.asarray(
        [
            state.get("score_location"),
            state.get("score_scale"),
        ],
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
        raise ValueError("Relation state contains invalid location, precision, or scale")
    return dict(state)


def score_relation_windows(
    standardized_series: np.ndarray,
    starts: list[int],
    state: Mapping[str, object],
) -> np.ndarray:
    series = _validate_series(standardized_series)
    channels = int(series.shape[1])
    window_size = int(state.get("window_size", -1))
    validated = validate_relation_state(
        state,
        channels=channels,
        window_size=window_size,
    )
    if validated["method"] == SINGLE_CHANNEL_RELATION_METHOD:
        return np.zeros(len(starts), dtype=np.float64)
    raw_scores = _raw_relation_window_scores(
        series,
        starts,
        location=np.asarray(validated["location"], dtype=np.float64),
        precision=np.asarray(validated["precision"], dtype=np.float64),
        window_size=window_size,
        aggregate_percentile=float(validated["aggregate_percentile"]),
    )
    robust = 1.0 + np.maximum(
        (raw_scores - float(validated["score_location"]))
        / float(validated["score_scale"]),
        0.0,
    )
    if not np.isfinite(robust).all():
        raise FloatingPointError("Normalized relation scores contain NaN/Inf")
    return robust.astype(np.float64, copy=False)


def fuse_enhanced_scores(
    regime_scores: np.ndarray,
    relation_scores: np.ndarray,
    *,
    relation_weight: float,
) -> np.ndarray:
    regime = np.asarray(regime_scores, dtype=np.float64).ravel()
    relation = np.asarray(relation_scores, dtype=np.float64).ravel()
    if regime.size == 0 or regime.shape != relation.shape:
        raise ValueError("Regime and relation scores must be non-empty and aligned")
    if not np.isfinite(regime).all() or not np.isfinite(relation).all():
        raise FloatingPointError("Enhanced score inputs contain NaN/Inf")
    weight = float(relation_weight)
    if not np.isfinite(weight) or weight not in {0.0, 0.25, 0.5}:
        raise ValueError("relation_weight must be one of 0, 0.25, or 0.5")
    fused = (1.0 - weight) * regime + weight * relation
    if not np.isfinite(fused).all():
        raise FloatingPointError("Enhanced score fusion produced NaN/Inf")
    return fused
