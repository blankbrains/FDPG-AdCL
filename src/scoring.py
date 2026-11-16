"""Shared leakage-free scoring and point-level threshold calibration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def robust_location_scale(values: torch.Tensor) -> tuple[float, float]:
    """Return a finite median and MAD-based scale."""
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("Robust statistics require a non-empty vector")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Robust statistics input contains NaN/Inf")
    median = torch.median(values)
    mad = torch.median((values - median).abs())
    scale = torch.clamp(1.4826 * mad, min=1e-6)
    return float(median), float(scale)


def aggregate_block_scores(
    blocks: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Join already-aggregated blocks with a normal separator point."""
    if not blocks:
        raise ValueError("At least one score block is required")
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for index, (scores, labels) in enumerate(blocks):
        score_array = np.asarray(scores, dtype=np.float64).ravel()
        label_array = np.asarray(labels, dtype=np.int32).ravel()
        if score_array.size == 0 or score_array.shape != label_array.shape:
            raise ValueError("Score and label blocks must be non-empty and aligned")
        if not np.isfinite(score_array).all():
            raise FloatingPointError("Score blocks contain NaN/Inf")
        if not np.isin(label_array, [0, 1]).all():
            raise ValueError("Label blocks must be binary")
        if index:
            score_parts.append(np.array([0.0], dtype=np.float64))
            label_parts.append(np.array([0], dtype=np.int32))
        score_parts.append(score_array)
        label_parts.append(label_array)
    return np.concatenate(score_parts), np.concatenate(label_parts)


def score_embeddings(
    embeddings: torch.Tensor,
    reference: torch.Tensor,
    prototype: torch.Tensor,
    *,
    scoring: str,
    k: int,
    mix_weight: float = 0.5,
    score_stats: Mapping[str, object] | None = None,
    distance_chunk_size: int = 1024,
) -> np.ndarray:
    if embeddings.ndim != 2 or reference.ndim != 2:
        raise ValueError("Embeddings and references must be two-dimensional")
    if embeddings.shape[1] != reference.shape[1]:
        raise ValueError("Embedding and reference dimensions do not match")
    if prototype.shape != (embeddings.shape[1],):
        raise ValueError("Prototype dimension does not match embeddings")
    if len(embeddings) < 1 or len(reference) < 1:
        raise ValueError("Embedding scoring requires non-empty inputs")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive")
    target_device = (
        embeddings.device
        if embeddings.device.type == "cuda"
        else reference.device
    )
    embeddings = torch.nn.functional.normalize(
        embeddings.to(target_device, dtype=torch.float32),
        dim=1,
    )
    reference = torch.nn.functional.normalize(
        reference.to(target_device, dtype=torch.float32),
        dim=1,
    )
    prototype = torch.nn.functional.normalize(
        prototype.to(target_device, dtype=torch.float32),
        dim=0,
    )
    if scoring not in {"cosine", "knn", "hybrid"}:
        raise ValueError("scoring must be 'knn', 'cosine', or 'hybrid'")
    cosine_scores = 1.0 - embeddings @ prototype
    if scoring == "cosine":
        scores = cosine_scores
    else:
        if k < 1 or k > len(reference):
            raise ValueError(f"k must be in [1, {len(reference)}], got {k}")
        knn_chunks = []
        for start in range(0, len(embeddings), distance_chunk_size):
            distances = torch.cdist(
                embeddings[start:start + distance_chunk_size],
                reference,
                p=2,
            )
            knn_chunks.append(
                distances.topk(k, largest=False, dim=1).values.mean(dim=1)
            )
        knn_scores = torch.cat(knn_chunks)
    if scoring == "knn":
        scores = knn_scores
    elif scoring == "hybrid":
        if score_stats is None:
            raise ValueError("Hybrid scoring requires training score statistics")
        weight = float(score_stats.get("mix_weight", mix_weight))
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("Hybrid score mix_weight must be in [0, 1]")
        knn_location = float(score_stats.get("knn_location", float("nan")))
        knn_scale = float(score_stats.get("knn_scale", float("nan")))
        cosine_location = float(score_stats.get("cosine_location", float("nan")))
        cosine_scale = float(score_stats.get("cosine_scale", float("nan")))
        values = np.array(
            [knn_location, knn_scale, cosine_location, cosine_scale],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values[1::2] <= 0):
            raise ValueError("Hybrid score statistics must be finite with positive scales")
        knn_robust = 1.0 + torch.relu(
            (knn_scores - knn_location) / knn_scale
        )
        cosine_robust = 1.0 + torch.relu(
            (cosine_scores - cosine_location) / cosine_scale
        )
        scores = weight * knn_robust + (1.0 - weight) * cosine_robust
    array = scores.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("Embedding scoring produced NaN/Inf")
    return array


def fit_score_statistics(
    reference: torch.Tensor,
    prototype: torch.Tensor,
    *,
    k: int,
    mix_weight: float = 0.5,
) -> dict[str, float | str]:
    """Fit robust hybrid-score statistics from training references only."""
    if reference.ndim != 2 or prototype.ndim != 1:
        raise ValueError("Reference and prototype shapes are invalid")
    if reference.shape[1] != prototype.shape[0]:
        raise ValueError("Reference and prototype dimensions do not match")
    if len(reference) <= k or k < 1:
        raise ValueError("Hybrid statistics require 1 <= k < reference count")
    if not np.isfinite(mix_weight) or not 0.0 <= mix_weight <= 1.0:
        raise ValueError("mix_weight must be finite and in [0, 1]")
    normalized_reference = torch.nn.functional.normalize(
        reference.float(),
        dim=1,
    )
    normalized_prototype = torch.nn.functional.normalize(
        prototype.float(),
        dim=0,
    )
    pairwise = torch.cdist(
        normalized_reference,
        normalized_reference,
        p=2,
    )
    pairwise.fill_diagonal_(float("inf"))
    knn_scores = pairwise.topk(k, largest=False, dim=1).values.mean(dim=1)
    cosine_scores = 1.0 - normalized_reference @ normalized_prototype

    knn_location, knn_scale = robust_location_scale(knn_scores)
    cosine_location, cosine_scale = robust_location_scale(cosine_scores)
    return {
        "method": "leave_one_out_robust_hybrid_v1",
        "mix_weight": float(mix_weight),
        "knn_location": knn_location,
        "knn_scale": knn_scale,
        "cosine_location": cosine_location,
        "cosine_scale": cosine_scale,
    }


def aggregate_window_scores(
    window_scores: np.ndarray,
    starts: list[int],
    *,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    scores = np.asarray(window_scores, dtype=np.float64).ravel()
    if series_length < 1:
        raise ValueError("series_length must be positive")
    if window_size < 2 or window_size > series_length:
        raise ValueError("window_size must be in [2, series_length]")
    if len(scores) != len(starts):
        raise ValueError("Window scores and starts have different lengths")
    if not np.isfinite(scores).all():
        raise FloatingPointError("Window scores contain NaN/Inf")
    total = np.zeros(series_length, dtype=np.float64)
    count = np.zeros(series_length, dtype=np.int64)
    for score, start in zip(scores, starts):
        end = int(start) + window_size
        if start < 0 or end > series_length:
            raise ValueError("Window location is out of aggregation range")
        total[start:end] += score
        count[start:end] += 1
    if np.any(count == 0):
        missing = int((count == 0).sum())
        raise RuntimeError(
            f"Sliding-window scoring left {missing} test points uncovered"
        )
    point_scores = total / count
    if not np.isfinite(point_scores).all():
        raise FloatingPointError("Point-score aggregation produced NaN/Inf")
    return point_scores


def calibrate_point_threshold(
    window_scores: np.ndarray,
    starts: list[int],
    *,
    series_length: int,
    window_size: int,
    percentile: float,
) -> tuple[np.ndarray, float]:
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("threshold percentile must be finite and in (0, 100)")
    point_scores = aggregate_window_scores(
        window_scores,
        starts,
        series_length=series_length,
        window_size=window_size,
    )
    threshold = float(np.percentile(point_scores, percentile))
    if not np.isfinite(threshold):
        raise FloatingPointError("Point-level threshold is NaN/Inf")
    return point_scores, threshold


def _validation_ecdf(
    reference_scores: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_scores, dtype=np.float64).ravel()
    targets = np.asarray(values, dtype=np.float64).ravel()
    if (
        reference.size == 0
        or targets.size == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("ECDF scoring requires finite non-empty arrays")
    ordered = np.sort(reference)
    ranks = np.searchsorted(ordered, targets, side="right")
    return ranks.astype(np.float64) / float(len(ordered) + 1)
