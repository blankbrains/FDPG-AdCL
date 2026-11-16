"""Point-local scoring and aggregation for FP-LIR-AdCL."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


COMPONENT_NAMES = ("repair", "idempotence", "prototype")


def point_components(
    x: torch.Tensor,
    repair: torch.Tensor,
    tokens: torch.Tensor,
    repaired_tokens: torch.Tensor,
    prototype: torch.Tensor,
    *,
    channel_topk_ratio: float,
) -> torch.Tensor:
    if x.shape != repair.shape or x.ndim != 3:
        raise ValueError("Input and repair must have equal [batch, channels, time] shapes")
    if tokens.shape != repaired_tokens.shape or tokens.ndim != 3:
        raise ValueError("Token tensors must have equal [batch, features, time] shapes")
    if tokens.shape[0] != x.shape[0] or tokens.shape[-1] != x.shape[-1]:
        raise ValueError("Point tokens are not aligned with input points")
    if prototype.shape != (tokens.shape[1],):
        raise ValueError("Prototype dimension does not match point tokens")
    if not 0.0 < channel_topk_ratio <= 1.0:
        raise ValueError("channel_topk_ratio must be in (0, 1]")

    residual = (x - repair).abs()
    selected = max(1, int(np.ceil(x.shape[1] * channel_topk_ratio)))
    repair_score = residual.topk(selected, dim=1).values.mean(dim=1)
    idempotence_score = 1.0 - (tokens * repaired_tokens).sum(dim=1)
    prototype_score = 1.0 - torch.einsum(
        "bdt,d->bt",
        tokens,
        torch.nn.functional.normalize(prototype.float(), dim=0),
    )
    result = torch.stack(
        [repair_score, idempotence_score, prototype_score],
        dim=-1,
    )
    if not torch.isfinite(result).all():
        raise FloatingPointError("FP-LIR score components contain NaN/Inf")
    return result


def fit_component_statistics(components: np.ndarray) -> dict[str, list[float] | str]:
    values = np.asarray(components, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("Score components must have shape [points, 3]")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Score components must be finite and non-empty")
    location = np.median(values, axis=0)
    scale = np.median(np.abs(values - location), axis=0) * 1.4826
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1e-6))
    return {
        "method": "training_median_mad_v1",
        "location": location.tolist(),
        "scale": scale.tolist(),
    }


def fuse_components(
    components: np.ndarray,
    statistics: dict[str, object],
    *,
    weights: Iterable[float],
) -> np.ndarray:
    values = np.asarray(components, dtype=np.float64)
    location = np.asarray(statistics.get("location"), dtype=np.float64)
    scale = np.asarray(statistics.get("scale"), dtype=np.float64)
    weight = np.asarray(list(weights), dtype=np.float64)
    expected = (len(COMPONENT_NAMES),)
    if (
        values.ndim != 2
        or values.shape[1] != len(COMPONENT_NAMES)
        or location.shape != expected
        or scale.shape != expected
        or weight.shape != expected
        or not np.isfinite(values).all()
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(weight).all()
        or np.any(scale <= 0)
        or np.any(weight < 0)
        or weight.sum() <= 0
    ):
        raise ValueError("Invalid FP-LIR component fusion state")
    robust = np.maximum((values - location) / scale, 0.0)
    scores = robust @ (weight / weight.sum())
    if not np.isfinite(scores).all():
        raise FloatingPointError("FP-LIR fused scores contain NaN/Inf")
    return scores


def aggregate_local_scores(
    window_scores: np.ndarray,
    starts: list[int],
    *,
    series_length: int,
    window_size: int,
) -> np.ndarray:
    scores = np.asarray(window_scores, dtype=np.float64)
    if scores.shape != (len(starts), window_size):
        raise ValueError("Local window scores must have shape [windows, window_size]")
    if not np.isfinite(scores).all():
        raise FloatingPointError("Local window scores contain NaN/Inf")
    total = np.zeros(series_length, dtype=np.float64)
    count = np.zeros(series_length, dtype=np.int64)
    for local, start in zip(scores, starts):
        end = int(start) + window_size
        if start < 0 or end > series_length:
            raise ValueError("Window location is out of aggregation range")
        total[start:end] += local
        count[start:end] += 1
    if np.any(count == 0):
        raise RuntimeError("FP-LIR local scoring left uncovered points")
    result = total / count
    if not np.isfinite(result).all():
        raise FloatingPointError("FP-LIR point aggregation produced NaN/Inf")
    return result
