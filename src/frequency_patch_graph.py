"""Frequency-patch dynamic sparse channel graph scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencyPatchGraphState:
    patch_size: int
    hop_size: int
    frequency_bins: int
    neighbor_indices: np.ndarray
    neighbor_priors: np.ndarray
    token_location: np.ndarray
    token_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    residual_location: np.ndarray
    residual_scale: np.ndarray
    reliability: np.ndarray
    attention_temperature: float
    topk_ratio: float
    attention_mode: str


def patch_starts(
    length: int,
    patch_size: int,
    hop_size: int,
) -> list[int]:
    if length < patch_size or patch_size < 4:
        raise ValueError("Series must contain at least one patch of size >= 4")
    if hop_size < 1 or hop_size > patch_size:
        raise ValueError("hop_size must be in [1, patch_size]")
    starts = list(range(0, length - patch_size + 1, hop_size))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _validate_data(data: np.ndarray, patch_size: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < patch_size
        or values.shape[1] < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Frequency patch graph requires finite [time, channels>=2] data"
        )
    return values


def extract_frequency_patch_tokens(
    data: np.ndarray,
    starts: list[int],
    *,
    patch_size: int,
    frequency_bins: int,
    chunk_size: int = 512,
) -> np.ndarray:
    values = _validate_data(data, patch_size)
    if (
        not starts
        or frequency_bins < 0
        or frequency_bins > patch_size // 2
        or chunk_size < 1
    ):
        raise ValueError("Invalid frequency patch token configuration")
    if any(start < 0 or start + patch_size > len(values) for start in starts):
        raise ValueError("Patch start is outside the series")
    token_parts: list[np.ndarray] = []
    for offset in range(0, len(starts), chunk_size):
        chunk_starts = starts[offset:offset + chunk_size]
        patches = np.stack(
            [values[start:start + patch_size] for start in chunk_starts],
            axis=0,
        )
        mean = patches.mean(axis=1)
        std = patches.std(axis=1)
        value_range = patches.max(axis=1) - patches.min(axis=1)
        mean_abs_difference = np.abs(np.diff(patches, axis=1)).mean(axis=1)
        if frequency_bins:
            spectrum = np.fft.rfft(patches, axis=1)
            magnitude = np.log1p(
                np.abs(spectrum[:, 1:frequency_bins + 1, :])
            )
            magnitude = np.transpose(magnitude, (0, 2, 1))
        else:
            magnitude = np.empty(
                (len(patches), patches.shape[2], 0),
                dtype=np.float64,
            )
        token_parts.append(
            np.concatenate(
                [
                    mean[..., None],
                    std[..., None],
                    value_range[..., None],
                    mean_abs_difference[..., None],
                    magnitude,
                ],
                axis=2,
            )
        )
    tokens = np.concatenate(token_parts, axis=0)
    if not np.isfinite(tokens).all():
        raise FloatingPointError("Frequency patch tokens contain NaN/Inf")
    return tokens


def neighbor_attention_weights(
    previous_target: np.ndarray,
    current_neighbors: np.ndarray,
    priors: np.ndarray,
    *,
    temperature: float,
    mode: str = "dynamic",
) -> np.ndarray:
    previous = np.asarray(previous_target, dtype=np.float64)
    neighbors = np.asarray(current_neighbors, dtype=np.float64)
    prior = np.asarray(priors, dtype=np.float64)
    if (
        previous.ndim != 2
        or neighbors.ndim != 3
        or previous.shape[0] != neighbors.shape[0]
        or previous.shape[1] != neighbors.shape[2]
        or prior.shape != (neighbors.shape[1],)
        or not np.isfinite(previous).all()
        or not np.isfinite(neighbors).all()
        or not np.isfinite(prior).all()
        or np.any(prior <= 0)
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid frequency-patch neighbor attention inputs")
    if mode == "prior":
        weights = np.broadcast_to(prior[None, :], neighbors.shape[:2]).copy()
        weights /= weights.sum(axis=1, keepdims=True)
        return weights
    if mode != "dynamic":
        raise ValueError("attention mode must be dynamic or prior")
    query_norm = np.linalg.norm(previous, axis=1, keepdims=True)
    key_norm = np.linalg.norm(neighbors, axis=2)
    dot = np.einsum("nd,nkd->nk", previous, neighbors)
    cosine = dot / np.maximum(query_norm * key_norm, 1e-8)
    logits = cosine / temperature + np.log(np.maximum(prior, 1e-6))[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    if not np.isfinite(weights).all():
        raise FloatingPointError("Frequency-patch attention contains NaN/Inf")
    return weights


def _dynamic_neighbor_aggregate(
    previous_target: np.ndarray,
    current_neighbors: np.ndarray,
    priors: np.ndarray,
    *,
    temperature: float,
    mode: str = "dynamic",
) -> np.ndarray:
    weights = neighbor_attention_weights(
        previous_target,
        current_neighbors,
        priors,
        temperature=temperature,
        mode=mode,
    )
    return np.einsum("nk,nkd->nd", weights, current_neighbors)


def _candidate_neighbors(similarity: np.ndarray, channel: int, count: int) -> np.ndarray:
    """Select other channels even when all candidate correlations are tied."""
    strength = np.abs(np.asarray(similarity, dtype=np.float64)).copy()
    if (
        strength.ndim != 1
        or not np.isfinite(strength).all()
        or not 0 <= channel < strength.size
        or not 1 <= count < strength.size
    ):
        raise ValueError("Invalid candidate-neighbor selection inputs")
    strength[channel] = -np.inf
    return np.argsort(strength)[-count:][::-1]


def fit_frequency_patch_graph(
    training_data: np.ndarray,
    *,
    patch_size: int = 32,
    hop_size: int = 8,
    frequency_bins: int = 8,
    neighbor_count: int = 8,
    ridge_alpha: float = 1.0,
    attention_temperature: float = 0.20,
    topk_ratio: float = 0.10,
    max_fit_patches: int = 20_000,
    attention_mode: str = "dynamic",
) -> FrequencyPatchGraphState:
    data = _validate_data(training_data, patch_size)
    channels = int(data.shape[1])
    if (
        neighbor_count < 1
        or not np.isfinite(ridge_alpha)
        or ridge_alpha <= 0
        or not np.isfinite(attention_temperature)
        or attention_temperature <= 0
        or not 0.0 < topk_ratio <= 1.0
        or max_fit_patches < 10
        or attention_mode not in {"dynamic", "prior"}
    ):
        raise ValueError("Invalid frequency patch graph configuration")
    all_starts = patch_starts(len(data), patch_size, hop_size)
    if len(all_starts) > max_fit_patches:
        indices = np.linspace(
            0,
            len(all_starts) - 1,
            max_fit_patches,
            dtype=np.int64,
        )
        starts = [all_starts[index] for index in indices]
    else:
        starts = all_starts
    tokens = extract_frequency_patch_tokens(
        data,
        starts,
        patch_size=patch_size,
        frequency_bins=frequency_bins,
    )
    if len(tokens) < 3:
        raise ValueError("At least three frequency patches are required")
    token_location = np.median(tokens, axis=0)
    token_scale = 1.4826 * np.median(
        np.abs(tokens - token_location[None, ...]),
        axis=0,
    )
    fallback = tokens.std(axis=0)
    token_scale = np.where(
        token_scale > 1e-8,
        token_scale,
        np.maximum(fallback, 1e-6),
    )
    normalized = (tokens - token_location[None, ...]) / token_scale[None, ...]
    flattened = np.transpose(normalized, (1, 0, 2)).reshape(channels, -1)
    similarity = np.corrcoef(flattened)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)

    selected_count = min(int(neighbor_count), channels - 1)
    token_dim = int(normalized.shape[2])
    feature_dim = token_dim * 2
    neighbor_indices = np.empty((channels, selected_count), dtype=np.int64)
    neighbor_priors = np.empty((channels, selected_count), dtype=np.float64)
    coefficients = np.empty(
        (channels, feature_dim, token_dim),
        dtype=np.float64,
    )
    intercepts = np.empty((channels, token_dim), dtype=np.float64)
    residual_location = np.empty(channels, dtype=np.float64)
    residual_scale = np.empty(channels, dtype=np.float64)
    reliability = np.empty(channels, dtype=np.float64)
    identity = np.eye(feature_dim, dtype=np.float64)

    for channel in range(channels):
        neighbors = _candidate_neighbors(similarity[channel], channel, selected_count)
        priors = np.maximum(np.abs(similarity[channel, neighbors]), 1e-4)
        priors /= priors.sum()
        neighbor_indices[channel] = neighbors
        neighbor_priors[channel] = priors
        previous_target = normalized[:-1, channel, :]
        current_neighbors = normalized[1:, neighbors, :]
        aggregate = _dynamic_neighbor_aggregate(
            previous_target,
            current_neighbors,
            priors,
            temperature=attention_temperature,
            mode=attention_mode,
        )
        features = np.concatenate([aggregate, previous_target], axis=1)
        target = normalized[1:, channel, :]
        feature_mean = features.mean(axis=0)
        target_mean = target.mean(axis=0)
        centered_features = features - feature_mean
        centered_target = target - target_mean
        gram = centered_features.T @ centered_features / len(features)
        rhs = centered_features.T @ centered_target / len(features)
        coefficient = np.linalg.solve(
            gram + float(ridge_alpha) * identity,
            rhs,
        )
        intercept = target_mean - feature_mean @ coefficient
        prediction = features @ coefficient + intercept
        residual = np.abs(target - prediction).mean(axis=1)
        location = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - location)))
        if scale <= 1e-8:
            scale = max(float(np.std(residual)), 1e-6)
        denominator = float(np.sum(centered_target ** 2))
        r_squared = (
            1.0 - float(np.sum((target - prediction) ** 2)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        coefficients[channel] = coefficient
        intercepts[channel] = intercept
        residual_location[channel] = location
        residual_scale[channel] = scale
        reliability[channel] = float(np.clip(r_squared, 0.0, 1.0))

    arrays = (
        neighbor_priors,
        token_location,
        token_scale,
        coefficients,
        intercepts,
        residual_location,
        residual_scale,
        reliability,
    )
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Frequency patch graph fitting produced NaN/Inf")
    return FrequencyPatchGraphState(
        patch_size=int(patch_size),
        hop_size=int(hop_size),
        frequency_bins=int(frequency_bins),
        neighbor_indices=neighbor_indices,
        neighbor_priors=neighbor_priors,
        token_location=token_location,
        token_scale=token_scale,
        coefficients=coefficients,
        intercepts=intercepts,
        residual_location=residual_location,
        residual_scale=residual_scale,
        reliability=reliability,
        attention_temperature=float(attention_temperature),
        topk_ratio=float(topk_ratio),
        attention_mode=attention_mode,
    )


def score_frequency_patch_graph_channels(
    data: np.ndarray,
    state: FrequencyPatchGraphState,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    values = _validate_data(data, state.patch_size)
    starts = patch_starts(len(values), state.patch_size, state.hop_size)
    tokens = extract_frequency_patch_tokens(
        values,
        starts,
        patch_size=state.patch_size,
        frequency_bins=state.frequency_bins,
    )
    channels = int(values.shape[1])
    token_dim = int(tokens.shape[2])
    if (
        state.neighbor_indices.shape[0] != channels
        or state.token_location.shape != (channels, token_dim)
        or state.token_scale.shape != (channels, token_dim)
        or state.coefficients.shape != (channels, token_dim * 2, token_dim)
        or state.intercepts.shape != (channels, token_dim)
        or state.residual_location.shape != (channels,)
        or state.residual_scale.shape != (channels,)
        or state.reliability.shape != (channels,)
        or np.any(state.token_scale <= 0)
        or np.any(state.residual_scale <= 0)
    ):
        raise ValueError("Frequency patch graph state is incompatible with data")
    normalized = (
        tokens - state.token_location[None, ...]
    ) / state.token_scale[None, ...]
    patch_channel_scores = np.empty((len(tokens), channels), dtype=np.float64)
    attention_weights = np.empty(
        (
            len(tokens),
            channels,
            state.neighbor_indices.shape[1],
        ),
        dtype=np.float64,
    )
    for channel in range(channels):
        previous_target = normalized[:-1, channel, :]
        current_neighbors = normalized[
            1:,
            state.neighbor_indices[channel],
            :,
        ]
        weights = neighbor_attention_weights(
            previous_target,
            current_neighbors,
            state.neighbor_priors[channel],
            temperature=state.attention_temperature,
            mode=state.attention_mode,
        )
        aggregate = np.einsum("nk,nkd->nd", weights, current_neighbors)
        features = np.concatenate([aggregate, previous_target], axis=1)
        prediction = (
            features @ state.coefficients[channel]
            + state.intercepts[channel]
        )
        residual = np.abs(normalized[1:, channel, :] - prediction).mean(axis=1)
        robust = np.maximum(
            (residual - state.residual_location[channel])
            / state.residual_scale[channel],
            0.0,
        )
        patch_channel_scores[1:, channel] = (
            robust * np.sqrt(state.reliability[channel])
        )
        attention_weights[1:, channel, :] = weights
    patch_channel_scores[0] = patch_channel_scores[1]
    attention_weights[0] = attention_weights[1]
    if (
        not np.isfinite(patch_channel_scores).all()
        or not np.isfinite(attention_weights).all()
    ):
        raise FloatingPointError("Frequency patch diagnostics contain NaN/Inf")
    return starts, patch_channel_scores, attention_weights


def score_frequency_patch_graph(
    data: np.ndarray,
    state: FrequencyPatchGraphState,
) -> np.ndarray:
    values = _validate_data(data, state.patch_size)
    starts, patch_channel_scores, _ = score_frequency_patch_graph_channels(
        values,
        state,
    )
    channels = int(values.shape[1])
    selected_count = max(1, int(np.ceil(channels * state.topk_ratio)))
    selected = np.partition(
        patch_channel_scores,
        channels - selected_count,
        axis=1,
    )[:, -selected_count:]
    patch_scores = selected.mean(axis=1)
    point_total = np.zeros(len(values), dtype=np.float64)
    point_count = np.zeros(len(values), dtype=np.int64)
    for start, score in zip(starts, patch_scores):
        point_total[start:start + state.patch_size] += score
        point_count[start:start + state.patch_size] += 1
    if np.any(point_count == 0):
        raise RuntimeError("Frequency patch graph left uncovered points")
    scores = point_total / point_count
    if not np.isfinite(scores).all():
        raise FloatingPointError("Frequency patch graph scores contain NaN/Inf")
    return scores
