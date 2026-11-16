"""Contrastive objective for the maintained AdCL-FD pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def overlap_negative_mask(starts: torch.Tensor, window_size: int) -> torch.Tensor:
    """Return pairs whose source windows do not overlap in time."""
    values = torch.as_tensor(starts, dtype=torch.long)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("starts must be a non-empty one-dimensional tensor")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    return (values[:, None] - values[None, :]).abs() >= int(window_size)


def overlap_fraction(starts: torch.Tensor, window_size: int) -> torch.Tensor:
    values = torch.as_tensor(starts, dtype=torch.long)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("starts must be a non-empty one-dimensional tensor")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    distance = (values[:, None] - values[None, :]).abs().float()
    return ((float(window_size) - distance).clamp_min(0.0) / float(window_size))


class AdCLFDV3Loss(nn.Module):
    """Overlap-aware contrastive loss with explicit adversarial negatives."""

    def __init__(
        self,
        temperature: float = 0.2,
        variance_weight: float = 0.1,
        variance_target: float = 0.05,
        overlap_policy: str = "mask",
        overlap_penalty: float = 2.0,
    ) -> None:
        super().__init__()
        values = torch.tensor(
            [temperature, variance_weight, variance_target, overlap_penalty],
            dtype=torch.float64,
        )
        if not torch.isfinite(values).all():
            raise ValueError("Loss hyperparameters must be finite")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if variance_weight < 0 or variance_target < 0:
            raise ValueError("variance_weight and variance_target must be non-negative")
        if overlap_policy not in {"mask", "soft"}:
            raise ValueError("overlap_policy must be mask or soft")
        if overlap_penalty < 0:
            raise ValueError("overlap_penalty must be non-negative")
        self.temperature = float(temperature)
        self.variance_weight = float(variance_weight)
        self.variance_target = float(variance_target)
        self.overlap_policy = str(overlap_policy)
        self.overlap_penalty = float(overlap_penalty)

    @staticmethod
    def _validate_views(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> None:
        if anchor.ndim != 2 or anchor.shape[0] < 1 or anchor.shape[1] < 2:
            raise ValueError(f"Expected [batch, embedding_dim] anchor, got {tuple(anchor.shape)}")
        if positive.shape != anchor.shape or negative.shape != anchor.shape:
            raise ValueError("Anchor, positive, and negative embeddings must have equal shapes")
        if not (
            torch.isfinite(anchor).all()
            and torch.isfinite(positive).all()
            and torch.isfinite(negative).all()
        ):
            raise FloatingPointError("Contrastive embeddings contain NaN/Inf")

    def _directional_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        paired_positive = (query * positive).sum(dim=1, keepdim=True)
        paired_negative = (query * negative).sum(dim=1, keepdim=True)
        batch_similarity = query @ positive.T
        if self.overlap_policy == "mask":
            non_overlap = overlap_negative_mask(starts, window_size).to(query.device)
            batch_negatives = batch_similarity.masked_fill(~non_overlap, -torch.inf)
        else:
            fractions = overlap_fraction(starts, window_size).to(query.device)
            batch_negatives = batch_similarity - self.overlap_penalty * fractions
            batch_negatives.fill_diagonal_(float("-inf"))
        logits = torch.cat(
            [paired_positive, paired_negative, batch_negatives],
            dim=1,
        ) / self.temperature
        targets = torch.zeros(len(query), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, targets)

    def contrastive_term(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        self._validate_views(anchor, positive, negative)
        starts_tensor = torch.as_tensor(starts, dtype=torch.long, device=anchor.device)
        if starts_tensor.shape != (len(anchor),):
            raise ValueError("starts length must match the embedding batch size")
        anchor_normalized = F.normalize(anchor, dim=1)
        positive_normalized = F.normalize(positive, dim=1)
        negative_normalized = F.normalize(negative, dim=1)
        forward = self._directional_loss(
            anchor_normalized,
            positive_normalized,
            negative_normalized,
            starts_tensor,
            window_size,
        )
        backward = self._directional_loss(
            positive_normalized,
            anchor_normalized,
            negative_normalized,
            starts_tensor,
            window_size,
        )
        return 0.5 * (forward + backward)

    def attack_objective(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        return self.contrastive_term(anchor, positive, negative, starts, window_size)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        contrastive = self.contrastive_term(
            anchor,
            positive,
            negative,
            starts,
            window_size,
        )
        normalized_pairs = F.normalize(torch.cat([anchor, positive], dim=0), dim=1)
        standard_deviation = torch.sqrt(
            normalized_pairs.var(dim=0, unbiased=False) + 1e-4
        )
        variance = F.relu(self.variance_target - standard_deviation).mean()
        loss = contrastive + self.variance_weight * variance
        if not torch.isfinite(loss):
            raise FloatingPointError("AdCLFDV3Loss produced NaN/Inf")
        return loss
