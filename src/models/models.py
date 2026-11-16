"""Maintained AdCL-FD contrastive encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.resent_time import ResNetRepresentation


class ContrastiveModel(nn.Module):
    ARCHITECTURE = "adcl_fd_v3"

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        head: str = "mlp",
        features_dim: int = 64,
        *,
        representation: str = "single",
        patch_sizes: tuple[int, ...] = (16, 32, 64),
        channel_features_dim: int = 0,
        per_channel_width: int = 8,
        stem_kernel_size: int = 8,
        residual_kernel_sizes: tuple[int, ...] = (8, 5, 3),
        block_width_multipliers: tuple[int, ...] = (1, 2, 2),
        pooled_statistics: tuple[str, ...] = (
            "attention_mean",
            "maximum",
            "standard_deviation",
        ),
        multiscale_minimum_patch_count: int = 4,
    ) -> None:
        super().__init__()
        if features_dim < 2:
            raise ValueError("features_dim must be at least 2")
        if head not in {"linear", "mlp"}:
            raise ValueError("head must be 'linear' or 'mlp'")
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.head = head
        self.features_dim = int(features_dim)
        self.representation = str(representation)
        self.patch_sizes = tuple(int(value) for value in patch_sizes)
        self.channel_features_dim = int(channel_features_dim)
        self.per_channel_width = int(per_channel_width)
        self.stem_kernel_size = int(stem_kernel_size)
        self.residual_kernel_sizes = tuple(
            int(value) for value in residual_kernel_sizes
        )
        self.block_width_multipliers = tuple(
            int(value) for value in block_width_multipliers
        )
        self.pooled_statistics = tuple(str(value) for value in pooled_statistics)
        self.multiscale_minimum_patch_count = int(
            multiscale_minimum_patch_count
        )
        if self.channel_features_dim < 0 or self.channel_features_dim == 1:
            raise ValueError("channel_features_dim must be 0 or at least 2")
        self.backbone = ResNetRepresentation(
            self.in_channels,
            self.mid_channels,
            representation=self.representation,
            patch_sizes=self.patch_sizes,
            per_channel_width=self.per_channel_width,
            stem_kernel_size=self.stem_kernel_size,
            residual_kernel_sizes=self.residual_kernel_sizes,
            block_width_multipliers=self.block_width_multipliers,
            pooled_statistics=self.pooled_statistics,
            multiscale_minimum_patch_count=(
                self.multiscale_minimum_patch_count
            ),
        )
        self.contrastive_head = (
            nn.Linear(self.backbone.output_dim, self.features_dim)
            if head == "linear"
            else nn.Sequential(
                nn.Linear(self.backbone.output_dim, self.backbone.output_dim),
                nn.GELU(),
                nn.Linear(self.backbone.output_dim, self.features_dim),
            )
        )
        channel_input_dim = self.backbone.per_channel_width * 3
        self.channel_contrastive_head = (
            None
            if self.channel_features_dim == 0
            else nn.Sequential(
                nn.Linear(channel_input_dim, channel_input_dim),
                nn.GELU(),
                nn.Linear(channel_input_dim, self.channel_features_dim),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(self.contrastive_head(self.backbone(x)), dim=1)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Contrastive model produced NaN/Inf embeddings")
        return embedding

    def forward_channel_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_contrastive_head is None:
            raise RuntimeError("Channel embeddings are disabled for this model")
        statistics = self.backbone.channel_statistics(x)
        embeddings = F.normalize(
            self.channel_contrastive_head(statistics),
            dim=-1,
        )
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError(
                "Contrastive model produced NaN/Inf channel embeddings"
            )
        return embeddings

    def get_config(self) -> dict[str, int | str | list[int] | list[str]]:
        config: dict[str, int | str | list[int] | list[str]] = {
            "architecture": self.ARCHITECTURE,
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "head": self.head,
            "features_dim": self.features_dim,
            "per_channel_width": self.per_channel_width,
            "stem_kernel_size": self.stem_kernel_size,
            "residual_kernel_sizes": list(self.residual_kernel_sizes),
            "block_width_multipliers": list(
                self.block_width_multipliers
            ),
            "pooled_statistics": list(self.pooled_statistics),
            "multiscale_minimum_patch_count": (
                self.multiscale_minimum_patch_count
            ),
        }
        if self.representation != "single":
            config["representation"] = self.representation
            config["patch_sizes"] = list(self.patch_sizes)
        if self.channel_features_dim > 0:
            config["channel_features_dim"] = self.channel_features_dim
        return config
