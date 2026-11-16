"""Residual time-series encoder used by the maintained AdCL-FD model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def conv1d_same_padding(input_tensor, weight, bias, stride, dilation, groups):
    kernel, dilation_value, stride_value = weight.size(2), dilation[0], stride[0]
    input_length = input_tensor.size(2)
    output_length = (input_length + stride_value - 1) // stride_value
    padding = max(0, (output_length - 1) * stride_value - input_length + dilation_value * (kernel - 1) + 1)
    if padding % 2:
        input_tensor = F.pad(input_tensor, (0, 1))
    return F.conv1d(input_tensor, weight, bias, stride=stride, padding=padding // 2, dilation=dilation, groups=groups)


class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return conv1d_same_padding(input_tensor, self.weight, self.bias, self.stride, self.dilation, self.groups)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            Conv1dSamePadding(in_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(value < 1 for value in kernel_sizes):
            raise ValueError("Residual kernel sizes must be positive")
        blocks = []
        current_channels = in_channels
        for kernel_size in kernel_sizes:
            blocks.append(
                ConvBlock(current_channels, out_channels, kernel_size)
            )
            current_channels = out_channels
        self.layers = nn.Sequential(*blocks)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Sequential(
            Conv1dSamePadding(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.layers(x) + self.residual(x))


class ResNetRepresentation(nn.Module):
    """Channel-aware residual encoder with attention/statistical pooling."""

    VALID_REPRESENTATIONS = frozenset({"single", "multiscale", "difference"})
    VALID_POOLED_STATISTICS = frozenset(
        {"attention_mean", "maximum", "standard_deviation"}
    )

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        *,
        representation: str = "single",
        patch_sizes: tuple[int, ...] = (16, 32, 64),
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
        if in_channels < 1 or mid_channels < 2:
            raise ValueError("in_channels must be positive and mid_channels at least 2")
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {sorted(self.VALID_REPRESENTATIONS)}"
            )
        normalized_patch_sizes = tuple(sorted({int(value) for value in patch_sizes}))
        if representation == "multiscale" and (
            not normalized_patch_sizes
            or any(value < 2 for value in normalized_patch_sizes)
        ):
            raise ValueError("multiscale representation requires patch sizes >= 2")
        normalized_kernels = tuple(int(value) for value in residual_kernel_sizes)
        normalized_multipliers = tuple(
            int(value) for value in block_width_multipliers
        )
        normalized_statistics = tuple(str(value) for value in pooled_statistics)
        if per_channel_width < 1 or stem_kernel_size < 1:
            raise ValueError("Encoder width and stem kernel must be positive")
        if not normalized_kernels or any(value < 1 for value in normalized_kernels):
            raise ValueError("Residual kernel sizes must be positive")
        if (
            not normalized_multipliers
            or any(value < 1 for value in normalized_multipliers)
        ):
            raise ValueError("Block width multipliers must be positive")
        if (
            not normalized_statistics
            or len(set(normalized_statistics)) != len(normalized_statistics)
            or any(
                value not in self.VALID_POOLED_STATISTICS
                for value in normalized_statistics
            )
        ):
            raise ValueError(
                "Pooled statistics must be unique supported statistic names"
            )
        if multiscale_minimum_patch_count < 1:
            raise ValueError("Minimum multiscale patch count must be positive")
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.representation = str(representation)
        self.patch_sizes = normalized_patch_sizes
        self.per_channel_width = int(per_channel_width)
        self.stem_kernel_size = int(stem_kernel_size)
        self.residual_kernel_sizes = normalized_kernels
        self.block_width_multipliers = normalized_multipliers
        self.pooled_statistics = normalized_statistics
        self.multiscale_minimum_patch_count = int(
            multiscale_minimum_patch_count
        )
        self.per_channel = nn.Sequential(
            Conv1dSamePadding(
                self.in_channels,
                self.in_channels * self.per_channel_width,
                kernel_size=self.stem_kernel_size,
                stride=1,
                groups=self.in_channels,
            ),
            nn.BatchNorm1d(self.in_channels * self.per_channel_width),
            nn.GELU(),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv1d(self.in_channels * self.per_channel_width, self.mid_channels, kernel_size=1),
            nn.BatchNorm1d(self.mid_channels),
            nn.GELU(),
        )
        residual_blocks = []
        current_channels = self.mid_channels
        for multiplier in self.block_width_multipliers:
            out_channels = self.mid_channels * multiplier
            residual_blocks.append(
                ResNetBlock(
                    current_channels,
                    out_channels,
                    self.residual_kernel_sizes,
                )
            )
            current_channels = out_channels
        self.layers = nn.Sequential(*residual_blocks)
        out_channels = current_channels
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, self.mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.mid_channels, 1, kernel_size=1),
        )
        self.output_dim = out_channels * len(self.pooled_statistics)
        if self.representation == "multiscale":
            self.patch_fusion = nn.Sequential(
                nn.Linear(self.output_dim * len(self.patch_sizes), self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
            )
            self.patch_gate = nn.Sequential(
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.Sigmoid(),
            )
        elif self.representation == "difference":
            self.difference_fusion = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
            )
            self.difference_gate = nn.Sequential(
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.Sigmoid(),
            )

    def _encode_representation(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.layers(self.channel_mix(self.per_channel(x)))
        weights = torch.softmax(self.attention(encoded), dim=-1)
        values = {
            "attention_mean": (encoded * weights).sum(dim=-1),
            "maximum": encoded.amax(dim=-1),
            "standard_deviation": encoded.std(dim=-1, unbiased=False),
        }
        return torch.cat(
            [values[name] for name in self.pooled_statistics],
            dim=-1,
        )

    def channel_statistics(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [batch, {self.in_channels}, time], got {tuple(x.shape)}"
            )
        if x.shape[-1] < 2 or not torch.isfinite(x).all():
            raise ValueError("Channel statistics require finite temporal inputs")
        encoded = self.per_channel(x).reshape(
            len(x),
            self.in_channels,
            self.per_channel_width,
            x.shape[-1],
        )
        statistics = torch.cat(
            [
                encoded.mean(dim=-1),
                encoded.amax(dim=-1),
                encoded.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        if not torch.isfinite(statistics).all():
            raise FloatingPointError("Channel statistics contain NaN/Inf")
        return statistics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Encoder expects {self.in_channels} channels, got {x.shape[1]}")
        if x.shape[-1] < 2 or not torch.isfinite(x).all():
            raise ValueError("Encoder input must have at least two finite time points")
        representation = self._encode_representation(x)
        if self.representation == "multiscale":
            patch_representations = []
            for patch_size in self.patch_sizes:
                patch_count = max(
                    self.multiscale_minimum_patch_count,
                    int(math.ceil(x.shape[-1] / patch_size)),
                )
                patch_input = F.adaptive_avg_pool1d(x, patch_count)
                patch_representations.append(self._encode_representation(patch_input))
            patch_representation = self.patch_fusion(
                torch.cat(patch_representations, dim=1)
            )
            gate = self.patch_gate(
                torch.cat([representation, patch_representation], dim=1)
            )
            representation = representation + gate * patch_representation
        elif self.representation == "difference":
            difference = F.pad(x[..., 1:] - x[..., :-1], (1, 0))
            difference_representation = self.difference_fusion(
                self._encode_representation(difference)
            )
            gate = self.difference_gate(
                torch.cat([representation, difference_representation], dim=1)
            )
            representation = representation + gate * difference_representation
        if not torch.isfinite(representation).all():
            raise FloatingPointError("Encoder produced NaN/Inf values")
        return representation


def resnet_ts(**kwargs: int) -> dict[str, object]:
    backbone = ResNetRepresentation(**kwargs)
    return {"backbone": backbone, "dim": backbone.output_dim}
