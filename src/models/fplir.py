"""Point-local repair model for FP-LIR-AdCL."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def deterministic_linear_interpolate_1d(
    x: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    """Match 1D linear interpolation without CUDA atomic-add backward."""
    if x.ndim < 1 or x.shape[-1] < 1 or size < 1:
        raise ValueError("Invalid deterministic interpolation dimensions")
    input_size = int(x.shape[-1])
    if input_size == size:
        return x

    positions = (
        (torch.arange(size, device=x.device, dtype=x.dtype) + 0.5)
        * (input_size / float(size))
        - 0.5
    ).clamp(0.0, float(input_size - 1))
    lower = positions.floor().to(dtype=torch.long)
    upper = (lower + 1).clamp(max=input_size - 1)
    upper_weight = positions - lower.to(dtype=x.dtype)
    lower_weight = 1.0 - upper_weight
    source = torch.arange(input_size, device=x.device).unsqueeze(0)
    interpolation = (
        (source == lower.unsqueeze(1)).to(dtype=x.dtype)
        * lower_weight.unsqueeze(1)
        + (source == upper.unsqueeze(1)).to(dtype=x.dtype)
        * upper_weight.unsqueeze(1)
    )
    return torch.matmul(x, interpolation.transpose(0, 1))


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if (
            channels < 4
            or dilation < 1
            or kernel_size < 1
            or kernel_size % 2 == 0
        ):
            raise ValueError("Invalid temporal block dimensions")
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class FPLIRModel(nn.Module):
    """Multi-scale temporal/frequency encoder with point-wise repair outputs."""

    ARCHITECTURE = "fplir_v1"

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int = 64,
        token_dim: int = 32,
        spectral_patch_size: int = 32,
        spectral_hop: int = 8,
        per_channel_width: int = 4,
        stem_kernel_size: int = 7,
        temporal_kernel_size: int = 5,
        temporal_dilations: tuple[int, ...] = (1, 2, 4),
        repair_kernel_size: int = 3,
        spectral_channel_statistics: tuple[str, ...] = ("mean", "maximum"),
    ) -> None:
        super().__init__()
        if in_channels < 1 or hidden_dim < 8 or token_dim < 4:
            raise ValueError("Invalid FP-LIR model dimensions")
        if spectral_patch_size < 4 or spectral_hop < 1:
            raise ValueError("Invalid spectral patch configuration")
        if spectral_hop > spectral_patch_size:
            raise ValueError("spectral_hop cannot exceed spectral_patch_size")
        normalized_dilations = tuple(
            int(value) for value in temporal_dilations
        )
        normalized_spectral_statistics = tuple(
            str(value) for value in spectral_channel_statistics
        )
        odd_kernels = (
            int(stem_kernel_size),
            int(temporal_kernel_size),
            int(repair_kernel_size),
        )
        if per_channel_width < 1:
            raise ValueError("per_channel_width must be positive")
        if any(value < 1 or value % 2 == 0 for value in odd_kernels):
            raise ValueError("FP-LIR temporal kernels must be positive and odd")
        if (
            not normalized_dilations
            or any(value < 1 for value in normalized_dilations)
        ):
            raise ValueError("FP-LIR temporal dilations must be positive")
        if (
            not normalized_spectral_statistics
            or len(set(normalized_spectral_statistics))
            != len(normalized_spectral_statistics)
            or any(
                value not in {"mean", "maximum"}
                for value in normalized_spectral_statistics
            )
        ):
            raise ValueError(
                "FP-LIR spectral statistics must be unique supported names"
            )
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.spectral_patch_size = int(spectral_patch_size)
        self.spectral_hop = int(spectral_hop)
        self.per_channel_width = int(per_channel_width)
        self.stem_kernel_size = int(stem_kernel_size)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.temporal_dilations = normalized_dilations
        self.repair_kernel_size = int(repair_kernel_size)
        self.spectral_channel_statistics = normalized_spectral_statistics

        stem_padding = (self.stem_kernel_size - 1) // 2
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.in_channels * self.per_channel_width,
                kernel_size=self.stem_kernel_size,
                padding=stem_padding,
                groups=self.in_channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.in_channels * self.per_channel_width,
                self.hidden_dim,
                kernel_size=1,
            ),
            nn.GroupNorm(1, self.hidden_dim),
            nn.GELU(),
        )
        frequency_bins = self.spectral_patch_size // 2 + 1
        self.spectral_projection = nn.Sequential(
            nn.Linear(
                frequency_bins * len(self.spectral_channel_statistics),
                self.hidden_dim,
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            *[
                ResidualTemporalBlock(
                    self.hidden_dim,
                    dilation,
                    self.temporal_kernel_size,
                )
                for dilation in self.temporal_dilations
            ]
        )
        repair_padding = (self.repair_kernel_size - 1) // 2
        self.repair_head = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=self.repair_kernel_size,
                padding=repair_padding,
            ),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.in_channels, kernel_size=1),
        )
        self.token_head = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.token_dim, kernel_size=1),
        )

    def _spectral_context(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = min(self.spectral_patch_size, int(x.shape[-1]))
        if patch_size < 4:
            raise ValueError("Input is too short for spectral patches")
        hop = min(self.spectral_hop, patch_size)
        patches = x.unfold(-1, patch_size, hop)
        if patch_size != self.spectral_patch_size:
            patches = F.pad(
                patches,
                (0, self.spectral_patch_size - patch_size),
            )
        spectrum = torch.fft.rfft(patches, n=self.spectral_patch_size, dim=-1)
        magnitude = torch.log1p(spectrum.abs())
        statistics = {
            "mean": magnitude.mean(dim=1),
            "maximum": magnitude.amax(dim=1),
        }
        spectral = self.spectral_projection(
            torch.cat(
                [
                    statistics[name]
                    for name in self.spectral_channel_statistics
                ],
                dim=-1,
            )
        )
        spectral = spectral.transpose(1, 2)
        return deterministic_linear_interpolate_1d(
            spectral,
            size=x.shape[-1],
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.ndim != 3
            or x.shape[1] != self.in_channels
            or x.shape[-1] < 4
        ):
            raise ValueError(
                f"Expected [batch, {self.in_channels}, time>=4], got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("FP-LIR input contains NaN/Inf")
        encoded = self.encoder(self.temporal_stem(x) + self._spectral_context(x))
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("FP-LIR encoder produced NaN/Inf")
        return encoded

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        repair = self.repair_head(encoded)
        tokens = F.normalize(self.token_head(encoded), dim=1)
        if not torch.isfinite(repair).all() or not torch.isfinite(tokens).all():
            raise FloatingPointError("FP-LIR model produced NaN/Inf")
        return repair, tokens

    def get_config(self) -> dict[str, int | str | list[int] | list[str]]:
        return {
            "architecture": self.ARCHITECTURE,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "token_dim": self.token_dim,
            "spectral_patch_size": self.spectral_patch_size,
            "spectral_hop": self.spectral_hop,
            "per_channel_width": self.per_channel_width,
            "stem_kernel_size": self.stem_kernel_size,
            "temporal_kernel_size": self.temporal_kernel_size,
            "temporal_dilations": list(self.temporal_dilations),
            "repair_kernel_size": self.repair_kernel_size,
            "spectral_channel_statistics": list(
                self.spectral_channel_statistics
            ),
        }


def build_fplir_from_config(config: dict[str, object]) -> FPLIRModel:
    required = {
        "architecture",
        "in_channels",
        "hidden_dim",
        "token_dim",
        "spectral_patch_size",
        "spectral_hop",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError("Incomplete FP-LIR model config: " + ", ".join(missing))
    if config.get("architecture") != FPLIRModel.ARCHITECTURE:
        raise ValueError("Unsupported FP-LIR architecture")
    values = [
        int(config["in_channels"]),
        int(config["hidden_dim"]),
        int(config["token_dim"]),
        int(config["spectral_patch_size"]),
        int(config["spectral_hop"]),
        int(config.get("per_channel_width", 4)),
        int(config.get("stem_kernel_size", 7)),
        int(config.get("temporal_kernel_size", 5)),
        int(config.get("repair_kernel_size", 3)),
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("FP-LIR model configuration is not finite")
    return FPLIRModel(
        in_channels=values[0],
        hidden_dim=values[1],
        token_dim=values[2],
        spectral_patch_size=values[3],
        spectral_hop=values[4],
        per_channel_width=values[5],
        stem_kernel_size=values[6],
        temporal_kernel_size=values[7],
        temporal_dilations=tuple(
            int(value)
            for value in config.get("temporal_dilations", (1, 2, 4))
        ),
        repair_kernel_size=values[8],
        spectral_channel_statistics=tuple(
            str(value)
            for value in config.get(
                "spectral_channel_statistics",
                ("mean", "maximum"),
            )
        ),
    )
