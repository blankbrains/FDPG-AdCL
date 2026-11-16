"""Differentiable frequency-phase perturbation used by AdCL-FD v3 PGD."""

from __future__ import annotations

import math

import torch
from torch import nn


class FreqPhaseHardNeg(nn.Module):
    """Create hard negatives with bounded segment-wise phase perturbations."""

    VALID_MODES = frozenset({"uniform", "gaussian"})
    VALID_WINDOWS = frozenset({"hamming", "rectangular"})

    def __init__(
        self,
        epsilon: float = 0.3,
        perturbation_mode: str = "uniform",
        seg_len: int = 32,
        overlap_ratio: float = 0.5,
        overlap_add_window: str = "hamming",
    ) -> None:
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if perturbation_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported perturbation_mode: {perturbation_mode}")
        if seg_len < 2:
            raise ValueError("seg_len must be at least 2")
        if (
            not math.isfinite(overlap_ratio)
            or not 0.0 <= overlap_ratio < 1.0
        ):
            raise ValueError("overlap_ratio must be finite and in [0, 1)")
        if overlap_add_window not in self.VALID_WINDOWS:
            raise ValueError(
                f"Unsupported overlap-add window: {overlap_add_window}"
            )
        self.epsilon = float(epsilon)
        self.perturbation_mode = perturbation_mode
        self.seg_len = int(seg_len)
        self.overlap_ratio = float(overlap_ratio)
        self.overlap_add_window = str(overlap_add_window)

    def _sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if generator is not None and device.type != "cpu":
            raise ValueError("A CPU generator cannot be used for CUDA augmentation")
        if self.perturbation_mode == "uniform":
            return (torch.rand(shape, device=device, dtype=dtype, generator=generator) * 2 - 1) * self.epsilon
        delta = torch.randn(shape, device=device, dtype=dtype, generator=generator) * (self.epsilon / 3)
        return delta.clamp(-self.epsilon, self.epsilon)

    @staticmethod
    def _work_tensor(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.reshape(1, 1, x.shape[-1])
        if x.ndim == 2:
            return x.unsqueeze(0)
        return x

    def segment_starts(self, length: int) -> list[int]:
        if length < 2:
            raise ValueError("Time dimension must contain at least two points")
        segment = min(self.seg_len, length)
        stride = max(
            int(math.floor(segment * (1.0 - self.overlap_ratio))),
            1,
        )
        starts = list(range(0, length - segment + 1, stride))
        final_start = length - segment
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def delta_shape(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim not in (1, 2, 3):
            raise ValueError(f"Expected [..., time] input, got {tuple(x.shape)}")
        work = self._work_tensor(x)
        segment = min(self.seg_len, work.shape[-1])
        return (
            len(self.segment_starts(work.shape[-1])),
            work.shape[0],
            work.shape[1],
            segment // 2 + 1,
        )

    def zero_delta(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.delta_shape(x), device=x.device, dtype=x.dtype)

    def random_delta(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        delta = self._sample(
            self.delta_shape(x),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        return self.project_delta(
            delta,
            segment_length=min(self.seg_len, x.shape[-1]),
        )

    def project_delta(
        self,
        delta: torch.Tensor,
        *,
        segment_length: int,
        max_epsilon: float | None = None,
    ) -> torch.Tensor:
        if delta.ndim != 4 or not delta.is_floating_point():
            raise ValueError("Phase delta must be a floating [segments, batch, channels, frequencies] tensor")
        if not torch.isfinite(delta).all():
            raise ValueError("Phase delta contains NaN/Inf values")
        if segment_length < 2:
            raise ValueError("segment_length must be at least 2")
        epsilon = self.epsilon if max_epsilon is None else float(max_epsilon)
        if not math.isfinite(epsilon) or not 0.0 < epsilon <= self.epsilon:
            raise ValueError(
                "max_epsilon must be finite and in (0, epsilon]"
            )
        projected = delta.clamp(-epsilon, epsilon)
        mask = torch.ones_like(projected)
        mask.select(-1, 0).zero_()
        if segment_length % 2 == 0:
            mask.select(-1, -1).zero_()
        return projected * mask

    def apply_delta(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (1, 2, 3):
            raise ValueError(f"Expected [..., time] input, got {tuple(x.shape)}")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input must be a finite floating-point tensor")
        original_shape = x.shape
        work = self._work_tensor(x)
        expected_shape = self.delta_shape(x)
        if tuple(delta.shape) != expected_shape:
            raise ValueError(
                f"Phase delta shape {tuple(delta.shape)} does not match expected {expected_shape}"
            )
        if delta.device != work.device or delta.dtype != work.dtype:
            raise ValueError("Phase delta must use the same device and dtype as the input")

        batch, _, length = work.shape
        segment = min(self.seg_len, length)
        starts = self.segment_starts(length)
        output = torch.zeros_like(work)
        weights = torch.zeros((batch, 1, length), device=work.device, dtype=work.dtype)
        window = (
            torch.ones(segment, device=work.device, dtype=work.dtype)
            if self.overlap_add_window == "rectangular"
            else torch.hamming_window(
                segment,
                periodic=False,
                device=work.device,
                dtype=work.dtype,
            )
        )
        bounded = self.project_delta(delta, segment_length=segment)
        for segment_index, start in enumerate(starts):
            chunk = work[..., start:start + segment]
            spectrum = torch.fft.rfft(chunk, dim=-1)
            phase = spectrum.angle() + bounded[segment_index]
            changed = spectrum.abs() * torch.exp(1j * phase)
            restored = torch.fft.irfft(changed, n=segment, dim=-1)
            output[..., start:start + segment] += restored * window
            weights[..., start:start + segment] += window
        if torch.any(weights <= 0):
            raise RuntimeError("Frequency overlap-add left uncovered points")
        result = output / weights
        if not torch.isfinite(result).all():
            raise FloatingPointError("Frequency perturbation produced NaN/Inf values")
        return result.reshape(original_shape)

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.apply_delta(x, self.random_delta(x, generator=generator))

    def extra_repr(self) -> str:
        return (
            f"epsilon={self.epsilon}, mode={self.perturbation_mode!r}, "
            f"seg_len={self.seg_len}, overlap_ratio={self.overlap_ratio}, "
            f"window={self.overlap_add_window!r}"
        )
