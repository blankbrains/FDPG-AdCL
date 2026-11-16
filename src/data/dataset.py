"""CPU-safe sliding-window datasets for AdCL-FD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def window_starts(length: int, window_size: int, stride: int, cover_end: bool = False) -> list[int]:
    if length < 0:
        raise ValueError("length cannot be negative")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size because it would leave gaps")
    if length < window_size:
        return []
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if cover_end and starts[-1] != final_start:
        starts.append(final_start)
    return starts


class WindowDataset(Dataset):
    """Return CPU tensors in model-native ``[channels, time]`` order."""

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int,
        *,
        labels: np.ndarray | None = None,
        starts: Sequence[int] | None = None,
        cover_end: bool = False,
        normal_only: bool = False,
    ) -> None:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] < 1:
            raise ValueError(f"Expected [time, channels] data, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("WindowDataset data contains NaN/Inf values")
        self.data = array
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int32).ravel()
        if self.labels is not None:
            if len(self.labels) != len(self.data):
                raise ValueError("Point labels and data must have the same length")
            if not np.isin(self.labels, [0, 1]).all():
                raise ValueError("Point labels must be binary")
        candidates = list(starts) if starts is not None else window_starts(
            len(array), self.window_size, self.stride, cover_end=cover_end
        )
        max_start = len(array) - self.window_size
        if any(start < 0 or start > max_start for start in candidates):
            raise ValueError("A requested window start is out of range")
        if normal_only:
            if self.labels is None:
                raise ValueError("normal_only requires point labels")
            candidates = [
                start for start in candidates
                if self.labels[start:start + self.window_size].max(initial=0) == 0
            ]
        self.starts = candidates

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        start = self.starts[index]
        window = self.data[start:start + self.window_size]
        output: dict[str, torch.Tensor | int] = {
            "x": torch.from_numpy(window.T.copy()),
            "start": start,
        }
        if self.labels is not None:
            output["label"] = torch.tensor(
                int(self.labels[start:start + self.window_size].max(initial=0)),
                dtype=torch.long,
            )
        return output


class ContrastiveWindowDataset(Dataset):
    """Generate deterministic weak-positive pairs lazily on CPU."""

    VALID_AUGMENTATIONS = frozenset({"noise", "multi_view"})

    def __init__(
        self,
        base: WindowDataset,
        *,
        noise_sigma: float,
        seed: int,
        augmentation: str = "noise",
        amplitude_scale: float = 0.0,
        mask_ratio: float = 0.0,
        max_shift: int = 0,
    ) -> None:
        if len(base) == 0:
            raise ValueError("Contrastive training requires at least one window")
        if noise_sigma < 0 or not np.isfinite(noise_sigma):
            raise ValueError("noise_sigma must be finite and non-negative")
        if augmentation not in self.VALID_AUGMENTATIONS:
            raise ValueError(
                f"augmentation must be one of {sorted(self.VALID_AUGMENTATIONS)}"
            )
        if amplitude_scale < 0 or not np.isfinite(amplitude_scale):
            raise ValueError("amplitude_scale must be finite and non-negative")
        if not 0.0 <= mask_ratio < 1.0 or not np.isfinite(mask_ratio):
            raise ValueError("mask_ratio must be finite and in [0, 1)")
        if max_shift < 0 or max_shift >= base.window_size:
            raise ValueError("max_shift must be in [0, window_size)")
        self.base = base
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.augmentation = str(augmentation)
        self.amplitude_scale = float(amplitude_scale)
        self.mask_ratio = float(mask_ratio)
        self.max_shift = int(max_shift)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        item = self.base[index]
        anchor = item["x"]
        assert isinstance(anchor, torch.Tensor)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch * 1_000_003 + index)
        positive = anchor.clone()
        if self.augmentation == "multi_view":
            if self.amplitude_scale > 0:
                scale = 1.0 + (
                    torch.rand(
                        (positive.shape[0], 1),
                        generator=generator,
                        dtype=positive.dtype,
                    )
                    * 2.0
                    - 1.0
                ) * self.amplitude_scale
                positive = positive * scale
            if self.max_shift > 0:
                shift = int(
                    torch.randint(
                        -self.max_shift,
                        self.max_shift + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
                if shift > 0:
                    shifted = positive.clone()
                    shifted[:, shift:] = positive[:, :-shift]
                    shifted[:, :shift] = positive[:, :1]
                    positive = shifted
                elif shift < 0:
                    amount = -shift
                    shifted = positive.clone()
                    shifted[:, :-amount] = positive[:, amount:]
                    shifted[:, -amount:] = positive[:, -1:]
                    positive = shifted
            if self.mask_ratio > 0:
                mask_length = max(
                    1,
                    min(
                        positive.shape[-1] - 1,
                        int(round(positive.shape[-1] * self.mask_ratio)),
                    ),
                )
                mask_start = int(
                    torch.randint(
                        0,
                        positive.shape[-1] - mask_length + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
                fill = positive.mean(dim=-1, keepdim=True)
                positive = positive.clone()
                positive[:, mask_start:mask_start + mask_length] = fill
        positive = positive + torch.randn(
            positive.shape,
            generator=generator,
            dtype=positive.dtype,
        ) * self.noise_sigma
        if not torch.isfinite(positive).all():
            raise FloatingPointError("Augmentation produced NaN/Inf values")
        return {
            "anchor": anchor,
            "positive": positive,
            "start": item["start"],
        }
