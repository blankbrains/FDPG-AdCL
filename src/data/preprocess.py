"""Strict dataset-level preprocessing for AdCL-FD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class CausalImputer:
    """Forward-fill missing values using statistics fitted on training data only."""

    def __init__(self) -> None:
        self.fill_values: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.fill_values is not None

    @staticmethod
    def _validate_shape(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim != 2 or array.size == 0 or array.shape[1] < 1:
            raise ValueError(f"Expected non-empty [time, channels] data, got {array.shape}")
        if np.isinf(array).any():
            raise ValueError("CausalImputer input contains Inf values")
        return array

    def fit(self, data: np.ndarray) -> "CausalImputer":
        array = self._validate_shape(data)
        fill_values = np.empty((1, array.shape[1]), dtype=np.float32)
        for channel in range(array.shape[1]):
            finite_values = array[np.isfinite(array[:, channel]), channel]
            if finite_values.size == 0:
                raise ValueError(f"Channel {channel} has no finite training values")
            fill_values[0, channel] = np.median(finite_values).astype(np.float32)
        self.fill_values = fill_values
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        array = self._validate_shape(data)
        if not self.fitted:
            raise RuntimeError("CausalImputer must be fitted before transform")
        assert self.fill_values is not None
        if array.shape[1] != self.fill_values.shape[1]:
            raise ValueError(
                f"Imputer expects {self.fill_values.shape[1]} channels, got {array.shape[1]}"
            )

        output = array.copy()
        row_indices = np.arange(len(output), dtype=np.int64)
        for channel in range(output.shape[1]):
            finite = np.isfinite(output[:, channel])
            previous = np.where(finite, row_indices, -1)
            np.maximum.accumulate(previous, out=previous)
            observed = previous >= 0
            output[observed, channel] = output[previous[observed], channel]
            output[~observed, channel] = self.fill_values[0, channel]
        if not np.isfinite(output).all():
            raise FloatingPointError("Causal imputation produced NaN/Inf values")
        return output

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted CausalImputer")
        assert self.fill_values is not None
        return {"fill_values": self.fill_values.copy(), "fitted": True}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or not state.get("fitted", False):
            raise ValueError("Invalid or unfitted imputer_state")
        if "fill_values" not in state:
            raise ValueError("imputer_state must contain fill_values")
        fill_values = np.asarray(state["fill_values"], dtype=np.float32)
        if (
            fill_values.ndim != 2
            or fill_values.shape[0] != 1
            or fill_values.shape[1] < 1
            or not np.isfinite(fill_values).all()
        ):
            raise ValueError("imputer_state contains invalid fill values")
        self.fill_values = fill_values.copy()

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CausalImputer":
        imputer = cls()
        imputer.load_state_dict(state)
        return imputer


class StandardScaler:
    """Per-channel standardization fitted only on training data."""

    def __init__(self, eps: float = 1e-8) -> None:
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = float(eps)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    @staticmethod
    def _validate_data(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim not in (2, 3):
            raise ValueError(f"Expected [time, channels] or [samples, time, channels], got {array.shape}")
        if array.size == 0:
            raise ValueError("Cannot process an empty array")
        if not np.isfinite(array).all():
            count = int((~np.isfinite(array)).sum())
            raise ValueError(f"Data contains {count} NaN/Inf values")
        return array

    def fit(self, data: np.ndarray) -> "StandardScaler":
        array = self._validate_data(data)
        axes = (0,) if array.ndim == 2 else (0, 1)
        self.mean = array.mean(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
        self.std = array.std(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
        self.std = np.where(self.std < self.eps, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        array = self._validate_data(data)
        if not self.fitted:
            raise RuntimeError("StandardScaler must be fitted before transform")
        assert self.mean is not None and self.std is not None
        expected = int(self.mean.shape[-1])
        if array.shape[-1] != expected:
            raise ValueError(f"Scaler expects {expected} channels, got {array.shape[-1]}")
        transformed = (array - self.mean) / self.std
        if not np.isfinite(transformed).all():
            raise FloatingPointError("Standardization produced NaN/Inf values")
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted StandardScaler")
        return {"mean": self.mean, "std": self.std, "eps": self.eps, "fitted": True}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or not state.get("fitted", False):
            raise ValueError("Invalid or unfitted scaler_state")
        if "mean" not in state or "std" not in state:
            raise ValueError("scaler_state must contain mean and std")
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        if mean.ndim not in (2, 3) or std.shape != mean.shape or mean.shape[-1] < 1:
            raise ValueError("scaler_state mean/std have incompatible shapes")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("scaler_state contains invalid statistics")
        eps = float(state.get("eps", 1e-8))
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("scaler_state eps must be finite and positive")
        self.mean = mean
        self.std = std
        self.eps = eps

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "StandardScaler":
        scaler = cls()
        scaler.load_state_dict(state)
        return scaler
