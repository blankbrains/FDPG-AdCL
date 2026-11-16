"""Strict checkpoint reconstruction for AdCL-FD v3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.models.models import ContrastiveModel


REQUIRED_MODEL_CONFIG = frozenset({"architecture", "in_channels", "mid_channels", "head", "features_dim", "window_size"})


def load_checkpoint_file(path: str | Path, device: str | torch.device = "cpu"):
    """Load full checkpoints on both PyTorch 2.0 and newer releases."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(checkpoint: Mapping[str, Any], device: str | torch.device = "cpu") -> ContrastiveModel:
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint is missing a valid model_config")
    if config.get("architecture") != ContrastiveModel.ARCHITECTURE:
        raise ValueError("Unsupported checkpoint architecture; only adcl_fd_v3 is accepted")
    missing = sorted(REQUIRED_MODEL_CONFIG.difference(config))
    if missing:
        raise ValueError("Incomplete model_config; missing: " + ", ".join(missing))
    model = ContrastiveModel(
        in_channels=int(config["in_channels"]),
        mid_channels=int(config["mid_channels"]),
        head=str(config["head"]),
        features_dim=int(config["features_dim"]),
        representation=str(config.get("representation", "single")),
        patch_sizes=tuple(
            int(value) for value in config.get("patch_sizes", (16, 32, 64))
        ),
        channel_features_dim=int(config.get("channel_features_dim", 0)),
        per_channel_width=int(config.get("per_channel_width", 8)),
        stem_kernel_size=int(config.get("stem_kernel_size", 8)),
        residual_kernel_sizes=tuple(
            int(value)
            for value in config.get("residual_kernel_sizes", (8, 5, 3))
        ),
        block_width_multipliers=tuple(
            int(value)
            for value in config.get("block_width_multipliers", (1, 2, 2))
        ),
        pooled_statistics=tuple(
            str(value)
            for value in config.get(
                "pooled_statistics",
                ("attention_mean", "maximum", "standard_deviation"),
            )
        ),
        multiscale_minimum_patch_count=int(
            config.get("multiscale_minimum_patch_count", 4)
        ),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint parameters do not match model_config") from exc
    model.to(device)
    model.eval()
    return model


def load_model_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = load_checkpoint_file(path, device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    return build_model_from_checkpoint(checkpoint, device=device), checkpoint
