"""Train the maintained AdCL-FD v3 contrastive encoder.

The only supported training path is:
unlabeled temporal windows -> weak positives + phase-PGD negatives ->
overlap-aware contrastive training -> validation-point threshold calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
)
from torch.utils.data import DataLoader, Subset

from data.dataset import ContrastiveWindowDataset, WindowDataset
from data.freq_hard_neg import FreqPhaseHardNeg
from data.loading import load_feature_pair
from data.preprocess import CausalImputer, StandardScaler
from data.splitting import UnifiedTemporalSplit, split_full_series
from losses import AdCLFDV3Loss
from models import ContrastiveModel, load_checkpoint_file
from scoring import (
    calibrate_point_threshold,
    fit_score_statistics,
    score_embeddings,
)


REQUIRED_CONFIG_KEYS = frozenset(
    {
        "dataset", "data_root", "in_channels", "window_size", "stride",
        "eval_stride", "mid_channels", "head", "features_dim", "batch_size",
        "num_epochs", "learning_rate", "weight_decay", "temperature",
        "noise_sigma", "phase_epsilon", "segment_length",
        "train_fraction", "post_train_blocks",
        "pgd_steps", "pgd_step_size", "variance_weight", "variance_target",
        "min_embedding_std", "min_effective_rank", "scoring", "k",
        "threshold_percentile", "patience", "seed", "reference_limit",
    }
)


def load_config(
    path: str | Path,
    *,
    variant: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    selected_variant = str(
        variant if variant is not None else config.get("experiment_variant", "RGCSF")
    )
    variant_path = Path(__file__).resolve().parents[1] / "configs" / "branches.yaml"
    if selected_variant:
        if not variant_path.is_file():
            raise FileNotFoundError(f"Experiment variant file not found: {variant_path}")
        with variant_path.open("r", encoding="utf-8") as handle:
            variant_config = yaml.safe_load(handle)
        if not isinstance(variant_config, dict) or not isinstance(
            variant_config.get("variants"), dict
        ):
            raise ValueError("Experiment variant file must contain a variants mapping")
        variants = variant_config["variants"]
        if selected_variant not in variants:
            raise ValueError(
                f"Unknown experiment variant {selected_variant!r}; "
                f"available={sorted(variants)}"
            )
        variant_overrides = variants[selected_variant]
        if not isinstance(variant_overrides, dict):
            raise ValueError(f"Variant {selected_variant!r} must be a mapping")
        config.update(variant_overrides)
    config["experiment_variant"] = selected_variant
    config.setdefault("representation", "single")
    config.setdefault("patch_sizes", [16, 32, 64])
    config.setdefault("global_per_channel_width", 8)
    config.setdefault("global_stem_kernel_size", 8)
    config.setdefault("global_residual_kernel_sizes", [8, 5, 3])
    config.setdefault("global_block_width_multipliers", [1, 2, 2])
    config.setdefault(
        "global_pooled_statistics",
        ["attention_mean", "maximum", "standard_deviation"],
    )
    config.setdefault("global_multiscale_minimum_patch_count", 4)
    config.setdefault("augmentation", "noise")
    config.setdefault("amplitude_scale", 0.0)
    config.setdefault("mask_ratio", 0.0)
    config.setdefault("max_shift", 0)
    config.setdefault("hybrid_mix_weight", 0.5)
    config.setdefault("overlap_policy", "mask")
    config.setdefault("overlap_penalty", 2.0)
    config.setdefault("channel_features_dim", 0)
    config.setdefault("channel_loss_weight", 0.0)
    config.setdefault("channel_sparse_ratio", 1.0)
    config.setdefault("fplir_per_channel_width", 4)
    config.setdefault("fplir_stem_kernel_size", 7)
    config.setdefault("fplir_temporal_kernel_size", 5)
    config.setdefault("fplir_temporal_dilations", [1, 2, 4])
    config.setdefault("fplir_repair_kernel_size", 3)
    config.setdefault(
        "fplir_spectral_channel_statistics",
        ["mean", "maximum"],
    )
    config.setdefault("max_train_windows", 0)
    config.setdefault("max_validation_windows", 0)
    config.setdefault("min_delta", 1e-6)
    config.setdefault("warmup_epochs", 0)
    config.setdefault("warmup_start_factor", 0.1)
    config.setdefault("phase_perturbation_mode", "uniform")
    config.setdefault("phase_segment_overlap_ratio", 0.5)
    config.setdefault("phase_overlap_add_window", "hamming")
    config.setdefault("adversarial_ramp_epochs", 0)
    config.setdefault("ema_decay", 0.0)
    config.setdefault("global_optimizer", "adam")
    config.setdefault("fplir_optimizer", "adamw")
    config.setdefault("global_scheduler", "cosine")
    config.setdefault("fplir_scheduler", "cosine")
    if overrides:
        unknown = sorted(set(overrides).difference(config))
        if unknown:
            raise ValueError(
                "Unknown config override keys: " + ", ".join(unknown)
            )
        config.update(dict(overrides))
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError("Config is missing required keys: " + ", ".join(missing))
    train_fraction = float(config["train_fraction"])
    if not np.isfinite(train_fraction) or train_fraction != 0.5:
        raise ValueError("train_fraction must be exactly 0.5")
    if int(config["post_train_blocks"]) != 2:
        raise ValueError("post_train_blocks must be exactly 2")
    positive_ints = (
        "in_channels", "window_size", "stride", "eval_stride", "mid_channels",
        "features_dim", "batch_size", "num_epochs", "pgd_steps", "k",
        "patience", "reference_limit", "global_per_channel_width",
        "global_stem_kernel_size",
        "global_multiscale_minimum_patch_count",
    )
    if any(int(config[key]) < 1 for key in positive_ints):
        raise ValueError("Channel, window, model, batch, epoch, patience, and reference sizes must be positive")
    if (
        int(config["window_size"]) < 2
        or int(config["stride"]) > int(config["window_size"])
        or int(config["eval_stride"]) > int(config["window_size"])
    ):
        raise ValueError("window_size must be at least 2 and strides cannot exceed it")
    if int(config["segment_length"]) < 2:
        raise ValueError("segment_length must be at least 2")
    if str(config["head"]) not in {"linear", "mlp"}:
        raise ValueError("head must be linear or mlp")
    if str(config["representation"]) not in {"single", "multiscale", "difference"}:
        raise ValueError("representation must be single, multiscale, or difference")
    patch_sizes = config["patch_sizes"]
    if not isinstance(patch_sizes, (list, tuple)):
        raise ValueError("patch_sizes must be a list or tuple")
    normalized_patch_sizes = sorted({int(value) for value in patch_sizes})
    if str(config["representation"]) == "multiscale" and (
        not normalized_patch_sizes
        or any(value < 2 for value in normalized_patch_sizes)
    ):
        raise ValueError("multiscale patch_sizes must contain integers >= 2")
    config["patch_sizes"] = normalized_patch_sizes
    residual_kernel_sizes = config["global_residual_kernel_sizes"]
    block_width_multipliers = config["global_block_width_multipliers"]
    pooled_statistics = config["global_pooled_statistics"]
    if not isinstance(residual_kernel_sizes, (list, tuple)):
        raise ValueError("global_residual_kernel_sizes must be a sequence")
    if not isinstance(block_width_multipliers, (list, tuple)):
        raise ValueError("global_block_width_multipliers must be a sequence")
    if not isinstance(pooled_statistics, (list, tuple)):
        raise ValueError("global_pooled_statistics must be a sequence")
    normalized_residual_kernels = [
        int(value) for value in residual_kernel_sizes
    ]
    normalized_block_multipliers = [
        int(value) for value in block_width_multipliers
    ]
    normalized_pooled_statistics = [str(value) for value in pooled_statistics]
    if (
        not normalized_residual_kernels
        or any(value < 1 for value in normalized_residual_kernels)
    ):
        raise ValueError("Global residual kernel sizes must be positive")
    if (
        not normalized_block_multipliers
        or any(value < 1 for value in normalized_block_multipliers)
    ):
        raise ValueError("Global block width multipliers must be positive")
    valid_pooled_statistics = {
        "attention_mean",
        "maximum",
        "standard_deviation",
    }
    if (
        not normalized_pooled_statistics
        or len(set(normalized_pooled_statistics))
        != len(normalized_pooled_statistics)
        or any(
            value not in valid_pooled_statistics
            for value in normalized_pooled_statistics
        )
    ):
        raise ValueError(
            "Global pooled statistics must be unique supported names"
        )
    config["global_residual_kernel_sizes"] = normalized_residual_kernels
    config["global_block_width_multipliers"] = normalized_block_multipliers
    config["global_pooled_statistics"] = normalized_pooled_statistics
    if str(config["augmentation"]) not in {"noise", "multi_view"}:
        raise ValueError("augmentation must be noise or multi_view")
    amplitude_scale = float(config["amplitude_scale"])
    mask_ratio = float(config["mask_ratio"])
    max_shift = int(config["max_shift"])
    if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
        raise ValueError("amplitude_scale must be finite and non-negative")
    if not np.isfinite(mask_ratio) or not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be finite and in [0, 1)")
    if max_shift < 0 or max_shift >= int(config["window_size"]):
        raise ValueError("max_shift must be in [0, window_size)")
    hybrid_mix_weight = float(config["hybrid_mix_weight"])
    if not np.isfinite(hybrid_mix_weight) or not 0.0 <= hybrid_mix_weight <= 1.0:
        raise ValueError("hybrid_mix_weight must be finite and in [0, 1]")
    if str(config["overlap_policy"]) not in {"mask", "soft"}:
        raise ValueError("overlap_policy must be mask or soft")
    overlap_penalty = float(config["overlap_penalty"])
    if not np.isfinite(overlap_penalty) or overlap_penalty < 0:
        raise ValueError("overlap_penalty must be finite and non-negative")
    positive_floats = (
        "learning_rate", "temperature", "phase_epsilon", "pgd_step_size",
        "min_embedding_std", "min_effective_rank",
    )
    if any(not np.isfinite(float(config[key])) or float(config[key]) <= 0 for key in positive_floats):
        raise ValueError("Positive floating-point hyperparameters must be finite and positive")
    nonnegative_floats = (
        "weight_decay", "noise_sigma", "variance_weight", "variance_target",
        "min_delta",
    )
    if any(not np.isfinite(float(config[key])) or float(config[key]) < 0 for key in nonnegative_floats):
        raise ValueError("Non-negative hyperparameters must be finite and non-negative")
    channel_features_dim = int(config["channel_features_dim"])
    channel_loss_weight = float(config["channel_loss_weight"])
    channel_sparse_ratio = float(config["channel_sparse_ratio"])
    if channel_features_dim < 0 or channel_features_dim == 1:
        raise ValueError("channel_features_dim must be 0 or at least 2")
    if (
        not np.isfinite(channel_loss_weight)
        or channel_loss_weight < 0
        or (channel_loss_weight > 0 and channel_features_dim == 0)
    ):
        raise ValueError(
            "channel_loss_weight must be non-negative and requires channel features"
        )
    if (
        not np.isfinite(channel_sparse_ratio)
        or not 0.0 < channel_sparse_ratio <= 1.0
    ):
        raise ValueError("channel_sparse_ratio must be in (0, 1]")
    if (
        int(config["max_train_windows"]) < 0
        or int(config["max_validation_windows"]) < 0
    ):
        raise ValueError("Window limits must be non-negative")
    grad_clip = float(config.get("grad_clip", 1.0))
    if not np.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    workers = int(config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    minimum_lr = float(config.get("min_learning_rate", 1e-6))
    if not np.isfinite(minimum_lr) or minimum_lr < 0 or minimum_lr > float(config["learning_rate"]):
        raise ValueError("min_learning_rate must be finite and in [0, learning_rate]")
    warmup_epochs = int(config["warmup_epochs"])
    warmup_start_factor = float(config["warmup_start_factor"])
    if warmup_epochs < 0 or warmup_epochs >= int(config["num_epochs"]):
        raise ValueError("warmup_epochs must be in [0, num_epochs)")
    if (
        not np.isfinite(warmup_start_factor)
        or not 0.0 < warmup_start_factor <= 1.0
    ):
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if str(config["phase_perturbation_mode"]) not in {
        "uniform",
        "gaussian",
    }:
        raise ValueError(
            "phase_perturbation_mode must be uniform or gaussian"
        )
    phase_overlap_ratio = float(config["phase_segment_overlap_ratio"])
    if (
        not np.isfinite(phase_overlap_ratio)
        or not 0.0 <= phase_overlap_ratio < 1.0
    ):
        raise ValueError(
            "phase_segment_overlap_ratio must be finite and in [0, 1)"
        )
    if str(config["phase_overlap_add_window"]) not in {
        "hamming",
        "rectangular",
    }:
        raise ValueError(
            "phase_overlap_add_window must be hamming or rectangular"
        )
    adversarial_ramp_epochs = int(config["adversarial_ramp_epochs"])
    if not 0 <= adversarial_ramp_epochs <= int(config["num_epochs"]):
        raise ValueError(
            "adversarial_ramp_epochs must be in [0, num_epochs]"
        )
    ema_decay = float(config["ema_decay"])
    if not np.isfinite(ema_decay) or not (
        ema_decay == 0.0 or 0.9 <= ema_decay < 1.0
    ):
        raise ValueError("ema_decay must be 0 or in [0.9, 1)")
    for key in ("global_optimizer", "fplir_optimizer"):
        optimizer_name = str(config[key]).lower()
        if optimizer_name not in {"adam", "adamw", "radam"}:
            raise ValueError(
                f"{key} must be adam, adamw, or radam"
            )
        config[key] = optimizer_name
    for key in ("global_scheduler", "fplir_scheduler"):
        scheduler_name = str(config[key]).lower()
        if scheduler_name not in {"cosine", "linear", "constant"}:
            raise ValueError(
                f"{key} must be cosine, linear, or constant"
            )
        config[key] = scheduler_name
    scoring = str(config["scoring"]).lower()
    if scoring not in {"knn", "cosine", "hybrid"}:
        raise ValueError("scoring must be knn, cosine, or hybrid")
    config["scoring"] = scoring
    if scoring == "knn" and int(config["k"]) >= int(config["reference_limit"]):
        raise ValueError("KNN requires k < reference_limit")
    percentile = float(config["threshold_percentile"])
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("threshold_percentile must be finite and in (0, 100)")
    return config


def parse_config_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw_value = item.partition("=")
        key = key.strip()
        if not separator or not key or not raw_value.strip():
            raise ValueError(
                "Config overrides must use key=yaml_value syntax"
            )
        if key in overrides:
            raise ValueError(f"Duplicate config override: {key}")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def normalize_legacy_training_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("global_per_channel_width", 8)
    normalized.setdefault("global_stem_kernel_size", 8)
    normalized.setdefault("global_residual_kernel_sizes", [8, 5, 3])
    normalized.setdefault("global_block_width_multipliers", [1, 2, 2])
    normalized.setdefault(
        "global_pooled_statistics",
        ["attention_mean", "maximum", "standard_deviation"],
    )
    normalized.setdefault("global_multiscale_minimum_patch_count", 4)
    normalized.setdefault("warmup_epochs", 0)
    normalized.setdefault("warmup_start_factor", 0.1)
    normalized.setdefault("phase_perturbation_mode", "uniform")
    normalized.setdefault("phase_segment_overlap_ratio", 0.5)
    normalized.setdefault("phase_overlap_add_window", "hamming")
    normalized.setdefault("adversarial_ramp_epochs", 0)
    normalized.setdefault("ema_decay", 0.0)
    normalized.setdefault("global_optimizer", "adam")
    normalized.setdefault("fplir_optimizer", "adamw")
    normalized.setdefault("global_scheduler", "cosine")
    normalized.setdefault("fplir_scheduler", "cosine")
    return normalized


def normalize_legacy_contrastive_model_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("per_channel_width", 8)
    normalized.setdefault("stem_kernel_size", 8)
    normalized.setdefault("residual_kernel_sizes", [8, 5, 3])
    normalized.setdefault("block_width_multipliers", [1, 2, 2])
    normalized.setdefault(
        "pooled_statistics",
        ["attention_mean", "maximum", "standard_deviation"],
    )
    normalized.setdefault("multiscale_minimum_patch_count", 4)
    return normalized


def adversarial_ramp_scale(epoch: int, ramp_epochs: int) -> float:
    if epoch < 0 or ramp_epochs < 0:
        raise ValueError("epoch and ramp_epochs must be non-negative")
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, (epoch + 1) / float(ramp_epochs))


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    if not np.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be finite and in [0, 1)")
    model_parameters = dict(model.named_parameters())
    ema_parameters = dict(ema_model.named_parameters())
    model_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    if (
        model_parameters.keys() != ema_parameters.keys()
        or model_buffers.keys() != ema_buffers.keys()
    ):
        raise ValueError("EMA model structure does not match source model")
    for name, ema_parameter in ema_parameters.items():
        source = model_parameters[name].detach()
        ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)
    for name, ema_buffer in ema_buffers.items():
        source = model_buffers[name].detach()
        if ema_buffer.is_floating_point():
            ema_buffer.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            ema_buffer.copy_(source)


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str = "cosine",
    num_epochs: int,
    minimum_lr: float,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> Any:
    normalized = str(name).lower()
    if normalized not in {"cosine", "linear", "constant"}:
        raise ValueError(
            "Scheduler must be cosine, linear, or constant"
        )
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs >= num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs)")
    if (
        not np.isfinite(minimum_lr)
        or minimum_lr < 0
        or not np.isfinite(warmup_start_factor)
        or not 0.0 < warmup_start_factor <= 1.0
    ):
        raise ValueError("Invalid epoch scheduler configuration")
    base_learning_rates = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    if (
        not base_learning_rates
        or any(
            not np.isfinite(value)
            or value <= 0
            or minimum_lr > value
            for value in base_learning_rates
        )
    ):
        raise ValueError(
            "Scheduler requires positive base learning rates and "
            "minimum_lr no greater than each base rate"
        )
    if normalized == "cosine" and warmup_epochs == 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=minimum_lr,
        )
    decay_epochs = num_epochs - warmup_epochs

    def schedule(base_lr: float) -> Callable[[int], float]:
        minimum_factor = minimum_lr / base_lr

        def factor(epoch: int) -> float:
            if epoch < warmup_epochs:
                progress = epoch / float(warmup_epochs)
                return warmup_start_factor + (
                    1.0 - warmup_start_factor
                ) * progress
            if normalized == "constant":
                return 1.0
            progress = min(
                1.0,
                (epoch - warmup_epochs) / float(decay_epochs),
            )
            if normalized == "linear":
                return minimum_factor + (
                    1.0 - minimum_factor
                ) * (1.0 - progress)
            return minimum_factor + 0.5 * (
                1.0 - minimum_factor
            ) * (1.0 + math.cos(math.pi * progress))

        return factor

    return LambdaLR(
        optimizer,
        lr_lambda=[
            schedule(float(group["lr"]))
            for group in optimizer.param_groups
        ],
    )


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    normalized = str(name).lower()
    if normalized not in {"adam", "adamw", "radam"}:
        raise ValueError("Optimizer must be adam, adamw, or radam")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Optimizer learning rate must be finite and positive")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            "Optimizer weight decay must be finite and non-negative"
        )
    optimizer_types = {
        "adam": Adam,
        "adamw": AdamW,
        "radam": RAdam,
    }
    return optimizer_types[normalized](
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_text_write(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    # CUBLAS requires this before the first CUDA operation for reproducible GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_unlabeled_feature_pair(
    config: Mapping[str, Any],
    entity: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    return load_feature_pair(
        str(config["dataset"]),
        config["data_root"],
        entity=entity,
        expected_channels=int(config["in_channels"]),
    )


def load_unlabeled_training_data(
    config: Mapping[str, Any],
    entity: str | None,
) -> np.ndarray:
    """Compatibility helper returning only the normal source features."""
    return load_unlabeled_feature_pair(config, entity)[0]


@dataclass(frozen=True)
class PreparedProtocolData:
    """Preprocessed datasets needed by the training phase."""

    imputer: CausalImputer
    scaler: StandardScaler
    split: UnifiedTemporalSplit
    train_dataset: WindowDataset
    validation_dataset: WindowDataset


def prepare_protocol_datasets(
    train_data: np.ndarray,
    test_data: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> PreparedProtocolData:
    """Prepare normal training windows and one isolated validation block."""
    window_size = int(config["window_size"])
    stride = int(config["stride"])
    normal_source = np.asarray(train_data, dtype=np.float32)
    suffix = np.asarray(test_data, dtype=np.float32)
    if normal_source.ndim == 1:
        normal_source = normal_source[:, None]
    if suffix.ndim == 1:
        suffix = suffix[:, None]
    if normal_source.ndim != 2 or normal_source.shape[1] < 1:
        raise ValueError(f"Expected [time, channels] training data, got {normal_source.shape}")
    if suffix.ndim != 2 or suffix.shape[1] != normal_source.shape[1]:
        raise ValueError(
            "Train and test feature arrays must be 2D with equal channel counts"
        )
    split = split_full_series(
        train_length=len(normal_source),
        test_length=len(suffix),
        seed=int(config["seed"] if seed is None else seed),
        train_fraction=float(config["train_fraction"]),
        post_train_blocks=int(config["post_train_blocks"]),
    )
    normal_prefix = normal_source[: min(len(normal_source), split.train_end)]
    if len(normal_prefix) < window_size:
        raise ValueError("Training data is too short for windowing")
    if split.validation.length < window_size:
        raise ValueError("Validation block is too short for windowing")

    imputer = CausalImputer().fit(normal_prefix)
    full_raw = np.concatenate([normal_source, suffix], axis=0)
    imputed_train = imputer.transform(normal_prefix)
    imputed_full = imputer.transform(full_raw)
    scaler = StandardScaler().fit(imputed_train)
    standardized_train = scaler.transform(imputed_train)
    standardized_full = scaler.transform(imputed_full)
    validation_raw = standardized_full[
        split.validation.start:split.validation.end
    ]

    train_dataset = WindowDataset(
        standardized_train, window_size, stride, cover_end=True
    )
    val_dataset = WindowDataset(
        validation_raw, window_size, stride, cover_end=True
    )
    if len(train_dataset) == 0:
        raise ValueError("No complete training windows are available")
    if len(val_dataset) == 0:
        raise ValueError("No complete validation windows are available")
    return PreparedProtocolData(
        imputer=imputer,
        scaler=scaler,
        split=split,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
    )


def limit_window_dataset(
    dataset: WindowDataset,
    limit: int,
) -> WindowDataset:
    if limit < 0:
        raise ValueError("Window limit must be non-negative")
    if limit == 0 or len(dataset) <= limit:
        return dataset
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=limit,
        dtype=np.int64,
    )
    starts = [dataset.starts[int(index)] for index in indices]
    if len(set(starts)) != limit:
        raise RuntimeError("Window limiting produced duplicate starts")
    return WindowDataset(
        dataset.data,
        dataset.window_size,
        dataset.stride,
        starts=starts,
    )


@contextmanager
def _frozen_batch_norm_stats(model: nn.Module):
    batch_norms = [
        module for module in model.modules() if isinstance(module, _BatchNorm)
    ]
    states = [module.training for module in batch_norms]
    try:
        for module in batch_norms:
            module.eval()
        yield
    finally:
        for module, training in zip(batch_norms, states):
            module.train(training)


def generate_adversarial_negative(
    model: nn.Module,
    phase_transform: FreqPhaseHardNeg,
    criterion: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    starts: torch.Tensor,
    *,
    window_size: int,
    steps: int,
    step_size: float,
    random_start: bool,
    channel_loss_weight: float = 0.0,
    channel_sparse_ratio: float = 1.0,
    strength_scale: float = 1.0,
) -> torch.Tensor:
    """Generate bounded phase-PGD negatives without accumulating model gradients."""
    if anchor.shape != positive.shape or anchor.ndim != 3:
        raise ValueError("anchor and positive must have equal [batch, channels, time] shapes")
    if steps < 1:
        raise ValueError("PGD steps must be positive")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("PGD step_size must be finite and positive")
    if not np.isfinite(strength_scale) or not 0.0 < strength_scale <= 1.0:
        raise ValueError("strength_scale must be finite and in (0, 1]")
    if not torch.isfinite(anchor).all() or not torch.isfinite(positive).all():
        raise ValueError("PGD inputs contain NaN/Inf")
    starts_tensor = torch.as_tensor(starts, dtype=torch.long, device=anchor.device)
    if starts_tensor.shape != (len(anchor),):
        raise ValueError("starts length must match the PGD batch size")
    attack_objective = getattr(criterion, "attack_objective", None)
    if not callable(attack_objective):
        raise TypeError("criterion must provide attack_objective")

    if (
        not np.isfinite(channel_loss_weight)
        or channel_loss_weight < 0
        or not np.isfinite(channel_sparse_ratio)
        or not 0.0 < channel_sparse_ratio <= 1.0
    ):
        raise ValueError("Invalid channel loss weight or sparse ratio")

    def combined_objective(
        negative_embedding: torch.Tensor,
        negative_channels: torch.Tensor | None,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        anchor_channels: torch.Tensor | None,
        positive_channels: torch.Tensor | None,
    ) -> torch.Tensor:
        objective = attack_objective(
            anchor_embedding,
            positive_embedding,
            negative_embedding,
            starts_tensor,
            window_size,
        )
        if channel_loss_weight > 0:
            if (
                anchor_channels is None
                or positive_channels is None
                or negative_channels is None
            ):
                raise RuntimeError("Channel embeddings are required by this objective")
            channel_count = int(anchor_channels.shape[1])
            channel_starts = starts_tensor.repeat_interleave(channel_count)
            objective = objective + float(channel_loss_weight) * attack_objective(
                anchor_channels.reshape(-1, anchor_channels.shape[-1]),
                positive_channels.reshape(-1, positive_channels.shape[-1]),
                negative_channels.reshape(-1, negative_channels.shape[-1]),
                channel_starts,
                window_size,
            )
        return objective

    with _frozen_batch_norm_stats(model):
        with torch.no_grad():
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            anchor_channels = (
                model.forward_channel_embeddings(anchor)
                if channel_loss_weight > 0
                else None
            )
            positive_channels = (
                model.forward_channel_embeddings(positive)
                if channel_loss_weight > 0
                else None
            )
        channel_mask = torch.ones(
            (1, len(anchor), anchor.shape[1], 1),
            device=anchor.device,
            dtype=anchor.dtype,
        )
        if channel_sparse_ratio < 1.0:
            probe = phase_transform.zero_delta(anchor).requires_grad_(True)
            probe_negative = phase_transform.apply_delta(anchor, probe)
            probe_objective = combined_objective(
                model(probe_negative),
                (
                    model.forward_channel_embeddings(probe_negative)
                    if channel_loss_weight > 0
                    else None
                ),
                anchor_embedding.detach(),
                positive_embedding.detach(),
                (
                    anchor_channels.detach()
                    if anchor_channels is not None
                    else None
                ),
                (
                    positive_channels.detach()
                    if positive_channels is not None
                    else None
                ),
            )
            probe_gradient = torch.autograd.grad(
                probe_objective,
                probe,
                only_inputs=True,
            )[0]
            if not torch.isfinite(probe_gradient).all():
                raise FloatingPointError(
                    "Sparse PGD channel-selection gradient contains NaN/Inf"
                )
            importance = probe_gradient.abs().mean(dim=(0, 3))
            selected_count = max(
                1,
                int(np.ceil(anchor.shape[1] * channel_sparse_ratio)),
            )
            selected = importance.topk(
                selected_count,
                dim=1,
                largest=True,
            ).indices
            sample_mask = torch.zeros_like(importance)
            sample_mask.scatter_(1, selected, 1.0)
            channel_mask = sample_mask.unsqueeze(0).unsqueeze(-1)
        delta = (
            phase_transform.random_delta(anchor)
            if random_start
            else phase_transform.zero_delta(anchor)
        ) * channel_mask * float(strength_scale)
        segment_length = min(phase_transform.seg_len, anchor.shape[-1])
        effective_epsilon = (
            phase_transform.epsilon * float(strength_scale)
        )
        best_objective = float("-inf")
        best_delta = delta.detach().clone()
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            negative = phase_transform.apply_delta(anchor, delta)
            negative_embedding = model(negative)
            objective = combined_objective(
                negative_embedding,
                (
                    model.forward_channel_embeddings(negative)
                    if channel_loss_weight > 0
                    else None
                ),
                anchor_embedding.detach(),
                positive_embedding.detach(),
                (
                    anchor_channels.detach()
                    if anchor_channels is not None
                    else None
                ),
                (
                    positive_channels.detach()
                    if positive_channels is not None
                    else None
                ),
            )
            objective_value = float(objective.detach())
            if objective_value > best_objective:
                best_objective = objective_value
                best_delta = delta.detach().clone()
            gradient = torch.autograd.grad(
                objective,
                delta,
                only_inputs=True,
            )[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("PGD phase gradient contains NaN/Inf")
            delta = phase_transform.project_delta(
                (
                    delta
                    + float(step_size)
                    * float(strength_scale)
                    * gradient.sign()
                )
                * channel_mask,
                segment_length=segment_length,
                max_epsilon=effective_epsilon,
            )
        with torch.no_grad():
            final_negative = phase_transform.apply_delta(anchor, delta)
            final_objective = float(
                combined_objective(
                    model(final_negative),
                    (
                        model.forward_channel_embeddings(final_negative)
                        if channel_loss_weight > 0
                        else None
                    ),
                    anchor_embedding,
                    positive_embedding,
                    anchor_channels,
                    positive_channels,
                )
            )
        if final_objective > best_objective:
            best_delta = delta.detach().clone()
        adversarial = phase_transform.apply_delta(anchor, best_delta).detach()
    if not torch.isfinite(adversarial).all():
        raise FloatingPointError("PGD produced NaN/Inf negative samples")
    return adversarial


def _batch_pairs(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = batch.get("anchor")
    positive = batch.get("positive")
    starts = batch.get("start")
    if not isinstance(anchor, torch.Tensor) or not isinstance(positive, torch.Tensor):
        raise TypeError("Batch anchor and positive fields must be tensors")
    if not isinstance(starts, torch.Tensor):
        starts = torch.as_tensor(starts, dtype=torch.long)
    return (
        anchor.to(device, non_blocking=True),
        positive.to(device, non_blocking=True),
        starts.to(device, dtype=torch.long, non_blocking=True),
    )


def _view_embeddings(
    model: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = model(torch.cat([anchor, positive, negative], dim=0))
    return torch.chunk(embeddings, 3, dim=0)


def _combined_contrastive_loss(
    model: nn.Module,
    criterion: nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    starts: torch.Tensor,
    window_size: int,
    *,
    channel_loss_weight: float,
) -> torch.Tensor:
    anchor_embedding, positive_embedding, negative_embedding = _view_embeddings(
        model,
        anchor,
        positive,
        negative,
    )
    loss = criterion(
        anchor_embedding,
        positive_embedding,
        negative_embedding,
        starts,
        window_size,
    )
    if channel_loss_weight > 0:
        channel_embeddings = model.forward_channel_embeddings(
            torch.cat([anchor, positive, negative], dim=0)
        )
        anchor_channels, positive_channels, negative_channels = torch.chunk(
            channel_embeddings,
            3,
            dim=0,
        )
        channel_count = int(anchor_channels.shape[1])
        channel_starts = starts.repeat_interleave(channel_count)
        loss = loss + float(channel_loss_weight) * criterion(
            anchor_channels.reshape(-1, anchor_channels.shape[-1]),
            positive_channels.reshape(-1, positive_channels.shape[-1]),
            negative_channels.reshape(-1, negative_channels.shape[-1]),
            channel_starts,
            window_size,
        )
    return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    phase_transform: FreqPhaseHardNeg,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    *,
    window_size: int,
    pgd_steps: int,
    pgd_step_size: float,
    channel_loss_weight: float = 0.0,
    channel_sparse_ratio: float = 1.0,
    adversarial_strength: float = 1.0,
    ema_model: nn.Module | None = None,
    ema_decay: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        anchor, positive, starts = _batch_pairs(batch, device)
        negative = generate_adversarial_negative(
            model,
            phase_transform,
            criterion,
            anchor,
            positive,
            starts,
            window_size=window_size,
            steps=pgd_steps,
            step_size=pgd_step_size,
            random_start=True,
            channel_loss_weight=channel_loss_weight,
            channel_sparse_ratio=channel_sparse_ratio,
            strength_scale=adversarial_strength,
        )
        loss = _combined_contrastive_loss(
            model,
            criterion,
            anchor,
            positive,
            negative,
            starts,
            window_size,
            channel_loss_weight=channel_loss_weight,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("Gradient norm became NaN/Inf")
        optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, ema_decay)
        batch_size = int(batch["anchor"].shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
    if total_items == 0:
        raise RuntimeError("Training loader produced no batches")
    return total_loss / total_items


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    phase_transform: FreqPhaseHardNeg,
    device: torch.device,
    *,
    window_size: int,
    pgd_steps: int,
    pgd_step_size: float,
    channel_loss_weight: float = 0.0,
    channel_sparse_ratio: float = 1.0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        anchor, positive, starts = _batch_pairs(batch, device)
        negative = generate_adversarial_negative(
            model,
            phase_transform,
            criterion,
            anchor,
            positive,
            starts,
            window_size=window_size,
            steps=pgd_steps,
            step_size=pgd_step_size,
            random_start=False,
            channel_loss_weight=channel_loss_weight,
            channel_sparse_ratio=channel_sparse_ratio,
        )
        with torch.no_grad():
            loss = _combined_contrastive_loss(
                model,
                criterion,
                anchor,
                positive,
                negative,
                starts,
                window_size,
                channel_loss_weight=channel_loss_weight,
            )
        batch_size = int(batch["anchor"].shape[0])
        total_loss += float(loss) * batch_size
        total_items += batch_size
    if total_items == 0:
        raise RuntimeError("Validation loader produced no batches")
    return total_loss / total_items


@torch.no_grad()
def collect_window_embeddings(
    model: nn.Module,
    dataset: WindowDataset | Subset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: list[torch.Tensor] = []
    starts: list[int] = []
    model.eval()
    for batch in loader:
        embeddings.append(model(batch["x"].to(device)).cpu())
        starts.extend(int(value) for value in batch["start"])
    if not embeddings:
        raise RuntimeError("No window embeddings were generated")
    result = torch.cat(embeddings, dim=0).float()
    if len(result) != len(starts) or not torch.isfinite(result).all():
        raise RuntimeError("Window embedding generation failed")
    return result, starts


@torch.no_grad()
def collect_reference_embeddings(
    model: nn.Module,
    dataset: WindowDataset,
    *,
    batch_size: int,
    limit: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if limit < 1:
        raise ValueError("reference_limit must be positive")
    if len(dataset) > limit:
        indices = np.linspace(0, len(dataset) - 1, num=limit, dtype=np.int64).tolist()
        source = Subset(dataset, indices)
    else:
        source = dataset
    reference, _ = collect_window_embeddings(
        model,
        source,
        batch_size=batch_size,
        device=device,
    )
    prototype = torch.nn.functional.normalize(reference.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    if not torch.isfinite(reference).all() or not torch.isfinite(prototype).all():
        raise FloatingPointError("Reference embeddings contain NaN/Inf")
    return reference, prototype


@torch.no_grad()
def collect_channel_reference_embeddings(
    model: ContrastiveModel,
    dataset: WindowDataset,
    *,
    batch_size: int,
    limit: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    if model.channel_features_dim < 2:
        raise ValueError("Model does not provide channel embeddings")
    if limit < 1:
        raise ValueError("reference_limit must be positive")
    if len(dataset) > limit:
        indices = np.linspace(
            0,
            len(dataset) - 1,
            num=limit,
            dtype=np.int64,
        ).tolist()
        source: WindowDataset | Subset = Subset(dataset, indices)
    else:
        source = dataset
    loader = DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    embeddings: list[torch.Tensor] = []
    starts: list[int] = []
    model.eval()
    for batch in loader:
        embeddings.append(
            model.forward_channel_embeddings(batch["x"].to(device)).cpu()
        )
        starts.extend(int(value) for value in batch["start"])
    if not embeddings:
        raise RuntimeError("No channel reference embeddings were generated")
    result = torch.cat(embeddings, dim=0).float()
    if (
        result.ndim != 3
        or len(result) != len(starts)
        or not torch.isfinite(result).all()
    ):
        raise RuntimeError("Channel reference embedding generation failed")
    return result, starts


def embedding_health(embeddings: torch.Tensor) -> dict[str, float]:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or embeddings.shape[1] < 2:
        raise ValueError("Embedding health requires [samples, embedding_dim] data")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("Embedding health input contains NaN/Inf")
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)
    mean_feature_std = normalized.std(dim=0, unbiased=False).mean()
    if len(normalized) > 1:
        similarity = normalized @ normalized.T
        mask = ~torch.eye(
            len(normalized),
            dtype=torch.bool,
            device=normalized.device,
        )
        mean_pairwise_cosine = similarity[mask].mean()
    else:
        mean_pairwise_cosine = torch.tensor(1.0, device=normalized.device)
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if float(total) <= 1e-12:
        effective_rank = torch.tensor(1.0, device=normalized.device)
    else:
        probabilities = singular_values / total
        effective_rank = torch.exp(
            -(
                probabilities
                * probabilities.clamp_min(1e-12).log()
            ).sum()
        )
    result = {
        "mean_feature_std": float(mean_feature_std),
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
        "effective_rank": float(effective_rank),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Embedding health produced NaN/Inf")
    return result


def validate_embedding_health(
    health: Mapping[str, float],
    *,
    min_embedding_std: float,
    min_effective_rank: float,
) -> None:
    mean_std = float(health.get("mean_feature_std", float("nan")))
    effective_rank = float(health.get("effective_rank", float("nan")))
    if not np.isfinite(mean_std) or not np.isfinite(effective_rank):
        raise ValueError("Embedding health statistics must be finite")
    if mean_std < min_embedding_std or effective_rank < min_effective_rank:
        raise RuntimeError(
            "Embedding representation collapsed: "
            f"mean_feature_std={mean_std:.6f}, effective_rank={effective_rank:.6f}"
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = torch.as_tensor(
        state["torch"],
        dtype=torch.uint8,
        device="cpu",
    ).contiguous()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


# Backward-compatible aliases for existing internal callers.
_rng_state = capture_rng_state
_restore_rng_state = restore_rng_state


def _training_checkpoint(
    *,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    imputer: CausalImputer,
    scaler: StandardScaler,
    config: Mapping[str, Any],
    split: UnifiedTemporalSplit,
    epoch: int,
    best_score: float,
    patience_counter: int,
    global_step: int,
    best_model_state: Mapping[str, torch.Tensor],
    ema_model: ContrastiveModel | None,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "adcl_fd_v3_training",
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_model_state_dict": (
            None
            if ema_model is None
            else copy.deepcopy(ema_model.state_dict())
        ),
        "best_model_state_dict": best_model_state,
        "model_config": {**model.get_config(), "window_size": int(config["window_size"])},
        "imputer_state": imputer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "criterion_config": {
            "temperature": float(config["temperature"]),
            "variance_weight": float(config["variance_weight"]),
            "variance_target": float(config["variance_target"]),
            "overlap_policy": str(config["overlap_policy"]),
            "overlap_penalty": float(config["overlap_penalty"]),
        },
        "training_config": dict(config),
        "split_protocol": split.to_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "patience_counter": int(patience_counter),
        "global_step": int(global_step),
        "rng_state": capture_rng_state(),
    }


def train(args: argparse.Namespace) -> Path:
    config = load_config(
        args.config,
        variant=args.variant,
        overrides=parse_config_overrides(args.override),
    )
    seed = int(config["seed"] if args.seed is None else args.seed)
    config["seed"] = seed
    set_seed(seed)
    device = resolve_device(args.device)
    entity = args.entity if args.entity is not None else config.get("entity")
    train_data, test_data = load_unlabeled_feature_pair(config, entity)
    prepared = prepare_protocol_datasets(
        train_data,
        test_data,
        config,
        seed=seed,
    )
    imputer = prepared.imputer
    scaler = prepared.scaler
    split = prepared.split
    base_train = prepared.train_dataset
    base_val = prepared.validation_dataset
    training_base = limit_window_dataset(
        base_train,
        int(config["max_train_windows"]),
    )
    validation_base = limit_window_dataset(
        base_val,
        int(config["max_validation_windows"]),
    )
    train_dataset = ContrastiveWindowDataset(
        training_base,
        noise_sigma=float(config["noise_sigma"]),
        seed=seed,
        augmentation=str(config["augmentation"]),
        amplitude_scale=float(config["amplitude_scale"]),
        mask_ratio=float(config["mask_ratio"]),
        max_shift=int(config["max_shift"]),
    )
    val_dataset = ContrastiveWindowDataset(
        validation_base,
        noise_sigma=float(config["noise_sigma"]),
        seed=seed + 17,
        augmentation=str(config["augmentation"]),
        amplitude_scale=float(config["amplitude_scale"]),
        mask_ratio=float(config["mask_ratio"]),
        max_shift=int(config["max_shift"]),
    )
    batch_size = int(config["batch_size"])
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")

    model = ContrastiveModel(
        in_channels=int(config["in_channels"]), mid_channels=int(config["mid_channels"]),
        head=str(config["head"]), features_dim=int(config["features_dim"]),
        representation=str(config["representation"]),
        patch_sizes=tuple(int(value) for value in config["patch_sizes"]),
        channel_features_dim=int(config["channel_features_dim"]),
        per_channel_width=int(config["global_per_channel_width"]),
        stem_kernel_size=int(config["global_stem_kernel_size"]),
        residual_kernel_sizes=tuple(
            int(value)
            for value in config["global_residual_kernel_sizes"]
        ),
        block_width_multipliers=tuple(
            int(value)
            for value in config["global_block_width_multipliers"]
        ),
        pooled_statistics=tuple(
            str(value) for value in config["global_pooled_statistics"]
        ),
        multiscale_minimum_patch_count=int(
            config["global_multiscale_minimum_patch_count"]
        ),
    ).to(device)
    ema_decay = float(config["ema_decay"])
    ema_model = (
        None
        if ema_decay == 0.0
        else copy.deepcopy(model).requires_grad_(False)
    )
    criterion = AdCLFDV3Loss(
        temperature=float(config["temperature"]),
        variance_weight=float(config["variance_weight"]),
        variance_target=float(config["variance_target"]),
        overlap_policy=str(config["overlap_policy"]),
        overlap_penalty=float(config["overlap_penalty"]),
    ).to(device)
    phase_transform = FreqPhaseHardNeg(
        epsilon=float(config["phase_epsilon"]),
        perturbation_mode=str(config["phase_perturbation_mode"]),
        seg_len=int(config["segment_length"]),
        overlap_ratio=float(config["phase_segment_overlap_ratio"]),
        overlap_add_window=str(config["phase_overlap_add_window"]),
    ).to(device)
    optimizer = build_optimizer(
        model.parameters(),
        name=str(config["global_optimizer"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_epoch_scheduler(
        optimizer,
        name=str(config["global_scheduler"]),
        num_epochs=int(config["num_epochs"]),
        minimum_lr=float(config.get("min_learning_rate", 1e-6)),
        warmup_epochs=int(config["warmup_epochs"]),
        warmup_start_factor=float(config["warmup_start_factor"]),
    )

    entity_name = str(entity or "default").replace("/", "_").replace("\\", "_")
    output_dir = Path(args.output_dir or config.get("output_dir") or Path("checkpoints") / str(config["dataset"]) / entity_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best_model.pt"

    start_epoch = 0
    best_score = float("inf")
    patience_counter = 0
    global_step = 0
    best_model_state = copy.deepcopy(model.state_dict())
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = load_checkpoint_file(resume_path, device)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Resume checkpoint root must be a mapping")
        if checkpoint.get("checkpoint_type") != "adcl_fd_v3_training":
            raise ValueError("Resume checkpoint is not an AdCL-FD v3 training checkpoint")
        checkpoint_model_config = checkpoint.get("model_config")
        if (
            not isinstance(checkpoint_model_config, Mapping)
            or normalize_legacy_contrastive_model_config(
                checkpoint_model_config
            )
            != {**model.get_config(), "window_size": int(config["window_size"])}
        ):
            raise ValueError("Resume checkpoint model configuration does not match the current config")
        checkpoint_training_config = checkpoint.get("training_config")
        if not isinstance(checkpoint_training_config, Mapping):
            raise ValueError("Resume checkpoint lacks training_config")
        if (
            normalize_legacy_training_config(checkpoint_training_config)
            != dict(config)
        ):
            raise ValueError("Resume checkpoint training configuration does not match the current config")
        resumed_split = UnifiedTemporalSplit.from_dict(
            checkpoint.get("split_protocol", {})
        )
        if resumed_split != split:
            raise ValueError("Resume checkpoint temporal split does not match current data")
        resumed_imputer = CausalImputer.from_state_dict(
            checkpoint.get("imputer_state", {})
        )
        resumed_scaler = StandardScaler.from_state_dict(
            checkpoint.get("scaler_state", {})
        )
        if not np.array_equal(resumed_imputer.fill_values, imputer.fill_values):
            raise ValueError("Resume checkpoint imputer state does not match current training data")
        if (
            not np.array_equal(resumed_scaler.mean, scaler.mean)
            or not np.array_equal(resumed_scaler.std, scaler.std)
        ):
            raise ValueError("Resume checkpoint scaler state does not match current training data")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if ema_model is not None:
            ema_state = checkpoint.get("ema_model_state_dict")
            if not isinstance(ema_state, Mapping):
                raise ValueError(
                    "Resume checkpoint lacks ema_model_state_dict"
                )
            ema_model.load_state_dict(ema_state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_model_state = checkpoint["best_model_state_dict"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        patience_counter = int(checkpoint["patience_counter"])
        global_step = int(checkpoint["global_step"])
        restore_rng_state(checkpoint["rng_state"])

    print(f"AdCL-FD v3 | dataset={config['dataset']} entity={entity_name} device={device}")
    print(
        f"train_windows={len(training_base)}/{len(base_train)} "
        f"val_windows={len(validation_base)}/{len(base_val)}"
    )
    for epoch in range(start_epoch, int(config["num_epochs"])):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(0)
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            phase_transform,
            optimizer,
            device,
            float(config.get("grad_clip", 1.0)),
            window_size=int(config["window_size"]),
            pgd_steps=int(config["pgd_steps"]),
            pgd_step_size=float(config["pgd_step_size"]),
            channel_loss_weight=float(config["channel_loss_weight"]),
            channel_sparse_ratio=float(config["channel_sparse_ratio"]),
            adversarial_strength=adversarial_ramp_scale(
                epoch,
                int(config["adversarial_ramp_epochs"]),
            ),
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        validation_model = ema_model if ema_model is not None else model
        val_loss = validate_epoch(
            validation_model,
            val_loader,
            criterion,
            phase_transform,
            device,
            window_size=int(config["window_size"]),
            pgd_steps=int(config["pgd_steps"]),
            pgd_step_size=float(config["pgd_step_size"]),
            channel_loss_weight=float(config["channel_loss_weight"]),
            channel_sparse_ratio=float(config["channel_sparse_ratio"]),
        )
        scheduler.step()
        global_step += len(train_loader)
        improved = val_loss < best_score - float(config.get("min_delta", 1e-6))
        if improved:
            best_score = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(
                validation_model.state_dict()
            )
        else:
            patience_counter += 1
        atomic_torch_save(
            _training_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                imputer=imputer, scaler=scaler,
                config=config, split=split, epoch=epoch, best_score=best_score,
                patience_counter=patience_counter, global_step=global_step,
                best_model_state=best_model_state,
                ema_model=ema_model,
            ),
            latest_path,
        )
        print(f"epoch={epoch + 1}/{config['num_epochs']} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_score:.6f}")
        if patience_counter >= int(config["patience"]):
            print(f"early_stop epoch={epoch + 1}")
            break

    model.load_state_dict(best_model_state, strict=True)
    reference, prototype = collect_reference_embeddings(
        model, base_train, batch_size=batch_size, limit=int(config["reference_limit"]), device=device
    )
    channel_reference = None
    channel_reference_starts = None
    if model.channel_features_dim > 0:
        channel_reference, channel_reference_starts = (
            collect_channel_reference_embeddings(
                model,
                base_train,
                batch_size=batch_size,
                limit=int(config["reference_limit"]),
                device=device,
            )
        )
        if len(channel_reference) != len(reference):
            raise RuntimeError(
                "Global and channel reference assets are not aligned"
            )
    health = embedding_health(reference)
    validate_embedding_health(
        health,
        min_embedding_std=float(config["min_embedding_std"]),
        min_effective_rank=float(config["min_effective_rank"]),
    )
    scoring = str(config["scoring"])
    k = int(config["k"])
    if scoring == "knn" and len(reference) < k:
        raise ValueError(
            f"KNN scoring requires at least k={k} training references, got {len(reference)}"
        )
    if scoring == "hybrid" and len(reference) <= k:
        raise ValueError(
            f"Hybrid scoring requires k={k} smaller than {len(reference)} references"
        )
    score_stats = (
        fit_score_statistics(
            reference,
            prototype,
            k=k,
            mix_weight=float(config["hybrid_mix_weight"]),
        )
        if scoring == "hybrid"
        else None
    )
    calibration_dataset = WindowDataset(
        base_val.data,
        int(config["window_size"]),
        int(config["eval_stride"]),
        cover_end=True,
    )
    calibration_embeddings, calibration_starts = collect_window_embeddings(
        model,
        calibration_dataset,
        batch_size=batch_size,
        device=device,
    )
    calibration_window_scores = score_embeddings(
        calibration_embeddings,
        reference,
        prototype,
        scoring=scoring,
        k=k,
        mix_weight=float(config["hybrid_mix_weight"]),
        score_stats=score_stats,
    )
    calibration_point_scores, threshold = calibrate_point_threshold(
        calibration_window_scores,
        calibration_starts,
        series_length=len(base_val.data),
        window_size=int(config["window_size"]),
        percentile=float(config["threshold_percentile"]),
    )
    decision_config = {
        "scoring": scoring,
        "k": k if scoring in {"knn", "hybrid"} else None,
        "eval_stride": int(config["eval_stride"]),
        "threshold_percentile": float(config["threshold_percentile"]),
        "hybrid_mix_weight": (
            float(config["hybrid_mix_weight"])
            if scoring == "hybrid"
            else None
        ),
    }
    final_checkpoint = {
        "checkpoint_type": "adcl_fd_v3_inference",
        "model_state_dict": model.state_dict(),
        "model_config": {**model.get_config(), "window_size": int(config["window_size"])},
        "imputer_state": imputer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "reference_embeddings": reference,
        "reference_prototype": prototype,
        "channel_reference_embeddings": channel_reference,
        "channel_reference_starts": channel_reference_starts,
        "score_statistics": score_stats,
        "decision_config": decision_config,
        "threshold": threshold,
        "threshold_source": "validation_point_scores",
        "validation_score_calibration": {
            "protocol": "adcl_fd_validation_point_scores_v1",
            "point_scores": np.asarray(
                calibration_point_scores,
                dtype=np.float64,
            ),
            "threshold_percentile": float(config["threshold_percentile"]),
        },
        "training_config": dict(config),
        "split_protocol": split.to_dict(),
        "dataset": str(config["dataset"]),
        "entity": entity,
        "best_score": best_score,
        "training_window_count": len(base_train),
        "validation_point_count": len(calibration_point_scores),
        "embedding_health": health,
    }
    atomic_torch_save(final_checkpoint, best_path)
    metadata = {
        "checkpoint": str(best_path), "dataset": str(config["dataset"]), "entity": entity,
        "best_validation_loss": best_score, "reference_count": len(reference),
        "validation_point_count": len(calibration_point_scores),
        "threshold": threshold, "decision_config": decision_config,
        "embedding_health": health, "seed": seed,
        "split_protocol": split.to_dict(),
    }
    atomic_text_write(json.dumps(metadata, indent=2, ensure_ascii=False), output_dir / "training_summary.json")
    print(f"saved={best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fully unlabeled AdCL-FD v3")
    parser.add_argument("--config", required=True, help="Dataset YAML under configs/")
    parser.add_argument(
        "--variant",
        default=None,
        help="Experiment variant from configs/branches.yaml",
    )
    parser.add_argument("--entity", default=None, help="Dataset entity/file name when required")
    parser.add_argument("--output-dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--resume", default=None, help="Path to latest.pt from the same config")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=YAML_VALUE",
        help="Override one validated config key after applying the variant",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
