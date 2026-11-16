"""Train FP-LIR-AdCL with the maintained unlabeled temporal protocol."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.training.adcl_pretext import (
    adversarial_ramp_scale,
    atomic_text_write,
    atomic_torch_save,
    build_epoch_scheduler,
    build_optimizer,
    capture_rng_state,
    limit_window_dataset,
    load_config,
    load_unlabeled_feature_pair,
    normalize_legacy_training_config,
    parse_config_overrides,
    prepare_protocol_datasets,
    resolve_device,
    restore_rng_state,
    set_seed,
    update_ema_model,
)
from src.data.dataset import WindowDataset
from src.data.freq_hard_neg import FreqPhaseHardNeg
from src.data.preprocess import CausalImputer, StandardScaler
from src.data.splitting import UnifiedTemporalSplit
from src.scoring.fplir_scoring import (
    aggregate_local_scores,
    fit_component_statistics,
    fuse_components,
    point_components,
)
from src.models.fplir import FPLIRModel
from src.models.loading import load_checkpoint_file


def _fplir_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "fplir_hidden_dim": 64,
        "fplir_token_dim": 32,
        "fplir_spectral_patch_size": 32,
        "fplir_spectral_hop": 8,
        "fplir_per_channel_width": 4,
        "fplir_stem_kernel_size": 7,
        "fplir_temporal_kernel_size": 5,
        "fplir_temporal_dilations": [1, 2, 4],
        "fplir_repair_kernel_size": 3,
        "fplir_spectral_channel_statistics": ["mean", "maximum"],
        "fplir_mask_ratio": 0.15,
        "fplir_context_weight": 0.25,
        "fplir_smooth_l1_beta": 1.0,
        "fplir_identity_weight": 0.15,
        "fplir_idempotence_weight": 0.30,
        "fplir_idempotence_token_weight": 1.0,
        "fplir_contrastive_weight": 0.20,
        "fplir_contrastive_variance_target": 0.05,
        "fplir_contrastive_variance_weight": 0.10,
        "fplir_channel_topk_ratio": 0.10,
        "fplir_score_weights": [0.45, 0.25, 0.30],
        "fplir_pgd_steps": 1,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    positive_ints = (
        "fplir_hidden_dim",
        "fplir_token_dim",
        "fplir_spectral_patch_size",
        "fplir_spectral_hop",
        "fplir_pgd_steps",
        "fplir_per_channel_width",
        "fplir_stem_kernel_size",
        "fplir_temporal_kernel_size",
        "fplir_repair_kernel_size",
    )
    if any(int(config[key]) < 1 for key in positive_ints):
        raise ValueError("FP-LIR integer hyperparameters must be positive")
    if int(config["fplir_spectral_patch_size"]) > int(config["window_size"]):
        raise ValueError("FP-LIR spectral patch cannot exceed window_size")
    if int(config["fplir_spectral_hop"]) > int(config["fplir_spectral_patch_size"]):
        raise ValueError("FP-LIR spectral hop cannot exceed patch size")
    for key in (
        "fplir_stem_kernel_size",
        "fplir_temporal_kernel_size",
        "fplir_repair_kernel_size",
    ):
        if int(config[key]) % 2 == 0:
            raise ValueError(f"{key} must be odd")
    temporal_dilations = config["fplir_temporal_dilations"]
    spectral_statistics = config["fplir_spectral_channel_statistics"]
    if not isinstance(temporal_dilations, (list, tuple)):
        raise ValueError("fplir_temporal_dilations must be a sequence")
    if not isinstance(spectral_statistics, (list, tuple)):
        raise ValueError(
            "fplir_spectral_channel_statistics must be a sequence"
        )
    normalized_dilations = [int(value) for value in temporal_dilations]
    normalized_statistics = [str(value) for value in spectral_statistics]
    if not normalized_dilations or any(
        value < 1 for value in normalized_dilations
    ):
        raise ValueError("FP-LIR temporal dilations must be positive")
    if (
        not normalized_statistics
        or len(set(normalized_statistics)) != len(normalized_statistics)
        or any(value not in {"mean", "maximum"} for value in normalized_statistics)
    ):
        raise ValueError(
            "FP-LIR spectral channel statistics are invalid"
        )
    config["fplir_temporal_dilations"] = normalized_dilations
    config["fplir_spectral_channel_statistics"] = normalized_statistics
    mask_ratio = float(config["fplir_mask_ratio"])
    topk_ratio = float(config["fplir_channel_topk_ratio"])
    if not 0.0 < mask_ratio < 1.0 or not 0.0 < topk_ratio <= 1.0:
        raise ValueError("FP-LIR mask and channel ratios are invalid")
    for key in (
        "fplir_context_weight",
        "fplir_identity_weight",
        "fplir_idempotence_weight",
        "fplir_idempotence_token_weight",
        "fplir_contrastive_weight",
        "fplir_contrastive_variance_target",
        "fplir_contrastive_variance_weight",
    ):
        if not np.isfinite(float(config[key])) or float(config[key]) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    smooth_l1_beta = float(config["fplir_smooth_l1_beta"])
    if not np.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0:
        raise ValueError("fplir_smooth_l1_beta must be finite and positive")
    score_weights = np.asarray(config["fplir_score_weights"], dtype=np.float64)
    if (
        score_weights.shape != (3,)
        or not np.isfinite(score_weights).all()
        or np.any(score_weights < 0)
        or score_weights.sum() <= 0
    ):
        raise ValueError("fplir_score_weights must contain three non-negative values")
    config["fplir_score_weights"] = score_weights.tolist()
    return config


def _model_config(
    model: FPLIRModel,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **model.get_config(),
        "window_size": int(config["window_size"]),
    }


def _normalize_legacy_fplir_model_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("per_channel_width", 4)
    normalized.setdefault("stem_kernel_size", 7)
    normalized.setdefault("temporal_kernel_size", 5)
    normalized.setdefault("temporal_dilations", [1, 2, 4])
    normalized.setdefault("repair_kernel_size", 3)
    normalized.setdefault(
        "spectral_channel_statistics",
        ["mean", "maximum"],
    )
    return normalized


def _training_checkpoint(
    *,
    model: FPLIRModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    imputer: CausalImputer,
    scaler: StandardScaler,
    config: Mapping[str, Any],
    split: UnifiedTemporalSplit,
    dataset: str,
    entity: str | None,
    epoch: int,
    best_validation_loss: float,
    patience_counter: int,
    best_model_state: Mapping[str, torch.Tensor],
    ema_model: FPLIRModel | None,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "fplir_v1_training",
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_model_state_dict": (
            None
            if ema_model is None
            else copy.deepcopy(ema_model.state_dict())
        ),
        "best_model_state_dict": best_model_state,
        "model_config": _model_config(model, config),
        "training_config": dict(config),
        "split_protocol": split.to_dict(),
        "imputer_state": imputer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "dataset": dataset,
        "entity": entity,
        "epoch": int(epoch),
        "best_validation_loss": float(best_validation_loss),
        "patience_counter": int(patience_counter),
        "rng_state": capture_rng_state(),
    }


def _restore_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: FPLIRModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Mapping[str, Any],
    split: UnifiedTemporalSplit,
    imputer: CausalImputer,
    scaler: StandardScaler,
    dataset: str,
    entity: str | None,
    ema_model: FPLIRModel | None,
) -> tuple[int, float, int, Mapping[str, torch.Tensor]]:
    if checkpoint.get("checkpoint_type") != "fplir_v1_training":
        raise ValueError("Resume checkpoint is not an FP-LIR training checkpoint")
    if checkpoint.get("dataset") != dataset or checkpoint.get("entity") != entity:
        raise ValueError("Resume checkpoint dataset/entity does not match")
    checkpoint_model_config = checkpoint.get("model_config")
    if (
        not isinstance(checkpoint_model_config, Mapping)
        or _normalize_legacy_fplir_model_config(checkpoint_model_config)
        != _model_config(model, config)
    ):
        raise ValueError("Resume checkpoint model configuration does not match")
    checkpoint_training_config = checkpoint.get("training_config")
    if not isinstance(checkpoint_training_config, Mapping):
        raise ValueError("Resume checkpoint lacks training_config")
    normalized_checkpoint_config = _fplir_config(
        normalize_legacy_training_config(checkpoint_training_config)
    )
    if normalized_checkpoint_config != dict(config):
        raise ValueError("Resume checkpoint training configuration does not match")
    resumed_split = UnifiedTemporalSplit.from_dict(
        checkpoint.get("split_protocol", {})
    )
    if resumed_split != split:
        raise ValueError("Resume checkpoint temporal split does not match")
    resumed_imputer = CausalImputer.from_state_dict(
        checkpoint.get("imputer_state", {})
    )
    resumed_scaler = StandardScaler.from_state_dict(
        checkpoint.get("scaler_state", {})
    )
    if not np.array_equal(resumed_imputer.fill_values, imputer.fill_values):
        raise ValueError("Resume checkpoint imputer state does not match")
    if (
        not np.array_equal(resumed_scaler.mean, scaler.mean)
        or not np.array_equal(resumed_scaler.std, scaler.std)
    ):
        raise ValueError("Resume checkpoint scaler state does not match")
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
    best_model_state = checkpoint.get("best_model_state_dict")
    if not isinstance(best_model_state, Mapping):
        raise ValueError("Resume checkpoint lacks best_model_state_dict")
    start_epoch = int(checkpoint["epoch"]) + 1
    best_validation_loss = float(checkpoint["best_validation_loss"])
    patience_counter = int(checkpoint["patience_counter"])
    if (
        start_epoch < 0
        or start_epoch > int(config["num_epochs"])
        or not np.isfinite(best_validation_loss)
        or patience_counter < 0
    ):
        raise ValueError("Resume checkpoint training progress is invalid")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("Resume checkpoint lacks rng_state")
    restore_rng_state(rng_state)
    return (
        start_epoch,
        best_validation_loss,
        patience_counter,
        best_model_state,
    )


def deterministic_mask(
    x: torch.Tensor,
    *,
    ratio: float,
    batch_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or not 0.0 < ratio < 1.0:
        raise ValueError("Invalid deterministic mask input")
    length = int(x.shape[-1])
    span = max(1, min(length - 1, int(round(length * ratio))))
    mask = torch.zeros_like(x, dtype=torch.bool)
    available = length - span + 1
    for batch_index in range(len(x)):
        for channel in range(x.shape[1]):
            start = (
                batch_offset * 17
                + batch_index * 13
                + channel * 7
            ) % available
            mask[batch_index, channel, start:start + span] = True
    corrupted = x.masked_fill(mask, 0.0)
    return corrupted, mask


def local_contrastive_loss(
    anchor_tokens: torch.Tensor,
    positive_tokens: torch.Tensor,
    negative_tokens: torch.Tensor,
    *,
    temperature: float,
    stride: int,
    variance_target: float,
    variance_weight: float,
) -> torch.Tensor:
    if (
        anchor_tokens.shape != positive_tokens.shape
        or anchor_tokens.shape != negative_tokens.shape
        or anchor_tokens.ndim != 3
    ):
        raise ValueError("Local contrastive tokens must have equal 3D shapes")
    if (
        temperature <= 0
        or stride < 1
        or variance_target < 0
        or variance_weight < 0
    ):
        raise ValueError("Invalid local contrastive configuration")
    anchor = anchor_tokens[..., ::stride]
    positive = positive_tokens[..., ::stride]
    negative = negative_tokens[..., ::stride]
    positive_similarity = (anchor * positive).sum(dim=1)
    negative_similarity = (anchor * negative).sum(dim=1)
    ranking = F.softplus(
        (negative_similarity - positive_similarity) / temperature
    ).mean()
    feature_std = anchor.transpose(1, 2).reshape(-1, anchor.shape[1]).std(
        dim=0,
        unbiased=False,
    )
    variance = torch.relu(variance_target - feature_std).mean()
    return ranking + variance_weight * variance


def generate_local_hard_negative(
    model: FPLIRModel,
    phase_transform: FreqPhaseHardNeg,
    anchor: torch.Tensor,
    anchor_tokens: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    channel_sparse_ratio: float,
    token_stride: int,
    strength_scale: float,
) -> torch.Tensor:
    if (
        steps < 1
        or step_size <= 0
        or not 0.0 < channel_sparse_ratio <= 1.0
        or not np.isfinite(strength_scale)
        or not 0.0 < strength_scale <= 1.0
    ):
        raise ValueError("Invalid FP-LIR adversarial configuration")
    delta = phase_transform.random_delta(anchor) * float(strength_scale)
    segment_length = min(phase_transform.seg_len, anchor.shape[-1])
    effective_epsilon = phase_transform.epsilon * float(strength_scale)
    channel_mask: torch.Tensor | None = None
    for step in range(steps):
        delta = delta.detach().requires_grad_(True)
        negative = phase_transform.apply_delta(anchor, delta)
        _, negative_tokens = model(negative)
        similarity = (
            anchor_tokens.detach()[..., ::token_stride]
            * negative_tokens[..., ::token_stride]
        ).sum(dim=1).mean()
        gradient = torch.autograd.grad(similarity, delta, only_inputs=True)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("FP-LIR phase gradient contains NaN/Inf")
        if step == 0 and channel_sparse_ratio < 1.0:
            importance = gradient.abs().mean(dim=(0, 3))
            selected_count = max(
                1,
                int(np.ceil(anchor.shape[1] * channel_sparse_ratio)),
            )
            selected = importance.topk(selected_count, dim=1).indices
            sample_mask = torch.zeros_like(importance)
            sample_mask.scatter_(1, selected, 1.0)
            channel_mask = sample_mask.unsqueeze(0).unsqueeze(-1)
        if channel_mask is not None:
            gradient = gradient * channel_mask
            delta = delta * channel_mask
        delta = phase_transform.project_delta(
            delta
            + step_size * float(strength_scale) * gradient.sign(),
            segment_length=segment_length,
            max_epsilon=effective_epsilon,
        )
    return phase_transform.apply_delta(anchor, delta.detach()).detach()


def batch_loss(
    model: FPLIRModel,
    phase_transform: FreqPhaseHardNeg,
    x: torch.Tensor,
    *,
    config: Mapping[str, Any],
    batch_offset: int,
    adversarial: bool,
    adversarial_strength: float,
) -> torch.Tensor:
    corrupted, mask = deterministic_mask(
        x,
        ratio=float(config["fplir_mask_ratio"]),
        batch_offset=batch_offset,
    )
    positive = x + torch.randn_like(x) * float(config["noise_sigma"])
    repair_corrupted, corrupted_tokens = model(corrupted)
    repair_clean, clean_tokens = model(x)
    _, positive_tokens = model(positive)
    repair_twice, repaired_tokens = model(repair_corrupted)

    smooth_l1_beta = float(config["fplir_smooth_l1_beta"])
    masked_error = F.smooth_l1_loss(
        repair_corrupted[mask],
        x[mask],
        beta=smooth_l1_beta,
    )
    context_error = F.smooth_l1_loss(
        repair_corrupted[~mask],
        x[~mask],
        beta=smooth_l1_beta,
    )
    reconstruction = (
        masked_error
        + float(config["fplir_context_weight"]) * context_error
    )
    identity = F.smooth_l1_loss(
        repair_clean,
        x,
        beta=smooth_l1_beta,
    )
    idempotence = F.smooth_l1_loss(
        repair_twice,
        repair_corrupted.detach(),
        beta=smooth_l1_beta,
    ) + float(config["fplir_idempotence_token_weight"]) * (
        1.0
        - (corrupted_tokens.detach() * repaired_tokens).sum(dim=1)
    ).mean()

    if adversarial:
        hard_negative = generate_local_hard_negative(
            model,
            phase_transform,
            x,
            clean_tokens.detach(),
            steps=int(config["fplir_pgd_steps"]),
            step_size=float(config["pgd_step_size"]),
            channel_sparse_ratio=float(config["channel_sparse_ratio"]),
            token_stride=int(config["fplir_spectral_hop"]),
            strength_scale=adversarial_strength,
        )
    else:
        hard_negative = phase_transform(x).detach()
    _, negative_tokens = model(hard_negative)
    contrastive = local_contrastive_loss(
        clean_tokens,
        positive_tokens,
        negative_tokens,
        temperature=float(config["temperature"]),
        stride=int(config["fplir_spectral_hop"]),
        variance_target=float(
            config["fplir_contrastive_variance_target"]
        ),
        variance_weight=float(
            config["fplir_contrastive_variance_weight"]
        ),
    )
    loss = (
        reconstruction
        + float(config["fplir_identity_weight"]) * identity
        + float(config["fplir_idempotence_weight"]) * idempotence
        + float(config["fplir_contrastive_weight"]) * contrastive
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("FP-LIR loss is NaN/Inf")
    return loss


def run_epoch(
    model: FPLIRModel,
    loader: DataLoader,
    phase_transform: FreqPhaseHardNeg,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    adversarial_strength: float = 1.0,
    ema_model: FPLIRModel | None = None,
    ema_decay: float = 0.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    grad_context = torch.enable_grad if training else torch.no_grad
    for batch_index, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(
                model,
                phase_transform,
                x,
                config=config,
                batch_offset=batch_index,
                adversarial=True,
                adversarial_strength=adversarial_strength,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.get("grad_clip", 1.0)),
            )
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
        else:
            with grad_context():
                loss = batch_loss(
                    model,
                    phase_transform,
                    x,
                    config=config,
                    batch_offset=batch_index,
                    adversarial=False,
                    adversarial_strength=1.0,
                )
        losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("FP-LIR epoch received no batches")
    return float(np.mean(losses))


@torch.no_grad()
def collect_prototype_and_components(
    model: FPLIRModel,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    channel_topk_ratio: float,
) -> tuple[torch.Tensor, dict[str, list[float] | str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    token_sum = torch.zeros(model.token_dim, dtype=torch.float64)
    token_count = 0
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        _, tokens = model(x)
        token_sum += tokens.double().sum(dim=(0, 2)).cpu()
        token_count += int(tokens.shape[0] * tokens.shape[-1])
    if token_count < 1:
        raise RuntimeError("No FP-LIR training tokens were collected")
    prototype = F.normalize((token_sum / token_count).float(), dim=0)

    component_parts: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device)
        repair, tokens = model(x)
        _, repaired_tokens = model(repair)
        component_parts.append(
            point_components(
                x,
                repair,
                tokens,
                repaired_tokens,
                prototype.to(device),
                channel_topk_ratio=channel_topk_ratio,
            ).cpu().numpy().reshape(-1, 3)
        )
    return prototype, fit_component_statistics(np.concatenate(component_parts))


@torch.no_grad()
def score_dataset_components(
    model: FPLIRModel,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    prototype: torch.Tensor,
    component_statistics: dict[str, object],
    channel_topk_ratio: float,
) -> tuple[np.ndarray, list[int]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    windows: list[np.ndarray] = []
    starts: list[int] = []
    location = np.asarray(component_statistics.get("location"), dtype=np.float64)
    scale = np.asarray(component_statistics.get("scale"), dtype=np.float64)
    if (
        location.shape != (3,)
        or scale.shape != (3,)
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("FP-LIR component statistics are invalid")
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        repair, tokens = model(x)
        _, repaired_tokens = model(repair)
        components = point_components(
            x,
            repair,
            tokens,
            repaired_tokens,
            prototype.to(device),
            channel_topk_ratio=channel_topk_ratio,
        )
        values = components.cpu().numpy().astype(np.float64, copy=False)
        robust = np.maximum(
            (values - location[None, None, :]) / scale[None, None, :],
            0.0,
        )
        windows.append(robust)
        starts.extend(int(value) for value in batch["start"])
    if not windows:
        raise RuntimeError("No FP-LIR windows were scored")
    return np.concatenate(windows, axis=0), starts


@torch.no_grad()
def score_dataset(
    model: FPLIRModel,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    prototype: torch.Tensor,
    component_statistics: dict[str, object],
    score_weights: list[float],
    channel_topk_ratio: float,
) -> tuple[np.ndarray, list[int]]:
    components, starts = score_dataset_components(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        prototype=prototype,
        component_statistics=component_statistics,
        channel_topk_ratio=channel_topk_ratio,
    )
    weights = np.asarray(score_weights, dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("FP-LIR score weights are invalid")
    normalized = weights / weights.sum()
    fused = np.einsum("btc,c->bt", components, normalized)
    if not np.isfinite(fused).all():
        raise FloatingPointError("FP-LIR fused window scores contain NaN/Inf")
    return fused, starts


def train(args: argparse.Namespace) -> Path:
    config = _fplir_config(
        load_config(
            args.config,
            variant=args.variant,
            overrides=parse_config_overrides(args.override),
        )
    )
    seed = int(config["seed"] if args.seed is None else args.seed)
    config["seed"] = seed
    set_seed(seed)
    device = resolve_device(args.device)
    entity = args.entity if args.entity is not None else config.get("entity")
    train_data, test_data = load_unlabeled_feature_pair(config, entity)
    prepared = prepare_protocol_datasets(train_data, test_data, config, seed=seed)
    training_base = limit_window_dataset(
        prepared.train_dataset,
        int(config["max_train_windows"]),
    )
    validation_base = limit_window_dataset(
        prepared.validation_dataset,
        int(config["max_validation_windows"]),
    )
    batch_size = int(config["batch_size"])
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        training_base,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        validation_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = FPLIRModel(
        in_channels=int(config["in_channels"]),
        hidden_dim=int(config["fplir_hidden_dim"]),
        token_dim=int(config["fplir_token_dim"]),
        spectral_patch_size=int(config["fplir_spectral_patch_size"]),
        spectral_hop=int(config["fplir_spectral_hop"]),
        per_channel_width=int(config["fplir_per_channel_width"]),
        stem_kernel_size=int(config["fplir_stem_kernel_size"]),
        temporal_kernel_size=int(config["fplir_temporal_kernel_size"]),
        temporal_dilations=tuple(
            int(value) for value in config["fplir_temporal_dilations"]
        ),
        repair_kernel_size=int(config["fplir_repair_kernel_size"]),
        spectral_channel_statistics=tuple(
            str(value)
            for value in config["fplir_spectral_channel_statistics"]
        ),
    ).to(device)
    ema_decay = float(config["ema_decay"])
    ema_model = (
        None
        if ema_decay == 0.0
        else copy.deepcopy(model).requires_grad_(False)
    )
    phase_transform = FreqPhaseHardNeg(
        epsilon=float(config["phase_epsilon"]),
        perturbation_mode=str(config["phase_perturbation_mode"]),
        seg_len=int(config["segment_length"]),
        overlap_ratio=float(config["phase_segment_overlap_ratio"]),
        overlap_add_window=str(config["phase_overlap_add_window"]),
    ).to(device)
    optimizer = build_optimizer(
        model.parameters(),
        name=str(config["fplir_optimizer"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_epoch_scheduler(
        optimizer,
        name=str(config["fplir_scheduler"]),
        num_epochs=int(config["num_epochs"]),
        minimum_lr=float(config.get("min_learning_rate", 1e-6)),
        warmup_epochs=int(config["warmup_epochs"]),
        warmup_start_factor=float(config["warmup_start_factor"]),
    )

    entity_name = str(entity or "default").replace("/", "_").replace("\\", "_")
    output_dir = Path(
        args.output_dir
        or Path("checkpoints_fplir") / str(config["dataset"]) / entity_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best_model.pt"
    start_epoch = 0
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        checkpoint = load_checkpoint_file(resume_path, device)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Resume checkpoint root must be a mapping")
        (
            start_epoch,
            best_loss,
            patience_counter,
            best_state,
        ) = _restore_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            split=prepared.split,
            imputer=prepared.imputer,
            scaler=prepared.scaler,
            dataset=str(config["dataset"]),
            entity=entity,
            ema_model=ema_model,
        )

    print(
        f"FP-LIR-AdCL | dataset={config['dataset']} entity={entity_name} "
        f"device={device} train_windows={len(training_base)} "
        f"val_windows={len(validation_base)}"
    )
    if patience_counter >= int(config["patience"]):
        start_epoch = int(config["num_epochs"])
    for epoch in range(start_epoch, int(config["num_epochs"])):
        train_loss = run_epoch(
            model,
            train_loader,
            phase_transform,
            device,
            config,
            optimizer=optimizer,
            adversarial_strength=adversarial_ramp_scale(
                epoch,
                int(config["adversarial_ramp_epochs"]),
            ),
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        validation_model = ema_model if ema_model is not None else model
        val_loss = run_epoch(
            validation_model,
            val_loader,
            phase_transform,
            device,
            config,
            optimizer=None,
        )
        scheduler.step()
        improved = val_loss < best_loss - float(config.get("min_delta", 1e-6))
        if improved:
            best_loss = val_loss
            best_state = copy.deepcopy(validation_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        atomic_torch_save(
            _training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                imputer=prepared.imputer,
                scaler=prepared.scaler,
                config=config,
                split=prepared.split,
                dataset=str(config["dataset"]),
                entity=entity,
                epoch=epoch,
                best_validation_loss=best_loss,
                patience_counter=patience_counter,
                best_model_state=best_state,
                ema_model=ema_model,
            ),
            latest_path,
        )
        print(
            f"epoch={epoch + 1}/{config['num_epochs']} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best={best_loss:.6f}"
        )
        if patience_counter >= int(config["patience"]):
            print(f"early_stop epoch={epoch + 1}")
            break

    model.load_state_dict(best_state, strict=True)
    reference_dataset = limit_window_dataset(
        prepared.train_dataset,
        int(config["reference_limit"]),
    )
    prototype, component_statistics = collect_prototype_and_components(
        model,
        reference_dataset,
        batch_size=batch_size,
        device=device,
        channel_topk_ratio=float(config["fplir_channel_topk_ratio"]),
    )
    calibration_dataset = WindowDataset(
        prepared.validation_dataset.data,
        int(config["window_size"]),
        int(config["eval_stride"]),
        cover_end=True,
    )
    calibration_windows, calibration_starts = score_dataset(
        model,
        calibration_dataset,
        batch_size=batch_size,
        device=device,
        prototype=prototype,
        component_statistics=component_statistics,
        score_weights=list(config["fplir_score_weights"]),
        channel_topk_ratio=float(config["fplir_channel_topk_ratio"]),
    )
    calibration_points = aggregate_local_scores(
        calibration_windows,
        calibration_starts,
        series_length=len(calibration_dataset.data),
        window_size=int(config["window_size"]),
    )
    threshold = float(
        np.percentile(calibration_points, float(config["threshold_percentile"]))
    )
    if not np.isfinite(threshold):
        raise FloatingPointError("FP-LIR validation threshold is NaN/Inf")
    checkpoint = {
        "checkpoint_type": "fplir_v1_inference",
        "model_state_dict": model.state_dict(),
        "model_config": _model_config(model, config),
        "training_config": dict(config),
        "split_protocol": prepared.split.to_dict(),
        "imputer_state": prepared.imputer.state_dict(),
        "scaler_state": prepared.scaler.state_dict(),
        "prototype": prototype,
        "component_statistics": component_statistics,
        "decision_config": {
            "eval_stride": int(config["eval_stride"]),
            "threshold_percentile": float(config["threshold_percentile"]),
            "score_weights": list(config["fplir_score_weights"]),
            "channel_topk_ratio": float(config["fplir_channel_topk_ratio"]),
        },
        "validation_score_calibration": {
            "protocol": "fplir_validation_point_scores_v1",
            "point_scores": calibration_points,
            "threshold_percentile": float(config["threshold_percentile"]),
        },
        "threshold": threshold,
        "threshold_source": "validation_point_scores",
        "dataset": str(config["dataset"]),
        "entity": entity,
        "best_validation_loss": best_loss,
    }
    atomic_torch_save(checkpoint, best_path)
    atomic_text_write(
        json.dumps(
            {
                "checkpoint": str(best_path),
                "dataset": str(config["dataset"]),
                "entity": entity,
                "best_validation_loss": best_loss,
                "threshold": threshold,
                "training_windows": len(training_base),
                "validation_points": len(calibration_points),
                "seed": seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        output_dir / "training_summary.json",
    )
    print(f"saved={best_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FP-LIR-AdCL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="FPLIR")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to latest.pt from the same candidate and entity",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=YAML_VALUE",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
