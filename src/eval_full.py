"""Immutable AdCL-FD v3 evaluation with the fixed 13-metric protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import WindowDataset
from data.loading import load_dataset
from data.preprocess import CausalImputer, StandardScaler
from data.splitting import UnifiedTemporalSplit, split_full_series
from enhanced_scoring import (
    fuse_enhanced_scores,
    score_regime_memory,
    score_relation_windows,
    validate_regime_memory,
    validate_relation_state,
)
from metrics import FORMAL_METRICS, compute_all_metrics
from models.loading import load_model_checkpoint
from rgcsf_scoring import (
    score_channel_regimes,
    score_regime_relations,
    score_rgcsf_fusion,
    validate_channel_regime_state,
    validate_regime_relation_state,
    validate_rgcsf_calibration,
)
from scoring import aggregate_block_scores, aggregate_window_scores, score_embeddings


def resolve_checkpoint_split(
    checkpoint: Mapping[str, Any],
    *,
    train_length: int,
    test_length: int,
) -> UnifiedTemporalSplit:
    """Rebuild and verify the immutable temporal split saved at training time."""
    raw_protocol = checkpoint.get("split_protocol")
    if not isinstance(raw_protocol, Mapping):
        raise ValueError("Checkpoint is missing a valid split_protocol")
    saved = UnifiedTemporalSplit.from_dict(dict(raw_protocol))
    expected = split_full_series(
        train_length=train_length,
        test_length=test_length,
        seed=saved.seed,
        train_fraction=saved.train_fraction,
        post_train_blocks=saved.post_train_blocks,
    )
    if saved != expected:
        raise ValueError("Checkpoint temporal split does not match current data")
    return saved


def slice_test_blocks(
    full_data: np.ndarray,
    split: UnifiedTemporalSplit,
) -> tuple[np.ndarray, ...]:
    """Return independent test arrays without creating cross-block windows."""
    array = np.asarray(full_data)
    if array.ndim < 1 or len(array) != split.full_length:
        raise ValueError("Full data length does not match temporal split")
    return tuple(array[block.start:block.end].copy() for block in split.test_blocks)


def _inference_assets(
    checkpoint: Mapping[str, Any],
    features_dim: int,
    channel_features_dim: int = 0,
):
    if checkpoint.get("checkpoint_type") != "adcl_fd_v3_inference":
        raise ValueError("Formal evaluation requires an adcl_fd_v3_inference checkpoint")
    required_checkpoint_fields = {
        "imputer_state",
        "scaler_state",
        "reference_embeddings",
        "reference_prototype",
        "decision_config",
        "threshold",
        "threshold_source",
        "embedding_health",
        "split_protocol",
    }
    missing_checkpoint_fields = sorted(
        required_checkpoint_fields.difference(checkpoint)
    )
    if missing_checkpoint_fields:
        raise ValueError(
            "Incomplete inference checkpoint; missing: "
            + ", ".join(missing_checkpoint_fields)
        )
    if checkpoint["threshold_source"] != "validation_point_scores":
        raise ValueError("Checkpoint threshold_source must be validation_point_scores")
    health = checkpoint["embedding_health"]
    if not isinstance(health, Mapping):
        raise ValueError("Checkpoint embedding_health must be a mapping")
    health_keys = {
        "mean_feature_std",
        "mean_pairwise_cosine",
        "effective_rank",
    }
    missing_health = sorted(health_keys.difference(health))
    if missing_health:
        raise ValueError(
            "Incomplete embedding_health; missing: " + ", ".join(missing_health)
        )
    if not all(np.isfinite(float(health[key])) for key in health_keys):
        raise ValueError("Checkpoint embedding_health contains NaN/Inf")
    imputer = CausalImputer.from_state_dict(checkpoint.get("imputer_state", {}))
    scaler = StandardScaler.from_state_dict(checkpoint.get("scaler_state", {}))
    reference = torch.as_tensor(checkpoint.get("reference_embeddings"), dtype=torch.float32)
    prototype = torch.as_tensor(checkpoint.get("reference_prototype"), dtype=torch.float32)
    if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] != features_dim:
        raise ValueError("Checkpoint reference_embeddings have an invalid shape")
    if prototype.shape != (features_dim,):
        raise ValueError("Checkpoint reference_prototype has an invalid shape")
    if not torch.isfinite(reference).all() or not torch.isfinite(prototype).all():
        raise FloatingPointError("Checkpoint reference assets contain NaN/Inf")
    reference = torch.nn.functional.normalize(reference, dim=1)
    prototype = torch.nn.functional.normalize(prototype, dim=0)
    decision = checkpoint.get("decision_config")
    if not isinstance(decision, Mapping):
        raise ValueError("Checkpoint is missing a valid decision_config")
    required = {"scoring", "k", "eval_stride", "threshold_percentile"}
    missing = sorted(required.difference(decision))
    if missing:
        raise ValueError("Incomplete decision_config; missing: " + ", ".join(missing))
    scoring = str(decision["scoring"])
    if scoring not in {
        "knn",
        "cosine",
        "hybrid",
        "regime_memory",
        "rgcsf",
    }:
        raise ValueError(
            "decision_config scoring must be knn, cosine, hybrid, "
            "regime_memory, or rgcsf"
        )
    eval_stride = int(decision["eval_stride"])
    if eval_stride < 1:
        raise ValueError("decision_config eval_stride must be positive")
    percentile = float(decision["threshold_percentile"])
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("decision_config threshold_percentile is invalid")
    mix_weight = decision.get("hybrid_mix_weight")
    regime_memory_state = None
    relation_state = None
    relation_weight = 0.0
    rgcsf_assets = None
    if scoring == "hybrid":
        if mix_weight is None:
            raise ValueError("Hybrid decision_config requires hybrid_mix_weight")
        mix_weight = float(mix_weight)
        if not np.isfinite(mix_weight) or not 0.0 <= mix_weight <= 1.0:
            raise ValueError("Hybrid decision_config mix weight is invalid")
        score_statistics = checkpoint.get("score_statistics")
        if not isinstance(score_statistics, Mapping):
            raise ValueError("Hybrid checkpoints require score_statistics")
        if score_statistics.get("method") != "leave_one_out_robust_hybrid_v1":
            raise ValueError("Unsupported hybrid score_statistics method")
    elif scoring == "regime_memory":
        regime_memory_state = checkpoint.get("regime_memory_state")
        relation_state = checkpoint.get("relation_state")
        relation_weight = float(
            decision.get("relation_weight", float("nan"))
        )
        if relation_weight not in {0.0, 0.25, 0.5}:
            raise ValueError(
                "Enhanced decision_config relation_weight must be 0, 0.25, or 0.5"
            )
        validate_regime_memory(
            regime_memory_state,
            features_dim=features_dim,
        )
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, Mapping) or "window_size" not in model_config:
            raise ValueError(
                "Enhanced checkpoints require model_config window_size"
            )
        scaler_channels = int(np.asarray(scaler.mean).shape[-1])
        validate_relation_state(
            relation_state,
            channels=scaler_channels,
            window_size=int(model_config["window_size"]),
        )
        score_statistics = None
    elif scoring == "rgcsf":
        if channel_features_dim < 2:
            raise ValueError("RG-CSF checkpoints require channel embeddings")
        regime_memory_state = checkpoint.get("regime_memory_state")
        validate_regime_memory(
            regime_memory_state,
            features_dim=features_dim,
        )
        channel_regime_state = checkpoint.get("channel_regime_state")
        regime_relation_state = checkpoint.get("regime_relation_state")
        rgcsf_calibration = checkpoint.get("rgcsf_calibration")
        validated_channel = validate_channel_regime_state(
            channel_regime_state
        )
        validated_relation = validate_regime_relation_state(
            regime_relation_state
        )
        validated_calibration = validate_rgcsf_calibration(
            rgcsf_calibration
        )
        scaler_channels = int(np.asarray(scaler.mean).shape[-1])
        if (
            int(validated_channel["channels"]) != scaler_channels
            or int(validated_channel["features_dim"])
            != int(channel_features_dim)
            or int(validated_relation["channels"]) != scaler_channels
            or int(validated_relation["regime_count"])
            != int(validated_channel["regime_count"])
        ):
            raise ValueError("RG-CSF checkpoint assets are incompatible")
        score_statistics = None
        rgcsf_assets = {
            "channel_regime_state": validated_channel,
            "regime_relation_state": validated_relation,
            "calibration": validated_calibration,
        }
    else:
        score_statistics = None
    normalized_decision = {
        "scoring": scoring,
        "k": None if scoring == "cosine" else int(decision["k"]),
        "eval_stride": eval_stride,
        "threshold_percentile": percentile,
        "hybrid_mix_weight": mix_weight,
        "relation_weight": relation_weight,
    }
    if scoring in {"knn", "hybrid", "regime_memory", "rgcsf"} and (
        normalized_decision["k"] is None
        or normalized_decision["k"] < 1
        or normalized_decision["k"] > len(reference)
    ):
        raise ValueError("decision_config k is incompatible with training references")
    if scoring == "hybrid" and normalized_decision["k"] >= len(reference):
        raise ValueError("Hybrid decision_config k must be smaller than references")
    threshold = float(checkpoint.get("threshold", float("nan")))
    if not np.isfinite(threshold):
        raise ValueError("Checkpoint threshold must be finite")
    return (
        imputer,
        scaler,
        reference,
        prototype,
        normalized_decision,
        threshold,
        score_statistics,
        regime_memory_state,
        relation_state,
        relation_weight,
        rgcsf_assets,
    )


@torch.no_grad()
def embed_test_windows(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    dataset = WindowDataset(data, window_size, stride, cover_end=True)
    if len(dataset) == 0:
        raise ValueError(f"Test series length {len(data)} is shorter than window_size={window_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: list[torch.Tensor] = []
    starts: list[int] = []
    model.eval()
    for batch in loader:
        embeddings.append(model(batch["x"].to(device)).cpu())
        starts.extend(int(value) for value in batch["start"])
    result = torch.cat(embeddings, dim=0)
    if len(result) != len(starts) or not torch.isfinite(result).all():
        raise RuntimeError("Test-window embedding generation failed")
    return result, starts


@torch.no_grad()
def embed_test_window_views(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    dataset = WindowDataset(data, window_size, stride, cover_end=True)
    if len(dataset) == 0:
        raise ValueError(
            f"Test series length {len(data)} is shorter than "
            f"window_size={window_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    global_embeddings: list[torch.Tensor] = []
    channel_embeddings: list[torch.Tensor] = []
    starts: list[int] = []
    model.eval()
    for batch in loader:
        inputs = batch["x"].to(device)
        global_embeddings.append(model(inputs).cpu())
        channel_embeddings.append(
            model.forward_channel_embeddings(inputs).cpu()
        )
        starts.extend(int(value) for value in batch["start"])
    global_result = torch.cat(global_embeddings, dim=0)
    channel_result = torch.cat(channel_embeddings, dim=0)
    if (
        len(global_result) != len(starts)
        or len(channel_result) != len(starts)
        or not torch.isfinite(global_result).all()
        or not torch.isfinite(channel_result).all()
    ):
        raise RuntimeError("RG-CSF window embedding generation failed")
    return global_result, channel_result, starts


def compute_entity_metrics(
    point_scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    allow_single_class: bool,
) -> dict[str, float] | None:
    label_array = np.asarray(labels, dtype=np.int32).ravel()
    if np.unique(label_array).size != 2:
        if allow_single_class:
            return None
        raise ValueError(
            "Entity metrics require both normal and anomalous labels; "
            "use dataset aggregation for single-class entities"
        )
    return compute_all_metrics(point_scores, label_array, threshold)


def save_score_artifact(
    path: str | Path,
    *,
    point_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    validation_point_scores: np.ndarray | None = None,
    threshold_percentile: float | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload: dict[str, np.ndarray] = {
        "point_scores": np.asarray(point_scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int32),
        "threshold": np.asarray(float(threshold), dtype=np.float64),
    }
    if (validation_point_scores is None) != (threshold_percentile is None):
        raise ValueError(
            "Validation point scores and threshold percentile must be provided together"
        )
    if validation_point_scores is not None:
        validation_scores = np.asarray(
            validation_point_scores,
            dtype=np.float64,
        ).ravel()
        percentile = float(threshold_percentile)
        if (
            validation_scores.size == 0
            or not np.isfinite(validation_scores).all()
            or not np.isfinite(percentile)
            or not 0.0 < percentile < 100.0
        ):
            raise ValueError("Validation score calibration is invalid")
        payload["validation_point_scores"] = validation_scores
        payload["threshold_percentile"] = np.asarray(
            percentile,
            dtype=np.float64,
        )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output)
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.distance_chunk_size < 1:
        raise ValueError("batch-size and distance-chunk-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=device)
    checkpoint_dataset = str(checkpoint.get("dataset", "")).lower()
    if checkpoint_dataset != args.dataset.lower():
        raise ValueError(f"Checkpoint dataset is {checkpoint_dataset!r}, not {args.dataset!r}")
    checkpoint_entity = checkpoint.get("entity")
    if checkpoint_entity is not None and args.entity != checkpoint_entity:
        raise ValueError(f"Checkpoint entity is {checkpoint_entity!r}, not {args.entity!r}")
    (
        imputer,
        scaler,
        reference,
        prototype,
        decision,
        threshold,
        score_statistics,
        regime_memory_state,
        relation_state,
        relation_weight,
        rgcsf_assets,
    ) = _inference_assets(
        checkpoint,
        model.features_dim,
        model.channel_features_dim,
    )
    train_data, test_data, train_labels, test_labels = load_dataset(
        args.dataset, args.data_root, entity=args.entity, expected_channels=model.in_channels
    )
    split = resolve_checkpoint_split(
        checkpoint,
        train_length=len(train_data),
        test_length=len(test_data),
    )
    full_data = np.concatenate([train_data, test_data], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)
    standardized = scaler.transform(imputer.transform(full_data))
    standardized_test_blocks = slice_test_blocks(standardized, split)
    label_test_blocks = slice_test_blocks(full_labels, split)
    model_config = checkpoint["model_config"]
    window_size = int(model_config["window_size"])
    stride = int(decision["eval_stride"])
    scored_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for block_data, block_labels in zip(standardized_test_blocks, label_test_blocks):
        channel_embeddings = None
        if decision["scoring"] == "rgcsf":
            embeddings, channel_embeddings, starts = embed_test_window_views(
                model,
                block_data,
                window_size=window_size,
                stride=stride,
                batch_size=args.batch_size,
                device=device,
            )
        else:
            embeddings, starts = embed_test_windows(
                model, block_data, window_size=window_size, stride=stride,
                batch_size=args.batch_size, device=device,
            )
        if decision["scoring"] == "regime_memory":
            assert regime_memory_state is not None
            assert relation_state is not None
            regime_scores = score_regime_memory(
                embeddings,
                regime_memory_state,
                distance_chunk_size=args.distance_chunk_size,
            )
            relation_scores = score_relation_windows(
                block_data,
                starts,
                relation_state,
            )
            window_scores = fuse_enhanced_scores(
                regime_scores,
                relation_scores,
                relation_weight=relation_weight,
            )
        elif decision["scoring"] == "rgcsf":
            assert regime_memory_state is not None
            assert rgcsf_assets is not None
            assert channel_embeddings is not None
            global_scores = score_regime_memory(
                embeddings,
                regime_memory_state,
                distance_chunk_size=args.distance_chunk_size,
            )
            (
                channel_scores,
                assignments,
                confidence,
            ) = score_channel_regimes(
                channel_embeddings,
                embeddings,
                regime_memory_state,
                rgcsf_assets["channel_regime_state"],
            )
            relation_scores = score_regime_relations(
                block_data,
                starts,
                assignments,
                rgcsf_assets["regime_relation_state"],
            )
            window_scores = score_rgcsf_fusion(
                global_scores,
                channel_scores,
                relation_scores,
                confidence,
                rgcsf_assets["calibration"],
            )
        else:
            window_scores = score_embeddings(
                embeddings, reference, prototype,
                scoring=str(decision["scoring"]),
                k=int(decision["k"] or 1),
                mix_weight=float(decision["hybrid_mix_weight"] or 0.5),
                score_stats=score_statistics,
                distance_chunk_size=args.distance_chunk_size,
            )
        block_scores = aggregate_window_scores(
            window_scores,
            starts,
            series_length=len(block_data),
            window_size=window_size,
        )
        scored_blocks.append((block_scores, block_labels))
    point_scores, test_labels = aggregate_block_scores(scored_blocks)
    scores_output = getattr(args, "scores_output", None)
    if scores_output:
        calibration = checkpoint.get("validation_score_calibration")
        validation_point_scores = None
        threshold_percentile = None
        if calibration is not None:
            if (
                not isinstance(calibration, Mapping)
                or calibration.get("protocol")
                != "adcl_fd_validation_point_scores_v1"
            ):
                raise ValueError(
                    "Checkpoint validation_score_calibration is invalid"
                )
            validation_point_scores = np.asarray(
                calibration.get("point_scores"),
                dtype=np.float64,
            ).ravel()
            threshold_percentile = float(
                calibration.get("threshold_percentile", float("nan"))
            )
            if (
                validation_point_scores.size == 0
                or not np.isfinite(validation_point_scores).all()
                or not np.isfinite(threshold_percentile)
                or not 0.0 < threshold_percentile < 100.0
            ):
                raise ValueError(
                    "Checkpoint validation score calibration does not match threshold"
                )
            expected_threshold = float(
                np.percentile(
                    validation_point_scores,
                    threshold_percentile,
                )
            )
            if not np.isclose(
                expected_threshold,
                threshold,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "Checkpoint validation score calibration does not match threshold"
                )
        save_score_artifact(
            scores_output,
            point_scores=point_scores,
            labels=test_labels,
            threshold=threshold,
            validation_point_scores=validation_point_scores,
            threshold_percentile=threshold_percentile,
        )
    defer_metrics = bool(getattr(args, "defer_metrics", False))
    metrics = (
        None
        if defer_metrics
        else compute_entity_metrics(
            point_scores,
            test_labels,
            threshold=threshold,
            allow_single_class=bool(getattr(args, "allow_single_class", False)),
        )
    )
    if metrics is not None and (tuple(metrics) != FORMAL_METRICS or len(metrics) != 13):
        raise RuntimeError("Evaluation did not return the exact 13-metric contract")
    result = {
        "protocol": (
            "adcl_fd_v3_13_metrics"
            if metrics is not None
            else "adcl_fd_v3_entity_scores"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": args.dataset.lower(),
        "entity": args.entity,
        "scoring": decision["scoring"],
        "k": decision["k"],
        "threshold": threshold,
        "threshold_source": str(
            checkpoint.get("threshold_source", "validation_point_scores")
        ),
        "threshold_percentile": decision["threshold_percentile"],
        "stride": stride,
        "window_size": window_size,
        "split_protocol": split.to_dict(),
        "validation_block_index": split.validation_block_index,
        "test_block_indices": [block.index for block in split.test_blocks],
        "reference_count": len(reference),
        "scores_output": str(Path(scores_output).resolve()) if scores_output else None,
        "metric_status": (
            "complete"
            if metrics is not None
            else "deferred_to_dataset_aggregation"
        ),
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AdCL-FD v3 with exactly 13 metrics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["smd", "msl", "smap", "swat", "wadi", "psm", "kpi", "yahoo", "ucr"])
    parser.add_argument("--data-root", required=True, help="Root directory for the selected dataset")
    parser.add_argument("--entity", default=None, help="Entity/file name when required")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scores-output",
        default=None,
        help="Optional compressed point-score artifact for dataset aggregation",
    )
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Write entity scores without metrics when test labels contain one class",
    )
    parser.add_argument(
        "--defer-metrics",
        action="store_true",
        help="Write entity score artifacts and defer all formal metrics to dataset aggregation",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
