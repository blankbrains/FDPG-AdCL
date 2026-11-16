"""Derive immutable RG-CSF scoring assets from a trained channel-aware encoder."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.loading import load_feature_pair
from src.data.preprocess import CausalImputer, StandardScaler
from src.scoring.enhanced_scoring import fit_regime_memory, score_regime_memory
from src.evaluation.eval_full import embed_test_window_views, resolve_checkpoint_split
from src.models.loading import load_model_checkpoint
from src.scoring.rgcsf_scoring import (
    fit_channel_regime_state,
    fit_regime_relation_state,
    fit_rgcsf_calibration,
    score_channel_regimes,
    score_regime_relations,
)


def derive_rgcsf_checkpoint(
    source_checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    train_data: np.ndarray,
    test_data: np.ndarray,
    *,
    cluster_count: int,
    min_cluster_size: int,
    regime_mix_weight: float,
    kmeans_n_init: int,
    top_fraction: float,
    relation_percentile: float,
    global_weight: float,
    relation_weight: float,
    channel_gate_quantile: float,
    relation_gate_quantile: float,
    gate_temperature: float,
    batch_size: int,
    distance_chunk_size: int,
    device: torch.device,
    reference_k: int | None = None,
) -> dict[str, Any]:
    if source_checkpoint.get("checkpoint_type") != "adcl_fd_v3_inference":
        raise ValueError("Source must be an adcl_fd_v3_inference checkpoint")
    if getattr(model, "channel_features_dim", 0) < 2:
        raise ValueError("Source model must provide channel embeddings")
    if batch_size < 1 or distance_chunk_size < 1:
        raise ValueError("Batch and distance chunk sizes must be positive")
    train = np.asarray(train_data, dtype=np.float32)
    test = np.asarray(test_data, dtype=np.float32)
    if (
        train.ndim != 2
        or test.ndim != 2
        or train.shape[1] != test.shape[1]
        or train.shape[1] < 2
    ):
        raise ValueError("RG-CSF requires aligned multivariate feature arrays")
    model_config = source_checkpoint.get("model_config")
    decision = source_checkpoint.get("decision_config")
    if not isinstance(model_config, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("Source checkpoint lacks model or decision configuration")
    window_size = int(model_config["window_size"])
    eval_stride = int(decision["eval_stride"])
    threshold_percentile = float(decision["threshold_percentile"])
    training_config = source_checkpoint.get("training_config")
    configured_k = (
        training_config.get("k")
        if isinstance(training_config, Mapping)
        else None
    )
    raw_k = (
        reference_k
        if reference_k is not None
        else decision.get("k", configured_k)
    )
    if raw_k is None:
        raw_k = configured_k
    if raw_k is None or int(raw_k) < 1:
        raise ValueError(
            "RG-CSF requires a positive reference neighbor count"
        )
    k = int(raw_k)
    split = resolve_checkpoint_split(
        source_checkpoint,
        train_length=len(train),
        test_length=len(test),
    )
    imputer = CausalImputer.from_state_dict(source_checkpoint["imputer_state"])
    scaler = StandardScaler.from_state_dict(source_checkpoint["scaler_state"])
    normal_prefix = train[: min(len(train), split.train_end)]
    standardized_training = scaler.transform(
        imputer.transform(normal_prefix)
    )
    standardized_full = scaler.transform(
        imputer.transform(np.concatenate([train, test], axis=0))
    )
    validation_data = standardized_full[
        split.validation.start:split.validation.end
    ]

    reference = torch.as_tensor(
        source_checkpoint.get("reference_embeddings"),
        dtype=torch.float32,
    )
    channel_reference = torch.as_tensor(
        source_checkpoint.get("channel_reference_embeddings"),
        dtype=torch.float32,
    )
    reference_starts = source_checkpoint.get("channel_reference_starts")
    if (
        reference.ndim != 2
        or channel_reference.ndim != 3
        or len(reference) != len(channel_reference)
        or not isinstance(reference_starts, list)
        or len(reference_starts) != len(reference)
    ):
        raise ValueError("Source checkpoint channel references are incomplete")
    regime_state = fit_regime_memory(
        reference,
        k=k,
        cluster_count=int(cluster_count),
        min_cluster_size=int(min_cluster_size),
        seed=int(split.seed),
        mix_weight=float(regime_mix_weight),
        kmeans_n_init=int(kmeans_n_init),
    )
    channel_state = fit_channel_regime_state(
        channel_reference,
        regime_state,
        top_fraction=float(top_fraction),
    )
    relation_state = fit_regime_relation_state(
        standardized_training,
        reference_starts,
        regime_state,
        window_size=window_size,
        aggregate_percentile=float(relation_percentile),
    )

    (
        validation_global_embeddings,
        validation_channel_embeddings,
        validation_starts,
    ) = embed_test_window_views(
        model,
        validation_data,
        window_size=window_size,
        stride=eval_stride,
        batch_size=batch_size,
        device=device,
    )
    validation_global_scores = score_regime_memory(
        validation_global_embeddings,
        regime_state,
        distance_chunk_size=distance_chunk_size,
    )
    (
        validation_channel_scores,
        validation_assignments,
        validation_confidence,
    ) = score_channel_regimes(
        validation_channel_embeddings,
        validation_global_embeddings,
        regime_state,
        channel_state,
    )
    validation_relation_scores = score_regime_relations(
        validation_data,
        validation_starts,
        validation_assignments,
        relation_state,
    )
    calibration = fit_rgcsf_calibration(
        validation_global_scores,
        validation_channel_scores,
        validation_relation_scores,
        validation_confidence,
        starts=validation_starts,
        series_length=len(validation_data),
        window_size=window_size,
        global_weight=float(global_weight),
        relation_weight=float(relation_weight),
        channel_gate_quantile=float(channel_gate_quantile),
        relation_gate_quantile=float(relation_gate_quantile),
        gate_temperature=float(gate_temperature),
        threshold_percentile=threshold_percentile,
    )

    derived = copy.deepcopy(dict(source_checkpoint))
    derived.update(
        {
            "regime_memory_state": regime_state,
            "channel_regime_state": channel_state,
            "regime_relation_state": relation_state,
            "rgcsf_calibration": calibration,
            "decision_config": {
                "scoring": "rgcsf",
                "k": k,
                "eval_stride": eval_stride,
                "threshold_percentile": threshold_percentile,
                "hybrid_mix_weight": None,
            },
            "threshold": float(calibration["threshold"]),
            "threshold_source": "validation_point_scores",
            "validation_point_count": len(
                np.asarray(calibration["validation_point_scores"])
            ),
            "validation_score_calibration": {
                "protocol": "adcl_fd_validation_point_scores_v1",
                "threshold_percentile": threshold_percentile,
                "point_scores": np.asarray(
                    calibration["validation_point_scores"],
                    dtype=np.float64,
                ),
            },
            "rgcsf_source": {
                "cluster_count": int(cluster_count),
                "min_cluster_size": int(min_cluster_size),
                "reference_k": int(k),
                "regime_mix_weight": float(regime_mix_weight),
                "kmeans_n_init": int(kmeans_n_init),
                "top_fraction": float(top_fraction),
                "relation_percentile": float(relation_percentile),
                "global_weight": float(global_weight),
                "relation_weight": float(relation_weight),
                "channel_gate_quantile": float(channel_gate_quantile),
                "relation_gate_quantile": float(relation_gate_quantile),
                "gate_temperature": float(gate_temperature),
            },
            "rgcsf_version": "regime_gated_channel_sparse_frequency_v1",
        }
    )
    return derived


def _atomic_torch_save(checkpoint: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    temporary.replace(output)


def derive_from_args(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint = load_model_checkpoint(
        args.source_checkpoint,
        device=device,
    )
    dataset = str(args.dataset).lower()
    if str(checkpoint.get("dataset", "")).lower() != dataset:
        raise ValueError("Source checkpoint dataset does not match --dataset")
    if checkpoint.get("entity") != args.entity:
        raise ValueError("Source checkpoint entity does not match --entity")
    train_data, test_data = load_feature_pair(
        dataset,
        args.data_root,
        entity=args.entity,
        expected_channels=model.in_channels,
    )
    derived = derive_rgcsf_checkpoint(
        checkpoint,
        model,
        train_data,
        test_data,
        cluster_count=args.cluster_count,
        min_cluster_size=args.min_cluster_size,
        regime_mix_weight=args.regime_mix_weight,
        kmeans_n_init=args.kmeans_n_init,
        top_fraction=args.top_fraction,
        relation_percentile=args.relation_percentile,
        global_weight=args.global_weight,
        relation_weight=args.relation_weight,
        channel_gate_quantile=args.channel_gate_quantile,
        relation_gate_quantile=args.relation_gate_quantile,
        gate_temperature=args.gate_temperature,
        batch_size=args.batch_size,
        distance_chunk_size=args.distance_chunk_size,
        device=device,
        reference_k=args.reference_k,
    )
    derived["rgcsf_source"]["checkpoint"] = str(
        Path(args.source_checkpoint).resolve()
    )
    output = Path(args.output)
    _atomic_torch_save(derived, output)
    print(f"saved={output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a strict RG-CSF multivariate scoring checkpoint"
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["smd", "msl", "smap", "swat", "wadi", "psm"],
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=16)
    parser.add_argument("--regime-mix-weight", type=float, default=0.50)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--relation-percentile", type=float, default=90.0)
    parser.add_argument("--global-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.20)
    parser.add_argument("--channel-gate-quantile", type=float, default=0.80)
    parser.add_argument("--relation-gate-quantile", type=float, default=0.95)
    parser.add_argument("--gate-temperature", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--distance-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--reference-k",
        type=int,
        default=None,
        help=(
            "Reference neighbor count when the source scoring mode does not "
            "persist decision_config.k"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    derive_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
