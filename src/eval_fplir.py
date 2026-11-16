"""Evaluate FP-LIR-AdCL with the fixed 13-metric protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import WindowDataset
from data.preprocess import CausalImputer, StandardScaler
from eval_full import (
    aggregate_block_scores,
    compute_entity_metrics,
    load_dataset,
    resolve_checkpoint_split,
    save_score_artifact,
    slice_test_blocks,
)
from fplir_scoring import aggregate_local_scores
from metrics import FORMAL_METRICS
from models.fplir import build_fplir_from_config
from models.loading import load_checkpoint_file
from train_fplir import score_dataset


def load_fplir_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    checkpoint = load_checkpoint_file(path, device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("FP-LIR checkpoint root must be a mapping")
    if checkpoint.get("checkpoint_type") != "fplir_v1_inference":
        raise ValueError("Checkpoint is not an FP-LIR v1 inference checkpoint")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("FP-LIR checkpoint lacks model_config")
    model = build_fplir_from_config(model_config).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError("FP-LIR checkpoint parameters are incompatible") from exc
    model.eval()
    return model, checkpoint


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint = load_fplir_checkpoint(args.checkpoint, device=device)
    if str(checkpoint.get("dataset", "")).lower() != args.dataset.lower():
        raise ValueError("Checkpoint dataset does not match requested dataset")
    if checkpoint.get("entity") != args.entity:
        raise ValueError("Checkpoint entity does not match requested entity")

    imputer = CausalImputer.from_state_dict(checkpoint.get("imputer_state", {}))
    scaler = StandardScaler.from_state_dict(checkpoint.get("scaler_state", {}))
    prototype = torch.as_tensor(checkpoint.get("prototype"), dtype=torch.float32)
    if prototype.shape != (model.token_dim,) or not torch.isfinite(prototype).all():
        raise ValueError("FP-LIR checkpoint prototype is invalid")
    component_statistics = checkpoint.get("component_statistics")
    decision = checkpoint.get("decision_config")
    calibration = checkpoint.get("validation_score_calibration")
    threshold = float(checkpoint.get("threshold", float("nan")))
    if (
        not isinstance(component_statistics, dict)
        or not isinstance(decision, Mapping)
        or not isinstance(calibration, Mapping)
        or calibration.get("protocol") != "fplir_validation_point_scores_v1"
        or checkpoint.get("threshold_source") != "validation_point_scores"
        or not np.isfinite(threshold)
    ):
        raise ValueError("FP-LIR checkpoint inference assets are incomplete")
    validation_scores = np.asarray(
        calibration.get("point_scores"),
        dtype=np.float64,
    ).ravel()
    percentile = float(calibration.get("threshold_percentile", float("nan")))
    if (
        validation_scores.size == 0
        or not np.isfinite(validation_scores).all()
        or not 0.0 < percentile < 100.0
        or not np.isclose(
            threshold,
            np.percentile(validation_scores, percentile),
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise ValueError("FP-LIR validation calibration does not match threshold")

    train_data, test_data, train_labels, test_labels = load_dataset(
        args.dataset,
        args.data_root,
        entity=args.entity,
        expected_channels=model.in_channels,
    )
    split = resolve_checkpoint_split(
        checkpoint,
        train_length=len(train_data),
        test_length=len(test_data),
    )
    full_data = np.concatenate([train_data, test_data], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)
    standardized = scaler.transform(imputer.transform(full_data))
    data_blocks = slice_test_blocks(standardized, split)
    label_blocks = slice_test_blocks(full_labels, split)
    window_size = int(checkpoint["model_config"]["window_size"])
    eval_stride = int(decision["eval_stride"])
    weights = [float(value) for value in decision["score_weights"]]
    topk_ratio = float(decision["channel_topk_ratio"])
    scored_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for block_data, block_labels in zip(data_blocks, label_blocks):
        dataset = WindowDataset(
            block_data,
            window_size,
            eval_stride,
            cover_end=True,
        )
        local_scores, starts = score_dataset(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
            prototype=prototype,
            component_statistics=component_statistics,
            score_weights=weights,
            channel_topk_ratio=topk_ratio,
        )
        point_scores = aggregate_local_scores(
            local_scores,
            starts,
            series_length=len(block_data),
            window_size=window_size,
        )
        scored_blocks.append((point_scores, block_labels))
    point_scores, labels = aggregate_block_scores(scored_blocks)
    if args.scores_output:
        save_score_artifact(
            args.scores_output,
            point_scores=point_scores,
            labels=labels,
            threshold=threshold,
            validation_point_scores=validation_scores,
            threshold_percentile=percentile,
        )
    metrics = (
        None
        if args.defer_metrics
        else compute_entity_metrics(
            point_scores,
            labels,
            threshold=threshold,
            allow_single_class=args.allow_single_class,
        )
    )
    if metrics is not None and tuple(metrics) != FORMAL_METRICS:
        raise RuntimeError("FP-LIR evaluation violated the 13-metric contract")
    result: dict[str, Any] = {
        "protocol": (
            "fplir_v1_13_metrics"
            if metrics is not None
            else "fplir_v1_entity_scores"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": args.dataset.lower(),
        "entity": args.entity,
        "scoring": "point_local_repair_idempotence_prototype",
        "threshold": threshold,
        "threshold_source": "validation_point_scores",
        "metrics": metrics,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FP-LIR-AdCL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default=None)
    parser.add_argument("--scores-output", default=None)
    parser.add_argument("--allow-single-class", action="store_true")
    parser.add_argument("--defer-metrics", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
