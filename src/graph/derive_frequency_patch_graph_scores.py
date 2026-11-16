"""Derive frequency-patch dynamic sparse graph score artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.evaluation.aggregate_results import load_calibrated_score_artifact
from src.data.loading import load_dataset
from src.data.preprocess import CausalImputer, StandardScaler
from src.evaluation.eval_full import resolve_checkpoint_split, save_score_artifact, slice_test_blocks
from src.graph.frequency_patch_graph import (
    fit_frequency_patch_graph,
    score_frequency_patch_graph,
)
from src.models.loading import load_checkpoint_file
from src.graph.graph_artifact_contract import (
    GRAPH_FIT_SCOPE, GRAPH_PROTOCOL, GRAPH_STATE_PROTOCOL,
    array_sha256, file_sha256, save_graph_state,
)
from src.scoring.scoring import aggregate_block_scores


def derive(args: argparse.Namespace) -> Path:
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Graph score output must have a .npz suffix")
    metadata_path = output_path.with_suffix(".json")
    state_path = output_path.with_suffix(".state")
    reserved_paths = (
        output_path,
        metadata_path,
        state_path,
        output_path.with_suffix(output_path.suffix + ".tmp"),
        metadata_path.with_suffix(metadata_path.suffix + ".tmp"),
    )
    for path in reserved_paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Graph derivation requires a fresh output path; preserve existing artifact: {path}"
            )
    percentile = float(args.threshold_percentile)
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("threshold-percentile must be finite and in (0, 100)")
    input_hashes = {
        "checkpoint": file_sha256(args.checkpoint),
        "base_scores": file_sha256(args.base_scores),
    }
    source_paths = {
        name: Path(__file__).resolve().with_name(name)
        for name in (
            "derive_frequency_patch_graph_scores.py",
            "frequency_patch_graph.py",
            "graph_artifact_contract.py",
        )
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    checkpoint = load_checkpoint_file(args.checkpoint, "cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a mapping")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint lacks model_config")
    train, test, train_labels, test_labels = load_dataset(
        args.dataset,
        args.data_root,
        entity=args.entity,
        expected_channels=int(model_config["in_channels"]),
    )
    data_hashes = {
        name: array_sha256(value) for name, value in (
            ("train", train), ("test", test),
            ("train_labels", train_labels), ("test_labels", test_labels),
        )
    }
    split = resolve_checkpoint_split(
        checkpoint,
        train_length=len(train),
        test_length=len(test),
    )
    full_data = np.concatenate([train, test], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)
    imputer = CausalImputer.from_state_dict(checkpoint["imputer_state"])
    scaler = StandardScaler.from_state_dict(checkpoint["scaler_state"])
    standardized = scaler.transform(imputer.transform(full_data))
    fit_started = time.perf_counter()
    state = fit_frequency_patch_graph(
        standardized[:split.normal_fit_end],
        patch_size=args.patch_size,
        hop_size=args.hop_size,
        frequency_bins=args.frequency_bins,
        neighbor_count=args.neighbor_count,
        ridge_alpha=args.ridge_alpha,
        attention_temperature=args.attention_temperature,
        topk_ratio=args.topk_ratio,
        max_fit_patches=args.max_fit_patches,
        attention_mode=args.attention_mode,
    )
    fit_seconds = time.perf_counter() - fit_started
    validation_data = standardized[split.validation.start:split.validation.end]
    validation_started = time.perf_counter()
    validation_scores = score_frequency_patch_graph(validation_data, state)
    validation_score_seconds = time.perf_counter() - validation_started
    threshold = float(
        np.percentile(validation_scores, percentile)
    )
    test_blocks = slice_test_blocks(standardized, split)
    test_started = time.perf_counter()
    test_scores = [score_frequency_patch_graph(block, state) for block in test_blocks]
    test_score_seconds = time.perf_counter() - test_started
    label_blocks = slice_test_blocks(full_labels, split)
    point_scores, labels = aggregate_block_scores(
        list(zip(test_scores, label_blocks))
    )
    base = load_calibrated_score_artifact(args.base_scores)
    if (
        point_scores.shape != base.point_scores.shape
        or validation_scores.shape != base.validation_point_scores.shape
        or not np.array_equal(labels, base.labels)
    ):
        raise RuntimeError("Frequency graph scores are not aligned with base scores")
    if input_hashes != {
        "checkpoint": file_sha256(args.checkpoint),
        "base_scores": file_sha256(args.base_scores),
    } or source_hashes != {
        name: file_sha256(path) for name, path in source_paths.items()
    }:
        raise RuntimeError("Graph derivation inputs or sources changed during execution")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph_state(state_path, state)
    output = save_score_artifact(
        output_path,
        point_scores=point_scores,
        labels=labels,
        threshold=threshold,
        validation_point_scores=validation_scores,
        threshold_percentile=percentile,
    )
    metadata = {
        "protocol": GRAPH_PROTOCOL,
        "fit_scope": GRAPH_FIT_SCOPE,
        "dataset": args.dataset,
        "entity": args.entity,
        "seed": split.seed,
        "train_end": split.train_end,
        "train_source_length": split.train_source_length,
        "normal_fit_end": split.normal_fit_end,
        "split_protocol": split.to_dict(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_scores": str(Path(args.base_scores).resolve()),
        "input_sha256": input_hashes,
        "score_sha256": file_sha256(output),
        "source_sha256": source_hashes,
        "data_sha256": data_hashes,
        "state_file": state_path.name,
        "state_protocol": GRAPH_STATE_PROTOCOL,
        "state_sha256": file_sha256(state_path),
        "fit_pairing": "consecutive_retained_training_patches",
        "threshold_percentile": percentile,
        "patch_size": args.patch_size,
        "hop_size": args.hop_size,
        "frequency_bins": args.frequency_bins,
        "neighbor_count": args.neighbor_count,
        "ridge_alpha": args.ridge_alpha,
        "attention_temperature": args.attention_temperature,
        "attention_mode": args.attention_mode,
        "topk_ratio": args.topk_ratio,
        "max_fit_patches": args.max_fit_patches,
        "fit_seconds": fit_seconds,
        "validation_score_seconds": validation_score_seconds,
        "test_score_seconds": test_score_seconds,
        "test_points": int(sum(len(block) for block in test_blocks)),
        "threshold": threshold,
        "mean_channel_reliability": float(state.reliability.mean()),
        "active_channel_fraction": float(np.mean(state.reliability > 0.05)),
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive frequency-patch dynamic sparse graph scores"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-scores", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--hop-size", type=int, default=8)
    parser.add_argument("--frequency-bins", type=int, default=8)
    parser.add_argument("--neighbor-count", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--attention-temperature", type=float, default=0.20)
    parser.add_argument(
        "--attention-mode",
        choices=["dynamic", "prior"],
        default="dynamic",
    )
    parser.add_argument("--topk-ratio", type=float, default=0.10)
    parser.add_argument("--max-fit-patches", type=int, default=20_000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    return parser


def main() -> None:
    derive(build_parser().parse_args())


if __name__ == "__main__":
    main()
