"""Validation-reliability-gated fusion of global and FP-LIR point scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_results import (
    CalibratedScoreArtifact,
    aggregate_entity_scores,
    load_calibrated_score_artifact,
)
from scoring import _validation_ecdf
from eval_full import save_score_artifact
from metrics import FORMAL_METRICS, compute_all_metrics


def local_reliability_weight(
    local_artifact: CalibratedScoreArtifact,
    *,
    maximum_weight: float = 0.5,
    shift_scale: float = 20.0,
    exponent: float = 4.0,
) -> float:
    if (
        not np.isfinite(maximum_weight)
        or not 0.0 <= maximum_weight <= 1.0
        or not np.isfinite(shift_scale)
        or shift_scale <= 0
        or not np.isfinite(exponent)
        or exponent <= 0
    ):
        raise ValueError("Invalid reliability-gate configuration")
    shift = float(local_artifact.threshold)
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("Local validation shift must be finite and positive")
    reliability = 1.0 / (1.0 + (shift / shift_scale) ** exponent)
    return float(maximum_weight * reliability)


def fuse_gated_artifact(
    global_artifact: CalibratedScoreArtifact,
    local_artifact: CalibratedScoreArtifact,
    *,
    maximum_local_weight: float = 0.5,
    shift_scale: float = 20.0,
    exponent: float = 4.0,
) -> tuple[CalibratedScoreArtifact, float]:
    if (
        global_artifact.point_scores.shape != local_artifact.point_scores.shape
        or global_artifact.labels.shape != local_artifact.labels.shape
        or global_artifact.validation_point_scores.shape
        != local_artifact.validation_point_scores.shape
        or not np.array_equal(global_artifact.labels, local_artifact.labels)
        or not np.isclose(
            global_artifact.threshold_percentile,
            local_artifact.threshold_percentile,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Global and local calibrated artifacts are not aligned")
    local_weight = local_reliability_weight(
        local_artifact,
        maximum_weight=maximum_local_weight,
        shift_scale=shift_scale,
        exponent=exponent,
    )
    global_test = _validation_ecdf(
        global_artifact.validation_point_scores,
        global_artifact.point_scores,
    )
    local_test = _validation_ecdf(
        local_artifact.validation_point_scores,
        local_artifact.point_scores,
    )
    global_validation = _validation_ecdf(
        global_artifact.validation_point_scores,
        global_artifact.validation_point_scores,
    )
    local_validation = _validation_ecdf(
        local_artifact.validation_point_scores,
        local_artifact.validation_point_scores,
    )
    fused_test = (
        (1.0 - local_weight) * global_test
        + local_weight * local_test
    )
    fused_validation = (
        (1.0 - local_weight) * global_validation
        + local_weight * local_validation
    )
    percentile = float(global_artifact.threshold_percentile)
    threshold = float(np.percentile(fused_validation, percentile))
    if (
        not np.isfinite(fused_test).all()
        or not np.isfinite(fused_validation).all()
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise FloatingPointError("Reliability-gated fusion produced invalid scores")
    return (
        CalibratedScoreArtifact(
            point_scores=fused_test,
            labels=global_artifact.labels.copy(),
            threshold=threshold,
            validation_point_scores=fused_validation,
            threshold_percentile=percentile,
        ),
        local_weight,
    )


def build_gated_ensemble(
    *,
    global_root: str | Path,
    local_root: str | Path,
    output: str | Path,
    maximum_local_weight: float = 0.5,
    shift_scale: float = 20.0,
    exponent: float = 4.0,
    defer_metrics: bool = False,
) -> dict[str, object]:
    global_path = Path(global_root)
    local_path = Path(local_root)
    output_path = Path(output)
    datasets = sorted(path.name for path in local_path.iterdir() if path.is_dir())
    if not datasets:
        raise ValueError("No local score datasets were found")
    reports: list[str] = []
    gate_state: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        global_files = sorted((global_path / dataset).glob("*.npz"))
        local_files = sorted((local_path / dataset).glob("*.npz"))
        if not local_files or [p.name for p in global_files] != [p.name for p in local_files]:
            raise ValueError(f"Global/local entity sets differ for dataset={dataset}")
        fused: list[CalibratedScoreArtifact] = []
        weights: list[float] = []
        score_output = output_path / "scores" / dataset
        score_output.mkdir(parents=True, exist_ok=True)
        for global_file, local_file in zip(global_files, local_files):
            artifact, weight = fuse_gated_artifact(
                load_calibrated_score_artifact(global_file),
                load_calibrated_score_artifact(local_file),
                maximum_local_weight=maximum_local_weight,
                shift_scale=shift_scale,
                exponent=exponent,
            )
            fused.append(artifact)
            weights.append(weight)
            save_score_artifact(
                score_output / global_file.name,
                point_scores=artifact.point_scores,
                labels=artifact.labels,
                threshold=artifact.threshold,
                validation_point_scores=artifact.validation_point_scores,
                threshold_percentile=artifact.threshold_percentile,
            )
        metrics = None
        if not defer_metrics:
            scores, labels = aggregate_entity_scores(
                [
                    (item.point_scores, item.labels, item.threshold)
                    for item in fused
                ]
            )
            metrics = compute_all_metrics(scores, labels, threshold=1.0)
            if tuple(metrics) != FORMAL_METRICS:
                raise RuntimeError("Gated fusion violated the 13-metric contract")
        report = {
            "protocol": (
                "fplir_validation_reliability_gated_ensemble_v1"
                if metrics is not None
                else "fplir_validation_reliability_gated_scores_v1"
            ),
            "dataset": dataset,
            "entity_count": len(fused),
            "maximum_local_weight": float(maximum_local_weight),
            "shift_scale": float(shift_scale),
            "exponent": float(exponent),
            "mean_local_weight": float(np.mean(weights)),
            "min_local_weight": float(np.min(weights)),
            "max_local_weight": float(np.max(weights)),
            "metric_status": (
                "complete" if metrics is not None else "deferred_to_fdpg"
            ),
            "metrics": metrics,
        }
        report_path = output_path / "datasets" / f"{dataset}_metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        reports.append(str(report_path.resolve()))
        gate_state[dataset] = {
            path.name: float(weight)
            for path, weight in zip(global_files, weights)
        }
    manifest = {
        "protocol": "fplir_validation_reliability_gated_ensemble_v1",
        "global_root": str(global_path.resolve()),
        "local_root": str(local_path.resolve()),
        "maximum_local_weight": float(maximum_local_weight),
        "shift_scale": float(shift_scale),
        "exponent": float(exponent),
        "metric_status": (
            "deferred_to_fdpg" if defer_metrics else "complete"
        ),
        "gate_state": gate_state,
        "dataset_reports": reports,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse global and FP-LIR scores with validation reliability"
    )
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-local-weight", type=float, default=0.5)
    parser.add_argument("--shift-scale", type=float, default=20.0)
    parser.add_argument("--exponent", type=float, default=4.0)
    parser.add_argument(
        "--defer-metrics",
        action="store_true",
        help="Write calibrated base scores and defer formal metrics to FDPG",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_gated_ensemble(
        global_root=args.global_root,
        local_root=args.local_root,
        output=args.output,
        maximum_local_weight=args.maximum_local_weight,
        shift_scale=args.shift_scale,
        exponent=args.exponent,
        defer_metrics=args.defer_metrics,
    )


if __name__ == "__main__":
    main()
