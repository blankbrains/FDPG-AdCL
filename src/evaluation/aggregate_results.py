"""Aggregate entity score artifacts into one formal 13-metric dataset report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.metrics import FORMAL_METRICS, compute_all_metrics


@dataclass(frozen=True)
class CalibratedScoreArtifact:
    point_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    validation_point_scores: np.ndarray
    threshold_percentile: float


def aggregate_entity_scores(
    series: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        raise ValueError("At least one entity score series is required")
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for index, (scores, labels, threshold) in enumerate(series):
        score_array = np.asarray(scores, dtype=np.float64).ravel()
        label_array = np.asarray(labels, dtype=np.int32).ravel()
        if score_array.size == 0 or score_array.shape != label_array.shape:
            raise ValueError("Entity scores and labels must be non-empty and aligned")
        if not np.isfinite(score_array).all():
            raise FloatingPointError("Entity scores contain NaN/Inf")
        if not np.isin(label_array, [0, 1]).all():
            raise ValueError("Entity labels must be binary")
        threshold_value = float(threshold)
        if not np.isfinite(threshold_value) or threshold_value <= 0:
            raise ValueError("Entity threshold must be finite and positive")
        if index:
            score_parts.append(np.array([0.0], dtype=np.float64))
            label_parts.append(np.array([0], dtype=np.int32))
        score_parts.append(score_array / threshold_value)
        label_parts.append(label_array)
    aggregated_scores = np.concatenate(score_parts)
    aggregated_labels = np.concatenate(label_parts)
    if np.unique(aggregated_labels).size != 2:
        raise ValueError("Dataset aggregation requires at least one anomalous point")
    return aggregated_scores, aggregated_labels


def load_score_artifact(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Score artifact not found: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {"point_scores", "labels", "threshold"}
        missing = sorted(required.difference(artifact.files))
        if missing:
            raise ValueError(
                f"Incomplete score artifact {artifact_path}; missing: {', '.join(missing)}"
            )
        scores = artifact["point_scores"]
        labels = artifact["labels"]
        threshold = float(np.asarray(artifact["threshold"]).item())
    return scores, labels, threshold


def load_calibrated_score_artifact(
    path: str | Path,
) -> CalibratedScoreArtifact:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Score artifact not found: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {
            "point_scores",
            "labels",
            "threshold",
            "validation_point_scores",
            "threshold_percentile",
        }
        missing = sorted(required.difference(artifact.files))
        if missing:
            raise ValueError(
                f"Score artifact {artifact_path} lacks validation calibration; "
                f"missing: {', '.join(missing)}"
            )
        point_scores = np.asarray(
            artifact["point_scores"],
            dtype=np.float64,
        ).ravel()
        labels = np.asarray(artifact["labels"], dtype=np.int32).ravel()
        threshold = float(np.asarray(artifact["threshold"]).item())
        validation_point_scores = np.asarray(
            artifact["validation_point_scores"],
            dtype=np.float64,
        ).ravel()
        threshold_percentile = float(
            np.asarray(artifact["threshold_percentile"]).item()
        )
    if (
        point_scores.size == 0
        or point_scores.shape != labels.shape
        or validation_point_scores.size == 0
        or not np.isfinite(point_scores).all()
        or not np.isfinite(validation_point_scores).all()
        or not np.isfinite(threshold)
        or threshold <= 0
        or not np.isfinite(threshold_percentile)
        or not 0.0 < threshold_percentile < 100.0
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError(
            f"Score artifact {artifact_path} contains invalid calibration data"
        )
    return CalibratedScoreArtifact(
        point_scores=point_scores,
        labels=labels,
        threshold=threshold,
        validation_point_scores=validation_point_scores,
        threshold_percentile=threshold_percentile,
    )


def aggregate_files(
    paths: list[str | Path],
    *,
    dataset: str,
    output: str | Path,
) -> dict[str, object]:
    ordered_paths = sorted(Path(path) for path in paths)
    scores, labels = aggregate_entity_scores(
        [load_score_artifact(path) for path in ordered_paths]
    )
    metrics = compute_all_metrics(scores, labels, threshold=1.0)
    if tuple(metrics) != FORMAL_METRICS:
        raise RuntimeError("Dataset aggregation violated the 13-metric contract")
    result: dict[str, object] = {
        "protocol": "adcl_fd_v3_dataset_13_metrics",
        "dataset": dataset.strip().lower(),
        "entity_count": len(ordered_paths),
        "score_normalization": "point_score_divided_by_entity_threshold",
        "entity_separator": "one_normal_point",
        "threshold": 1.0,
        "metrics": metrics,
        "score_artifacts": [str(path.resolve()) for path in ordered_paths],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate AdCL-FD v3 entity scores into 13 dataset metrics"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate_files(args.input, dataset=args.dataset, output=args.output)


if __name__ == "__main__":
    main()
