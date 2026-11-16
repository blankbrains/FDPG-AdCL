"""Build exact reports for the frozen threshold/channel diagnostic variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_results import (
    aggregate_entity_scores,
    load_calibrated_score_artifact,
)
from scoring import _validation_ecdf
from eval_full import save_score_artifact
from metrics import FORMAL_METRICS, compute_all_metrics


def fuse_validation_aligned_scores(
    base_test: np.ndarray,
    channel_test: np.ndarray,
    base_validation: np.ndarray,
    channel_validation: np.ndarray,
    *,
    mode: str,
    channel_weight: float,
    threshold_percentile: float,
    boundary_strength: float,
    boundary_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(value, dtype=np.float64).ravel()
        for value in (
            base_test,
            channel_test,
            base_validation,
            channel_validation,
        )
    ]
    if (
        arrays[0].shape != arrays[1].shape
        or arrays[2].shape != arrays[3].shape
        or any(value.size == 0 or not np.isfinite(value).all() for value in arrays)
    ):
        raise ValueError("Fusion inputs must be finite and aligned")
    if mode == "linear":
        fused_test = (
            (1.0 - channel_weight) * arrays[0]
            + channel_weight * arrays[1]
        )
        fused_validation = (
            (1.0 - channel_weight) * arrays[2]
            + channel_weight * arrays[3]
        )
    elif mode == "boundary":
        if (
            not np.isfinite(boundary_strength)
            or not 0.0 <= boundary_strength <= 1.0
            or not np.isfinite(boundary_sigma)
            or boundary_sigma <= 0
        ):
            raise ValueError("Invalid boundary-gate configuration")
        boundary = float(np.percentile(arrays[2], threshold_percentile))
        test_gate = np.exp(
            -0.5 * ((arrays[0] - boundary) / boundary_sigma) ** 2
        )
        validation_gate = np.exp(
            -0.5 * ((arrays[2] - boundary) / boundary_sigma) ** 2
        )
        fused_test = arrays[0] + boundary_strength * test_gate * (
            arrays[1] - arrays[0]
        )
        fused_validation = arrays[2] + boundary_strength * validation_gate * (
            arrays[3] - arrays[2]
        )
    else:
        raise ValueError("mode must be linear or boundary")
    if not np.isfinite(fused_test).all() or not np.isfinite(fused_validation).all():
        raise FloatingPointError("Threshold/channel fusion produced NaN/Inf")
    return fused_test, fused_validation


def evaluate(
    *,
    base_root: str | Path,
    channel_root: str | Path,
    output: str | Path,
    channel_weight: float,
    threshold_percentile: float,
    mode: str = "linear",
    boundary_strength: float = 0.5,
    boundary_sigma: float = 0.01,
) -> dict[str, object]:
    if (
        not np.isfinite(channel_weight)
        or not 0.0 <= channel_weight <= 1.0
        or not np.isfinite(threshold_percentile)
        or not 0.0 < threshold_percentile < 100.0
    ):
        raise ValueError("Invalid threshold/channel diagnostic configuration")
    base_path = Path(base_root)
    channel_path = Path(channel_root)
    output_path = Path(output)
    datasets = sorted(path.name for path in channel_path.iterdir() if path.is_dir())
    reports: list[str] = []
    for dataset in datasets:
        base_files = sorted((base_path / dataset).glob("*.npz"))
        channel_files = sorted((channel_path / dataset).glob("*.npz"))
        if not base_files or [p.name for p in base_files] != [p.name for p in channel_files]:
            raise ValueError(f"Base/channel entity sets differ for dataset={dataset}")
        fused_series: list[tuple[np.ndarray, np.ndarray, float]] = []
        score_output = output_path / "scores" / dataset
        score_output.mkdir(parents=True, exist_ok=True)
        for base_file, channel_file in zip(base_files, channel_files):
            base = load_calibrated_score_artifact(base_file)
            channel = load_calibrated_score_artifact(channel_file)
            if (
                base.point_scores.shape != channel.point_scores.shape
                or base.validation_point_scores.shape
                != channel.validation_point_scores.shape
                or not np.array_equal(base.labels, channel.labels)
            ):
                raise ValueError("Base/channel score assets are not aligned")
            base_test = _validation_ecdf(
                base.validation_point_scores,
                base.point_scores,
            )
            channel_test = _validation_ecdf(
                channel.validation_point_scores,
                channel.point_scores,
            )
            base_validation = _validation_ecdf(
                base.validation_point_scores,
                base.validation_point_scores,
            )
            channel_validation = _validation_ecdf(
                channel.validation_point_scores,
                channel.validation_point_scores,
            )
            fused_test, fused_validation = fuse_validation_aligned_scores(
                base_test,
                channel_test,
                base_validation,
                channel_validation,
                mode=mode,
                channel_weight=channel_weight,
                threshold_percentile=threshold_percentile,
                boundary_strength=boundary_strength,
                boundary_sigma=boundary_sigma,
            )
            threshold = float(
                np.percentile(fused_validation, threshold_percentile)
            )
            save_score_artifact(
                score_output / base_file.name,
                point_scores=fused_test,
                labels=base.labels,
                threshold=threshold,
                validation_point_scores=fused_validation,
                threshold_percentile=threshold_percentile,
            )
            fused_series.append((fused_test, base.labels, threshold))
        scores, labels = aggregate_entity_scores(fused_series)
        metrics = compute_all_metrics(scores, labels, threshold=1.0)
        if tuple(metrics) != FORMAL_METRICS:
            raise RuntimeError("Diagnostic report violated the 13-metric contract")
        report = {
            "protocol": "threshold_sparse_channel_diagnostic_v1",
            "dataset": dataset,
            "entity_count": len(fused_series),
            "channel_weight": float(channel_weight),
            "threshold_percentile": float(threshold_percentile),
            "fusion_mode": mode,
            "boundary_strength": (
                float(boundary_strength) if mode == "boundary" else None
            ),
            "boundary_sigma": (
                float(boundary_sigma) if mode == "boundary" else None
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
    manifest = {
        "protocol": "threshold_sparse_channel_diagnostic_v1",
        "channel_weight": float(channel_weight),
        "threshold_percentile": float(threshold_percentile),
        "fusion_mode": mode,
        "boundary_strength": (
            float(boundary_strength) if mode == "boundary" else None
        ),
        "boundary_sigma": float(boundary_sigma) if mode == "boundary" else None,
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
        description="Evaluate the frozen threshold/channel diagnostic"
    )
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--channel-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channel-weight", type=float, default=0.10)
    parser.add_argument("--threshold-percentile", type=float, default=97.5)
    parser.add_argument(
        "--mode",
        choices=["linear", "boundary"],
        default="linear",
    )
    parser.add_argument("--boundary-strength", type=float, default=0.5)
    parser.add_argument("--boundary-sigma", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        base_root=args.base_root,
        channel_root=args.channel_root,
        output=args.output,
        channel_weight=args.channel_weight,
        threshold_percentile=args.threshold_percentile,
        mode=args.mode,
        boundary_strength=args.boundary_strength,
        boundary_sigma=args.boundary_sigma,
    )


if __name__ == "__main__":
    main()
