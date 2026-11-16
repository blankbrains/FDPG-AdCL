"""Run the complete local FDPG-AdCL implementation in an isolated output directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ("msl", "smap", "smd", "swat", "psm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for checkpoints, scores, reports and logs")
    parser.add_argument("--entity", help="Optional file in train/ for MSL, SMAP or SMD")
    parser.add_argument("--config", type=Path, help="Dataset YAML, defaults to configs/<dataset>.yaml")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and print commands without creating outputs or training")
    return parser


def resolve_entities(dataset: str, data_root: Path, entity: str | None) -> list[str | None]:
    if not data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_root}")
    if dataset in ("psm", "swat"):
        if entity is not None:
            raise ValueError("Use --entity only for MSL, SMAP and SMD")
        return [None]
    suffixes = (".txt", ".csv") if dataset == "smd" else (".npy",)
    folder = data_root / "train"
    if not folder.is_dir():
        raise ValueError(f"Missing training folder: {folder}")
    if entity is not None:
        if Path(entity).name != entity or "/" in entity or "\\" in entity:
            raise ValueError("--entity must be a filename, without directory components")
        candidates = [folder / entity]
        if Path(entity).suffix == "":
            candidates += [folder / (entity + suffix) for suffix in suffixes]
        if not any(path.is_file() and path.suffix in suffixes for path in candidates):
            raise ValueError(f"Entity file is missing or has an unsupported extension: {entity}")
        return [entity]
    entities = sorted(path.name for path in folder.iterdir()
                      if path.is_file() and path.suffix in suffixes)
    if not entities:
        raise ValueError(f"No supported entity files in {folder}")
    return entities


def build_stages(args: argparse.Namespace, entities: list[str | None]) -> list[dict]:
    """Compose stage CLIs using the manuscript dataset-specific parameter table."""
    from src.training.adcl_pretext import load_config

    resolved = load_config(args.config, variant="FPLIR")
    parameters = resolved["paper_pipeline"]
    stages = []
    output = args.output
    global_root = output / "global" / "scores"
    local_root = output / "local" / "scores"
    base_root = output / "base" / "scores"
    graph_root = output / "graph" / "scores"

    def add(name: str, script: str, arguments: list) -> None:
        stages.append({"name": name, "command": [sys.executable, "-B", "-X", "utf8",
                                               "-m", script, *map(str, arguments)]})

    for index, entity in enumerate(entities):
        safe_name = entity or "default"
        entity_args = ["--entity", entity] if entity is not None else []
        train_args = ["--config", args.config, "--device", args.device,
                      "--seed", args.seed, "--override", "data_root=" + json.dumps(str(args.data_root))]
        data_args = ["--dataset", args.dataset, "--data-root", args.data_root, *entity_args]
        encoder_dir = output / "checkpoints" / "encoder" / safe_name
        local_dir = output / "checkpoints" / "local" / safe_name
        rgcsf_path = output / "checkpoints" / "rgcsf" / safe_name / "model.pt"
        add(f"encoder_{index}", "src.training.adcl_pretext",
            [*train_args, "--variant", "RGCSF", *entity_args, "--output-dir", encoder_dir])
        add(f"rgcsf_{index}", "src.evaluation.derive_rgcsf_checkpoint",
            ["--source-checkpoint", encoder_dir / "best_model.pt", *data_args,
             "--device", args.device, "--output", rgcsf_path])
        add(f"global_scores_{index}", "src.evaluation.eval_full",
            ["--checkpoint", rgcsf_path, *data_args, "--device", args.device,
             "--output", output / "global" / "entities" / f"{safe_name}.json",
             "--scores-output", global_root / args.dataset / f"{safe_name}.npz", "--defer-metrics"])
        add(f"local_train_{index}", "src.training.train_fplir",
            [*train_args, "--variant", "FPLIR", *entity_args, "--output-dir", local_dir])
        add(f"local_scores_{index}", "src.evaluation.eval_fplir",
            ["--checkpoint", local_dir / "best_model.pt", *data_args, "--device", args.device,
             "--output", output / "local" / "entities" / f"{safe_name}.json",
             "--scores-output", local_root / args.dataset / f"{safe_name}.npz", "--defer-metrics"])

    add("base_fusion", "src.evaluation.gated_fplir_ensemble",
        ["--global-root", global_root, "--local-root", local_root,
         "--output", output / "base", "--defer-metrics",
         "--maximum-local-weight", parameters["maximum_local_weight"],
         "--shift-scale", parameters["shift_scale"], "--exponent", parameters["exponent"]])
    for index, entity in enumerate(entities):
        safe_name = entity or "default"
        entity_args = ["--entity", entity] if entity is not None else []
        add(f"graph_{index}", "src.graph.derive_frequency_patch_graph_scores",
            ["--checkpoint", output / "checkpoints" / "local" / safe_name / "best_model.pt",
             "--dataset", args.dataset, "--data-root", args.data_root, *entity_args,
             "--base-scores", base_root / args.dataset / f"{safe_name}.npz",
             "--output", graph_root / args.dataset / f"{safe_name}.npz",
             "--patch-size", parameters["patch_size"], "--hop-size", parameters["hop_size"],
             "--frequency-bins", parameters["frequency_bins"],
             "--neighbor-count", parameters["neighbor_count"]])
    add("boundary_fusion", "src.evaluation.evaluate_threshold_channel_diagnostic",
        ["--base-root", base_root, "--channel-root", graph_root,
         "--output", output / "final", "--mode", "boundary",
         "--boundary-strength", parameters["boundary_strength"],
         "--boundary-sigma", parameters["boundary_sigma"],
         "--threshold-percentile", parameters["threshold_percentile"]])
    return stages


def validate_data(args: argparse.Namespace, entities: list[str | None], channels: int,
                  window_size: int) -> None:
    from src.data.loading import load_dataset
    from src.data.splitting import split_full_series

    for entity in entities:
        train, test, _, _ = load_dataset(args.dataset, args.data_root, entity=entity,
                                       expected_channels=channels)
        split = split_full_series(train_length=len(train), test_length=len(test), seed=args.seed)
        lengths = [split.normal_fit_end, *[block.length for block in split.all_post_train_blocks]]
        if min(lengths) < window_size:
            raise ValueError(f"Entity {entity!r} is too short for window_size={window_size}: {lengths}")


def execute_stages(stages: list[dict], output: Path, metadata: dict) -> None:
    """Reserve one fresh run directory and retain logs and state on failure."""
    output.mkdir(parents=True, exist_ok=False)
    log_root = output / "logs"
    log_root.mkdir()
    state = {**metadata, "status": "running", "completed": [], "current": None, "stages": stages}
    state_path = output / "run_state.json"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"

    def save() -> None:
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    try:
        save()
        for stage in stages:
            state["current"] = stage["name"]
            save()
            log_path = log_root / (stage["name"] + ".log")
            print(f"Running {stage['name']} | log {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(stage["command"], cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state["completed"].append(stage["name"])
            save()
        state["status"] = "complete"
        state["current"] = None
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = str(exc)
        raise
    finally:
        save()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from src.training.adcl_pretext import load_config
    import torch

    args.data_root = args.data_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.config = (args.config or ROOT / "configs" / f"{args.dataset}.yaml").resolve()
    try:
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        if args.output.exists():
            raise ValueError(f"Choose a new output directory: {args.output}")
        entities = resolve_entities(args.dataset, args.data_root, args.entity)
        overrides = {"data_root": str(args.data_root), "seed": args.seed}
        global_config = load_config(args.config, variant="RGCSF", overrides=overrides)
        local_config = load_config(args.config, variant="FPLIR", overrides=overrides)
        if global_config["dataset"].lower() != args.dataset:
            raise ValueError("--config dataset does not match --dataset")
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Use a cpu or cuda device")
        stages = build_stages(args, entities)
        metadata = {"dataset": args.dataset, "entities": entities, "seed": args.seed,
                    "device": str(device), "data_root": str(args.data_root),
                    "protocol": "fdpg_adcl_chronological_50_10_40_v1",
                    "global_config": global_config, "local_config": local_config}
        print("Protocol: chronological 50/10/40 (training/validation/test).", flush=True)
        if args.dry_run:
            print(json.dumps({**metadata, "stages": stages}, indent=2, ensure_ascii=False))
            return 0
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Install the CUDA runtime or select --device cpu")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device does not exist")
        validate_data(args, entities, int(global_config["in_channels"]), int(global_config["window_size"]))
        execute_stages(stages, args.output, metadata)
        print(f"Completed. Report: {args.output / 'final' / 'datasets' / (args.dataset + '_metrics.json')}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed artifacts and the active stage log were retained.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
