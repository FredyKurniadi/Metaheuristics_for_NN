from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from experiment import run_single_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run METAHEURISTIK2 experiments")
    p.add_argument("--config", type=str, default="train/configs/experiment.yaml")
    return p.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_model_dir(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("model_")])
    if not existing:
        return models_dir / "model_001"
    latest = max(int(n.split("_")[1]) for n in existing)
    return models_dir / f"model_{latest + 1:03d}"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    run_dir = next_model_dir(Path(cfg["output"].get("models_dir", "models")))
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    seed_stride = int(cfg.get("seed_stride_per_problem", 100))
    for p_idx, problem in enumerate(cfg["problems"].keys()):
        problem_seed_offset = p_idx * seed_stride
        for method in cfg["optimizers"].keys():
            out_dir = run_dir / problem / method
            summary = run_single_experiment(cfg, problem, method, out_dir, problem_seed_offset)
            summaries.append(summary)
            print(f"[done] {problem} | {method} | best_loss={summary['best_loss']:.6f}")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"results": summaries}, f, indent=2)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[output] {run_dir}")


if __name__ == "__main__":
    main()
