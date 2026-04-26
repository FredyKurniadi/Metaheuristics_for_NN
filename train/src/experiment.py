from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import autograd.numpy as anp
import numpy as np

from core import generate_dataset, mse, normalize_to_unit_interval
from nn_model import forward_from_vector, param_dim, random_init_vector
from optimizers import run_ga, run_gradient_autograd, run_pso
from pca_viz import pca_transform, save_loss_curve, save_pca_path_3d, save_pca_path_3d_gif, save_y_curve_animation


def run_single_experiment(cfg: Dict, problem: str, method: str, output_dir: Path, seed_offset: int) -> Dict:
    rng = np.random.default_rng(int(cfg["seed"]) + seed_offset)

    layer_sizes = list(cfg["model"]["layers"])
    activation = str(cfg["model"].get("activation", "tanh"))
    bounds = tuple(float(v) for v in cfg["model"]["param_bounds"])

    data = generate_dataset(problem, cfg, rng)
    x = data["x"]
    x_model = normalize_to_unit_interval(x)
    y_obs = data["y_obs"]
    y_true = data["y_clean"]

    training_cfg = cfg.get("training", {})
    restarts = int(training_cfg.get("restarts_by_method", {}).get(method, training_cfg.get("restarts", 3)))
    init_std_default = float(cfg["model"].get("init_std", 0.5))
    init_std = float(cfg["model"].get("init_std_by_method", {}).get(method, init_std_default))
    best_result = None
    best_restart_idx = -1

    x_ag = anp.asarray(x_model, dtype=float)
    y_ag = anp.asarray(y_obs, dtype=float)

    for r_idx in range(restarts):
        init_vec = random_init_vector(
            layer_sizes=layer_sizes,
            mean=float(cfg["model"].get("init_mean", 0.0)),
            std=init_std,
            bounds=bounds,
            rng=rng,
        )

        def objective_np(params: np.ndarray) -> float:
            y_pred = forward_from_vector(params, x_model, layer_sizes, activation=activation, xp=np)
            return mse(y_obs, y_pred)

        def objective_ag(params):
            y_pred = forward_from_vector(params, x_ag, layer_sizes, activation=activation, xp=anp)
            diff = y_ag - y_pred
            return anp.mean(diff * diff)

        opt_cfg = cfg["optimizers"][method]
        if method == "gradient_autograd":
            result = run_gradient_autograd(objective_ag, init_vec, bounds, opt_cfg, rng=rng)
        elif method == "pso":
            result = run_pso(objective_np, init_vec, bounds, opt_cfg, rng)
        elif method == "ga":
            result = run_ga(objective_np, init_vec, bounds, opt_cfg, rng)
        else:
            raise ValueError(f"Unknown method: {method}")

        if best_result is None or result.best_loss < best_result.best_loss:
            best_result = result
            best_restart_idx = r_idx

    result = best_result
    opt_cfg = cfg["optimizers"][method]

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "best_params.npy", result.best_params)
    _save_history(output_dir / "history.csv", result.history_loss)

    coords, _, _, explained = pca_transform(result.history_params, n_components=int(cfg["visualization"].get("pca_components", 3)))
    _save_pca_coords(output_dir / "pca_coords.csv", coords)

    save_loss_curve(result.history_loss, output_dir / "loss_curve.png", f"{problem} | {method.upper()} | Loss")
    save_pca_path_3d(coords, output_dir / "pca_path_3d.png", f"{problem} | {method.upper()} | PCA 3D Path")
    save_pca_path_3d_gif(
        coords,
        output_dir / "pca_path_3d.gif",
        f"{problem} | {method.upper()} | PCA 3D Convergence",
        fps=int(cfg["visualization"].get("fps", 12)),
    )

    y_preds = [forward_from_vector(p, x_model, layer_sizes, activation=activation, xp=np) for p in result.history_params]
    save_y_curve_animation(
        x=x,
        y_obs=y_obs,
        y_true=y_true,
        y_preds=y_preds,
        out_path=output_dir / "y_pred_vs_true.gif",
        title=f"{problem} | {method.upper()} | NN Prediction",
        fps=int(cfg["visualization"].get("fps", 12)),
    )

    summary = {
        "problem": problem,
        "method": method,
        "param_dim": int(param_dim(layer_sizes)),
        "best_loss": float(result.best_loss),
        "best_a_true": float(data["a_true"]),
        "best_b_true": float(data["b_true"]),
        "iterations": int(opt_cfg["iterations"]),
        "population": int(opt_cfg["population"]) if "population" in opt_cfg else None,
        "selected_restart": int(best_restart_idx),
        "restarts": int(restarts),
        "pca_explained_variance_ratio": [float(v) for v in explained],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _save_history(path: Path, history_loss: np.ndarray):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "best_loss"])
        for i, loss in enumerate(history_loss):
            w.writerow([i, float(loss)])


def _save_pca_coords(path: Path, coords: np.ndarray):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "pc1", "pc2", "pc3"])
        for i, row in enumerate(coords):
            w.writerow([i, float(row[0]), float(row[1]), float(row[2])])
