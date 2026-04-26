from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_target(problem: str, x: np.ndarray, a: float, b: float) -> np.ndarray:
    if problem == "soal_1":
        return np.sin(a * x) * np.cos(b * x)
    if problem == "soal_2":
        exp_arg = np.clip(-a * (x**2), -8.0, 8.0)
        return np.exp(exp_arg) * np.sin(b * x)
    raise ValueError(f"Unknown problem: {problem}")


def make_noise(size: int, noise_cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    noise_type = noise_cfg.get("type", "gaussian")
    params = noise_cfg.get("params", {})
    if noise_type == "gaussian":
        return rng.normal(float(params.get("mean", 0.0)), float(params.get("std", 0.05)), size=size)
    if noise_type == "uniform":
        return rng.uniform(float(params.get("low", -0.05)), float(params.get("high", 0.05)), size=size)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def normalize_to_unit_interval(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = max(x_max - x_min, 1e-12)
    return 2.0 * (x - x_min) / span - 1.0


def generate_dataset(problem: str, cfg: Dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    p_cfg = cfg["problems"][problem]
    x_min, x_max = p_cfg["x_range"]
    num_samples = int(cfg["data"]["num_samples"])
    a_true = float(p_cfg["true_params"]["a"])
    b_true = float(p_cfg["true_params"]["b"])

    x = np.linspace(float(x_min), float(x_max), num_samples)
    y_clean = evaluate_target(problem, x, a_true, b_true)
    y_obs = y_clean + make_noise(num_samples, cfg["data"]["noise"], rng)

    return {
        "x": x,
        "y_clean": y_clean,
        "y_obs": y_obs,
        "a_true": a_true,
        "b_true": b_true,
    }
