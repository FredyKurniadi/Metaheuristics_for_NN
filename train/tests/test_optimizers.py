from __future__ import annotations

import sys
from pathlib import Path

import autograd.numpy as anp
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from optimizers import run_ga, run_gradient_autograd, run_pso


def test_optimizers_on_quadratic() -> None:
    target = np.array([1.0, -0.5, 0.7, -1.2], dtype=float)
    init = np.array([2.2, 1.8, -1.4, 2.0], dtype=float)
    bounds = (-3.0, 3.0)

    def objective_np(x: np.ndarray) -> float:
        return float(np.mean((x - target) ** 2))

    def objective_ag(x):
        d = x - anp.asarray(target)
        return anp.mean(d * d)

    rng = np.random.default_rng(42)

    pso = run_pso(objective_np, init, bounds, {"population": 30, "iterations": 80, "w": 0.7, "c1": 1.5, "c2": 1.5}, rng)
    ga = run_ga(objective_np, init, bounds, {"population": 30, "iterations": 80, "elite_ratio": 0.2, "mutation_rate": 0.2, "mutation_scale": 0.1}, rng)
    gd = run_gradient_autograd(
        objective_ag,
        init,
        bounds,
        {"iterations": 220, "learning_rate": 0.04, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        rng=np.random.default_rng(123),
    )

    assert pso.best_loss < 1e-2
    assert ga.best_loss < 5e-2
    assert gd.best_loss < 2e-3
