from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import autograd.numpy as anp
from autograd import grad
import numpy as np

ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class OptimResult:
    method: str
    best_params: np.ndarray
    best_loss: float
    history_params: np.ndarray
    history_loss: np.ndarray


def _clip(vec: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(vec, bounds[0], bounds[1])


def run_gradient_autograd(
    objective_ag: Callable,
    init_params: np.ndarray,
    bounds: tuple[float, float],
    cfg: Dict,
    rng: np.random.Generator | None = None,
) -> OptimResult:
    iterations = int(cfg["iterations"])
    lr = float(cfg.get("learning_rate", 0.01))
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("epsilon", 1e-8))
    lr_decay = float(cfg.get("lr_decay", 1.0))
    plateau_patience = int(cfg.get("plateau_patience", 40))
    perturb_scale = float(cfg.get("perturb_scale", 0.0))

    theta = _clip(np.asarray(init_params, dtype=float).copy(), bounds)
    if rng is None:
        rng = np.random.default_rng(0)
    grad_fn = grad(objective_ag)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    best = theta.copy()
    best_loss = float(objective_ag(theta))
    no_improve = 0
    history_p = [best.copy()]
    history_l = [best_loss]

    for t in range(1, iterations + 1):
        g = np.asarray(grad_fn(theta), dtype=float)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        step_lr = lr * (lr_decay ** (t - 1))
        theta = _clip(theta - step_lr * m_hat / (np.sqrt(v_hat) + eps), bounds)

        cur_loss = float(objective_ag(theta))
        if cur_loss < best_loss:
            best_loss = cur_loss
            best = theta.copy()
            no_improve = 0
        else:
            no_improve += 1

        if perturb_scale > 0.0 and no_improve >= plateau_patience:
            theta = _clip(theta + rng.normal(0.0, perturb_scale, size=theta.shape), bounds)
            no_improve = 0

        history_p.append(best.copy())
        history_l.append(best_loss)

    return OptimResult(
        method="gradient_autograd",
        best_params=best,
        best_loss=best_loss,
        history_params=np.asarray(history_p, dtype=float),
        history_loss=np.asarray(history_l, dtype=float),
    )


def run_pso(objective: ObjectiveFn, init_params: np.ndarray, bounds: tuple[float, float], cfg: Dict, rng: np.random.Generator) -> OptimResult:
    pop = int(cfg["population"])
    iters = int(cfg["iterations"])
    w = float(cfg.get("w", 0.72))
    c1 = float(cfg.get("c1", 1.6))
    c2 = float(cfg.get("c2", 1.6))

    dim = len(init_params)
    spread = float(cfg.get("init_spread", 0.35))
    particles = np.asarray(init_params, dtype=float).reshape(1, -1) + rng.normal(0.0, spread, size=(pop, dim))
    particles[0] = np.asarray(init_params, dtype=float)
    particles = _clip(particles, bounds)
    velocity = rng.normal(0.0, 0.1, size=(pop, dim))

    pbest = particles.copy()
    pbest_loss = np.array([objective(p) for p in pbest], dtype=float)
    g_idx = int(np.argmin(pbest_loss))
    gbest = pbest[g_idx].copy()
    gbest_loss = float(pbest_loss[g_idx])

    history_p = [gbest.copy()]
    history_l = [gbest_loss]

    for _ in range(iters):
        r1 = rng.uniform(size=(pop, dim))
        r2 = rng.uniform(size=(pop, dim))
        velocity = w * velocity + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest.reshape(1, -1) - particles)
        particles = _clip(particles + velocity, bounds)

        losses = np.array([objective(p) for p in particles], dtype=float)
        improved = losses < pbest_loss
        pbest[improved] = particles[improved]
        pbest_loss[improved] = losses[improved]

        g_idx = int(np.argmin(pbest_loss))
        if pbest_loss[g_idx] < gbest_loss:
            gbest = pbest[g_idx].copy()
            gbest_loss = float(pbest_loss[g_idx])

        history_p.append(gbest.copy())
        history_l.append(gbest_loss)

    return OptimResult(
        method="pso",
        best_params=gbest,
        best_loss=gbest_loss,
        history_params=np.asarray(history_p, dtype=float),
        history_loss=np.asarray(history_l, dtype=float),
    )


def _tournament(population: np.ndarray, losses: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    idx = rng.integers(0, len(population), size=k)
    return population[idx[int(np.argmin(losses[idx]))]]


def run_ga(objective: ObjectiveFn, init_params: np.ndarray, bounds: tuple[float, float], cfg: Dict, rng: np.random.Generator) -> OptimResult:
    pop = int(cfg["population"])
    iters = int(cfg["iterations"])
    elite_ratio = float(cfg.get("elite_ratio", 0.2))
    mut_rate = float(cfg.get("mutation_rate", 0.12))
    mut_scale = float(cfg.get("mutation_scale", 0.08))
    stagnation_patience = int(cfg.get("stagnation_patience", 35))
    mutation_boost = float(cfg.get("mutation_boost", 1.5))
    mutation_scale_max = float(cfg.get("mutation_scale_max", 0.5))

    dim = len(init_params)
    spread = float(cfg.get("init_spread", 0.45))
    population = np.asarray(init_params, dtype=float).reshape(1, -1) + rng.normal(0.0, spread, size=(pop, dim))
    population[0] = np.asarray(init_params, dtype=float)
    population = _clip(population, bounds)

    losses = np.array([objective(ind) for ind in population], dtype=float)
    b_idx = int(np.argmin(losses))
    best = population[b_idx].copy()
    best_loss = float(losses[b_idx])
    no_improve = 0

    history_p = [best.copy()]
    history_l = [best_loss]

    for _ in range(iters):
        order = np.argsort(losses)
        population = population[order]
        losses = losses[order]

        elite_count = max(1, int(round(elite_ratio * pop)))
        next_pop = [population[i].copy() for i in range(elite_count)]

        while len(next_pop) < pop:
            p1 = _tournament(population, losses, rng)
            p2 = _tournament(population, losses, rng)
            alpha = rng.uniform(size=dim)
            child = alpha * p1 + (1.0 - alpha) * p2

            mask = rng.uniform(size=dim) < mut_rate
            child = child + mask * rng.normal(0.0, mut_scale, size=dim)
            child = _clip(child, bounds)
            next_pop.append(child)

        population = np.asarray(next_pop, dtype=float)
        losses = np.array([objective(ind) for ind in population], dtype=float)

        c_idx = int(np.argmin(losses))
        if losses[c_idx] < best_loss:
            best_loss = float(losses[c_idx])
            best = population[c_idx].copy()
            no_improve = 0
            mut_scale = float(cfg.get("mutation_scale", 0.08))
        else:
            no_improve += 1

        if no_improve >= stagnation_patience:
            mut_scale = min(mutation_scale_max, mut_scale * mutation_boost)
            no_improve = 0

        history_p.append(best.copy())
        history_l.append(best_loss)

    return OptimResult(
        method="ga",
        best_params=best,
        best_loss=best_loss,
        history_params=np.asarray(history_p, dtype=float),
        history_loss=np.asarray(history_l, dtype=float),
    )
