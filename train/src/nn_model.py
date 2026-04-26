from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def param_dim(layer_sizes: Sequence[int]) -> int:
    total = 0
    for i in range(len(layer_sizes) - 1):
        in_dim = int(layer_sizes[i])
        out_dim = int(layer_sizes[i + 1])
        total += in_dim * out_dim + out_dim
    return total


def vector_to_params(vec, layer_sizes: Sequence[int], xp=np) -> List[Tuple]:
    params = []
    idx = 0
    for i in range(len(layer_sizes) - 1):
        in_dim = int(layer_sizes[i])
        out_dim = int(layer_sizes[i + 1])

        w_count = in_dim * out_dim
        b_count = out_dim

        w = vec[idx : idx + w_count].reshape((in_dim, out_dim))
        idx += w_count
        b = vec[idx : idx + b_count]
        idx += b_count
        params.append((w, b))
    return params


def forward_from_vector(vec, x, layer_sizes: Sequence[int], activation: str = "tanh", xp=np):
    params = vector_to_params(vec, layer_sizes, xp=xp)
    h = x.reshape((-1, 1))

    for li, (w, b) in enumerate(params):
        h = h @ w + b
        if li < len(params) - 1:
            if activation == "tanh":
                h = xp.tanh(h)
            elif activation == "relu":
                h = xp.maximum(h, 0.0)
            elif activation == "sin":
                h = xp.sin(h)
            else:
                raise ValueError(f"Unsupported activation: {activation}")

    return h.reshape((-1,))


def random_init_vector(
    layer_sizes: Sequence[int],
    mean: float,
    std: float,
    bounds: Tuple[float, float],
    rng: np.random.Generator,
):
    dim = param_dim(layer_sizes)
    vec = rng.normal(loc=mean, scale=std, size=dim)
    low, high = float(bounds[0]), float(bounds[1])
    return np.clip(vec, low, high)
