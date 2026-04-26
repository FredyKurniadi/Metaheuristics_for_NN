from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nn_model import forward_from_vector, param_dim


def test_param_dim_positive() -> None:
    layers = [1, 8, 8, 1]
    assert param_dim(layers) == 97


def test_forward_output_shape() -> None:
    layers = [1, 4, 1]
    dim = param_dim(layers)
    vec = np.linspace(-0.1, 0.1, dim)
    x = np.linspace(-1.0, 1.0, 21)
    y = forward_from_vector(vec, x, layers, activation="tanh", xp=np)
    assert y.shape == x.shape
