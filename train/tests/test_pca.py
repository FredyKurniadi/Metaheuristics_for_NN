from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pca_viz import pca_transform


def test_pca_output_dims() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 20))
    coords, comps, mean, explained = pca_transform(x, n_components=3)

    assert coords.shape == (50, 3)
    assert comps.shape[0] == 3
    assert mean.shape[0] == 20
    assert explained.shape[0] == 3
