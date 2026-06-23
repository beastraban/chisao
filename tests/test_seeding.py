"""Seeders produce in-bounds populations of the expected shape."""

import numpy as np

from chisao import carry_tiger_rays, carry_tiger_seed, random_seed

BOUNDS = [(-5.0, 5.0), (-3.0, 7.0), (-1.0, 1.0)]


def _in_bounds(x, bounds):
    b = np.asarray(bounds, float)
    return np.all(x >= b[:, 0] - 1e-9) and np.all(x <= b[:, 1] + 1e-9)


def test_random_seed_shape_and_bounds():
    x0 = random_seed(BOUNDS, n=128, use_gpu=False, seed=1)
    assert x0.shape == (128, 3)
    assert _in_bounds(x0, BOUNDS)


def test_carry_tiger_rays_shapes_match():
    starts, ends = carry_tiger_rays(BOUNDS, n_rays=12, use_gpu=False)
    assert starts.shape == ends.shape
    assert starts.shape[1] == 3
    assert _in_bounds(ends, BOUNDS)


def test_carry_tiger_seed_in_bounds():
    x0 = carry_tiger_seed(None, BOUNDS, n_rays=12, n_samples_per_ray=20, use_gpu=False, seed=2)
    assert x0.ndim == 2 and x0.shape[1] == 3
    assert x0.shape[0] > 0
    assert _in_bounds(x0, BOUNDS)


def test_carry_tiger_seed_reproducible():
    a = carry_tiger_seed(None, BOUNDS, n_rays=12, use_gpu=False, seed=7)
    b = carry_tiger_seed(None, BOUNDS, n_rays=12, use_gpu=False, seed=7)
    assert np.allclose(a, b)


def test_carry_tiger_seed_with_values():
    def f(X):
        return -np.sum(X**2, axis=1)

    x0, f0 = carry_tiger_seed(
        f, BOUNDS, n_rays=8, n_samples_per_ray=10, use_gpu=False, seed=3, return_values=True
    )
    assert f0.shape[0] == x0.shape[0]
