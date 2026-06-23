"""End-to-end mode recovery on CPU (NumPy backend).

Smoke / regression tests, not the full SFU benchmark. They run on the NumPy
backend so they execute in CI without a GPU.

Note on seeders: the paper's central finding is that *seeding matters*. The
``carry_tiger`` seeder reliably recovers the global mode on landscapes where a
uniformly-random population does not. These tests therefore assert recovery for
``carry_tiger`` and only assert well-formed output (not recovery) for ``random``.
"""

import numpy as np

from chisao import optimize


def _recovered(peaks, target, tol):
    if peaks is None or len(peaks) == 0:
        return False
    d = np.max(np.abs(np.asarray(peaks) - np.asarray(target)), axis=1)
    return bool(np.any(d < tol))


def neg_rastrigin(X):
    A = 10.0
    d = X.shape[1]
    return -(A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1))


def sphere(X):
    return -np.sum(X**2, axis=1)


def test_sphere_unimodal_recovery():
    """Strictly concave: the single maximum at the origin must be found."""
    bounds = [(-5.0, 5.0)] * 4
    peaks, logL = optimize(sphere, bounds, seeder="carry_tiger", seed=0, n_oscillations=2)
    assert _recovered(peaks, np.zeros(4), tol=1e-2)


def test_rastrigin_carry_tiger_recovers_global():
    """Carry-Tiger seeding recovers the Rastrigin global max at the origin."""
    bounds = [(-5.12, 5.12)] * 2
    peaks, logL = optimize(neg_rastrigin, bounds, seeder="carry_tiger", seed=0, n_oscillations=3)
    assert _recovered(peaks, np.zeros(2), tol=5e-2)


def test_random_seeder_returns_well_formed_output():
    """The random seeder must run and return a well-formed (peaks, logL) pair."""
    bounds = [(-5.12, 5.12)] * 2
    peaks, logL = optimize(neg_rastrigin, bounds, seeder="random", seed=0, n_oscillations=3)
    assert peaks is not None
    peaks = np.asarray(peaks)
    logL = np.asarray(logL)
    # Either no peaks, or a [K, d] block with one logL per peak.
    assert len(peaks) == len(logL)
    if len(peaks):
        assert peaks.shape[1] == 2
