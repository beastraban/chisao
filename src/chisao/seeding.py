"""
chisao.seeding
==============

Initial-population seeders for ChiSao (``sticky_hands``).

ChiSao takes an initial population ``x0`` and oscillates it toward all
significant modes of a black-box function. *Where the population starts
matters.* On landscapes with large gradient-free regions (Ackley) or
product-coupled local minima (Griewank), a uniformly random population
never reaches the global basin, while a population seeded by structured
rays cast through the domain geometry does. This is the ``carry_tiger``
seeder of the SunBURST inference pipeline, reproduced here so that the
standalone package recovers the paper's ``carry_tiger`` results.

Two seeders are provided:

``random_seed``
    Uniform i.i.d. sampling over the box. The baseline.

``carry_tiger_seed``
    Structured ray-based initialization. Rays are cast between hypercube
    vertices, vertex-to-edge, wall-to-wall, and as a "sunburst" of
    QR-orthonormal directions from the domain center; the population is
    the set of points sampled along those rays. Lifted verbatim (in
    behaviour) from ``CarryTigerToMountain._generate_rays_discovery`` and
    ``._initial_sampling`` in the SunBURST source, decoupled from the
    evidence machinery.

The convenience wrapper :func:`optimize` ties a seeder to ``sticky_hands``
and returns the discovered modes, mirroring the two-seeder comparison
(``random`` vs ``carry_tiger``) used throughout the paper.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union, Dict, Any

import numpy as np

try:  # optional GPU backend
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

from .core import sticky_hands


__all__ = [
    "get_array_module",
    "random_seed",
    "carry_tiger_seed",
    "carry_tiger_rays",
    "optimize",
]


# ---------------------------------------------------------------------------
# Array-module selection
# ---------------------------------------------------------------------------
def get_array_module(use_gpu: Optional[bool] = None):
    """Return the array module to use.

    Parameters
    ----------
    use_gpu : bool or None
        ``True``  -> require CuPy (raises if unavailable).
        ``False`` -> force NumPy.
        ``None``  -> CuPy if available, else NumPy.
    """
    if use_gpu is True:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("use_gpu=True but CuPy is not available.")
        return cp
    if use_gpu is False:
        return np
    return cp if _CUPY_AVAILABLE else np


def _as_bounds(bounds, xp) -> "np.ndarray":
    """Coerce bounds to an ``[D, 2]`` array on the chosen backend."""
    b = xp.asarray(bounds, dtype=xp.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds must have shape [D, 2], got {tuple(b.shape)}")
    return b


def _adaptive_n_rays(dim: int) -> int:
    """Default ray count: 10 + log2(d), matching the SunBURST default."""
    return int(10 + np.log2(max(2, dim)))


# ---------------------------------------------------------------------------
# Random (uniform) seeder
# ---------------------------------------------------------------------------
def random_seed(
    bounds,
    n: int = 200,
    use_gpu: Optional[bool] = None,
    seed: Optional[int] = None,
):
    """Uniform i.i.d. population over the box.

    Parameters
    ----------
    bounds : array-like, shape [D, 2]
        Per-dimension ``[min, max]``.
    n : int
        Population size.
    use_gpu : bool or None
        Backend selection (see :func:`get_array_module`).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    x0 : ndarray, shape [n, D]
    """
    xp = get_array_module(use_gpu)
    b = _as_bounds(bounds, xp)
    if seed is not None:
        xp.random.seed(seed)
    lo, hi = b[:, 0], b[:, 1]
    return xp.random.uniform(lo, hi, size=(n, b.shape[0])).astype(xp.float64)


# ---------------------------------------------------------------------------
# Carry-Tiger structured ray seeder
# ---------------------------------------------------------------------------
def _qr_random_basis(dim: int, xp) -> "np.ndarray":
    """Random orthonormal basis via QR of a Gaussian matrix. Columns are vectors."""
    A = xp.random.randn(dim, dim).astype(xp.float64)
    Q, _ = xp.linalg.qr(A)
    return Q


def carry_tiger_rays(
    bounds,
    n_rays: Optional[int] = None,
    use_gpu: Optional[bool] = None,
    xp=None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Generate Carry-Tiger discovery rays.

    Four ray families (matching SunBURST ``_generate_rays_discovery``):

    * vertex-to-vertex  (40% of ``n_rays``)
    * vertex-to-edge    (30%)
    * wall-to-wall      (remainder)
    * sunburst          (``n_rays`` extra rays: QR-orthonormal directions
      from the domain center, random signs)

    Returns ``(ray_starts, ray_ends)``, each ``[N_rays, D]``.
    """
    if xp is None:
        xp = get_array_module(use_gpu)
    b = _as_bounds(bounds, xp)
    D = b.shape[0]
    if n_rays is None:
        n_rays = _adaptive_n_rays(D)

    n_v2v = int(0.4 * n_rays)
    n_v2e = int(0.3 * n_rays)
    n_w2w = n_rays - n_v2v - n_v2e

    all_starts, all_ends = [], []

    # --- vertex-to-vertex ---
    if n_v2v > 0:
        s_choice = xp.random.randint(0, 2, size=(n_v2v, D))
        e_choice = xp.random.randint(0, 2, size=(n_v2v, D))
        all_starts.append(xp.where(s_choice == 0, b[:, 0], b[:, 1]))
        all_ends.append(xp.where(e_choice == 0, b[:, 0], b[:, 1]))

    # --- vertex-to-edge ---
    if n_v2e > 0:
        s_choice = xp.random.randint(0, 2, size=(n_v2e, D))
        v2e_starts = xp.where(s_choice == 0, b[:, 0], b[:, 1])
        v2e_ends = xp.random.uniform(b[:, 0], b[:, 1], size=(n_v2e, D)).astype(xp.float64)
        row = xp.arange(n_v2e)
        fixed_dims = xp.random.randint(0, D, size=n_v2e)
        fixed_sides = xp.random.randint(0, 2, size=n_v2e)
        v2e_ends[row, fixed_dims] = xp.where(
            fixed_sides == 0, b[fixed_dims, 0], b[fixed_dims, 1]
        )
        all_starts.append(v2e_starts)
        all_ends.append(v2e_ends)

    # --- wall-to-wall ---
    if n_w2w > 0:
        w_starts = xp.random.uniform(b[:, 0], b[:, 1], size=(n_w2w, D)).astype(xp.float64)
        w_ends = xp.random.uniform(b[:, 0], b[:, 1], size=(n_w2w, D)).astype(xp.float64)
        row = xp.arange(n_w2w)
        sd = xp.random.randint(0, D, size=n_w2w)
        ss = xp.random.randint(0, 2, size=n_w2w)
        w_starts[row, sd] = xp.where(ss == 0, b[sd, 0], b[sd, 1])
        ed = xp.random.randint(0, D, size=n_w2w)
        es = xp.random.randint(0, 2, size=n_w2w)
        w_ends[row, ed] = xp.where(es == 0, b[ed, 0], b[ed, 1])
        all_starts.append(w_starts)
        all_ends.append(w_ends)

    # --- sunburst: QR-orthonormal directions from center, random signs ---
    center = (b[:, 0] + b[:, 1]) / 2.0
    box_width = b[:, 1] - b[:, 0]
    mx = xp.max(box_width)
    ray_length = float(mx.get() if hasattr(mx, "get") else mx) / 2.0
    Q = _qr_random_basis(D, xp)
    starts = xp.tile(center, (n_rays, 1))
    dir_idx = xp.arange(n_rays) % D
    directions = Q[:, dir_idx].T
    signs = xp.where(xp.random.randint(0, 2, size=n_rays) == 0, 1.0, -1.0)
    ends = center[None, :] + signs[:, None] * directions * ray_length
    ends = xp.clip(ends, b[:, 0], b[:, 1])
    all_starts.append(starts)
    all_ends.append(ends)

    return xp.vstack(all_starts), xp.vstack(all_ends)


def carry_tiger_seed(
    func: Optional[Callable],
    bounds,
    n_rays: Optional[int] = None,
    n_samples_per_ray: int = 50,
    use_gpu: Optional[bool] = None,
    seed: Optional[int] = None,
    return_values: bool = False,
):
    """Structured ray-based initial population (the ``carry_tiger`` seeder).

    Casts Carry-Tiger discovery rays through the domain and samples
    ``n_samples_per_ray`` points along each ray. The resulting points are
    the initial population handed to :func:`chisao.sticky_hands`.

    Parameters
    ----------
    func : callable or None
        Log-likelihood, batched: ``func(X[n, D]) -> [n]``. Only required
        when ``return_values=True``; otherwise the population is geometric
        and ``func`` may be ``None``.
    bounds : array-like, shape [D, 2]
    n_rays : int or None
        Base ray count per family; ``None`` -> ``10 + log2(d)``. Total rays
        produced are roughly ``2 * n_rays`` (geometric + sunburst).
    n_samples_per_ray : int
        Points sampled along each ray (linspace in ``[0, 1]``).
    use_gpu : bool or None
        Backend selection.
    seed : int or None
        RNG seed.
    return_values : bool
        If ``True``, also evaluate ``func`` on the population and return
        ``(x0, f0)``.

    Returns
    -------
    x0 : ndarray, shape [N, D]
        ``N = n_total_rays * n_samples_per_ray``.
    f0 : ndarray, shape [N]
        Only if ``return_values=True``.
    """
    xp = get_array_module(use_gpu)
    b = _as_bounds(bounds, xp)
    D = b.shape[0]
    if seed is not None:
        xp.random.seed(seed)

    ray_starts, ray_ends = carry_tiger_rays(b, n_rays=n_rays, xp=xp)
    t = xp.linspace(0.0, 1.0, n_samples_per_ray)
    dirs = ray_ends - ray_starts
    # [n_rays, n_samples, D]
    pts = ray_starts[:, None, :] + t[None, :, None] * dirs[:, None, :]
    pts = xp.clip(pts, b[:, 0], b[:, 1])
    x0 = pts.reshape(-1, D)

    if return_values:
        if func is None:
            raise ValueError("func must be provided when return_values=True")
        f0 = func(x0)
        return x0, f0
    return x0


# ---------------------------------------------------------------------------
# High-level optimize() wrapper
# ---------------------------------------------------------------------------
_SEEDERS: Dict[str, Callable] = {
    "carry_tiger": carry_tiger_seed,
    "random": random_seed,
}


def optimize(
    func: Callable,
    bounds,
    seeder: Union[str, Callable] = "carry_tiger",
    n_rays: Optional[int] = None,
    n_samples_per_ray: int = 50,
    n_random: int = 200,
    use_gpu: Optional[bool] = None,
    seed: Optional[int] = None,
    return_full: bool = False,
    **sticky_hands_kwargs: Any,
):
    """Seed a population and run ChiSao to convergence.

    Convenience entry point: builds ``x0`` with the requested seeder and
    calls :func:`chisao.sticky_hands`. ``func`` must return **log**-likelihood,
    batched over the leading axis.

    Parameters
    ----------
    func : callable
        ``func(X[n, D]) -> [n]`` log-likelihood.
    bounds : array-like, shape [D, 2]
    seeder : {"carry_tiger", "random"} or callable
        Population seeder. A callable must accept ``(func, bounds, use_gpu, seed)``
        (extra keywords are tolerated) and return ``x0``.
    n_rays, n_samples_per_ray :
        Forwarded to :func:`carry_tiger_seed`.
    n_random :
        Population size for the ``random`` seeder.
    use_gpu : bool or None
    seed : int or None
    return_full : bool
        If ``True``, return the full ``sticky_hands`` result dict instead of
        ``(peaks, L_peaks)``.
    **sticky_hands_kwargs :
        Passed through to :func:`chisao.sticky_hands` (e.g. ``n_oscillations``,
        ``stick_tolerance``, ``verbose``).

    Returns
    -------
    peaks, L_peaks : ndarrays
        Or the full result dict if ``return_full=True``.
    """
    xp = get_array_module(use_gpu)
    b = _as_bounds(bounds, xp)

    if callable(seeder):
        x0 = seeder(func=func, bounds=b, use_gpu=use_gpu, seed=seed)
    elif seeder == "carry_tiger":
        x0 = carry_tiger_seed(
            func, b, n_rays=n_rays, n_samples_per_ray=n_samples_per_ray,
            use_gpu=use_gpu, seed=seed,
        )
    elif seeder == "random":
        x0 = random_seed(b, n=n_random, use_gpu=use_gpu, seed=seed)
    else:
        raise ValueError(
            f"Unknown seeder {seeder!r}; expected one of {sorted(_SEEDERS)} or a callable."
        )

    result = sticky_hands(func, x0, bounds=b, **sticky_hands_kwargs)
    if return_full:
        return result
    return result["peaks"], result["L_peaks"]
