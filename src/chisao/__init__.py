"""
ChiSao
======

A GPU-native parallel optimizer for multimodal black-box functions via
convergence--anticonvergence oscillation ("sticky hands").

ChiSao runs an entire population of samples simultaneously and exploits a
deliberate convergence/anti-convergence oscillation cycle to escape local
traps while freezing confirmed modes. It is designed to map directly onto
GPU execution (one batch = one wall-clock step) and works as a drop-in
exploration module for any optimizer that supplies an initial population.

Quick start
-----------
>>> import numpy as np
>>> from chisao import optimize
>>> # log-likelihood, batched: f(X[n, d]) -> [n]
>>> def neg_rastrigin(X):
...     A = 10.0
...     return -(A * X.shape[1] + np.sum(X**2 - A * np.cos(2*np.pi*X), axis=1))
>>> bounds = [(-5.12, 5.12)] * 4
>>> peaks, logL = optimize(neg_rastrigin, bounds, seeder="carry_tiger", seed=0)

The package separates cleanly into:

* :func:`sticky_hands` -- the core oscillation optimizer (takes a population).
* :mod:`chisao.seeding` -- population seeders, including the structured
  ``carry_tiger`` ray seeder that the paper shows is essential on flat /
  coupled landscapes (Ackley, Griewank).
* :func:`optimize` -- a convenience wrapper that seeds + runs ChiSao.

GPU acceleration is automatic when CuPy is installed (``pip install chisao[gpu]``);
otherwise everything runs on NumPy.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import numpy as _np

from . import core as _core
from . import single_whip as _single_whip
from ._gpu import GPU_OK
from .core import (
    SINGLEWHIP_AVAILABLE,
    SINGLEWHIP_VERSION,
    GPUCapability,
    SampleBank,
    deduplicate_peaks_L_infinity,
    estimate_peak_width,
    get_gpu_info,
    gradient_ascent_batch,
    lbfgs_batch,
    optimize_batch,
    sticky_hands,
)
from .core import __version__ as __core_version__  # vendored optimizer lineage
from .seeding import (
    carry_tiger_rays,
    carry_tiger_seed,
    get_array_module,
    optimize,
    random_seed,
)
from .single_whip import (
    SingleWhip,
    randcoord_line_search_batch,
)

try:
    __version__ = _pkg_version("chisao")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.1.0"

# Graceful CPU fallback: if CuPy is absent OR installed-but-broken, force the
# vendored optimizer onto NumPy so it degrades to CPU instead of crashing.
if not GPU_OK:
    _core.GPU_AVAILABLE = False
    _core.cp = _np
    _single_whip.GPU_AVAILABLE = False
    _single_whip.cp = _np

__all__ = [
    "__version__",
    "__core_version__",
    "GPU_OK",
    # core optimizer
    "sticky_hands",
    "optimize",
    "lbfgs_batch",
    "gradient_ascent_batch",
    "deduplicate_peaks_L_infinity",
    "estimate_peak_width",
    "optimize_batch",
    "SampleBank",
    # seeding
    "carry_tiger_seed",
    "carry_tiger_rays",
    "random_seed",
    "get_array_module",
    # singlewhip toolkit
    "SingleWhip",
    "randcoord_line_search_batch",
    # gpu / capability
    "get_gpu_info",
    "GPUCapability",
    "SINGLEWHIP_AVAILABLE",
    "SINGLEWHIP_VERSION",
]
