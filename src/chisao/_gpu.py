"""
Working-GPU probe and CPU-fallback policy.

CuPy can *import* successfully yet fail at runtime when its CUDA libraries are
missing or mismatched (for example, two different ``cupy-cudaNNx`` wheels
installed at once). An import-only check then advertises a GPU that does not
actually work, and the first real device op crashes instead of degrading to CPU.

``probe_gpu`` returns ``True`` only if CuPy imports *and* a small elementwise op
plus a cuSOLVER call (``cupy.linalg.qr`` -- the library that fails on broken
installs) execute cleanly. The result, ``GPU_OK``, is the single source of truth
the rest of the package uses to decide NumPy vs CuPy.

Set the environment variable ``CHISAO_FORCE_CPU=1`` to force the CPU path
regardless of hardware.
"""

from __future__ import annotations

import os
import warnings


def probe_gpu() -> bool:
    """True only if a *working* CuPy/GPU is available."""
    if os.environ.get("CHISAO_FORCE_CPU"):
        return False
    try:
        import cupy as cp
    except Exception:
        return False
    try:
        _ = cp.zeros(2, dtype=cp.float64) + 1.0
        cp.linalg.qr(cp.eye(2, dtype=cp.float64))  # exercises cuSOLVER
        cp.cuda.Stream.null.synchronize()
        return True
    except Exception as exc:  # imported but unusable (e.g. broken CUDA libs)
        warnings.warn(
            f"CuPy is installed but a GPU operation failed "
            f"({type(exc).__name__}: {exc}). ChiSao will run on CPU. "
            "Check your CUDA/CuPy install -- a single matching cupy-cudaNNx "
            "wheel for your driver is usually the fix."
        )
        return False


GPU_OK: bool = probe_gpu()
