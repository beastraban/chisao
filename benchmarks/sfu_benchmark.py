"""
sfu_benchmark.py — ChiSao full benchmark on the SFU optimization test suite
============================================================================

Covers all tractable functions from:
  https://www.sfu.ca/~ssurjano/optimization.html

Functions are grouped by category. Fixed-dimension functions (2D only, fixed-D)
are run at their canonical dimension. Variable-dimension functions are run at
configurable dims (default: 2 4 8 16).

Excluded (no closed-form global minimum or special structure):
  - Gramacy & Lee (1D only, not interesting for ChiSao)
  - Perm 0,d,β / Perm d,β  (known to be degenerate for gradient methods)
  - Power Sum (underdetermined)
  - Shekel (table-lookup constants — included separately)

Usage:
    python sfu_benchmark.py --source-dir .
    python sfu_benchmark.py --source-dir . --dims 2 4 8 --trials 10
    python sfu_benchmark.py --source-dir . --group multimodal --dims 4 8
    python sfu_benchmark.py --source-dir . --funcs rastrigin schwefel ackley
    python sfu_benchmark.py --source-dir . --out results.json
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings

import numpy as np

try:
    import cupy as cp
    try:
        GPU = cp.cuda.runtime.getDeviceCount() > 0   # imported but no visible device (CVD=-1) -> CPU
    except Exception:
        GPU = False
    if not GPU:
        cp = np
        warnings.warn('CuPy present but no CUDA device; running on CPU.')
except Exception:
    cp = np
    GPU = False
    warnings.warn('CuPy not available; running on CPU.')

SEED = 42

def get_xp(x):
    if GPU:
        return cp.get_array_module(x)
    return np

def gpu_clear():
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ─────────────────────────────────────────────────────────────────
# Load package (standalone chisao.py or sunburst package)
# ─────────────────────────────────────────────────────────────────

def load_package(source_dir=None):
    # Force the patched package when CHISAO_SRC is set -- bypasses the standalone/
    # sunburst shadowing below so the WHOLE gamut provably runs on the package.
    _forced = os.environ.get('CHISAO_SRC')
    if _forced:
        _forced = os.path.abspath(_forced)
        if _forced not in sys.path:
            sys.path.insert(0, _forced)
        for key in [k for k in list(sys.modules) if k == 'chisao' or k.startswith('chisao.')]:
            del sys.modules[key]
        import chisao as _pkg
        print(f'Loaded PATCHED package chisao from CHISAO_SRC={_forced} -> {_pkg.__file__}')
        return _pkg.sticky_hands

    if source_dir is not None:
        source_dir = os.path.abspath(source_dir)
        if os.path.exists(os.path.join(source_dir, 'chisao.py')):
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
            for key in list(sys.modules.keys()):
                if key in ('chisao', 'single_whip'):
                    del sys.modules[key]
            mod = importlib.import_module('chisao')
            print(f'Loaded standalone chisao.py from {source_dir}')
            print(f'GPU available : {mod.GPU_AVAILABLE}')
            print(f'SingleWhip    : {mod.SINGLEWHIP_AVAILABLE}')
            return mod.sticky_hands

        # sunburst package layout
        candidate = source_dir
        for _ in range(4):
            pkg = os.path.join(candidate, 'sunburst')
            if os.path.isdir(pkg):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                mod = importlib.import_module('sunburst.utils.chisao')
                print(f'Loaded sunburst package from {candidate}')
                return mod.sticky_hands
            candidate = os.path.dirname(candidate)

    mod = importlib.import_module('chisao')
    return mod.sticky_hands


# ─────────────────────────────────────────────────────────────────
# Function implementations  (maximization: return -f)
# ─────────────────────────────────────────────────────────────────

# ── Many Local Minima ──────────────────────────────────────────

def rastrigin(x):
    xp = get_xp(x)
    D = x.shape[1]
    return -(10 * D + xp.sum(x**2 - 10 * xp.cos(2 * np.pi * x), axis=1))

def ackley(x):
    xp = get_xp(x)
    D = x.shape[1]
    a, b, c = 20, 0.2, 2 * np.pi
    t1 = -a * xp.exp(-b * xp.sqrt(xp.sum(x**2, axis=1) / D))
    t2 = -xp.exp(xp.sum(xp.cos(c * x), axis=1) / D)
    return -(t1 + t2 + a + np.e)

def schwefel(x):
    xp = get_xp(x)
    D = x.shape[1]
    return -(418.9829 * D - xp.sum(x * xp.sin(xp.sqrt(xp.abs(x))), axis=1))

def griewank(x):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(1, D + 1, dtype=x.dtype)
    sum_sq = xp.sum(x**2, axis=1) / 4000
    prod_cos = xp.prod(xp.cos(x / xp.sqrt(idx)), axis=1)
    return -(sum_sq - prod_cos + 1)

def levy(x):
    xp = get_xp(x)
    D = x.shape[1]
    w = 1 + (x - 1) / 4
    term1 = xp.sin(np.pi * w[:, 0])**2
    term2 = xp.sum((w[:, :-1] - 1)**2 * (1 + 10 * xp.sin(np.pi * w[:, :-1] + 1)**2), axis=1)
    term3 = (w[:, -1] - 1)**2 * (1 + xp.sin(2 * np.pi * w[:, -1])**2)
    return -(term1 + term2 + term3)

def styblinski_tang(x):
    xp = get_xp(x)
    return -0.5 * xp.sum(x**4 - 16 * x**2 + 5 * x, axis=1)

def michalewicz(x, m=10):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(1, D + 1, dtype=x.dtype)
    sin_term = xp.sin((x**2 * idx) / np.pi)
    return xp.sum(xp.sin(x) * sin_term**(2 * m), axis=1)  # already negative (maximizing)

# ── Fixed 2D ──────────────────────────────────────────────────

def easom(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return xp.cos(x1) * xp.cos(x2) * xp.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def cross_in_tray(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    inner = xp.abs(xp.sin(x1) * xp.sin(x2) * xp.exp(xp.abs(100 - xp.sqrt(x1**2 + x2**2) / np.pi)))
    return 0.0001 * (inner + 1)**0.1  # positive, maximizing

def drop_wave(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    r2 = x1**2 + x2**2
    return (1 + xp.cos(12 * xp.sqrt(r2))) / (0.5 * r2 + 2)

def eggholder(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    t1 = -(x2 + 47) * xp.sin(xp.sqrt(xp.abs(x2 + x1 / 2 + 47)))
    t2 = -x1 * xp.sin(xp.sqrt(xp.abs(x1 - (x2 + 47))))
    return -(t1 + t2)

def holder_table(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    inner = xp.abs(1 - xp.sqrt(x1**2 + x2**2) / np.pi)
    return xp.abs(xp.sin(x1) * xp.cos(x2) * xp.exp(inner))

def schaffer2(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    val = 0.5 + (xp.sin(x1**2 - x2**2)**2 - 0.5) / (1 + 0.001 * (x1**2 + x2**2))**2
    return -val

def schaffer4(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    val = 0.5 + (xp.cos(xp.sin(xp.abs(x1**2 - x2**2)))**2 - 0.5) / \
          (1 + 0.001 * (x1**2 + x2**2))**2
    return -val

def shubert(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    i = xp.arange(1, 6, dtype=x.dtype)
    s1 = xp.sum(i * xp.cos((i + 1) * x1[:, None] + i), axis=1)
    s2 = xp.sum(i * xp.cos((i + 1) * x2[:, None] + i), axis=1)
    return -(s1 * s2)

def bukin6(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -(100 * xp.sqrt(xp.abs(x2 - 0.01 * x1**2)) + 0.01 * xp.abs(x1 + 10))

# ── Bowl-shaped ────────────────────────────────────────────────

def sphere(x):
    xp = get_xp(x)
    return -xp.sum(x**2, axis=1)

def rosenbrock(x):
    xp = get_xp(x)
    xi  = x[:, :-1]
    xi1 = x[:, 1:]
    return -xp.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, axis=1)

def zakharov(x):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(1, D + 1, dtype=x.dtype)
    s1 = xp.sum(x**2, axis=1)
    s2 = xp.sum(0.5 * idx * x, axis=1)
    return -(s1 + s2**2 + s2**4)

def dixon_price(x):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(2, D + 1, dtype=x.dtype)
    term1 = (x[:, 0] - 1)**2
    term2 = xp.sum(idx * (2 * x[:, 1:]**2 - x[:, :-1])**2, axis=1)
    return -(term1 + term2)

def trid(x):
    xp = get_xp(x)
    term1 = xp.sum((x - 1)**2, axis=1)
    term2 = xp.sum(x[:, 1:] * x[:, :-1], axis=1)
    return -(term1 - term2)

def rotated_hyper_ellipsoid(x):
    xp = get_xp(x)
    D = x.shape[1]
    # sum_{i=1}^{D} sum_{j=1}^{i} x_j^2
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(D):
        result += xp.sum(x[:, :i+1]**2, axis=1)
    return -result

def sum_squares(x):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(1, D + 1, dtype=x.dtype)
    return -xp.sum(idx * x**2, axis=1)

# ── Plate-shaped ───────────────────────────────────────────────

def booth(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -((x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2)

def matyas(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)

def mccormick(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -(xp.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1)

# ── Valley-shaped ──────────────────────────────────────────────

def camel3(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -(2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2)

def camel6(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -((4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2)

# ── Steep ridges/drops ─────────────────────────────────────────

def de_jong5(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    A = np.zeros((2, 25))
    a = np.arange(-2, 3)
    A[0] = np.tile(a, 5)
    A[1] = np.repeat(a, 5)
    A_xp = xp.array(A, dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for j in range(25):
        t = xp.sum((x - A_xp[:, j])**2, axis=1)
        result += 1.0 / (j + 1 + t**5 + 1e-10)  # stabilized
    val = (0.002 + result)**(-1)
    return -val

# ── Other ──────────────────────────────────────────────────────

def langermann(x, m=5):
    xp = get_xp(x)
    # Standard 2D constants
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]], dtype=np.float64)
    c = np.array([1, 2, 5, 2, 3], dtype=np.float64)
    A_xp = xp.array(A[:m], dtype=x.dtype)
    c_xp = xp.array(c[:m], dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(m):
        d = xp.sum((x - A_xp[i])**2, axis=1)
        result += c_xp[i] * xp.exp(-d / np.pi) * xp.cos(np.pi * d)
    return result  # already wants to be maximized (peaks are positive)

def levy13(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    val = (xp.sin(3*np.pi*x1)**2
           + (x1-1)**2 * (1 + xp.sin(3*np.pi*x2)**2)
           + (x2-1)**2 * (1 + xp.sin(2*np.pi*x2)**2))
    return -val

def bohachevsky1(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    val = (x1**2 + 2*x2**2
           - 0.3*xp.cos(3*np.pi*x1)
           - 0.4*xp.cos(4*np.pi*x2) + 0.7)
    return -val

def sum_of_different_powers(x):
    xp = get_xp(x)
    D = x.shape[1]
    idx = xp.arange(2, D + 2, dtype=x.dtype)
    return -xp.sum(xp.abs(x)**idx, axis=1)

def hartmann4(x):
    xp = get_xp(x)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5],
                  [0.05, 10, 17, 0.1],
                  [3, 3.5, 1.7, 10],
                  [17, 8, 0.05, 10]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124],
                          [2329, 4135, 8307, 3736],
                          [2348, 1451, 3522, 2883],
                          [4047, 8828, 8732, 5743]])
    A_xp = xp.array(A, dtype=x.dtype)
    P_xp = xp.array(P, dtype=x.dtype)
    alpha_xp = xp.array(alpha, dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(4):
        inner = xp.sum(A_xp[i] * (x - P_xp[i])**2, axis=1)
        result += alpha_xp[i] * xp.exp(-inner)
    return result

def shekel(x, m=10):
    xp = get_xp(x)
    # Standard D=4 constants
    C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    A = np.array([[4, 4, 4, 4],
                  [1, 1, 1, 1],
                  [8, 8, 8, 8],
                  [6, 6, 6, 6],
                  [3, 7, 3, 7],
                  [2, 9, 2, 9],
                  [5, 5, 3, 3],
                  [8, 1, 8, 1],
                  [6, 2, 6, 2],
                  [7, 3.6, 7, 3.6]], dtype=np.float64)
    A_xp = xp.array(A[:m], dtype=x.dtype)
    C_xp = xp.array(C[:m], dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(m):
        d = xp.sum((x - A_xp[i])**2, axis=1)
        result += 1.0 / (d + C_xp[i])
    return result

def beale(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    t1 = (1.5   - x1 + x1*x2)**2
    t2 = (2.25  - x1 + x1*x2**2)**2
    t3 = (2.625 - x1 + x1*x2**3)**2
    return -(t1 + t2 + t3)

def branin(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    a, b = 1.0, 5.1 / (4 * np.pi**2)
    c, r = 5 / np.pi, 6.0
    s, t = 10.0, 1 / (8 * np.pi)
    val = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*xp.cos(x1) + s
    return -val

def goldstein_price(x):
    xp = get_xp(x)
    x1, x2 = x[:, 0], x[:, 1]
    A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return -(A * B)

def hartmann3(x):
    xp = get_xp(x)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                          [4699, 4387, 7470],
                          [1091, 8732, 5547],
                          [381,  5743, 8828]])
    A_xp = xp.array(A, dtype=x.dtype)
    P_xp = xp.array(P, dtype=x.dtype)
    alpha_xp = xp.array(alpha, dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(4):
        inner = xp.sum(A_xp[i] * (x - P_xp[i])**2, axis=1)
        result += alpha_xp[i] * xp.exp(-inner)
    return result  # already positive (we maximize)

def hartmann6(x):
    xp = get_xp(x)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3,  17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124,  8283, 5886],
                          [2329, 4135, 8307, 3736, 1004, 9991],
                          [2348, 1451, 3522, 2883, 3047, 6650],
                          [4047, 8828, 8732, 5743, 1091, 381]])
    A_xp = xp.array(A, dtype=x.dtype)
    P_xp = xp.array(P, dtype=x.dtype)
    alpha_xp = xp.array(alpha, dtype=x.dtype)
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(4):
        inner = xp.sum(A_xp[i] * (x - P_xp[i])**2, axis=1)
        result += alpha_xp[i] * xp.exp(-inner)
    return result

def colville(x):
    xp = get_xp(x)
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    val = (100*(x1**2 - x2)**2 + (x1-1)**2 + (x3-1)**2
           + 90*(x3**2 - x4)**2 + 10.1*((x2-1)**2 + (x4-1)**2)
           + 19.8*(x2-1)*(x4-1))
    return -val

def powell(x):
    xp = get_xp(x)
    # D must be multiple of 4
    D = x.shape[1]
    result = xp.zeros(x.shape[0], dtype=x.dtype)
    for i in range(D // 4):
        x1 = x[:, 4*i]
        x2 = x[:, 4*i+1]
        x3 = x[:, 4*i+2]
        x4 = x[:, 4*i+3]
        result += ((x1 + 10*x2)**2 + 5*(x3 - x4)**2
                   + (x2 - 2*x3)**4 + 10*(x1 - x4)**4)
    return -result


# ─────────────────────────────────────────────────────────────────
# Function registry
# ─────────────────────────────────────────────────────────────────

# tol: L-infinity tolerance for success check
# fixed_D: if set, only run at this dimension
# optimum: global minimum location (for maximization: the argmax)
# opt_val: function VALUE at optimum (for max: the max value)
# multi_opt: True if multiple equivalent global optima exist

FUNC_REGISTRY = {
    # ── Many Local Minima ──
    'rastrigin': dict(
        func=rastrigin, group='multimodal',
        bounds=(-5.12, 5.12), optimum_x=0.0, tol=0.5,
        description='Highly multimodal, regular grid of minima',
    ),
    'ackley': dict(
        func=ackley, group='multimodal',
        bounds=(-32.768, 32.768), optimum_x=0.0, tol=1.0,
        description='Nearly flat outer region, deep central hole',
    ),
    'schwefel': dict(
        func=schwefel, group='multimodal',
        bounds=(-500.0, 500.0), optimum_x=420.9687, tol=2.0,
        description='Deceptive: global optimum far from next-best',
    ),
    'griewank': dict(
        func=griewank, group='multimodal',
        bounds=(-600.0, 600.0), optimum_x=0.0, tol=1.0,
        description='Widespread regular local minima',
    ),
    'levy': dict(
        func=levy, group='multimodal',
        bounds=(-10.0, 10.0), optimum_x=1.0, tol=0.2,
        description='Many local minima with sinusoidal structure',
    ),
    'styblinski_tang': dict(
        func=styblinski_tang, group='multimodal',
        bounds=(-5.0, 5.0), optimum_x=-2.903534, tol=0.2,
        description='Multiple local minima, moderate difficulty',
    ),
    'michalewicz': dict(
        func=michalewicz, group='multimodal',
        bounds=(0.0, np.pi), optimum_x=None, tol=0.1,  # no closed-form x*
        multi_opt=True,
        description='d! local minima, steep ridges (m=10)',
    ),
    # ── Fixed 2D multimodal ──
    'easom': dict(
        func=easom, group='multimodal_2d', fixed_D=2,
        bounds=(-100.0, 100.0), optimum_x=np.pi, tol=0.5,
        description='Unimodal but tiny basin relative to domain',
    ),
    'cross_in_tray': dict(
        func=cross_in_tray, group='multimodal_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=None, tol=0.1,
        multi_opt=True,
        description='4 equivalent global optima',
    ),
    'drop_wave': dict(
        func=drop_wave, group='multimodal_2d', fixed_D=2,
        bounds=(-5.12, 5.12), optimum_x=0.0, tol=0.2,
        description='Unimodal with concentric wave structure',
    ),
    'eggholder': dict(
        func=eggholder, group='multimodal_2d', fixed_D=2,
        bounds=(-512.0, 512.0), optimum_x=None, tol=2.0,
        multi_opt=True,
        description='Difficult, many local optima',
    ),
    'holder_table': dict(
        func=holder_table, group='multimodal_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=None, tol=0.1,
        multi_opt=True,
        description='4 equivalent global optima',
    ),
    'schaffer2': dict(
        func=schaffer2, group='multimodal_2d', fixed_D=2,
        bounds=(-100.0, 100.0), optimum_x=0.0, tol=0.2,
        description='Unimodal with oscillating gradient',
    ),
    'schaffer4': dict(
        func=schaffer4, group='multimodal_2d', fixed_D=2,
        bounds=(-100.0, 100.0), optimum_x=None, tol=0.2,
        multi_opt=True,
        description='Two global optima near axes',
    ),
    'levy13': dict(
        func=levy13, group='multimodal_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=1.0, tol=0.2,
        description='Levy N.13 — 2D variant',
    ),
    'langermann': dict(
        func=langermann, group='multimodal_2d', fixed_D=2,
        bounds=(0.0, 10.0), optimum_x=None, tol=0.2,
        multi_opt=True,
        description='5 local maxima, oscillatory decay envelope',
    ),
    'de_jong5': dict(
        func=de_jong5, group='multimodal_2d', fixed_D=2,
        bounds=(-65.536, 65.536), optimum_x=None, tol=1.0,
        multi_opt=True,
        description='25 wells, steep drops — De Jong N.5',
    ),
    'shubert': dict(
        func=shubert, group='multimodal_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=None, tol=0.2,
        multi_opt=True,
        description='18 equivalent global optima',
    ),
    'bukin6': dict(
        func=bukin6, group='multimodal_2d', fixed_D=2,
        bounds_asym=((-15.0, -5.0), (-3.0, 3.0)),
        optimum_x=None, tol=0.2,
        description='Ridge along parabola, narrow global minimum',
    ),
    # ── Bowl-shaped ──
    'bohachevsky': dict(
        func=bohachevsky1, group='bowl_2d', fixed_D=2,
        bounds=(-100.0, 100.0), optimum_x=0.0, tol=0.2,
        description='Bowl with cosine perturbations, global min at 0',
    ),
    'sum_of_different_powers': dict(
        func=sum_of_different_powers, group='bowl',
        bounds=(-1.0, 1.0), optimum_x=0.0, tol=0.05,
        description='Unimodal, increasing powers — easy',
    ),
    'sphere': dict(
        func=sphere, group='bowl',
        bounds=(-5.12, 5.12), optimum_x=0.0, tol=0.1,
        description='Convex, unimodal — sanity check',
    ),
    'rosenbrock': dict(
        func=rosenbrock, group='valley',
        bounds=(-5.0, 10.0), optimum_x=1.0, tol=0.2,
        description='Narrow parabolic valley, easy to find hard to converge',
    ),
    'zakharov': dict(
        func=zakharov, group='bowl',
        bounds=(-5.0, 10.0), optimum_x=0.0, tol=0.2,
        description='No local minima except global',
    ),
    'dixon_price': dict(
        func=dixon_price, group='valley',
        bounds=(-10.0, 10.0), optimum_x=None, tol=0.2,
        description='Global min: x_i = 2^(-(2^i-2)/2^i)',
    ),
    'trid': dict(
        func=trid, group='bowl',
        bounds=None, bounds_fn=lambda D: (-D * D, D * D),  # D-dependent: [-D^2, D^2]
        optimum_x=None, tol=0.5,
        description='Global min: x_i = i*(D+1-i)',
    ),
    'rotated_hyper_ellipsoid': dict(
        func=rotated_hyper_ellipsoid, group='bowl',
        bounds=(-65.536, 65.536), optimum_x=0.0, tol=0.2,
        description='Unimodal bowl with increasing ridge',
    ),
    'sum_squares': dict(
        func=sum_squares, group='bowl',
        bounds=(-10.0, 10.0), optimum_x=0.0, tol=0.2,
        description='Weighted sum of squares, unimodal',
    ),
    # ── Plate-shaped ──
    'booth': dict(
        func=booth, group='plate_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=None, tol=0.2,
        description='Global min at (1, 3)',
    ),
    'matyas': dict(
        func=matyas, group='plate_2d', fixed_D=2,
        bounds=(-10.0, 10.0), optimum_x=0.0, tol=0.2,
        description='Nearly flat plate, global min at origin',
    ),
    'mccormick': dict(
        func=mccormick, group='plate_2d', fixed_D=2,
        bounds_asym=((-1.5, 4.0), (-3.0, 4.0)),
        optimum_x=None, tol=0.2,
        description='Global min at (-0.547, -1.547)',
    ),
    # ── Valley-shaped ──
    'camel3': dict(
        func=camel3, group='valley_2d', fixed_D=2,
        bounds=(-5.0, 5.0), optimum_x=0.0, tol=0.2,
        description='Three-hump camel, global min at origin',
    ),
    'camel6': dict(
        func=camel6, group='valley_2d', fixed_D=2,
        bounds_asym=((-3.0, 3.0), (-2.0, 2.0)),
        optimum_x=None, tol=0.1,
        multi_opt=True,
        description='Six-hump camel, 2 global optima',
    ),
    # ── Other ──
    'beale': dict(
        func=beale, group='other_2d', fixed_D=2,
        bounds=(-4.5, 4.5), optimum_x=None, tol=0.2,
        description='Global min at (3, 0.5)',
    ),
    'branin': dict(
        func=branin, group='other_2d', fixed_D=2,
        bounds_asym=((-5.0, 10.0), (0.0, 15.0)),
        optimum_x=None, tol=0.3,
        multi_opt=True,
        description='3 equivalent global optima',
    ),
    'goldstein_price': dict(
        func=goldstein_price, group='other_2d', fixed_D=2,
        bounds=(-2.0, 2.0), optimum_x=None, tol=0.2,
        description='Global min at (0, -1)',
    ),
    'hartmann3': dict(
        func=hartmann3, group='other', fixed_D=3,
        bounds=(0.0, 1.0), optimum_x=None, tol=0.05,
        description='6 local minima, D=3',
    ),
    'hartmann4': dict(
        func=hartmann4, group='other', fixed_D=4,
        bounds=(0.0, 1.0), optimum_x=None, tol=0.05,
        multi_opt=True,
        description='4-D Hartmann, 4 local minima',
    ),
    'hartmann6': dict(
        func=hartmann6, group='other', fixed_D=6,
        bounds=(0.0, 1.0), optimum_x=None, tol=0.05,
        multi_opt=True,
        description='6-D Hartmann, 6 local minima',
    ),
    'shekel': dict(
        func=shekel, group='other', fixed_D=4,
        bounds=(0.0, 10.0), optimum_x=None, tol=0.2,
        multi_opt=True,
        description='Shekel m=10, D=4 — 10 local minima',
    ),
    'colville': dict(
        func=colville, group='other', fixed_D=4,
        bounds=(-10.0, 10.0), optimum_x=1.0, tol=0.2,
        description='Global min at (1,1,1,1)',
    ),
    'powell': dict(
        func=powell, group='other',
        bounds=(-4.0, 5.0), optimum_x=0.0, tol=0.2,
        description='D must be multiple of 4; global min at 0',
        dims_override=[4, 8, 16],
    ),
}

# Fix duplicate michalewicz key (registry dict only keeps last)
# Already correctly defined above — the second entry overwrites the first. Fine.


# ─────────────────────────────────────────────────────────────────
# Success check
# ─────────────────────────────────────────────────────────────────

def make_optimum(cfg, D):
    """Build the expected optimum array for a given D."""
    ox = cfg.get('optimum_x')
    if ox is None:
        return None
    return np.full(D, float(ox))

def recovered(peaks, cfg, D):
    if cfg.get('multi_opt'):
        # Success = found at least 1 peak (no unique target to check)
        return peaks is not None and len(peaks) > 0

    opt = make_optimum(cfg, D)
    if opt is None:
        return peaks is not None and len(peaks) > 0

    if peaks is None or len(peaks) == 0:
        return False

    p = np.array(peaks.get() if hasattr(peaks, 'get') else peaks)
    dists = np.max(np.abs(p - opt[None, :]), axis=1)
    return bool(np.any(dists < cfg['tol']))


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def make_bounds_xp(lo, hi, D):
    arr = np.array([[lo, hi]] * D, dtype=np.float64)
    return cp.asarray(arr) if GPU else arr

def make_bounds_asym_xp(bounds_asym):
    arr = np.array([[b[0], b[1]] for b in bounds_asym], dtype=np.float64)
    return cp.asarray(arr) if GPU else arr

def _lohi(cfg, D):
    """Resolve (lo, hi), honoring a D-dependent bounds_fn (e.g. trid: [-D^2, D^2])."""
    bf = cfg.get('bounds_fn')
    return bf(D) if bf is not None else cfg['bounds']

def get_bnd(cfg, D):
    asym = cfg.get('bounds_asym')
    if asym is not None:
        return make_bounds_asym_xp(asym)
    lo, hi = _lohi(cfg, D)
    return make_bounds_xp(lo, hi, D)

# ─────────────────────────────────────────────────────────────────
# Seeding strategies
# ─────────────────────────────────────────────────────────────────

def seed_random(cfg, D, N, seed):
    """Uniform random in bounds."""
    rng = np.random.default_rng(seed)
    asym = cfg.get('bounds_asym')
    if asym is not None:
        cols = [rng.uniform(b[0], b[1], N) for b in asym]
        x0 = np.column_stack(cols)
    else:
        lo, hi = _lohi(cfg, D)
        x0 = rng.uniform(lo, hi, (N, D))
    return cp.asarray(x0, dtype=cp.float64) if GPU else x0.astype(np.float64)


def seed_carry_tiger(cfg, D, seed):
    """
    CarryTiger ray-casting seeding (4-component recipe):
      v2v  (40%): vertex-to-vertex
      v2e  (30%): vertex-to-edge
      w2w  (30%): wall-to-wall
      sunburst: n_rays rays from center via QR orthonormal directions

    Samples n_samples_per_ray points along each ray.
    Returns all sample points as x0.
    """
    xp = cp if GPU else np
    rng_np = np.random.default_rng(seed)

    # Build bounds array [D, 2]
    asym = cfg.get('bounds_asym')
    if asym is not None:
        bounds_np = np.array([[b[0], b[1]] for b in asym], dtype=np.float64)
    else:
        lo, hi = _lohi(cfg, D)
        bounds_np = np.array([[lo, hi]] * D, dtype=np.float64)

    bounds_xp = xp.array(bounds_np)

    n_rays = int(10 + np.log2(max(2, D)))
    n_samples_per_ray = 50
    n_v2v = int(0.4 * n_rays)
    n_v2e = int(0.3 * n_rays)
    n_w2w = n_rays - n_v2v - n_v2e

    all_starts = []
    all_ends = []

    # v2v
    if n_v2v > 0:
        sc = rng_np.integers(0, 2, size=(n_v2v, D))
        ec = rng_np.integers(0, 2, size=(n_v2v, D))
        s = np.where(sc == 0, bounds_np[:, 0], bounds_np[:, 1])
        e = np.where(ec == 0, bounds_np[:, 0], bounds_np[:, 1])
        all_starts.append(s); all_ends.append(e)

    # v2e
    if n_v2e > 0:
        sc = rng_np.integers(0, 2, size=(n_v2e, D))
        s = np.where(sc == 0, bounds_np[:, 0], bounds_np[:, 1])
        e = rng_np.uniform(bounds_np[:, 0], bounds_np[:, 1], size=(n_v2e, D))
        fd = rng_np.integers(0, D, size=n_v2e)
        fs = rng_np.integers(0, 2, size=n_v2e)
        for i in range(n_v2e):
            e[i, fd[i]] = bounds_np[fd[i], fs[i]]
        all_starts.append(s); all_ends.append(e)

    # w2w
    if n_w2w > 0:
        s = rng_np.uniform(bounds_np[:, 0], bounds_np[:, 1], size=(n_w2w, D))
        e = rng_np.uniform(bounds_np[:, 0], bounds_np[:, 1], size=(n_w2w, D))
        sd = rng_np.integers(0, D, size=n_w2w); ss = rng_np.integers(0, 2, size=n_w2w)
        ed = rng_np.integers(0, D, size=n_w2w); es = rng_np.integers(0, 2, size=n_w2w)
        for i in range(n_w2w):
            s[i, sd[i]] = bounds_np[sd[i], ss[i]]
            e[i, ed[i]] = bounds_np[ed[i], es[i]]
        all_starts.append(s); all_ends.append(e)

    # sunburst from center
    center = (bounds_np[:, 0] + bounds_np[:, 1]) / 2.0
    ray_length = float(np.max(bounds_np[:, 1] - bounds_np[:, 0])) / 2.0
    A = rng_np.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)  # [D, D] orthonormal
    sb_starts = np.tile(center, (n_rays, 1))
    dir_indices = np.arange(n_rays) % D
    directions = Q[:, dir_indices].T  # [n_rays, D]
    signs = rng_np.choice([-1.0, 1.0], size=n_rays)
    sb_ends = center[None, :] + signs[:, None] * directions * ray_length
    sb_ends = np.clip(sb_ends, bounds_np[:, 0], bounds_np[:, 1])
    all_starts.append(sb_starts); all_ends.append(sb_ends)

    # Sample along all rays
    ray_starts = np.vstack(all_starts)   # [N_rays_total, D]
    ray_ends   = np.vstack(all_ends)
    n_total_rays = len(ray_starts)

    t = np.linspace(0, 1, n_samples_per_ray)
    # [n_total_rays, n_samples_per_ray, D]
    dirs = ray_ends - ray_starts
    samples = ray_starts[:, None, :] + t[None, :, None] * dirs[:, None, :]
    samples = np.clip(samples, bounds_np[:, 0], bounds_np[:, 1])
    x0 = samples.reshape(-1, D)

    return cp.asarray(x0, dtype=cp.float64) if GPU else x0.astype(np.float64)


def run_one(sticky_hands, fname, cfg, D, seed, seeder='random'):
    np.random.seed(seed)
    if GPU:
        cp.random.seed(seed)

    N = 50 * (10 + int(np.ceil(np.log2(max(2, D)))))
    bnd = get_bnd(cfg, D)

    if seeder == 'random':
        x0 = seed_random(cfg, D, N, seed)
    else:
        x0 = seed_carry_tiger(cfg, D, seed)

    params = dict(
        method             = 'lbfgs',
        n_converge         = 10,
        n_anticonverge     = 5,
        n_oscillations     = 3,
        stick_tolerance    = 1e-3,
        reseed_strategy    = 'sunburst',
        cannon_through_sky = True,
        bounds             = bnd,
        estimate_widths    = False,
        verbose            = False,
    )

    t0 = time.perf_counter()
    try:
        result = sticky_hands(cfg['func'], x0, **params)
        peaks = result.get('peaks')
    except Exception as e:
        print(f'    [ERROR {fname} D={D} seed={seeder}] {e}')
        gpu_clear()
        return None, time.perf_counter() - t0

    gpu_clear()
    return peaks, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────

SEEDERS = ['random', 'carry_tiger']


def _diag(peaks, cfg, D, ok):
    """Return a short diagnostic string for a single trial."""
    n_peaks = len(peaks) if peaks is not None else 0
    if not ok and peaks is not None and n_peaks > 0 and not cfg.get('multi_opt'):
        opt = make_optimum(cfg, D)
        if opt is not None:
            try:
                p = np.array(peaks.get() if hasattr(peaks, 'get') else peaks)
                dists = np.max(np.abs(p - opt[None, :]), axis=1)
                return f'Linf={np.min(dists):.2f}'
            except Exception:
                pass
    if not ok and (peaks is None or n_peaks == 0):
        return 'no peaks'
    return ''


def _cell_complete(entry, n_trials):
    """A (function, D) cell is done iff BOTH seeders have finalized stats at this n_trials."""
    if not isinstance(entry, dict):
        return False
    for s in SEEDERS:
        sd = entry.get(s)
        if not isinstance(sd, dict) or 'rate' not in sd or sd.get('n_trials') != n_trials:
            return False
    return True


def run_benchmark(sticky_hands, func_names, dims, n_trials, out_path=None):
    # results[fname][D][seeder] = {'rate', 'mean_wall', 'n_trials'}
    results = {}

    # ── resume: reload completed (function, D) cells from a prior interrupted run ──
    prior = {}
    if out_path and os.path.exists(out_path):
        try:
            raw = json.load(open(out_path))
            for f, dd in raw.items():
                prior[f] = {}
                for dk, ent in dd.items():
                    try:
                        prior[f][int(dk)] = ent          # JSON stringifies int D-keys; restore
                    except (TypeError, ValueError):
                        prior[f][dk] = ent
            n_done = sum(1 for f in prior for dk in prior[f]
                         if _cell_complete(prior[f][dk], n_trials))
            print(f'[resume] loaded {out_path}: {n_done} completed cell(s) will be skipped', flush=True)
        except Exception as e:
            print(f'[resume] could not read {out_path} ({e}); starting fresh', flush=True)
            prior = {}

    def _save():
        if out_path:
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    for fname in func_names:
        cfg = FUNC_REGISTRY[fname]
        fixed_D = cfg.get('fixed_D')
        dims_for_func = [fixed_D] if fixed_D else cfg.get('dims_override', dims)

        results[fname] = {}

        for D in dims_for_func:
            # resume: skip cells already finished on disk (never lose more than one cell)
            pcell = prior.get(fname, {}).get(D)
            if _cell_complete(pcell, n_trials):
                results[fname][D] = pcell
                print(f'  -> {fname} D={D}:  random={pcell["random"]["rate"]:.0%}  '
                      f'carry_tiger={pcell["carry_tiger"]["rate"]:.0%}  [resumed]', flush=True)
                _save()
                continue

            results[fname][D] = {s: {'successes': 0, 'walls': []} for s in SEEDERS}

            print(f'\n{"="*70}')
            print(f'  {fname}  D={D}  ({n_trials} trials)  [{cfg["description"]}]')
            print(f'{"="*70}')
            print(f'  {"trial":>7}  {"rand":>5}  {"rand diag":<14}  {"CT":>5}  {"CT diag"}')
            print(f'  {"-"*7}  {"-"*5}  {"-"*14}  {"-"*5}  {"-"*14}')

            for t in range(n_trials):
                seed = SEED + t
                row_parts = [f'  {t+1:>2}/{n_trials}']

                for seeder in SEEDERS:
                    peaks, wall = run_one(sticky_hands, fname, cfg, D, seed, seeder)
                    ok = recovered(peaks, cfg, D)
                    results[fname][D][seeder]['successes'] += int(ok)
                    results[fname][D][seeder]['walls'].append(wall)

                    n_peaks = len(peaks) if peaks is not None else 0
                    marker = 'v' if ok else 'x'
                    diag = _diag(peaks, cfg, D, ok)
                    row_parts.append(f'  {marker} {wall:.1f}s  {diag:<14}')

                print(''.join(row_parts))

            # Finalise per-seeder stats
            for seeder in SEEDERS:
                s_data = results[fname][D][seeder]
                n_s = s_data['successes']
                mean_w = float(np.mean(s_data['walls']))
                rate = n_s / n_trials
                results[fname][D][seeder] = {
                    'rate': rate, 'mean_wall': mean_w, 'n_trials': n_trials
                }

            rr = results[fname][D]['random']['rate']
            rc = results[fname][D]['carry_tiger']['rate']
            print(f'  -> {fname} D={D}:  random={rr:.0%}  carry_tiger={rc:.0%}')
            _save()                        # per-cell checkpoint -> resume never loses >1 cell

    # ── Summary table ──────────────────────────────────────────────────────
    W = 75
    print(f'\n\n{"="*W}')
    print(f'  SUMMARY  ({n_trials} trials each)')
    print(f'{"="*W}')
    print(f'  {"Function":<28}  {"D":>4}  {"Random":>7}  {"CarryTgr":>8}  {"d":>5}  Group')
    print(f'  {"-"*28}  {"-"*4}  {"-"*7}  {"-"*8}  {"-"*5}  -----')
    for fname, dresults in results.items():
        cfg = FUNC_REGISTRY[fname]
        for D, sr in dresults.items():
            rr = sr['random']['rate']
            rc = sr['carry_tiger']['rate']
            delta = rc - rr
            sign = '+' if delta >= 0 else ''
            print(f'  {fname:<28}  {D:>4}  {rr:>6.0%}  {rc:>7.0%}  {sign}{delta:>4.0%}  '
                  f'{cfg["group"]}')

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {out_path}')

    return results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

ALL_FUNCS = list(FUNC_REGISTRY.keys())
ALL_GROUPS = sorted(set(cfg['group'] for cfg in FUNC_REGISTRY.values()))

def main():
    parser = argparse.ArgumentParser(description='ChiSao SFU benchmark suite')
    parser.add_argument('--source-dir', default=None,
                        help='Directory containing chisao.py or sunburst package root')
    parser.add_argument('--dims', nargs='+', type=int, default=[2, 4, 8],
                        help='Dimensions to test (default: 2 4 8)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Trials per function/dim (default: 5)')
    parser.add_argument('--funcs', nargs='+', default=None,
                        choices=ALL_FUNCS, metavar='FUNC',
                        help=f'Functions to run. Choices: {ALL_FUNCS}')
    parser.add_argument('--group', default=None, choices=ALL_GROUPS,
                        help='Run only functions in this group')
    parser.add_argument('--out', default='sfu_benchmark.json',
                        help='JSON output path (incremental, crash-safe)')
    args = parser.parse_args()

    sticky_hands = load_package(args.source_dir)

    if args.funcs:
        funcs = args.funcs
    elif args.group:
        funcs = [f for f, cfg in FUNC_REGISTRY.items() if cfg['group'] == args.group]
    else:
        funcs = ALL_FUNCS

    run_benchmark(sticky_hands, funcs, args.dims, args.trials, args.out)

if __name__ == '__main__':
    main()
