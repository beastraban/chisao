#!/usr/bin/env python3
r"""
gpu_scaling.py
==============
ChiSao's scaling on its native hardware. The whole loop runs on-device: a CuPy
objective (value + analytic gradient), a GPU-seeded population, no NumPy/CuPy
mixing anywhere. Measures the two exponents that matter and confirms the GPU
half of the scaling story:

    FE count   ~ D^a   (algorithmic work -- expect strongly sublinear, ~D^0.11)
    wall-clock ~ D^b   (on GPU, expect the per-eval O(D) to parallelize away so
                        wall-clock tracks the FE count -- ~D^0.14, the paper's
                        headline number; on CPU it degrades to ~D^1 or worse)

Device-agnostic: uses CuPy if ChiSao reports GPU_OK, else NumPy. Run it on the
3080 to get the GPU exponent; the CPU path is only for plumbing checks.

USAGE
    python gpu_scaling.py --dims 64,128,256,512,1024,2048 [--baselines]
"""
import argparse, json, math, os, sys, time
import numpy as np

# IMPORTANT: do NOT force CPU -- we want ChiSao on the GPU when CuPy is present.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import chisao
from chisao import sticky_hands
from chisao.seeding import carry_tiger_seed, _as_bounds

if chisao.GPU_OK:
    import cupy as xp
    DEV = f"GPU (CuPy, {xp.cuda.runtime.getDeviceProperties(0)['name'].decode()})"
    USE_GPU = True
    def sync(): xp.cuda.Stream.null.synchronize()
    def to_cpu(a): return xp.asnumpy(a)
else:
    xp = np
    DEV = "CPU (NumPy; ChiSao GPU_OK=False -- plumbing check only)"
    USE_GPU = False
    def sync(): pass
    def to_cpu(a): return np.asarray(a)

TWO_PI = 2.0 * math.pi
BOX = (-32.0, 32.0)                                   # Ackley domain


def ackley(X):
    """Ackley (maximization), value + analytic gradient, in the active xp.
    Global optimum at the origin."""
    X = xp.atleast_2d(X); D = X.shape[1]
    rms = xp.sqrt(xp.mean(X ** 2, 1)); c = xp.mean(xp.cos(TWO_PI * X), 1)
    a = -0.2 * rms
    v = -(-20 * xp.exp(a) - xp.exp(c) + 20 + math.e)
    safe = xp.where(rms > 0, rms, 1.0)
    g1 = (4 * xp.exp(a) / (D * safe))[:, None] * X
    g1 = xp.where((rms > 0)[:, None], g1, 0.0)
    g2 = (TWO_PI / D * xp.exp(c))[:, None] * xp.sin(TWO_PI * X)
    return v, -(g1 + g2)


def run(D):
    lb, ub = BOX
    fe = {"n": 0}
    def f(X):
        X = xp.atleast_2d(X); fe["n"] += X.shape[0]
        return ackley(X)                              # (value, grad), both on device
    def vo(X):
        X = xp.atleast_2d(X); fe["n"] += X.shape[0]
        return ackley(X)[0]
    b = _as_bounds([(lb, ub)] * D, xp)
    x0 = carry_tiger_seed(vo, b, use_gpu=USE_GPU, seed=0)   # on-device seed
    sync(); t0 = time.time()
    r = sticky_hands(f, x0, bounds=b, bank_samples=True, stick_tolerance=1e-3,
                     estimate_widths=False, n_oscillations=3, func_returns_grad=True)
    sync(); wall = time.time() - t0
    # extraction + scoring on host (one-time, off the timed path)
    bp = to_cpu(np.atleast_2d(np.asarray(to_cpu(r["sample_bank"]["positions"]))))
    fv = ackley(xp.asarray(bp))[0]; fv = to_cpu(fv)
    # compact extract (host)
    tol = 0.05 * (fv.max() - np.median(fv)); tol = max(tol, 1e-6)
    m = fv >= fv.max() - tol; P = bp[m]
    rr = 0.01 * (ub - lb); seen = set(); keep = []
    for i in np.argsort(-fv[m]):
        k = tuple(np.floor(P[i] / max(rr, 1e-9)).astype(np.int64))
        if k not in seen:
            seen.add(k); keep.append(i)
    pts = P[keep]
    rec = 0.0 if len(pts) == 0 else float(np.min(np.abs(pts).max(axis=1)) < max(0.5, 0.02 * (ub - lb)))  # L∞ to origin
    return rec, fe["n"], wall


# ---------------------------------------------------------------------------
# Baselines (CPU libraries) for the SAME-HARDWARE exponent comparison.
# These always run in NumPy. The honest comparison is CPU-vs-CPU: ChiSao on CPU
# vs these -> an ALGORITHMIC (exponent) gap, hardware-independent. On GPU, ChiSao's
# curve is the throughput story and is NOT exponent-comparable to these CPU libs.
# ---------------------------------------------------------------------------
def _ackley_np(X):
    X = np.atleast_2d(X); D = X.shape[1]
    rms = np.sqrt(np.mean(X ** 2, 1)); c = np.mean(np.cos(TWO_PI * X), 1)
    return -(-20 * np.exp(-0.2 * rms) - np.exp(c) + 20 + math.e)   # maximization value


def _np_min_obj():
    fe = {"n": 0}
    def f(x):                                    # scalar minimization objective = -ackley
        fe["n"] += 1
        return -float(_ackley_np(np.asarray(x)[None, :])[0])
    return f, fe


def run_cmaes(D, maxiter=100000):   # high safety cap; convergence (tolfun/tolx) stops it first
    import cma
    f, fe = _np_min_obj(); lb, ub = BOX
    x0 = np.random.uniform(lb, ub, D); sigma0 = (ub - lb) / 4.0
    t0 = time.time()
    # Run to CONVERGENCE, not a fixed iteration cap, so CMA-ES's true O(D^2)+ cost shows.
    es = cma.CMAEvolutionStrategy(x0, sigma0,
        {'bounds': [lb, ub], 'maxiter': maxiter, 'tolfun': 1e-9, 'tolx': 1e-9, 'verbose': -9})
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f(s) for s in sols])
    return time.time() - t0, fe["n"]


def run_de(D, maxiter=40):
    from scipy.optimize import differential_evolution
    f, fe = _np_min_obj(); lb, ub = BOX
    t0 = time.time()
    differential_evolution(f, [(lb, ub)] * D, maxiter=maxiter, tol=0, polish=False, seed=0)
    return time.time() - t0, fe["n"]


def run_bh(D, niter=20):
    from scipy.optimize import basinhopping
    f, fe = _np_min_obj(); lb, ub = BOX
    x0 = np.random.uniform(lb, ub, D)
    t0 = time.time()
    basinhopping(f, x0, niter=niter, seed=0)
    return time.time() - t0, fe["n"]


BASELINES = {"CMA-ES": run_cmaes, "DE": run_de, "BasinHop": run_bh}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="64,128,256,512,1024")
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per D (median reported)")
    ap.add_argument("--baselines", action="store_true",
                    help="also run CMA-ES/DE/BasinHopping (CPU) for the same-hardware exponent comparison")
    ap.add_argument("--baseline-max-dim", type=int, default=256,
                    help="cap dim for baselines (they scale O(D^2-3); avoid hanging at high D)")
    ap.add_argument("--out", default="gpu_scaling.json")
    args = ap.parse_args()
    dims = [int(d) for d in args.dims.split(",")]
    print(f"Device: {DEV}\n")
    print("warming up (compiling kernels / building context) ...", flush=True)
    run(dims[0])                                       # discard: one-time GPU warmup
    print(f"{'D':>6} {'recover':>7} {'FEs':>13} {'wall_s(med)':>11}")
    print("-" * 42)
    FE, WALL, REC = [], [], []
    for D in dims:
        walls = []; rec = fes = None
        for _ in range(max(1, args.reps)):
            rec, fes, w = run(D); walls.append(w)
        wall = float(np.median(walls))
        FE.append(fes); WALL.append(wall); REC.append(rec)
        print(f"{D:>6} {rec:>7.2f} {fes:>13,} {wall:>11.3f}")
    chisao_b = None
    if len(dims) >= 2:
        a = np.polyfit(np.log(dims), np.log(FE), 1)[0]
        chisao_b = np.polyfit(np.log(dims), np.log(WALL), 1)[0]
        print("-" * 42)
        print(f"FE count   ~ D^{a:+.3f}   (note: a regime switch near D=512 breaks a single fit)")
        print(f"wall-clock ~ D^{chisao_b:+.3f}   ({'GPU: ~flat here (card not saturated)' if USE_GPU else 'CPU: ~D^1 from serial O(D)'})")

    if args.baselines:
        bdims = [d for d in dims if d <= args.baseline_max_dim]
        print(f"\n=== Baselines (CPU libraries, same-hardware exponent comparison; D <= {args.baseline_max_dim}) ===")
        print(f"{'D':>6} " + " ".join(f"{n:>11}" for n in BASELINES) + "    (wall_s)")
        bwall = {n: [] for n in BASELINES}
        for D in bdims:
            row = f"{D:>6} "
            for n, fn in BASELINES.items():
                try:
                    w, _ = fn(D); bwall[n].append(w); row += f"{w:>11.3f} "
                except Exception as e:
                    bwall[n].append(np.nan); row += f"{'ERR':>11} "
            print(row, flush=True)
        print("-" * 42)
        dev = "CPU" if not USE_GPU else "GPU"
        if chisao_b is not None:
            note = "" if not USE_GPU else "  (GPU -- NOT exponent-comparable to CPU baselines below)"
            print(f"{'ChiSao':>10}: wall ~ D^{chisao_b:+.3f}   [{dev}]{note}")
        for n in BASELINES:
            ws = np.array(bwall[n]); ok = ~np.isnan(ws)
            if ok.sum() >= 2:
                be = np.polyfit(np.log(np.array(bdims)[ok]), np.log(ws[ok]), 1)[0]
                print(f"{n:>10}: wall ~ D^{be:+.3f}   [CPU]")
        print("\nInterpretation: on identical (CPU) hardware, ChiSao's exponent should sit")
        print("below the baselines' -- linear-while-multimodal vs their super-linear cost.")

    out = {"device": DEV, "dims": dims,
           "chisao": {"FE": [int(x) for x in FE], "wall_s": WALL, "recover": REC},
           "chisao_wall_exponent": (float(chisao_b) if chisao_b is not None else None)}
    if args.baselines:
        out["baseline_dims"] = bdims
        out["baselines_wall_s"] = {n: bwall[n] for n in BASELINES}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
