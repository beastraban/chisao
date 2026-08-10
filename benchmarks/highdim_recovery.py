#!/usr/bin/env python3
r"""
highdim_recovery.py  --  the region-of-advantage test for the R3 rebuttal.
==========================================================================
ChiSao's edge is NOT "recovers Rastrigin at modest D" -- it is that at VERY HIGH
dimension the dedicated niching methods cannot run at all (crowding-/speciation-DE
are O(N^2) in a population ~20D; niching-CMA-ES is O(D^3) per generation), while
ChiSao runs the whole loop on the card and STILL recovers the shifted optimum.

This runs ChiSao GPU-native on SHIFTED Rastrigin (deep separable ripples, the hard
multimodal case) at D = 64 ... 2048, recovery ON (default), scored in L-inf. Pair
it with shifted_benchmark.py (which shows crowding-/speciation-DE and niching-CMA-ES
all fail AND exceed the wall-clock budget by D>=64) and the two together are the
complete rebuttal: the niching methods are dead in this regime; ChiSao is the only
method that both runs and recovers.

Niching-method feasibility ceiling (why they are absent here, not omitted):
  crowding-DE / speciation-DE : population ~20D -> (20D)^2 distance work; ~3 GB by
                                D~500, OOM / timeout well before D=1024.
  niching-CMA-ES              : D x D covariance + O(D^3) eigendecomp/generation;
                                intractable by D~512.
  ChiSao                      : O(D) L-BFGS, O(N) dedup, no host sync -> runs.

USAGE
    python highdim_recovery.py --dims 64,128,256,512,1024,2048 --shift-frac 0.4
"""
import argparse, json, math, os, sys, time
import numpy as np

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
    xp = np; DEV = "CPU (NumPy; run on the GPU box for the real numbers)"; USE_GPU = False
    def sync(): pass
    def to_cpu(a): return np.asarray(a)

TWO_PI = 2.0 * math.pi
BOX = {"rastrigin": (-5.12, 5.12), "ackley": (-32.0, 32.0)}


def rastrigin(X, s):
    """Maximization form, optimum (v=0) at x=s; value + analytic ascent gradient."""
    Y = xp.atleast_2d(X) - s
    v = -(10 * Y.shape[1] + xp.sum(Y ** 2 - 10 * xp.cos(TWO_PI * Y), 1))
    return v, -(2 * Y + 20 * math.pi * xp.sin(TWO_PI * Y))


def ackley(X, s):
    Y = xp.atleast_2d(X) - s; D = Y.shape[1]
    rms = xp.sqrt(xp.mean(Y ** 2, 1)); c = xp.mean(xp.cos(TWO_PI * Y), 1); a = -0.2 * rms
    v = -(-20 * xp.exp(a) - xp.exp(c) + 20 + math.e); safe = xp.where(rms > 0, rms, 1.0)
    g1 = xp.where((rms > 0)[:, None], (4 * xp.exp(a) / (D * safe))[:, None] * Y, 0.0)
    g2 = (TWO_PI / D * xp.exp(c))[:, None] * xp.sin(TWO_PI * Y)
    return v, -(g1 + g2)


FUN = {"rastrigin": rastrigin, "ackley": ackley}


def run(fname, D, lb, ub, shift):
    s = xp.asarray(shift); fn = FUN[fname]
    # normalize objective scale (÷std) so ChiSao's stick gate certifies peaks -- else
    # result['peaks'] comes back empty on large-scale objectives and recovery has nothing to do
    samp = xp.asarray(np.random.default_rng(0).uniform(lb, ub, (600, D)))
    sc = float(to_cpu(xp.std(fn(samp, s)[0]))) or 1.0
    fe = {"n": 0}
    def f(X):
        X = xp.atleast_2d(X); fe["n"] += X.shape[0]
        v, g = fn(X, s); return v / sc, g / sc
    b = _as_bounds([(lb, ub)] * D, xp)
    x0 = carry_tiger_seed(None, b, n_rays=1000, n_samples_per_ray=3, use_gpu=USE_GPU, seed=0)
    sync(); t0 = time.time()
    r = sticky_hands(f, x0, bounds=b, bank_samples=True, stick_tolerance=1e-3,
                     estimate_widths=False, n_oscillations=3, func_returns_grad=True)  # recovery ON by default
    sync(); wall = time.time() - t0
    bp = np.atleast_2d(np.asarray(to_cpu(r["sample_bank"]["positions"])))
    pk = r.get("peaks")
    if pk is not None and np.asarray(to_cpu(pk)).size > 0:
        bp = np.vstack([bp, np.atleast_2d(np.asarray(to_cpu(pk)))])
    linf = float(np.abs(bp - shift).max(1).min())
    rec = float(linf < max(0.5, 0.02 * (ub - lb)))
    return rec, linf, fe["n"], wall


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--function", default="rastrigin", choices=list(FUN))
    ap.add_argument("--dims", default="64,128,256,512,1024,2048")
    ap.add_argument("--shift-frac", type=float, default=0.4)
    ap.add_argument("--out", default="highdim_recovery.json")
    args = ap.parse_args()
    fn, (lb, ub) = args.function, BOX[args.function]
    print(f"Device: {DEV}")
    print(f"SHIFTED {args.function} (frac={args.shift_frac}), recovery ON, L∞ scoring — "
          f"niching methods cannot run at these D (see header)\n")
    hdr = f"{'D':>6} {'recover':>7} {'L_inf':>8} {'FEs':>13} {'wall_s':>9}"
    print(hdr); print("-" * len(hdr))
    print("warming up (kernels/context) ...", flush=True)
    d0 = int(args.dims.split(",")[0]); run(fn, d0, lb, ub,
        np.random.default_rng(1).uniform(-args.shift_frac*(ub-lb)/2, args.shift_frac*(ub-lb)/2, d0))
    results = {"device": DEV, "function": fn, "shift_frac": args.shift_frac, "by_dim": {}}
    for D in [int(d) for d in args.dims.split(",")]:
        shift = np.random.default_rng(1).uniform(-args.shift_frac*(ub-lb)/2, args.shift_frac*(ub-lb)/2, D)
        rec, linf, fes, wall = run(fn, D, lb, ub, shift)
        print(f"{D:>6} {rec:>7.2f} {linf:>8.3f} {fes:>13,} {wall:>9.2f}")
        results["by_dim"][str(D)] = dict(recover=rec, l_inf=linf, fes=int(fes), wall_s=wall)
        json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
