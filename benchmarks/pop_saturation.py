#!/usr/bin/env python3
r"""
pop_saturation.py
=================
Saturate the card by growing the POPULATION at a fixed dimension, and ask two
questions the dimension-sweep can't:

  1. Throughput -- how does wall-clock grow with population N? On GPU, small N is
     overhead-bound (flat); large N saturates and goes ~linear. Shows the card is
     actually used, and how big a population you can afford.
  2. Robustness-for-free -- does a LARGE population recover cases a small one
     missed? ChiSao's failures came with N~1e3. Full GPU parallelism lets you
     run N~1e5 for near-flat wall-clock. If that recovers the shifted / hard
     optimum, the parallelism is buying back coverage, not just speed.

Device-agnostic (CuPy if chisao.GPU_OK else NumPy). Run it on the 3080.

USAGE
    python pop_saturation.py --function ackley --D 64 --shift-frac 0.4 \
        --pops 200,1000,5000,20000,100000
"""
import argparse, json, math, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
import chisao
from chisao import sticky_hands
from chisao.seeding import random_seed, carry_tiger_seed, _as_bounds

if chisao.GPU_OK:
    import cupy as xp
    DEV = f"GPU (CuPy, {xp.cuda.runtime.getDeviceProperties(0)['name'].decode()})"
    USE_GPU = True
    def sync(): xp.cuda.Stream.null.synchronize()
    def to_cpu(a): return xp.asnumpy(a)
else:
    xp = np; DEV = "CPU (NumPy; plumbing check only)"; USE_GPU = False
    def sync(): pass
    def to_cpu(a): return np.asarray(a)

TWO_PI = 2.0 * math.pi
BOX = {"ackley": (-32.0, 32.0), "rastrigin": (-5.12, 5.12)}


def ackley(X, s):
    X = xp.atleast_2d(X) - s; D = X.shape[1]
    rms = xp.sqrt(xp.mean(X ** 2, 1)); c = xp.mean(xp.cos(TWO_PI * X), 1)
    a = -0.2 * rms
    v = -(-20 * xp.exp(a) - xp.exp(c) + 20 + math.e)
    safe = xp.where(rms > 0, rms, 1.0)
    g1 = xp.where((rms > 0)[:, None], (4 * xp.exp(a) / (D * safe))[:, None] * X, 0.0)
    g2 = (TWO_PI / D * xp.exp(c))[:, None] * xp.sin(TWO_PI * X)
    return v, -(g1 + g2)


def rastrigin(X, s):
    X = xp.atleast_2d(X) - s
    v = -(10 * X.shape[1] + xp.sum(X ** 2 - 10 * xp.cos(TWO_PI * X), 1))
    return v, -(2 * X + 20 * math.pi * xp.sin(TWO_PI * X))


FUN = {"ackley": ackley, "rastrigin": rastrigin}


def run(fn, D, lb, ub, N, shift, seeder, spr, full):
    s = xp.asarray(shift)
    fe = {"n": 0}
    def f(X):
        X = xp.atleast_2d(X); fe["n"] += X.shape[0]; return fn(X, s)
    b = _as_bounds([(lb, ub)] * D, xp)
    if seeder == "carry_tiger":
        # N is the RAY count (coverage lever); samples-per-ray small on purpose
        x0 = carry_tiger_seed(None, b, n_rays=N, n_samples_per_ray=spr,
                              use_gpu=USE_GPU, seed=0)
    else:
        x0 = random_seed(b, n=N, use_gpu=USE_GPU, seed=0)    # N restarts (multistart rule)
    pop = int(np.atleast_2d(to_cpu(x0)).shape[0])
    # --full turns on the exploration boosters (Repulse Monkey + Golden Rooster
    # reseeding, and Cannon-Through-the-Sky boundary rescue)
    boosters = dict(reseed_strategy="sunburst", cannon_through_sky=True) if full else {}
    sync(); t0 = time.time()
    r = sticky_hands(f, x0, bounds=b, bank_samples=True, stick_tolerance=1e-3,
                     estimate_widths=False, n_oscillations=3, func_returns_grad=True,
                     **boosters)
    sync(); wall = time.time() - t0
    bp = to_cpu(np.asarray(to_cpu(r["sample_bank"]["positions"])))
    shift_cpu = np.asarray(shift)
    tol = max(0.5, 0.02 * (ub - lb))
    rec = 0.0 if bp.size == 0 else float(np.min(np.abs(bp - shift_cpu).max(axis=1)) < tol)  # L∞
    return rec, fe["n"], wall, pop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--function", default="ackley", choices=list(FUN))
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--seeder", default="random", choices=["random", "carry_tiger"])
    ap.add_argument("--sweep", default="200,1000,5000,20000,100000",
                    help="random: restart counts N.  carry_tiger: RAY counts n_rays.")
    ap.add_argument("--samples-per-ray", type=int, default=3,
                    help="carry_tiger only; kept small (coverage is in the rays, not here)")
    ap.add_argument("--shift-frac", type=float, default=0.4,
                    help="off-center optimum (0 = centered)")
    ap.add_argument("--full", action="store_true",
                    help="turn on ALL exploration boosters: Repulse Monkey + Golden "
                         "Rooster reseeding + Cannon-Through-the-Sky")
    ap.add_argument("--out", default=None,
                    help="output JSON; if omitted, auto-named by function/seeder/config/D so runs never clobber")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"pop_saturation_{args.function}_{args.seeder}_{'full' if args.full else 'base'}_D{args.D}.json"
    fn, (lb, ub) = FUN[args.function], BOX[args.function]
    D = args.D; spr = args.samples_per_ray
    shift = np.random.default_rng(1).uniform(-args.shift_frac * (ub - lb) / 2,
                                             args.shift_frac * (ub - lb) / 2, D)
    vals = [int(p) for p in args.sweep.split(",")]
    tag = "centered" if args.shift_frac == 0 else f"shifted |s|={np.linalg.norm(shift):.1f}"
    lever = "n_rays" if args.seeder == "carry_tiger" else "N restarts"
    cfg = "FULL (reseed + cannon ON)" if args.full else "base (reseed + cannon OFF; HLC on)"
    print(f"Device: {DEV}")
    print(f"{args.function}  D={D}  ({tag})  seeder={args.seeder}  sweeping {lever}  config={cfg}.")
    print("Question: does more COVERAGE recover the off-center basin, and at what wall-cost?\n")
    print("warming up (CUDA context) ...", flush=True); run(fn, D, lb, ub, vals[0], shift, args.seeder, spr, args.full)
    print(f"{lever:>12} {'pop':>10} {'recover':>7} {'FEs':>14} {'wall_s':>9}")
    print("-" * 56)
    results = {"device": DEV, "function": args.function, "D": D, "seeder": args.seeder,
               "shift_frac": args.shift_frac, "full": args.full, "by_sweep": {}}
    for v in vals:
        run(fn, D, lb, ub, v, shift, args.seeder, spr, args.full)   # per-N warmup: compile kernels for THIS shape (untimed)
        rec, fes, wall, pop = run(fn, D, lb, ub, v, shift, args.seeder, spr, args.full)
        print(f"{v:>12,} {pop:>10,} {rec:>7.2f} {fes:>14,} {wall:>9.3f}")
        results["by_sweep"][str(v)] = dict(pop=int(pop), recover=rec, fes=int(fes), wall_s=wall)
        json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
