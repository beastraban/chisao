#!/usr/bin/env python3
r"""
portfolio_recovery.py — run ChiSao with anti-convergence ON and OFF from the same
seed, union the certified peaks, and report recovery of Full, anti-off, and the
UNION across the Group-A suite at d = 8,16,32,64.

Motivation: the ablation shows Schwefel and Lévy fail on OPPOSITE settings of
anti-convergence (Ph.6). A device-resident optimizer can run both configurations
for a constant-factor cost and keep whichever finds the better mode; because the
failures are disjoint, the union should recover more than either config alone.

USAGE (run where sfu_benchmark.py lives, same as ablation.py):
    python portfolio_recovery.py --source-dir . --dims 8,16,32,64 --trials 5
"""
import argparse, importlib.util, inspect, json, os, warnings
import numpy as np
try:
    import cupy as cp
    try:
        GPU = cp.cuda.runtime.getDeviceCount() > 0   # imported but no visible device (CVD=-1) -> CPU
    except Exception:
        GPU = False
    if not GPU:
        cp = np
except Exception:
    cp = np; GPU = False; warnings.warn("CuPy not available; running on CPU.")

GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]


def load_sfu(sd):
    spec = importlib.util.spec_from_file_location("sfu", os.path.join(os.path.abspath(sd), "sfu_benchmark.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def gpu_clear():
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def peaks_of(r):
    p = r.get("peaks")
    if p is None:
        return None
    p = np.atleast_2d(np.asarray(p.get() if hasattr(p, "get") else p))
    return p if p.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=".")
    ap.add_argument("--dims", default="8,16,32,64")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--functions", default=",".join(GROUP_A))
    ap.add_argument("--out", default="portfolio_recovery.json")
    ap.add_argument("--scale-invariant", action="store_true", dest="scale_invariant",
                    help="run both configs with the affine-invariant gate on")
    args = ap.parse_args()

    sfu = load_sfu(args.source_dir); sh = sfu.load_package(args.source_dir)
    sig = set(inspect.signature(sh).parameters)
    dims = [int(d) for d in args.dims.split(",")]
    funcs = [f for f in args.functions.split(",") if f in GROUP_A]

    def base_for(bnd):
        b = dict(method="lbfgs", n_converge=10, n_anticonverge=5, n_oscillations=3,
                 stick_tolerance=1e-3, reseed_strategy="sunburst", cannon_through_sky=True,
                 cloud_enabled=True, bounds=bnd, estimate_widths=False, verbose=False)
        return {k: v for k, v in b.items() if k in sig or k == "bounds"}

    results = {}
    for fname in funcs:
        cfg = sfu.FUNC_REGISTRY[fname]; results[fname] = {}
        for D in dims:
            bnd = sfu.get_bnd(cfg, D); base = base_for(bnd)
            if args.scale_invariant and "scale_invariant" in sig:
                base["scale_invariant"] = True
            full, anti, uni = [], [], []
            for t in range(args.trials):
                seed = 1000 + t; np.random.seed(seed)
                if GPU: cp.random.seed(seed)
                x0 = sfu.seed_carry_tiger(cfg, D, seed)          # same seed for both configs
                pF = peaks_of(sh(cfg["func"], x0, **base)); gpu_clear()
                b2 = dict(base); b2["n_anticonverge"] = 0
                pA = peaks_of(sh(cfg["func"], x0, **b2)); gpu_clear()
                parts = [p for p in (pF, pA) if p is not None]
                pU = np.vstack(parts) if parts else None
                full.append(int(sfu.recovered(pF, cfg, D)))
                anti.append(int(sfu.recovered(pA, cfg, D)))
                uni.append(int(sfu.recovered(pU, cfg, D)))
            results[fname][D] = dict(full=float(np.mean(full)), anti_off=float(np.mean(anti)),
                                     union=float(np.mean(uni)))
            print(f"{fname:>14} d={D:>3} | Full {np.mean(full)*100:>3.0f}%   -anti {np.mean(anti)*100:>3.0f}%"
                  f"   UNION {np.mean(uni)*100:>3.0f}%", flush=True)
            json.dump(results, open(args.out, "w"), indent=2)

    print(f"\n=== UNION recovery (%) — portfolio of Full and anti-off, {args.trials} trials ===")
    print(f"{'function':>14} " + " ".join(f"d={d:>3}" for d in dims))
    for fname in funcs:
        print(f"{fname:>14} " + " ".join(f"{results[fname][d]['union']*100:>5.0f}" for d in dims))


if __name__ == "__main__":
    main()
