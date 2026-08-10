#!/usr/bin/env python3
r"""
noise_sensitivity.py — HONEST noise-robustness check for ChiSao.

Fixes the two flaws in chisao_benchmark.py's Table 5:
  (1) scoring is L-inf (Chebyshev), never L2;
  (2) stick_tolerance is FIXED and small, NOT slaved to sigma — so the
      localization radius never exceeds the recovery budget.

Additive value noise:  f_noisy(x) = f_clean(x) + N(0, sigma)  per sample.
Config = the SunBURST noise config (method=randcoord, reseed=repulse_monkey,
line_search=armijo, tight stick_tolerance), which is where the real
noise-robustness lives.

USAGE (run where the package sfu_benchmark.py lives, package on PYTHONPATH):
    PYTHONPATH=/path/to/chisao/src python noise_sensitivity.py \
        --source-dir . --funcs rastrigin --dims 10 --trials 10 \
        --sigmas 0,0.1,0.2,0.5,1.0
"""
import argparse, importlib.util, inspect, json, os, warnings
import numpy as np
try:
    import cupy as cp; GPU = True
except ImportError:
    cp = np; GPU = False; warnings.warn("CuPy not available; running on CPU.")


def load_sfu(sd):
    spec = importlib.util.spec_from_file_location(
        "sfu_benchmark", os.path.join(os.path.abspath(sd), "sfu_benchmark.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def peaks_of(r):
    p = r.get("peaks") if isinstance(r, dict) else r
    if p is None:
        return None
    p = np.atleast_2d(np.asarray(p.get() if hasattr(p, "get") else p))
    return p if p.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=".")
    ap.add_argument("--funcs", default="rastrigin")
    ap.add_argument("--dims", default="10")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--sigmas", default="0,0.1,0.2,0.5,1.0")
    ap.add_argument("--stick", type=float, default=1e-3,
                    help="FIXED stick tolerance, decoupled from sigma")
    ap.add_argument("--scale-invariant", action="store_true", dest="scale_invariant")
    ap.add_argument("--out", default="noise_sensitivity.json")
    args = ap.parse_args()

    sfu = load_sfu(args.source_dir)
    sh = sfu.load_package(args.source_dir)
    sig = set(inspect.signature(sh).parameters)
    dims = [int(d) for d in args.dims.split(",")]
    funcs = [f for f in args.funcs.split(",") if f in sfu.FUNC_REGISTRY]
    sigmas = [float(s) for s in args.sigmas.split(",")]

    # SunBURST noise config, filtered to what the loaded copy accepts
    cfg_base = dict(method="randcoord", n_converge=10, n_anticonverge=5,
                    n_oscillations=3, stick_tolerance=args.stick,
                    reseed_strategy="sunburst", line_search="armijo",
                    cloud_enabled=True, estimate_widths=False, verbose=False)
    if args.scale_invariant:
        cfg_base["scale_invariant"] = True

    results = {}
    for fname in funcs:
        fcfg = sfu.FUNC_REGISTRY[fname]; results[fname] = {}
        for D in dims:
            bnd = sfu.get_bnd(fcfg, D)
            base = {k: v for k, v in {**cfg_base, "bounds": bnd}.items()
                    if k in sig or k == "bounds"}
            results[fname][D] = {}
            for sigma in sigmas:
                recs = []
                for t in range(args.trials):
                    seed = 1000 + t
                    np.random.seed(seed)
                    if GPU: cp.random.seed(seed)
                    clean = fcfg["func"]

                    def noisy(x, _s=sigma, _f=clean):
                        v = _f(x)
                        n = x.shape[0]
                        xp = cp if (GPU and hasattr(x, "get")) else np
                        return v + xp.asarray(
                            np.random.normal(0, _s, n).astype(v.dtype if hasattr(v, "dtype") else np.float64))

                    x0 = sfu.seed_carry_tiger(fcfg, D, seed)
                    try:
                        p = peaks_of(sh(noisy, x0, **base))
                        rec = int(sfu.recovered(p, fcfg, D))   # L-inf, sfu's own scorer
                    except Exception as e:
                        print(f"  [ERR {fname} d={D} s={sigma} t={t}] {e}", flush=True)
                        rec = 0
                    recs.append(rec)
                rate = float(np.mean(recs))
                results[fname][D][sigma] = rate
                print(f"{fname:>12} d={D:>3} sigma={sigma:<4} | recovery {rate*100:>5.0f}%  "
                      f"({int(np.sum(recs))}/{args.trials})", flush=True)
                json.dump(results, open(args.out, "w"), indent=2)

    print(f"\n=== L-inf noise robustness ({args.trials} trials, stick={args.stick}) ===")
    print(f"{'function':>12} d " + " ".join(f"s={s:<4}" for s in sigmas))
    for fname in funcs:
        for D in dims:
            print(f"{fname:>12} {D:>3} " +
                  " ".join(f"{results[fname][D][s]*100:>5.0f}" for s in sigmas))


if __name__ == "__main__":
    main()
