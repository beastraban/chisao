#!/usr/bin/env python3
r"""
parity_lbfgs.py -- memetically-fair single-global recovery parity.
==================================================================
Backs the manuscript sentence "single-global recovery quality is not the claim;
a memetically fair baseline matches it." The fair comparison is memetic-vs-memetic:
each classical baseline is run to CONVERGENCE and then its best solution is polished
with the SAME L-BFGS ChiSao uses. No FE cap -- the baselines get every chance, so a
tie is honest and unattackable.

Scored on SINGLE-GLOBAL recovery only (did the method land the global optimum within
the function's L_infinity tolerance) -- NOT mode-cataloguing, which stays ChiSao's
separate claim. Scoring reuses sfu_benchmark.recovered() (L_infinity, cfg['tol']) so
the numbers are directly comparable to the paper's recovery tables.

ChiSao's own single-global column is read from an existing sfu_benchmark_{cpu,gpu}.json
(its 'recovered' flag is already single-global-in-L_infinity), so this script only runs
the polished baselines.

Baselines (all + L-BFGS-B polish, jac from the analytic/finite-diff gradient):
  * CMA-ES        -> run to convergence, polish es.result.xbest
  * DE            -> scipy differential_evolution(polish=True)  (built-in L-BFGS-B)
  * multistart    -> N random starts, L-BFGS-B each, keep best   (the strongest memetic baseline)

CPU-only by construction (baselines are CPU libraries). Resume-safe: reloads its JSON
and skips finished (function, D, method) cells, checkpointing after every cell.

USAGE
    set PYTHONPATH=D:\Dropbox\chisao\src
    python -u parity_lbfgs.py --dims 2,8,32,64 --trials 10 \
        --chisao-json sfu_benchmark_cpu.json --out parity_lbfgs.json
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only, ALWAYS: baselines are CPU libs and
# recovery quality is device-independent. Hard-set (not setdefault) so a lingering
# CUDA_VISIBLE_DEVICES=0 from a prior GPU run in the same shell can't force GPU mode,
# which would feed NumPy scalars into GPU-mode objectives and break the baselines.

_HERE = os.path.dirname(os.path.abspath(__file__))

# Group A (scalable multimodal) -- the sharp single-global set, matching the paper's tables.
GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]


def load_sfu():
    path = os.path.join(_HERE, "sfu_benchmark.py")
    spec = importlib.util.spec_from_file_location("sfu_benchmark", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def neg_val_grad(x, cfg):
    """Return (-value, -grad) at a single point x, for MINIMIZATION of -f.
    Handles (value, grad) or value-only returns, numpy or cupy."""
    X = np.atleast_2d(np.asarray(x, dtype=np.float64))
    r = cfg["func"](X)
    if isinstance(r, (tuple, list)):
        v = r[0]; g = r[1] if len(r) > 1 else None
    else:
        v = r; g = None
    v = np.asarray(v.get() if hasattr(v, "get") else v).ravel()
    fval = -float(v[0])
    if g is not None:
        g = np.asarray(g.get() if hasattr(g, "get") else g).reshape(-1)
        return fval, -g
    return fval, None


def bounds_list(sfu, cfg, D):
    b = sfu.get_bnd(cfg, D)
    b = np.asarray(b.get() if hasattr(b, "get") else b)
    return [(float(b[i, 0]), float(b[i, 1])) for i in range(D)]


def clip_to(x, bnds):
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])
    return np.clip(np.asarray(x, float), lo, hi)


def lbfgs_polish(x0, cfg, bnds):
    """Polish a candidate with L-BFGS-B on -f (the same local optimizer ChiSao uses)."""
    from scipy.optimize import minimize
    x0 = clip_to(x0, bnds)
    _, g = neg_val_grad(x0, cfg)
    use_jac = g is not None
    fun = (lambda z: neg_val_grad(z, cfg)) if use_jac else (lambda z: neg_val_grad(z, cfg)[0])
    try:
        res = minimize(fun, x0, method="L-BFGS-B", jac=use_jac, bounds=bnds,
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        return res.x
    except Exception:
        return x0


def run_cmaes(cfg, D, bnds, seed):
    import cma
    lb = [b[0] for b in bnds]; ub = [b[1] for b in bnds]
    rng = np.random.default_rng(seed)
    x0 = rng.uniform([b[0] for b in bnds], [b[1] for b in bnds])
    sigma0 = float(np.mean([hi - lo for lo, hi in bnds])) / 4.0
    es = cma.CMAEvolutionStrategy(list(x0), sigma0,
        {"bounds": [lb, ub], "maxiter": 100000, "tolfun": 1e-9, "tolx": 1e-9,
         "seed": int(seed) % (2**31), "verbose": -9})
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [neg_val_grad(s, cfg)[0] for s in sols])
    xbest = np.asarray(es.result.xbest, dtype=np.float64)
    return lbfgs_polish(xbest, cfg, bnds)


def run_de(cfg, D, bnds, seed):
    from scipy.optimize import differential_evolution
    res = differential_evolution(lambda z: neg_val_grad(z, cfg)[0], bnds,
                                 maxiter=200, tol=1e-9, polish=True, seed=int(seed) % (2**31))
    return np.asarray(res.x, dtype=np.float64)   # polish=True already does L-BFGS-B


def run_multistart(cfg, D, bnds, seed, n_starts=64):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])
    best_x, best_f = None, np.inf
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi)
        xp = lbfgs_polish(x0, cfg, bnds)
        fx = neg_val_grad(xp, cfg)[0]
        if fx < best_f:
            best_f, best_x = fx, xp
    return best_x


METHODS = {"cma_lbfgs": run_cmaes, "de_lbfgs": run_de, "multistart_lbfgs": run_multistart}


def load_chisao_column(path):
    """Best-seeder single-global recovery per (fname, D) from an sfu_benchmark JSON."""
    col = {}
    if not path or not os.path.exists(path):
        return col
    try:
        d = json.load(open(path))
    except Exception:
        return col
    for f, dd in d.items():
        col[f] = {}
        for dk, seeders in dd.items():
            rates = [s["rate"] for s in seeders.values()
                     if isinstance(s, dict) and "rate" in s]
            if rates:
                col[f][str(dk)] = max(rates)      # best seeder = ChiSao's single-global recovery
    return col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="2,8,32,64")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--functions", default=",".join(GROUP_A))
    ap.add_argument("--multistart", type=int, default=64, help="restarts for the multistart+L-BFGS baseline")
    ap.add_argument("--chisao-json", default="sfu_benchmark_cpu.json",
                    help="existing sfu_benchmark JSON to read ChiSao's single-global column from")
    ap.add_argument("--out", default="parity_lbfgs.json")
    args = ap.parse_args()

    sfu = load_sfu()
    sfu.load_package(_HERE)                       # ensure the patched package is importable
    dims = [int(x) for x in args.dims.split(",")]
    funcs = [f for f in args.functions.split(",") if f in sfu.FUNC_REGISTRY]
    n = args.trials
    out = os.path.join(_HERE, args.out)

    # resume
    results = {}
    if os.path.exists(out):
        try:
            results = json.load(open(out))
            done = sum(1 for f in results for dk in results[f] for mk, v in results[f][dk].items()
                       if mk in METHODS and isinstance(v, dict) and v.get("n_trials") == n)
            print(f"[resume] loaded {out}: {done} completed baseline cell(s) will be skipped", flush=True)
        except Exception as e:
            print(f"[resume] could not read {out} ({e}); starting fresh", flush=True)
            results = {}

    chisao_col = load_chisao_column(os.path.join(_HERE, args.chisao_json))

    def save():
        json.dump(results, open(out, "w"), indent=2)

    print(f"Memetic-parity (single-global, L_infinity). {len(funcs)} functions x {dims} x {n} trials.")
    print(f"{'function':<18} {'D':>4} {'chisao':>7} {'cma+lb':>7} {'de+lb':>7} {'multi+lb':>9}")
    print("-" * 60)
    for fname in funcs:
        cfg = sfu.FUNC_REGISTRY[fname]
        results.setdefault(fname, {})
        for D in dims:
            dk = str(D)
            cell = results[fname].setdefault(dk, {})
            # ChiSao column (read-through, from existing JSON)
            cell["chisao"] = {"rate": chisao_col.get(fname, {}).get(dk), "source": args.chisao_json}
            bnds = bounds_list(sfu, cfg, D)
            for mk, fn in METHODS.items():
                if isinstance(cell.get(mk), dict) and cell[mk].get("n_trials") == n:
                    continue                     # resume skip
                succ = 0
                for t in range(n):
                    seed = 1000 + t
                    try:
                        if mk == "multistart_lbfgs":
                            x = fn(cfg, D, bnds, seed, n_starts=args.multistart)
                        else:
                            x = fn(cfg, D, bnds, seed)
                        succ += int(sfu.recovered(np.atleast_2d(x), cfg, D))
                    except Exception as e:
                        print(f"  [ERR {fname} d={D} {mk} t={t}] {e}", flush=True)
                cell[mk] = {"rate": succ / n, "n_trials": n}
                save()
            def _fmt(v): return "  -  " if v is None else f"{v*100:>5.0f}"
            print(f"{fname:<18} {D:>4} {_fmt(cell['chisao']['rate']):>7} "
                  f"{cell['cma_lbfgs']['rate']*100:>6.0f}% {cell['de_lbfgs']['rate']*100:>6.0f}% "
                  f"{cell['multistart_lbfgs']['rate']*100:>8.0f}%", flush=True)
    save()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
