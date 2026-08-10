#!/usr/bin/env python3
r"""
parity_lbfgs.py -- memetically-fair single-global recovery parity, with a wall-cap.
==================================================================================
Backs the manuscript sentence "single-global recovery quality is not the claim;
a memetically fair baseline matches it." The comparison is memetic-vs-memetic:
each classical baseline is run to CONVERGENCE and then its best solution is polished
with the SAME L-BFGS ChiSao uses.

Two regimes, selected by --wall-cap:
  * no cap (default): baselines run to convergence at unbounded budget -> the honest
    low/moderate-d parity table (matches the paper's recovery-table dimensions).
  * --wall-cap SECONDS: each baseline must COMPLETE its run (CMA-ES/DE converge;
    multistart finish its restarts) within SECONDS, else the cell is marked
    "timeout" (infeasible) and displayed as "cap". Once a method times out at a
    dimension it is marked infeasible for every HIGHER dimension of that function
    (monotone: O(d^3)/O(N^2) cost only grows), so this draws the feasibility wall
    directly inside the table -- the high-dimensional non-degeneracy edge.

Scored on SINGLE-GLOBAL recovery only (global optimum within the function's
L_infinity tolerance), reusing sfu_benchmark.recovered() so the numbers match the
paper's recovery tables. ChiSao's own column is read from an existing
sfu_benchmark_{cpu,gpu}.json; where a dimension is absent there (e.g. d>64) it shows
"n/a" -- ChiSao's high-d recovery is documented separately (shifted Rastrigin to
d=2048, section High-Dimensional Recovery).

CPU-only by construction. Resume-safe (per (function, D, method) cell).

USAGE
    set PYTHONPATH=D:\Dropbox\chisao\src
    python -u parity_lbfgs.py --dims 2,8,32,64 --trials 10 \
        --chisao-json sfu_benchmark_cpu.json --out parity_lbfgs.json
    # high-dimensional feasibility-wall run:
    python -u parity_lbfgs.py --dims 64,128,256,512 --trials 5 --wall-cap 120 \
        --chisao-json sfu_benchmark_cpu.json --out parity_highdim.json
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only, ALWAYS: baselines are CPU libs and
# recovery quality is device-independent. Hard-set (not setdefault) so a lingering
# CUDA_VISIBLE_DEVICES=0 from a prior GPU run in the same shell can't force GPU mode,
# which would feed NumPy scalars into GPU-mode objectives and break the baselines.

_HERE = os.path.dirname(os.path.abspath(__file__))
GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]


def load_sfu():
    path = os.path.join(_HERE, "sfu_benchmark.py")
    spec = importlib.util.spec_from_file_location("sfu_benchmark", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def neg_val_grad(x, cfg):
    """(-value, -grad) at a single point, for MINIMIZATION of -f. Handles
    (value, grad) or value-only returns, numpy or cupy."""
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


# Each runner returns (x_or_None, completed_bool). completed=False => hit the wall-cap.
def run_cmaes(cfg, D, bnds, seed, cap=None):
    import cma
    lb = [b[0] for b in bnds]; ub = [b[1] for b in bnds]
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lb, ub); sigma0 = float(np.mean([hi - lo for lo, hi in bnds])) / 4.0
    es = cma.CMAEvolutionStrategy(list(x0), sigma0,
        {"bounds": [lb, ub], "maxiter": 100000, "tolfun": 1e-9, "tolx": 1e-9,
         "seed": int(seed) % (2**31), "verbose": -9})
    t0 = time.time()
    while not es.stop():
        if cap is not None and time.time() - t0 > cap:
            return None, False                       # did not converge within budget
        sols = es.ask()
        es.tell(sols, [neg_val_grad(s, cfg)[0] for s in sols])
    xbest = np.asarray(es.result.xbest, dtype=np.float64)
    return lbfgs_polish(xbest, cfg, bnds), True


def run_de(cfg, D, bnds, seed, cap=None):
    from scipy.optimize import differential_evolution
    t0 = time.time(); flag = {"to": False}
    def cb(xk, convergence=None):
        if cap is not None and time.time() - t0 > cap:
            flag["to"] = True; return True           # stop DE early
        return False
    res = differential_evolution(lambda z: neg_val_grad(z, cfg)[0], bnds,
                                 maxiter=200, tol=1e-9, polish=True,
                                 seed=int(seed) % (2**31), callback=cb)
    if flag["to"]:
        return None, False
    return np.asarray(res.x, dtype=np.float64), True   # polish=True already L-BFGS-polishes


def run_multistart(cfg, D, bnds, seed, cap=None, n_starts=64):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])
    best_x, best_f = None, np.inf; t0 = time.time()
    for i in range(n_starts):
        if cap is not None and time.time() - t0 > cap:
            return None, False                       # could not finish its restarts in budget
        xp = lbfgs_polish(rng.uniform(lo, hi), cfg, bnds)
        fx = neg_val_grad(xp, cfg)[0]
        if fx < best_f:
            best_f, best_x = fx, xp
    return best_x, True


METHODS = {"cma_lbfgs": run_cmaes, "de_lbfgs": run_de, "multistart_lbfgs": run_multistart}


def load_chisao_column(path):
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
            rates = [s["rate"] for s in seeders.values() if isinstance(s, dict) and "rate" in s]
            if rates:
                col[f][str(dk)] = max(rates)          # best seeder = ChiSao single-global recovery
    return col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="2,8,32,64")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--functions", default=",".join(GROUP_A))
    ap.add_argument("--multistart", type=int, default=64)
    ap.add_argument("--wall-cap", type=float, default=None, dest="wall_cap",
                    help="per-run wall-clock budget (s); a baseline that cannot COMPLETE within it "
                         "is marked infeasible ('cap') and skipped for all higher dimensions")
    ap.add_argument("--chisao-json", default="sfu_benchmark_cpu.json")
    ap.add_argument("--out", default="parity_lbfgs.json")
    args = ap.parse_args()

    sfu = load_sfu()
    sfu.load_package(_HERE)
    dims = [int(x) for x in args.dims.split(",")]
    funcs = [f for f in args.functions.split(",") if f in sfu.FUNC_REGISTRY]
    n = args.trials
    out = os.path.join(_HERE, args.out)

    # resume
    results = {}
    if os.path.exists(out):
        try:
            results = json.load(open(out))
            print(f"[resume] loaded {out}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {out} ({e}); starting fresh", flush=True)
            results = {}

    chisao_col = load_chisao_column(os.path.join(_HERE, args.chisao_json))

    def done(cell, mk):
        v = cell.get(mk)
        return isinstance(v, dict) and (v.get("status") == "timeout" or v.get("n_trials") == n)

    def save():
        json.dump(results, open(out, "w"), indent=2)

    cap = args.wall_cap
    print(f"Memetic-parity (single-global, L_infinity). {len(funcs)} functions x {dims} x {n} trials."
          + (f"  wall-cap={cap}s/run." if cap else "  unbounded budget."))
    print(f"{'function':<18} {'D':>4} {'chisao':>7} {'cma+lb':>8} {'de+lb':>8} {'multi+lb':>9}")
    print("-" * 62)
    walled = set()                                   # (function, method) that have hit the cap
    for fname in funcs:
        cfg = sfu.FUNC_REGISTRY[fname]
        results.setdefault(fname, {})
        for D in sorted(dims):
            dk = str(D)
            cell = results[fname].setdefault(dk, {})
            cr = chisao_col.get(fname, {}).get(dk)
            cell["chisao"] = {"rate": cr, "source": args.chisao_json}
            bnds = bounds_list(sfu, cfg, D)
            for mk, fn in METHODS.items():
                if done(cell, mk):
                    if cell[mk].get("status") == "timeout":
                        walled.add((fname, mk))
                    continue
                if (fname, mk) in walled:            # monotone: infeasible at lower d -> infeasible here
                    cell[mk] = {"rate": None, "status": "timeout", "reason": "inherited", "n_trials": n}
                    save(); continue
                succ = 0; timed_out = False
                for t in range(n):
                    seed = 1000 + t
                    try:
                        if mk == "multistart_lbfgs":
                            x, ok = fn(cfg, D, bnds, seed, cap=cap, n_starts=args.multistart)
                        else:
                            x, ok = fn(cfg, D, bnds, seed, cap=cap)
                        if not ok:
                            timed_out = True; break
                        succ += int(sfu.recovered(np.atleast_2d(x), cfg, D))
                    except Exception as e:
                        print(f"  [ERR {fname} d={D} {mk} t={t}] {e}", flush=True)
                if timed_out:
                    cell[mk] = {"rate": None, "status": "timeout", "n_trials": n}
                    walled.add((fname, mk))
                else:
                    cell[mk] = {"rate": succ / n, "status": "ok", "n_trials": n}
                save()
            def fmt(mk):
                v = cell[mk]
                if v.get("status") == "timeout":
                    return "cap"
                return f"{v['rate']*100:.0f}%"
            cs = "n/a" if cell["chisao"]["rate"] is None else f"{cell['chisao']['rate']*100:.0f}"
            print(f"{fname:<18} {D:>4} {cs:>7} {fmt('cma_lbfgs'):>8} {fmt('de_lbfgs'):>8} "
                  f"{fmt('multistart_lbfgs'):>9}", flush=True)
    save()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
