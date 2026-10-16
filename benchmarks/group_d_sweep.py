"""
group_d_sweep.py
================

SFU Group D (14 fixed low-dimensional structured functions), five methods,
scored against the KNOWN global optimum in BOTH L_inf and L_2.

Methods
    chisao_random       public `chisao` package (sticky_hands), random seeding
    chisao_carry_tiger  public `chisao` package (sticky_hands), carry_tiger seeding
    DE                  SciPy differential_evolution (default strategy)
    BH                  SciPy basinhopping, restarted until budget is spent
    CMA-ES              `cma` with IPOP-style restarts until budget is spent

Design
    * Functions, bounds, per-function tolerances, seeders and ChiSao settings are
      imported from sfu_benchmark.py (same folder) so Group D uses exactly the code
      behind the other tables.
    * Recovery = the closest returned peak (ChiSao) / best-seen point (baselines)
      lies within `tol` of a global optimum, in L_inf and, separately, in L_2.
      The SAME radius `tol` is used for both norms (cfg['tol'] from sfu_benchmark).
      Raw per-trial distances are stored, so the tables can be re-scored at any
      tolerance with --rescore, without rerunning.
    * Budgets: each baseline is capped at the number of function evaluations that
      ChiSao spent in the same trial (the larger of its two seeders), counted by a
      wrapper that counts rows of every batched call (finite-difference gradient
      evaluations included). Baselines return the best point seen when the cap hits.
    * Global optima: analytic locations, polished with L-BFGS-B where well
      conditioned; discovered by dense multistart for hartmann4/6 and shekel.
      Printed at startup for inspection and saved in the output JSON.
    * Checkpointing (default ON): the JSON is rewritten atomically after EVERY
      trial; a rerun with the same --out resumes where it stopped. Seeds are
      deterministic (crc32), so a resumed run reproduces an uninterrupted one.

Run (from the benchmarks/ folder of the repo, GPU machine):
    python group_d_sweep.py                       # full sweep, 10 trials
    python group_d_sweep.py --trials 1 --funcs booth matyas   # quick check
    python group_d_sweep.py --rescore --out group_d_results.json --tol-scale 0.5
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfu_benchmark as sfu            # noqa: E402  (repo's own benchmark module)

from scipy.optimize import basinhopping, differential_evolution, minimize  # noqa: E402

try:
    import cma
    CMA_OK = True
except Exception:
    CMA_OK = False

GROUP_D = ["booth", "matyas", "mccormick", "camel3", "camel6", "beale", "branin",
           "goldstein_price", "hartmann3", "hartmann4", "hartmann6", "shekel",
           "colville", "powell"]
CHISAO_METHODS = ["chisao_random", "chisao_carry_tiger"]
BASELINES = ["DE", "BH", "CMA-ES"]
METHODS = CHISAO_METHODS + BASELINES


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def to_host(a):
    """Device array -> NumPy (no-op for NumPy)."""
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


def cfg_dims(cfg):
    if cfg.get("fixed_D"):
        return [cfg["fixed_D"]]
    return list(cfg.get("dims_override", []))


def bounds_np(cfg, D):
    asym = cfg.get("bounds_asym")
    if asym is not None:
        return np.array([[b[0], b[1]] for b in asym], dtype=np.float64)
    lo, hi = sfu._lohi(cfg, D)
    return np.array([[lo, hi]] * D, dtype=np.float64)


def f_host(func, X):
    """Evaluate the (maximisation-form) benchmark function on host arrays."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.asarray(to_host(func(X))).reshape(-1)


def trial_seed(fname, D, t):
    return zlib.crc32(f"{fname}|{D}|{t}".encode()) & 0x7FFFFFFF


# ------------------------------------------------------------------
# global optima (reference)
# ------------------------------------------------------------------
PI = np.pi
ANALYTIC = {
    "booth":           [[1.0, 3.0]],
    "matyas":          [[0.0, 0.0]],
    "mccormick":       [[-0.54719, -1.54719]],
    "camel3":          [[0.0, 0.0]],
    "camel6":          [[0.0898, -0.7126], [-0.0898, 0.7126]],
    "beale":           [[3.0, 0.5]],
    "branin":          [[-PI, 12.275], [PI, 2.275], [9.42478, 2.475]],
    "goldstein_price": [[0.0, -1.0]],
    "hartmann3":       [[0.114614, 0.555649, 0.852547]],
}
EXACT = {"colville", "powell"}            # flat / ill-conditioned: use exact x*
DISCOVER = {"hartmann4", "hartmann6", "shekel"}


def _polish(func, x0, lo, hi):
    res = minimize(lambda x: -f_host(func, x)[0], x0, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options=dict(ftol=1e-15, gtol=1e-12, maxiter=500))
    return res.x, -res.fun


def reference_optima(fname, cfg, D, n_starts=3000, seed=0):
    b = bounds_np(cfg, D)
    lo, hi = b[:, 0], b[:, 1]
    func = cfg["func"]
    if fname == "colville":
        xs = [np.ones(D)]
    elif fname == "powell":
        xs = [np.zeros(D)]
    elif fname in ANALYTIC:
        xs = []
        for s in ANALYTIC[fname]:
            s = np.asarray(s, dtype=np.float64)
            xp_, fp_ = _polish(func, s, lo, hi)
            xs.append(xp_ if np.max(np.abs(xp_ - s)) < 1e-2 else s)
    else:                                   # discover numerically
        rng = np.random.default_rng(seed)
        pts = []
        for _ in range(n_starts):
            x0 = rng.uniform(lo, hi)
            x, f = _polish(func, x0, lo, hi)
            pts.append((f, x))
        best = max(p[0] for p in pts)
        xs = []
        for f, x in sorted(pts, key=lambda p: -p[0]):
            if f < best - 1e-7 * (1 + abs(best)):
                break
            if all(np.max(np.abs(x - y)) > 1e-3 for y in xs):
                xs.append(x)
    xs = np.array(xs, dtype=np.float64)
    fvals = f_host(func, xs)
    return xs, fvals


# ------------------------------------------------------------------
# distance scoring
# ------------------------------------------------------------------
def score_points(P, xstar):
    """Closest point in P to any global optimum: returns (d_inf, d_2).

    Each norm is minimised independently over (point, optimum) pairs."""
    P = np.atleast_2d(P)
    if P.size == 0:
        return float("inf"), float("inf")
    diff = P[:, None, :] - xstar[None, :, :]          # [n_pts, n_opt, D]
    d_inf = np.max(np.abs(diff), axis=2).min()
    d_2 = np.sqrt((diff ** 2).sum(axis=2)).min()
    return float(d_inf), float(d_2)


# ------------------------------------------------------------------
# evaluation counting / budgets
# ------------------------------------------------------------------
class Budget(Exception):
    pass


class CountingFunc:
    """Wrap a batched maximisation-form function; count evaluated rows."""

    def __init__(self, func):
        self.func = func
        self.n = 0

    def __call__(self, x):
        self.n += 1 if getattr(x, "ndim", 2) == 1 else int(x.shape[0])
        return self.func(x)


class BestSeen:
    """Single-point objective (minimisation form) with budget + best-so-far."""

    def __init__(self, func, budget):
        self.func, self.budget = func, budget
        self.n, self.best_f, self.best_x = 0, np.inf, None

    def __call__(self, x):
        if self.budget is not None and self.n >= self.budget:
            raise Budget()
        self.n += 1
        x = np.asarray(x, dtype=np.float64)
        v = -float(f_host(self.func, x)[0])
        if v < self.best_f:
            self.best_f, self.best_x = v, x.copy()
        return v


# ------------------------------------------------------------------
# method runners   (each returns dict with points / n_evals / wall / error)
# ------------------------------------------------------------------
def run_chisao(sticky_hands, fname, cfg, D, seed, seeder):
    counted = CountingFunc(cfg["func"])
    cfg_c = dict(cfg)
    cfg_c["func"] = counted
    t0 = time.perf_counter()
    peaks, wall = sfu.run_one(sticky_hands, fname, cfg_c, D, seed, seeder=seeder)
    wall = time.perf_counter() - t0
    if peaks is None:
        return dict(points=np.zeros((0, D)), n_evals=counted.n, wall=wall,
                    error="no_peaks_or_exception")
    return dict(points=to_host(peaks).reshape(-1, D), n_evals=counted.n,
                wall=wall, error=None)


def run_de(fname, cfg, D, seed, budget):
    b = bounds_np(cfg, D)
    obj = BestSeen(cfg["func"], budget)
    t0 = time.perf_counter()
    err = None
    try:
        differential_evolution(obj, bounds=list(map(tuple, b)), seed=seed,
                               maxiter=1000, tol=1e-8, polish=True)
    except Budget:
        pass
    except Exception as e:                           # noqa: BLE001
        err = repr(e)
    pts = obj.best_x[None, :] if obj.best_x is not None else np.zeros((0, D))
    return dict(points=pts, n_evals=obj.n, wall=time.perf_counter() - t0, error=err)


def run_bh(fname, cfg, D, seed, budget):
    b = bounds_np(cfg, D)
    lo, hi = b[:, 0], b[:, 1]
    rng = np.random.default_rng(seed)
    obj = BestSeen(cfg["func"], budget)
    t0 = time.perf_counter()
    err = None
    try:
        while budget is None or obj.n < budget:
            x0 = rng.uniform(lo, hi)
            basinhopping(obj, x0, niter=50, seed=int(rng.integers(1, 2**31)),
                         minimizer_kwargs=dict(method="L-BFGS-B",
                                               bounds=list(zip(lo, hi))))
            if budget is None:
                break
    except Budget:
        pass
    except Exception as e:                           # noqa: BLE001
        err = repr(e)
    pts = obj.best_x[None, :] if obj.best_x is not None else np.zeros((0, D))
    return dict(points=pts, n_evals=obj.n, wall=time.perf_counter() - t0, error=err)


def run_cma(fname, cfg, D, seed, budget):
    if not CMA_OK:
        return dict(points=np.zeros((0, D)), n_evals=0, wall=0.0, error="cma_missing")
    b = bounds_np(cfg, D)
    lo, hi = b[:, 0], b[:, 1]
    rng = np.random.default_rng(seed)
    obj = BestSeen(cfg["func"], budget)
    sigma0 = float(np.max(hi - lo) / 4.0)
    base_pop = 4 + int(3 * np.log(D))
    t0 = time.perf_counter()
    err = None
    try:
        k = 0
        while budget is None or obj.n < budget:
            es = cma.CMAEvolutionStrategy(
                rng.uniform(lo, hi).tolist(), sigma0,
                dict(bounds=[lo.tolist(), hi.tolist()], popsize=base_pop * 2 ** k,
                     verbose=-9, seed=int(rng.integers(1, 2**31)),
                     maxfevals=(None if budget is None else max(1, budget - obj.n))))
            es.optimize(obj)
            k += 1
            if budget is None:
                break
    except Budget:
        pass
    except Exception as e:                           # noqa: BLE001
        err = repr(e)
    pts = obj.best_x[None, :] if obj.best_x is not None else np.zeros((0, D))
    return dict(points=pts, n_evals=obj.n, wall=time.perf_counter() - t0, error=err)


# ------------------------------------------------------------------
# checkpoint I/O
# ------------------------------------------------------------------
def save_atomic(path, data, retries=40, wait=0.25):
    """Atomic checkpoint. On Windows, Dropbox/antivirus can briefly lock the
    destination (PermissionError on os.replace), so retry, then fall back to a
    direct overwrite, and never let a failed checkpoint kill a long run."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(wait)
    try:                                    # last resort: non-atomic overwrite
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
        try:
            os.remove(tmp)
        except OSError:
            pass
    except PermissionError:
        print(f"  [warn] checkpoint to {path} blocked (file locked); "
              f"will retry after the next trial", flush=True)


def load_or_new(path, meta):
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        data.setdefault("trials", {})
        data["meta"].update({k: v for k, v in meta.items() if k != "created"})
        print(f"# resuming from {path}: "
              f"{sum(len(v) for v in data['trials'].values())} trials already done")
        return data
    return dict(meta=meta, optima={}, trials={})


# ------------------------------------------------------------------
# summary / rescoring
# ------------------------------------------------------------------
def tol_of(cfg, scale):
    return float(cfg["tol"]) * scale


def summarise(data, tol_scale=1.0, out=sys.stdout):
    reg = sfu.FUNC_REGISTRY
    print(f"\n# Recovery (%)  columns per method: Linf | L2 | mean wall (s)   "
          f"[tol = cfg['tol'] x {tol_scale}]", file=out)
    hdr = f"{'function':<16}{'D':>3}  " + "".join(f"{m:>26}" for m in METHODS)
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for fname in GROUP_D:
        for D in cfg_dims(reg[fname]):
            tol = tol_of(reg[fname], tol_scale)
            cells = []
            for m in METHODS:
                rows = data["trials"].get(f"{fname}|{D}|{m}", [])
                if not rows:
                    cells.append(f"{'--':>26}")
                    continue
                n = len(rows)
                ri = 100.0 * sum(r["d_inf"] <= tol for r in rows) / n
                r2 = 100.0 * sum(r["d_2"] <= tol for r in rows) / n
                w = np.mean([r["wall"] for r in rows])
                cells.append(f"{ri:>7.0f} |{r2:>5.0f} |{w:>6.1f}  ")
            print(f"{fname:<16}{D:>3}  " + "".join(cells), file=out)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--funcs", nargs="+", default=None, choices=GROUP_D)
    ap.add_argument("--out", default="group_d_results.json")
    ap.add_argument("--source-dir", default=None)
    ap.add_argument("--budget-cap", type=int, default=0,
                    help="optional upper cap on the baseline eval budget (0 = none)")
    ap.add_argument("--rescore", action="store_true",
                    help="do not run; re-print the tables from --out")
    ap.add_argument("--tol-scale", type=float, default=1.0,
                    help="multiply cfg['tol'] by this when summarising")
    args = ap.parse_args()

    if args.rescore:
        with open(args.out) as fh:
            summarise(json.load(fh), args.tol_scale)
        return

    sticky_hands = sfu.load_package(args.source_dir)
    funcs = args.funcs or GROUP_D
    reg = sfu.FUNC_REGISTRY

    print("# SFU Group D sweep (public chisao package)")
    print(f"# device  : {'GPU (CuPy)' if sfu.GPU else 'CPU (NumPy)'}")
    print(f"# cma     : {'available' if CMA_OK else 'MISSING - CMA-ES rows will be errors'}")
    print(f"# trials  : {args.trials}   functions: {len(funcs)}   methods: {len(METHODS)}")
    print(f"# output  : {args.out}  (checkpointed after every trial)\n", flush=True)

    meta = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        trials=args.trials,
        recovery_rule=("closest returned peak / best-seen point within cfg['tol'] "
                       "of a known global optimum, in L_inf and in L_2 (same radius)"),
        budget_rule=("baselines capped at max(evals of the two ChiSao runs) in the "
                     "same trial; --budget-cap=%d" % args.budget_cap),
        chisao_params="sfu_benchmark.run_one defaults",
    )
    data = load_or_new(args.out, meta)

    # reference optima
    print("# reference global optima")
    for fname in funcs:
        for D in cfg_dims(reg[fname]):
            key = f"{fname}|{D}"
            if key not in data["optima"]:
                xs, fv = reference_optima(fname, reg[fname], D)
                data["optima"][key] = dict(x=xs.tolist(), f=fv.tolist())
            o = data["optima"][key]
            print(f"  {key:<20} n_opt={len(o['x'])}  f*={max(o['f']):.6f}  "
                  f"tol={reg[fname]['tol']}")
    save_atomic(args.out, data)
    print(flush=True)

    t_start = time.perf_counter()
    for fname in funcs:
        cfg = reg[fname]
        for D in cfg_dims(cfg):
            xstar = np.array(data["optima"][f"{fname}|{D}"]["x"])
            for t in range(args.trials):
                seed = trial_seed(fname, D, t)
                rec = {m: data["trials"].setdefault(f"{fname}|{D}|{m}", [])
                       for m in METHODS}
                for m in METHODS:
                    if len(rec[m]) > t:
                        continue                      # already done (resume)
                    if m in CHISAO_METHODS:
                        r = run_chisao(sticky_hands, fname, cfg, D, seed,
                                       "random" if m == "chisao_random"
                                       else "carry_tiger")
                    else:
                        done = [rec[c][t]["n_evals"] for c in CHISAO_METHODS
                                if len(rec[c]) > t]
                        budget = max(done) if done else None
                        if budget is not None and args.budget_cap:
                            budget = min(budget, args.budget_cap)
                        r = {"DE": run_de, "BH": run_bh, "CMA-ES": run_cma}[m](
                            fname, cfg, D, seed, budget)
                    d_inf, d_2 = score_points(r["points"], xstar)
                    rec[m].append(dict(
                        trial=t, seed=seed, d_inf=d_inf, d_2=d_2,
                        n_peaks=int(len(r["points"])), n_evals=int(r["n_evals"]),
                        wall=float(r["wall"]), error=r["error"]))
                    save_atomic(args.out, data)       # checkpoint after every trial
                print(f"  {fname:<16} D={D:<3} trial {t + 1}/{args.trials} done  "
                      f"[{(time.perf_counter() - t_start) / 60:.1f} min]", flush=True)

    summarise(data, 1.0)
    print(f"\n# elapsed {(time.perf_counter() - t_start) / 60:.1f} min   wrote {args.out}")


if __name__ == "__main__":
    main()
