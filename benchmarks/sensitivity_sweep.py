"""
sensitivity_sweep.py
====================

Fills Appendix C (Hyperparameter Sensitivity) of the ChiSao paper.

For each of 7 Group A benchmark functions, sweeps four hyperparameters
one-at-a-time from defaults, under BOTH paper seeders:
  - "random"      = CarryTigerToMountain(multi_scale=False)
  - "carry_tiger" = CarryTigerToMountain(multi_scale=True)
matching chisao_benchmark.py:255-257. 30 trials per cell.
Produces sensitivity_results.json and per-function tables on stdout.

NOTE: the quality-gate threshold delta (= 0.1) is hardcoded in
chisao.py around line 3179. Sweeping it requires exposing it as a
parameter to sticky_hands; left out of this sweep intentionally.

Total: 7 functions x 12 cells x 30 trials x 2 seeders ~= 5040 trials,
       ~60-90 min on a single RTX 3080.

Run:
    python sensitivity_sweep.py
or with overrides:
    python sensitivity_sweep.py --trials 30 --N 200 --out sensitivity_results.json
    python sensitivity_sweep.py --functions rastrigin,ackley --seeders carry_tiger
"""

import argparse
import importlib
import json
import os
import sys
import time
import gc
from typing import Dict

import numpy as np

try:
    import cupy as cp
    GPU = True
except Exception:
    cp = None
    GPU = False

xp = cp if GPU else np


# ------------------------------------------------------------------
# 1. Load sunburst.utils.chisao.sticky_hands and
#    sunburst.modules.carry_tiger.CarryTigerToMountain.
# ------------------------------------------------------------------
def _add_likely_parents_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for parent in (here, os.path.dirname(here),
                   os.path.dirname(os.path.dirname(here)),
                   os.path.dirname(os.path.dirname(os.path.dirname(here)))):
        if os.path.isdir(os.path.join(parent, "sunburst")) and parent not in sys.path:
            sys.path.insert(0, parent)

_add_likely_parents_to_path()

def load_sticky_hands():
    try:
        return importlib.import_module("sunburst.utils.chisao").sticky_hands
    except ModuleNotFoundError:
        return importlib.import_module("chisao").sticky_hands

def load_carry_tiger():
    try:
        return importlib.import_module("sunburst.modules.carry_tiger").CarryTigerToMountain
    except ModuleNotFoundError:
        return None

sticky_hands = load_sticky_hands()
CarryTiger   = load_carry_tiger()


# ------------------------------------------------------------------
# 2. Benchmark functions (batched, GPU-aware).
#    Each f: array (N, d) -> array (N,) returning LOG-LIKELIHOOD.
#    Definitions match Appendix D of main.tex (all negated for max).
# ------------------------------------------------------------------
def _get_xp(x):
    return cp if GPU and isinstance(x, cp.ndarray) else np

def rastrigin(x):
    _xp = _get_xp(x)
    A, d = 10.0, x.shape[1]
    return -(A * d + _xp.sum(x * x - A * _xp.cos(2 * _xp.pi * x), axis=1))

def ackley(x):
    _xp = _get_xp(x)
    a, b, c, d = 20.0, 0.2, 2 * _xp.pi, x.shape[1]
    s1 = _xp.sum(x * x, axis=1)
    s2 = _xp.sum(_xp.cos(c * x), axis=1)
    return -(-a * _xp.exp(-b * _xp.sqrt(s1 / d)) - _xp.exp(s2 / d) + a + _xp.e)

def schwefel(x):
    _xp = _get_xp(x)
    d = x.shape[1]
    return -(418.9829 * d - _xp.sum(x * _xp.sin(_xp.sqrt(_xp.abs(x))), axis=1))

def styblinski_tang(x):
    _xp = _get_xp(x)
    return -0.5 * _xp.sum(x ** 4 - 16 * x * x + 5 * x, axis=1)

def levy(x):
    _xp = _get_xp(x)
    w  = 1.0 + (x - 1.0) / 4.0
    t1 = _xp.sin(_xp.pi * w[:, 0]) ** 2
    t2 = _xp.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * _xp.sin(_xp.pi * w[:, :-1] + 1) ** 2), axis=1)
    t3 = (w[:, -1] - 1) ** 2 * (1 + _xp.sin(2 * _xp.pi * w[:, -1]) ** 2)
    return -(t1 + t2 + t3)

def griewank(x):
    _xp = _get_xp(x)
    d   = x.shape[1]
    idx = _xp.arange(1, d + 1).astype(x.dtype)
    s1  = _xp.sum(x * x, axis=1) / 4000.0
    s2  = _xp.prod(_xp.cos(x / _xp.sqrt(idx)), axis=1)
    return -(s1 - s2 + 1.0)

def michalewicz(x):
    _xp = _get_xp(x)
    d   = x.shape[1]
    m   = 10
    idx = _xp.arange(1, d + 1).astype(x.dtype)
    return _xp.sum(_xp.sin(x) * _xp.sin(idx * x * x / _xp.pi) ** (2 * m), axis=1)


# ------------------------------------------------------------------
# 3. Function metadata.
#    test_dim matches the paper's "where ChiSao is contested":
#      d=32 for the easy ones, d=8 for Schwefel (only dim it's non-zero).
# ------------------------------------------------------------------
FUNCTIONS: Dict[str, dict] = {
    "rastrigin": dict(
        f=rastrigin, bounds=(-5.12, 5.12), d=32, tol=0.5,
        opt=lambda d: np.zeros(d),
    ),
    "ackley": dict(
        f=ackley, bounds=(-32.768, 32.768), d=32, tol=0.5,
        opt=lambda d: np.zeros(d),
    ),
    "levy": dict(
        f=levy, bounds=(-10.0, 10.0), d=32, tol=0.5,
        opt=lambda d: np.ones(d),
    ),
    "griewank": dict(
        f=griewank, bounds=(-600.0, 600.0), d=32, tol=1.0,
        opt=lambda d: np.zeros(d),
    ),
    "styblinski_tang": dict(
        f=styblinski_tang, bounds=(-5.0, 5.0), d=32, tol=0.2,
        opt=lambda d: np.full(d, -2.903534),
    ),
    "schwefel": dict(
        f=schwefel, bounds=(-500.0, 500.0), d=8, tol=2.0,    # d=8: only dim where it survives
        opt=lambda d: np.full(d, 420.9687),
    ),
    "michalewicz": dict(
        f=michalewicz, bounds=(0.0, float(np.pi)), d=32, tol=0.5,
        opt=lambda d: None,                                    # multiple equiv. optima
    ),
}


# ------------------------------------------------------------------
# 4. Sweep grid (matches the paper's Appendix C table).
# ------------------------------------------------------------------
DEFAULTS = dict(
    n_oscillations  = 3,
    n_anticonverge  = 5,
    anti_momentum   = 0.9,
    cloud_K         = 10,
    # Other params held fixed at paper defaults:
    n_converge      = 10,
    n_cloud         = 3,
    stick_tolerance = 1e-6,
    cloud_enabled   = True,
    reseed_strategy = "repulse_monkey",
    method          = "randcoord",
    line_search     = "armijo",
)

SWEEP_GRID = [
    ("n_oscillations", [1, 3, 5]),
    ("n_anticonverge", [1, 5, 10]),
    ("anti_momentum",  [0.5, 0.9, 0.99]),
    ("cloud_K",        [5, 10, 20]),
]


# ------------------------------------------------------------------
# 5. Trial runner (both seeders route through CarryTiger, differing
#    only in multi_scale, matching chisao_benchmark.py:255-257).
# ------------------------------------------------------------------
def _peaks_to_numpy(peaks):
    if peaks is None:
        return None
    if GPU and isinstance(peaks, cp.ndarray):
        return cp.asnumpy(peaks)
    return np.asarray(peaks)

def _is_recovered(peaks_np, fmeta: dict) -> bool:
    if peaks_np is None or peaks_np.size == 0:
        return False
    opt = fmeta["opt"](fmeta["d"])
    if opt is None:                                  # michalewicz: peak_count > 0
        return True
    d_inf = np.max(np.abs(peaks_np - opt[None, :]), axis=1)
    return bool(np.any(d_inf < fmeta["tol"]))

def _run_via_carry(fmeta, seed, overrides, multi_scale: bool):
    if CarryTiger is None:
        raise RuntimeError("CarryTigerToMountain not importable.")
    np.random.seed(seed)
    if GPU:
        cp.random.seed(seed)
    lo, hi    = fmeta["bounds"]
    d         = fmeta["d"]
    bounds_np = np.array([[lo, hi]] * d, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        tiger = CarryTiger(
            func=fmeta["f"],
            bounds=bounds_np,
            use_gpu=GPU,
            n_oscillations=overrides.get("n_oscillations", DEFAULTS["n_oscillations"]),
            multi_scale=multi_scale,
            stick_tolerance=1e-3,    # matches run_chisao_variant in chisao_benchmark.py
        )
        for k, v in overrides.items():
            if k == "n_oscillations":
                continue
            tiger.chisao_params[k] = v
        peaks, L_peaks, widths = tiger.detect_modes(verbose=False)
    except Exception:
        return False, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    ok   = _is_recovered(_peaks_to_numpy(peaks), fmeta)
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
    return ok, wall

def run_one_trial_random(fmeta, N, seed, overrides):
    return _run_via_carry(fmeta, seed, overrides, multi_scale=False)

def run_one_trial_carry(fmeta, N, seed, overrides):
    return _run_via_carry(fmeta, seed, overrides, multi_scale=True)

SEEDERS = {
    "random":      run_one_trial_random,
    "carry_tiger": run_one_trial_carry,
}


# ------------------------------------------------------------------
# 6. Cell runner.
# ------------------------------------------------------------------
def _purge_memory():
    """Drain CuPy memory pools and force a Python GC pass after each cell."""
    gc.collect()
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

def run_cell(fname, fmeta, seeder, N, n_trials, param, value, base_seed=0):
    runner = SEEDERS[seeder]
    successes, walls = 0, []
    for t in range(n_trials):
        seed = abs(hash((fname, seeder, param, value, t, base_seed))) % 2**31
        ok, wall = runner(fmeta, N, seed, {param: value})
        walls.append(wall)
        if ok:
            successes += 1
    return successes / n_trials * 100.0, float(np.mean(walls))


# ------------------------------------------------------------------
# 7. Main.
# ------------------------------------------------------------------
def _cell_already_done(results, fname, seeder, param, value):
    """True if (fname, seeder, param, value) already has a recorded recovery rate."""
    try:
        cells = results["results"][fname][seeder][param]
    except KeyError:
        return False
    for entry in cells:
        if entry.get("value") == value and entry.get("recovery_pct") is not None:
            return True
    return False


def _load_resume(path: str):
    """Load existing results file for --resume. Returns the results dict
       or None if no usable file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # sanity check structure
        if "results" not in data:
            return None
        return data
    except Exception as e:
        print(f"# could not read existing {path} for resume: {e}", flush=True)
        return None


def _checkpoint(path: str, results: dict):
    """Atomic write: dump to .tmp then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",    type=int, default=30, help="trials per cell")
    ap.add_argument("--N",         type=int, default=200, help="population size (unused after switch to CarryTiger)")
    ap.add_argument("--out",       type=str, default="sensitivity_results.json")
    ap.add_argument("--functions", type=str, default="all",
                    help="comma-separated, or 'all' (default: all 7)")
    ap.add_argument("--seeders",   type=str, default="random,carry_tiger",
                    help="comma-separated subset of {random, carry_tiger}")
    ap.add_argument("--resume",    action="store_true",
                    help="if set, read existing --out file and skip already-completed cells")
    args = ap.parse_args()

    func_names = list(FUNCTIONS.keys()) if args.functions == "all" \
                 else [s.strip() for s in args.functions.split(",")]
    seeders    = [s.strip() for s in args.seeders.split(",")]
    for fn in func_names:
        if fn not in FUNCTIONS:
            sys.exit(f"unknown function: {fn}; available: {list(FUNCTIONS)}")
    for sd in seeders:
        if sd not in SEEDERS:
            sys.exit(f"unknown seeder: {sd}; available: {list(SEEDERS)}")
    if CarryTiger is None:
        sys.exit("CarryTigerToMountain not importable; this sweep requires the sunburst package on sys.path.")

    # --- Resume bookkeeping
    results = None
    if args.resume:
        results = _load_resume(args.out)
        if results is None:
            print(f"# --resume requested but {args.out} not usable; starting fresh", flush=True)
    if results is None:
        results = {"meta": {
            "trials":           args.trials,
            "defaults":         DEFAULTS,
            "delta_quality_gate": "0.1 (HARDCODED in chisao.py:3179, not swept)",
            "sweep_grid":       SWEEP_GRID,
            "seeder_mapping":   {"random": "CarryTiger(multi_scale=False)",
                                 "carry_tiger": "CarryTiger(multi_scale=True)"},
            "functions":        {fn: {k: FUNCTIONS[fn][k] for k in ("bounds","d","tol")}
                                 for fn in func_names},
        }, "results": {}}

    print(f"# ChiSao multi-function sensitivity sweep")
    print(f"# device  : {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")
    print(f"# trials  : {args.trials} per cell")
    print(f"# defaults: {DEFAULTS}")
    print(f"# delta   : 0.1 (HARDCODED, not swept)")
    print(f"# functions: {func_names}")
    print(f"# seeders : {seeders}")
    print(f"# resume  : {args.resume}")
    n_cells = sum(len(vs) for _, vs in SWEEP_GRID)
    print(f"# total cells: {n_cells * len(func_names) * len(seeders)}"
          f" = {n_cells} x {len(func_names)} fns x {len(seeders)} seeders")
    print(f"# estimated trials: {n_cells * len(func_names) * len(seeders) * args.trials}")
    print(flush=True)

    t_start = time.perf_counter()
    n_skipped = 0
    for fname in func_names:
        fmeta = FUNCTIONS[fname]
        results["results"].setdefault(fname, {})
        for seeder in seeders:
            print(f"=== {fname}  (d={fmeta['d']}, tol={fmeta['tol']})  seeder={seeder} ===", flush=True)
            results["results"][fname].setdefault(seeder, {})
            print(f"{'Parameter':<24} {'Value':>8}   {'Recovery %':>10}  {'Mean wall (s)':>14}", flush=True)
            print("-" * 64, flush=True)
            for param, values in SWEEP_GRID:
                results["results"][fname][seeder].setdefault(param, [])
                for v in values:
                    if args.resume and _cell_already_done(results, fname, seeder, param, v):
                        # find and print the existing entry so the log stays complete
                        existing = next((e for e in results["results"][fname][seeder][param]
                                         if e.get("value") == v), None)
                        if existing:
                            tag   = "(default)" if v == DEFAULTS[param] else ""
                            label = f"{param} = {v} {tag}".strip()
                            print(f"{label:<24} {'SKIP':>8}   "
                                  f"{existing['recovery_pct']:>9.1f}%  "
                                  f"{existing['mean_wall_s']:>14.2f}", flush=True)
                        n_skipped += 1
                        continue
                    rate, wall = run_cell(fname, fmeta, seeder,
                                          args.N, args.trials, param, v)
                    tag   = "(default)" if v == DEFAULTS[param] else ""
                    label = f"{param} = {v} {tag}".strip()
                    print(f"{label:<24} {'':>8}   {rate:>9.1f}%  {wall:>14.2f}", flush=True)
                    # Append-or-replace: if a stale entry exists for this value, overwrite it
                    cell = dict(value=v,
                                is_default=v == DEFAULTS[param],
                                recovery_pct=rate,
                                mean_wall_s=wall)
                    existing_idx = next((i for i, e in enumerate(
                                            results["results"][fname][seeder][param])
                                         if e.get("value") == v), None)
                    if existing_idx is None:
                        results["results"][fname][seeder][param].append(cell)
                    else:
                        results["results"][fname][seeder][param][existing_idx] = cell
                    # Per-cell checkpoint + memory purge
                    _checkpoint(args.out, results)
                    _purge_memory()
                print(flush=True)

    print(f"\n# total elapsed: {(time.perf_counter() - t_start)/60.0:.1f} min "
          f"(skipped {n_skipped} cells via --resume)" if args.resume
          else f"\n# total elapsed: {(time.perf_counter() - t_start)/60.0:.1f} min")
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
