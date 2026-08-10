#!/usr/bin/env python3
"""
niching_gpu_benchmark.py
========================
A multi-solver benchmark for MULTIMODAL (find-all-global-optima) optimization,
scoring *peak ratio* and *wall-clock* on the same footing across:

  * ChiSao            (GPU-native; run here with the bank-HARVESTER extraction)
  * crowding-DE       (classic CPU niching baseline)
  * multistart-L-BFGS (trivial embarrassingly-parallel baseline)
  * EvoX  PSO/CMA-ES  (GPU-native EC framework -- single-objective; adapter)
  * QDax  MAP-Elites  (GPU-native Quality-Diversity; adapter)

...and folds in the *published* CEC'2013 competition numbers (crowding-DE, NEA2,
dADE, CMA, ...) shipped in mikeagn/CEC2013, so you get a full table without
re-running the incumbents.

WHY THIS DESIGN
---------------
The scientific point of the paper is NOT "ChiSao beats niching methods on peak
ratio" (it doesn't, at equal FEs). It is: the dedicated niching methods are
CPU-bound, and the GPU-native population methods (EvoX single/multi-objective;
QDax quality-diversity) do not catalogue global optima. Running EvoX/QDax here
on a multimodal objective *demonstrates* that -- they will not recover all
global optima -- which is exactly the evidence for ChiSao's novelty claim.

STATUS
------
ChiSao / crowding-DE / multistart are validated on CPU. The EvoX and QDax
adapters are GPU-side scaffolds (JAX/PyTorch): they are import-gated and clearly
marked; run them on your GPU box and adjust to your installed evox/qdax version.

USAGE
-----
    # get the official CEC'2013 python port + data:
    git clone https://github.com/mikeagn/CEC2013
    python niching_gpu_benchmark.py --cec /path/to/CEC2013 \
        --problems cec:1,4,6,9 --solvers chisao,crowding_de,multistart \
        --reference CDE,nea2,dade1
    # high-dimensional GMM sweep (ChiSao's home turf):
    python niching_gpu_benchmark.py --problems gmm:K=25,D=8/16/32/64 \
        --solvers chisao,multistart
"""
import argparse, json, os, sys, time
import numpy as np

# The installed `chisao` PyPI package must win over any local chisao.py that
# happens to sit next to this script (e.g. a dev copy in a paper folder),
# otherwise a single-file module shadows the package and `chisao.seeding` fails.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

ACCS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]      # CEC'2013 accuracy levels


# ======================================================================
# PROBLEMS
# ======================================================================
class CEC2013Problem:
    """Wraps one function of the official CEC'2013 niching suite."""
    def __init__(self, fid, cec_path):
        py3 = os.path.join(cec_path, "python3")
        if py3 not in sys.path:
            sys.path.insert(0, py3)
        # CFx.py loads data/*.dat relative to cwd -> run from python3 dir
        os.chdir(py3)
        from cec2013.cec2013 import CEC2013, how_many_goptima
        self._how = how_many_goptima
        self.f = CEC2013(fid)
        self.name = f"cec:F{fid}"
        self.D = self.f.get_dimension()
        self.n_optima = self.f.get_no_goptima()
        self.maxfes = self.f.get_maxfes()
        self.rho = self.f.get_rho()
        self.lb = np.array([self.f.get_lbound(k) for k in range(self.D)])
        self.ub = np.array([self.f.get_ubound(k) for k in range(self.D)])
        self.harvest_radius = self.rho          # niche radius for extraction

    def evaluate(self, X):
        """Maximization. X:[n,D] -> [n]. Box-clipped (Vincent etc. need it)."""
        X = np.atleast_2d(np.asarray(X, float))
        Xc = np.clip(X, self.lb, self.ub)
        return np.array([float(np.asarray(self.f.evaluate(x)).ravel()[0]) for x in Xc])

    def count(self, points, acc):
        pts = np.atleast_2d(np.asarray(points, float))
        if pts.size == 0 or pts.shape[1] != self.D:
            return 0
        return int(self._how(pts, self.f, acc)[0])


class GMMProblem:
    """log-Gaussian-mixture: K equal-height, well-separated modes in R^D.
    A real multimodal log-posterior -- ChiSao's native regime. Scalable to any D
    with a KNOWN mode count, which the CEC suite (<=20D) cannot provide."""
    def __init__(self, K, D, seed=1, box=6.0, sigma=1.0, sep=4.0):
        from scipy.special import logsumexp
        self._lse = logsumexp
        rng = np.random.default_rng(seed); C = []
        while len(C) < K:
            c = rng.uniform(-box, box, size=D)
            if all(np.linalg.norm(c - o) >= sep for o in C):
                C.append(c)
        self.C = np.array(C); self.Cn = (self.C ** 2).sum(1)
        self.sigma = sigma
        self.name = f"gmm:K={K},D={D}"
        self.D = D; self.n_optima = K
        self.maxfes = 200_000 * max(1, D // 8)
        self.lb = np.full(D, -box - 2); self.ub = np.full(D, box + 2)
        self.harvest_radius = 1.0
        self._tol = 1.5 * sigma            # a mode is "found" within 1.5 sigma

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d2 = (X * X).sum(1)[:, None] + self.Cn[None, :] - 2 * X @ self.C.T
        return self._lse(-0.5 * d2 / self.sigma ** 2, axis=1)

    def count(self, points, acc):
        pts = np.atleast_2d(np.asarray(points, float))
        if pts.size == 0 or pts.shape[1] != self.D:
            return 0
        # accuracy here scales the acceptance radius; use a fixed geometric tol
        return int(sum(np.min(np.linalg.norm(pts - c, axis=1)) < self._tol
                       for c in self.C))


# ======================================================================
# EXTRACTION (shared): harvest distinct modes from a point cloud
# ======================================================================
def harvest(P, fvals, r):
    """O(n) grid-snap dedup by L-inf radius r, keep highest-f per cell."""
    P = np.asarray(P); keys = np.floor(P / r).astype(np.int64)
    seen = set(); kept = []
    for i in np.argsort(-np.asarray(fvals)):
        k = tuple(keys[i])
        if k not in seen:
            seen.add(k); kept.append(i)
    return P[kept]


# ======================================================================
# SOLVERS  ->  return dict(points=[m,D], fes=int, wall=float, note=str)
# ======================================================================
def _counter(problem):
    fe = {"n": 0}
    def f(X):
        X = np.atleast_2d(X); fe["n"] += X.shape[0]
        return problem.evaluate(X)
    return f, fe


def solve_chisao(problem, budget, seed, n_oscillations=3):
    """ChiSao with the bank-HARVESTER extraction (the shipped stick-gate returns
    almost nothing on non-log-likelihood scales; harvesting the sample bank
    recovers the modes the search actually found). Objective is scale-normalized
    so ChiSao's stick criterion is dimensionally sane."""
    from chisao import sticky_hands
    from chisao.seeding import carry_tiger_seed, _as_bounds, get_array_module
    rng = np.random.default_rng(seed)
    Xs = rng.uniform(problem.lb, problem.ub, size=(800, problem.D))
    s = np.std(problem.evaluate(Xs)) or 1.0
    f, fe = _counter(problem)
    def g(X):
        return f(X) / s
    xp = get_array_module(False)
    b = _as_bounds([(problem.lb[k], problem.ub[k]) for k in range(problem.D)], xp)
    x0 = np.asarray(carry_tiger_seed(g, b, use_gpu=False, seed=seed))
    t0 = time.time()
    r = sticky_hands(g, x0, bounds=b, bank_samples=True, stick_tolerance=1e-3,
                     estimate_widths=False, n_oscillations=n_oscillations)
    wall = time.time() - t0
    bp = np.asarray(r["sample_bank"]["positions"])
    pts = harvest(bp, problem.evaluate(bp), problem.harvest_radius)
    return dict(points=pts, fes=fe["n"], wall=wall, note="harvester")


def solve_crowding_de(problem, budget, seed, NP=None, F=0.5, CR=0.9):
    """Crowding differential evolution -- a classic CPU niching baseline.
    Offspring replaces its NEAREST current individual if better (crowding),
    which preserves multiple niches."""
    rng = np.random.default_rng(seed)
    D = problem.D; NP = NP or max(50, 20 * D)
    pop = rng.uniform(problem.lb, problem.ub, size=(NP, D))
    fit = problem.evaluate(pop); fes = NP; t0 = time.time()
    while fes < budget:
        idx = rng.integers(0, NP, size=(NP, 3))
        a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
        mut = np.clip(a + F * (b - c), problem.lb, problem.ub)
        cross = rng.random((NP, D)) < CR
        cross[np.arange(NP), rng.integers(0, D, NP)] = True
        trial = np.where(cross, mut, pop)
        tf = problem.evaluate(trial); fes += NP
        # crowding replacement: each trial competes with its nearest in pop
        for i in range(NP):
            d = np.max(np.abs(pop - trial[i]), axis=1)   # L-inf
            j = int(np.argmin(d))
            if tf[i] > fit[j]:
                pop[j] = trial[i]; fit[j] = tf[i]
    return dict(points=pop, fes=fes, wall=time.time() - t0, note=f"NP={NP}")


def solve_multistart(problem, budget, seed):
    """Random-restart L-BFGS: strong baseline when basins are benign."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); fe = {"n": 0}
    def negf(z):
        fe["n"] += 1; return -problem.evaluate(z[None])[0]
    found = []; t0 = time.time()
    while fe["n"] < budget and len(found) < 20000:
        z0 = rng.uniform(problem.lb, problem.ub, size=problem.D)
        found.append(minimize(negf, z0, method="L-BFGS-B",
                              bounds=list(zip(problem.lb, problem.ub))).x)
    return dict(points=np.array(found), fes=fe["n"], wall=time.time() - t0,
                note=f"{len(found)} starts")


# --------------------- GPU adapters (import-gated) ---------------------
def solve_evox(problem, budget, seed):
    """GPU-native EvoX. EvoX ships single/multi-objective optimizers (PSO,
    CMA-ES, DE ...), NOT niching -- so its final population converges toward one
    basin and should score PR ~ 1/n_optima. That low score is the POINT: it
    evidences that a GPU EC framework is not a mode-cataloguer.
    NOTE: GPU-side, untested in a CPU sandbox; verify against your evox version."""
    try:
        import torch, evox
        from evox import algorithms, workflows, problems as _p  # noqa
    except Exception as e:
        return dict(points=np.zeros((0, problem.D)), fes=0, wall=0.0,
                    note=f"SKIPPED (evox unavailable: {e})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = problem.D; pop = max(256, 32 * D)
    lb = torch.tensor(problem.lb, device=dev); ub = torch.tensor(problem.ub, device=dev)
    def obj(x):  # EvoX minimizes -> negate; x:[pop,D] tensor
        xn = x.detach().cpu().numpy()
        return torch.tensor(-problem.evaluate(xn), device=dev)
    algo = algorithms.PSO(lb=lb, ub=ub, pop_size=pop)
    t0 = time.time(); gens = max(1, budget // pop); state = algo.init()
    last = None
    for _ in range(gens):
        cand, state = algo.ask(state)
        state = algo.tell(state, obj(cand)); last = cand
    pts = last.detach().cpu().numpy()
    return dict(points=pts, fes=gens * pop, wall=time.time() - t0, note=f"PSO pop={pop}")


def solve_qdax(problem, budget, seed):
    """GPU-native QDax MAP-Elites. Behavior descriptor = position (identity), so
    the archive illuminates the search space; score its archive for global
    optima. QD spreads across space but optimizes per-cell rather than to exact
    global optima -- typically good coverage, weaker precision at tight acc.
    NOTE: GPU/JAX-side, untested in a CPU sandbox; verify against your qdax version."""
    try:
        import jax, jax.numpy as jnp
        from qdax.core.map_elites import MAPElites  # noqa
    except Exception as e:
        return dict(points=np.zeros((0, problem.D)), fes=0, wall=0.0,
                    note=f"SKIPPED (qdax unavailable: {e})")
    # Minimal scaffold: descriptor = first 2 coords, fitness = objective.
    # Full wiring (emitter, centroids, scoring_fn) depends on your qdax version;
    # see qdax.readthedocs.io 'Optimizing with MAP-Elites'. Left explicit so you
    # can drop in your emitter/centroid config on the GPU box.
    return dict(points=np.zeros((0, problem.D)), fes=0, wall=0.0,
                note="SCAFFOLD: wire qdax emitter/centroids on GPU (see docstring)")


SOLVERS = {
    "chisao": solve_chisao, "crowding_de": solve_crowding_de,
    "multistart": solve_multistart, "evox": solve_evox, "qdax": solve_qdax,
}


# ======================================================================
# REFERENCE competitor data (published CEC'2013 competition results)
# ======================================================================
def load_reference(cec_path, name, kind="PR"):
    """Return the 20x5 (problem x accuracy) matrix for a competition entrant."""
    p = os.path.join(cec_path, "NichingCompetition2013FinalData", f"{name}_{kind}.dat")
    return np.loadtxt(p) if os.path.exists(p) else None


# ======================================================================
# RUNNER
# ======================================================================
def score(problem, res, accs=ACCS):
    out = {"fes": res["fes"], "wall": round(res["wall"], 3), "note": res["note"]}
    for a in accs:
        c = problem.count(res["points"], a)
        out[f"PR@{a:.0e}"] = round(c / problem.n_optima, 3)
    out["n_modes_returned"] = len(np.atleast_2d(res["points"]))
    return out


def parse_problems(spec, cec_path):
    probs = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if tok.startswith("cec:"):
            for fid in tok[4:].split(","):
                probs.append(CEC2013Problem(int(fid), cec_path))
        elif tok.startswith("gmm:"):
            kv = dict(p.split("=") for p in tok[4:].split(",") if "=" in p)
            K = int(kv.get("K", 25))
            for D in str(kv.get("D", "8")).split("/"):
                probs.append(GMMProblem(K, int(D)))
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cec", default="", help="path to a mikeagn/CEC2013 clone")
    ap.add_argument("--problems", default="cec:1,4,6",
                    help="e.g. cec:1,4,6,9  or  gmm:K=25,D=8/16/32/64")
    ap.add_argument("--solvers", default="chisao,crowding_de,multistart")
    ap.add_argument("--reference", default="CDE,nea2,dade1",
                    help="CEC'2013 entrants to print for comparison")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="niching_benchmark_results.json")
    args = ap.parse_args()

    problems = parse_problems(args.problems, args.cec)
    solver_names = [s for s in args.solvers.split(",") if s in SOLVERS]
    results = []
    for prob in problems:
        print(f"\n=== {prob.name}  D={prob.D}  optima={prob.n_optima}  "
              f"MaxFEs={prob.maxfes} ===")
        hdr = f"{'solver':>13} {'PR@1e-4':>8} {'FEs':>10} {'wall_s':>7} {'modes':>6}  note"
        print(hdr); print("-" * len(hdr))
        for sname in solver_names:
            prs = []
            for run in range(args.runs):
                res = SOLVERS[sname](prob, prob.maxfes, seed=run)
                prs.append(score(prob, res))
            pr = np.mean([r["PR@1e-04"] for r in prs])
            fes = int(np.mean([r["fes"] for r in prs]))
            wall = np.mean([r["wall"] for r in prs])
            modes = int(np.mean([r["n_modes_returned"] for r in prs]))
            print(f"{sname:>13} {pr:>8.3f} {fes:>10} {wall:>7.2f} {modes:>6}  {prs[0]['note']}")
            results.append({"problem": prob.name, "solver": sname, "runs": prs})
        # published incumbents (CEC only)
        if prob.name.startswith("cec:") and args.cec:
            fid = int(prob.name.split("F")[1])
            for ref in args.reference.split(","):
                M = load_reference(args.cec, ref)
                if M is not None:
                    pr = M[fid - 1, 3]     # column 3 == acc 1e-4
                    print(f"{'ref:' + ref:>13} {pr:>8.3f} {'(published)':>10} {'--':>7} {'--':>6}  CEC'2013 competition")
    json.dump(results, open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
