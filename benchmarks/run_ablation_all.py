#!/usr/bin/env python3
r"""
run_ablation_all.py — extend the ChiSao ablation to d = 8,16,32,64 and plot it.
==============================================================================
One script that does the whole R3-comment-3 job: it drives the SAME ablation as
ablation.py (dedup-off via stick_tolerance=0, the seven Group-A functions from
sfu_benchmark.py, mode-recovery via sfu.recovered, carry_tiger seeding), across
d = 8,16,32,64, then writes the extended heatmap (Figure-5 style, one panel per
dimension) plus a JSON and a text table.

The recovery phase (this session's addition) is forced OFF if the loaded package
exposes it, so this always ablates the PUBLISHED six-phase algorithm regardless of
whether core.py is patched.

USAGE (run where sfu_benchmark.py lives; package on PYTHONPATH or via --source-dir):
    python run_ablation_all.py --source-dir . --dims 8,16,32,64 --seeder carry_tiger --trials 10
"""
import argparse, importlib.util, inspect, json, os, sys, time, warnings
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

# One row per configuration; each removes exactly one phase from Full (dedup off
# = stick_tolerance 0.0, matching ablation.py).
CONFIGS = [
    ("Full ChiSao",              {}),
    ("- Anti-conv (Ph.6)",       {"n_anticonverge": 0}),
    ("- Hands Like Clouds (Ph.5)", {"cloud_enabled": False}),
    ("- Reseeding (Ph.4)",       {"reseed_strategy": None}),
    ("- Deduplication (Ph.3)",   {"stick_tolerance": 0.0}),
    ("Single osc. (n=1)",        {"n_oscillations": 1}),
    ("Plain multistart",         {"n_oscillations": 1, "n_anticonverge": 0,
                                  "cloud_enabled": False, "reseed_strategy": None}),
]
GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]
LABELS  = ["Rastrigin", "Ackley", "Lévy", "Griewank", "Styblinski-Tang", "Schwefel", "Michalewicz"]


def load_sfu(source_dir):
    path = os.path.join(os.path.abspath(source_dir), "sfu_benchmark.py")
    spec = importlib.util.spec_from_file_location("sfu_benchmark", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def gpu_clear():
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=".")
    ap.add_argument("--dims", default="8,16,32,64")
    ap.add_argument("--seeder", choices=["random", "carry_tiger"], default="carry_tiger")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--functions", default=",".join(GROUP_A),
                    help="subset of Group-A (default: all)")
    ap.add_argument("--out-prefix", default="ablation_extended")
    ap.add_argument("--fig", default="fig_ablation_extended.png")
    args = ap.parse_args()

    sfu = load_sfu(args.source_dir)
    sticky_hands = sfu.load_package(args.source_dir)
    sig = set(inspect.signature(sticky_hands).parameters)
    force_off = {"recover_stuck": False} if "recover_stuck" in sig else {}
    if force_off:
        print("note: recover_stuck present in loaded package -> forcing OFF (clean 6-phase ablation)")

    dims = [int(d) for d in args.dims.split(",")]
    funcs = [f for f in args.functions.split(",") if f in GROUP_A]

    # ---------- resume: reload any completed cells from a prior (interrupted) run ----------
    out_json = f"{args.out_prefix}.json"
    results = {}   # results[str(D)][cfg_label][fname] = {"rate","mean_wall","n_trials"}
    n_reloaded = 0
    if os.path.exists(out_json):
        try:
            results = json.load(open(out_json))
            for D in results:
                for cl in results[D]:
                    for fn in results[D][cl]:
                        v = results[D][cl][fn]
                        if v and v.get("n_trials") == args.trials:
                            n_reloaded += 1
            print(f"[resume] loaded {out_json}: {n_reloaded} completed cell(s) will be skipped",
                  flush=True)
        except Exception as e:
            print(f"[resume] could not read {out_json} ({e}); starting fresh", flush=True)
            results = {}

    def _done(D, cfg_label, fname):
        v = results.get(str(D), {}).get(cfg_label, {}).get(fname)
        return bool(v) and v.get("n_trials") == args.trials

    def _save():
        json.dump(results, open(out_json, "w"), indent=2)

    # GPU warmup: one throwaway run so kernel compilation doesn't inflate the first timed cell
    if GPU and funcs:
        try:
            _c = sfu.FUNC_REGISTRY[funcs[0]]; _b = sfu.get_bnd(_c, dims[0])
            _x0 = (sfu.seed_carry_tiger(_c, dims[0], 0) if args.seeder == "carry_tiger"
                   else sfu.seed_random(_c, dims[0], 200, 0))
            _wp = {k: v for k, v in dict(method="lbfgs", n_converge=10, n_anticonverge=5,
                    n_oscillations=3, stick_tolerance=1e-3, reseed_strategy="sunburst",
                    cannon_through_sky=True, cloud_enabled=True, bounds=_b,
                    estimate_widths=False, verbose=False, **force_off).items()
                   if k in sig or k == "bounds"}
            print("warming up GPU (kernel compilation) ...", flush=True)
            sticky_hands(_c["func"], _x0, **_wp); gpu_clear()
        except Exception as e:
            print(f"[warmup skipped: {e}]", flush=True)

    for D in dims:
        results.setdefault(str(D), {})
        for cfg_label, ov in CONFIGS:
            results[str(D)].setdefault(cfg_label, {})
            for fname in funcs:
                if _done(D, cfg_label, fname):
                    v = results[str(D)][cfg_label][fname]
                    print(f"d={D:>3} | {cfg_label:<26} | {fname:<16} "
                          f"{v['rate']*100:>5.0f}%  [resumed]", flush=True)
                    continue
                if fname not in sfu.FUNC_REGISTRY:
                    results[str(D)][cfg_label][fname] = None; _save(); continue
                cfg = sfu.FUNC_REGISTRY[fname]; bnd = sfu.get_bnd(cfg, D)
                N = 50 * (10 + int(np.ceil(np.log2(max(2, D)))))
                if args.seeder == "carry_tiger":
                    seeder_fn = lambda D, seed, _c=cfg: sfu.seed_carry_tiger(_c, D, seed)
                else:
                    seeder_fn = lambda D, seed, _c=cfg, _N=N: sfu.seed_random(_c, D, _N, seed)
                base = dict(method="lbfgs", n_converge=10, n_anticonverge=5, n_oscillations=3,
                            stick_tolerance=1e-3, reseed_strategy="sunburst",
                            cannon_through_sky=True, cloud_enabled=True, bounds=bnd,
                            estimate_widths=False, verbose=False, **force_off)
                base = {k: v for k, v in base.items() if k in sig or k == "bounds"}  # version-robust
                recs = []; tt = []
                for t in range(args.trials):
                    seed = 1000 + t
                    np.random.seed(seed)
                    if GPU: cp.random.seed(seed)
                    x0 = seeder_fn(D, seed)
                    params = {**base, **ov}
                    _t0 = time.perf_counter()
                    try:
                        r = sticky_hands(cfg["func"], x0, **params); peaks = r.get("peaks")
                    except Exception as e:
                        print(f"  [ERR d={D} {cfg_label} {fname} seed={seed}] {e}", flush=True)
                        peaks = None
                    gpu_clear()
                    tt.append(time.perf_counter() - _t0)
                    recs.append(int(sfu.recovered(peaks, cfg, D)))
                rate = float(np.mean(recs)); mean_t = float(np.mean(tt))
                results[str(D)][cfg_label][fname] = {"rate": rate, "mean_wall": mean_t,
                                                     "n_trials": args.trials}
                _save()   # checkpoint after EVERY function -> resume never loses more than one cell
                print(f"d={D:>3} | {cfg_label:<26} | {fname:<16} {rate*100:>5.0f}%  "
                      f"mean {mean_t:>5.1f}s/trial  ({sum(tt):.0f}s)", flush=True)

    # ---------- text table (always) ----------
    for D in dims:
        print(f"\n=== d = {D} — mode-recovery rate (%), {args.seeder} seeder, {args.trials} trials ===")
        print(f"{'config':<26} " + " ".join(f"{l[:7]:>8}" for l in LABELS))
        for cfg_label, _ in CONFIGS:
            row = f"{cfg_label:<26} "
            for fname in funcs:
                v = results[str(D)][cfg_label].get(fname)
                r = v["rate"] if v else None
                row += f"{'   -' if r is None else f'{r*100:>7.0f}':>8} "
            print(row)

    # ---------- extended heatmap (Figure-5 style) ----------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nD = len(dims)
        fig, axes = plt.subplots(1, nD, figsize=(3.7 * nD, 4.8), squeeze=False)
        cmap = plt.get_cmap("RdYlGn")
        im = None
        for ci, D in enumerate(dims):
            ax = axes[0][ci]
            def _rate(cl, f):
                v = results[str(D)][cl].get(f); return v["rate"] if v else np.nan
            M = np.array([[_rate(cl, f) for f in funcs] for cl, _ in CONFIGS], dtype=float)
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(funcs)))
            ax.set_xticklabels([LABELS[GROUP_A.index(f)] for f in funcs], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(CONFIGS)))
            ax.set_yticklabels([c for c, _ in CONFIGS] if ci == 0 else [""] * len(CONFIGS), fontsize=8)
            ax.set_title(f"d = {D}", fontsize=11)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    ax.text(j, i, "–" if np.isnan(v) else f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="black")
        fig.suptitle(f"Ablation: mode-recovery rate vs dimension ({args.seeder} seeder, {args.trials} trials)",
                     fontsize=12)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="recovery rate")
        fig.savefig(args.fig, dpi=150, bbox_inches="tight")
        print(f"\nwrote {args.fig}")
    except ImportError:
        print("\n[matplotlib not installed — JSON + text table written; "
              "`pip install matplotlib` and re-run for the figure]")
    print(f"wrote {args.out_prefix}.json")


if __name__ == "__main__":
    main()
