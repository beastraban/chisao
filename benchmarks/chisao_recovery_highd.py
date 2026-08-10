#!/usr/bin/env python3
r"""
chisao_recovery_highd.py
========================
ChiSao single-global recovery at high dimension, on the GPU, for the seven Group-A
functions -- written in the sfu_benchmark JSON layout so parity_lbfgs.py can read it
directly as its ChiSao column. Reuses sfu_benchmark's own run_one() (the canonical
ChiSao invocation) and recovered() (single-global, L_infinity), so the numbers are
identical in construction to the paper's recovery tables.

Run on the GPU (this is ChiSao's home turf; high-d is fast here). If your shell has
CUDA_VISIBLE_DEVICES=-1 left over from a CPU run, set it back to 0 first.

USAGE
    set PYTHONPATH=D:\Dropbox\chisao\src
    set CUDA_VISIBLE_DEVICES=0
    python -u chisao_recovery_highd.py --dims 64,128,256,512 --trials 5 \
        --out sfu_benchmark_highd_gpu.json
then re-read it into the parity table:
    python -u parity_lbfgs.py --dims 64,128,256,512 --trials 5 --wall-cap 120 \
        --chisao-json sfu_benchmark_highd_gpu.json --out parity_highdim.json
(baselines resume-skip; only the ChiSao column is refreshed.)
"""
import argparse, importlib.util, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]


def load_sfu():
    p = os.path.join(_HERE, "sfu_benchmark.py")
    spec = importlib.util.spec_from_file_location("sfu_benchmark", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="64,128,256,512")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--functions", default=",".join(GROUP_A))
    ap.add_argument("--out", default="sfu_benchmark_highd_gpu.json")
    args = ap.parse_args()

    sfu = load_sfu()
    sticky = sfu.load_package(_HERE)
    funcs = [f for f in args.functions.split(",") if f in sfu.FUNC_REGISTRY]
    dims = [int(x) for x in args.dims.split(",")]
    n = args.trials
    out = os.path.join(_HERE, args.out)

    print("device:", "GPU" if getattr(sfu, "GPU", False) else "CPU (set CUDA_VISIBLE_DEVICES=0 for GPU!)",
          flush=True)
    res = {}
    if os.path.exists(out):
        try:
            res = json.load(open(out)); print(f"[resume] loaded {out}", flush=True)
        except Exception:
            res = {}

    for f in funcs:
        cfg = sfu.FUNC_REGISTRY[f]; res.setdefault(f, {})
        for D in dims:
            dk = str(D); res[f].setdefault(dk, {})
            for seeder in sfu.SEEDERS:                      # ['random', 'carry_tiger']
                if res[f][dk].get(seeder, {}).get("n_trials") == n:
                    continue                                # resume
                succ = 0; walls = []
                for t in range(n):
                    try:
                        peaks, wall = sfu.run_one(sticky, f, cfg, D, sfu.SEED + t, seeder)
                        succ += int(sfu.recovered(peaks, cfg, D)); walls.append(wall)
                    except Exception as e:
                        print(f"  [ERR {f} d={D} {seeder} t={t}] {e}", flush=True)
                        walls.append(float("nan"))
                res[f][dk][seeder] = {"rate": succ / n,
                                      "mean_wall": float(np.nanmean(walls)) if walls else 0.0,
                                      "n_trials": n}
                json.dump(res, open(out, "w"), indent=2)
                print(f"{f:<16} d={D:<5} {seeder:<12} rate={succ / n:.0%}  "
                      f"mean {np.nanmean(walls):.1f}s", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
