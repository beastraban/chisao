#!/usr/bin/env python3
r"""
noise_table.py — the CORRECT noise-robustness table (paper Table 8).

Replaces chisao_benchmark.py's Table 5, which was invalid (L2 metric + stick
tolerance slaved to sigma). Here:
  - noise is sensed by assess_noise.py (no SunBURST needed);
  - the sensed fd_eps is passed as `epsilon` so gradients aren't amplified by 1/eps;
  - the convergence + final gate uses `grad_threshold` = 10x sensed gradient noise;
  - `dedup_radius` widens to the noise localization radius;
  - recovery is scored in L-inf (never L2), "were BOTH modes found" within a
    noise-dependent tolerance.

Target: 2-peak likelihood in d=6, peaks at +/- 2*e1 (as in the paper).

Requires the PATCHED package (grad_threshold/epsilon/dedup_radius on sticky_hands)
and assess_noise.py on the path.

USAGE (package on PYTHONPATH; assess_noise.py alongside):
    python noise_table.py --dim 6 --sigmas 0,0.1,0.2,0.5,1.0 --trials 10 --method lbfgs
NOTE: method=randcoord does NOT yet thread epsilon (single_whip.py); use lbfgs for noisy runs.
"""
import argparse, importlib, inspect, json, os, sys, warnings
import numpy as np
# The noise result is HARDWARE-INDEPENDENT and the 2-peak objective is pure NumPy.
# Force CPU (set BEFORE importing chisao/cupy) so a GPU-enabled ChiSao doesn't feed
# the numpy objective device arrays -> "Unsupported type numpy.ndarray".
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "-1" hides the GPU; "" does NOT (CuPy treats empty as unrestricted)
cp = np; GPU = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.environ.get("CHISAO_SRC", r"D:\Dropbox\chisao\src")
if _HERE not in sys.path:
    sys.path.append(_HERE)              # for assess_noise, kept OFF index 0
sys.path.insert(0, _PKG)               # patched package chisao must win over any local chisao.py
import chisao
from assess_noise import assess_noise
print(f"[noise_table] chisao loaded from: {chisao.__file__}")
sh = chisao.sticky_hands
SIG = set(inspect.signature(sh).parameters)
if "grad_threshold" not in SIG:
    raise SystemExit("This chisao has NO grad_threshold param -> UNPATCHED copy loaded. "
                     "Set CHISAO_SRC to the patched package src (e.g. D:\\Dropbox\\chisao\\src).")


def two_peak(D, w, sep, sigma):
    mu1 = np.zeros(D); mu1[0] = sep
    mu2 = np.zeros(D); mu2[0] = -sep
    def f(x, _s=sigma):
        x = np.atleast_2d(x)
        g1 = -0.5 * np.sum((x - mu1) ** 2, axis=1) / w ** 2
        g2 = -0.5 * np.sum((x - mu2) ** 2, axis=1) / w ** 2
        L = np.logaddexp(g1, g2)
        return L + (np.random.normal(0, _s, x.shape[0]) if _s > 0 else 0.0)
    return f, mu1, mu2


def grad_noise_linf(f, x, eps, D, reps=15):
    est = []
    for _ in range(reps):
        g = np.zeros(D)
        for d in range(D):
            e = np.zeros(D); e[d] = eps
            g[d] = (f((x + e)[None, :])[0] - f((x - e)[None, :])[0]) / (2 * eps)
        est.append(np.max(np.abs(g)))
    return float(np.std(est))


def found(peaks, mu, tol):
    if peaks is None or len(peaks) == 0:
        return False
    p = np.array(peaks.get() if hasattr(peaks, "get") else peaks)
    if p.ndim == 1:
        p = p[None, :]
    return bool(np.any(np.max(np.abs(p - mu), axis=1) < tol))   # L-inf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--sep", type=float, default=2.0)
    ap.add_argument("--sigmas", default="0,0.1,0.2,0.5,1.0")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--method", default="lbfgs")
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--out", default="noise_table.json")
    args = ap.parse_args()

    D = args.dim
    sigmas = [float(s) for s in args.sigmas.split(",")]
    print(f"2-peak d={D} likelihood, peaks at +/-{args.sep}*e1, width {args.width}, "
          f"{args.trials} trials, method={args.method}")
    print(f"{'sigma':>6} {'recovery':>9} {'mean d(+)':>10} {'mean d(-)':>10} {'grad_thr':>9}  (tol=0.5+2sigma, L-inf)")
    results = {}
    for sigma in sigmas:
        fd = max(sigma, 1e-3); tol = 0.5 + 2 * sigma; ded = 0.4 + sigma
        recs = []; d1 = []; d2 = []; gts = []
        for t in range(args.trials):
            seed = 1000 + t; np.random.seed(seed)
            if GPU: cp.random.seed(seed)
            f, mu1, mu2 = two_peak(D, args.width, args.sep, sigma)
            if sigma > 0:
                gthr = max(10 * grad_noise_linf(f, mu1 + 0.3, fd, D), 1e-3)
                dr = ded
            else:
                gthr = None; dr = None
            gts.append(gthr if gthr else 0.0)
            base = dict(method=args.method, n_converge=15, n_anticonverge=5, n_oscillations=3,
                        stick_tolerance=1e-3, reseed_strategy="sunburst", line_search="armijo",
                        cloud_enabled=True, estimate_widths=False, verbose=False,
                        bounds=np.array([[-5. * args.sep, 5. * args.sep]] * D),
                        epsilon=fd, grad_threshold=gthr, dedup_radius=dr)
            base = {k: v for k, v in base.items() if k in SIG or k == "bounds"}
            x0 = np.random.uniform(-5 * args.sep, 5 * args.sep, (args.N, D))
            try:
                pk = sh(f, x0, **base).get("peaks")
            except Exception as e:
                print(f"  [ERR sigma={sigma} t={t}] {e}"); pk = None
            r1 = found(pk, mu1, tol); r2 = found(pk, mu2, tol)
            recs.append(r1 and r2)
            if pk is not None and len(pk):
                p = np.array(pk.get() if hasattr(pk, "get") else pk)
                d1.append(float(np.min(np.max(np.abs(p - mu1), axis=1))))
                d2.append(float(np.min(np.max(np.abs(p - mu2), axis=1))))
        rate = float(np.mean(recs)) * 100
        results[sigma] = dict(recovery=rate, mean_d_plus=(np.mean(d1) if d1 else None),
                              mean_d_minus=(np.mean(d2) if d2 else None), grad_threshold=float(np.mean(gts)))
        print(f"{sigma:>6} {rate:>8.0f}% {(np.mean(d1) if d1 else float('nan')):>10.3f} "
              f"{(np.mean(d2) if d2 else float('nan')):>10.3f} {np.mean(gts):>9.2f}", flush=True)
        json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
