#!/usr/bin/env python3
r"""
run_all.py -- overnight driver for the ENTIRE ChiSao benchmark gamut.
=====================================================================
Runs every benchmark sub-script as a subprocess. Hardened for unattended runs:

  * LIVE output           -- children run UNBUFFERED (-u / PYTHONUNBUFFERED), so lines
                             stream to the console / log in real time (no silent wait)
  * continue-on-failure   -- one bad step never stops the rest
  * per-step TIMEOUT       -- a HUNG step is killed and the gamut moves on
  * live STATUS file       -- run_all_status.txt rewritten before/after every step;
                             its last line shows where/when it stopped if the process dies
  * RESUME                 -- --resume skips steps already marked OK in the status file
  * device tagging         -- outputs suffixed _gpu / _cpu so CPU and GPU never clobber
  * incremental saves      -- every sub-script writes its JSON as it goes

ONE command does everything (GPU pass then CPU pass):

    set PYTHONPATH=D:\Dropbox\chisao\src
    cd /d D:\Dropbox\chisao\benchmarks
    python -u run_all.py --device both  > gamut_overnight.log 2>&1

Resume after an interruption (same command + --resume):
    python -u run_all.py --device both --resume  >> gamut_overnight.log 2>&1

Defaults: --trials 10, --reps 10, dense --scale-dims, --timeout 28800 (8h/step), --settle 5.
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DENSE_SCALE = "64,128,192,256,384,512,768,1024,1536,2048"   # brackets the d=512 regime switch

RESET_WARMUP = (
    "import cupy as cp\n"
    "cp.get_default_memory_pool().free_all_blocks()\n"
    "cp.get_default_pinned_memory_pool().free_all_blocks()\n"
    "a = cp.random.rand(2048, 2048, dtype=cp.float32)\n"
    "s = float((a @ a).sum()); cp.cuda.Stream.null.synchronize()\n"
    "print('  [gpu reset+warmup] pools freed, kernels warm (warm-sum=%.3e)' % s, flush=True)\n"
)


def steps(dev, trials, reps, scale_dims, recovery_dims):
    rec = recovery_dims.split(",")                 # sfu_benchmark wants SPACE-separated dims
    t, r = str(trials), str(reps)
    return [
        ("recovery",  "sfu_benchmark.py",
         ["--source-dir", ".", "--dims", *rec, "--trials", t, "--out", f"sfu_benchmark_{dev}.json"],
         {"cpu", "gpu"}),
        ("ablation",  "run_ablation_all.py",
         ["--source-dir", ".", "--dims", "8,16,32,64", "--trials", t,
          "--out-prefix", f"ablation_{dev}", "--fig", f"fig_ablation_{dev}.png"], {"cpu", "gpu"}),
        ("portfolio", "portfolio_recovery.py",
         ["--source-dir", ".", "--dims", "8,16,32,64", "--trials", str(max(3, trials // 2)),
          "--out", f"portfolio_{dev}.json"], {"cpu", "gpu"}),
        ("noise",     "noise_table.py",
         ["--dim", "6", "--sigmas", "0,0.1,0.2,0.5,1.0", "--trials", "10", "--method", "lbfgs",
          "--out", "noise_table.json"], {"cpu"}),
        ("scaling",   "gpu_scaling.py",
         (["--dims", scale_dims, "--reps", r, "--out", "gpu_scaling_gpu.json"] if dev == "gpu"
          else ["--dims", "64,128,256,512", "--reps", r, "--baselines", "--baseline-max-dim", "512",
                "--out", "gpu_scaling_cpu.json"]), {"cpu", "gpu"}),
        ("highdim",   "highdim_recovery.py",
         ["--dims", scale_dims, "--out", f"highdim_recovery_{dev}.json"], {"gpu"}),
        ("pop_base",  "pop_saturation.py",
         ["--function", "ackley", "--D", "64", "--shift-frac", "0.4",
          "--sweep", "200,1000,5000,20000,100000"], {"gpu"}),
        ("pop_full",  "pop_saturation.py",
         ["--function", "ackley", "--D", "64", "--shift-frac", "0.4",
          "--sweep", "200,1000,5000,20000,100000", "--full", "--seeder", "carry_tiger"], {"gpu"}),
    ]


def _stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(path, header, rows, current):
    try:
        with open(path, "w") as f:
            f.write(header + "\n")
            f.write(f"LAST UPDATE : {_stamp()}\n")
            f.write(f"CURRENT     : {current}\n")
            f.write("-" * 60 + "\n")
            for dev, name, status, dt in rows:
                f.write(f"  {dev:<4} {name:<12} {status:<14} {dt:>8.0f}s\n")
    except Exception:
        pass


def parse_completed(path):
    """Return {(dev,name): status} from a prior status file's row lines (for --resume)."""
    done = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[0] in ("cpu", "gpu"):
                    done[(p[0], p[1])] = p[2]
    except Exception:
        pass
    return done


def _step_output(name, a, dev):
    """Best-effort path to a step's primary output JSON, so --resume can skip on
    file existence (robust) instead of fragile status-file text."""
    if "--out" in a:
        return os.path.join(_HERE, a[a.index("--out") + 1])
    if "--out-prefix" in a:
        return os.path.join(_HERE, a[a.index("--out-prefix") + 1] + ".json")
    if name in ("pop_base", "pop_full"):        # pop_saturation auto-names its output
        fn = a[a.index("--function") + 1] if "--function" in a else "ackley"
        sd = a[a.index("--seeder") + 1] if "--seeder" in a else "random"
        cf = "full" if "--full" in a else "base"
        D = a[a.index("--D") + 1] if "--D" in a else "64"
        return os.path.join(_HERE, f"pop_saturation_{fn}_{sd}_{cf}_D{D}.json")
    return None


_GROUP_A = ["rastrigin", "ackley", "levy", "griewank", "styblinski_tang", "schwefel", "michalewicz"]


def _step_done(name, a, dev):
    """A step is done only if its output JSON exists AND is COMPLETE. Completeness
    (not mere existence) matters because recovery/ablation write incrementally, so a
    half-written file must never be treated as done on --resume."""
    # ablation resumes internally (per-cell) -> always re-enter; the subprocess
    # skips finished cells and exits in seconds if all are done.
    if name == "ablation":
        return False
    p = _step_output(name, a, dev)
    if not p or not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    if not d:
        return False
    if name == "recovery":
        # need all 7 Group-A functions, each covering every requested dim
        dims = a[a.index("--dims") + 1: a.index("--trials")] if "--dims" in a and "--trials" in a else []
        if not all(f in d for f in _GROUP_A):
            return False
        return all(all(str(x) in d[f] for x in dims) for f in _GROUP_A)
    return True   # single-shot outputs (portfolio, scaling, highdim, pop, noise)


def _gpu_reset_warmup(env):
    try:
        subprocess.run([sys.executable, "-u", "-c", RESET_WARMUP], env=env, cwd=_HERE, timeout=180)
    except Exception as e:
        print(f"  [reset+warmup skipped: {e}]", flush=True)


def _run_one_device(dev, only, args, rows, completed):
    env = dict(os.environ)
    env["CHISAO_SRC"] = args.chisao_src
    env["PYTHONPATH"] = args.chisao_src + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0" if dev == "gpu" else "-1"
    env["PYTHONUNBUFFERED"] = "1"                              # live child output
    env["PYTHONUTF8"] = "1"                                    # so unicode (checkmarks) survive a pipe on Windows cp1252
    env["PYTHONIOENCODING"] = "utf-8"

    plan = [(n, s, a) for (n, s, a, dv) in steps(dev, args.trials, args.reps, args.scale_dims, args.recovery_dims)
            if dev in dv and (only is None or n in only)]

    print("\n" + "=" * 76, flush=True)
    print(f"[{_stamp()}]  ChiSao GAMUT  |  device={dev}  CVD={env['CUDA_VISIBLE_DEVICES']}  "
          f"trials={args.trials} reps={args.reps}  timeout={args.timeout}s  resume={args.resume}", flush=True)
    print(f"  scale-dims={args.scale_dims}", flush=True)
    print(f"  steps: {', '.join(n for n, _, _ in plan)}", flush=True)
    print("=" * 76, flush=True)

    try:
        chk = subprocess.run([sys.executable, "-c",
                              "import chisao; print('  child sees', chisao.__file__, '| GPU_OK:', chisao.GPU_OK)"],
                             env=env, cwd=_HERE, capture_output=True, text=True, timeout=120)
        print(chk.stdout.strip() or (chk.stderr.strip().splitlines()[-1] if chk.stderr.strip() else ""), flush=True)
    except Exception as e:
        print(f"  [device check skipped: {e}]", flush=True)

    n_bad = 0
    to = args.timeout if args.timeout > 0 else None
    for name, script, a in plan:
        if args.resume and (_step_done(name, a, dev) or completed.get((dev, name)) in ("OK", "SKIP(done)")):
            why = "output JSON on disk" if _step_done(name, a, dev) else "marked OK in status file"
            print(f"\n[{_stamp()}] [{name}] SKIP ({why})", flush=True)
            rows.append((dev, name, "OK", 0.0)); write_status(args.status, args._header, rows, f"skip {dev}/{name}")
            continue
        if dev == "gpu" and not args.dry_run:
            _gpu_reset_warmup(env)
            if args.settle > 0:
                time.sleep(args.settle)
        cmd = [sys.executable, "-u", script] + a          # -u => unbuffered child
        print("\n" + "#" * 76, flush=True)
        print(f"# [{_stamp()}] STEP {name} ({dev})", flush=True)
        print("# " + " ".join(cmd), flush=True)
        print("#" * 76, flush=True)
        write_status(args.status, args._header, rows, f"RUNNING {dev}/{name} since {_stamp()} (timeout {to}s)")
        if args.dry_run:
            rows.append((dev, name, "DRY", 0.0)); continue
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(cmd, env=env, cwd=_HERE, timeout=to).returncode
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            print(f"\n[{_stamp()}] [{name}] TIMED OUT after {to}s -- killed, continuing", flush=True)
        dt = time.perf_counter() - t0
        if status != "OK":
            n_bad += 1
        rows.append((dev, name, status, dt))
        write_status(args.status, args._header, rows, f"{dev}/{name} -> {status}")
        print(f"\n[{_stamp()}] [{name}] {status} in {dt:.0f}s", flush=True)

    print("\n" + "=" * 76, flush=True)
    print(f"[{_stamp()}] SUMMARY (device={dev})", flush=True)
    for d2, name, status, dt in [r for r in rows if r[0] == dev]:
        print(f"  {name:<12} {status:<14} {dt:>9.0f}s", flush=True)
    print(f"  -> {'ALL OK' if n_bad == 0 else f'{n_bad} not-OK'}  (outputs tagged _{dev})", flush=True)
    return n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu", "both"], required=False,
                    help="'both' runs the GPU pass then the CPU pass in one shot")
    ap.add_argument("--only", default=None, help="comma-separated step names to run")
    ap.add_argument("--resume", action="store_true", help="skip steps already marked OK in the status file")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--reps", type=int, default=10, help="timed repeats for the scaling curve (median)")
    ap.add_argument("--scale-dims", default=DENSE_SCALE, dest="scale_dims")
    ap.add_argument("--recovery-dims", default="2,4,8,16,32,64", dest="recovery_dims")
    ap.add_argument("--settle", type=float, default=5.0, help="seconds to sleep after mem-free+warmup, per GPU step")
    ap.add_argument("--timeout", type=int, default=28800, help="per-step kill limit in seconds (0 = no limit)")
    ap.add_argument("--status", default=os.path.join(_HERE, "run_all_status.txt"))
    ap.add_argument("--chisao-src", default=os.path.abspath(os.path.join(_HERE, "..", "src")))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, s, a, dv in steps("gpu", args.trials, args.reps, args.scale_dims, args.recovery_dims):
            print(f"  {n:<10} [{'/'.join(sorted(dv))}]  {s} {' '.join(a)}")
        return
    if not args.device:
        ap.error("--device {cpu,gpu,both} is required (or use --list)")

    only = set(args.only.split(",")) if args.only else None
    devices = ["gpu", "cpu"] if args.device == "both" else [args.device]
    completed = parse_completed(args.status) if args.resume else {}
    if args.resume:
        print(f"[{_stamp()}] RESUME: {sum(1 for v in completed.values() if v=='OK')} step(s) already OK will be skipped", flush=True)
    args._header = f"ChiSao GAMUT  device={args.device}  started {_stamp()}  passes: {', '.join(devices)}"
    rows = []
    write_status(args.status, args._header, rows, "STARTING")

    t_start = time.perf_counter()
    total_bad = sum(_run_one_device(dev, only, args, rows, completed) for dev in devices)
    mins = (time.perf_counter() - t_start) / 60

    final = f"COMPLETE at {_stamp()} in {mins:.1f} min -- {'ALL OK' if total_bad == 0 else f'{total_bad} step(s) NOT OK'}"
    write_status(args.status, args._header, rows, final)
    print("\n" + "=" * 76, flush=True)
    print(f"[{_stamp()}] GAMUT {final}", flush=True)
    print(f"  status file: {args.status}", flush=True)
    print("=" * 76, flush=True)
    sys.stdout.write("\a"); sys.stdout.flush()   # terminal bell on finish
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
