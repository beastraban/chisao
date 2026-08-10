# ChiSao benchmark suite

One-command driver (resume-safe, per-cell checkpoints):
    set PYTHONPATH=<repo>\src
    python -u run_all.py --device both --resume

Benchmarks:
    sfu_benchmark.py         42-function mode-recovery grid (L-inf), random vs carry_tiger seeders
    run_ablation_all.py      six-phase ablation, d=8..64
    portfolio_recovery.py    Full u (-anti) union recovery (Schwefel)
    highdim_recovery.py      shifted-Rastrigin recovery to d=2048 (GPU)
    pop_saturation.py        population/throughput saturation
    gpu_scaling.py           wall-clock scaling; --baselines, --wall-cap S, --baselines-only
    parity_lbfgs.py          memetically-fair single-global parity (baselines + L-BFGS polish), L-inf, --wall-cap
    chisao_recovery_highd.py ChiSao single-global recovery at high d (GPU), sfu_benchmark JSON layout
    noise_table.py / assess_noise.py   noise robustness + threshold calibration
    niching_gpu_benchmark.py niching-method comparison
figure_scripts/  regenerate the manuscript figures from the JSON outputs.

Scoring is L-inf (Chebyshev) throughout. ChiSao: CUDA_VISIBLE_DEVICES=0 (GPU) or -1 (CPU); baselines are CPU libs.
