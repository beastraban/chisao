# ChiSao SFU Benchmark

Reproduces the ChiSao mode-recovery results from the paper on the
[Simon Fraser University optimization test suite](https://www.sfu.ca/~ssurjano/optimization.html):
42 functions, both seeders (`random` and `carry_tiger`), across dimension.

This harness covers the **ChiSao** columns of the paper (random vs carry\_tiger
recovery rate and wall-clock). The CPU baselines (Differential Evolution,
Basin-Hopping, CMA-ES) are reported separately and are not part of this script.

## Run

From the repository root, with the package installed (`pip install -e .`):

```bash
# quick smoke (seconds on CPU)
python benchmarks/sfu_benchmark.py --funcs rastrigin sphere ackley --dims 2 --trials 2

# a multimodal group across low dimensions
python benchmarks/sfu_benchmark.py --group multimodal --dims 2 4 8 --trials 10

# the full suite, paper dimensions, JSON output
python benchmarks/sfu_benchmark.py --dims 2 4 8 16 32 64 --trials 10 --out sfu_results.json
```

CLI options:

| Flag           | Meaning                                                        |
| -------------- | ------------------------------------------------------------- |
| `--funcs`      | Specific functions (names from the registry).                 |
| `--group`      | One of `multimodal`, `multimodal_2d`, `bowl`, `valley`, ...   |
| `--dims`       | Dimensions to test (default `2 4 8`).                         |
| `--trials`     | Independent trials per function/dim (default `5`).            |
| `--out`        | Write full results to a JSON file.                            |
| `--source-dir` | Point at a working tree instead of the installed package.     |

By default the harness does `import chisao` (the installed package). GPU is used
automatically when CuPy is present; otherwise it runs on NumPy (the high
dimensions are slow on CPU — use a GPU for `d >= 16`).

## Paper run settings

Each trial seeds a population and runs `chisao.sticky_hands` with the settings
below (see `run_one` in `sfu_benchmark.py`). These are the settings that
reproduce the paper — in particular, the `random` seeder only matches the
paper's numbers with `reseed_strategy='sunburst'` and `cannon_through_sky=True`:

```python
N = 50 * (10 + ceil(log2(max(2, D))))      # population size
params = dict(
    method='lbfgs',
    n_converge=10,
    n_anticonverge=5,
    n_oscillations=3,
    stick_tolerance=1e-3,
    reseed_strategy='sunburst',            # Repulse Monkey
    cannon_through_sky=True,
    estimate_widths=False,
)
```

The two seeders (`seed_random`, `seed_carry_tiger`) are defined in the script.
`seed_carry_tiger` is the same 4-component ray recipe exposed by the package as
`chisao.carry_tiger_seed` (vertex-to-vertex, vertex-to-edge, wall-to-wall, and a
QR-orthonormal sunburst from the domain center), with
`n_rays = 10 + log2(d)` and 50 samples per ray.

## Success criterion

A trial succeeds if a recovered peak lies within the function's L∞ tolerance of
a global optimum (`tol` in `FUNC_REGISTRY`). Functions with multiple equivalent
optima (`multi_opt=True`) are matched against the nearest of their global optima.
