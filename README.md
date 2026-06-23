# ChiSao

[![PyPI](https://img.shields.io/pypi/v/chisao.svg)](https://pypi.org/project/chisao/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**C**onvergence-**H**alt-**I**nvert-**S**tick-**A**nd-**O**scillate — a GPU-native
population optimizer for finding **all** modes of a multimodal black-box function.

ChiSao runs an entire sample batch simultaneously and exploits a deliberate
convergence/anti-convergence oscillation cycle to escape local traps while
*freezing* confirmed modes. On a GPU, an entire population advances in the same
wall-clock time as a single sample, so cataloguing every mode of a hard
landscape costs roughly what one gradient step would cost on a CPU.

> The name is from Wing Chun *Chí Sǎo* ("sticky hands"): sensitivity drills that
> train simultaneous contact maintenance and redirection — the explore-and-freeze
> asymmetry at the heart of the algorithm.

ChiSao began as the mode-discovery engine of the
[SunBURST](https://pypi.org/project/sunburst-bayes/) Bayesian-evidence pipeline
and is published here as a standalone optimizer.

## Install

```bash
pip install chisao
```

GPU acceleration is optional. Install the CuPy wheel matching your CUDA driver
(check `nvidia-smi`):

```bash
pip install "chisao[cuda13]"   # CUDA 13.x driver
pip install "chisao[cuda12]"   # CUDA 12.x driver
pip install "chisao[cuda11]"   # CUDA 11.x driver
```

Without CuPy, ChiSao runs on NumPy (CPU) automatically. If CuPy is installed but
its CUDA libraries are broken or mismatched, ChiSao detects this at import and
falls back to CPU instead of crashing (check `chisao.GPU_OK`, or force CPU with
the `CHISAO_FORCE_CPU=1` environment variable).

## Quick start

```python
import numpy as np
from chisao import optimize

# Objective must return LOG-likelihood, batched: f(X[n, d]) -> [n].
# (Maximization. Negate a cost function to maximize it.)
def neg_rastrigin(X):
    A = 10.0
    d = X.shape[1]
    return -(A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1))

bounds = [(-5.12, 5.12)] * 8

peaks, logL = optimize(neg_rastrigin, bounds, seeder="carry_tiger", seed=0)
print(peaks.shape, "modes found; best logL =", logL.max())
```

## Seeding matters

ChiSao optimizes whatever **initial population** you hand it. *Where that
population starts is a first-class algorithmic choice, not a detail.* On
landscapes with large gradient-free regions (Ackley) or product-coupled minima
(Griewank), a uniformly random population never reaches the global basin, while
a population seeded by structured rays cast through the domain geometry does.

Two seeders ship in `chisao.seeding`:

| Seeder         | What it does                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `random`       | Uniform i.i.d. over the box. The baseline.                                   |
| `carry_tiger`  | Structured rays between vertices, vertex-to-edge, wall-to-wall, and a QR-orthonormal "sunburst" from the center — the seeding the paper shows is essential on flat/coupled landscapes. |

```python
from chisao import optimize, carry_tiger_seed, random_seed, sticky_hands

# High-level: seed + run in one call
peaks, logL = optimize(f, bounds, seeder="carry_tiger")

# Or build the population yourself and call the core optimizer directly
x0 = carry_tiger_seed(f, bounds, n_samples_per_ray=50, seed=0)
result = sticky_hands(f, x0, bounds=np.asarray(bounds, float), n_oscillations=3)
peaks, logL, widths = result["peaks"], result["L_peaks"], result["widths"]
```

## How it works

Each oscillation cycle runs six phases in fixed order:

1. **Convergence** — the whole population takes batched L-BFGS steps toward local maxima.
2. **Stick detection** — samples at true peaks (gradient-norm + quality gate) are *frozen*.
3. **Deduplication** — coincident peaks are merged in the L∞ metric (the load-bearing step).
4. **Reseeding** — *Repulse Monkey* (many free samples) or *Golden Rooster* (near exhaustion) refill the population.
5. **Hands Like Clouds** — unfrozen samples ascend a Gaussian-smoothed gradient to see global structure through local ripples.
6. **Anti-convergence** — momentum-based descent disperses unfrozen samples across valleys into new basins.

Frozen samples sit out the exploration phases but keep refining their peak.
The last oscillation skips phases 4–6 so the final output is clean.

## API

| Object | Purpose |
| ------ | ------- |
| `optimize(func, bounds, seeder=...)` | Seed a population and run ChiSao; returns `(peaks, L_peaks)`. |
| `sticky_hands(func, x0, ...)` | The core oscillation optimizer over a given population. |
| `carry_tiger_seed`, `random_seed`, `carry_tiger_rays` | Population seeders. |
| `lbfgs_batch`, `gradient_ascent_batch`, `optimize_batch` | Batched local optimizers. |
| `deduplicate_peaks_L_infinity`, `estimate_peak_width` | Mode post-processing. |
| `SingleWhip`, `randcoord_line_search_batch` | GPU batch toolkit (line search, scales, distances). |
| `get_gpu_info`, `get_array_module`, `GPU_OK` | Backend introspection / NumPy-or-CuPy selection. |

`func` must return **log**-likelihood and accept a batched input `X[n, d]`.

On a GPU, results are returned as CuPy arrays (call `.get()` for host copies);
on CPU they are NumPy arrays.

## Reproducing the paper

`benchmarks/sfu_benchmark.py` runs the 42-function Simon Fraser University suite
for both seeders across dimension. See `benchmarks/README.md` for the exact
settings used in the paper.

## Citation

```bibtex
@article{wolfson_chisao,
  title  = {ChiSao: A GPU-Native Parallel Optimizer for Multimodal Black-Box
            Functions via Convergence-Anticonvergence Oscillation},
  author = {Wolfson, Ira},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
