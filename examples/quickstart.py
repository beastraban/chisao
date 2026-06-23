"""
ChiSao quickstart: recover the global mode of a 2-D multimodal landscape,
comparing the random and carry_tiger seeders.

Run:  python examples/quickstart.py
"""

import numpy as np

from chisao import optimize


def neg_rastrigin(X):
    """Rastrigin as a LOG-likelihood to maximize (global max at the origin)."""
    A = 10.0
    d = X.shape[1]
    return -(A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1))


def main():
    bounds = [(-5.12, 5.12)] * 2

    for seeder in ("random", "carry_tiger"):
        peaks, logL = optimize(neg_rastrigin, bounds, seeder=seeder, seed=0, n_oscillations=3)
        n = 0 if peaks is None else len(peaks)
        if n:
            best = int(np.argmax(logL))
            loc = np.round(peaks[best], 3)
            print("[%-12s] modes found: %3d | best logL = %+.4f at %s" % (seeder, n, logL.max(), loc))
        else:
            print("[%-12s] modes found:   0 | (no peaks cleared the quality gate)" % seeder)


if __name__ == "__main__":
    main()
