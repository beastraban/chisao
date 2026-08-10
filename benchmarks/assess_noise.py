#!/usr/bin/env python3
"""
Assess noise levels in a likelihood function to determine appropriate
filter thresholds for SunBURST.

Usage:
    noise_stats = assess_noise(log_likelihood, test_point, bounds)
    print(f"Recommended stick_tolerance: {noise_stats['grad_noise_linf'] * 10:.2e}")
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional

def assess_noise(
    log_likelihood: Callable,
    test_point: np.ndarray,
    bounds: np.ndarray,
    n_samples: int = 20,
    eps_range: Tuple[float, float] = (1e-6, 1e-3),
    verbose: bool = True
) -> Dict[str, float]:
    """
    Assess noise characteristics of a likelihood function.
    
    Measures:
    1. Function value noise (if stochastic)
    2. Gradient noise via finite differences
    3. Hessian diagonal noise
    
    Args:
        log_likelihood: Function to assess
        test_point: Point to evaluate at (should be in a smooth region)
        bounds: Parameter bounds [D, 2]
        n_samples: Number of samples for noise estimation
        eps_range: (min_eps, max_eps) for finite difference steps
        verbose: Print diagnostic info
        
    Returns:
        Dictionary with noise statistics and recommended thresholds
    """
    test_point = np.atleast_1d(test_point)
    D = len(test_point)
    
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Likelihood Function Noise Assessment")
        print("=" * 70)
        print(f"Dimension: {D}")
        print(f"Test point: {test_point}")
    
    # =========================================================================
    # 1. Function Value Noise
    # =========================================================================
    if verbose:
        print("\n[1. Function Value Noise]")
        print("  Evaluating at same point multiple times...")
    
    f_values = np.array([log_likelihood(test_point.reshape(1, -1))[0] for _ in range(n_samples)])
    f_mean = np.mean(f_values)
    f_std = np.std(f_values)
    f_range = np.max(f_values) - np.min(f_values)
    
    results['function_std'] = f_std
    results['function_range'] = f_range
    
    if verbose:
        print(f"  Mean: {f_mean:.6e}")
        print(f"  Std:  {f_std:.6e}")
        print(f"  Range: {f_range:.6e}")
        if f_std < 1e-10:
            print("  → Function appears deterministic")
        else:
            print(f"  → Function has stochastic noise: σ ≈ {f_std:.2e}")
    
    # =========================================================================
    # 2. Gradient Noise (L2 and L∞ norms)
    # =========================================================================
    if verbose:
        print("\n[2. Gradient Noise via Finite Differences]")
    
    # Test multiple epsilon values
    eps_values = np.logspace(np.log10(eps_range[0]), np.log10(eps_range[1]), 5)
    
    gradient_estimates_l2 = []
    gradient_estimates_linf = []
    
    for eps in eps_values:
        # Central difference gradient
        grad = np.zeros(D)
        for d in range(D):
            e_d = np.zeros(D)
            e_d[d] = 1.0
            
            x_plus = test_point + eps * e_d
            x_minus = test_point - eps * e_d
            
            # Clip to bounds
            x_plus = np.clip(x_plus, bounds[:, 0], bounds[:, 1])
            x_minus = np.clip(x_minus, bounds[:, 0], bounds[:, 1])
            
            f_plus = log_likelihood(x_plus.reshape(1, -1))[0]
            f_minus = log_likelihood(x_minus.reshape(1, -1))[0]
            
            grad[d] = (f_plus - f_minus) / (2 * eps)
        
        gradient_estimates_l2.append(np.linalg.norm(grad))
        gradient_estimates_linf.append(np.max(np.abs(grad)))
    
    grad_l2_std = np.std(gradient_estimates_l2)
    grad_linf_std = np.std(gradient_estimates_linf)
    grad_l2_range = np.max(gradient_estimates_l2) - np.min(gradient_estimates_l2)
    grad_linf_range = np.max(gradient_estimates_linf) - np.min(gradient_estimates_linf)
    
    results['grad_noise_l2'] = grad_l2_std
    results['grad_noise_linf'] = grad_linf_std
    results['grad_l2_range'] = grad_l2_range
    results['grad_linf_range'] = grad_linf_range
    
    if verbose:
        print(f"  Tested eps: {eps_values}")
        print(f"  Gradient L2 norms: {gradient_estimates_l2}")
        print(f"  Gradient L∞ norms: {gradient_estimates_linf}")
        print(f"  L2 variation: {grad_l2_std:.2e} (range: {grad_l2_range:.2e})")
        print(f"  L∞ variation: {grad_linf_std:.2e} (range: {grad_linf_range:.2e})")
    
    # =========================================================================
    # 3. Hessian Diagonal Noise
    # =========================================================================
    if verbose:
        print("\n[3. Hessian Diagonal Noise]")
    
    # Use a reasonable epsilon
    eps = 1e-4
    
    f_center = log_likelihood(test_point.reshape(1, -1))[0]
    hessian_diag_estimates = []
    
    for _ in range(5):  # Multiple estimates
        diag_H = np.zeros(D)
        for d in range(D):
            e_d = np.zeros(D)
            e_d[d] = 1.0
            
            x_plus = test_point + eps * e_d
            x_minus = test_point - eps * e_d
            
            x_plus = np.clip(x_plus, bounds[:, 0], bounds[:, 1])
            x_minus = np.clip(x_minus, bounds[:, 0], bounds[:, 1])
            
            f_plus = log_likelihood(x_plus.reshape(1, -1))[0]
            f_minus = log_likelihood(x_minus.reshape(1, -1))[0]
            
            diag_H[d] = (f_plus - 2*f_center + f_minus) / (eps**2)
        
        hessian_diag_estimates.append(diag_H)
    
    hessian_diag_estimates = np.array(hessian_diag_estimates)
    hessian_noise = np.std(hessian_diag_estimates, axis=0)
    hessian_noise_mean = np.mean(hessian_noise)
    hessian_noise_max = np.max(hessian_noise)
    
    results['hessian_noise_mean'] = hessian_noise_mean
    results['hessian_noise_max'] = hessian_noise_max
    
    if verbose:
        print(f"  Hessian diagonal estimates (5 trials):")
        for i, h in enumerate(hessian_diag_estimates):
            print(f"    Trial {i+1}: {h}")
        print(f"  Per-dimension noise: {hessian_noise}")
        print(f"  Mean noise: {hessian_noise_mean:.2e}")
        print(f"  Max noise:  {hessian_noise_max:.2e}")
    
    # =========================================================================
    # 4. Recommended Thresholds
    # =========================================================================
    # Conservative rule: threshold = 10× measured noise
    # For gradient filters, use L∞ norm (that's what ChiSao/GreenDragon use)
    
    stick_tol_recommended = max(grad_linf_range * 10, 1e-6)  # CarryTiger uses L∞
    grad_tol_recommended = max(grad_l2_std * 10, 1e-6)       # GreenDragon uses L2
    saddle_tol_recommended = max(hessian_noise_max * 10, 1e-6)
    
    # CRITICAL: Ensure grad_threshold >= stick_tolerance
    # CarryTiger filters first (L∞ norm), then GreenDragon filters (L2 norm)
    # If grad_threshold < stick_tolerance, peaks passing CarryTiger will fail GreenDragon
    grad_tol_recommended = max(grad_tol_recommended, stick_tol_recommended)
    
    # For fd_eps in fast mode: use function noise level or 1e-4, whichever is larger
    # Optimal eps ~ σ^(1/4) for second derivatives, but conservatively use σ
    if f_std > 1e-10:
        fd_eps_recommended = max(f_std, 1e-5)
    else:
        # For deterministic functions with numerical noise from FD
        # Use gradient noise as proxy
        fd_eps_recommended = max(grad_linf_std, 1e-5)
    
    results['stick_tolerance_recommended'] = stick_tol_recommended
    results['grad_threshold_recommended'] = grad_tol_recommended
    results['saddle_threshold_recommended'] = saddle_tol_recommended
    results['fd_eps_recommended'] = fd_eps_recommended
    
    if verbose:
        print("\n" + "=" * 70)
        print("RECOMMENDED THRESHOLDS (10× measured noise)")
        print("=" * 70)
        print(f"stick_tolerance  = {stick_tol_recommended:.2e}  # CarryTiger gradient filter")
        print(f"grad_threshold   = {grad_tol_recommended:.2e}  # GreenDragon gradient filter")
        if grad_tol_recommended > stick_tol_recommended * 1.01:
            print(f"                   (increased to {grad_tol_recommended:.2e} ≥ stick_tolerance for consistency)")
        print(f"saddle_threshold = {saddle_tol_recommended:.2e}  # GreenDragon saddle filter")
        print(f"fd_eps           = {fd_eps_recommended:.2e}  # GreenDragon FD step (fast mode)")
        print("\nUsage:")
        print("  compute_evidence(")
        print("      log_likelihood, bounds,")
        print(f"      stick_tolerance={stick_tol_recommended:.2e},")
        print(f"      grad_threshold={grad_tol_recommended:.2e},")
        print(f"      saddle_threshold={saddle_tol_recommended:.2e},")
        print(f"      fd_eps={fd_eps_recommended:.2e},")
        print("  )")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Example: test on noisy Gaussian
    import numpy as np
    
    np.random.seed(42)
    noise_level = 1e-4
    
    def noisy_gaussian(x):
        """Gaussian with additive noise"""
        x = np.atleast_2d(x)
        clean = -0.5 * np.sum(x**2, axis=1)
        noise = np.random.randn(len(x)) * noise_level
        return clean + noise
    
    test_point = np.zeros(6)
    bounds = np.array([[-5, 5]] * 6)
    
    results = assess_noise(
        noisy_gaussian,
        test_point,
        bounds,
        verbose=True
    )
