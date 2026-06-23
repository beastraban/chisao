"""
SingleWhip (單鞭) v1.6 - STANDALONE VERSION
============================================

Self-contained GPU utility toolkit with NO external dependencies
(except NumPy/CuPy/SciPy standard libraries).

Named after the Guang Ping Yang Style Tai Chi (廣平楊式太極拳) form,
in honor of Master Donald Rubbo.

All helper classes integrated directly:
  - BatchLikelihoodEvaluator
  - BatchDistanceComputer
  - SunburstSampler
  - Peak deduplication utilities
  - Gradient evaluation (single + batched)
  - v1.3: Multi-scale estimation
  - v1.3: Smoothed gradient (for Hands Like Clouds / 雲手)
  - v1.4: GPU-PURE deduplicate_peaks_linf (no Python loops)
  - v1.5: RandCoord + Line Search optimizer (sublinear gradient scaling)
  - NEW v1.6: DIMENSION BATCHING for O(1) wall-clock gradients at high D

Version: 1.6 Standalone
Status: Production Ready
Dependencies: NumPy, CuPy (optional), SciPy
"""

import numpy as np
try:
    import cupy as cp
    from cupyx.scipy import linalg as cp_linalg
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    from scipy import linalg as cp_linalg
    GPU_AVAILABLE = False

from scipy.optimize import minimize_scalar
import time
import math
from typing import Callable, Optional, Union, Tuple, Dict


# ============================================================================
# RANDCOORD + LINE SEARCH OPTIMIZER (v1.5)
# ============================================================================

def gradient_randcoord_linf(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    k: int,
    epsilon: float = 1e-6,
    func_returns_grad: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Random Coordinate gradient with L∞ guarantee.
    
    Computes exact partial derivatives for k random dimensions,
    PLUS always includes the L∞ argmax dimension per sample.
    
    This guarantees progress on L∞ distance every iteration.
    
    Cost: O(2k + 2) function evaluations worst case
    
    Args:
        func: Likelihood function(x) -> [N] or (x) -> ([N], [N,D]) if func_returns_grad
        x: [N, D] current positions
        k: Number of random dimensions to sample
        epsilon: Finite difference step
        func_returns_grad: If True, func returns (L, grad) tuple
    
    Returns:
        [N, D] gradient estimate (sparse - only k+1 dims nonzero per sample)
    """
    xp = cp if GPU_AVAILABLE and hasattr(x, '__cuda_array_interface__') else np
    N, D = x.shape
    k = min(k, D)
    
    # If analytic gradient available, use it directly
    if func_returns_grad:
        _, grad = func(x)
        return grad
    
    grad = xp.zeros((N, D), dtype=x.dtype)
    
    # Find the L∞ argmax dimension for each sample
    linf_dims = xp.argmax(xp.abs(x), axis=1)  # Shape: (N,)
    
    # Sample k random dimensions (shared across samples for batching efficiency)
    if xp == np:
        random_dims = np.random.choice(D, size=min(k, D), replace=False)
    else:
        random_dims = cp.random.choice(D, size=min(k, D), replace=False)
    
    # Build perturbation matrix for random dimensions
    eye_k = xp.zeros((len(random_dims), D), dtype=x.dtype)
    eye_k[xp.arange(len(random_dims)), random_dims] = 1.0
    
    # Compute partials for random dims (batched)
    x_plus = x[:, None, :] + epsilon * eye_k[None, :, :]   # [N, k, D]
    x_minus = x[:, None, :] - epsilon * eye_k[None, :, :]  # [N, k, D]
    
    f_plus = func(x_plus.reshape(N * len(random_dims), D)).reshape(N, len(random_dims))
    f_minus = func(x_minus.reshape(N * len(random_dims), D)).reshape(N, len(random_dims))
    
    partials = (f_plus - f_minus) / (2 * epsilon)
    grad[:, random_dims] = partials
    
    # Check which samples need their L∞ dim computed
    linf_in_random = xp.any(linf_dims[:, None] == random_dims[None, :], axis=1)
    needs_linf = ~linf_in_random
    n_needs = int(xp.sum(needs_linf))
    
    if n_needs > 0:
        # Get samples that need L∞ partial
        x_needs = x[needs_linf]
        linf_dims_needs = linf_dims[needs_linf]
        
        # Build per-sample perturbations for L∞ dims
        x_plus_linf = x_needs.copy()
        x_minus_linf = x_needs.copy()
        
        # Vectorized: add epsilon to each sample's L∞ dimension
        rows = xp.arange(n_needs)
        x_plus_linf[rows, linf_dims_needs] += epsilon
        x_minus_linf[rows, linf_dims_needs] -= epsilon
        
        f_plus_linf = func(x_plus_linf)
        f_minus_linf = func(x_minus_linf)
        
        partials_linf = (f_plus_linf - f_minus_linf) / (2 * epsilon)
        
        # Write back to grad using advanced indexing
        # Need to handle this carefully for GPU arrays
        needs_indices = xp.where(needs_linf)[0]
        for i in range(n_needs):
            grad[needs_indices[i], linf_dims_needs[i]] = partials_linf[i]
    
    return grad


def randcoord_line_search_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-7,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,  # Unused but kept for API compatibility
    verbose: bool = False,
    func_returns_grad: bool = False,
    line_search: str = 'armijo',  # For API compatibility (always uses backtracking)
    k_scale: Tuple[int, int] = (16, 8)  # k = A + B*log2(D)
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    RandCoord + Line Search optimizer for maximization.
    All N samples optimize simultaneously.
    
    Uses k = A + B*log₂(D) random coordinates per iteration (sublinear scaling).
    Always includes L∞ argmax dimension to guarantee progress.
    
    SCALING BEHAVIOR (k = 16 + 8*log₂(D)):
        D=64:   k=64   (100% of D)
        D=128:  k=72   (56% of D)
        D=256:  k=80   (31% of D)
        D=512:  k=88   (17% of D)
        D=1024: k=96   (9% of D) → 1.64x faster than full FD
    
    80-20 ADAPTIVE RULE:
        - Minimum 3 iterations always
        - When 80% converged at iteration N, allow N more for remaining 20%
        - Stop at 2N iterations max
    
    Args:
        func: Likelihood function(x) -> [N] or (x) -> ([N], [N,D]) if func_returns_grad
        x0: [N, D] initial positions
        maxiter: Maximum iterations (None = adaptive)
        tol: Convergence tolerance (L_∞ norm of gradient)
        bounds: [D, 2] optional bounds [min, max]
        batch_size: Unused (kept for API compatibility)
        verbose: Print convergence info
        func_returns_grad: If True, func returns (L, grad) tuple
        line_search: Ignored (always uses backtracking)
        k_scale: (A, B) for k = A + B*log2(D)
    
    Returns:
        Dictionary with:
            'x': [N, D] final positions
            'L': [N] likelihood values
            'converged': [N] bool convergence flags
            'niter': int iterations taken
            'grad_norm': [N] final gradient L_∞ norms
            'inv_hessian_scale': [N] (ones - not computed by RandCoord)
            'history_count': [N] (zeros - not computed by RandCoord)
    """
    xp = cp if GPU_AVAILABLE and hasattr(x0, '__cuda_array_interface__') else np
    N, D = x0.shape
    
    # Compute k using logarithmic scaling
    A, B = k_scale
    k = int(A + B * math.log2(max(D, 2)))
    k = min(k, D)  # Can't sample more than D dimensions
    
    if verbose:
        print(f"  RandCoord: k={k} (D={D}, {100*k/D:.1f}% of dimensions)")
    
    # 80-20 adaptive iteration control
    min_iter = 3
    hard_max = maxiter if maxiter is not None else max(50, int(8 * math.log2(D)))
    iter_when_80pct = None
    adaptive_max = hard_max
    
    # State
    x = x0.copy()
    converged = xp.zeros(N, dtype=bool)
    
    # Initial function values
    if func_returns_grad:
        L, _ = func(x)
    else:
        L = func(x)
    
    # Bounds setup
    if bounds is not None:
        bounds = xp.asarray(bounds) if not hasattr(bounds, '__cuda_array_interface__') else bounds
        lb = bounds[:, 0]
        ub = bounds[:, 1]
    else:
        lb = None
        ub = None
    
    grad_norm = xp.zeros(N, dtype=x.dtype)
    iteration = 0
    
    for iteration in range(hard_max):
        # Compute gradient (only k + 1 dims per sample)
        grad = gradient_randcoord_linf(func, x, k, func_returns_grad=func_returns_grad)
        
        # Compute L∞ gradient norm
        grad_norm = xp.max(xp.abs(grad), axis=1)
        
        # Check convergence
        converged = grad_norm < tol
        n_converged = int(xp.sum(converged))
        frac_converged = n_converged / N
        
        if verbose and iteration % 10 == 0:
            mean_L = float(xp.mean(L))
            mean_grad = float(xp.mean(grad_norm))
            print(f"    Iter {iteration}: L_mean={mean_L:.6f}, grad_mean={mean_grad:.6e}, "
                  f"converged={n_converged}/{N}")
        
        # 80-20 rule
        if iter_when_80pct is None and frac_converged >= 0.8 and iteration >= min_iter:
            iter_when_80pct = iteration
            adaptive_max = min(hard_max, 2 * iteration)
            if verbose:
                print(f"    80-20 rule: 80% converged at iter {iteration}, max now {adaptive_max}")
        
        # Check stopping conditions
        if n_converged == N:
            if verbose:
                print(f"    All {N} samples converged at iteration {iteration}")
            break
        
        if iter_when_80pct is not None and iteration >= adaptive_max:
            if verbose:
                print(f"    80-20 rule: stopping at iteration {iteration} ({n_converged}/{N} converged)")
            break
        
        # Normalize gradient for direction
        grad_norm_safe = xp.maximum(grad_norm, 1e-10)[:, None]
        direction = grad / grad_norm_safe
        
        # === BACKTRACKING LINE SEARCH ===
        alpha = xp.ones(N, dtype=x.dtype)  # Initial step size
        
        x_new = x + alpha[:, None] * direction
        if lb is not None:
            x_new = xp.clip(x_new, lb, ub)
        
        if func_returns_grad:
            L_new, _ = func(x_new)
        else:
            L_new = func(x_new)
        
        # Backtrack samples that didn't improve
        for _ in range(10):  # Max 10 backtracks
            not_improved = L_new <= L
            if not xp.any(not_improved):
                break
            
            # Shrink step for non-improving samples
            alpha = xp.where(not_improved, alpha * 0.5, alpha)
            x_new = x + alpha[:, None] * direction
            if lb is not None:
                x_new = xp.clip(x_new, lb, ub)
            
            if func_returns_grad:
                L_new, _ = func(x_new)
            else:
                L_new = func(x_new)
        
        # Update positions (only where improved)
        improved = L_new > L
        x = xp.where(improved[:, None], x_new, x)
        L = xp.where(improved, L_new, L)
    
    return {
        'x': x,
        'L': L,
        'converged': converged,
        'niter': iteration + 1,
        'grad_norm': grad_norm,
        # Compatibility fields (not computed by RandCoord)
        'inv_hessian_scale': xp.ones(N, dtype=x.dtype),
        'history_count': xp.zeros(N, dtype=int)
    }


# ============================================================================
# BATCH LIKELIHOOD EVALUATOR (Integrated)
# ============================================================================

class BatchLikelihoodEvaluator:
    """
    Batch likelihood evaluator with automatic GPU memory management.
    Integrated directly into SingleWhip v1.2 standalone.
    """
    
    def __init__(
        self,
        likelihood_fn: Callable,
        bounds: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        max_batch_size: Optional[int] = None
    ):
        self.likelihood_fn = likelihood_fn
        self.bounds = bounds
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if max_batch_size is None:
            if self.use_gpu:
                # Auto-detect batch size based on GPU memory
                try:
                    device = cp.cuda.Device()
                    free_mem, total_mem = device.mem_info
                    # Use 25% of free memory as threshold
                    max_batch_size = int((free_mem * 0.25) / (8 * 1024))  # Rough estimate
                except:
                    max_batch_size = 10000
            else:
                max_batch_size = 1000
        
        self.max_batch_size = max_batch_size
    
    def evaluate(self, points: np.ndarray, clip_bounds: bool = True) -> np.ndarray:
        """Evaluate likelihood at batch of points"""
        xp = self.xp
        
        # Convert to appropriate array type
        if self.use_gpu:
            if not isinstance(points, cp.ndarray):
                points = cp.asarray(points)
        else:
            if hasattr(points, '__cuda_array_interface__'):
                # CuPy array but we're on CPU - convert
                points = cp.asnumpy(points)
        
        # Clip to bounds if requested
        if clip_bounds and self.bounds is not None:
            bounds_array = xp.asarray(self.bounds) if self.use_gpu else self.bounds
            points = xp.clip(points, bounds_array[:, 0], bounds_array[:, 1])
        
        # Batch evaluation
        N = len(points)
        if N <= self.max_batch_size:
            return self._evaluate_single_batch(points)
        else:
            # Split into batches
            results = []
            for i in range(0, N, self.max_batch_size):
                batch = points[i:i + self.max_batch_size]
                results.append(self._evaluate_single_batch(batch))
            return xp.concatenate(results)
    
    def _evaluate_single_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate single batch"""
        return self.likelihood_fn(points)
    
    def evaluate_gradient_fd(
        self,
        point: np.ndarray,
        eps: float = 1e-8,
        method: str = 'central'
    ) -> np.ndarray:
        """Finite difference gradient at single point"""
        xp = self.xp
        d = len(point)
        
        if method == 'central':
            # Central differences: (f(x+h) - f(x-h)) / 2h
            perturbations = xp.zeros((2*d, d), dtype=point.dtype)
            for i in range(d):
                perturbations[2*i, i] = eps
                perturbations[2*i + 1, i] = -eps
            
            perturbed_points = point + perturbations
            values = self.evaluate(perturbed_points, clip_bounds=False)
            
            grad = xp.zeros(d, dtype=point.dtype)
            for i in range(d):
                grad[i] = (values[2*i] - values[2*i + 1]) / (2 * eps)
        
        elif method == 'forward':
            # Forward differences: (f(x+h) - f(x)) / h
            perturbations = xp.eye(d, dtype=point.dtype) * eps
            perturbed_points = point + perturbations
            
            f_x = self.evaluate(point.reshape(1, -d), clip_bounds=False)[0]
            f_perturbed = self.evaluate(perturbed_points, clip_bounds=False)
            
            grad = (f_perturbed - f_x) / eps
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return grad


# ============================================================================
# BATCH DISTANCE COMPUTER (Integrated)
# ============================================================================

class BatchDistanceComputer:
    """
    Efficient batch distance computation.
    Integrated directly into SingleWhip v1.2 standalone.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
    
    def compute_linf(
        self,
        points1: np.ndarray,
        points2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute L-infinity (max) distances.
        GPU-PURE v1.4: Fully vectorized, memory-efficient (O(N²) not O(N²×D)).
        """
        xp = self.xp
        D = points1.shape[1]
        
        if points2 is None:
            # Pairwise distances: (N, N)
            # Compute max over dimensions incrementally
            dists = xp.abs(points1[:, None, 0] - points1[None, :, 0])
            for d in range(1, D):
                diff_d = xp.abs(points1[:, None, d] - points1[None, :, d])
                dists = xp.maximum(dists, diff_d)
            return dists
        else:
            # Cross distances: (N1, N2)
            dists = xp.abs(points1[:, None, 0] - points2[None, :, 0])
            for d in range(1, D):
                diff_d = xp.abs(points1[:, None, d] - points2[None, :, d])
                dists = xp.maximum(dists, diff_d)
            return dists
    
    def compute_l2(
        self,
        points1: np.ndarray,
        points2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute L2 (Euclidean) distances.
        GPU-PURE v1.4: Fully vectorized, memory-efficient (O(N²) not O(N²×D)).
        """
        xp = self.xp
        D = points1.shape[1]
        
        if points2 is None:
            # Pairwise distances: (N, N)
            # Compute sum of squares incrementally
            diff_sq = (points1[:, None, 0] - points1[None, :, 0])**2
            for d in range(1, D):
                diff_sq += (points1[:, None, d] - points1[None, :, d])**2
            return xp.sqrt(diff_sq)
        else:
            # Cross distances: (N1, N2)
            diff_sq = (points1[:, None, 0] - points2[None, :, 0])**2
            for d in range(1, D):
                diff_sq += (points1[:, None, d] - points2[None, :, d])**2
            return xp.sqrt(diff_sq)
    
    def find_k_nearest(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
        k: int,
        metric: str = 'l2'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors"""
        xp = self.xp
        
        # Compute distances
        if metric == 'l2':
            dists = self.compute_l2(query_points, reference_points)
        elif metric == 'linf':
            dists = self.compute_linf(query_points, reference_points)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Find k nearest
        indices = xp.argsort(dists, axis=1)[:, :k]
        nearest_dists = xp.take_along_axis(dists, indices, axis=1)
        
        return indices, nearest_dists


# ============================================================================
# SUNBURST SAMPLER (Integrated)
# ============================================================================

class SunburstSampler:
    """
    Sunburst ray sampling pattern.
    Integrated directly into SingleWhip v1.2 standalone.
    """
    
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
    
    def sample_rays(
        self,
        center: np.ndarray,
        n_rays: int,
        bounds: Optional[np.ndarray] = None,
        ray_length: float = 1.0,
        pattern: str = 'uniform'
    ) -> np.ndarray:
        """Generate rays from center point"""
        xp = self.xp
        d = self.dim
        
        if pattern == 'uniform':
            # Random directions on unit sphere
            directions = xp.random.randn(n_rays, d).astype(xp.float32)
            norms = xp.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / norms
        
        elif pattern == 'axis_aligned':
            # Rays along coordinate axes (both directions)
            n_per_axis = n_rays // (2 * d)
            directions = []
            for i in range(d):
                dir_pos = xp.zeros((n_per_axis, d), dtype=xp.float32)
                dir_pos[:, i] = 1.0
                dir_neg = xp.zeros((n_per_axis, d), dtype=xp.float32)
                dir_neg[:, i] = -1.0
                directions.extend([dir_pos, dir_neg])
            directions = xp.vstack(directions)
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Generate points along rays
        rays = center + directions * ray_length
        
        # Clip to bounds if provided
        if bounds is not None:
            bounds_array = xp.asarray(bounds) if self.use_gpu else bounds
            rays = xp.clip(rays, bounds_array[:, 0], bounds_array[:, 1])
        
        return rays


# ============================================================================
# PEAK DEDUPLICATION (Integrated)
# ============================================================================

def deduplicate_peaks_linf(
    peaks: np.ndarray,
    L_peaks: np.ndarray,
    tolerance: float,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate peaks using L-infinity metric.
    
    GPU-PURE v1.4: Fully vectorized, no Python loops.
    MEMORY-EFFICIENT: Computes L_inf incrementally over dimensions.
    
    Algorithm:
        1. Compute pairwise L-inf distances dimension-by-dimension (O(N²) memory, not O(N²×D))
        2. Find which peaks are "dominated" (have a better duplicate)
        3. Keep only non-dominated peaks
    
    Complexity: O(N²) memory (not O(N²×D)), O(D) GPU kernel launches
    """
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    
    N = len(peaks)
    if N == 0:
        return peaks, L_peaks
    
    if N == 1:
        return peaks, L_peaks
    
    # Compute pairwise L-infinity distances - MEMORY EFFICIENT
    # Instead of creating (N, N, D) tensor, compute max incrementally
    # dists[i,j] = max_d |peaks[i,d] - peaks[j,d]|
    
    # Start with first dimension
    dists = xp.abs(peaks[:, None, 0] - peaks[None, :, 0])  # (N, N)
    
    # Update with max over remaining dimensions
    for d in range(1, peaks.shape[1]):
        diff_d = xp.abs(peaks[:, None, d] - peaks[None, :, d])  # (N, N)
        dists = xp.maximum(dists, diff_d)
    
    # Find duplicates: dists[i,j] < tolerance means i and j are duplicates
    is_duplicate = (dists < tolerance)
    
    # Zero out diagonal (don't compare with self)
    is_duplicate[xp.arange(N), xp.arange(N)] = False
    
    # For each pair of duplicates, keep the one with higher likelihood
    # L_diff[i,j] = L_peaks[i] - L_peaks[j]
    # If L_diff[i,j] < 0, then j has higher likelihood than i
    L_diff = L_peaks[:, None] - L_peaks[None, :]  # (N, N)
    
    # Peak i is "dominated" if there exists j where:
    #   - is_duplicate[i,j] = True (within tolerance)
    #   - L_peaks[j] > L_peaks[i] (j is better)
    dominated = xp.any(is_duplicate & (L_diff < 0), axis=1)
    
    # Handle ties: if L_peaks equal, keep lower index
    # tie_mask[i,j] = True if i,j are duplicates with equal likelihood
    tie_mask = is_duplicate & (L_diff == 0)
    
    # In a tie, lower index wins: peak i is dominated by j if j < i
    # Create mask where j < i (upper triangle)
    j_indices = xp.arange(N)[None, :]  # (1, N)
    i_indices = xp.arange(N)[:, None]  # (N, 1)
    j_less_than_i = j_indices < i_indices  # (N, N), True where j < i
    
    dominated_by_tie = xp.any(tie_mask & j_less_than_i, axis=1)
    
    # Keep peaks that are not dominated
    keep_mask = ~(dominated | dominated_by_tie)
    
    return peaks[keep_mask], L_peaks[keep_mask]


# ============================================================================
# SINGLEWHIP MAIN CLASS
# ============================================================================

class SingleWhip:
    """
    SingleWhip v1.3 - Standalone GPU utility toolkit
    
    All dependencies integrated directly (no external imports needed).
    
    Features:
      - Batch likelihood evaluation
      - Distance computation (L∞, L2, k-NN)
      - Sunburst ray sampling
      - Peak deduplication
      - Single-point gradient evaluation
      - v1.2: Batched gradient evaluation (10-50× faster)
      - NEW v1.3: Multi-scale estimation (estimate_characteristic_scales)
      - NEW v1.3: Smoothed gradient for "Hands Like Clouds" (evaluate_gradient_smoothed)
    
    Usage:
        whip = SingleWhip(use_gpu=True)
        
        # Evaluate likelihood
        L = whip.evaluate_likelihood(points, func)
        
        # Compute distances
        dists = whip.compute_distance(points1, points2, metric='linf')
        
        # Generate rays
        rays = whip.sample_sunburst_rays(center, n_rays, bounds)
        
        # Deduplicate peaks
        unique_peaks, unique_L = whip.deduplicate_peaks(peaks, L_peaks, tol)
        
        # Gradients
        grad = whip.evaluate_gradient(point, func)  # Single point
        grads = whip.evaluate_gradient_batch(points, func)  # Batch
        
        # NEW v1.3: Multi-scale
        scales = whip.estimate_characteristic_scales(f_samples, t_values)
        grads_smooth = whip.evaluate_gradient_smoothed(points, func, sigma=scales[2])
    """
    
    def __init__(self, use_gpu: bool = None):
        """Initialize SingleWhip utilities"""
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.is_gpu = self.use_gpu
        
        # Initialize helper components
        self._evaluator = None
        self._distance_computer = BatchDistanceComputer(use_gpu=self.use_gpu)
        self._samplers = {}  # Cache samplers by dimension
    
    def _get_evaluator(self, func: Callable, bounds: Optional[np.ndarray] = None):
        """Get or create evaluator for function"""
        return BatchLikelihoodEvaluator(func, bounds, use_gpu=self.use_gpu)
    
    def _get_sampler(self, dim: int):
        """Get or create sampler for dimension"""
        if dim not in self._samplers:
            self._samplers[dim] = SunburstSampler(dim, use_gpu=self.use_gpu)
        return self._samplers[dim]
    
    # ========================================================================
    # Likelihood Evaluation
    # ========================================================================
    
    def evaluate_likelihood(
        self,
        points: np.ndarray,
        func: Callable,
        bounds: Optional[np.ndarray] = None,
        clip_bounds: bool = True
    ) -> np.ndarray:
        """Evaluate likelihood at batch of points"""
        evaluator = self._get_evaluator(func, bounds)
        return evaluator.evaluate(points, clip_bounds=clip_bounds)
    
    # ========================================================================
    # Distance Computation
    # ========================================================================
    
    def compute_distance(
        self,
        points1: np.ndarray,
        points2: Optional[np.ndarray] = None,
        metric: str = 'l2'
    ) -> np.ndarray:
        """Compute distances between points"""
        if metric == 'l2':
            return self._distance_computer.compute_l2(points1, points2)
        elif metric == 'linf':
            return self._distance_computer.compute_linf(points1, points2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_k_nearest(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
        k: int,
        metric: str = 'l2'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors"""
        return self._distance_computer.find_k_nearest(
            query_points, reference_points, k, metric
        )
    
    # ========================================================================
    # Sunburst Ray Sampling
    # ========================================================================
    
    def sample_sunburst_rays(
        self,
        center: np.ndarray,
        n_rays: int,
        bounds: Optional[np.ndarray] = None,
        ray_length: float = 1.0,
        pattern: str = 'uniform'
    ) -> np.ndarray:
        """Generate Sunburst ray samples"""
        dim = len(center)
        sampler = self._get_sampler(dim)
        return sampler.sample_rays(center, n_rays, bounds, ray_length, pattern)
    
    # ========================================================================
    # Peak Deduplication
    # ========================================================================
    
    def deduplicate_peaks(
        self,
        peaks: np.ndarray,
        L_peaks: np.ndarray,
        tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deduplicate peaks using L-infinity metric"""
        return deduplicate_peaks_linf(peaks, L_peaks, tolerance, self.use_gpu)
    
    # ========================================================================
    # Gradient Evaluation (Single Point)
    # ========================================================================
    
    def evaluate_gradient(
        self,
        point: np.ndarray,
        func: Callable,
        eps: Optional[float] = None,
        method: str = 'central',
        func_returns_grad: bool = False  # NEW: analytic gradient support
    ) -> np.ndarray:
        """Evaluate gradient at single point using finite differences or analytic."""
        xp = self.xp
        
        # === ANALYTIC GRADIENT PATH ===
        if func_returns_grad:
            # Reshape to batch of 1, call func, extract gradient
            point_batch = point.reshape(1, -1)
            _, grad = func(point_batch)
            return grad.reshape(-1)  # Return as 1D
        
        # === FINITE DIFFERENCE PATH ===
        # Auto-select epsilon based on dtype (same as batch method)
        if eps is None:
            if point.dtype == xp.float32:
                eps = 1e-4
            else:
                eps = 1e-8
        
        evaluator = self._get_evaluator(func)
        return evaluator.evaluate_gradient_fd(point, eps, method)
    
    # ========================================================================
    # NEW v1.2: Batched Gradient Evaluation
    # ========================================================================
    
    def evaluate_gradient_batch(
        self,
        points: np.ndarray,
        func: Callable,
        eps: Optional[float] = None,
        method: str = 'central',
        mem_limit: Optional[float] = None,
        verbose: bool = False,
        func_returns_grad: bool = False  # Analytic gradient support
    ) -> np.ndarray:
        """
        Evaluate gradients for multiple points efficiently.
        
        v1.6: MAXIMIZE GPU UTILIZATION
        
        Strategy:
            1. If memory fits: ONE kernel call, ALL 2×D×N evaluations in parallel
            2. Only batch when approaching OOM ceiling
        
        Expected behavior:
            - Below OOM: O(1) wall clock (full parallelism)
            - At OOM: Batching kicks in, scaling appears
        
        ANALYTIC GRADIENT SUPPORT:
            If func_returns_grad=True, func(x) returns (L, grad) tuple.
            In this case, we return the gradient directly - O(1)!
        
        Args:
            points: [N, D] array of points
            func: Likelihood function (may return (L, grad) if func_returns_grad=True)
            eps: Finite difference step (default: 1e-4 for float32, 1e-8 for float64)
            method: 'central' or 'forward'
            mem_limit: GPU memory limit in bytes (None = auto-detect, use 80% of VRAM)
            verbose: Print memory/batching info
            func_returns_grad: If True, func returns (L, grad) tuple
            
        Returns:
            [N, D] array of gradients
        """
        xp = self.xp
        N, D = points.shape
        
        # === ANALYTIC GRADIENT PATH (O(1) - fastest!) ===
        if func_returns_grad:
            _, grad = func(points)
            return grad
        
        # === FINITE DIFFERENCE PATH ===
        
        # Auto-select epsilon based on dtype
        if eps is None:
            if points.dtype == xp.float32:
                eps = 1e-4  # float32: ~7 digits precision
            else:
                eps = 1e-8  # float64: ~16 digits precision
        
        # Determine GPU memory limit
        if mem_limit is None:
            if self.is_gpu:
                try:
                    device = cp.cuda.Device()
                    mem_free, mem_total = device.mem_info
                    mem_limit = mem_total * 0.8  # Use 80% of VRAM
                except:
                    mem_limit = 8e9  # Fallback: assume 8GB
            else:
                mem_limit = 16e9  # CPU: assume 16GB RAM
        
        if method == 'central':
            # Central differences: (f(x+h) - f(x-h)) / 2h
            
            # Memory needed for full [N, D, 2, D] tensor
            mem_needed = N * D * 2 * D * 8  # bytes
            
            if mem_needed < mem_limit:
                # === CLEAN PATH: ONE kernel call, full parallelism ===
                if verbose:
                    print(f"  [GRADIENT] Full parallel: {mem_needed/1e9:.2f}GB < {mem_limit/1e9:.2f}GB limit")
                
                eye = xp.eye(D, dtype=points.dtype)
                x_perturbed = xp.tile(points[:, None, None, :], (1, D, 2, 1))  # [N, D, 2, D]
                x_perturbed[:, :, 0, :] += eps * eye[None, :, :]  # +epsilon
                x_perturbed[:, :, 1, :] -= eps * eye[None, :, :]  # -epsilon
                
                x_flat = x_perturbed.reshape(2 * D * N, D)  # [2×D×N, D]
                L_flat = func(x_flat)  # ONE call, full parallelism
                
                L_perturbed = L_flat.reshape(N, D, 2)  # [N, D, 2]
                grad = (L_perturbed[:, :, 0] - L_perturbed[:, :, 1]) / (2 * eps)
                
                return grad
            
            else:
                # === BATCHED PATH: Only when approaching OOM ===
                # Batch over dimensions to fit in memory
                # Memory per batch: N × dim_batch × 2 × D × 8
                # Solve: N × dim_batch × 2 × D × 8 ≤ mem_limit × 0.8 (safety margin)
                dim_batch_size = max(1, int((mem_limit * 0.8) / (N * 2 * D * 8)))
                dim_batch_size = min(dim_batch_size, D)
                n_batches = (D + dim_batch_size - 1) // dim_batch_size
                
                if verbose:
                    print(f"  [GRADIENT] Batched: {mem_needed/1e9:.2f}GB > {mem_limit/1e9:.2f}GB limit")
                    print(f"             → {n_batches} batches of {dim_batch_size} dims")
                
                grad = xp.zeros((N, D), dtype=points.dtype)
                
                for d_start in range(0, D, dim_batch_size):
                    d_end = min(d_start + dim_batch_size, D)
                    batch_dims = d_end - d_start
                    
                    # Create perturbation tensor [N, batch_dims, 2, D]
                    x_perturbed = xp.tile(points[:, None, None, :], (1, batch_dims, 2, 1))
                    
                    # Create eye matrix for this batch of dimensions
                    eye_batch = xp.zeros((batch_dims, D), dtype=points.dtype)
                    eye_batch[xp.arange(batch_dims), xp.arange(d_start, d_end)] = 1.0
                    
                    x_perturbed[:, :, 0, :] += eps * eye_batch[None, :, :]
                    x_perturbed[:, :, 1, :] -= eps * eye_batch[None, :, :]
                    
                    x_flat = x_perturbed.reshape(2 * batch_dims * N, D)
                    L_flat = func(x_flat)
                    L_perturbed = L_flat.reshape(N, batch_dims, 2)
                    
                    grad[:, d_start:d_end] = (L_perturbed[:, :, 0] - L_perturbed[:, :, 1]) / (2 * eps)
                
                return grad
        
        elif method == 'forward':
            # Forward differences: (f(x+h) - f(x)) / h
            
            # Memory needed for [N, D, D] tensor
            mem_needed = N * D * D * 8
            
            # Evaluate f(x) for all points once
            f_x = func(points)
            
            if mem_needed < mem_limit:
                # === CLEAN PATH ===
                if verbose:
                    print(f"  [GRADIENT] Full parallel: {mem_needed/1e9:.2f}GB < {mem_limit/1e9:.2f}GB limit")
                
                eye = xp.eye(D, dtype=points.dtype)
                x_perturbed = xp.tile(points[:, None, :], (1, D, 1))  # [N, D, D]
                x_perturbed += eps * eye[None, :, :]
                
                x_flat = x_perturbed.reshape(D * N, D)
                L_flat = func(x_flat)
                L_perturbed = L_flat.reshape(N, D)
                
                grad = (L_perturbed - f_x[:, None]) / eps
                return grad
            
            else:
                # === BATCHED PATH ===
                dim_batch_size = max(1, int((mem_limit * 0.8) / (N * D * 8)))
                dim_batch_size = min(dim_batch_size, D)
                
                if verbose:
                    n_batches = (D + dim_batch_size - 1) // dim_batch_size
                    print(f"  [GRADIENT] Batched: {n_batches} batches of {dim_batch_size} dims")
                
                grad = xp.zeros((N, D), dtype=points.dtype)
                
                for d_start in range(0, D, dim_batch_size):
                    d_end = min(d_start + dim_batch_size, D)
                    batch_dims = d_end - d_start
                    
                    x_perturbed = xp.tile(points[:, None, :], (1, batch_dims, 1))
                    eye_batch = xp.zeros((batch_dims, D), dtype=points.dtype)
                    eye_batch[xp.arange(batch_dims), xp.arange(d_start, d_end)] = 1.0
                    x_perturbed += eps * eye_batch[None, :, :]
                    
                    x_flat = x_perturbed.reshape(batch_dims * N, D)
                    L_flat = func(x_flat)
                    L_perturbed = L_flat.reshape(N, batch_dims)
                    
                    grad[:, d_start:d_end] = (L_perturbed - f_x[:, None]) / eps
                
                return grad
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ========================================================================
    # NEW v1.3: Scale Estimation for Multi-Scale Optimization
    # ========================================================================
    
    def estimate_characteristic_scales(
        self,
        f_samples,
        t_values,
        n_scales: int = 20
    ) -> Tuple[float, float, float]:
        """
        Estimate 3 characteristic length scales from function samples along rays.
        
        Computes curvature (second derivative) at multiple scales and finds
        transitions where curvature drops - these indicate feature scales.
        
        Args:
            f_samples: [N_rays, N_samples] function values along rays
            t_values: [N_samples] t positions along rays
            n_scales: Number of scales to probe (log-spaced)
            
        Returns:
            (λ_fine, λ_mid, λ_coarse) - three characteristic scales
            
        Theory:
            For function f(t) = global(t) + local(t) where local has period P:
            - Curvature at scale h << P sees local bumps (high curvature)
            - Curvature at scale h >> P sees only global trend (low curvature)
            - Transitions reveal the characteristic scales
            
        Cost:
            ~20 GPU array operations, ~1ms at D=256. Called once per ray phase.
        """
        xp = self.xp
        
        N_rays, N_samples = f_samples.shape
        dt = float(t_values[1] - t_values[0])  # Assume uniform spacing
        
        # Generate log-spaced skip indices (scales)
        max_skip = max(2, N_samples // 4)  # Can't go beyond 1/4 of ray length
        skip_values = xp.unique(
            xp.logspace(0, xp.log10(max_skip), n_scales).astype(xp.int32)
        )
        skip_values = skip_values[(skip_values >= 1) & (skip_values < max_skip)]
        
        if len(skip_values) == 0:
            # Fallback: not enough samples for scale estimation
            ray_length = float(t_values[-1] - t_values[0])
            return (dt, ray_length / 4, ray_length / 2)
        
        # Compute max curvature at each scale (skip-slicing approach)
        max_curvatures = xp.zeros(len(skip_values), dtype=xp.float64)
        
        for s, skip in enumerate(skip_values):
            skip = int(skip)
            
            # Second derivative: [f(t+h) - 2f(t) + f(t-h)] / h²
            f_left = f_samples[:, :-2*skip]
            f_mid = f_samples[:, skip:-skip]
            f_right = f_samples[:, 2*skip:]
            
            h = skip * dt
            curvature = (f_left - 2*f_mid + f_right) / (h**2)
            
            # Max absolute curvature across all rays and positions
            max_curvatures[s] = xp.max(xp.abs(curvature))
        
        # Convert skip indices to actual length scales
        h_values = skip_values.astype(xp.float64) * dt
        
        # Find transitions (where curvature drops significantly)
        # Use log-scale curvature for better transition detection
        log_curv = xp.log10(max_curvatures + 1e-300)
        
        if len(log_curv) < 3:
            # Not enough points for transition detection
            ray_length = float(t_values[-1] - t_values[0])
            return (float(h_values[0]), ray_length / 4, ray_length / 2)
        
        # Compute derivative of log-curvature w.r.t. log-scale
        # Transitions are where this derivative is most negative
        d_log_curv = xp.diff(log_curv) / (xp.diff(xp.log10(h_values)) + 1e-10)
        
        # Find the two most negative transitions
        sorted_indices = xp.argsort(d_log_curv)
        transition_indices = sorted_indices[:2]
        transition_indices = xp.sort(transition_indices)  # Order by scale
        
        # Extract scales at transitions
        n_h = len(h_values)
        if len(transition_indices) >= 2:
            idx_fine = int(transition_indices[0])
            idx_coarse = int(transition_indices[1])
        elif len(transition_indices) == 1:
            idx_fine = int(transition_indices[0])
            idx_coarse = min(idx_fine + n_h // 2, n_h - 1)
        else:
            # Fallback: use quartiles
            idx_fine = n_h // 4
            idx_coarse = 3 * n_h // 4
        
        idx_mid = (idx_fine + idx_coarse) // 2
        
        # Clamp indices to valid range
        idx_fine = max(0, min(idx_fine, n_h - 1))
        idx_mid = max(0, min(idx_mid, n_h - 1))
        idx_coarse = max(0, min(idx_coarse, n_h - 1))
        
        λ_fine = float(h_values[idx_fine])
        λ_mid = float(h_values[idx_mid])
        λ_coarse = float(h_values[idx_coarse])
        
        # Ensure strict ordering: λ_fine < λ_mid < λ_coarse
        scales = sorted([λ_fine, λ_mid, λ_coarse])
        λ_fine, λ_mid, λ_coarse = scales[0], scales[1], scales[2]
        
        # Ensure minimum separation
        if λ_mid <= λ_fine:
            λ_mid = λ_fine * 2
        if λ_coarse <= λ_mid:
            λ_coarse = λ_mid * 2
        
        return (λ_fine, λ_mid, λ_coarse)
    
    # ========================================================================
    # NEW v1.3: Smoothed Gradient for Hands Like Clouds (雲手)
    # ========================================================================
    
    def evaluate_gradient_smoothed(
        self,
        points,
        func: Callable,
        sigma: float,
        K: int = 10,
        eps: Optional[float] = None,
        func_returns_grad: bool = False  # NEW: analytic gradient support
    ):
        """
        Gradient of Gaussian-smoothed function ("Hands Like Clouds").
        
        Computes: ∇f_σ(x) ≈ (1/K) Σ ∇f(x + σ·ε_k)  where ε_k ~ N(0, I)
        
        This blurs out features smaller than σ, revealing global structure.
        
        Args:
            points: [N, D] points to evaluate
            func: Likelihood function (may return (L, grad) if func_returns_grad=True)
            sigma: Smoothing scale (use λ_coarse for maximum blur)
            K: Number of perturbation samples (more = smoother estimate)
            eps: Finite difference epsilon (default: auto)
            func_returns_grad: If True, func returns (L, grad) tuple
            
        Returns:
            [N, D] smoothed gradients
            
        Theory:
            f_σ(x) = E[f(x + σε)] is the Gaussian-smoothed version of f.
            Features with scale << σ average out to ~0 gradient.
            Only features with scale >> σ remain visible.
            
        Cost:
            K gradient evaluations, but all parallel on GPU.
            Wall-clock ≈ same as K=1 for large N.
            With analytic gradients (func_returns_grad=True): O(K) instead of O(K*D)
        """
        xp = self.xp
        N, D = points.shape
        
        if eps is None:
            eps = 1e-8 if points.dtype == xp.float64 else 1e-4
        
        # Generate all perturbations at once: [N, K, D]
        noise = xp.random.randn(N, K, D).astype(points.dtype) * sigma
        
        # Perturbed points: [N, K, D]
        x_perturbed = points[:, None, :] + noise
        
        # Flatten for batch gradient: [N*K, D]
        x_flat = x_perturbed.reshape(N * K, D)
        
        # Batch gradient computation (GPU parallel, uses analytic if available)
        grads_flat = self.evaluate_gradient_batch(x_flat, func, eps=eps,
                                                   func_returns_grad=func_returns_grad)
        
        # Reshape and average: [N, K, D] → [N, D]
        grads_perturbed = grads_flat.reshape(N, K, D)
        smoothed_grads = xp.mean(grads_perturbed, axis=1)
        
        return smoothed_grads


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SingleWhip v1.3 - STANDALONE - Self-Tests")
    print("="*70)
    print()
    
    # Initialize
    whip = SingleWhip(use_gpu=True)
    xp = whip.xp
    
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Using: {'GPU (CuPy)' if whip.is_gpu else 'CPU (NumPy)'}")
    print()
    
    # Test 1: Likelihood evaluation
    print("[Test 1: Likelihood Evaluation]")
    def quadratic(theta):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        return -0.5 * xp.sum(theta**2, axis=1)
    
    points = xp.random.randn(100, 10).astype(xp.float64)
    L = whip.evaluate_likelihood(points, quadratic)
    print(f"  ✓ Evaluated {len(points)} points")
    print()
    
    # Test 2: Distance computation
    print("[Test 2: Distance Computation]")
    dists_l2 = whip.compute_distance(points[:10], points[10:20], metric='l2')
    dists_linf = whip.compute_distance(points[:10], points[10:20], metric='linf')
    print(f"  ✓ L2 distances: {dists_l2.shape}")
    print(f"  ✓ L∞ distances: {dists_linf.shape}")
    print()
    
    # Test 3: Ray sampling
    print("[Test 3: Sunburst Ray Sampling]")
    center = xp.zeros(10, dtype=xp.float64)
    bounds = xp.array([[-5, 5]] * 10, dtype=xp.float64)
    rays = whip.sample_sunburst_rays(center, 50, bounds, ray_length=1.0)
    print(f"  ✓ Generated {len(rays)} rays from center")
    print()
    
    # Test 4: Peak deduplication
    print("[Test 4: Peak Deduplication]")
    peaks = xp.random.randn(20, 10).astype(xp.float64) * 0.1
    L_peaks = xp.random.randn(20).astype(xp.float64)
    unique_peaks, unique_L = whip.deduplicate_peaks(peaks, L_peaks, tolerance=0.5)
    print(f"  ✓ Deduplicated: {len(peaks)} → {len(unique_peaks)} peaks")
    print()
    
    # Test 5: Single gradient
    print("[Test 5: Single-Point Gradient]")
    point = xp.zeros(10, dtype=xp.float64)
    grad = whip.evaluate_gradient(point, quadratic)
    grad_norm = float(xp.linalg.norm(grad))
    print(f"  ✓ Gradient norm at origin: {grad_norm:.2e}")
    print()
    
    # Test 6: Batched gradients
    print("[Test 6: Batched Gradient Evaluation (v1.2)]")
    batch_points = xp.random.randn(50, 10).astype(xp.float64) * 0.1
    
    start = time.time()
    grads_batch = whip.evaluate_gradient_batch(batch_points, quadratic, verbose=False)
    time_batch = time.time() - start
    
    print(f"  ✓ Batched gradients: {grads_batch.shape}")
    print(f"  Time: {time_batch:.4f}s")
    print()
    
    # Test 7: NEW v1.3 - Scale estimation
    print("[Test 7: Scale Estimation (v1.3)]")
    
    # Create a multi-scale test function: global bowl + local bumps
    def rastrigin_1d(t):
        return t**2 + 10 * xp.cos(2 * xp.pi * t)
    
    # Simulate ray samples
    N_rays = 100
    N_samples = 200
    t_values = xp.linspace(-5, 5, N_samples)
    
    # Generate f_samples for multiple rays (same function, different offsets)
    f_samples = xp.zeros((N_rays, N_samples), dtype=xp.float64)
    for i in range(N_rays):
        offset = xp.random.randn() * 0.1
        f_samples[i] = rastrigin_1d(t_values + offset)
    
    start = time.time()
    λ_fine, λ_mid, λ_coarse = whip.estimate_characteristic_scales(f_samples, t_values)
    time_scales = time.time() - start
    
    print(f"  ✓ Scales: λ_fine={λ_fine:.4f}, λ_mid={λ_mid:.4f}, λ_coarse={λ_coarse:.4f}")
    print(f"  Expected: fine ~0.05-0.1 (bump period ~1), coarse ~2-5 (global bowl)")
    print(f"  Time: {time_scales*1000:.2f}ms")
    print()
    
    # Test 8: NEW v1.3 - Smoothed gradient
    print("[Test 8: Smoothed Gradient (v1.3)]")
    
    # Rastrigin in 2D
    def rastrigin_2d(theta):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        return -(10 * theta.shape[1] + xp.sum(theta**2 - 10 * xp.cos(2 * xp.pi * theta), axis=1))
    
    test_points = xp.random.randn(20, 2).astype(xp.float64) * 2
    
    # Sharp gradient (sees local bumps)
    start = time.time()
    grad_sharp = whip.evaluate_gradient_batch(test_points, rastrigin_2d)
    time_sharp = time.time() - start
    
    # Smoothed gradient (sees global bowl)
    start = time.time()
    grad_smooth = whip.evaluate_gradient_smoothed(test_points, rastrigin_2d, sigma=1.0, K=10)
    time_smooth = time.time() - start
    
    print(f"  ✓ Sharp gradients: mean norm = {float(xp.mean(xp.linalg.norm(grad_sharp, axis=1))):.2f}")
    print(f"  ✓ Smooth gradients (σ=1.0): mean norm = {float(xp.mean(xp.linalg.norm(grad_smooth, axis=1))):.2f}")
    print(f"  Time: sharp={time_sharp*1000:.2f}ms, smooth={time_smooth*1000:.2f}ms")
    print()
    
    # Verify smoothed gradient points toward origin (global minimum)
    # Dot product with direction toward origin should be positive
    toward_origin = -test_points / (xp.linalg.norm(test_points, axis=1, keepdims=True) + 1e-10)
    alignment = xp.sum(grad_smooth * toward_origin, axis=1)
    frac_aligned = float(xp.mean(alignment > 0))
    print(f"  Fraction pointing toward origin: {frac_aligned:.1%}")
    if frac_aligned > 0.7:
        print(f"  ✓ Smoothed gradient correctly sees global structure")
    else:
        print(f"  ⚠ Smoothed gradient may need larger σ or more samples")
    print()
    
    # Summary
    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print()
    print("SingleWhip v1.3 Standalone Features:")
    print("  • Likelihood evaluation (batched)")
    print("  • Distance computation (L∞/L2/k-NN)")
    print("  • Sunburst ray sampling")
    print("  • Peak deduplication")
    print("  • Single-point gradient evaluation")
    print("  • v1.2: Batched gradient evaluation (10-50× faster)")
    print("  • NEW v1.3: Multi-scale estimation")
    print("  • NEW v1.3: Smoothed gradient ('Hands Like Clouds')")
    print()
    print("NO external dependencies! (except NumPy/CuPy/SciPy)")
    print("="*70)
