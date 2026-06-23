"""
ChiSao (黏手) - Sticky Hands Optimization Module v3.3

GPU-accelerated batched optimization with convergence-anticonvergence heuristics.
Fully parallelized, memory-aware, and GPU-capability-adaptive.

Named after the Wing Chun "sticky hands" technique - combining convergence 
with strategic exploration through anti-convergence oscillations.

The heuristics within this module are named after Guang Ping Yang Style 
Tai Chi (廣平楊式太極拳) forms, in honor of Master Donald Rubbo.

    雲手 (Yún Shǒu) - Hands Like Clouds: Multi-scale smoothing
    金雞獨立 (Jīn Jī Dú Lì) - Golden Rooster Stands on One Leg: GPU-filling
    倒攆猴 (Dào Niǎn Hóu) - Step Back Repulse Monkey: Sample reseeding
    單鞭 (Dān Biān) - Single Whip: Low-level GPU toolkit
    抱虎歸山 (Bào Hǔ Guī Shān) - Carry Tiger to Mountain: Ray casting

v3.3 Changes:
    - NEW: lbfgs_batch return_trajectories option for rotation detection
      Stores intermediate positions at each L-BFGS iteration
      Used by Module 3 (BendTheBow) for perpendicular step fraction analysis
      to detect rotated Gaussians requiring full Hessian computation

v3.2 Changes:
    - Hybrid L-BFGS history update: vectorized where() for D>=100 (fixes
      O(D^1.5) scattered-write bottleneck), fancy indexing for D<100
    - Adaptive dim_batch_size: auto-selects based on available memory (2GB target)
      instead of hardcoded 64, reducing kernel launches from O(D/64) to O(1)

Features:
    - Pure GPU parallelization (no Python loops over samples)
    - Automatic GPU capability detection and optimization
    - L-BFGS and Gradient Ascent optimizers
    - Sticky hands convergence-anticonvergence heuristic
    - Repulse Monkey sample count maintenance via reseeding
    - Golden Rooster (金雞獨立) GPU-filling from converged peaks
    - Cannon Through the Sky boundary-stuck sample rescue
    - Peak width estimation under L_∞ metric
    - Memory-safe batching with multiple strategies
    - CUDA stream support for advanced GPUs
    - Runtime calibration and caching
    - Extensible optimizer framework
    - CPU fallback support
    - External reseeding strategy support (Callable)
    - NEW v2.2: Hands Like Clouds (雲手) phase for multi-scale optimization
    - NEW v2.5: Memory-efficient gradient computation O(N×D) instead of O(N×D²)

Version History:
    1.5.1: History tracking, bug fixes
    1.6.0: Unified reseed_strategy (None | str | Callable)
           - External strategy support for Module 1 integration
           - Removed separate repulse_monkey parameter
           - Strategy=None disables reseeding
    1.7.0: Width estimation from L-BFGS inverse Hessian
           - lbfgs_batch returns inv_hessian_scale (gamma = s^T y / y^T y)
           - sticky_hands uses sqrt(gamma) for widths when available
           - Falls back to estimate_peak_width() if dedup changes peak count
    1.8.0: Robust Hessian transfer during deduplication
           - lbfgs_batch returns history_count (0 = no movement, garbage Hessian)
           - deduplicate_peaks_L_infinity transfers widest valid Hessian to winner
           - Fixed L=0 stuck mask bug (removed vestigial L != 0 check)
           - Adaptive final dedup: uses min_width/100 as tolerance after width estimation
    2.0.0: SingleWhip integration for gradient/Hessian acceleration
           - Integrated SingleWhip for 1-call gradients (vs d calls)
           - Integrated SingleWhip for 1-call Hessians (vs d² calls)
           - ~128× faster gradients, ~16,384× faster width estimation at d=128
           - All ChiSao algorithms preserved (sticky hands, Repulse Monkey, Cannon)
           - Backward compatible: falls back to legacy if SingleWhip unavailable
    2.1.0: Golden Rooster (金雞獨立) for maximum GPU utilization
           - When N_unconverged < 5, use converged peaks as sunburst sources
           - Refills GPU to original sample count (free parallelization)
           - Ray length = longest prior dimension / 2
           - New samples marked unstuck for exploration
           - Adaptive: disables if exploration doesn't find new peaks
           - Philosophy: GPU has thousands of cores - use what's freely given!
    2.2.0: Hands Like Clouds (雲手) for multi-scale optimization
           - Uses SingleWhip's smoothed gradient to see global structure
           - Blurs out local oscillations (Rastrigin, Ackley bumps)
           - Nudges unstuck samples toward global basin before anti-convergence
           - Auto-estimates cloud_sigma from sample spread if not provided
           - Configurable: n_cloud iterations, cloud_K perturbations
           - Philosophy: "Clouds obscure details but reveal the mountain"
    2.3.0: Two-phase mode support for multi-scale problems
           - Added mode='full' (default) or mode='hlc_only' parameter
           - hlc_only: Extended Hands Like Clouds, minimal local refinement
           - For Ackley/Rastrigin: Phase 1 finds global, Phase 2 finds local
           - Integrates with CarryTiger v1.7 multi_scale parameter
    2.4.0: Sample banking for evidence calculation
           - Added SampleBank class for GPU-resident sample storage
           - Added bank_samples parameter to sticky_hands
           - Banks samples at key stages: initial, post_converge, post_dedup, final
           - For integration with Module 3 (BendTheBowShootTheTiger)
    2.6.0: Pi Chuan (劈拳) - Gradient-free optimizers for slow likelihoods
           - Named after Hsing-I's Splitting Fist - direct, no wasted motion
           - Added 'nelder_mead' optimizer (scipy Nelder-Mead per sample)
           - Added 'powell' optimizer (scipy Powell per sample)
           - Added 'direct' optimizer (pattern search, no gradients)
           - Added 'none' optimizer (no refinement, return as-is)
           - For expensive likelihoods like Planck/CAMB (~0.5s per eval)
    3.0.0: Vectorized deduplication - pure GPU, no Python loops
           - deduplicate_peaks_L_infinity now O(N²) GPU ops instead of Python loops
           - Eliminates CPU↔GPU transfers in dedup inner loop
           - ~100× faster deduplication for large sample counts
           - All other algorithms unchanged

Author: Ira Wolfson, Braude College of Engineering
Date: December 2025
Version: 3.1.0
"""

import numpy as np
import warnings
from typing import Callable, Optional, Union, Dict, Tuple, Any
import time
import json
import os

# GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. ChiSao will run on CPU (much slower).")

# SingleWhip integration (v2.0, upgraded to v1.6 for ChiSao v3.2 clean-path gradients)
SINGLEWHIP_VERSION = None
SINGLEWHIP_AVAILABLE = False
try:
    # Package-relative import
    from .single_whip import SingleWhip
    SINGLEWHIP_AVAILABLE = True
    SINGLEWHIP_VERSION = '1.6'
except ImportError:
    try:
        # Fallback for standalone usage
        from single_whip_GPU_v1_6 import SingleWhip
        SINGLEWHIP_AVAILABLE = True
        SINGLEWHIP_VERSION = '1.6'
    except ImportError:
        warnings.warn(
            "SingleWhip not available. ChiSao will use legacy gradient computation."
        )

# RandCoord optimizer (v3.1 - sublinear gradient scaling)
RANDCOORD_AVAILABLE = False
try:
    from .single_whip import randcoord_line_search_batch
    RANDCOORD_AVAILABLE = True
except ImportError:
    try:
        from single_whip_GPU_v1_6 import randcoord_line_search_batch
        RANDCOORD_AVAILABLE = True
    except ImportError:
        pass


# ============================================================================
# SAMPLE BANKING FOR EVIDENCE CALCULATION (v2.4)
# ============================================================================

class SampleBank:
    """
    GPU-resident storage for samples collected during optimization.
    Used for evidence calculation in Module 3 (BendTheBowShootTheTiger).
    
    Stores: positions, log_L values, source stage, oscillation number, converged flags.
    """
    
    def __init__(self, D: int, initial_capacity: int = 10000, xp=None):
        """
        Initialize sample bank.
        
        Args:
            D: Dimensionality
            initial_capacity: Initial storage capacity (grows dynamically)
            xp: Array module (np or cp)
        """
        self.xp = xp if xp is not None else (cp if GPU_AVAILABLE else np)
        self.D = D
        self.capacity = initial_capacity
        self.n_samples = 0
        
        # Pre-allocate GPU arrays
        self.positions = self.xp.zeros((initial_capacity, D), dtype=self.xp.float64)
        self.log_L = self.xp.zeros(initial_capacity, dtype=self.xp.float64)
        self.source_stage = self.xp.zeros(initial_capacity, dtype=self.xp.int32)
        self.source_oscillation = self.xp.zeros(initial_capacity, dtype=self.xp.int32)
        self.converged = self.xp.zeros(initial_capacity, dtype=self.xp.bool_)
        
        # Stage ID mapping
        self.STAGE_IDS = {
            'initial': 0,
            'post_converge': 1,
            'post_dedup': 2,
            'post_golden_rooster': 3,
            'post_reseed': 4,
            'post_clouds': 5,
            'post_anticonverge': 6,
            'post_cannon': 7,
            'final': 8
        }
    
    def _grow_if_needed(self, n_new: int):
        """Double capacity if needed to fit n_new samples."""
        if self.n_samples + n_new > self.capacity:
            new_capacity = max(self.capacity * 2, self.n_samples + n_new)
            
            # Grow each array
            new_positions = self.xp.zeros((new_capacity, self.D), dtype=self.xp.float64)
            new_positions[:self.n_samples] = self.positions[:self.n_samples]
            self.positions = new_positions
            
            new_log_L = self.xp.zeros(new_capacity, dtype=self.xp.float64)
            new_log_L[:self.n_samples] = self.log_L[:self.n_samples]
            self.log_L = new_log_L
            
            new_stage = self.xp.zeros(new_capacity, dtype=self.xp.int32)
            new_stage[:self.n_samples] = self.source_stage[:self.n_samples]
            self.source_stage = new_stage
            
            new_osc = self.xp.zeros(new_capacity, dtype=self.xp.int32)
            new_osc[:self.n_samples] = self.source_oscillation[:self.n_samples]
            self.source_oscillation = new_osc
            
            new_conv = self.xp.zeros(new_capacity, dtype=self.xp.bool_)
            new_conv[:self.n_samples] = self.converged[:self.n_samples]
            self.converged = new_conv
            
            self.capacity = new_capacity
    
    def add_samples(self, positions, log_L_values, stage: str, oscillation: int, converged_mask=None):
        """
        Add samples to the bank.
        
        Args:
            positions: [N, D] sample positions
            log_L_values: [N] log-likelihood values
            stage: Stage name (e.g., 'post_converge')
            oscillation: Oscillation number
            converged_mask: [N] boolean mask of converged samples (optional)
        """
        n_new = positions.shape[0]
        if n_new == 0:
            return
            
        self._grow_if_needed(n_new)
        
        start = self.n_samples
        end = start + n_new
        
        self.positions[start:end] = positions
        self.log_L[start:end] = log_L_values
        self.source_stage[start:end] = self.STAGE_IDS.get(stage, -1)
        self.source_oscillation[start:end] = oscillation
        
        if converged_mask is not None:
            self.converged[start:end] = converged_mask
        else:
            self.converged[start:end] = False
            
        self.n_samples = end
    
    def get_bank_dict(self) -> dict:
        """Return trimmed arrays as a dictionary."""
        n = self.n_samples
        return {
            'positions': self.positions[:n].copy(),
            'log_L': self.log_L[:n].copy(),
            'source_stage': self.source_stage[:n].copy(),
            'source_oscillation': self.source_oscillation[:n].copy(),
            'converged': self.converged[:n].copy(),
            'n_samples': n,
            'stage_ids': self.STAGE_IDS.copy()
        }
    
    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        n = self.n_samples
        bytes_used = (
            n * self.D * 8 +  # positions (float64)
            n * 8 +          # log_L (float64)
            n * 4 +          # source_stage (int32)
            n * 4 +          # source_oscillation (int32)
            n * 1            # converged (bool)
        )
        return bytes_used / (1024 * 1024)


# ============================================================================
# GPU CAPABILITY DETECTION
# ============================================================================

class GPUCapability:
    """
    Detect and classify GPU capabilities.
    Auto-tune parallelism strategy based on hardware.
    """
    
    def __init__(self):
        self.tier = self._detect_gpu_tier()
        self.specs = self._get_gpu_specs()
        self.optimal_config = self._compute_optimal_config()
    
    def _detect_gpu_tier(self) -> str:
        """
        Classify GPU into performance tiers.
        
        Tiers:
            TIER_1: Consumer (GTX 1080, RTX 2060, etc.)
            TIER_2: Prosumer (RTX 3080, RTX 4090, etc.)
            TIER_3: Professional (A100, A6000, etc.)
            TIER_4: Data Center (H100, H200, etc.)
            CPU_ONLY: No GPU available
        """
        if not GPU_AVAILABLE:
            return 'CPU_ONLY'
        
        device = cp.cuda.Device()
        compute_capability = '.'.join(map(str, device.compute_capability))
        total_memory_gb = device.mem_info[1] / 1e9
        
        # Parse compute capability
        cc_major = int(device.compute_capability[0])
        
        # Classification heuristics
        if cc_major >= 9:  # Hopper (H100)
            return 'TIER_4'
        elif cc_major >= 8:  # Ampere (A100, RTX 30xx, 40xx)
            if total_memory_gb >= 40:
                return 'TIER_3'
            else:
                return 'TIER_2'
        elif cc_major >= 7:  # Volta/Turing
            return 'TIER_2' if total_memory_gb >= 16 else 'TIER_1'
        else:
            return 'TIER_1'
    
    def _get_gpu_specs(self) -> Dict[str, Any]:
        """Extract detailed GPU specifications."""
        if not GPU_AVAILABLE:
            return {
                'name': 'CPU',
                'memory_gb': None,
                'sm_count': 0,
                'compute_capability': None,
                'cuda_cores_approx': 0
            }
        
        device = cp.cuda.Device()
        attrs = device.attributes
        
        # Get device name using runtime API
        device_id = device.id
        try:
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)
            device_name = device_props['name'].decode('utf-8') if isinstance(device_props['name'], bytes) else str(device_props['name'])
        except:
            device_name = f"GPU_{device_id}"
        
        return {
            'name': device_name,
            'memory_gb': device.mem_info[1] / 1e9,
            'sm_count': attrs['MultiProcessorCount'],
            'compute_capability': '.'.join(map(str, device.compute_capability)),
            'max_threads_per_block': attrs['MaxThreadsPerBlock'],
            'max_threads_per_sm': attrs.get('MaxThreadsPerMultiProcessor', 0),
            'warp_size': attrs['WarpSize'],
            'cuda_cores_approx': self._estimate_cuda_cores(attrs, device.compute_capability)
        }
    
    def _estimate_cuda_cores(self, attrs: dict, compute_capability: tuple) -> int:
        """Estimate CUDA core count (varies by architecture)."""
        sm_count = attrs['MultiProcessorCount']
        
        # Cores per SM varies by architecture
        cores_per_sm_map = {
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere (A100)
            (8, 6): 128,  # Ampere (RTX 30xx)
            (8, 9): 128,  # Ada Lovelace (RTX 40xx)
            (9, 0): 128,  # Hopper (H100)
        }
        
        cores_per_sm = cores_per_sm_map.get(compute_capability, 64)
        return sm_count * cores_per_sm
    
    def _compute_optimal_config(self) -> Dict[str, Any]:
        """Compute optimal parallelism configuration based on GPU tier."""
        tier_configs = {
            'CPU_ONLY': {
                'batch_strategy': 'fixed',
                'batch_size': 100,
                'n_streams': 1,
                'max_parallel_evals': 1000,
                'gradient_batch_size': 100,
                'use_streams': False
            },
            'TIER_1': {  # GTX 1080 (8GB, ~2560 cores)
                'batch_strategy': 'adaptive',
                'batch_size': 1024,
                'n_streams': 2,
                'max_parallel_evals': 10000,
                'gradient_batch_size': 2048,
                'use_streams': False
            },
            'TIER_2': {  # RTX 3080 (10-12GB, ~8704 cores)
                'batch_strategy': 'auto',
                'batch_size': 4096,
                'n_streams': 4,
                'max_parallel_evals': 50000,
                'gradient_batch_size': 10000,
                'use_streams': True
            },
            'TIER_3': {  # A100 (40-80GB, ~6912 cores)
                'batch_strategy': 'auto',
                'batch_size': 16384,
                'n_streams': 8,
                'max_parallel_evals': 500000,
                'gradient_batch_size': 100000,
                'use_streams': True
            },
            'TIER_4': {  # H100 (80GB, ~16896 cores)
                'batch_strategy': 'auto',
                'batch_size': 65536,
                'n_streams': 16,
                'max_parallel_evals': 2000000,
                'gradient_batch_size': 500000,
                'use_streams': True
            }
        }
        
        config = tier_configs[self.tier].copy()
        
        # Fine-tune based on actual memory
        if GPU_AVAILABLE:
            memory_gb = self.specs['memory_gb']
            memory_scale = memory_gb / 12.0  # Baseline: 12GB
            config['batch_size'] = int(config['batch_size'] * memory_scale)
            config['max_parallel_evals'] = int(config['max_parallel_evals'] * memory_scale)
            config['gradient_batch_size'] = int(config['gradient_batch_size'] * memory_scale)
        
        return config


# ============================================================================
# ANALYTIC GRADIENT SUPPORT
# ============================================================================

def eval_L_only(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    func_returns_grad: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Evaluate function and return only likelihood (not gradient).
    
    Use this when you need L but not grad, and func might return (L, grad) tuple.
    
    Args:
        func: Likelihood function
        x: [N, D] sample positions
        func_returns_grad: If True, func returns (L, grad) tuple
        
    Returns:
        L: [N] likelihood values
    """
    if func_returns_grad:
        L, _ = func(x)
        return L
    else:
        return func(x)


def evaluate_with_gradient(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    func_returns_grad: bool = False
) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray', None]]:
    """
    Evaluate function and get gradient if available.
    
    Args:
        func: Likelihood function. Can return:
              - Just L: func(x) -> [N]
              - L and grad: func(x) -> ([N], [N, D])
        x: [N, D] sample positions
        func_returns_grad: If True, func returns (L, grad) tuple
        
    Returns:
        L: [N] likelihood values
        grad: [N, D] gradients if available, else None
    """
    result = func(x)
    
    if func_returns_grad or isinstance(result, tuple):
        # Function returns (L, grad)
        L, grad = result
        return L, grad
    else:
        # Function returns only L
        return result, None


def get_gradient(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    func_returns_grad: bool = False,
    cached_grad: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    epsilon: Optional[float] = None,
    batch_size: Optional[int] = None,
    dim_batch_size: int = 64,
    sample_batch_size: Optional[int] = None,
    verbose: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Get gradient - use analytic if available, else finite differences.
    
    This is the main entry point for gradient computation in ChiSao.
    
    Args:
        func: Likelihood function (may return gradient)
        x: [N, D] sample positions
        func_returns_grad: If True, func returns (L, grad) tuple
        cached_grad: Pre-computed gradient (from previous evaluate_with_gradient call)
        epsilon: Finite difference step (if needed)
        batch_size: Sample batch size for finite diff
        dim_batch_size: Dimension batch size for finite diff
        sample_batch_size: Sample batch size (alias)
        verbose: Print debug info
        
    Returns:
        grad: [N, D] gradients
    """
    # If we have a cached gradient from a previous evaluation, use it
    if cached_grad is not None:
        return cached_grad
    
    # If function returns gradient, call it
    if func_returns_grad:
        _, grad = evaluate_with_gradient(func, x, func_returns_grad=True)
        if grad is not None:
            return grad
    
    # Fallback to finite differences
    return compute_gradient_batch(
        func, x,
        epsilon=epsilon,
        batch_size=batch_size,
        dim_batch_size=dim_batch_size,
        sample_batch_size=sample_batch_size,
        verbose=verbose
    )


# ============================================================================
# GRADIENT COMPUTATION (FINITE DIFFERENCES)
# ============================================================================

def compute_gradient_batch(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    epsilon: Optional[float] = None,
    batch_size: Optional[int] = None,
    method: str = 'central',
    verbose: bool = False,
    dim_batch_size: Optional[int] = None,  # None = auto based on memory
    sample_batch_size: Optional[int] = None  # NEW: Batch samples for high-D
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute gradients for N samples using finite differences.
    
    DOUBLE-BATCHED v3.0: Batches both dimensions AND samples for memory control.
    
    Memory per batch: sample_batch × dim_batch × D × 8 bytes
    At D=1024 with sample_batch=256, dim_batch=64: 256×64×1024×8 = 128 MB
    
    Args:
        func: Likelihood function(x) -> [N] or scalar
        x: [N, D] sample positions
        epsilon: Finite difference step size (None = auto based on dtype)
        batch_size: Alias for sample_batch_size (backward compat)
        method: 'central' or 'forward' differences
        verbose: Print debug information
        dim_batch_size: Number of dimensions to process at once (None = auto)
        sample_batch_size: Number of samples to process at once (None = auto)
    
    Returns:
        grad: [N, D] gradients for all samples
    """
    xp = cp.get_array_module(x) if GPU_AVAILABLE else np
    N, D = x.shape
    
    # Auto-select epsilon based on dtype precision
    if epsilon is None:
        if x.dtype == xp.float32:
            epsilon = 1e-4
        else:
            epsilon = 1e-8
    
    # Auto-select dim_batch_size based on available memory
    # Memory = N × dim_batch × D × 8 × 2 (plus and minus perturbations)
    # Target 2GB to leave headroom on 8GB cards
    if dim_batch_size is None:
        target_bytes = 2 * 1e9
        dim_batch_size = max(64, int(target_bytes / (N * D * 8 * 2)))
        dim_batch_size = min(dim_batch_size, D)  # Don't exceed D
    
    # Auto-select sample batch size based on D and available memory
    if sample_batch_size is None:
        if batch_size is not None:
            sample_batch_size = batch_size
        else:
            # Target 2GB, same as dim_batch
            target_bytes = 2 * 1e9
            sample_batch_size = max(64, int(target_bytes / (dim_batch_size * D * 8 * 2)))
            sample_batch_size = min(sample_batch_size, N)  # Don't exceed N
    
    n_sample_batches = (N + sample_batch_size - 1) // sample_batch_size
    n_dim_batches = (D + dim_batch_size - 1) // dim_batch_size
    
    if verbose:
        mem_per_batch_mb = sample_batch_size * dim_batch_size * D * 8 * 2 / 1e6
        print(f"  [Gradient] N={N}, D={D}")
        print(f"  [Gradient] sample_batch={sample_batch_size}, dim_batch={dim_batch_size}")
        print(f"  [Gradient] {n_sample_batches} sample batches × {n_dim_batches} dim batches")
        print(f"  [Gradient] ~{mem_per_batch_mb:.1f} MB per batch")
    
    grad = xp.zeros((N, D), dtype=x.dtype)
    
    # Process samples in batches
    for s_start in range(0, N, sample_batch_size):
        s_end = min(s_start + sample_batch_size, N)
        x_batch = x[s_start:s_end]  # (batch_N, D)
        batch_N = s_end - s_start
        
        if method == 'central':
            # Process dimensions in batches
            for d_start in range(0, D, dim_batch_size):
                d_end = min(d_start + dim_batch_size, D)
                d_batch = d_end - d_start
                
                # Create perturbations for this batch of dimensions
                eye_batch = xp.eye(D, dtype=x.dtype)[d_start:d_end]  # (d_batch, D)
                
                # Broadcast: (batch_N, 1, D) + epsilon * (1, d_batch, D)
                x_plus = x_batch[:, None, :] + epsilon * eye_batch[None, :, :]
                x_minus = x_batch[:, None, :] - epsilon * eye_batch[None, :, :]
                
                # Reshape for function call: (batch_N * d_batch, D)
                x_plus_flat = x_plus.reshape(batch_N * d_batch, D)
                x_minus_flat = x_minus.reshape(batch_N * d_batch, D)
                
                # Two function calls for this batch
                f_plus = func(x_plus_flat).reshape(batch_N, d_batch)
                f_minus = func(x_minus_flat).reshape(batch_N, d_batch)
                
                # Gradient for these samples and dimensions
                grad[s_start:s_end, d_start:d_end] = (f_plus - f_minus) / (2 * epsilon)
        
        else:  # forward
            # Base function value for this sample batch
            f_base = func(x_batch)  # (batch_N,)
            
            for d_start in range(0, D, dim_batch_size):
                d_end = min(d_start + dim_batch_size, D)
                d_batch = d_end - d_start
                
                eye_batch = xp.eye(D, dtype=x.dtype)[d_start:d_end]
                x_plus = x_batch[:, None, :] + epsilon * eye_batch[None, :, :]
                x_plus_flat = x_plus.reshape(batch_N * d_batch, D)
                
                f_plus = func(x_plus_flat).reshape(batch_N, d_batch)
                grad[s_start:s_end, d_start:d_end] = (f_plus - f_base[:, None]) / epsilon
    
    return grad


# ============================================================================
# OPTIMIZER FRAMEWORK (Extensible)
# ============================================================================

# Registry for optimizer methods
_OPTIMIZERS = {}

def register_optimizer(name: str):
    """Decorator to register new optimization methods."""
    def decorator(func):
        _OPTIMIZERS[name] = func
        return func
    return decorator


# ============================================================================
# L-BFGS OPTIMIZER
# ============================================================================

def lbfgs_two_loop_recursion(
    grad: Union[np.ndarray, 'cp.ndarray'],
    s_history: Union[np.ndarray, 'cp.ndarray'],
    y_history: Union[np.ndarray, 'cp.ndarray'],
    rho_history: Union[np.ndarray, 'cp.ndarray'],
    history_count: Union[np.ndarray, 'cp.ndarray']
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    L-BFGS two-loop recursion for computing search direction.
    Vectorized over all N samples simultaneously.
    
    Args:
        grad: [N, D] current gradients
        s_history: [N, m, D] position differences
        y_history: [N, m, D] gradient differences
        rho_history: [N, m] scaling factors
        history_count: [N] number of valid history entries per sample
    
    Returns:
        search_dir: [N, D] search directions
    """
    xp = cp.get_array_module(grad) if GPU_AVAILABLE else np
    N, D = grad.shape
    m = s_history.shape[1]
    
    q = grad.copy()
    alpha = xp.zeros((N, m), dtype=grad.dtype)
    
    # First loop (backward through history)
    for i in range(m-1, -1, -1):
        # Only process samples with sufficient history
        valid = history_count > i
        
        if xp.any(valid):
            # alpha[valid, i] = rho[valid, i] * (s[valid, i] · q[valid])
            alpha[valid, i] = rho_history[valid, i] * xp.sum(
                s_history[valid, i] * q[valid], axis=1
            )
            q[valid] -= alpha[valid, i, None] * y_history[valid, i]
    
    # Initial Hessian approximation: H_0 = γI
    # γ = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
    valid = history_count > 0
    if xp.any(valid):
        last_idx = (history_count[valid] - 1).astype(int)
        s_last = s_history[valid, last_idx]
        y_last = y_history[valid, last_idx]
        
        sy = xp.sum(s_last * y_last, axis=1)
        yy = xp.sum(y_last * y_last, axis=1)
        gamma = sy / (yy + 1e-10)
        
        r = q.copy()
        r[valid] = gamma[:, None] * q[valid]
    else:
        r = q
    
    # Second loop (forward through history)
    for i in range(m):
        valid = history_count > i
        
        if xp.any(valid):
            beta = rho_history[valid, i] * xp.sum(
                y_history[valid, i] * r[valid], axis=1
            )
            r[valid] += s_history[valid, i] * (alpha[valid, i] - beta)[:, None]
    
    return r


# ============================================================================
# SINGLEWHIP INTEGRATION (v2.0)
# ============================================================================

def _compute_gradient_smart(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    whip: Optional['SingleWhip'] = None,
    batch_size: Optional[int] = None,
    epsilon: Optional[float] = None,
    func_returns_grad: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute gradient using best available method:
    1. Analytic gradient (if func_returns_grad=True) - O(1)
    2. SingleWhip (if available) - fast finite differences
    3. Legacy batched finite differences - O(D)
    
    Args:
        func: Likelihood function (may return gradient)
        x: Points to evaluate (N, d) or (d,)
        whip: SingleWhip instance (if available)
        batch_size: Batch size for legacy method
        epsilon: Epsilon for finite differences
        func_returns_grad: If True, func returns (L, grad) tuple
        
    Returns:
        grad: Gradient array, same shape as x
    """
    # Priority 1: Analytic gradient (fastest!)
    if func_returns_grad:
        _, grad = func(x)
        return grad
    
    # Priority 2: SingleWhip acceleration
    if whip is not None and SINGLEWHIP_AVAILABLE:
        xp = whip.xp
        
        if epsilon is None:
            epsilon = 1e-8
        
        if x.ndim == 1:
            return whip.evaluate_gradient(x, func, eps=epsilon)
        else:
            return whip.evaluate_gradient_batch(x, func, eps=epsilon)
    
    # Priority 3: Legacy batched finite differences
    return compute_gradient_batch(func, x, epsilon=epsilon, batch_size=batch_size)


def _compute_hessian_diagonal_smart(
    func: Callable,
    peaks: Union[np.ndarray, 'cp.ndarray'],
    whip: Optional['SingleWhip'] = None,
    eps: float = 1e-5,
    batch_size: Optional[int] = None
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute Hessian diagonal using direct finite differences.
    
    OPTIMIZED v2.5: O(2D) function evaluations instead of O(D²).
    Uses: H_ii = (f(x+h*e_i) - 2*f(x) + f(x-h*e_i)) / h²
    
    Args:
        func: Likelihood function
        peaks: [N, D] peak locations
        whip: SingleWhip instance (unused, kept for API compatibility)
        eps: Finite difference epsilon
        batch_size: Unused, kept for API compatibility
        
    Returns:
        hessian_diag: [N, D] Hessian diagonal elements
    """
    xp = cp.get_array_module(peaks) if GPU_AVAILABLE else np
    N, D = peaks.shape
    
    # Compute f(x) at all peaks
    f_x = func(peaks)  # (N,)
    
    # Compute Hessian diagonal using central differences
    # H_ii = (f(x+h*e_i) - 2*f(x) + f(x-h*e_i)) / h²
    hessian_diag = xp.zeros((N, D), dtype=peaks.dtype)
    
    for d in range(D):
        # Perturb dimension d
        x_plus = peaks.copy()
        x_minus = peaks.copy()
        x_plus[:, d] += eps
        x_minus[:, d] -= eps
        
        # Evaluate
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        # Second derivative
        hessian_diag[:, d] = (f_plus - 2*f_x + f_minus) / (eps**2)
    
    return hessian_diag


@register_optimizer('lbfgs')
def lbfgs_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    m: int = 10,
    tol: float = 1e-7,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    func_returns_grad: bool = False,  # Analytic gradient support
    line_search: str = 'armijo',  # 'armijo' (safe) or 'fixed' (fast)
    return_trajectories: bool = False  # NEW: Store intermediate positions for rotation detection
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Batched L-BFGS optimization for maximization.
    All N samples optimize simultaneously.
    
    80-20 ADAPTIVE RULE (v3.0):
        - Minimum 3 iterations always
        - When 80% converged at iteration N, allow N more for remaining 20%
        - Stop at 2N iterations max
    
    ANALYTIC GRADIENT SUPPORT (v3.1):
        If func_returns_grad=True, func(x) should return (L, grad) tuple.
        This is O(1) vs O(D) for finite differences - massive speedup!
    
    LINE SEARCH (v3.2):
        - 'armijo': Backtracking line search (RECOMMENDED - better convergence)
        - 'fixed': Fixed step with safety clamp (experimental, not recommended)
        
        Testing showed Armijo converges in fewer iterations, making it faster
        overall despite the per-iteration line search cost.
    
    TRAJECTORY STORAGE (v3.3):
        If return_trajectories=True, intermediate positions are stored for
        rotation detection in Module 3. Used for perpendicular step fraction
        analysis to detect rotated Gaussians.
    
    Args:
        func: Likelihood function(x) -> [N] or (x) -> ([N], [N,D]) if func_returns_grad
        x0: [N, D] initial positions
        maxiter: Maximum iterations (None = adaptive based on convergence)
        m: L-BFGS history size
        tol: Convergence tolerance (L_∞ norm of gradient)
        bounds: [D, 2] optional bounds [min, max]
        batch_size: Batch size for gradient computation (finite diff only)
        verbose: Print convergence info
        func_returns_grad: If True, func returns (L, grad) tuple
        line_search: 'armijo' (recommended) or 'fixed' (experimental)
        return_trajectories: If True, store and return intermediate positions
    
    Returns:
        Dictionary with:
            'x': [N, D] final positions
            'L': [N] likelihood values
            'converged': [N] bool convergence flags
            'niter': int iterations taken
            'grad_norm': [N] final gradient L_∞ norms
            'inv_hessian_scale': [N] gamma = s^T y / y^T y (≈ σ² for Gaussian)
            'history_count': [N] int, number of s,y pairs stored (0 = no movement)
            'trajectories': [N] list of [n_iters, D] arrays (if return_trajectories=True)
    """
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    # 80-20 adaptive iteration control
    min_iter = 3
    hard_max = maxiter if maxiter is not None else max(20, int(4 * np.log2(D)))
    iter_when_80pct = None  # Track when 80% converged
    adaptive_max = hard_max  # Will be updated when 80% converge
    
    # State
    x = x0.copy()
    
    # Trajectory storage (for rotation detection) - VECTORIZED, no Python loops
    if return_trajectories:
        # Pre-allocate on CPU - will store snapshots at each iteration
        # Shape: (max_iters+1, N, D) - we'll trim at the end
        max_iters_estimate = maxiter if maxiter is not None else max(20, int(4 * np.log2(D)))
        x0_cpu = x0.get() if hasattr(x0, 'get') else np.asarray(x0)
        trajectory_buffer = np.zeros((max_iters_estimate + 1, N, D), dtype=np.float64)
        trajectory_buffer[0] = x0_cpu
        traj_idx = 1
    
    # L-BFGS history
    s_history = xp.zeros((N, m, D), dtype=x.dtype)
    y_history = xp.zeros((N, m, D), dtype=x.dtype)
    rho_history = xp.zeros((N, m), dtype=x.dtype)
    history_count = xp.zeros(N, dtype=int)
    history_idx = xp.zeros(N, dtype=int)
    
    # Initial gradient - use analytic if available
    if func_returns_grad:
        _, grad = func(x)
    else:
        grad = compute_gradient_batch(func, x, batch_size=batch_size)
    
    converged = xp.zeros(N, dtype=bool)
    
    for iteration in range(hard_max):
        # Check convergence (L_∞ norm)
        grad_norm_inf = xp.max(xp.abs(grad), axis=1)
        newly_converged = (grad_norm_inf < tol) & ~converged
        converged |= newly_converged
        
        n_converged = int(xp.sum(converged))
        convergence_pct = n_converged / N
        
        # 80-20 rule: when 80% converge, set deadline for remaining 20%
        if iter_when_80pct is None and convergence_pct >= 0.8:
            iter_when_80pct = max(iteration, min_iter)
            adaptive_max = min(2 * iter_when_80pct, hard_max)
            if verbose:
                print(f"  80% converged at iter {iteration}, deadline={adaptive_max}")
        
        # Check stopping conditions
        if xp.all(converged):
            if verbose:
                print(f"All samples converged at iteration {iteration}")
            break
        
        # 80-20 deadline reached
        if iter_when_80pct is not None and iteration >= adaptive_max:
            if verbose:
                print(f"80-20 deadline reached at iter {iteration} ({n_converged}/{N} converged)")
            break
        
        # Minimum iterations not yet reached
        if iteration < min_iter:
            pass  # Continue regardless of convergence
        
        # Compute search direction using L-BFGS
        search_dir = lbfgs_two_loop_recursion(
            grad, s_history, y_history, rho_history, history_count
        )
        
        # Safety check: ensure search direction is ascending (for maximization)
        # grad^T * search_dir should be positive
        grad_dot_dir_check = xp.sum(grad * search_dir, axis=1)
        needs_flip = grad_dot_dir_check < 0
        if xp.any(needs_flip):
            # If search direction is descending, negate it
            search_dir[needs_flip] = -search_dir[needs_flip]
        
        # === LINE SEARCH ===
        if line_search == 'fixed':
            # FIXED STEP (fast): Trust L-BFGS direction, clamp step size
            step_norms = xp.linalg.norm(search_dir, axis=1)
            
            # Max step = 10% of domain size (or 1.0 if no bounds)
            if bounds is not None:
                domain_size = float(xp.max(bounds[:, 1] - bounds[:, 0]))
                max_step = 0.1 * domain_size
            else:
                max_step = 1.0
            
            # Adaptive alpha: min(1.0, max_step / ||direction||)
            alpha = xp.minimum(1.0, max_step / (step_norms + 1e-10))
            
            # Single step, no backtracking
            x_new = x + alpha[:, None] * search_dir
            if bounds is not None:
                x_new = xp.clip(x_new, bounds[:, 0], bounds[:, 1])
        
        else:  # 'armijo' (default)
            # ARMIJO BACKTRACKING (safe): Up to 20 iterations
            alpha = xp.ones(N, dtype=x.dtype)
            c1 = 1e-4
            
            # Get current L (extract from tuple if func returns grad)
            if func_returns_grad:
                L_current, _ = func(x)
            else:
                L_current = func(x)
            grad_dot_dir = xp.sum(grad * search_dir, axis=1)
            
            for _ in range(20):  # Max line search iterations
                x_new = x + alpha[:, None] * search_dir
                
                # Apply bounds if provided
                if bounds is not None:
                    x_new = xp.clip(x_new, bounds[:, 0], bounds[:, 1])
                
                # Get L_new (extract from tuple if func returns grad)
                if func_returns_grad:
                    L_new, _ = func(x_new)
                else:
                    L_new = func(x_new)
                
                # Armijo condition
                sufficient_decrease = L_new >= L_current + c1 * alpha * grad_dot_dir
                
                if xp.all(sufficient_decrease | converged):
                    break
                
                # Reduce step size for samples that didn't satisfy Armijo
                alpha = xp.where(sufficient_decrease | converged, alpha, alpha * 0.5)
            
            # Final position update
            x_new = x + alpha[:, None] * search_dir
            if bounds is not None:
                x_new = xp.clip(x_new, bounds[:, 0], bounds[:, 1])
        
        # Compute new gradient - use analytic if available
        if func_returns_grad:
            _, grad_new = func(x_new)
        else:
            grad_new = compute_gradient_batch(func, x_new, batch_size=batch_size)
        
        # Update L-BFGS history
        s_k = x_new - x
        y_k = grad - grad_new  # CORRECT sign for maximization (opposite of minimization!)
        
        # Curvature condition: s^T y > 0
        sy = xp.sum(s_k * y_k, axis=1)
        valid_update = sy > 1e-10
        
        # === HYBRID HISTORY UPDATE ===
        # Small D: fancy indexing is fast (low overhead)
        # Large D: scattered writes kill performance, use vectorized where()
        if D >= 100:
            # VECTORIZED (avoids slow scattered writes at high D)
            slot_indices = history_idx[:, None]  # [N, 1]
            slot_range = xp.arange(m)[None, :]   # [1, m]
            slot_mask = (slot_indices == slot_range)  # [N, m] boolean
            update_mask = valid_update[:, None] & slot_mask  # [N, m]
            
            s_history = xp.where(update_mask[:, :, None], s_k[:, None, :], s_history)
            y_history = xp.where(update_mask[:, :, None], y_k[:, None, :], y_history)
            rho_k = 1.0 / (sy + 1e-10)
            rho_history = xp.where(update_mask, rho_k[:, None], rho_history)
            
            history_count = xp.where(valid_update, xp.minimum(history_count + 1, m), history_count)
            history_idx = xp.where(valid_update, (history_idx + 1) % m, history_idx)
        else:
            # FANCY INDEXING (fast for small arrays)
            if xp.any(valid_update):
                idx = history_idx[valid_update]
                s_history[valid_update, idx] = s_k[valid_update]
                y_history[valid_update, idx] = y_k[valid_update]
                rho_history[valid_update, idx] = 1.0 / (sy[valid_update] + 1e-10)
                
                history_count[valid_update] = xp.minimum(history_count[valid_update] + 1, m)
                history_idx[valid_update] = (idx + 1) % m
        
        # Update state
        x = x_new
        grad = grad_new
        
        # Store trajectory positions (for rotation detection) - VECTORIZED
        if return_trajectories:
            x_cpu = x.get() if hasattr(x, 'get') else np.asarray(x)
            if traj_idx < trajectory_buffer.shape[0]:
                trajectory_buffer[traj_idx] = x_cpu
                traj_idx += 1
        
        if verbose and iteration % 10 == 0:
            n_converged = xp.sum(converged)
            mean_grad_norm = xp.mean(grad_norm_inf[~converged]) if n_converged < N else 0
            print(f"Iter {iteration}: {n_converged}/{N} converged, mean grad L∞: {mean_grad_norm:.2e}")
    
    # Final evaluation
    if func_returns_grad:
        L_final, _ = func(x)
    else:
        L_final = func(x)
    grad_norm_final = xp.max(xp.abs(grad), axis=1)
    
    # Compute inverse Hessian scale (gamma) from final s,y history
    # gamma = (s^T y) / (y^T y) approximates the diagonal of H^{-1}
    # For a Gaussian with log L = -0.5*x^T*Σ^{-1}*x, gamma ≈ σ²
    inv_hessian_scale = xp.ones(N, dtype=x.dtype)  # Default to 1
    
    valid = history_count > 0
    if xp.any(valid):
        # Get most recent s,y pair for each sample
        last_idx = xp.clip(history_count - 1, 0, m - 1).astype(int)
        
        # Extract s_last and y_last for valid samples
        valid_indices = xp.where(valid)[0]
        s_last = s_history[valid_indices, last_idx[valid_indices]]
        y_last = y_history[valid_indices, last_idx[valid_indices]]
        
        sy = xp.sum(s_last * y_last, axis=1)
        yy = xp.sum(y_last * y_last, axis=1)
        
        # gamma = s^T y / y^T y
        gamma = sy / (yy + 1e-10)
        
        # Ensure positive (should be for well-conditioned problems)
        gamma = xp.abs(gamma)
        
        inv_hessian_scale[valid_indices] = gamma
    
    # Build result dict
    result = {
        'x': x,
        'L': L_final,
        'converged': converged,
        'niter': iteration + 1,
        'grad_norm': grad_norm_final,
        'inv_hessian_scale': inv_hessian_scale,
        'history_count': history_count
    }
    
    # Add trajectories if requested (for rotation detection)
    if return_trajectories:
        # Trim buffer to actual iterations and convert to list of per-sample arrays
        # trajectory_buffer shape: (traj_idx, N, D) -> list of N arrays of shape (traj_idx, D)
        trimmed = trajectory_buffer[:traj_idx]  # (n_steps, N, D)
        result['trajectories'] = [trimmed[:, i, :] for i in range(N)]  # List of (n_steps, D)
    
    return result


# ============================================================================
# GRADIENT ASCENT OPTIMIZER
# ============================================================================

@register_optimizer('gradient_ascent')
def gradient_ascent_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    learning_rate: Union[str, float] = 'adaptive',
    tol: float = 1e-7,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Batched Gradient Ascent optimization.
    All N samples optimize simultaneously.
    
    Args:
        func: Likelihood function(x) -> [N]
        x0: [N, D] initial positions
        maxiter: Maximum iterations (None = adaptive: max(10, 3*log2(D)))
        learning_rate: 'adaptive' or fixed float value
        tol: Convergence tolerance (L_∞ norm of gradient)
        bounds: [D, 2] optional bounds
        batch_size: Batch size for gradient computation
        verbose: Print convergence info
    
    Returns:
        Dictionary (same format as lbfgs_batch)
    """
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    # Adaptive iteration count if not specified
    if maxiter is None:
        maxiter = max(10, int(3 * np.log2(D)))
        if verbose:
            print(f"  Adaptive maxiter: {maxiter} (D={D})")
    
    x = x0.copy()
    
    # Initialize learning rates
    if learning_rate == 'adaptive':
        lr = xp.ones(N, dtype=x.dtype) * 0.1
    else:
        lr = xp.ones(N, dtype=x.dtype) * learning_rate
    
    L_prev = func(x)
    converged = xp.zeros(N, dtype=bool)
    
    for iteration in range(maxiter):
        # Compute gradient
        grad = compute_gradient_batch(func, x, batch_size=batch_size)
        
        # Check convergence
        grad_norm_inf = xp.max(xp.abs(grad), axis=1)
        newly_converged = (grad_norm_inf < tol) & ~converged
        converged |= newly_converged
        
        if xp.all(converged):
            if verbose:
                print(f"All samples converged at iteration {iteration}")
            break
        
        # Update positions
        x_new = x + lr[:, None] * grad
        
        # Apply bounds
        if bounds is not None:
            x_new = xp.clip(x_new, bounds[:, 0], bounds[:, 1])
        
        # Evaluate new positions
        L_new = func(x_new)
        
        # Adaptive learning rate adjustment
        if learning_rate == 'adaptive':
            improved = L_new > L_prev
            lr = xp.where(improved, lr * 1.1, lr * 0.5)
            lr = xp.clip(lr, 1e-6, 1.0)
        
        # Accept update
        x = x_new
        L_prev = L_new
        
        if verbose and iteration % 10 == 0:
            n_converged = xp.sum(converged)
            mean_grad_norm = xp.mean(grad_norm_inf[~converged]) if n_converged < N else 0
            print(f"Iter {iteration}: {n_converged}/{N} converged, mean grad L∞: {mean_grad_norm:.2e}")
    
    grad_norm_final = xp.max(xp.abs(compute_gradient_batch(func, x, batch_size=batch_size)), axis=1)
    
    return {
        'x': x,
        'L': L_prev,
        'converged': converged,
        'niter': iteration + 1,
        'grad_norm': grad_norm_final
    }


# ============================================================================
# PI CHUAN (劈拳) - SPLITTING FIST - Gradient-Free Optimizers (v2.6)
# ============================================================================
#
# Named after the first of the Five Fists (五行拳) of Hsing-I Chuan (形意拳).
# Pi Chuan is a direct, powerful downward splitting strike - no wasted motion.
#
# Like the splitting fist, these methods strike directly at the optimum
# without the overhead of gradient computation. For slow likelihoods
# (Planck/CAMB ~0.5s per eval), gradients cost 12× per sample.
# Pi Chuan methods find the peak with minimal evaluations.
#
# The Five Fists:
#   劈拳 Pi Chuan (Splitting) - Metal - Direct, cutting through
#   崩拳 Beng Chuan (Crushing) - Wood - Explosive, forward
#   鑽拳 Zuan Chuan (Drilling) - Water - Spiraling, penetrating  
#   炮拳 Pao Chuan (Pounding) - Fire - Explosive, outward
#   橫拳 Heng Chuan (Crossing) - Earth - Crossing, stable
#
# ============================================================================

@register_optimizer('nelder_mead')
def nelder_mead_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-6,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Gradient-free Nelder-Mead optimization for slow likelihoods.
    
    Runs scipy.optimize.minimize with method='Nelder-Mead' on each sample.
    Much fewer function evaluations than L-BFGS for expensive likelihoods.
    
    For D=6: ~100-200 evaluations per sample vs ~1000+ for L-BFGS with gradients.
    
    Args:
        func: Likelihood function(x) -> [N] (we MAXIMIZE, so negate internally)
        x0: [N, D] initial positions
        maxiter: Maximum iterations per sample (None = 200*D)
        tol: Convergence tolerance
        bounds: [D, 2] optional bounds (applied via clipping, not constraints)
        batch_size: Ignored (for interface compatibility)
        verbose: Print progress
        
    Returns:
        Dictionary with:
            'x': [N, D] final positions
            'L': [N] likelihood values
            'converged': [N] bool convergence flags
            'niter': int total iterations
            'grad_norm': [N] zeros (no gradients computed)
    """
    from scipy.optimize import minimize
    
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    if maxiter is None:
        maxiter = 200 * D
    
    # Convert to NumPy for scipy
    if hasattr(x0, 'get'):
        x0_np = x0.get()
    else:
        x0_np = np.asarray(x0)
    
    if bounds is not None:
        if hasattr(bounds, 'get'):
            bounds_np = bounds.get()
        else:
            bounds_np = np.asarray(bounds)
    else:
        bounds_np = None
    
    # Results storage
    x_final = np.zeros_like(x0_np)
    L_final = np.zeros(N)
    converged = np.zeros(N, dtype=bool)
    total_nfev = 0
    
    # Single-point wrapper for scipy (NEGATED for minimization)
    def neg_likelihood_single(x_single):
        x_batch = x_single.reshape(1, -1)
        if bounds_np is not None:
            x_batch = np.clip(x_batch, bounds_np[:, 0], bounds_np[:, 1])
        result = func(x_batch)
        if hasattr(result, 'get'):
            result = result.get()
        return -float(result[0])  # Negate for minimization
    
    for i in range(N):
        if verbose and (i % 10 == 0 or i == N-1):
            print(f"  Nelder-Mead: sample {i+1}/{N}")
        
        result = minimize(
            neg_likelihood_single,
            x0_np[i],
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol}
        )
        
        x_opt = result.x
        if bounds_np is not None:
            x_opt = np.clip(x_opt, bounds_np[:, 0], bounds_np[:, 1])
        
        x_final[i] = x_opt
        L_final[i] = -result.fun  # Un-negate
        # Check convergence: scipy success OR significant improvement
        L_initial = -neg_likelihood_single(x0_np[i])
        converged[i] = result.success or (L_final[i] > L_initial + tol)
        total_nfev += result.nfev
    
    if verbose:
        print(f"  Nelder-Mead complete: {np.sum(converged)}/{N} converged, {total_nfev} total evals")
    
    # Convert back to GPU if needed
    if GPU_AVAILABLE and hasattr(x0, 'get'):
        x_final = cp.asarray(x_final)
        L_final = cp.asarray(L_final)
        converged = cp.asarray(converged)
    
    return {
        'x': x_final,
        'L': L_final,
        'converged': converged,
        'niter': total_nfev,
        'grad_norm': xp.zeros(N, dtype=x0.dtype),  # No gradients
        'inv_hessian_scale': xp.ones(N, dtype=x0.dtype),  # Default
        'history_count': xp.zeros(N, dtype=int)
    }


@register_optimizer('powell')
def powell_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-6,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Gradient-free Powell optimization for slow likelihoods.
    
    Uses scipy.optimize.minimize with method='Powell'.
    Performs sequential 1D line searches along each coordinate.
    
    Args:
        func: Likelihood function(x) -> [N] (we MAXIMIZE)
        x0: [N, D] initial positions
        maxiter: Maximum iterations per sample
        tol: Convergence tolerance
        bounds: [D, 2] optional bounds
        batch_size: Ignored
        verbose: Print progress
        
    Returns:
        Same structure as lbfgs_batch
    """
    from scipy.optimize import minimize
    
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    if maxiter is None:
        maxiter = 200 * D
    
    # Convert to NumPy
    if hasattr(x0, 'get'):
        x0_np = x0.get()
    else:
        x0_np = np.asarray(x0)
    
    if bounds is not None:
        if hasattr(bounds, 'get'):
            bounds_np = bounds.get()
        else:
            bounds_np = np.asarray(bounds)
    else:
        bounds_np = None
    
    x_final = np.zeros_like(x0_np)
    L_final = np.zeros(N)
    converged = np.zeros(N, dtype=bool)
    total_nfev = 0
    
    def neg_likelihood_single(x_single):
        x_batch = x_single.reshape(1, -1)
        if bounds_np is not None:
            x_batch = np.clip(x_batch, bounds_np[:, 0], bounds_np[:, 1])
        result = func(x_batch)
        if hasattr(result, 'get'):
            result = result.get()
        return -float(result[0])
    
    for i in range(N):
        if verbose and (i % 10 == 0 or i == N-1):
            print(f"  Powell: sample {i+1}/{N}")
        
        result = minimize(
            neg_likelihood_single,
            x0_np[i],
            method='Powell',
            options={'maxiter': maxiter, 'xtol': tol, 'ftol': tol}
        )
        
        x_opt = result.x
        if bounds_np is not None:
            x_opt = np.clip(x_opt, bounds_np[:, 0], bounds_np[:, 1])
        
        x_final[i] = x_opt
        L_final[i] = -result.fun
        # Check convergence: scipy success OR significant improvement
        L_initial = -neg_likelihood_single(x0_np[i])
        converged[i] = result.success or (L_final[i] > L_initial + tol)
        total_nfev += result.nfev
    
    if verbose:
        print(f"  Powell complete: {np.sum(converged)}/{N} converged, {total_nfev} total evals")
    
    if GPU_AVAILABLE and hasattr(x0, 'get'):
        x_final = cp.asarray(x_final)
        L_final = cp.asarray(L_final)
        converged = cp.asarray(converged)
    
    return {
        'x': x_final,
        'L': L_final,
        'converged': converged,
        'niter': total_nfev,
        'grad_norm': xp.zeros(N, dtype=x0.dtype),
        'inv_hessian_scale': xp.ones(N, dtype=x0.dtype),
        'history_count': xp.zeros(N, dtype=int)
    }


@register_optimizer('direct')
def direct_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-6,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    step_size: float = 0.01,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Simple direct pattern search - no gradients, minimal evaluations.
    
    For each sample:
    1. Evaluate current point
    2. Try ±step along each dimension
    3. Move to best neighbor
    4. Repeat until no improvement or maxiter
    
    Very cheap: ~2D evaluations per iteration (not 12D like L-BFGS gradients).
    
    Args:
        func: Likelihood function(x) -> [N]
        x0: [N, D] initial positions
        maxiter: Maximum iterations (default: 50)
        tol: Improvement tolerance for convergence
        bounds: [D, 2] optional bounds
        batch_size: Ignored
        step_size: Initial step size (fraction of domain if bounds given)
        verbose: Print progress
        
    Returns:
        Same structure as lbfgs_batch
    """
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    if maxiter is None:
        maxiter = 50
    
    x = x0.copy()
    
    # Compute step sizes
    if bounds is not None:
        domain_size = bounds[:, 1] - bounds[:, 0]
        steps = step_size * domain_size
    else:
        steps = xp.ones(D, dtype=x0.dtype) * step_size
    
    # Initial likelihood
    L = func(x)
    converged = xp.zeros(N, dtype=bool)
    
    for iteration in range(maxiter):
        improved_any = False
        
        for d in range(D):
            # Try +step
            x_plus = x.copy()
            x_plus[:, d] += steps[d]
            if bounds is not None:
                x_plus[:, d] = xp.clip(x_plus[:, d], bounds[d, 0], bounds[d, 1])
            L_plus = func(x_plus)
            
            # Try -step
            x_minus = x.copy()
            x_minus[:, d] -= steps[d]
            if bounds is not None:
                x_minus[:, d] = xp.clip(x_minus[:, d], bounds[d, 0], bounds[d, 1])
            L_minus = func(x_minus)
            
            # Accept best improvement
            improve_plus = (L_plus > L + tol) & ~converged
            improve_minus = (L_minus > L + tol) & ~converged & ~improve_plus
            
            # Prefer larger improvement
            prefer_minus = improve_minus & (L_minus > L_plus)
            improve_plus = improve_plus & ~prefer_minus
            improve_minus = improve_minus | prefer_minus
            
            if xp.any(improve_plus):
                x = xp.where(improve_plus[:, None], x_plus, x)
                L = xp.where(improve_plus, L_plus, L)
                improved_any = True
            
            if xp.any(improve_minus):
                x = xp.where(improve_minus[:, None], x_minus, x)
                L = xp.where(improve_minus, L_minus, L)
                improved_any = True
        
        if not improved_any:
            # Reduce step size
            steps = steps * 0.5
            if xp.max(steps) < tol:
                converged[:] = True
                break
        
        if verbose and iteration % 10 == 0:
            print(f"  Direct iter {iteration}: best L = {float(xp.max(L)):.4f}")
    
    converged = xp.ones(N, dtype=bool)  # All "converged" (no gradient to check)
    
    if verbose:
        print(f"  Direct complete: {iteration+1} iterations, best L = {float(xp.max(L)):.4f}")
    
    return {
        'x': x,
        'L': L,
        'converged': converged,
        'niter': iteration + 1,
        'grad_norm': xp.zeros(N, dtype=x0.dtype),
        'inv_hessian_scale': xp.ones(N, dtype=x0.dtype),
        'history_count': xp.zeros(N, dtype=int)
    }


@register_optimizer('none')
def none_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-6,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    No optimization - just evaluate and return.
    
    For cases where ray exploration already found the peak and
    no refinement is needed. Useful for very expensive likelihoods.
    
    Args:
        func: Likelihood function(x) -> [N]
        x0: [N, D] initial positions
        All other args: Ignored
        
    Returns:
        Same structure as lbfgs_batch (with samples as-is)
    """
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    x = x0.copy()
    
    # Clip to bounds if provided
    if bounds is not None:
        x = xp.clip(x, bounds[:, 0], bounds[:, 1])
    
    # Single evaluation
    L = func(x)
    
    if verbose:
        print(f"  No optimization: {N} samples, best L = {float(xp.max(L)):.4f}")
    
    return {
        'x': x,
        'L': L,
        'converged': xp.ones(N, dtype=bool),  # All "converged"
        'niter': 1,
        'grad_norm': xp.zeros(N, dtype=x0.dtype),
        'inv_hessian_scale': xp.ones(N, dtype=x0.dtype),
        'history_count': xp.zeros(N, dtype=int)
    }


# ============================================================================
# RANDCOORD OPTIMIZER (v3.1 - Sublinear Gradient Scaling)
# ============================================================================

@register_optimizer('randcoord')
def randcoord_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    maxiter: Optional[int] = None,
    tol: float = 1e-7,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    func_returns_grad: bool = False,
    line_search: str = 'armijo',
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    RandCoord + Line Search optimizer (wrapper for SingleWhip v1.5).
    
    Uses k = 16 + 8*log₂(D) random coordinates per iteration.
    Proven faster than L-BFGS at D ≥ 384:
        D=512:  1.14x speedup
        D=768:  1.17x speedup
        D=1024: 1.64x speedup
    
    Falls back to L-BFGS if SingleWhip v1.5 not available.
    """
    if not RANDCOORD_AVAILABLE:
        if verbose:
            print("  RandCoord not available, falling back to L-BFGS")
        return lbfgs_batch(
            func, x0, maxiter=maxiter, tol=tol, bounds=bounds,
            batch_size=batch_size, verbose=verbose,
            func_returns_grad=func_returns_grad, line_search=line_search
        )
    
    return randcoord_line_search_batch(
        func, x0,
        maxiter=maxiter,
        tol=tol,
        bounds=bounds,
        verbose=verbose,
        func_returns_grad=func_returns_grad,
        k_scale=(16, 8)
    )


# ============================================================================
# WIDTH ESTIMATION
# ============================================================================

def estimate_peak_width(
    func: Callable,
    peaks: Union[np.ndarray, 'cp.ndarray'],
    method: str = 'gradient_probe',
    grad_threshold: float = 1e-5,
    L_drop_factor: float = np.e,
    n_probe_steps: int = 50,
    max_probe_distance: float = 1.0,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    whip: Optional['SingleWhip'] = None,
    batch_size: Optional[int] = None
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Estimate peak widths under L_∞ metric.
    Returns [longest_axis, shortest_axis] for each peak.
    
    Args:
        func: Likelihood function
        peaks: [N, D] peak locations
        method: 'gradient_probe', 'likelihood_drop', or 'hessian'
        grad_threshold: Gradient threshold for 'gradient_probe'
        L_drop_factor: Likelihood drop factor for 'likelihood_drop'
        n_probe_steps: Number of probe steps per direction
        max_probe_distance: Maximum distance to probe
        bounds: Optional domain bounds
        whip: SingleWhip instance for acceleration (v2.0)
        batch_size: Batch size for legacy gradient computation
    
    Returns:
        widths: [N, 2] where widths[:, 0] = longest, widths[:, 1] = shortest
    """
    xp = cp.get_array_module(peaks) if GPU_AVAILABLE else np
    N, D = peaks.shape
    
    if method == 'gradient_probe':
        # Probe along each axis until gradient exceeds threshold
        steps = xp.linspace(0, max_probe_distance, n_probe_steps, dtype=peaks.dtype)
        
        # Create probe points [N, D, 2, S, D]
        # For each peak, each dimension, probe ±direction with S steps
        width_per_dim = xp.zeros((N, D), dtype=peaks.dtype)
        
        for d in range(D):
            # Probe in +d and -d directions
            probe_plus = peaks[:, None, :] + steps[None, :, None] * xp.eye(D, dtype=peaks.dtype)[d]
            probe_minus = peaks[:, None, :] - steps[None, :, None] * xp.eye(D, dtype=peaks.dtype)[d]
            
            # Stack: [N, 2*S, D]
            probe_both = xp.concatenate([probe_plus, probe_minus], axis=1)
            
            # Apply bounds if provided
            if bounds is not None:
                probe_both = xp.clip(probe_both, bounds[:, 0], bounds[:, 1])
            
            # Flatten for batch evaluation: [N*2*S, D]
            probe_flat = probe_both.reshape(N * 2 * n_probe_steps, D)
            
            # Compute gradients
            grad_flat = compute_gradient_batch(func, probe_flat)
            grad_norm = xp.max(xp.abs(grad_flat), axis=1)  # L_∞ norm
            
            # Reshape: [N, 2*S]
            grad_norm = grad_norm.reshape(N, 2 * n_probe_steps)
            
            # Find first exceeding threshold
            exceeds = grad_norm > grad_threshold
            first_exceed = xp.argmax(exceeds, axis=1)
            
            # Handle never exceeding case
            never_exceeds = ~xp.any(exceeds, axis=1)
            first_exceed = xp.where(never_exceeds, n_probe_steps - 1, first_exceed)
            
            # Convert to distance
            width_per_dim[:, d] = steps[first_exceed % n_probe_steps]
        
        # Extract longest and shortest
        widths_longest = xp.max(width_per_dim, axis=1)
        widths_shortest = xp.min(width_per_dim, axis=1)
    
    elif method == 'likelihood_drop':
        # Similar to gradient_probe but using likelihood drop
        steps = xp.linspace(0, max_probe_distance, n_probe_steps, dtype=peaks.dtype)
        L_peaks = func(peaks)
        threshold = L_peaks / L_drop_factor
        
        width_per_dim = xp.zeros((N, D), dtype=peaks.dtype)
        
        for d in range(D):
            probe_plus = peaks[:, None, :] + steps[None, :, None] * xp.eye(D, dtype=peaks.dtype)[d]
            probe_minus = peaks[:, None, :] - steps[None, :, None] * xp.eye(D, dtype=peaks.dtype)[d]
            probe_both = xp.concatenate([probe_plus, probe_minus], axis=1)
            
            if bounds is not None:
                probe_both = xp.clip(probe_both, bounds[:, 0], bounds[:, 1])
            
            probe_flat = probe_both.reshape(N * 2 * n_probe_steps, D)
            L_flat = func(probe_flat)
            L_reshaped = L_flat.reshape(N, 2 * n_probe_steps)
            
            # Find first drop below threshold
            below_threshold = L_reshaped < threshold[:, None]
            first_drop = xp.argmax(below_threshold, axis=1)
            never_drops = ~xp.any(below_threshold, axis=1)
            first_drop = xp.where(never_drops, n_probe_steps - 1, first_drop)
            
            width_per_dim[:, d] = steps[first_drop % n_probe_steps]
        
        widths_longest = xp.max(width_per_dim, axis=1)
        widths_shortest = xp.min(width_per_dim, axis=1)
    
    elif method == 'hessian':
        # Hessian-based approximation (less accurate for L_∞)
        # v2.0: Use smart Hessian computation (SingleWhip if available)
        epsilon = 1e-5
        
        # Smart Hessian diagonal computation (1 call vs d² calls!)
        hessian_diag = _compute_hessian_diagonal_smart(
            func, peaks, whip=whip, eps=epsilon, batch_size=batch_size
        )
        
        # Width ~ 1/sqrt(|hessian|)
        width_per_dim = 1.0 / xp.sqrt(xp.abs(hessian_diag) + 1e-10)
        
        widths_longest = xp.max(width_per_dim, axis=1)
        widths_shortest = xp.min(width_per_dim, axis=1)
    
    else:
        raise ValueError(f"Unknown width estimation method: {method}")
    
    return xp.stack([widths_longest, widths_shortest], axis=1)


# ============================================================================
# STICKY HANDS HEURISTIC
# ============================================================================

def _compute_diagonal_hessian_batch(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    epsilon: float = 1e-5,
    batch_size: Optional[int] = None,
    func_returns_grad: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute diagonal of Hessian matrix for batch of points.
    Uses finite differences: H_ii ≈ (f(x+e_i) - 2f(x) + f(x-e_i)) / h²
    
    Cheaper than full Hessian: O(D) instead of O(D²)
    
    Args:
        func: Likelihood function
        x: [N, D] positions
        epsilon: Step size for finite differences
        batch_size: Batch size for function evaluations
        func_returns_grad: If True, func returns (L, grad) tuple
    
    Returns:
        [N, D] diagonal Hessian elements
    """
    xp = cp.get_array_module(x) if GPU_AVAILABLE else np
    N, D = x.shape
    
    # Compute f(x) - use eval_L_only to handle tuple returns
    f_x = eval_L_only(func, x, func_returns_grad)
    
    # For each dimension, compute second derivative
    H_diag = xp.zeros((N, D), dtype=x.dtype)
    
    for d in range(D):
        # Create perturbations
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[:, d] += epsilon
        x_minus[:, d] -= epsilon
        
        # Evaluate - use eval_L_only
        f_plus = eval_L_only(func, x_plus, func_returns_grad)
        f_minus = eval_L_only(func, x_minus, func_returns_grad)
        
        # Second derivative
        H_diag[:, d] = (f_plus - 2*f_x + f_minus) / (epsilon**2)
    
    return H_diag


def _is_saddle_point_batch(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    epsilon: float = 1e-5,
    curvature_threshold: float = 1e-6,
    func_returns_grad: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Detect saddle points via diagonal Hessian check.
    For MAXIMIZATION: saddle if not all diagonal elements are negative.
    
    A true peak for maximization has all H_ii < 0 (negative curvature in all directions).
    A saddle point has mixed curvature (some positive, some negative).
    
    Args:
        func: Likelihood function
        x: [N, D] positions to check
        epsilon: Step size for Hessian computation
        curvature_threshold: Threshold for considering curvature significant
        func_returns_grad: If True, func returns (L, grad) tuple
    
    Returns:
        [N] bool array - True if sample is at saddle point
    """
    xp = cp.get_array_module(x) if GPU_AVAILABLE else np
    
    H_diag = _compute_diagonal_hessian_batch(func, x, epsilon=epsilon, 
                                              func_returns_grad=func_returns_grad)
    
    # For maximization, peak has all H_ii < 0
    # Saddle has at least one H_ii >= 0
    is_saddle = ~xp.all(H_diag < -curvature_threshold, axis=1)
    
    return is_saddle


def _reseed_sunburst(
    unconverged_samples: Union[np.ndarray, 'cp.ndarray'],
    K_lost: int,
    ray_length: float,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']],
    max_attempts: int = 10
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Sunburst reseeding: generate random rays from unconverged samples.
    
    FULLY VECTORIZED - no Python loops. All operations are single GPU calls.
    
    Strategy (Repulse Monkey 摊手伏虎):
        1. From each unconverged sample, shoot rays in random directions
        2. Sample uniformly along each ray
        3. Clip to bounds (single GPU call)
        4. Distribute K_lost rays across N_unconverged samples (round-robin)
    
    Args:
        unconverged_samples: [N_unconverged, D] positions to seed from
        K_lost: Number of new samples to generate (duplicates removed)
        ray_length: Initial ray length
        bounds: [D, 2] prior bounds
        max_attempts: (unused, kept for API compatibility)
    
    Returns:
        [K_lost, D] new sample positions
    """
    xp = cp.get_array_module(unconverged_samples) if GPU_AVAILABLE else np
    N_unconverged, D = unconverged_samples.shape
    
    if N_unconverged == 0:
        # No unconverged samples - return random samples in bounds
        if bounds is not None:
            return xp.random.uniform(
                bounds[:, 0], bounds[:, 1], 
                size=(K_lost, D)
            ).astype(unconverged_samples.dtype)
        else:
            return xp.random.randn(K_lost, D).astype(unconverged_samples.dtype)
    
    # === FULLY VECTORIZED GENERATION ===
    
    # 1. Assign each ray to a source (round-robin)
    source_indices = xp.arange(K_lost) % N_unconverged
    origins = unconverged_samples[source_indices]  # [K_lost, D]
    
    # 2. Generate all random directions at once
    directions = xp.random.randn(K_lost, D).astype(unconverged_samples.dtype)
    norms = xp.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms  # Normalize to unit sphere
    
    # 3. Generate all t values at once
    t = xp.random.uniform(0, ray_length, size=(K_lost,)).astype(unconverged_samples.dtype)
    
    # 4. Compute all new samples at once
    new_samples = origins + t[:, None] * directions
    
    # 5. Clip to bounds (single GPU call)
    if bounds is not None:
        new_samples = xp.clip(new_samples, bounds[:, 0], bounds[:, 1])
    
    return new_samples


def _golden_rooster_reseed(
    converged_peaks: Union[np.ndarray, 'cp.ndarray'],
    unconverged_samples: Union[np.ndarray, 'cp.ndarray'],
    K_to_generate: int,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']],
    max_attempts: int = 10
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Golden Rooster (金雞獨立) reseeding: shoot rays in ORTHONORMAL directions from peaks.
    
    v2.6 FIX: Uses QR decomposition to generate orthonormal basis vectors instead
    of random directions. This ensures systematic exploration of ALL dimensions,
    which is critical for finding modes in high-D spaces.
    
    In 256D with random directions, the probability of hitting a mode at a specific
    location is essentially zero. With QR-orthonormal directions, we guarantee
    coverage of all D independent directions from each source.
    
    FULLY VECTORIZED - no Python loops.
    
    Strategy:
        1. Combine converged peaks + unconverged samples as sources
        2. Generate orthonormal basis via QR decomposition of random matrix
        3. For each source, shoot rays along basis vectors (both ±directions)
        4. Distribute K_to_generate rays across sources and directions
    
    Args:
        converged_peaks: [N_converged, D] peak positions (stuck samples)
        unconverged_samples: [N_unconverged, D] unconverged positions
        K_to_generate: Number of new samples to generate
        bounds: [D, 2] prior bounds (required for ray length)
        max_attempts: (unused, kept for API compatibility)
    
    Returns:
        [K_to_generate, D] new sample positions
    """
    xp = cp.get_array_module(converged_peaks) if GPU_AVAILABLE else np
    
    # Combine sources: converged peaks + unconverged samples
    if len(unconverged_samples) > 0:
        sources = xp.vstack([converged_peaks, unconverged_samples])
    else:
        sources = converged_peaks
    
    N_sources, D = sources.shape
    
    if N_sources == 0 or bounds is None:
        # Fallback: random samples in bounds
        if bounds is not None:
            return xp.random.uniform(
                bounds[:, 0], bounds[:, 1],
                size=(K_to_generate, D)
            ).astype(converged_peaks.dtype)
        else:
            return xp.random.randn(K_to_generate, D).astype(converged_peaks.dtype)
    
    # Ray length = longest prior dimension / 2
    prior_widths = bounds[:, 1] - bounds[:, 0]
    ray_length = float(xp.max(prior_widths)) / 2.0
    
    # === QR-BASED ORTHONORMAL DIRECTIONS ===
    
    # Generate orthonormal basis via QR decomposition
    # Random matrix -> QR gives uniformly distributed orthonormal basis
    random_matrix = xp.random.randn(D, D).astype(converged_peaks.dtype)
    
    # QR decomposition - Q is orthonormal
    Q, _ = xp.linalg.qr(random_matrix)
    
    # Q is [D, D] orthonormal matrix - each column is a basis vector
    # We have 2D directions: ±Q[:, i] for i = 0..D-1
    
    # Build direction pool: both positive and negative of each basis vector
    # directions_pool: [2*D, D]
    directions_pool = xp.vstack([Q.T, -Q.T])  # [2D, D]
    n_directions = 2 * D
    
    # === FULLY VECTORIZED DISTRIBUTION ===
    
    # Assign each ray to (source, direction) pair via indexing
    # Pattern: cycle through directions first, then sources
    ray_indices = xp.arange(K_to_generate)
    
    # Source assignment: which source does ray k come from?
    source_indices = ray_indices % N_sources
    
    # Direction assignment: which direction does ray k use?
    # Cycle through all 2D directions for each source
    dir_indices = (ray_indices // N_sources) % n_directions
    
    # Gather origins and directions (fully vectorized)
    origins = sources[source_indices]  # [K_to_generate, D]
    directions = directions_pool[dir_indices]  # [K_to_generate, D]
    
    # Random distances along rays (avoid starting exactly at origin)
    t = xp.random.uniform(0.1 * ray_length, ray_length, size=(K_to_generate,)).astype(converged_peaks.dtype)
    
    # Compute all new samples at once
    new_samples = origins + t[:, None] * directions
    
    # Clip to bounds (single GPU call)
    new_samples = xp.clip(new_samples, bounds[:, 0], bounds[:, 1])
    
    return new_samples


def _reseed_perturb(
    unconverged_samples: Union[np.ndarray, 'cp.ndarray'],
    K_lost: int,
    noise_scale: float,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']],
    max_attempts: int = 10
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Perturb reseeding: add Gaussian noise to unconverged samples.
    
    Strategy (Repulse Monkey variant):
        1. Randomly select from unconverged samples (with replacement)
        2. Add Gaussian noise: new = sample + N(0, noise_scale²)
        3. If outside bounds, reduce noise_scale and retry
    
    Good for: Local exploration around unconverged points, smooth landscapes
    
    Args:
        unconverged_samples: [N_unconverged, D] positions to seed from
        K_lost: Number of new samples to generate
        noise_scale: Standard deviation of Gaussian noise (ray_length parameter)
        bounds: [D, 2] prior bounds
        max_attempts: Max noise reductions if outside bounds
    
    Returns:
        [K_lost, D] new sample positions
    """
    xp = cp.get_array_module(unconverged_samples) if GPU_AVAILABLE else np
    N_unconverged, D = unconverged_samples.shape
    
    if N_unconverged == 0:
        # No unconverged samples - return random samples
        if bounds is not None:
            return xp.random.uniform(
                bounds[:, 0], bounds[:, 1], 
                size=(K_lost, D)
            ).astype(unconverged_samples.dtype)
        else:
            return xp.random.randn(K_lost, D).astype(unconverged_samples.dtype)
    
    new_samples = []
    
    for _ in range(K_lost):
        # Randomly select an unconverged sample (with replacement)
        idx = xp.random.randint(0, N_unconverged)
        origin = unconverged_samples[idx]
        
        # Try adding Gaussian noise with boundary checking
        current_scale = noise_scale
        valid = False
        
        for attempt in range(max_attempts):
            # Add Gaussian noise
            noise = xp.random.randn(D).astype(unconverged_samples.dtype) * current_scale
            new_sample = origin + noise
            
            # Check bounds
            if bounds is not None:
                if xp.all((new_sample >= bounds[:, 0]) & (new_sample <= bounds[:, 1])):
                    valid = True
                    break
                else:
                    # Reduce noise and retry
                    current_scale *= 0.5
            else:
                valid = True
                break
        
        if valid:
            new_samples.append(new_sample)
        else:
            # Failed - use origin with tiny perturbation
            new_samples.append(origin + 1e-6 * xp.random.randn(D).astype(unconverged_samples.dtype))
    
    return xp.stack(new_samples, axis=0)


def _reseed_radial(
    unconverged_samples: Union[np.ndarray, 'cp.ndarray'],
    K_lost: int,
    ray_length: float,
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']],
    max_attempts: int = 10
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Radial reseeding: shoot rays along coordinate axes and main diagonals.
    
    Strategy (Repulse Monkey variant):
        1. For each unconverged sample, generate rays in structured directions:
           - Positive/negative coordinate axes: ±e_i
           - Main diagonals: [±1, ±1, ..., ±1] normalized
        2. Sample uniformly along each ray
        3. Distribute K_lost rays across available directions
    
    Directions (for D dimensions):
        - 2D axes: ±e_1, ±e_2, ..., ±e_D
        - Main diagonals: [±1, ±1, ..., ±1] (limited to first 3 dims)
        - Total: 2D + 2^min(D,3) directions per sample
    
    Good for: Axis-aligned features, structured exploration, escaping saddles
    
    Args:
        unconverged_samples: [N_unconverged, D] positions to seed from
        K_lost: Number of new samples to generate
        ray_length: Ray length
        bounds: [D, 2] prior bounds
        max_attempts: Max halvings if outside bounds
    
    Returns:
        [K_lost, D] new sample positions
    """
    xp = cp.get_array_module(unconverged_samples) if GPU_AVAILABLE else np
    N_unconverged, D = unconverged_samples.shape
    
    if N_unconverged == 0:
        # No unconverged samples - return random samples
        if bounds is not None:
            return xp.random.uniform(
                bounds[:, 0], bounds[:, 1], 
                size=(K_lost, D)
            ).astype(unconverged_samples.dtype)
        else:
            return xp.random.randn(K_lost, D).astype(unconverged_samples.dtype)
    
    # Build direction set: coordinate axes + main diagonals
    directions = []
    
    # Coordinate axes: ±e_i for i=1..D
    for i in range(D):
        direction = xp.zeros(D, dtype=unconverged_samples.dtype)
        direction[i] = 1.0
        directions.append(direction)
        directions.append(-direction)
    
    # Main diagonals: [±1, ±1, ..., ±1] (limit to first 3 dims to avoid explosion)
    n_diag_dims = min(D, 3)
    for signs in range(2**n_diag_dims):
        diagonal = xp.ones(D, dtype=unconverged_samples.dtype)
        for i in range(n_diag_dims):
            if signs & (1 << i):
                diagonal[i] = -1.0
        diagonal = diagonal / xp.linalg.norm(diagonal)  # Normalize
        directions.append(diagonal)
    
    directions = xp.stack(directions, axis=0)  # [N_directions, D]
    N_directions = len(directions)
    
    # Intelligent distribution of K_lost samples across (N_unconverged × N_directions) combinations
    total_combinations = N_unconverged * N_directions
    
    new_samples = []
    
    for k in range(K_lost):
        # Cycle through combinations
        combo_idx = k % total_combinations
        sample_idx = combo_idx // N_directions
        direction_idx = combo_idx % N_directions
        
        origin = unconverged_samples[sample_idx]
        direction = directions[direction_idx]
        
        # Sample uniformly along ray with boundary checking
        current_length = ray_length
        valid = False
        
        for attempt in range(max_attempts):
            # Uniform distance along ray
            t = xp.random.uniform(0, current_length)
            new_sample = origin + t * direction
            
            # Check bounds
            if bounds is not None:
                if xp.all((new_sample >= bounds[:, 0]) & (new_sample <= bounds[:, 1])):
                    valid = True
                    break
                else:
                    # Halve distance and retry
                    current_length *= 0.5
            else:
                valid = True
                break
        
        if valid:
            new_samples.append(new_sample)
        else:
            # Failed - use origin with small perturbation
            new_samples.append(origin + 1e-6 * xp.random.randn(D).astype(unconverged_samples.dtype))
    
    return xp.stack(new_samples, axis=0)


# Dictionary of reseeding strategies
_RESEED_STRATEGIES = {
    'sunburst': _reseed_sunburst,
    'perturb': _reseed_perturb,
    'radial': _reseed_radial,
    # Future strategies can be added here:
    # 'sticky': _reseed_sticky_anticonverge,
    # 'adaptive': _reseed_adaptive
}


def deduplicate_peaks_L_infinity(
    peaks: Union[np.ndarray, 'cp.ndarray'],
    L_peaks: Union[np.ndarray, 'cp.ndarray'],
    tolerance: float = 1e-6,
    keep_best: bool = True,
    verbose: bool = False,
    return_mask: bool = False,
    inv_hessian_scale: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    hessian_valid_mask: Optional[Union[np.ndarray, 'cp.ndarray']] = None
) -> Union[
    Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']],
    Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]
]:
    """
    Deduplicate peaks using L_∞ (max) metric with grid-based spatial hashing.
    
    Two peaks are considered duplicates if their L_∞ distance is below tolerance:
        L_∞(a, b) = max_i |a_i - b_i| < tolerance
    
    Algorithm: O(N log N) via spatial hashing
        1. Create grid with cell size = tolerance
        2. Hash peaks to grid cells
        3. Check only peaks in same/neighboring cells
        4. Keep best peak from each cluster
        5. Transfer widest valid Hessian to winner (v1.8)
    
    Args:
        peaks: [N, D] peak locations
        L_peaks: [N] likelihood values at peaks
        tolerance: L_∞ distance threshold for merging
        keep_best: If True, keep peak with highest L; otherwise keep first
        verbose: Print deduplication stats
        return_mask: If True, also return the boolean mask of kept peaks
        inv_hessian_scale: [N] optional inverse Hessian scale values (modified in-place)
        hessian_valid_mask: [N] optional bool mask, True = valid Hessian (modified in-place)
    
    Returns:
        Base: (peaks_dedup, L_dedup)
        If return_mask=True: adds keep_mask to tuple
        If inv_hessian_scale provided: adds ihs_dedup to tuple
        
        Full signature: (peaks_dedup, L_dedup, [keep_mask], [ihs_dedup])
    """
    xp = cp.get_array_module(peaks) if GPU_AVAILABLE else np
    N, D = peaks.shape
    
    if N == 0:
        empty_mask = xp.array([], dtype=bool)
        if return_mask and inv_hessian_scale is not None:
            return peaks, L_peaks, empty_mask, xp.array([], dtype=peaks.dtype)
        elif return_mask:
            return peaks, L_peaks, empty_mask
        elif inv_hessian_scale is not None:
            return peaks, L_peaks, xp.array([], dtype=peaks.dtype)
        return peaks, L_peaks
    
    # Grid-based spatial hashing
    # Cell size = tolerance ensures L_∞ < tolerance only within same cell (centered)
    # Use ROUND instead of FLOOR to center cells at multiples of cell_size
    cell_size = tolerance
    
    # Convert peaks to grid coordinates (centered cells)
    # Use int64 to avoid overflow with large coordinates
    # Handle inf/nan values (which can occur from division)
    # NumPy would warn about these, but we explicitly handle them
    grid_float = peaks / cell_size
    
    # Replace inf/nan with finite values to avoid issues
    mask_inf = ~xp.isfinite(grid_float)
    if xp.any(mask_inf):
        grid_float = grid_float.copy()  # Avoid modifying input
        grid_float[mask_inf] = 0.0
    
    # Convert to integer grid coordinates
    grid_coords = xp.round(grid_float).astype(xp.int64)
    
    # Create unique cell IDs (hash to 1D)
    # Use prime number multiplication to reduce collisions
    cell_ids = grid_coords[:, 0].copy()
    for d in range(1, D):
        cell_ids = cell_ids * 73856093 + grid_coords[:, d] * 19349663
    
    # Sort by cell ID for efficient grouping
    sort_idx = xp.argsort(cell_ids)
    peaks_sorted = peaks[sort_idx]
    L_sorted = L_peaks[sort_idx]
    cell_ids_sorted = cell_ids[sort_idx]
    
    # Mark duplicates - VECTORIZED (v3.0)
    keep_mask = xp.ones(N, dtype=bool)
    
    # Process each unique cell
    unique_cells, cell_starts = xp.unique(cell_ids_sorted, return_index=True)
    cell_starts = xp.concatenate([cell_starts, xp.array([N])])
    n_cells = len(unique_cells)
    
    # Get cell sizes to identify multi-peak cells
    cell_sizes = cell_starts[1:] - cell_starts[:-1]
    multi_peak_cells = xp.where(cell_sizes > 1)[0]
    
    # Process multi-peak cells with vectorized operations
    for cell_idx in multi_peak_cells:
        start = int(cell_starts[cell_idx])
        end = int(cell_starts[cell_idx + 1])
        n_cell = end - start
        
        # Get cell data
        cell_peaks = peaks_sorted[start:end]
        cell_L = L_sorted[start:end]
        cell_sort_idx = sort_idx[start:end]
        
        # Memory check: [n_cell, n_cell, D] tensor would use n_cell^2 * D * 8 bytes
        # Limit to ~256MB to be safe
        max_cell_for_vectorized = int(np.sqrt(256e6 / (D * 8)))
        
        if n_cell <= max_cell_for_vectorized:
            # Vectorized pairwise L∞ distance: [n_cell, n_cell]
            # diff[j, k, d] = cell_peaks[j, d] - cell_peaks[k, d]
            diff = cell_peaks[:, None, :] - cell_peaks[None, :, :]  # [n, n, D]
            L_inf_dists = xp.max(xp.abs(diff), axis=2)  # [n, n]
            
            # Find duplicates (upper triangle only, excluding diagonal)
            is_duplicate = L_inf_dists < tolerance
            is_duplicate = xp.triu(is_duplicate, k=1)  # Upper triangle, k=1 excludes diagonal
            
            # For each duplicate pair, decide who to keep
            # Local keep mask for this cell
            cell_keep = xp.ones(n_cell, dtype=bool)
            
            if keep_best:
                # Build elimination mask using vectorized ops
                L_comparison = cell_L[:, None] >= cell_L[None, :]
                k_eliminated = xp.any(is_duplicate & L_comparison, axis=0)
                j_eliminated = xp.any(is_duplicate & ~L_comparison, axis=1)
                cell_keep = ~(k_eliminated | j_eliminated)
                
                # Handle Hessian transfer if needed
                if inv_hessian_scale is not None and hessian_valid_mask is not None:
                    dup_pairs = xp.argwhere(is_duplicate)
                    n_dup_pairs = len(dup_pairs)
                    
                    if n_dup_pairs > 0 and n_dup_pairs <= 1000:
                        for pair in dup_pairs:
                            j_local, k_local = int(pair[0]), int(pair[1])
                            idx_j = cell_sort_idx[j_local]
                            idx_k = cell_sort_idx[k_local]
                            
                            if cell_L[j_local] >= cell_L[k_local]:
                                winner_idx, loser_idx = idx_j, idx_k
                            else:
                                winner_idx, loser_idx = idx_k, idx_j
                            
                            loser_valid = bool(hessian_valid_mask[loser_idx])
                            winner_valid = bool(hessian_valid_mask[winner_idx])
                            loser_wider = inv_hessian_scale[loser_idx] > inv_hessian_scale[winner_idx]
                            
                            if loser_valid and (not winner_valid or loser_wider):
                                inv_hessian_scale[winner_idx] = inv_hessian_scale[loser_idx]
                                hessian_valid_mask[winner_idx] = True
                    elif n_dup_pairs > 1000:
                        # Fast path for massive duplicate sets
                        survivor_mask = cell_keep
                        if xp.any(survivor_mask):
                            survivor_local_idx = int(xp.argmax(survivor_mask))
                            survivor_global_idx = cell_sort_idx[survivor_local_idx]
                            
                            cell_global_idx = cell_sort_idx
                            cell_valid = hessian_valid_mask[cell_global_idx]
                            if xp.any(cell_valid):
                                cell_ihs = inv_hessian_scale[cell_global_idx]
                                cell_ihs_masked = xp.where(cell_valid, cell_ihs, -xp.inf)
                                widest_local_idx = int(xp.argmax(cell_ihs_masked))
                                widest_global_idx = cell_global_idx[widest_local_idx]
                                
                                inv_hessian_scale[survivor_global_idx] = inv_hessian_scale[widest_global_idx]
                                hessian_valid_mask[survivor_global_idx] = True
            else:
                k_eliminated = xp.any(is_duplicate, axis=0)
                cell_keep = ~k_eliminated
                
                if inv_hessian_scale is not None and hessian_valid_mask is not None:
                    dup_pairs = xp.argwhere(is_duplicate)
                    n_dup_pairs = len(dup_pairs)
                    
                    if n_dup_pairs > 0 and n_dup_pairs <= 1000:
                        for pair in dup_pairs:
                            j_local, k_local = int(pair[0]), int(pair[1])
                            idx_j = cell_sort_idx[j_local]
                            idx_k = cell_sort_idx[k_local]
                            
                            k_valid = bool(hessian_valid_mask[idx_k])
                            j_valid = bool(hessian_valid_mask[idx_j])
                            k_wider = inv_hessian_scale[idx_k] > inv_hessian_scale[idx_j]
                            
                            if k_valid and (not j_valid or k_wider):
                                inv_hessian_scale[idx_j] = inv_hessian_scale[idx_k]
                                hessian_valid_mask[idx_j] = True
                    elif n_dup_pairs > 1000:
                        survivor_global_idx = cell_sort_idx[0]
                        cell_global_idx = cell_sort_idx
                        cell_valid = hessian_valid_mask[cell_global_idx]
                        if xp.any(cell_valid):
                            cell_ihs = inv_hessian_scale[cell_global_idx]
                            cell_ihs_masked = xp.where(cell_valid, cell_ihs, -xp.inf)
                            widest_local_idx = int(xp.argmax(cell_ihs_masked))
                            widest_global_idx = cell_global_idx[widest_local_idx]
                            inv_hessian_scale[survivor_global_idx] = inv_hessian_scale[widest_global_idx]
                            hessian_valid_mask[survivor_global_idx] = True
        else:
            # Large cell: all samples are effectively duplicates (within tolerance of grid cell)
            # Just keep the one with highest L
            cell_keep = xp.zeros(n_cell, dtype=bool)
            if keep_best:
                best_idx = int(xp.argmax(cell_L))
            else:
                best_idx = 0  # Keep first
            cell_keep[best_idx] = True
            
            # Transfer widest valid Hessian to survivor
            if inv_hessian_scale is not None and hessian_valid_mask is not None:
                survivor_global_idx = cell_sort_idx[best_idx]
                cell_global_idx = cell_sort_idx
                cell_valid = hessian_valid_mask[cell_global_idx]
                if xp.any(cell_valid):
                    cell_ihs = inv_hessian_scale[cell_global_idx]
                    cell_ihs_masked = xp.where(cell_valid, cell_ihs, -xp.inf)
                    widest_local_idx = int(xp.argmax(cell_ihs_masked))
                    widest_global_idx = cell_global_idx[widest_local_idx]
                    inv_hessian_scale[survivor_global_idx] = inv_hessian_scale[widest_global_idx]
                    hessian_valid_mask[survivor_global_idx] = True
        
        # Apply cell keep mask to global keep mask
        keep_mask[cell_sort_idx[~cell_keep]] = False
    
    # Extract kept peaks
    peaks_dedup = peaks[keep_mask]
    L_dedup = L_peaks[keep_mask]
    
    if verbose:
        n_removed = N - xp.sum(keep_mask)
        print(f"  Deduplication: {N} → {len(peaks_dedup)} peaks ({n_removed} removed)")
    
    # Build return tuple based on options
    if return_mask and inv_hessian_scale is not None:
        return peaks_dedup, L_dedup, keep_mask, inv_hessian_scale[keep_mask]
    elif return_mask:
        return peaks_dedup, L_dedup, keep_mask
    elif inv_hessian_scale is not None:
        return peaks_dedup, L_dedup, inv_hessian_scale[keep_mask]
    return peaks_dedup, L_dedup


def _detect_boundary_stuck(x, bounds, wall_threshold, xp=np):
    """
    Detect samples within wall_threshold of any boundary.
    
    A sample is "boundary-stuck" if it's within wall_threshold distance
    of ANY boundary (lower or upper) in ANY dimension.
    
    Args:
        x: Sample positions (N, d)
        bounds: Domain bounds (d, 2) with [lower, upper] for each dimension
        wall_threshold: Distance threshold for "stuck" detection
        xp: numpy or cupy
    
    Returns:
        stuck_mask: Boolean array (N,) indicating stuck samples
        n_stuck: Number of stuck samples
    """
    if bounds is None:
        # No bounds, no stuck samples
        return xp.zeros(len(x), dtype=bool), 0
    
    lower = bounds[:, 0]  # (d,)
    upper = bounds[:, 1]  # (d,)
    
    # Distance to lower and upper boundaries in each dimension
    dist_to_lower = xp.abs(x - lower)  # (N, d)
    dist_to_upper = xp.abs(x - upper)  # (N, d)
    
    # Stuck if within threshold of ANY boundary in ANY dimension
    stuck_at_lower = xp.any(dist_to_lower < wall_threshold, axis=1)  # (N,)
    stuck_at_upper = xp.any(dist_to_upper < wall_threshold, axis=1)  # (N,)
    
    stuck_mask = stuck_at_lower | stuck_at_upper
    n_stuck = int(xp.sum(stuck_mask))
    
    return stuck_mask, n_stuck


def _choose_distant_vertex(stuck_point, bounds, xp=np):
    """
    Choose a random NON-NEIGHBORING vertex (distant corner).
    
    Strategy:
        - For each dimension, determine if stuck_point is closer to lower or upper bound
        - Choose the OPPOSITE bound for that dimension
        - This ensures the vertex is in a "distant" corner
    
    Args:
        stuck_point: Position of stuck sample (d,)
        bounds: Domain bounds (d, 2)
        xp: numpy or cupy
    
    Returns:
        distant_vertex: (d,) array representing distant corner
    """
    lower = bounds[:, 0]  # (d,)
    upper = bounds[:, 1]  # (d,)
    d = len(stuck_point)
    
    # For each dimension, find which boundary is closer
    dist_to_lower = xp.abs(stuck_point - lower)
    dist_to_upper = xp.abs(stuck_point - upper)
    
    # Choose opposite boundary (if closer to lower, pick upper; vice versa)
    distant_vertex = xp.where(dist_to_lower < dist_to_upper, upper, lower)
    
    return distant_vertex


def _cannon_through_sky(
    x,
    bounds,
    cannon_ray_length=1.0,
    cannon_max_attempts=10,
    cannon_wall_threshold=1e-3,
    verbose=False,
    xp=np
):
    """
    Cannon Through the Sky (炮打天空): Rescue boundary-stuck samples.
    
    VECTORIZED v3.0: Processes all stuck samples in parallel on GPU.
    
    After anti-convergence, some samples may be pushed against domain boundaries.
    This function "cannons" them away by shooting rays to distant corners.
    
    Algorithm (vectorized):
        1. Detect samples within cannon_wall_threshold of boundaries
        2. For ALL stuck samples at once:
            a. Choose distant vertices (opposite corners)
            b. Compute ray vectors: stuck_points → distant_vertices
            c. Start at cannon_ray_length, halve for out-of-bounds samples
            d. Iterate until all in bounds or max attempts reached
    
    Args:
        x: Sample positions (N, d)
        bounds: Domain bounds (d, 2)
        cannon_ray_length: Initial fraction of ray length to travel (0 to 1)
        cannon_max_attempts: Max ray halving attempts if out of bounds
        cannon_wall_threshold: Distance threshold for boundary detection
        verbose: Print diagnostics
        xp: numpy or cupy
    
    Returns:
        x_cannoned: Updated sample positions (N, d)
        n_cannoned: Number of samples that were cannoned
    """
    N, d = x.shape
    
    # Detect boundary-stuck samples
    stuck_mask, n_stuck = _detect_boundary_stuck(x, bounds, cannon_wall_threshold, xp=xp)
    
    if n_stuck == 0:
        if verbose:
            print("    No boundary-stuck samples detected")
        return x, 0
    
    if verbose:
        print(f"    Detected {n_stuck} boundary-stuck samples")
    
    # Extract stuck samples
    stuck_indices = xp.where(stuck_mask)[0]
    stuck_points = x[stuck_indices]  # (n_stuck, d)
    
    lower = bounds[:, 0]  # (d,)
    upper = bounds[:, 1]  # (d,)
    
    # === VECTORIZED: Choose distant vertices for ALL stuck samples ===
    # For each dimension, find which boundary is closer
    dist_to_lower = xp.abs(stuck_points - lower)  # (n_stuck, d)
    dist_to_upper = xp.abs(stuck_points - upper)  # (n_stuck, d)
    
    # Choose opposite boundary (if closer to lower, pick upper; vice versa)
    distant_vertices = xp.where(dist_to_lower < dist_to_upper, upper, lower)  # (n_stuck, d)
    
    # === VECTORIZED: Compute ray vectors ===
    ray_vectors = distant_vertices - stuck_points  # (n_stuck, d)
    ray_lengths = xp.linalg.norm(ray_vectors, axis=1)  # (n_stuck,)
    
    # Handle edge case: samples already at vertex
    valid_rays = ray_lengths > 1e-10
    if not xp.any(valid_rays):
        if verbose:
            print("    All stuck samples already at vertices")
        return x, 0
    
    # Normalize ray directions (only for valid rays)
    ray_directions = xp.zeros_like(ray_vectors)
    ray_directions[valid_rays] = ray_vectors[valid_rays] / ray_lengths[valid_rays, None]
    
    # === VECTORIZED: Iterative ray length adjustment ===
    # Start with full cannon_ray_length for all samples
    current_lengths = xp.full(n_stuck, cannon_ray_length, dtype=x.dtype)
    success = xp.zeros(n_stuck, dtype=bool)
    new_positions = stuck_points.copy()
    
    for attempt in range(cannon_max_attempts):
        # Compute candidate positions for non-successful samples
        active = valid_rays & ~success
        if not xp.any(active):
            break
        
        # Compute new positions: stuck + length * full_ray_length * direction
        candidates = stuck_points + (current_lengths[:, None] * ray_lengths[:, None]) * ray_directions
        
        # Check bounds
        in_lower = candidates >= lower  # (n_stuck, d)
        in_upper = candidates <= upper  # (n_stuck, d)
        in_bounds = xp.all(in_lower & in_upper, axis=1)  # (n_stuck,)
        
        # Mark successful samples
        newly_success = active & in_bounds
        success |= newly_success
        new_positions[newly_success] = candidates[newly_success]
        
        # Halve ray length for unsuccessful samples
        still_active = active & ~in_bounds
        current_lengths[still_active] *= 0.5
    
    # === Update original array ===
    x_cannoned = x.copy()
    x_cannoned[stuck_indices[success]] = new_positions[success]
    
    n_cannoned = int(xp.sum(success))
    n_failed = int(xp.sum(valid_rays & ~success))
    
    if verbose:
        print(f"    Successfully cannoned {n_cannoned}/{n_stuck} samples")
        if n_failed > 0:
            print(f"    Failed to cannon {n_failed} samples (kept original positions)")
    
    return x_cannoned, n_cannoned


def sticky_hands(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    method: str = 'randcoord',
    n_converge: int = 10,
    n_anticonverge: int = 5,
    n_oscillations: int = 3,
    stick_tolerance: float = 1e-6,
    reseed_strategy: Optional[Union[str, Callable]] = None,
    reseed_ray_length: float = 0.1,
    reseed_max_attempts: int = 10,
    cannon_through_sky: bool = False,
    cannon_ray_length: float = 1.0,
    cannon_max_attempts: int = 10,
    cannon_wall_threshold: float = 1e-3,
    anti_momentum: float = 0.9,
    anti_step_size: float = 0.1,
    # v2.2: Clouding parameters
    cloud_sigma: Optional[float] = None,
    n_cloud: int = 2,
    cloud_K: int = 10,
    cloud_step_size: float = 0.1,
    cloud_enabled: bool = True,
    # v2.3: Two-phase mode support
    mode: str = 'full',  # 'full' or 'hlc_only'
    # Other parameters
    width_method: str = 'gradient_probe',
    bounds: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    batch_size: Optional[int] = None,
    estimate_widths: bool = True,
    verbose: bool = False,
    track_history: bool = False,
    # v2.4: Sample banking
    bank_samples: bool = False,
    # v3.1: Analytic gradient support
    func_returns_grad: bool = False,
    # v3.2: Line search mode
    line_search: str = 'armijo',  # 'armijo' (safe) or 'fixed' (fast)
    **optimizer_kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Sticky Hands: Convergence-anticonvergence oscillation strategy.
    
    IMPORTANT - LOG-LIKELIHOOD OPERATION:
        ChiSao expects func to return LOG-LIKELIHOOD (log(L)), not likelihood (L).
        This matches SunBURST deployment: Module 0 → log(L) → Module 1 (ChiSao).
        
        The 10% high-likelihood threshold operates in log-space:
          - Likelihood space: L >= 0.1 * L_max
          - Log-likelihood space: log(L) >= log(L_max) + log(0.1)
          - Equivalently: log(L) >= log(L_max) - 2.303
        
        Why log-likelihood?
          - Prevents numerical underflow in high dimensions
          - Standard practice in Bayesian inference
          - Required for evidence calculation anyway
    
    Strategy:
        1. Converge for n_converge iterations
        2. "Stick" converged samples (freeze them)
        3. Anti-converge non-stuck samples for n_anticonverge iterations
        4. Release all, repeat K times
        5. Optional: Repulse Monkey reseeding to maintain sample count
        6. NEW v2.2: Hands Like Clouds phase between reseeding and anti-convergence
    
    Hands Like Clouds (雲手) - v2.2:
        Uses smoothed gradients to see global structure through local noise.
        After reseeding, before anti-convergence, takes n_cloud gradient ascent
        steps using a Gaussian-smoothed gradient. This blurs out small-scale
        oscillations (like Rastrigin/Ackley bumps) to reveal global basins.
        Only applies to UNSTUCK samples.
    
    Repulse Monkey (摊手伏虎):
        After deduplication removes K duplicates, reseed K new samples from
        unconverged points (not stuck or at saddle points) to maintain sample count.
        Enabled by setting reseed_strategy to a string or Callable.
    
    Args:
        func: Log-likelihood function (MUST return log(L), not L)
        x0: [N, D] initial positions
        method: 'lbfgs' or 'gradient_ascent'
        n_converge: Iterations per convergence phase
        n_anticonverge: Iterations per anti-convergence phase
        n_oscillations: Number of full oscillation cycles (K)
        stick_tolerance: Gradient L_∞ threshold for "stuck"
        reseed_strategy: Reseeding strategy for Repulse Monkey:
            - None: Disable reseeding (Repulse Monkey off)
            - str: Use internal strategy ('sunburst', 'perturb', 'radial')
            - Callable: External function with signature:
                reseed_func(unconverged_samples, K_lost, ray_length, bounds, max_attempts) → new_samples
        reseed_ray_length: Initial ray length for sunburst reseeding
        reseed_max_attempts: Max halvings if reseeded sample outside bounds
        cannon_through_sky: Enable Cannon Through the Sky boundary rescue
        cannon_ray_length: Fraction of ray length to travel when cannoning (default: 1.0)
        cannon_max_attempts: Max ray halving attempts if out of bounds (default: 10)
        cannon_wall_threshold: Distance threshold for boundary detection (default: 1e-3)
        anti_momentum: Momentum coefficient for anti-convergence (0=none, 0.99=heavy, default: 0.9)
        anti_step_size: Step size for anti-convergence gradient steps (default: 0.1)
        cloud_sigma: Smoothing scale for Hands Like Clouds (auto-estimated from sample spread if None)
        n_cloud: Number of Hands Like Clouds gradient steps (default: 2)
        cloud_K: Number of perturbation samples for smoothed gradient (default: 10)
        cloud_step_size: Step size for Hands Like Clouds gradient ascent (default: 0.1)
        cloud_enabled: Enable/disable Hands Like Clouds phase (default: True, requires SingleWhip v1.3)
        mode: Operation mode (v2.3):
            - 'full': Normal operation (HLC + convergence + anti-convergence) [default]
            - 'hlc_only': Extended Hands Like Clouds only (global features, no local exploration)
                          For two-phase detection: Phase 1 uses hlc_only to find global optima
        width_method: Method for width estimation
        bounds: Optional domain bounds
        batch_size: Batch size for gradient computation
        estimate_widths: Whether to estimate peak widths (disable for high-D benchmarks)
        verbose: Print progress
        track_history: If True, record sample positions at each stage (for debugging/visualization)
        **optimizer_kwargs: Additional args for optimizer
    
    Returns:
        Dictionary with:
            'peaks': [N_converged, D] peak locations
            'L_peaks': [N_converged] likelihood at peaks
            'widths': [N_converged, 2] [longest, shortest] widths
            'converged_mask': [N] bool which samples converged
            'n_oscillations_completed': int
            'n_reseeded': int (if reseed_strategy is not None)
            'n_cannoned': int (if cannon_through_sky=True)
            'cloud_sigma': float (estimated or provided Hands Like Clouds scale)
    """
    xp = cp.get_array_module(x0) if GPU_AVAILABLE else np
    N, D = x0.shape
    
    x = x0.copy()
    
    # v2.0: Initialize SingleWhip for gradient/Hessian acceleration
    whip = None
    if SINGLEWHIP_AVAILABLE:
        try:
            whip = SingleWhip(use_gpu=GPU_AVAILABLE)
            if verbose:
                print(f"ChiSao v3.2: SingleWhip v{SINGLEWHIP_VERSION} acceleration enabled (GPU={whip.is_gpu})")
        except Exception as e:
            warnings.warn(f"Failed to initialize SingleWhip: {e}. Using legacy gradient computation.")
            whip = None
    elif verbose:
        print("ChiSao v3.2: SingleWhip not available, using legacy gradient computation")
    
    stuck_mask = xp.zeros(N, dtype=bool)
    total_reseeded = 0  # Track total samples reseeded
    
    # History tracking
    history = [] if track_history else None
    if track_history:
        xp_cpu = x.get() if hasattr(x, 'get') else x  # GPU→CPU if needed
        history.append({'stage': 'initial', 'oscillation': 0, 'samples': xp_cpu.copy()})
    
    # v2.4: Sample banking for evidence calculation
    sample_bank = None
    if bank_samples:
        sample_bank = SampleBank(D=D, xp=xp)
        # Bank initial samples
        L_init = eval_L_only(func, x, func_returns_grad)
        sample_bank.add_samples(x, L_init, 'initial', 0)
    
    if method not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {method}. Available: {list(_OPTIMIZERS.keys())}")
    
    optimizer_func = _OPTIMIZERS[method]
    
    # Track cannoned samples
    total_cannoned = 0
    
    # v1.8: Track inv_hessian_scale and validity across oscillations
    inv_hessian_agg = None  # Will be initialized on first convergence
    hessian_valid_agg = None  # Tracks which samples have valid (non-default) Hessian
    
    # v2.1: Adaptive Golden Rooster - disable if exploration doesn't find new peaks
    gr_enabled = True
    gr_fired_last_oscillation = False
    n_peaks_before_gr = 0
    
    # v2.2: Hands Like Clouds setup
    # Auto-estimate cloud_sigma from initial sample spread if not provided
    actual_cloud_sigma = cloud_sigma
    cloud_available = cloud_enabled and whip is not None and hasattr(whip, 'evaluate_gradient_smoothed')
    
    if cloud_enabled and whip is None:
        if verbose:
            print("ChiSao v3.2: Hands Like Clouds disabled (requires SingleWhip v1.3+)")
        cloud_available = False
    elif cloud_enabled and not hasattr(whip, 'evaluate_gradient_smoothed'):
        if verbose:
            print("ChiSao v3.2: Hands Like Clouds disabled (SingleWhip missing evaluate_gradient_smoothed)")
        cloud_available = False
    
    if cloud_available and actual_cloud_sigma is None:
        # Estimate sigma from median pairwise distance / 2
        # This gives a scale that's meaningful for the current sample spread
        if N >= 2:
            # Sample a subset for efficiency (don't need all pairs)
            n_sample = min(100, N)
            idx = xp.random.choice(N, size=n_sample, replace=False) if N > n_sample else xp.arange(N)
            x_sample = x[idx]
            
            # Compute pairwise L2 distances
            diff = x_sample[:, None, :] - x_sample[None, :, :]  # [n, n, D]
            dists = xp.sqrt(xp.sum(diff**2, axis=2))  # [n, n]
            
            # Median of upper triangle (non-zero distances)
            upper_mask = xp.triu(xp.ones((n_sample, n_sample), dtype=bool), k=1)
            upper_dists = dists[upper_mask]
            actual_cloud_sigma = float(xp.median(upper_dists)) / 2
            
            # Clamp to reasonable range
            if bounds is not None:
                domain_scale = float(xp.mean(bounds[:, 1] - bounds[:, 0]))
                actual_cloud_sigma = max(0.01 * domain_scale, min(actual_cloud_sigma, 0.5 * domain_scale))
            
            if verbose:
                print(f"ChiSao v3.2: Hands Like Clouds enabled (auto σ={actual_cloud_sigma:.4f})")
        else:
            actual_cloud_sigma = 1.0  # Fallback
            if verbose:
                print(f"ChiSao v3.2: Hands Like Clouds enabled (default σ={actual_cloud_sigma})")
    elif cloud_available and verbose:
        print(f"ChiSao v3.2: Hands Like Clouds enabled (provided σ={actual_cloud_sigma:.4f})")
    
    # v2.3: HLC-only mode adjustments
    # Extended Hands Like Clouds for global feature discovery, minimal local refinement
    if mode == 'hlc_only':
        n_oscillations = 1        # Single pass
        n_converge = 3            # Minimal convergence
        n_anticonverge = 0        # No anti-convergence (skip local exploration)
        n_cloud = n_cloud * 5     # 5× more HLC iterations for global discovery
        if verbose:
            print(f"ChiSao v3.2: HLC-only mode (n_cloud={n_cloud}, n_converge={n_converge}, n_anticonverge=0)")
    elif mode != 'full':
        raise ValueError(f"Unknown mode: {mode}. Available: 'full', 'hlc_only'")
    
    for k in range(n_oscillations):
        if verbose:
            print(f"\n=== Oscillation {k+1}/{n_oscillations} ===")
        
        # === CONVERGENCE PHASE ===
        if verbose:
            print(f"Convergence phase ({n_converge} max iterations)...")
        
        # Release sticking for fresh convergence
        stuck_mask[:] = False
        
        # Run optimizer for n_converge iterations
        result = optimizer_func(
            func, x,
            maxiter=n_converge,
            tol=stick_tolerance / 10,
            bounds=bounds,
            batch_size=batch_size,
            verbose=False,
            func_returns_grad=func_returns_grad,
            line_search=line_search,
            **optimizer_kwargs
        )
        
        x = result['x']
        grad_norm = result['grad_norm']
        L = result['L']
        
        # Print actual iterations used (80-20 rule may reduce this)
        actual_iters = result.get('niter', n_converge)
        if verbose and actual_iters < n_converge:
            print(f"  (80-20 rule: used {actual_iters}/{n_converge} iterations)")
        
        # v1.8: Extract inv_hessian_scale and history_count from this convergence phase
        ihs = result.get('inv_hessian_scale', None)
        history_count = result.get('history_count', None)
        if ihs is not None:
            # Validity based on history_count (0 = no movement, garbage Hessian)
            valid_new = history_count > 0 if history_count is not None else (ihs != 1.0)
            
            if inv_hessian_agg is None:
                inv_hessian_agg = ihs.copy()
                hessian_valid_agg = valid_new.copy() if hasattr(valid_new, 'copy') else xp.array(valid_new)
            else:
                # Update with new values where they're valid
                inv_hessian_agg = xp.where(valid_new, ihs, inv_hessian_agg)
                hessian_valid_agg = hessian_valid_agg | valid_new
        
        if track_history:
            xp_cpu = x.get() if hasattr(x, 'get') else x
            history.append({'stage': 'post_converge', 'oscillation': k+1, 'samples': xp_cpu.copy()})
        
        # v2.4: Bank samples after convergence
        if bank_samples:
            sample_bank.add_samples(x, L, 'post_converge', k+1, stuck_mask)
        
        # Update stuck mask (only for high-likelihood convergences)
        # Low-L peaks (< 10% of max) stay active for reseeding
        # 
        # LOG-SPACE THRESHOLD:
        # In likelihood space: L >= 0.1 * L_max
        # In log-likelihood space: log(L) >= log(L_max) + log(0.1)
        # Since log(0.1) ≈ -2.303, the operational demand becomes:
        #   log(L) >= log(L_max) - 2.303
        max_L = xp.max(L)
        log_threshold = max_L + xp.log(0.1)  # log(L_max) + log(0.1) = log(0.1 * L_max)
        is_high_likelihood = L >= log_threshold
        newly_stuck = (grad_norm < stick_tolerance) & is_high_likelihood
        stuck_mask |= newly_stuck
        
        n_stuck = xp.sum(stuck_mask)
        if verbose:
            n_high_L = xp.sum(is_high_likelihood)
            print(f"  {n_stuck}/{N} samples stuck (at high-L peaks, {n_high_L} above 10% threshold)")
        
        # Deduplicate converged samples
        N_before_dedup = len(x)
        dedup_tol = stick_tolerance / 2
        
        # v1.8: Pass Hessian info to dedup for widest-valid transfer
        if inv_hessian_agg is not None and hessian_valid_agg is not None:
            # Make copies so dedup can modify them in-place
            ihs_for_dedup = inv_hessian_agg.copy()
            valid_for_dedup = hessian_valid_agg.copy()
            x, L, dedup_mask, ihs_dedup = deduplicate_peaks_L_infinity(
                x, L, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True,
                inv_hessian_scale=ihs_for_dedup, hessian_valid_mask=valid_for_dedup
            )
            inv_hessian_agg = ihs_dedup
            hessian_valid_agg = valid_for_dedup[dedup_mask]
        else:
            x, L, dedup_mask = deduplicate_peaks_L_infinity(
                x, L, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True
            )
        
        N_after_dedup = len(x)
        K_lost = N_before_dedup - N_after_dedup
        stuck_mask = stuck_mask[dedup_mask]  # Adjust mask using same dedup_mask
        
        # v2.1: Check if Golden Rooster helped (compare peak counts after dedup)
        if gr_fired_last_oscillation:
            if N_after_dedup <= n_peaks_before_gr:
                gr_enabled = False
                if verbose:
                    print(f"  [Adaptive GR: {N_after_dedup} peaks ≤ {n_peaks_before_gr} before → disabled]")
            else:
                if verbose:
                    print(f"  [Adaptive GR: {N_after_dedup} peaks > {n_peaks_before_gr} → still active]")
            gr_fired_last_oscillation = False
        
        if track_history:
            xp_cpu = x.get() if hasattr(x, 'get') else x
            history.append({'stage': 'post_dedup', 'oscillation': k+1, 'samples': xp_cpu.copy()})
        
        # v2.4: Bank samples after deduplication
        if bank_samples:
            sample_bank.add_samples(x, L, 'post_dedup', k+1, stuck_mask)
        
        # === REPULSE MONKEY RESEEDING ===
        # Maintain sample count by reseeding from unconverged points
        if reseed_strategy is not None and K_lost > 0 and k < n_oscillations - 1:  # Not on last oscillation
            if verbose:
                print(f"  Repulse Monkey: {K_lost} duplicates removed, reseeding...")
            
            # Identify unconverged samples: NOT stuck OR at saddle points
            is_saddle = _is_saddle_point_batch(func, x, epsilon=stick_tolerance,
                                               func_returns_grad=func_returns_grad)
            unconverged_mask = ~stuck_mask | is_saddle
            N_unconverged = xp.sum(unconverged_mask)
            
            if verbose:
                n_not_stuck = xp.sum(~stuck_mask)
                n_saddle = xp.sum(is_saddle)
                print(f"    Unconverged: {N_unconverged} samples ({n_not_stuck} not stuck, {n_saddle} saddles)")
            
            # === GOLDEN ROOSTER (金雞獨立) ===
            # When unconverged < 5, use converged peaks as additional sunburst sources
            # Philosophy: GPU has thousands of cores - use what's freely given!
            # v2.1: Adaptive - disable if exploration doesn't find new peaks
            if N_unconverged < 5 and gr_enabled:
                n_peaks_before_gr = len(x)  # Track for next oscillation
                gr_fired_last_oscillation = True
                
                converged_peaks = x[stuck_mask]
                unconverged_samples = x[unconverged_mask] if N_unconverged > 0 else xp.zeros((0, D), dtype=x.dtype)
                
                if verbose:
                    print(f"    Golden Rooster (金雞獨立): {len(converged_peaks)} peaks + {N_unconverged} unconverged → {K_lost} new samples")
                
                # Generate new samples from peaks + unconverged
                new_samples = _golden_rooster_reseed(
                    converged_peaks,
                    unconverged_samples,
                    K_lost,
                    bounds,
                    reseed_max_attempts
                )
                
                # Append to existing samples
                x = xp.vstack([x, new_samples])
                
                # Evaluate likelihood for new samples
                L_new = eval_L_only(func, new_samples, func_returns_grad)
                L = xp.concatenate([L, L_new])
                
                # Golden Rooster samples are UNSTUCK (can explore via anti-convergence)
                stuck_mask_new = xp.zeros(K_lost, dtype=bool)
                stuck_mask = xp.concatenate([stuck_mask, stuck_mask_new])
                
                # Extend Hessian arrays
                if inv_hessian_agg is not None:
                    ihs_new = xp.ones(K_lost, dtype=inv_hessian_agg.dtype)
                    inv_hessian_agg = xp.concatenate([inv_hessian_agg, ihs_new])
                if hessian_valid_agg is not None:
                    valid_new = xp.zeros(K_lost, dtype=bool)
                    hessian_valid_agg = xp.concatenate([hessian_valid_agg, valid_new])
                
                N = len(x)
                total_reseeded += K_lost
                
                if verbose:
                    print(f"    Generated {K_lost} new samples, total now: {N}")
                
                if track_history:
                    xp_cpu = x.get() if hasattr(x, 'get') else x
                    history.append({'stage': 'post_golden_rooster', 'oscillation': k+1, 'samples': xp_cpu.copy()})
            
            elif N_unconverged > 0:
                unconverged_samples = x[unconverged_mask]
                
                # Resolve reseeding function
                if callable(reseed_strategy):
                    # External callable provided (e.g., from Module 1)
                    reseed_func = reseed_strategy
                elif isinstance(reseed_strategy, str):
                    # Internal strategy by name
                    if reseed_strategy not in _RESEED_STRATEGIES:
                        raise ValueError(
                            f"Unknown reseed_strategy: {reseed_strategy}. "
                            f"Available: {list(_RESEED_STRATEGIES.keys())}"
                        )
                    reseed_func = _RESEED_STRATEGIES[reseed_strategy]
                else:
                    raise TypeError(
                        f"reseed_strategy must be None, str, or Callable, got {type(reseed_strategy)}"
                    )
                
                # Generate new samples
                new_samples = reseed_func(
                    unconverged_samples,
                    K_lost,
                    reseed_ray_length,
                    bounds,
                    reseed_max_attempts
                )
                
                # Append to existing samples
                x = xp.vstack([x, new_samples])
                
                # Evaluate likelihood for new samples
                L_new = eval_L_only(func, new_samples, func_returns_grad)
                L = xp.concatenate([L, L_new])
                
                # Extend stuck_mask (new samples are not stuck)
                stuck_mask_new = xp.zeros(K_lost, dtype=bool)
                stuck_mask = xp.concatenate([stuck_mask, stuck_mask_new])
                
                # Extend inv_hessian_agg with default values for new samples
                if inv_hessian_agg is not None:
                    ihs_new = xp.ones(K_lost, dtype=inv_hessian_agg.dtype)
                    inv_hessian_agg = xp.concatenate([inv_hessian_agg, ihs_new])
                # Extend hessian_valid_agg with False for new samples (no history yet)
                if hessian_valid_agg is not None:
                    valid_new = xp.zeros(K_lost, dtype=bool)
                    hessian_valid_agg = xp.concatenate([hessian_valid_agg, valid_new])
                
                N = len(x)  # Update sample count
                total_reseeded += K_lost  # Track reseeded samples
                
                if verbose:
                    print(f"    Generated {K_lost} new samples, total now: {N}")
                
                if track_history:
                    xp_cpu = x.get() if hasattr(x, 'get') else x
                    history.append({'stage': 'post_reseed', 'oscillation': k+1, 'samples': xp_cpu.copy()})
            else:
                if verbose:
                    print(f"    WARNING: No unconverged samples to reseed from!")
        
        # Update N after potential reseeding
        N = len(x)
        
        # === HANDS LIKE CLOUDS PHASE (v2.2) ===
        # Use smoothed gradient to see global structure through local noise
        # Only applies to UNSTUCK samples (stuck stay frozen at peaks)
        if cloud_available and n_cloud > 0 and k < n_oscillations - 1:  # Not on last oscillation
            if verbose:
                print(f"Hands Like Clouds (雲手, {n_cloud} iters, σ={actual_cloud_sigma:.4f})...")
            
            n_unstuck = int(xp.sum(~stuck_mask))
            if n_unstuck > 0:
                for i in range(n_cloud):
                    # Compute smoothed gradient (sees global basin, not local bumps)
                    grad_smooth = whip.evaluate_gradient_smoothed(
                        x, func, sigma=actual_cloud_sigma, K=cloud_K,
                        func_returns_grad=func_returns_grad
                    )
                    
                    # Gradient ascent step (ascending toward global maximum)
                    # Only update UNSTUCK samples
                    x_new = x + cloud_step_size * grad_smooth
                    x = xp.where(stuck_mask[:, None], x, x_new)
                    
                    # Apply bounds
                    if bounds is not None:
                        x = xp.clip(x, bounds[:, 0], bounds[:, 1])
                
                if verbose:
                    print(f"  Moved {n_unstuck} samples through clouds ({int(xp.sum(stuck_mask))} frozen)")
                
                if track_history:
                    xp_cpu = x.get() if hasattr(x, 'get') else x
                    history.append({'stage': 'post_clouds', 'oscillation': k+1, 'samples': xp_cpu.copy()})
            else:
                if verbose:
                    print(f"  No unstuck samples for clouds")
        
        # === ANTI-CONVERGENCE PHASE ===
        if verbose:
            print(f"Anti-convergence phase ({n_anticonverge} iterations, μ={anti_momentum})...")
        
        # NOTE: stuck_mask NOT released here!
        # Stuck samples stay frozen, only unstuck samples explore
        
        # Run anti-convergence with MOMENTUM
        # Momentum allows samples to "overshoot" valleys and find new peaks
        v = xp.zeros_like(x)  # Initialize velocity
        
        for i in range(n_anticonverge):
            # v2.0: Use smart gradient (SingleWhip if available, or analytic if provided)
            grad = _compute_gradient_smart(func, x, whip=whip, batch_size=batch_size,
                                           func_returns_grad=func_returns_grad)
            
            # Momentum update (descending with inertia)
            # Only update velocity for UNSTUCK samples
            v_update = anti_momentum * v - anti_step_size * grad
            v = xp.where(stuck_mask[:, None], 0.0, v_update)
            
            # Only move UNSTUCK samples (stuck stay frozen at peaks)
            x_new = x + v
            x = xp.where(stuck_mask[:, None], x, x_new)
            
            # Apply bounds (and zero velocity at boundaries)
            if bounds is not None:
                x_clipped = xp.clip(x, bounds[:, 0], bounds[:, 1])
                # Zero velocity for components that hit boundary
                hit_boundary = (x != x_clipped)
                v = xp.where(hit_boundary, 0.0, v)
                x = x_clipped
        
        if verbose:
            n_moved = xp.sum(~stuck_mask)
            print(f"  Anti-convergence complete ({n_moved} samples moved, {xp.sum(stuck_mask)} frozen)")
        
        if track_history:
            xp_cpu = x.get() if hasattr(x, 'get') else x
            history.append({'stage': 'post_anticonverge', 'oscillation': k+1, 'samples': xp_cpu.copy()})
        
        # === CANNON THROUGH THE SKY ===
        # Rescue boundary-stuck samples after anti-convergence
        if cannon_through_sky and k < n_oscillations - 1:  # Not on last oscillation
            if verbose:
                print(f"Cannon Through the Sky (rescuing boundary-stuck samples)...")
            
            x, n_cannoned = _cannon_through_sky(
                x,
                bounds=bounds,
                cannon_ray_length=cannon_ray_length,
                cannon_max_attempts=cannon_max_attempts,
                cannon_wall_threshold=cannon_wall_threshold,
                verbose=verbose,
                xp=xp
            )
            
            # Re-evaluate likelihood for cannoned samples
            if n_cannoned > 0:
                L = eval_L_only(func, x, func_returns_grad)
            
            total_cannoned += n_cannoned
            
            if verbose:
                print(f"  Cannoned {n_cannoned} samples")
            
            if track_history:
                xp_cpu = x.get() if hasattr(x, 'get') else x
                history.append({'stage': 'post_cannon', 'oscillation': k+1, 'samples': xp_cpu.copy()})
    
    # Deduplicate before final convergence
    dedup_tol = stick_tolerance / 2
    L_temp = xp.zeros(len(x))  # Temporary L values for deduplication
    x, _, dedup_mask_pre = deduplicate_peaks_L_infinity(x, L_temp, tolerance=dedup_tol, keep_best=False, verbose=verbose, return_mask=True)
    if inv_hessian_agg is not None:
        inv_hessian_agg = inv_hessian_agg[dedup_mask_pre]
    if hessian_valid_agg is not None:
        hessian_valid_agg = hessian_valid_agg[dedup_mask_pre]
    
    # === FINAL CONVERGENCE ===
    if verbose:
        print("\n=== Final convergence ===")
    
    final_result = optimizer_func(
        func, x,
        maxiter=100,
        tol=stick_tolerance / 10,
        bounds=bounds,
        batch_size=batch_size,
        verbose=verbose,
        func_returns_grad=func_returns_grad,
        line_search=line_search,
        **optimizer_kwargs
    )
    
    peaks = final_result['x']
    L_peaks = final_result['L']
    converged_mask = final_result['converged']
    
    # Get inv_hessian_scale and history_count for width estimation (v1.8 feature)
    inv_hessian_scale = final_result.get('inv_hessian_scale', None)
    history_count_final = final_result.get('history_count', None)
    
    # Create validity mask for final convergence
    if inv_hessian_scale is not None:
        hessian_valid_final = history_count_final > 0 if history_count_final is not None else (inv_hessian_scale != 1.0)
        # Update aggregated validity (samples that moved in final convergence)
        if hessian_valid_agg is not None:
            hessian_valid_agg = hessian_valid_agg | hessian_valid_final
            # Update aggregated values where final has valid and wider
            inv_hessian_agg = xp.where(hessian_valid_final & (inv_hessian_scale > inv_hessian_agg), inv_hessian_scale, inv_hessian_agg)
        else:
            hessian_valid_agg = hessian_valid_final.copy() if hasattr(hessian_valid_final, 'copy') else xp.array(hessian_valid_final)
    
    if track_history:
        xp_cpu = peaks.get() if hasattr(peaks, 'get') else peaks
        history.append({'stage': 'final', 'oscillation': n_oscillations+1, 'samples': xp_cpu.copy()})
    
    # Final deduplication (track mask for inv_hessian_scale, with Hessian transfer)
    dedup_tol = stick_tolerance / 2
    if inv_hessian_agg is not None and hessian_valid_agg is not None:
        ihs_for_dedup = inv_hessian_agg.copy()
        valid_for_dedup = hessian_valid_agg.copy()
        peaks, L_peaks, dedup_mask1, ihs_dedup1 = deduplicate_peaks_L_infinity(
            peaks, L_peaks, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True,
            inv_hessian_scale=ihs_for_dedup, hessian_valid_mask=valid_for_dedup
        )
        inv_hessian_agg = ihs_dedup1
        hessian_valid_agg = valid_for_dedup[dedup_mask1]
    else:
        peaks, L_peaks, dedup_mask1 = deduplicate_peaks_L_infinity(
            peaks, L_peaks, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True
        )
    if inv_hessian_scale is not None:
        inv_hessian_scale = inv_hessian_scale[dedup_mask1]
    
    # Re-converge deduplicated peaks (dedup may have picked non-converged sample)
    if len(peaks) > 0:
        reconv_result = optimizer_func(
            func, peaks,
            maxiter=20,
            tol=stick_tolerance / 10,
            bounds=bounds,
            batch_size=batch_size,
            verbose=False,
            func_returns_grad=func_returns_grad,
            line_search=line_search
        )
        peaks = reconv_result['x']
        L_peaks = reconv_result['L']
    
    # === FINAL PEAK VALIDATION ===
    # A real peak has:
    # a) Gradient ≈ 0 (stationary point)
    # b) Above noise floor (not numerical garbage)
    # 
    # Note: We do NOT filter by likelihood ratio - a low-weight mode is still a valid peak
    # The likelihood VALUE is irrelevant for peak detection; it only affects evidence weight
    #
    xp = cp.get_array_module(peaks) if GPU_AVAILABLE else np
    D = peaks.shape[1] if len(peaks) > 0 else 1
    
    # Noise floor: dimension-dependent absolute threshold
    # In D dimensions, even random points have log_L ~ -D * constant
    # Use -10 * D as a generous floor (anything below is garbage)
    noise_floor = -10.0 * D
    
    if len(peaks) > 0:
        above_noise = L_peaks >= noise_floor
        n_below_noise = int(xp.sum(~above_noise))
        
        if verbose and n_below_noise > 0:
            print(f"\n=== Noise floor filter (L >= {noise_floor:.1f}) ===")
            print(f"  Removed {n_below_noise} peaks below noise floor, kept {int(xp.sum(above_noise))}")
        
        peaks = peaks[above_noise]
        L_peaks = L_peaks[above_noise]
        if inv_hessian_scale is not None:
            inv_hessian_scale = inv_hessian_scale[above_noise]
        if inv_hessian_agg is not None:
            inv_hessian_agg = inv_hessian_agg[above_noise]
        if hessian_valid_agg is not None:
            hessian_valid_agg = hessian_valid_agg[above_noise]
    
    # Second deduplication pass (duplicates may have been separated by low-L peaks)
    if len(peaks) > 0:
        if inv_hessian_agg is not None and hessian_valid_agg is not None:
            ihs_for_dedup2 = inv_hessian_agg.copy()
            valid_for_dedup2 = hessian_valid_agg.copy()
            peaks, L_peaks, dedup_mask2, ihs_dedup2 = deduplicate_peaks_L_infinity(
                peaks, L_peaks, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True,
                inv_hessian_scale=ihs_for_dedup2, hessian_valid_mask=valid_for_dedup2
            )
            inv_hessian_agg = ihs_dedup2
            hessian_valid_agg = valid_for_dedup2[dedup_mask2]
        else:
            peaks, L_peaks, dedup_mask2 = deduplicate_peaks_L_infinity(
                peaks, L_peaks, tolerance=dedup_tol, keep_best=True, verbose=verbose, return_mask=True
            )
        if inv_hessian_scale is not None:
            inv_hessian_scale = inv_hessian_scale[dedup_mask2]
    
    # === WIDTH ESTIMATION ===
    # v1.7: Use inv_hessian_scale from L-BFGS (gamma = s^T y / y^T y ≈ σ²)
    # v1.8: Use hessian_valid_agg to determine if we have real Hessian info
    # Prefer aggregated values from oscillation loop over final (which may be default 1.0)
    xp = cp.get_array_module(peaks) if GPU_AVAILABLE else np
    
    # Choose best source: aggregated (from oscillations) or final (from final convergence)
    ihs_final = inv_hessian_scale  # From final convergence
    ihs_agg = inv_hessian_agg      # From oscillation loop
    
    # Use aggregated if available and has any valid values (based on hessian_valid_agg)
    # Fall back to final if aggregated has no valid values
    if ihs_agg is not None and len(ihs_agg) > 0 and hessian_valid_agg is not None and xp.any(hessian_valid_agg):
        ihs_to_use = ihs_agg
        source = "aggregated"
    elif ihs_final is not None and len(ihs_final) > 0 and xp.any(ihs_final != 1.0):
        # Fallback to final (use old check for backward compat with non-lbfgs optimizers)
        ihs_to_use = ihs_final
        source = "final"
    else:
        ihs_to_use = None
        source = None
    
    if estimate_widths and ihs_to_use is not None:
        # Use L-BFGS inverse Hessian scale for width estimation
        width_scalar = xp.sqrt(xp.abs(ihs_to_use))  # abs() for safety
        widths = xp.stack([width_scalar, width_scalar], axis=1)  # [longest, shortest] = same for isotropic
        if verbose:
            mean_width = float(xp.mean(width_scalar))
            print(f"\n=== Peak widths from L-BFGS Hessian ({source}, mean σ ≈ {mean_width:.4f}) ===")
    elif estimate_widths and len(peaks) > 0:
        # Fallback: estimate widths via probing
        if verbose:
            print("\n=== Estimating peak widths (fallback to probing) ===")
        widths = estimate_peak_width(
            func, peaks,
            method=width_method,
            bounds=bounds,
            whip=whip,
            batch_size=batch_size
        )
    else:
        # Skip width estimation
        widths = xp.zeros((len(peaks), 2), dtype=peaks.dtype if len(peaks) > 0 else xp.float64)
        if verbose:
            print("\n=== Skipping peak width estimation ===")
    
    # === ADAPTIVE FINAL DEDUPLICATION ===
    # Now that we have width estimates, use 1/100 of minimum width as tolerance
    # This catches duplicates that slipped through with the tight fixed tolerance
    if estimate_widths and len(peaks) > 1:
        min_width = float(xp.min(widths[widths > 0])) if xp.any(widths > 0) else 0
        if min_width > 0:
            adaptive_tol = min_width / 100
            n_before = len(peaks)
            
            if inv_hessian_agg is not None and hessian_valid_agg is not None:
                ihs_for_adaptive = inv_hessian_agg.copy()
                valid_for_adaptive = hessian_valid_agg.copy()
                peaks, L_peaks, adaptive_mask, ihs_adaptive = deduplicate_peaks_L_infinity(
                    peaks, L_peaks, tolerance=adaptive_tol, keep_best=True, verbose=False, return_mask=True,
                    inv_hessian_scale=ihs_for_adaptive, hessian_valid_mask=valid_for_adaptive
                )
                inv_hessian_agg = ihs_adaptive
                hessian_valid_agg = valid_for_adaptive[adaptive_mask]
            else:
                peaks, L_peaks, adaptive_mask = deduplicate_peaks_L_infinity(
                    peaks, L_peaks, tolerance=adaptive_tol, keep_best=True, verbose=False, return_mask=True
                )
            
            # Update widths to match surviving peaks
            widths = widths[adaptive_mask]
            
            if verbose and n_before != len(peaks):
                print(f"\n=== Adaptive dedup (tol={adaptive_tol:.2e} = min_width/100) ===")
                print(f"  {n_before} → {len(peaks)} peaks")
            
            # Re-converge after adaptive dedup (may have picked non-converged samples)
            if len(peaks) > 0:
                reconv_result = optimizer_func(
                    func, peaks,
                    maxiter=50,
                    tol=stick_tolerance / 10,
                    bounds=bounds,
                    batch_size=batch_size,
                    verbose=False,
                    func_returns_grad=func_returns_grad,
                    line_search=line_search
                )
                peaks = reconv_result['x']
                L_peaks = reconv_result['L']
    
    # === FINAL GRADIENT FILTER ===
    # Remove all samples that don't have |grad L|_∞ < tolerance
    # Use dimension-adaptive tolerance: numerical precision degrades in high-D
    grad_norm_inf = xp.array([])  # Initialize for edge case
    if len(peaks) > 0:
        if func_returns_grad:
            _, final_grad = func(peaks)
        else:
            final_grad = compute_gradient_batch(func, peaks, batch_size=batch_size)
        grad_norm_inf = xp.max(xp.abs(final_grad), axis=1)
        
        # Dimension-adaptive: 1e-6 at 8D, scales with sqrt(D/8)
        D = peaks.shape[1]
        grad_tol = stick_tolerance * max(1.0, np.sqrt(D / 8.0))
        converged_final = grad_norm_inf < grad_tol
        
        n_before_grad_filter = len(peaks)
        peaks = peaks[converged_final]
        L_peaks = L_peaks[converged_final]
        grad_norm_inf = grad_norm_inf[converged_final]
        if widths is not None and len(widths) == n_before_grad_filter:
            widths = widths[converged_final]
        if inv_hessian_agg is not None and len(inv_hessian_agg) == n_before_grad_filter:
            inv_hessian_agg = inv_hessian_agg[converged_final]
        if hessian_valid_agg is not None and len(hessian_valid_agg) == n_before_grad_filter:
            hessian_valid_agg = hessian_valid_agg[converged_final]
        
        n_removed = n_before_grad_filter - len(peaks)
        if verbose and n_removed > 0:
            print(f"\n=== Final gradient filter (|grad|_∞ < {grad_tol:.0e}) ===")
            print(f"  Removed {n_removed} unconverged samples, kept {len(peaks)}")
    
    # === FINAL DEDUPLICATION ===
    # Run one more dedup pass after gradient filter
    if len(peaks) > 1:
        n_before_final_dedup = len(peaks)
        dedup_tol_final = stick_tolerance / 2
        
        if inv_hessian_agg is not None and hessian_valid_agg is not None:
            ihs_for_final = inv_hessian_agg.copy()
            valid_for_final = hessian_valid_agg.copy()
            peaks, L_peaks, final_dedup_mask, ihs_final = deduplicate_peaks_L_infinity(
                peaks, L_peaks, tolerance=dedup_tol_final, keep_best=True, verbose=False, return_mask=True,
                inv_hessian_scale=ihs_for_final, hessian_valid_mask=valid_for_final
            )
            inv_hessian_agg = ihs_final
            hessian_valid_agg = valid_for_final[final_dedup_mask]
            if widths is not None and len(widths) == n_before_final_dedup:
                widths = widths[final_dedup_mask]
            grad_norm_inf = grad_norm_inf[final_dedup_mask]
        else:
            peaks, L_peaks, final_dedup_mask = deduplicate_peaks_L_infinity(
                peaks, L_peaks, tolerance=dedup_tol_final, keep_best=True, verbose=False, return_mask=True
            )
            if widths is not None and len(widths) == n_before_final_dedup:
                widths = widths[final_dedup_mask]
            grad_norm_inf = grad_norm_inf[final_dedup_mask]
        
        if verbose and n_before_final_dedup != len(peaks):
            print(f"\n=== Final deduplication (tol={dedup_tol_final:.0e}) ===")
            print(f"  {n_before_final_dedup} → {len(peaks)} peaks")
    
    # Update converged_mask to match final peaks (all are converged now)
    converged_mask = xp.ones(len(peaks), dtype=bool)
    
    # v2.4: Bank final samples
    if bank_samples:
        sample_bank.add_samples(peaks, L_peaks, 'final', n_oscillations+1, converged_mask)
    
    result = {
        'peaks': peaks,
        'L_peaks': L_peaks,
        'widths': widths,
        'converged_mask': converged_mask,
        'n_oscillations_completed': n_oscillations,
        'grad_norm': grad_norm_inf
    }
    
    # Add reseeding statistics if Repulse Monkey was used
    if reseed_strategy is not None:
        result['n_reseeded'] = total_reseeded
    
    # Add cannoning statistics if Cannon Through the Sky was used
    if cannon_through_sky:
        result['n_cannoned'] = total_cannoned
    
    # Add history if tracking was enabled
    if track_history:
        result['history'] = history
    
    # v2.2: Add clouding info
    if cloud_available:
        result['cloud_sigma'] = actual_cloud_sigma
    
    # v2.4: Add sample bank
    if bank_samples and sample_bank is not None:
        result['sample_bank'] = sample_bank.get_bank_dict()
        result['sample_bank_memory_mb'] = sample_bank.memory_mb()
    
    return result


# ============================================================================
# WALL ESCAPE (Future Implementation Stub)
# ============================================================================

def escape_walls(
    func: Callable,
    x: Union[np.ndarray, 'cp.ndarray'],
    bounds: Union[np.ndarray, 'cp.ndarray'],
    wall_threshold: float = 1e-3,
    escape_strategy: str = 'uniform',
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Detect samples stuck on prior walls and resample them.
    
    NOTE: Wall resampling is implemented via SkyCannon module.
    This function is a legacy stub - use SkyCannon for wall escape.
    
    Args:
        func: Likelihood function
        x: [N, D] current positions
        bounds: [D, 2] domain bounds
        wall_threshold: Distance threshold to be considered "on wall"
        escape_strategy: Strategy for resampling ('uniform', 'likelihood_weighted', etc.)
    
    Returns:
        Dictionary with:
            'x': [N, D] positions (wall-stuck samples resampled)
            'was_stuck': [N] bool which samples were on walls
            'n_escaped': int number of samples resampled
    """
    warnings.warn(
        "escape_walls() is a legacy stub - use SkyCannon module for wall resampling."
    )
    
    xp = cp.get_array_module(x) if GPU_AVAILABLE else np
    
    return {
        'x': x,
        'was_stuck': xp.zeros(len(x), dtype=bool),
        'n_escaped': 0
    }


# ============================================================================
# MEMORY MANAGEMENT & BATCHING
# ============================================================================

def estimate_safe_batch_size(
    N: int,
    D: int,
    memory_limit: Optional[float] = None,
    gpu_capability: Optional[GPUCapability] = None
) -> int:
    """
    Estimate safe batch size based on GPU memory and compute capability.
    
    Args:
        N: Number of samples
        D: Dimensionality
        memory_limit: Max memory to use (GB), None = auto-detect
        gpu_capability: GPU capability object, None = auto-detect
    
    Returns:
        Safe batch size
    """
    if gpu_capability is None:
        gpu_capability = GPUCapability()
    
    # Memory-based limit
    if memory_limit is None:
        if GPU_AVAILABLE:
            available_memory = gpu_capability.specs['memory_gb'] * 0.7 * 1e9
        else:
            try:
                import psutil
                available_memory = psutil.virtual_memory().available * 0.5
            except:
                available_memory = 8e9  # Default 8GB
    else:
        available_memory = memory_limit * 1e9
    
    # Memory per sample (rough estimate)
    m = 10  # L-BFGS history
    bytes_per_sample = D * 8 * (2 + 2*m + 4)  # Position, grad, history, temps
    bytes_per_sample += 2 * D * D * 8  # Gradient computation
    
    memory_limited = int(available_memory / bytes_per_sample)
    
    # Compute-based limit
    if GPU_AVAILABLE:
        cuda_cores = gpu_capability.specs['cuda_cores_approx']
        compute_limited = max(100, cuda_cores // 10)
    else:
        compute_limited = 100
    
    # Take minimum
    safe_batch = min(memory_limited, compute_limited, N)
    safe_batch = max(1, safe_batch)
    
    return safe_batch


# ============================================================================
# MAIN API FUNCTION
# ============================================================================

def optimize_batch(
    func: Callable,
    x0: Union[np.ndarray, 'cp.ndarray'],
    method: str = 'lbfgs',
    sticky_hands_mode: bool = False,
    batch_strategy: str = 'auto',
    batch_size: Optional[int] = None,
    memory_limit: Optional[float] = None,
    gpu_capability: Optional[GPUCapability] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
    """
    Main API function for ChiSao optimization.
    
    Args:
        func: Likelihood function(x) -> [N]
        x0: [N, D] initial positions
        method: 'lbfgs' or 'gradient_ascent'
        sticky_hands_mode: Use sticky hands heuristic?
        batch_strategy: 'auto', 'single', 'fixed', 'adaptive'
        batch_size: Fixed batch size (for 'fixed' strategy)
        memory_limit: Max GPU memory (GB)
        gpu_capability: GPU capability object
        verbose: Print info
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimization results (format depends on sticky_hands_mode)
    """
    # Detect GPU
    if gpu_capability is None:
        gpu_capability = GPUCapability()
    
    if verbose:
        print(f"ChiSao Optimization")
        print(f"  GPU: {gpu_capability.specs['name']}")
        print(f"  Tier: {gpu_capability.tier}")
        print(f"  Memory: {gpu_capability.specs.get('memory_gb', 'N/A')}")
        print(f"  Method: {method}")
        print(f"  Sticky Hands: {sticky_hands_mode}")
    
    # Determine batch size
    if batch_strategy == 'auto':
        if batch_size is None:
            batch_size = estimate_safe_batch_size(
                x0.shape[0], x0.shape[1], memory_limit, gpu_capability
            )
    elif batch_strategy == 'single':
        batch_size = x0.shape[0]
    elif batch_strategy == 'fixed':
        assert batch_size is not None, "Must specify batch_size for 'fixed' strategy"
    
    if verbose:
        print(f"  Batch size: {batch_size}")
    
    # Run optimization
    if sticky_hands_mode:
        return sticky_hands(func, x0, method=method, batch_size=batch_size, 
                           verbose=verbose, **kwargs)
    else:
        if method == 'lbfgs':
            return lbfgs_batch(func, x0, batch_size=batch_size, 
                              verbose=verbose, **kwargs)
        elif method == 'gradient_ascent':
            return gradient_ascent_batch(func, x0, batch_size=batch_size,
                                        verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


# ============================================================================
# MODULE INFO
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get current GPU information."""
    gpu_cap = GPUCapability()
    return {
        'gpu_available': GPU_AVAILABLE,
        'tier': gpu_cap.tier,
        'specs': gpu_cap.specs,
        'optimal_config': gpu_cap.optimal_config
    }


__version__ = '3.2.0'
__author__ = 'Ira Wolfson'
__all__ = [
    'lbfgs_batch',
    'gradient_ascent_batch',
    'sticky_hands',
    'estimate_peak_width',
    'escape_walls',
    'optimize_batch',
    'get_gpu_info',
    'GPUCapability',
    'compute_gradient_batch',
    'estimate_safe_batch_size',
    # v2.0 additions
    '_compute_gradient_smart',
    '_compute_hessian_diagonal_smart',
]
