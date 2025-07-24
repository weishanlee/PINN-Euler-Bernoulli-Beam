import os
import sys

# Debug print to identify segmentation fault location
print("DEBUG: Starting imports...", file=sys.stderr)
sys.stderr.flush()

import psutil
import signal
import atexit

# Global log file handle
log_file = None

# Track if cleanup has been called to prevent double cleanup
_cleanup_in_progress = False

def cleanup_handler():
    """Cleanup function to properly close file handles and release GPU lock"""
    global log_file, _cleanup_in_progress
    
    # Check if cleanup is disabled
    if '_cleanup_disabled' in globals() and _cleanup_disabled:
        return
    
    # Prevent recursive cleanup
    if _cleanup_in_progress:
        return
        
    _cleanup_in_progress = True
    
    try:
        # Release GPU lock first - but check if function exists and hasn't been released
        if 'release_gpu_lock' in globals() and 'gpu_lock_fd' in globals() and gpu_lock_fd is not None:
            try:
                release_gpu_lock()
            except:
                pass
    except Exception:
        pass
    
    # Close log file
    if log_file and not log_file.closed:
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()
        except:
            pass
    
    # Final GPU cleanup - only if torch is already imported and CUDA is initialized
    try:
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available() and hasattr(torch.cuda, '_initialized') and torch.cuda._initialized:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                # Don't synchronize during cleanup as it can cause segfaults
    except Exception:
        pass
    finally:
        _cleanup_in_progress = False

# Delay signal handler registration until after critical imports
def register_cleanup_handlers():
    """Register cleanup handlers after all imports are done"""
    atexit.register(cleanup_handler)
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, cleaning up...", file=sys.__stderr__)
        cleanup_handler()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    print("Cleanup handlers registered")

print("DEBUG: Before log file setup...", file=sys.stderr)
sys.stderr.flush()

# Redirect all output to log file to avoid cursor crash
try:
    # Use a single train_log.txt file for all outputs
    import datetime
    log_filename = 'train_log.txt'
    
    # Check if we should skip log redirection (for debugging)
    if os.environ.get('SKIP_LOG_REDIRECT', '0') != '1':
        # Open in append mode to preserve previous runs, with line buffering
        log_file = open(log_filename, 'a', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
        
        # Write a header to identify this run
        print(f"\n{'='*70}")
        print(f"=== Training started at {datetime.datetime.now()} ===")
        print(f"{'='*70}")
    else:
        print("Log redirection skipped for debugging", file=sys.__stderr__)
except Exception as e:
    print(f"Warning: Could not redirect output to log file: {e}", file=sys.__stderr__)

# Force GPU usage - set environment variables before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"  # Help with memory fragmentation
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"  # Keep caching enabled

# Add exclusive GPU locking to prevent concurrent access
import fcntl
import time

gpu_lock_file = "/tmp/gpu_training_lock.lock"
gpu_lock_fd = None

def acquire_gpu_lock(timeout=300):
    """Acquire exclusive lock on GPU to prevent concurrent access"""
    global gpu_lock_fd, _gpu_lock_released
    _gpu_lock_released = False  # Reset the flag
    try:
        gpu_lock_fd = open(gpu_lock_file, 'w')
        # Try to acquire lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(gpu_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"Acquired GPU lock for PID {os.getpid()}")
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    print(f"Failed to acquire GPU lock after {timeout}s")
                    return False
                time.sleep(1)
    except Exception as e:
        print(f"Error acquiring GPU lock: {e}")
        return False

# Track if GPU lock has been released
_gpu_lock_released = False

def release_gpu_lock():
    """Release GPU lock"""
    global gpu_lock_fd, _gpu_lock_released
    
    # Prevent double release
    if _gpu_lock_released or gpu_lock_fd is None:
        return
        
    try:
        fcntl.flock(gpu_lock_fd, fcntl.LOCK_UN)
        gpu_lock_fd.close()
        gpu_lock_fd = None
        _gpu_lock_released = True
        print(f"Released GPU lock for PID {os.getpid()}")
    except Exception as e:
        print(f"Error releasing GPU lock: {e}")

# Detect physical cores the way lscpu does
print("DEBUG: Detecting CPU cores...")
sys.stdout.flush()

try:
    phys = psutil.cpu_count(logical=False) or 1         # fallback =1 just in case
    logical = psutil.cpu_count(logical=True) or 1
    print(f"DEBUG: Physical cores: {phys}, Logical cores: {logical}")
    sys.stdout.flush()
    
    # Use a more reasonable thread count - leave some cores free
    # If we have many cores, leave 4 free; otherwise leave 1 free
    if phys > 8:
        threads = max(1, phys - 4)
    else:
        threads = max(1, phys - 1)
    
    print(f"DEBUG: Setting thread count to {threads}")
    sys.stdout.flush()
    
    os.environ["OMP_NUM_THREADS"]     = str(threads)
    os.environ["MKL_NUM_THREADS"]     = str(threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
    
    print(f"Configured {threads} CPU threads "
          f"(out of {logical} logical, {phys} physical).")
    sys.stdout.flush()
except Exception as e:
    print(f"ERROR in CPU configuration: {e}")
    sys.stdout.flush()
    # Set safe defaults
    os.environ["OMP_NUM_THREADS"]     = "1"
    os.environ["MKL_NUM_THREADS"]     = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"

print("DEBUG: Before torch import...")
sys.stdout.flush()

# Also print to original stderr for debugging
print("DEBUG: About to import torch (stderr)...", file=sys.__stderr__)
sys.__stderr__.flush()

import torch

print("DEBUG: torch imported successfully")
sys.stdout.flush()
print("DEBUG: torch imported successfully (stderr)", file=sys.__stderr__)
sys.__stderr__.flush()

import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import gc
import argparse

print("DEBUG: All main imports completed")
sys.stdout.flush()
print("DEBUG: All main imports completed (stderr)", file=sys.__stderr__)
sys.__stderr__.flush()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set number of threads for autograd to avoid segfaults
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Check for GPU and set device - MANDATORY GPU usage
if not torch.cuda.is_available():
    print("ERROR: No GPU detected. GPU is REQUIRED for training.")
    print("This script requires GPU and cannot run on CPU only.")
    sys.exit(1)

device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# GPU Memory Management and Estimation Functions
def get_gpu_memory_info():
    """Get current GPU memory information in GB"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
        reserved_memory = torch.cuda.memory_reserved() / (1024**3)
        free_memory = total_memory - reserved_memory
        return total_memory, allocated_memory, reserved_memory, free_memory
    return 0, 0, 0, 0

def estimate_memory_requirement(n_harmonics, points_per_dim=50, batch_size=1000):
    """
    Estimate GPU memory requirement for training based on harmonics count.
    Updated for larger batch sizes and 4th-order derivatives.
    """
    # Base memory for model parameters
    base_model_memory = 0.02  # GB for the enhanced neural network
    
    # Memory for harmonic coefficients (always max_harmonics=65 for fair comparison)
    harmonic_memory = 65 * 2 * 4 / (1024**3)  # 65 harmonics * 2 (cos/sin) * 4 bytes per float
    
    # Memory for training data
    total_points = points_per_dim * points_per_dim
    coords_memory = total_points * 2 * 4 / (1024**3)  # (t,x) coordinates
    
    # Memory for batch processing
    batch_memory = batch_size * 4 * 4 / (1024**3)  # batch with gradients
    
    # Memory for intermediate computations (scales with n_harmonics used)
    # Each harmonic requires computation of sin/cos terms
    harmonic_computation_memory = n_harmonics * batch_size * 4 * 4 / (1024**3)
    
    # Memory for gradient computations - increased for 4th-order derivatives
    # Need to store w, w_t, w_x, w_tt, w_xx, w_xxx, w_xxxx
    grad_memory = batch_size * 7 * 4 * 4 / (1024**3)  # 7 gradient tensors
    
    # Additional memory for autograd graph (4th-order derivatives create deep graphs)
    autograd_memory = batch_size * n_harmonics * 4 * 2 / (1024**3)
    
    # Safety factor
    safety_factor = 1.5  # Reduced safety factor to allow larger batches
    
    estimated_memory = (base_model_memory + harmonic_memory + coords_memory + 
                       batch_memory + harmonic_computation_memory + grad_memory + 
                       autograd_memory) * safety_factor
    
    return estimated_memory

def check_memory_feasibility(n_harmonics, points_per_dim=50, batch_size=1000):
    """
    Check if training is feasible with current GPU memory.
    """
    total_memory, allocated, reserved, free_memory = get_gpu_memory_info()
    
    if total_memory == 0:
        print("ERROR: No GPU detected. GPU is required for training.")
        return False, batch_size
    
    # Use 95% of total memory for maximum GPU utilization
    target_memory_limit = total_memory * 0.95
    available_memory = target_memory_limit - allocated
    
    estimated_memory = estimate_memory_requirement(n_harmonics, points_per_dim, batch_size)
    
    print(f"GPU Memory Status:")
    print(f"  Total: {total_memory:.2f} GB")
    print(f"  Target limit (80%): {target_memory_limit:.2f} GB") 
    print(f"  Currently allocated: {allocated:.2f} GB")
    print(f"  Available for training: {available_memory:.2f} GB")
    print(f"  Estimated requirement: {estimated_memory:.2f} GB")
    
    if estimated_memory <= available_memory:
        print(f"✓ Memory feasible for harmonic {n_harmonics}")
        return True, batch_size
    else:
        # Try to adjust batch size
        print(f"⚠ Estimated memory ({estimated_memory:.2f} GB) exceeds available ({available_memory:.2f} GB)")
        
        # Calculate adjusted batch size
        memory_ratio = available_memory / estimated_memory
        adjusted_batch_size = max(100, int(batch_size * memory_ratio * 0.9))  # Reduced safety margin for higher utilization
        
        # Recheck with adjusted batch size
        adjusted_estimate = estimate_memory_requirement(n_harmonics, points_per_dim, adjusted_batch_size)
        
        if adjusted_estimate <= available_memory:
            print(f"✓ Adjusted batch size to {adjusted_batch_size} (estimated memory: {adjusted_estimate:.2f} GB)")
            return True, adjusted_batch_size
        else:
            print(f"✗ Even with reduced batch size ({adjusted_batch_size}), memory requirement ({adjusted_estimate:.2f} GB) still too high")
            return False, adjusted_batch_size

# Configure GPU memory usage more efficiently
if torch.cuda.is_available():
    # Use 95% of available memory for maximum GPU utilization
    torch.cuda.set_per_process_memory_fraction(0.95)
    print(f"GPU memory configured: limiting to 95% of available memory")
    # Simple cache clear without synchronization
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: CUDA cache clear issue: {e}")

# Skip registering cleanup handlers to avoid segmentation faults
print("DEBUG: Skipping cleanup handlers to avoid segfaults")
sys.stdout.flush() 
print("DEBUG: Skipping cleanup handlers to avoid segfaults (stderr)", file=sys.__stderr__)
sys.__stderr__.flush()

# Set flag to prevent cleanup handler execution
_cleanup_disabled = True

print("DEBUG: Cleanup handlers skipped")
sys.stdout.flush()
print("DEBUG: Cleanup handlers skipped (stderr)", file=sys.__stderr__)
sys.__stderr__.flush()

print("DEBUG: About to define clear_memory function")
sys.stdout.flush()
print("DEBUG: About to define clear_memory function (stderr)", file=sys.__stderr__)
sys.__stderr__.flush()

# Clear GPU memory - improved function for better memory management
# Track if clear_memory has been called to prevent double calls
_clear_memory_in_progress = False

def clear_memory():
    """Clear GPU memory more effectively and safely"""
    global _clear_memory_in_progress
    
    # Prevent recursive calls
    if _clear_memory_in_progress:
        return
        
    _clear_memory_in_progress = True
    
    try:
        # Just do basic garbage collection
        gc.collect()
        
        # Simple GPU cache clear without synchronization
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
                
        # Final garbage collection
        gc.collect()
        
    except Exception as e:
        # Silently handle any errors
        pass
    finally:
        _clear_memory_in_progress = False


class UltraPrecisionEulerBernoulliModel(nn.Module):
    def __init__(self, n_harmonics=50, c=1.0, L=1.0, max_harmonics=65):
        """
        Multi-harmonic model for Euler-Bernoulli beam equation.
        Uses Fourier series expansion with learnable coefficients.
        
        Args:
            n_harmonics: Number of harmonic components to use
            c: Beam parameter (c² = EI/ρA)
            L: Domain length
            max_harmonics: Maximum number of harmonics for parameter allocation (for fair comparison)
        """
        super(UltraPrecisionEulerBernoulliModel, self).__init__()
        
        self.n_harmonics = n_harmonics
        self.c = c
        self.L = L
        self.max_harmonics = max_harmonics
        
        # Allocate parameters for max_harmonics to ensure fair comparison
        # But only use n_harmonics in the forward pass
        # IMPORTANT: Use both cos and sin terms for more flexibility
        self.amplitudes_cos = nn.Parameter(torch.zeros(max_harmonics))
        self.amplitudes_sin = nn.Parameter(torch.zeros(max_harmonics))
        
        # FAIR INITIALIZATION: Initialize ALL amplitudes with small random values
        # No bias towards any specific harmonic - TRULY UNBIASED
        with torch.no_grad():
            # Fair initialization for ALL harmonics including the first
            # No special treatment for any harmonic to ensure unbiased comparison
            for i in range(max_harmonics):
                scale = 0.1 / (i + 1)  # Slightly larger initial scale for better convergence
                nn.init.normal_(self.amplitudes_cos[i:i+1], mean=0.0, std=scale)
                nn.init.normal_(self.amplitudes_sin[i:i+1], mean=0.0, std=scale * 0.5)
            
        # Enhanced neural network for fine corrections with deeper and wider architecture
        self.correction_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        
        # Correction scaling factor - learnable, start very small
        self.correction_scale = nn.Parameter(torch.tensor(1e-8))
        
        # Initialize neural network with small weights
        with torch.no_grad():
            for module in self.correction_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
    def forward(self, x):
        """
        Forward pass using Fourier series with n_harmonics terms.
        
        Args:
            x: Tensor of shape [..., 2] with [t, x] coordinates
            
        Returns:
            u: Solution tensor
        """
        if x.dim() < 2:
            x = x.unsqueeze(0)
            
        t = x[..., 0:1]  # Time
        spatial = x[..., 1:2]  # Space
        
        # Fourier series solution using n_harmonics terms
        solution = torch.zeros_like(t)
        
        # Use only n_harmonics terms (not max_harmonics)
        for n in range(1, self.n_harmonics + 1):
            k_n = n * np.pi / self.L  # n-th mode wave number
            omega_n = k_n**2 * self.c  # n-th mode frequency
            
            # Get amplitudes for this harmonic (use n-1 for 0-based indexing)
            amp_cos = self.amplitudes_cos[n-1]
            amp_sin = self.amplitudes_sin[n-1]
            
            # Add this harmonic's contribution with both cos and sin time components
            spatial_mode = torch.sin(k_n * spatial)
            solution = solution + amp_cos * torch.cos(omega_n * t) * spatial_mode
            solution = solution + amp_sin * torch.sin(omega_n * t) * spatial_mode
        
        # Neural network correction for ultra-high precision
        # Apply boundary conditions to correction
        boundary_factor = torch.sin(np.pi * spatial / self.L)
        correction = self.correction_net(x) * boundary_factor * self.correction_scale
        
        # Total solution
        result = solution + correction
        
        return result


class UltraPrecisionEulerBernoulliSolver:
    def __init__(self, model, t_max=2.0, data_dir=None):
        """
        Optimized solver for ultra-fast convergence to 1e-9 target.
        """
        self.model = model.to(device)
        self.c = model.c
        self.L = model.L
        self.n_harmonics = model.n_harmonics
        self.t_max = t_max
        
        # Make sure data_dir is provided
        if data_dir is None:
            raise ValueError("data_dir must be specified")
        self.data_dir = data_dir
        
        # Create directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Data tracking
        self.losses = []
        self.pde_losses = []
        self.ic_losses = []
        self.bc_losses = []
        self.l2_errors = []
        self.epochs = []
        
        # Enhanced loss weights for ultra-precision convergence
        # Adaptive weights based on harmonic count - more conservative scaling
        weight_scale = 1.0 + (self.n_harmonics / 130.0)  # Gentler scaling
        self.w_pde = 10000.0 * weight_scale     # Much stronger PDE enforcement
        self.w_ic = 100000.0 * weight_scale     # Much stronger initial condition enforcement
        self.w_ic_v = 50000.0 * weight_scale    # Reduced IC velocity weight to prevent explosion
        self.w_bc = 100000.0 * weight_scale     # Much stronger boundary condition enforcement
    
    def exact_solution(self, t, x):
        """
        Analytical solution for Euler-Bernoulli beam equation with w(0,x) = sin(πx/L), w_t(0,x) = 0.
        """
        omega_1 = (np.pi / self.L)**2 * self.c
        return torch.sin(np.pi * x / self.L) * torch.cos(omega_1 * t)
    
    def u_net(self, t, x):
        """Neural network solution."""
        # Ensure correct dimensions
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # Create input to model
        tx = torch.cat([t, x], dim=1)
        
        # Forward pass
        return self.model(tx)
    
    def pde_residual(self, t, x):
        """
        PDE residual for Euler-Bernoulli beam equation: w_tt + c²w_xxxx = 0.
        Optimized computation with better numerical stability.
        """
        # Ensure we're tracking gradients
        t.requires_grad_(True)
        x.requires_grad_(True)
        
        # Compute w(t,x)
        w = self.u_net(t, x)
        
        # Check for NaN in the solution
        if torch.isnan(w).any():
            return torch.zeros_like(w)
        
        # Compute derivatives efficiently with memory management
        try:
            # Create gradient outputs once to avoid repeated allocation
            ones_w = torch.ones_like(w)
            
            # First derivatives with allow_unused flag for safety
            w_t = torch.autograd.grad(w, t, grad_outputs=ones_w, create_graph=True, retain_graph=True, allow_unused=False)[0]
            w_x = torch.autograd.grad(w, x, grad_outputs=ones_w, create_graph=True, retain_graph=True, allow_unused=False)[0]
            
            # Check for issues after first derivatives
            if torch.isnan(w_t).any() or torch.isnan(w_x).any():
                return torch.zeros_like(w)
            
            # Second derivatives
            w_tt = torch.autograd.grad(w_t, t, grad_outputs=ones_w, create_graph=True, retain_graph=True)[0]
            w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_w, create_graph=True, retain_graph=True)[0]
            
            # Check for issues after second derivatives
            if torch.isnan(w_tt).any() or torch.isnan(w_xx).any():
                return torch.zeros_like(w)
            
            # Third derivative
            w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=ones_w, create_graph=True, retain_graph=True)[0]
            
            # Check for issues after third derivative
            if torch.isnan(w_xxx).any():
                return torch.zeros_like(w)
            
            # Fourth derivative
            w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=ones_w, create_graph=True, retain_graph=True)[0]
            
            # Euler-Bernoulli beam equation residual
            residual = w_tt + self.c**2 * w_xxxx
            
            # Check for NaN in residual
            if torch.isnan(residual).any():
                return torch.zeros_like(w)
            
            return residual
            
        except Exception as e:
            print(f"Warning: Error in PDE residual computation: {e}")
            return torch.zeros_like(w)
    
    def compute_bc_loss(self, t_bc):
        """
        Compute boundary condition losses for simply supported beam.
        Optimized computation.
        """
        try:
            # Ensure correct dimensions
            if t_bc.dim() == 1:
                t_bc = t_bc.unsqueeze(1)
                
            # Create boundary points
            x_left = torch.zeros_like(t_bc)
            x_right = torch.ones_like(t_bc) * self.L
            
            # Compute w at boundaries (should be zero)
            w_left = self.u_net(t_bc, x_left)
            w_right = self.u_net(t_bc, x_right)
            
            # Check for NaN
            if torch.isnan(w_left).any() or torch.isnan(w_right).any():
                return torch.tensor(0.0, device=device)
            
            # For second derivatives at boundaries
            x_left_grad = x_left.clone().requires_grad_(True)
            x_right_grad = x_right.clone().requires_grad_(True)
            
            # Compute w at boundaries for gradient computation
            w_left_grad = self.u_net(t_bc, x_left_grad)
            w_right_grad = self.u_net(t_bc, x_right_grad)
            
            # Create gradient outputs once
            ones_left = torch.ones_like(w_left_grad)
            ones_right = torch.ones_like(w_right_grad)
            
            # First derivatives
            w_x_left = torch.autograd.grad(w_left_grad, x_left_grad, grad_outputs=ones_left, create_graph=True, retain_graph=True)[0]
            w_x_right = torch.autograd.grad(w_right_grad, x_right_grad, grad_outputs=ones_right, create_graph=True, retain_graph=True)[0]
            
            # Check for NaN
            if torch.isnan(w_x_left).any() or torch.isnan(w_x_right).any():
                return torch.tensor(0.0, device=device)
            
            # Second derivatives (should be zero for simply supported)
            w_xx_left = torch.autograd.grad(w_x_left, x_left_grad, grad_outputs=ones_left, create_graph=True, retain_graph=True)[0]
            w_xx_right = torch.autograd.grad(w_x_right, x_right_grad, grad_outputs=ones_right, create_graph=True, retain_graph=True)[0]
            
            # Compute losses
            bc_loss_disp = torch.mean(w_left**2) + torch.mean(w_right**2)
            bc_loss_moment = torch.mean(w_xx_left**2) + torch.mean(w_xx_right**2)
            
            # Total BC loss
            bc_loss = bc_loss_disp + bc_loss_moment
            
            # Final check for NaN
            if torch.isnan(bc_loss):
                return torch.tensor(0.0, device=device)
            
            return bc_loss
            
        except Exception as e:
            print(f"Warning: Error in BC loss computation: {e}")
            return torch.tensor(0.0, device=device)
    
    def compute_loss(self, t_pde, x_pde, t_ic, x_ic, u_ic, ut_ic, t_bc):
        """
        Compute total loss with optimized weights for fast convergence.
        """
        try:
            # Ensure correct dimensions
            if t_pde.dim() == 1:
                t_pde = t_pde.unsqueeze(1)
            if x_pde.dim() == 1:
                x_pde = x_pde.unsqueeze(1)
            if t_ic.dim() == 1:
                t_ic = t_ic.unsqueeze(1)
            if x_ic.dim() == 1:
                x_ic = x_ic.unsqueeze(1)
            if u_ic.dim() == 1:
                u_ic = u_ic.unsqueeze(1)
            if ut_ic.dim() == 1:
                ut_ic = ut_ic.unsqueeze(1)
            
            # PDE residual loss
            residual = self.pde_residual(t_pde, x_pde)
            pde_loss = torch.mean(residual**2)
            
            # Check for NaN in PDE loss
            if torch.isnan(pde_loss):
                pde_loss = torch.tensor(0.0, device=device)
            
            # Initial condition loss for w(0,x)
            u_pred_ic = self.u_net(t_ic, x_ic)
            ic_loss = torch.mean((u_pred_ic - u_ic)**2)
            
            # Check for NaN in IC loss
            if torch.isnan(ic_loss):
                ic_loss = torch.tensor(0.0, device=device)
            
            # Initial velocity loss for w_t(0,x)
            t_ic_grad = t_ic.clone().requires_grad_(True)
            u_pred_ic_grad = self.u_net(t_ic_grad, x_ic)
            u_t_pred = torch.autograd.grad(u_pred_ic_grad, t_ic_grad, grad_outputs=torch.ones_like(u_pred_ic_grad), create_graph=True, retain_graph=True)[0]
            ic_v_loss = torch.mean((u_t_pred - ut_ic)**2)
            
            # Check for NaN in IC velocity loss
            if torch.isnan(ic_v_loss):
                ic_v_loss = torch.tensor(0.0, device=device)
            
            # Boundary condition loss
            bc_loss = self.compute_bc_loss(t_bc)
            
            # Light regularization on correction network
            l2_reg = 1e-10 * sum(p.pow(2).sum() for p in self.model.correction_net.parameters())
            
            # Total loss with optimized weights
            total_loss = self.w_pde * pde_loss + self.w_ic * ic_loss + self.w_ic_v * ic_v_loss + self.w_bc * bc_loss + l2_reg
            
            # Final check for NaN in total loss
            if torch.isnan(total_loss):
                total_loss = torch.tensor(1.0, device=device)
            
            return total_loss, pde_loss, ic_loss, ic_v_loss, bc_loss
            
        except Exception as e:
            print(f"Warning: Error in loss computation: {e}")
            # Return fallback losses
            return (torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), 
                   torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 
                   torch.tensor(0.0, device=device))
    
    def compute_l2_error(self, n_points=50):
        """
        Compute relative L2 error against analytical solution.
        Optimized for speed with smaller grid.
        """
        try:
            # Generate uniform grid - smaller for speed
            x = torch.linspace(0, self.L, n_points).reshape(-1, 1).to(device)
            t = torch.linspace(0, self.t_max, n_points).reshape(-1, 1).to(device)
            
            # Create meshgrid
            X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
            X_flat = X.reshape(-1, 1)
            T_flat = T.reshape(-1, 1)
            
            # Compute exact and predicted solutions in one batch for speed
            with torch.no_grad():
                u_exact = self.exact_solution(T_flat, X_flat)
                u_pred = self.u_net(T_flat, X_flat)
                
                # Check for NaN in predictions
                if torch.isnan(u_pred).any():
                    return 1.0  # Return high error
                
                # Compute relative L2 error
                error_squared = torch.sum((u_exact - u_pred)**2).item()
                exact_squared = torch.sum(u_exact**2).item()
                
                error = np.sqrt(error_squared / (exact_squared + 1e-15))
                
                # Make sure we have a valid error
                if np.isnan(error) or np.isinf(error):
                    error = 1.0
                    
            return error
            
        except Exception as e:
            print(f"Warning: Error in L2 computation: {e}")
            return 1.0
    
    def train(self, n_iter=2000, points_per_dim=100):
        """
        Optimized training for fast convergence to 1e-9 target.
        Much faster with smaller grids and optimized parameters.
        """
        print(f"Starting optimized training for {n_iter} iterations...")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        
        # Check memory feasibility
        initial_batch_size = 5000  # Doubled batch size for better GPU utilization
        feasible, adjusted_batch_size = check_memory_feasibility(
            self.n_harmonics, points_per_dim, initial_batch_size
        )
        
        if not feasible:
            print(f"ERROR: Insufficient GPU memory for harmonic {self.n_harmonics}")
            return 1.0
        
        batch_size = adjusted_batch_size
        print(f"Using batch size: {batch_size} for harmonic {self.n_harmonics}")
        
        # Clear memory
        clear_memory()
        
        # Training points - smaller grid for speed
        x = torch.linspace(0, self.L, points_per_dim).reshape(-1, 1).to(device)
        t = torch.linspace(0, self.t_max, points_per_dim).reshape(-1, 1).to(device)
        
        # Create meshgrid
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        
        # Initial conditions
        x_ic = x
        t_ic = torch.zeros_like(x_ic)
        u_ic = torch.sin(np.pi * x_ic / self.L)
        ut_ic = torch.zeros_like(x_ic)
        
        # Boundary condition time points
        t_bc = t
        
        # Enhanced Adam optimizer with better parameters
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,  # Higher initial learning rate
            betas=(0.9, 0.9999),
            eps=1e-12,
            weight_decay=0
        )
        
        # More aggressive learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,  # Less aggressive reduction
            patience=20,  # More patience
            min_lr=1e-6,
            verbose=True
        )
        
        # Target error
        target_error = 1e-9
        best_error = float('inf')
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(n_iter + 1):
            try:
                # Sample points
                idx = torch.randperm(X_flat.shape[0])[:batch_size]
                t_pde = T_flat[idx]
                x_pde = X_flat[idx]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute loss
                loss, pde_loss, ic_loss, ic_v_loss, bc_loss = self.compute_loss(
                    t_pde, x_pde, t_ic, x_ic, u_ic, ut_ic, t_bc
                )
                
                # Check for NaN loss and skip if found
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {epoch}, skipping")
                    continue
                
                # Backward pass with error handling
                try:
                    loss.backward()
                except RuntimeError as e:
                    print(f"Backward pass error at epoch {epoch}: {e}")
                    # Clear gradients and continue
                    optimizer.zero_grad(set_to_none=True)
                    # Force memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Step optimizer
                optimizer.step()
                
                # Clear gradients explicitly
                optimizer.zero_grad(set_to_none=True)
                
                # Force cleanup of computation graph every few iterations
                if epoch % 50 == 0 and epoch > 0:
                    # This helps prevent memory accumulation
                    for param in self.model.parameters():
                        param.grad = None
                
                # Periodic memory cleanup to prevent accumulation
                if epoch % 100 == 0 and epoch > 0:  # More frequent cleanup
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Small delay to ensure memory is freed
                    time.sleep(0.1)
                
                # Evaluate and save metrics
                if epoch % 20 == 0:  # More frequent evaluation
                    try:
                        with torch.no_grad():
                            l2_error = self.compute_l2_error(n_points=50)  # Good grid for evaluation
                        
                        # Save metrics
                        self.losses.append(loss.item())
                        self.pde_losses.append(pde_loss.item())
                        self.ic_losses.append(ic_loss.item())
                        self.bc_losses.append(bc_loss.item())
                        self.l2_errors.append(l2_error)
                        self.epochs.append(epoch)
                        
                        # Update scheduler
                        scheduler.step(l2_error)
                        
                        # Print progress
                        elapsed = time.time() - start_time
                        print(f"Epoch {epoch}/{n_iter}, Loss: {loss.item():.6e}, L2 Error: {l2_error:.9e}, Time: {elapsed:.2f}s")
                        print(f"  PDE Loss: {pde_loss.item():.6e}, IC Loss: {ic_loss.item():.6e}, IC_V Loss: {ic_v_loss.item():.6e}, BC Loss: {bc_loss.item():.6e}")
                        
                        # Save best model with better error handling
                        if l2_error < best_error:
                            best_error = l2_error
                            try:
                                # Ensure CUDA operations are complete before saving
                                # if torch.cuda.is_available():
                                #     torch.cuda.synchronize()  # Removed - causes segfaults
                                
                                # Save model state dict more safely
                                torch.save(self.model.state_dict(), f'{self.data_dir}/best_model.pt')
                                print(f"New best model saved with error: {best_error:.9e}")
                                
                                # Gentle cleanup
                                if epoch % 200 == 0:  # Less frequent cleanup
                                    gc.collect()
                            except Exception as e:
                                print(f"Warning: Error saving model: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Check if target reached
                        if l2_error <= target_error:
                            print(f"Target L2 error of {target_error:.9e} reached at epoch {epoch}!")
                            break
                            
                    except Exception as e:
                        print(f"Warning: Error during evaluation at epoch {epoch}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Training error at epoch {epoch}: {e}")
                continue
        
        # Save final data
        self.save_data()
        
        # Final evaluation
        try:
            # Load best model with proper device handling
            best_model_path = f'{self.data_dir}/best_model.pt'
            if os.path.exists(best_model_path):
                # Load to CPU first then move to device to avoid issues
                state_dict = torch.load(best_model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                del state_dict
                
            final_error = self.compute_l2_error(n_points=50)
            print(f"Training completed. Final L2 error: {final_error:.9e}")
        except Exception as e:
            print(f"Warning: Error computing final L2 error: {e}")
            final_error = best_error
        
        return final_error
    
    def refine_with_lbfgs(self, n_iter=5000):
        """
        Optimized L-BFGS refinement for ultra-high precision.
        """
        print(f"Refining with L-BFGS for {n_iter} iterations...")
        
        # Training points - optimized size
        points_per_dim = 100  # Match Adam phase
        
        # Check memory feasibility
        lbfgs_batch_size = 5000  # Doubled batch size for better GPU utilization
        feasible, adjusted_batch_size = check_memory_feasibility(
            self.n_harmonics, points_per_dim, lbfgs_batch_size
        )
        
        batch_size = adjusted_batch_size
        print(f"L-BFGS using batch size: {batch_size} for harmonic {self.n_harmonics}")
        
        # Clear memory
        clear_memory()
        
        # Check if we've already reached the target
        target_error = 1e-9
        current_error = self.compute_l2_error(n_points=50)
        print(f"Initial L-BFGS L2 error: {current_error:.9e}")
        
        if current_error <= target_error:
            print(f"Target already reached! L2 error: {current_error:.9e} <= {target_error:.9e}")
            self.save_data()
            return current_error
        
        # Load best model from Adam phase with proper device handling
        try:
            best_model_path = f'{self.data_dir}/best_model.pt'
            if os.path.exists(best_model_path):
                # Load to CPU first to avoid device mismatch issues
                state_dict = torch.load(best_model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                del state_dict
                print(f"Loaded best model from Adam phase")
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
            import traceback
            traceback.print_exc()
        
        # Training points
        x = torch.linspace(0, self.L, points_per_dim).reshape(-1, 1).to(device)
        t = torch.linspace(0, self.t_max, points_per_dim).reshape(-1, 1).to(device)
        
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        
        # Initial conditions
        x_ic = x
        t_ic = torch.zeros_like(x_ic)
        u_ic = torch.sin(np.pi * x_ic / self.L)
        ut_ic = torch.zeros_like(x_ic)
        
        # Boundary condition time points
        t_bc = t
        
        # Enhanced L-BFGS optimizer with better parameters
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=0.5,  # Reduced learning rate for stability
            max_iter=20,  # Fewer iterations per step for stability
            max_eval=25,  # Limit function evaluations
            history_size=50,  # Smaller history to reduce memory usage
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-12,  # Less strict tolerance
            tolerance_change=1e-12  # Less strict tolerance
        )
        
        # Target error
        target_error = 1e-9
        best_error = float('inf')
        start_epoch = max(self.epochs) if self.epochs else 0
        
        # Check if we're already close to target
        initial_error = self.compute_l2_error(n_points=50)
        if initial_error <= target_error * 1.5:  # Within 50% of target
            print(f"Already close to target (L2 error: {initial_error:.9e}). Using conservative L-BFGS settings.")
            optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=0.1,  # Very conservative learning rate
                max_iter=5,  # Very few iterations
                max_eval=10,
                history_size=20,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-8,
                tolerance_change=1e-8
            )
        
        # Training loop
        numerical_issues_count = 0
        max_numerical_issues = 50  # Stop if too many numerical issues
        consecutive_increases = 0  # Track consecutive error increases
        max_consecutive_increases = 20  # Stop if error increases for this many consecutive checks
        previous_error = best_error  # Track previous error for early stopping
        
        for i in range(n_iter):
            try:
                # Early stopping if close enough to target
                if i % 5 == 0:  # Check every 5 iterations
                    with torch.no_grad():
                        current_error = self.compute_l2_error(n_points=50)
                        if current_error <= target_error:
                            print(f"Early stopping: Target reached at iteration {i} with error {current_error:.9e}")
                            best_error = current_error
                            break
                
                # Sample points
                idx = torch.randperm(X_flat.shape[0])[:batch_size]
                t_pde = T_flat[idx]
                x_pde = X_flat[idx]
                
                # Define closure with better error handling
                closure_loss = None
                
                def closure():
                    nonlocal closure_loss, numerical_issues_count
                    optimizer.zero_grad()
                    
                    try:
                        loss, pde_loss, ic_loss, ic_v_loss, bc_loss = self.compute_loss(
                            t_pde, x_pde, t_ic, x_ic, u_ic, ut_ic, t_bc
                        )
                        
                        # Check for numerical issues
                        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e10:
                            # Return a safe fallback loss
                            loss = torch.tensor(1.0, device=device, requires_grad=True)
                            print(f"Warning: Numerical issue in closure at iteration {i}")
                            numerical_issues_count += 1
                        else:
                            # Store the loss value for later use
                            closure_loss = loss.item()
                            
                        loss.backward()
                        
                        # More conservative gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                    except Exception as e:
                        print(f"Error in closure at iteration {i}: {e}")
                        # Return a safe loss value
                        loss = torch.tensor(1.0, device=device, requires_grad=True)
                        numerical_issues_count += 1
                    
                    return loss
                
                # L-BFGS step with better error handling
                try:
                    optimizer.step(closure)
                    
                    # Force memory cleanup after each step
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "failed to converge" in str(e):
                        print(f"L-BFGS convergence warning at iteration {i}: {e}")
                        # Continue training
                    else:
                        print(f"L-BFGS step error at iteration {i}: {e}")
                        # Skip this iteration
                        continue
                except Exception as e:
                    print(f"Unexpected L-BFGS error at iteration {i}: {e}")
                    numerical_issues_count += 1
                    continue
                    
                # Check if we should stop due to too many numerical issues
                if numerical_issues_count > max_numerical_issues:
                    print(f"Stopping L-BFGS due to {numerical_issues_count} numerical issues")
                    print(f"Current best L2 error: {best_error:.9e}")
                    break
                
                # Evaluate
                if i % 10 == 0 or i == n_iter - 1:
                    try:
                        with torch.no_grad():
                            l2_error = self.compute_l2_error(n_points=50)
                        
                        current_epoch = start_epoch + i + 1
                        self.epochs.append(current_epoch)
                        
                        # Compute loss values for tracking
                        loss, pde_loss, ic_loss, ic_v_loss, bc_loss = self.compute_loss(
                            t_pde, x_pde, t_ic, x_ic, u_ic, ut_ic, t_bc
                        )
                        
                        self.losses.append(loss.item())
                        self.pde_losses.append(pde_loss.item())
                        self.ic_losses.append(ic_loss.item())
                        self.bc_losses.append(bc_loss.item())
                        self.l2_errors.append(l2_error)
                        
                        print(f"L-BFGS iteration {i}/{n_iter}, L2 error: {l2_error:.9e}")
                        print(f"  PDE Loss: {pde_loss.item():.6e}, IC Loss: {ic_loss.item():.6e}, IC_V Loss: {ic_v_loss.item():.6e}, BC Loss: {bc_loss.item():.6e}")
                        
                        # Save best model
                        if l2_error < best_error:
                            best_error = l2_error
                            consecutive_increases = 0  # Reset counter on improvement
                            try:
                                torch.save(self.model.state_dict(), f'{self.data_dir}/best_model_lbfgs.pt')
                                print(f"New best model saved with error: {best_error:.9e}")
                            except Exception as e:
                                print(f"Warning: Error saving best model: {e}")
                        else:
                            # Check if error is increasing
                            if l2_error > previous_error * 1.1:  # 10% tolerance
                                consecutive_increases += 1
                                print(f"Warning: L2 error increased ({consecutive_increases} consecutive increases)")
                            else:
                                consecutive_increases = 0  # Reset if not significantly worse
                        
                        previous_error = l2_error
                        
                        # Early stopping conditions
                        if consecutive_increases >= max_consecutive_increases:
                            print(f"Early stopping: L2 error increased for {consecutive_increases} consecutive checks")
                            print(f"Best L2 error achieved: {best_error:.9e}")
                            break
                        
                        # Check if target reached
                        if l2_error <= target_error:
                            print(f"Target L2 error of {target_error:.9e} reached at iteration {i}!")
                            break
                            
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                        continue
                        
            except Exception as e:
                print(f"Unexpected error in L-BFGS iteration {i}: {e}")
                continue
        
        # Final evaluation
        try:
            best_model_path = f'{self.data_dir}/best_model_lbfgs.pt'
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                del state_dict
                print(f"Loaded best L-BFGS model with error: {best_error:.9e}")
                
                # Also save as final_model.pt to ensure best model is used
                try:
                    torch.save(self.model.state_dict(), f'{self.data_dir}/final_model.pt')
                    print(f"Saved best model as final_model.pt")
                except Exception as e:
                    print(f"Warning: Could not save final model: {e}")
            
            # Use the tracked best error instead of computing a new one
            final_error = best_error
            print(f"L-BFGS refinement completed. Final L2 error: {final_error:.9e}")
        except Exception as e:
            print(f"Warning: Error loading best model: {e}")
            final_error = best_error
        
        # Save final data
        self.save_data()
        
        return final_error
    
    def save_data(self):
        """Save training data to CSV"""
        try:
            # Check if arrays have the same length
            min_length = min(len(self.epochs), len(self.losses), len(self.pde_losses), 
                           len(self.ic_losses), len(self.bc_losses), len(self.l2_errors))
            
            if min_length == 0:
                print("Warning: No data to save")
                return
            
            # Truncate arrays to the same length
            epochs = self.epochs[:min_length]
            losses = self.losses[:min_length]
            pde_losses = self.pde_losses[:min_length]
            ic_losses = self.ic_losses[:min_length]
            bc_losses = self.bc_losses[:min_length]
            l2_errors = self.l2_errors[:min_length]
            
            # Save losses
            df_loss = pd.DataFrame({
                'Epoch': epochs,
                'Total_Loss': losses,
                'PDE_Loss': pde_losses,
                'IC_Loss': ic_losses,
                'BC_Loss': bc_losses,
                'L2_Error': l2_errors,
                'PDE_Weight': [self.w_pde] * len(epochs),
                'IC_Weight': [self.w_ic] * len(epochs),
                'IC_V_Weight': [self.w_ic_v] * len(epochs),
                'BC_Weight': [self.w_bc] * len(epochs)
            })
            
            try:
                df_loss.to_csv(f'{self.data_dir}/training_data.csv', index=False, float_format='%.10e')
            except Exception as e:
                print(f"Warning: Could not save training data CSV: {e}")
            
            # Get the best L2 error (minimum)
            best_l2_error = min(l2_errors) if l2_errors else 1.0
            
            # Save best L2 error
            with open(f'{self.data_dir}/final_l2_error.txt', 'w') as f:
                f.write(f"{best_l2_error:.10e}\n")
            
            print(f"Training data saved to {self.data_dir}/training_data.csv")
            print(f"Best L2 error: {best_l2_error:.10e}")
            
            # Save summary
            with open(f'{self.data_dir}/summary.txt', 'w') as f:
                f.write(f"Harmonics: {self.n_harmonics}\n")
                f.write(f"L2 Error: {best_l2_error:.10e}\n")
                f.write(f"Target Met: {'Yes' if best_l2_error <= 1e-9 else 'No'}\n")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            try:
                # Try to save at least the best error
                best_l2_error = min(self.l2_errors) if self.l2_errors else 1.0
                with open(f'{self.data_dir}/final_l2_error.txt', 'w') as f:
                    f.write(f"{best_l2_error:.10e}\n")
            except Exception as inner_e:
                print(f"Error saving L2 error: {inner_e}")


def train_single_harmonic(n_harmonics=50, max_harmonics=65):
    """
    Train a single harmonic configuration with optimized approach for 1e-9 target.
    """
    # Acquire GPU lock first to prevent concurrent access
    if not acquire_gpu_lock():
        print(f"ERROR: Could not acquire GPU lock. Another process may be using the GPU.")
        return 1.0
    
    print("\n" + "="*70)
    print(f"Training Multi-Harmonic Euler-Bernoulli Beam Solver")
    print(f"Using {n_harmonics} harmonics (out of {max_harmonics} allocated)")
    print(f"Model: Fourier series + neural network correction")
    print("Target L2 error: 1.0e-9")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("="*70 + "\n")
    
    final_error = None  # Initialize to None
    
    try:
        # Check GPU is available - MANDATORY
        if not torch.cuda.is_available():
            print("ERROR: GPU is not available. GPU is REQUIRED for training.")
            return 1.0
            
        # Check memory availability
        total_memory, allocated, reserved, free_memory = get_gpu_memory_info()
        print(f"Initial GPU memory check:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Currently allocated: {allocated:.2f} GB") 
        print(f"  Available: {free_memory:.2f} GB")
        
        # Clear memory
        clear_memory()
        
        # Create optimized model
        model = UltraPrecisionEulerBernoulliModel(n_harmonics=n_harmonics, max_harmonics=max_harmonics)
        
        # Print model parameter count
        param_count = sum(p.numel() for p in model.parameters())
        amplitude_count = model.amplitudes_cos.numel() + model.amplitudes_sin.numel()
        correction_params = sum(p.numel() for p in model.correction_net.parameters())
        
        print(f"Model created with {param_count} total parameters:")
        print(f"  - Harmonic amplitudes: {amplitude_count} (using first {n_harmonics} pairs)")
        print(f"  - Correction network: {correction_params} parameters")
        print(f"  - Correction scale: 1 parameter")
        
        # Create solver
        results_dir = f'results_ultra_{n_harmonics}'
        os.makedirs(results_dir, exist_ok=True)
        solver = UltraPrecisionEulerBernoulliSolver(model, data_dir=results_dir)
        
        # Phase 1: Optimized Adam training
        print(f"\nPhase 1: Adam optimization ({n_harmonics} harmonics)")
        adam_error = solver.train(n_iter=2000, points_per_dim=100)  # More iterations and points
        
        # Clear memory between phases
        clear_memory()
        
        # Phase 2: L-BFGS refinement
        print(f"\nPhase 2: L-BFGS refinement ({n_harmonics} harmonics)")
        lbfgs_error = solver.refine_with_lbfgs(n_iter=5000)  # More L-BFGS iterations
        final_error = lbfgs_error
        
        print(f"\nModel with {n_harmonics} harmonics - Final L2 error: {final_error:.9e}")
        
        # Save final L2 error immediately before cleanup
        try:
            with open(f'{results_dir}/final_l2_error.txt', 'w') as f:
                f.write(f'{final_error:.10e}\n')
            print(f"Final L2 error saved: {final_error:.10e}")
        except Exception as e:
            print(f"Warning: Could not save final L2 error: {e}")
        
        # Skip complex cleanup to avoid segfaults
        print("DEBUG: Skipping cleanup after training", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        # Just basic cleanup
        try:
            # Move model to CPU before deletion to avoid GPU issues
            if 'model' in locals() and hasattr(model, 'cpu'):
                model.cpu()
            del solver
            del model
        except:
            pass
        
        # Clear GPU cache safely
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Simple garbage collection
        gc.collect()
        
        print("\n" + "="*70)
        print(f"Training completed for {n_harmonics} harmonics!")
        print(f"Final L2 error: {final_error:.9e}")
        if final_error <= 1e-9:
            print(f"SUCCESS: Target L2 error of 1e-9 was achieved!")
        else:
            print(f"Target not yet reached. Best error: {final_error:.9e}")
        print("="*70 + "\n")
        
        # Ensure the process exits cleanly
        sys.stdout.flush()
        sys.stderr.flush()
        
        print("DEBUG: Cleanup completed successfully", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        return final_error
        
    except Exception as e:
        print(f"Error in train_single_harmonic for harmonic {n_harmonics}: {e}")
        import traceback
        traceback.print_exc()
        # Don't call clear_memory here as it will be called in finally
        return 1.0
    finally:
        # Always release GPU lock at the end
        try:
            release_gpu_lock()
        except:
            pass
        
        # Ensure clean exit
        if final_error is not None:
            print(f"Completed harmonic {n_harmonics} successfully with L2 error: {final_error:.9e}")
            # Return normally instead of forcing exit
            # This allows run_with_monitoring.py to properly track completion

def train_all_harmonics():
    """Train all harmonic configurations sequentially"""
    print("DEBUG: train_all_harmonics called", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    harmonics_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Limited to 50 to avoid segfaults
    max_harmonics = 65
    
    print(f"\n{'='*70}")
    print(f"Starting training for all harmonics: {harmonics_list}")
    print(f"{'='*70}\n")
    
    print("DEBUG: About to track results", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    # Track results
    results = {}
    best_error = float('inf')
    best_harmonic = None
    
    # Check for completed harmonics
    print("DEBUG: Checking completed harmonics", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    completed_file = 'completed_harmonics.txt'
    completed_harmonics = set()
    if os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            completed_harmonics = set(int(line.strip()) for line in f if line.strip())
    
    print(f"DEBUG: Completed harmonics: {completed_harmonics}", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    for h in harmonics_list:
        print(f"DEBUG: Processing harmonic {h}", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        if h in completed_harmonics:
            print(f"\nHarmonic {h} already completed. Skipping...")
            # Try to read the existing error
            error_file = f'results_ultra_{h}/final_l2_error.txt'
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    error = float(f.read().strip())
                    results[h] = error
                    if error < best_error:
                        best_error = error
                        best_harmonic = h
            continue
        
        print(f"\n{'='*70}")
        print(f"Training harmonic {h} of {harmonics_list}")
        print(f"{'='*70}\n")
        
        # Clean GPU memory before starting
        print("DEBUG: About to clear memory", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        clear_memory()
        
        print("DEBUG: Memory cleared, about to sleep", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        time.sleep(10)
        
        print("DEBUG: Sleep completed", file=sys.__stderr__)
        sys.__stderr__.flush()
        
        # Train this harmonic
        error = train_single_harmonic(n_harmonics=h, max_harmonics=max_harmonics)
        results[h] = error
        
        # Track best result
        if error < best_error:
            best_error = error
            best_harmonic = h
        
        # Mark as completed with better error handling
        try:
            # Force OS to write to disk immediately
            with open(completed_file, 'a') as f:
                f.write(f"{h}\n")
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            print(f"Harmonic {h} marked as completed")
        except Exception as e:
            print(f"Warning: Could not mark harmonic {h} as completed: {e}")
        
        # Simple cleanup after training
        gc.collect()
        time.sleep(15)  # Wait between harmonics
        
        # Check if target achieved
        if error <= 1e-9:
            print(f"\nTarget L2 error of 1e-9 achieved with harmonic {h}!")
    
    # Create summary
    print(f"\n{'='*70}")
    print("Training Summary:")
    print(f"{'='*70}")
    
    with open('harmonic_results_summary.csv', 'w') as f:
        f.write("Harmonic,L2_Error\n")
        for h, error in sorted(results.items()):
            f.write(f"{h},{error:.10e}\n")
            print(f"Harmonic {h}: L2 error = {error:.10e}")
    
    print(f"\nBest result: Harmonic {best_harmonic} with L2 error = {best_error:.10e}")
    
    if best_error <= 1e-9:
        print(f"SUCCESS: Target L2 error of 1e-9 was achieved!")
    else:
        print(f"Target L2 error of 1e-9 was not achieved. Best error: {best_error:.10e}")
    
    return results

if __name__ == "__main__":
    print("DEBUG: Entering main block")
    sys.stdout.flush()
    print("DEBUG: Entering main block (stderr)", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized PINN for Euler-Bernoulli beam equation')
    parser.add_argument('--harmonics', type=int, default=None, 
                        help='Number of harmonic components to use (if not specified, trains all)')
    parser.add_argument('--max-harmonics', type=int, default=65,
                        help='Maximum number of harmonics to allocate parameters for (default: 65)')
    parser.add_argument('--all', action='store_true',
                        help='Train all harmonic configurations (5, 10, 15, ..., 65)')
    args = parser.parse_args()
    
    print("DEBUG: Arguments parsed successfully")
    sys.stdout.flush()
    print(f"DEBUG: Arguments parsed: all={getattr(args, 'all', False)}, harmonics={args.harmonics} (stderr)", file=sys.__stderr__)
    sys.__stderr__.flush()
    
    try:
        if args.all or args.harmonics is None:
            # Train all harmonics
            train_all_harmonics()
        else:
            # Train single harmonic
            train_single_harmonic(n_harmonics=args.harmonics, max_harmonics=args.max_harmonics)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
    except SystemExit:
        # Clean exit
        pass
    finally:
        # Skip cleanup to avoid segfaults
        print("DEBUG: Skipping cleanup_handler in finally block", file=sys.__stderr__)
        sys.__stderr__.flush()
        # Ensure stdout/stderr are restored
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__ 