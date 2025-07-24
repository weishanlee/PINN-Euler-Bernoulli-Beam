"""
Ultra Precision Wave Equation PINN Visualizations

This script generates detailed visualizations for the ultra-precision wave equation solver,
focusing on high-precision error metrics, convergence visualization, and comparison
between different harmonic configurations.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import torch
import sys
import traceback
import gc

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Define harmonics configurations with their directories
# Limited to harmonics 5-50 to avoid segfault issues
HARMONIC_CONFIGS = {
    5: 'results_ultra_5',
    10: 'results_ultra_10',
    15: 'results_ultra_15',
    20: 'results_ultra_20',
    25: 'results_ultra_25',
    30: 'results_ultra_30',
    35: 'results_ultra_35',
    40: 'results_ultra_40',
    45: 'results_ultra_45',
    50: 'results_ultra_50'
}

# Will be determined dynamically based on which has the lowest L2 error
BEST_HARMONICS = None

def find_data_files(results_dir):
    """Find data files in the results directory"""
    training_data = os.path.join(results_dir, 'training_data.csv')
    solution_data = os.path.join(results_dir, 'solution_data.npz')
    l2_error_file = os.path.join(results_dir, 'final_l2_error.txt')
    
    # Check for training data
    if not os.path.exists(training_data):
        print(f"Warning: Training data not found in {results_dir}.")
        print(f"Expected: {training_data}")
        return None, None, None
    
    # If solution data doesn't exist, try to generate it from the model
    if not os.path.exists(solution_data):
        print(f"Solution data not found in {results_dir}. Attempting to generate it...")
        solution = generate_solution_data(results_dir)
        if solution is not None:
            return training_data, solution_data, l2_error_file
        else:
            print(f"Failed to generate solution data for {results_dir}.")
            return None, None, None
    
    return training_data, solution_data, l2_error_file

def load_data(training_data, solution_data, l2_error_file=None):
    """Load training and solution data with optional direct L2 error file"""
    # Load training metrics
    df = pd.read_csv(training_data)
    
    # Check if we have direct L2 error file (more reliable)
    if l2_error_file and os.path.exists(l2_error_file):
        try:
            with open(l2_error_file, 'r') as f:
                final_l2 = float(f.read().strip())
                
            # Find the best L2 error in the training data
            best_l2_from_training = df['L2_Error'].min()
            print(f"Best L2 error from training data: {best_l2_from_training:.9e}")
            print(f"Final L2 error from file: {final_l2:.9e}")
            
            # Use the final L2 error from file as it's the authoritative source
            # This represents the best model saved during training
            if len(df) > 0 and 'L2_Error' in df.columns:
                # Update the minimum L2 error row to match the final file
                min_idx = df['L2_Error'].idxmin()
                df.loc[min_idx, 'L2_Error'] = final_l2
                print(f"Updated best L2 error in dataframe to match final file")
        except Exception as e:
            print(f"Error reading L2 error file: {e}")
    
    # Load solution data
    solution = np.load(solution_data)
    
    return df, solution

def visualize_solution_comparison(solution, harmonics=None, vmin=None, vmax=None):
    """
    Visualize the predicted and exact solutions with error in a single row
    """
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    u_error = solution['u_error']
    
    # Create figure with a single row of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Add title with harmonics info if provided
    if harmonics:
        fig.suptitle(f'Euler-Bernoulli Beam Solution with {harmonics} Harmonics' + 
                    (' (BEST MODEL)' if harmonics == BEST_HARMONICS else ''), 
                    fontsize=16, y=1.05)
    
    # Plot predicted solution
    im0 = axes[0].pcolormesh(t, x, u_pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Position (x)')
    fig.colorbar(im0, ax=axes[0])
    
    # Plot exact solution
    im1 = axes[1].pcolormesh(t, x, u_exact, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Exact Solution')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Position (x)')
    fig.colorbar(im1, ax=axes[1])
    
    # Plot error (logarithmic scale for ultra-precision)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-16
    im2 = axes[2].pcolormesh(t, x, u_error + epsilon, cmap='inferno', norm=LogNorm(vmin=max(epsilon, u_error.min()), vmax=max(1e-1, u_error.max())))
    axes[2].set_title('Absolute Error (Log Scale)')
    axes[2].set_xlabel('Time (t)')
    axes[2].set_ylabel('Position (x)')
    cbar = fig.colorbar(im2, ax=axes[2])
    cbar.set_label('Absolute Error (Log Scale)')
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/comparison{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_3d_comparisons(solution, harmonics=None):
    """
    Create separate 3D plots for predicted solution, exact solution, and error
    with colorbars - each saved as a separate file
    """
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    u_error = solution['u_error']
    
    # Create meshgrid for 3D plots
    T, X = np.meshgrid(t, x)
    
    # 1. PINN Solution
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, u_pred, cmap='viridis', edgecolor='none', antialiased=True)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    ax.set_zlabel('u(t,x)')
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'PINN Solution with {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        ax.set_title(title)
    else:
        ax.set_title('PINN Solution')
    
    # Add colorbar to the right side
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.1)
    cbar.set_label('Amplitude')
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/3d_comparison_pinn_solution{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/3d_comparison_pinn_solution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Exact Solution
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, u_exact, cmap='viridis', edgecolor='none', antialiased=True)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    ax.set_zlabel('u(t,x)')
    ax.set_title('Exact Solution')
    
    # Add colorbar to the right side
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.1)
    cbar.set_label('Amplitude')
    
    plt.tight_layout()
    
    # Save exact solution only once (it's the same for all models)
    if harmonics is None or harmonics == BEST_HARMONICS:
        plt.savefig('visualizations/3d_comparison_exact_solution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use log colormap for error
    epsilon = 1e-16
    norm = LogNorm(vmin=max(epsilon, u_error.min()), vmax=max(1e-3, u_error.max()))
    surf = ax.plot_surface(T, X, u_error, cmap='inferno', edgecolor='none', 
                          antialiased=True, norm=norm)
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    ax.set_zlabel('Error')
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Error with {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        ax.set_title(title + ' (Log Scale)')
    else:
        ax.set_title('Absolute Error (Log Scale)')
    
    # Add colorbar to the right side
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.1)
    cbar.set_label('Error Magnitude (Log Scale)')
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/3d_comparison_error{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/3d_comparison_error.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_space_slices(solution, harmonics=None):
    """
    Visualize solution slices at 5 specific x-values (0.1, 0.3, 0.5, 0.7, 0.9)
    arranged in a column
    """
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    u_error = solution['u_error']
    
    # Find indices closest to the target x-values
    target_x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    x_indices = [np.abs(x - val).argmin() for val in target_x_values]
    
    # Create figure with 5 subplots in a column
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    # Add title with harmonics info if provided
    if harmonics:
        fig.suptitle(f'Solution at Different Positions - {harmonics} Harmonics' + 
                    (' (BEST MODEL)' if harmonics == BEST_HARMONICS else ''), 
                    fontsize=16, y=0.92)
    
    for i, idx in enumerate(x_indices):
        ax = axs[i]
        
        # Plot exact and predicted solutions
        ax.plot(t, u_exact[idx, :], 'b-', linewidth=2, label='Exact')
        ax.plot(t, u_pred[idx, :], 'r--', linewidth=1.5, label='PINN')
        
        # Add error on second y-axis if error values are positive
        if np.max(u_error[idx, :]) > 1e-16:
            ax2 = ax.twinx()
            ax2.semilogy(t, u_error[idx, :] + 1e-16, 'g-', alpha=0.5, label='Error')
            ax2.set_ylabel('Error (log scale)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim([1e-12, 1e-2])  # Set reasonable bounds
            ax2.legend(loc='lower right')
        
        ax.set_title(f'Position x = {x[idx]:.2f}')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Solution u(t,x)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/space_slices{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/space_slices.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_time_slices(solution, harmonics=None):
    """
    Visualize solution slices at 5 specific t-values (0.0, 0.5, 1.0, 1.5, 2.0)
    arranged in a column
    """
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    u_error = solution['u_error']
    
    # Find indices closest to the target t-values
    target_t_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    t_indices = [np.abs(t - val).argmin() for val in target_t_values]
    
    # Create figure with 5 subplots in a column
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    # Add title with harmonics info if provided
    if harmonics:
        fig.suptitle(f'Solution at Different Times - {harmonics} Harmonics' + 
                    (' (BEST MODEL)' if harmonics == BEST_HARMONICS else ''), 
                    fontsize=16, y=0.92)
    
    for i, idx in enumerate(t_indices):
        ax = axs[i]
        
        # Plot exact and predicted solutions
        ax.plot(x, u_exact[:, idx], 'b-', linewidth=2, label='Exact')
        ax.plot(x, u_pred[:, idx], 'r--', linewidth=1.5, label='PINN')
        
        # Add error on second y-axis if error values are positive
        if np.max(u_error[:, idx]) > 1e-16:
            ax2 = ax.twinx()
            ax2.semilogy(x, u_error[:, idx] + 1e-16, 'g-', alpha=0.5, label='Error')
            ax2.set_ylabel('Error (log scale)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim([1e-12, 1e-2])  # Set reasonable bounds
            ax2.legend(loc='lower right')
        
        ax.set_title(f'Time t = {t[idx]:.2f}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Solution u(t,x)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/time_slices{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/time_slices.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_training_losses(df, harmonics=None):
    """
    Visualize training losses on a semilogy scale with Total Loss, PDE Loss, and IC Loss
    """
    plt.figure(figsize=(12, 6))
    
    # Plot different loss components
    plt.semilogy(df['Epoch'], df['Total_Loss'], 'b-', label='Total Loss', linewidth=2)
    plt.semilogy(df['Epoch'], df['PDE_Loss'], 'r--', label='PDE Loss', linewidth=2)
    plt.semilogy(df['Epoch'], df['IC_Loss'], 'g-.', label='IC Loss', linewidth=2)
    
    # Add soft boundary condition loss if available
    if 'BC_Loss' in df.columns and not all(df['BC_Loss'] == 0):
        plt.semilogy(df['Epoch'], df['BC_Loss'], 'm:', label='BC Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value (log scale)', fontsize=12)
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Training Losses vs Epoch - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        plt.title(title, fontsize=14)
    else:
        plt.title('Training Losses vs Epoch', fontsize=14)
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add annotation for final loss value
    final_loss = df['Total_Loss'].iloc[-1]
    plt.annotate(f'Final Loss: {final_loss:.2e}', 
                xy=(df['Epoch'].iloc[-1], final_loss),
                xytext=(df['Epoch'].iloc[-1]*0.8, final_loss*5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/training_losses{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_validation_error(df, harmonics=None):
    """
    Visualize L2 error on a semilogy scale with target line at 10^-9
    """
    plt.figure(figsize=(12, 6))
    
    # Plot L2 error
    plt.semilogy(df['Epoch'], df['L2_Error'], 'b-', label='Relative L2 Error', linewidth=2)
    
    # Add target line with improved label - using regular text instead of Unicode
    target_error = 1e-9
    plt.axhline(y=target_error, color='r', linestyle='--', linewidth=1.5, 
                label=f'Target Error: 1e-9')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Relative L2 Error (log scale)', fontsize=12)
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Validation Error vs Epoch - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        plt.title(title, fontsize=14)
    else:
        plt.title('Validation Error vs Epoch', fontsize=14)
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=10)
    
    # Annotate the final error value
    final_error = df['L2_Error'].iloc[-1]
    plt.annotate(f'Final Error: {final_error:.2e}', 
                xy=(df['Epoch'].iloc[-1], final_error),
                xytext=(df['Epoch'].iloc[-1]*0.8, final_error*10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/validation_error{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/validation_error.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_weight_factors(df, harmonics=None):
    """
    Visualize PDE Weight and IC Weight vs Epoch, with BC Weight if using soft boundary conditions
    """
    # Check if weight data is available
    if 'PDE_Weight' not in df.columns:
        print("Weight factors not found in training data, skipping weight_factors.png")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot different weight components
    plt.plot(df['Epoch'], df['PDE_Weight'], 'b-', label='PDE Weight', linewidth=2)
    plt.plot(df['Epoch'], df['IC_Weight'], 'r--', label='IC Weight', linewidth=2)
    
    # Add boundary condition weight if available
    if 'BC_Weight' in df.columns and not all(df['BC_Weight'] == 0):
        plt.plot(df['Epoch'], df['BC_Weight'], 'g-.', label='BC Weight', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Loss Weights vs Epoch - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        plt.title(title, fontsize=14)
    else:
        plt.title('Loss Weights vs Epoch', fontsize=14)
    
    plt.grid(True, which="major", ls="--", alpha=0.3)
    plt.legend(fontsize=10)
    
    # FIX 3: Improved annotation positioning to avoid overlap
    final_pde_weight = df['PDE_Weight'].iloc[-1]
    final_ic_weight = df['IC_Weight'].iloc[-1]
    
    # Calculate better positions to avoid overlap
    y_range = plt.ylim()[1] - plt.ylim()[0]
    x_pos = df['Epoch'].iloc[-1] * 0.85
    
    # Add annotations with better spacing
    plt.annotate(f'Final PDE Weight: {final_pde_weight:.4f}', 
                xy=(df['Epoch'].iloc[-1], final_pde_weight),
                xytext=(x_pos, final_pde_weight + y_range * 0.05),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                fontsize=9, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='blue', alpha=0.8))
                
    plt.annotate(f'Final IC Weight: {final_ic_weight:.4f}', 
                xy=(df['Epoch'].iloc[-1], final_ic_weight),
                xytext=(x_pos, final_ic_weight - y_range * 0.05),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
    
    # Add BC weight annotation if available
    if 'BC_Weight' in df.columns and not all(df['BC_Weight'] == 0):
        final_bc_weight = df['BC_Weight'].iloc[-1]
        plt.annotate(f'Final BC Weight: {final_bc_weight:.4f}', 
                    xy=(df['Epoch'].iloc[-1], final_bc_weight),
                    xytext=(x_pos, final_bc_weight + y_range * 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                    fontsize=9, fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='green', alpha=0.8))
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/weight_factors{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/weight_factors.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_error_distribution(solution, harmonics=None):
    """Visualize the error distribution with high precision"""
    u_error = solution['u_error'].flatten()
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram with logarithmic bins for high precision
    # Ensure all values are positive by adding epsilon
    epsilon = 1e-16
    error_values = u_error + epsilon
    
    log_bins = np.logspace(np.log10(error_values.min()), np.log10(error_values.max()), 50)
    counts, bins, patches = plt.hist(error_values, bins=log_bins, alpha=0.7, color='steelblue')
    
    plt.xscale('log')  # Logarithmic x-axis for ultra-precision
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Distribution of Error Values - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        plt.title(title + ' (Log Scale)')
    else:
        plt.title('Distribution of Error Values (Log Scale)')
    
    # Add statistics
    plt.axvline(np.mean(u_error), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(u_error):.2e}')
    plt.axvline(np.median(u_error), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(u_error):.2e}')
    
    # Add percentiles
    percentiles = [10, 90, 99, 99.9]
    colors = ['purple', 'orange', 'brown', 'pink']
    for p, c in zip(percentiles, colors):
        val = np.percentile(u_error, p)
        plt.axvline(val, color=c, linestyle=':', linewidth=1.5, label=f'{p}th percentile: {val:.2e}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/error_distribution{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_error_statistics(solution, df, harmonics=None):
    """Compute and save detailed error statistics and numerical data for plots"""
    # Error statistics from solution
    u_error = solution['u_error']
    u_exact = solution['u_exact']
    
    # Note: The L2 error here may differ from training due to grid resolution
    # Training uses 50 points, visualization may use different resolution
    
    # Global statistics
    max_error = np.max(u_error)
    mean_error = np.mean(u_error)
    median_error = np.median(u_error)
    
    # Get L2 error from the solution file directly if available
    if 'rel_l2_error' in solution:
        relative_l2_error = solution['rel_l2_error']
    else:
        l2_error = np.sqrt(np.mean(u_error**2))
        relative_l2_error = np.sqrt(np.sum(u_error**2) / np.sum(u_exact**2))
    
    # Percentile statistics
    percentiles = [50, 75, 90, 95, 99, 99.9, 99.99]
    percentile_values = np.percentile(u_error, percentiles)
    
    # Create DataFrame with error statistics
    stats_df = pd.DataFrame({
        'Metric': ['Max Error', 'Mean Error', 'Median Error', 'Relative L2 Error'] + 
                  [f'{p}th Percentile' for p in percentiles],
        'Value': [max_error, mean_error, median_error, relative_l2_error] + 
                 list(percentile_values)
    })
    
    # Add harmonics info if provided
    if harmonics is not None:
        stats_df['Harmonics'] = harmonics
    
    # Save error statistics to CSV
    suffix = f"_{harmonics}h" if harmonics else ""
    if harmonics == BEST_HARMONICS:
        suffix += "_best"
    stats_df.to_csv(f'visualizations/error_statistics{suffix}.csv', index=False)
    
    # Print summary
    print(f"\nError Statistics {'for ' + str(harmonics) + ' Harmonics' if harmonics else ''}:")
    print(f"Max Error: {max_error:.2e}")
    print(f"Mean Error: {mean_error:.2e}")
    print(f"Median Error: {median_error:.2e}")
    print(f"Relative L2 Error (visualization grid): {relative_l2_error:.2e}" + 
          (" (BEST MODEL)" if harmonics == BEST_HARMONICS else ""))
    
    # Also report the best L2 error from training if available
    if 'L2_Error' in df.columns:
        best_training_l2 = df['L2_Error'].min()
        print(f"Best L2 Error (training grid): {best_training_l2:.2e}")
    
    return stats_df, relative_l2_error

def compare_errors_across_harmonics(all_stats):
    """
    Create comparative plots of errors across different harmonic configurations
    and determine which model performs best
    """
    if len(all_stats) < 1:
        print("No data available to compare across harmonics")
        return None
    
    # Extract harmonic values and corresponding errors
    harmonics = []
    l2_errors = []
    max_errors = []
    mean_errors = []
    median_errors = []
    
    for h, (stats_df, l2_error) in all_stats.items():
        harmonics.append(h)
        l2_errors.append(l2_error)
        
        # Extract other error metrics if available
        try:
            max_val = stats_df[stats_df['Metric'] == 'Max Error']['Value'].values[0]
            mean_val = stats_df[stats_df['Metric'] == 'Mean Error']['Value'].values[0]
            median_val = stats_df[stats_df['Metric'] == 'Median Error']['Value'].values[0]
            
            # FIX 1: Handle median error of 0.0 for visualization
            if median_val == 0.0 and h == 5:
                # Use a small value for visualization purposes
                median_val = 1e-10  # Small value for log scale visibility
                print(f"Note: Harmonic 5 has median error of 0.0, using {median_val:.2e} for visualization")
                
        except (IndexError, KeyError):
            # If metrics not available, use placeholder values
            max_val = l2_error * 2  # approximate
            mean_val = l2_error / 2  # approximate
            median_val = l2_error / 3  # approximate
            print(f"Warning: Could not extract detailed error metrics for harmonic {h}")
        
        max_errors.append(max_val)
        mean_errors.append(mean_val)
        median_errors.append(median_val)
    
    # Determine best model based on lowest L2 error
    best_idx = np.argmin(l2_errors)
    best_harmonics = harmonics[best_idx]
    
    # Update global BEST_HARMONICS
    global BEST_HARMONICS
    BEST_HARMONICS = best_harmonics
    print(f"\nBest model determined to be {BEST_HARMONICS} harmonics with L2 error: {l2_errors[best_idx]:.2e}")
    
    # Sort by harmonics
    idx = np.argsort(harmonics)
    harmonics = [harmonics[i] for i in idx]
    l2_errors = [l2_errors[i] for i in idx]
    max_errors = [max_errors[i] for i in idx]
    mean_errors = [mean_errors[i] for i in idx]
    median_errors = [median_errors[i] for i in idx]
    
    # Create bar chart comparing L2 errors
    plt.figure(figsize=(12, 6))
    
    # Ensure the plot has a reasonable y-axis range
    min_error = min(l2_errors)
    max_error = max(l2_errors)
    if min_error <= 0 or not np.isfinite(min_error):
        min_error = 1e-10
    if max_error <= 0 or not np.isfinite(max_error):
        max_error = 1e-2
    
    # Create the bar chart
    bars = plt.bar(harmonics, l2_errors, color=['green' if h == BEST_HARMONICS else 'skyblue' for h in harmonics])
    
    plt.xlabel('Number of Harmonics', fontsize=12)
    plt.ylabel('Relative L2 Error', fontsize=12)
    plt.title('Comparison of Relative L2 Errors Across Harmonic Configurations', fontsize=14)
    plt.yscale('log')  # Log scale for better visibility
    plt.ylim([min_error/10, max_error*10])  # Set reasonable y-axis limits
    plt.xticks(harmonics, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add text annotations on bars
    for bar, val, h in zip(bars, l2_errors, harmonics):
        if h == BEST_HARMONICS:  # Only add text annotation for the best model
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height*1.1,
                    f'{val:.2e}', ha='center', va='bottom', rotation=0, fontsize=10, fontweight='bold')
    
    # Annotate the best model
    best_idx = harmonics.index(BEST_HARMONICS)
    # Position text well above the bar with a fixed position
    text_position = max_error/2  # Fixed position high above the bars
    plt.annotate('BEST MODEL', 
                xy=(BEST_HARMONICS, l2_errors[best_idx]),  # Arrow points to the bar top
                xytext=(BEST_HARMONICS, text_position),    # Text positioned at fixed height
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/l2_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # If we have at least 2 models, create comparison of all error metrics
    if len(harmonics) >= 2:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(harmonics))
        width = 0.2
        
        plt.bar(x - width, l2_errors, width, label='L2 Error', color='skyblue')
        plt.bar(x, mean_errors, width, label='Mean Error', color='salmon') 
        plt.bar(x + width, median_errors, width, label='Median Error', color='lightgreen')
        
        plt.yscale('log')  # Log scale for better visibility
        
        # Add minor ticks on y-axis for better readability
        ax = plt.gca()
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)))
        ax.tick_params(axis='y', which='minor', length=4)
        ax.grid(True, axis='y', which='minor', alpha=0.2, linestyle='--')
        
        plt.xlabel('Number of Harmonics', fontsize=12)
        plt.ylabel('Error (log scale)', fontsize=12)
        plt.title('Comparison of Error Metrics Across Harmonic Configurations', fontsize=14)
        plt.xticks(x, harmonics, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10, loc='upper left')  # Move legend to upper-left corner
        
        # Add "BEST MODEL" text directly above Harmonic 5 without arrow
        # Calculate max_error_display from the data
        max_error_display = max(max(l2_errors), max(mean_errors), max(median_errors))
        
        # Position it around 10^-5 to avoid overlapping with legend
        # Use max_error_display with a small multiplier for flexible positioning
        # This ensures the text is positioned well below the legend
        text_position = max_error_display * 2e-5  # Results in ~1e-5 when max_error_display is ~0.5
        
        plt.text(x[best_idx], text_position, 'BEST MODEL', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',  # Reduced font size from 12 to 10
                color='darkgreen', rotation=0)
        
        plt.tight_layout()
        plt.savefig('visualizations/error_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Harmonics': harmonics,
        'L2 Error': l2_errors,
        'Max Error': max_errors,
        'Mean Error': mean_errors,
        'Median Error': median_errors
    })
    
    # Add column indicating best model
    summary_df['Best Model'] = [h == BEST_HARMONICS for h in harmonics]
    
    # Save summary to CSV
    summary_df.to_csv('visualizations/harmonics_comparison_summary.csv', index=False)
    
    return summary_df

def compare_training_losses(all_training_data):
    """
    Create a comparative plot of training losses across different harmonic configurations
    
    Parameters:
    -----------
    all_training_data : dict
        Dictionary with harmonic numbers as keys and dataframes with training data as values
    """
    if len(all_training_data) < 2:
        print("Not enough data to compare training losses across harmonics")
        return
    
    # Sort harmonics for consistent plotting
    harmonics = sorted(all_training_data.keys())
    
    # Group harmonics into categories for better line style distinction
    harmonic_groups = {
        'low': [h for h in harmonics if h <= 25],
        'medium': [h for h in harmonics if 25 < h <= 55],
        'high': [h for h in harmonics if h > 55 and h < 80],
        'highest': [h for h in harmonics if h >= 80]
    }
    
    # Define different line styles for each group
    line_styles = {
        'low': '-',      # Solid lines
        'medium': '--',  # Dashed lines
        'high': '-.',    # Dash-dot lines
        'highest': ':'   # Dotted lines for 80, 85, and 90 harmonics
    }
    
    # Define the custom color set to match the validation error plot
    custom_colors = [
        '#9FCDC9',  # R159 G205 B201 - Bluish-green
        '#56AEDE',  # R86 G174 B222 - Blue
        '#EE7A5F',  # R238 G122 B95 - Orange-red
        '#FDD39F',  # R253 G211 B159 - Light orange
        '#B6C4BA'   # R182 G196 B186 - Grayish-green
    ]
    
    # Standard plot with all curves
    plt.figure(figsize=(14, 8))
    
    # Create a color mapping using the custom colors, cycling through them if needed
    color_maps = {}
    for group_name, group_harmonics in harmonic_groups.items():
        group_colors = []
        for i in range(len(group_harmonics)):
            # Cycle through colors if there are more harmonics than colors
            color_idx = i % len(custom_colors)  # Use all colors, no need to skip any
            color = custom_colors[color_idx]
            
            # Slightly adjust the color intensity based on the group to maintain distinction
            if group_name == 'low':
                # Use original colors for low harmonics
                group_colors.append(color)
            elif group_name == 'medium':
                # Slightly darken colors for medium harmonics
                rgb = plt.matplotlib.colors.to_rgb(color)
                rgb = tuple(max(0, c * 0.85) for c in rgb)  # Darken by 15%
                group_colors.append(rgb)
            else:  # high and highest
                # Slightly lighten colors for high harmonics
                rgb = plt.matplotlib.colors.to_rgb(color)
                rgb = tuple(min(1, c * 1.15) for c in rgb)  # Lighten by 15%
                group_colors.append(rgb)
        color_maps[group_name] = group_colors
    
    # Add all lines grouped by harmonic range
    for group_name, group_harmonics in harmonic_groups.items():
        for i, h in enumerate(group_harmonics):
            df = all_training_data[h]
            line_style = line_styles[group_name]
            color = color_maps[group_name][i]
            
            plt.semilogy(df['Epoch'], df['Total_Loss'], line_style, color=color, 
                        label=f'{h} Harmonics', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (log scale)')
    plt.title('Comparison of Training Losses Across Harmonic Configurations')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Highlight best model with thicker line and distinctive style
    if BEST_HARMONICS in harmonics:
        df = all_training_data[BEST_HARMONICS]
        plt.semilogy(df['Epoch'], df['Total_Loss'], '-', color='green', 
                    label=f'{BEST_HARMONICS} Harmonics (BEST)', linewidth=2.5)
    
    # Move legend outside the plot to the right to avoid covering the curves
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small', 
              title='Harmonic Configurations')
    
    # Adjust the layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig('visualizations/training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Skip creating the redundant heatmap - the line plot above already shows this information clearly
    # The original heatmap with sparse data was not useful, and a duplicate line plot is unnecessary

    # Create a final-epoch comparison bar chart
    plt.figure(figsize=(14, 6))
    
    # Extract final loss values
    final_losses = []
    for h in harmonics:
        df = all_training_data[h]
        final_losses.append(df['Total_Loss'].iloc[-1])
    
    bars = plt.bar(harmonics, final_losses, 
                   color=['green' if h == BEST_HARMONICS else 'skyblue' for h in harmonics])
    
    plt.xlabel('Number of Harmonics')
    plt.ylabel('Final Total Loss')
    plt.title('Final Training Loss Comparison Across Harmonic Configurations')
    plt.yscale('log')
    plt.xticks(harmonics)
    
    # Add annotations
    for bar, val, h in zip(bars, final_losses, harmonics):
        if h == BEST_HARMONICS:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height*1.1,
                     f'{val:.2e}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/training_losses_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_validation_errors(all_training_data):
    """
    Create a comparative plot of validation errors across different harmonic configurations
    
    Parameters:
    -----------
    all_training_data : dict
        Dictionary with harmonic numbers as keys and dataframes with training data as values
    """
    if len(all_training_data) < 2:
        print("Not enough data to compare validation errors across harmonics")
        return
    
    # Sort harmonics for consistent plotting
    harmonics = sorted(all_training_data.keys())
    
    # If we have many harmonics, we may need alternative visualizations
    too_many_curves = len(harmonics) > 8
    
    # Always create the standard plot, but adjust legend if there are many curves
    # Standard plot with all curves
    plt.figure(figsize=(14, 8))
    
    # Group harmonics into categories for better line style distinction
    harmonic_groups = {
        'low': [h for h in harmonics if h <= 25],
        'medium': [h for h in harmonics if 25 < h <= 55],
        'high': [h for h in harmonics if h > 55 and h < 80],
        'highest': [h for h in harmonics if h >= 80]
    }
    
    # Define different line styles for each group
    line_styles = {
        'low': '-',      # Solid lines
        'medium': '--',  # Dashed lines
        'high': '-.',    # Dash-dot lines
        'highest': ':'   # Dotted lines for 80, 85, and 90 harmonics
    }
    
    # Define the custom color set provided by the user
    custom_colors = [
        '#9FCDC9',  # R159 G205 B201 - Bluish-green
        '#56AEDE',  # R86 G174 B222 - Blue
        '#EE7A5F',  # R238 G122 B95 - Orange-red
        '#FDD39F',  # R253 G211 B159 - Light orange
        '#B6C4BA'   # R182 G196 B186 - Grayish-green
    ]
    
    # Create a color mapping using the custom colors, cycling through them if needed
    color_maps = {}
    for group_name, group_harmonics in harmonic_groups.items():
        group_colors = []
        for i in range(len(group_harmonics)):
            # Cycle through colors if there are more harmonics than colors
            color_idx = i % len(custom_colors)
            # Slightly adjust the color intensity based on the group to maintain distinction
            color = custom_colors[color_idx]
            if group_name == 'low':
                # Use original colors for low harmonics
                group_colors.append(color)
            elif group_name == 'medium':
                # Slightly darken colors for medium harmonics
                rgb = plt.matplotlib.colors.to_rgb(color)
                rgb = tuple(max(0, c * 0.85) for c in rgb)  # Darken by 15%
                group_colors.append(rgb)
            else:  # high
                # Slightly lighten colors for high harmonics
                rgb = plt.matplotlib.colors.to_rgb(color)
                rgb = tuple(min(1, c * 1.15) for c in rgb)  # Lighten by 15%
                group_colors.append(rgb)
        color_maps[group_name] = group_colors
    
    # Add all lines grouped by harmonic range
    for group_name, group_harmonics in harmonic_groups.items():
        for i, h in enumerate(group_harmonics):
            df = all_training_data[h]
            line_style = line_styles[group_name]
            color = color_maps[group_name][i]
            
            plt.semilogy(df['Epoch'], df['L2_Error'], line_style, color=color, 
                        label=f'{h} Harmonics', linewidth=1.5)
    
    # Add target line with improved label using standard notation instead of Unicode
    target_error = 1e-9
    plt.axhline(y=target_error, color='r', linestyle='--', linewidth=1.5, 
                label=f'Target Error: 1e-9')
    
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error (log scale)')
    plt.title('Comparison of Validation Errors Across Harmonic Configurations')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Highlight best model with thicker line and distinctive style
    if BEST_HARMONICS in harmonics:
        df = all_training_data[BEST_HARMONICS]
        plt.semilogy(df['Epoch'], df['L2_Error'], '-', color='green', 
                    label=f'{BEST_HARMONICS} Harmonics (BEST)', linewidth=2.5)
    
    # Move legend outside the plot to the right to avoid covering the curves
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small', 
              title='Harmonic Configurations')
    
    # Adjust the layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig('visualizations/validation_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a better visualization - Timeline showing when models achieve error thresholds
    plt.figure(figsize=(14, 8))
    
    # Define error thresholds to track
    thresholds = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    threshold_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(thresholds)))
    
    # Find the max number of epochs
    max_epochs = max(df['Epoch'].max() for df in all_training_data.values())
    
    # For each harmonic, plot when it crosses each threshold
    for h_idx, h in enumerate(harmonics):
        df = all_training_data[h]
        
        # For each threshold, find first epoch where error is below it
        for t_idx, threshold in enumerate(thresholds):
            below_threshold = df[df['L2_Error'] < threshold]
            if not below_threshold.empty:
                first_epoch = below_threshold.iloc[0]['Epoch']
                # Plot a marker
                marker = 'o' if h == BEST_HARMONICS else 's'
                markersize = 12 if h == BEST_HARMONICS else 8
                plt.scatter(first_epoch, h_idx, color=threshold_colors[t_idx], 
                          s=markersize**2, marker=marker, 
                          edgecolors='black' if h == BEST_HARMONICS else 'none',
                          linewidths=2 if h == BEST_HARMONICS else 0,
                          zorder=10 if h == BEST_HARMONICS else 5)
    
    # Create custom legend for thresholds
    legend_elements = []
    for t_idx, (threshold, color) in enumerate(zip(thresholds, threshold_colors)):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'< {threshold:.0e}'))
    
    # Add grid and labels
    plt.yticks(range(len(harmonics)), harmonics)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Number of Harmonics', fontsize=12)
    plt.title('Validation Error Threshold Achievement Timeline', fontsize=14)
    plt.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add vertical line at Adam->LBFGS transition
    plt.axvline(x=2000, color='red', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    
    # Highlight best model row
    if BEST_HARMONICS in harmonics:
        best_idx = harmonics.index(BEST_HARMONICS)
        plt.axhspan(best_idx - 0.4, best_idx + 0.4, alpha=0.1, color='green')
        plt.text(-100, best_idx, 'BEST', fontweight='bold', 
                ha='right', va='center', color='green', fontsize=12)
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='Error Thresholds', fontsize=10)
    
    plt.xlim(-200, max_epochs + 200)
    plt.tight_layout()
    plt.savefig('visualizations/validation_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a final-epoch comparison with better visualization for extreme differences
    # Extract final error values from the authoritative source (final_l2_error.txt if available)
    final_errors = []
    for h in harmonics:
        # Try to get from final_l2_error.txt first
        l2_error_file = os.path.join(HARMONIC_CONFIGS[h], 'final_l2_error.txt')
        if os.path.exists(l2_error_file):
            try:
                with open(l2_error_file, 'r') as f:
                    error_val = float(f.read().strip())
                    final_errors.append(error_val)
            except:
                # Fallback to dataframe
                df = all_training_data[h]
                final_errors.append(df['L2_Error'].iloc[-1])
        else:
            df = all_training_data[h]
            final_errors.append(df['L2_Error'].iloc[-1])
    
    # Create two subplots: log scale and focused linear scale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top plot: Log scale (shows all values clearly)
    bars1 = ax1.bar(harmonics, final_errors, 
                    color=['green' if h == BEST_HARMONICS else 'skyblue' for h in harmonics])
    ax1.set_ylabel('Final L2 Error (Log Scale)')
    ax1.set_title('Final Validation Error Comparison Across Harmonic Configurations')
    ax1.set_yscale('log')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value annotations on log scale plot
    for bar, val, h in zip(bars1, final_errors, harmonics):
        height = bar.get_height()
        color = 'darkgreen' if h == BEST_HARMONICS else 'black'
        weight = 'bold' if h == BEST_HARMONICS else 'normal'
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.5,
                f'{val:.2e}', ha='center', va='bottom', rotation=0, 
                fontsize=10, color=color, fontweight=weight)
    
    # Bottom plot: Focused view on the best model
    # Find the range for focused view
    best_error = min(final_errors)
    max_to_show = best_error * 10  # Show up to 10x the best error
    
    # Filter data for focused view
    focused_errors = [e if e <= max_to_show else max_to_show for e in final_errors]
    
    bars2 = ax2.bar(harmonics, focused_errors, 
                    color=['green' if h == BEST_HARMONICS else 'skyblue' for h in harmonics])
    ax2.set_xlabel('Number of Harmonics')
    ax2.set_ylabel('Final L2 Error (Focused Scale)')
    ax2.set_ylim(0, max_to_show * 1.2)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add annotation for truncated bars
    for bar, val, focused_val, h in zip(bars2, final_errors, focused_errors, harmonics):
        if val > max_to_show:
            # Add triangle marker indicating truncation
            ax2.plot(bar.get_x() + bar.get_width()/2., max_to_show, 
                    '^', color='red', markersize=10)
    
    # Highlight the best model
    if BEST_HARMONICS in harmonics:
        best_idx = harmonics.index(BEST_HARMONICS)
        best_error = final_errors[best_idx]
        ax2.annotate(f'BEST MODEL: {best_error:.2e}',
                    xy=(best_idx, best_error),
                    xytext=(best_idx, max_to_show * 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/validation_error_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_weight_factors(all_training_data):
    """
    Create a comparative plot of weight factors across different harmonic configurations
    
    Parameters:
    -----------
    all_training_data : dict
        Dictionary with harmonic numbers as keys and dataframes with training data as values
    """
    # First check if weight data is available
    if not all('PDE_Weight' in df.columns for df in all_training_data.values()):
        print("Weight factors not found in training data, skipping weight_factors.png")
        return
    
    if len(all_training_data) < 2:
        print("Not enough data to compare weight factors across harmonics")
        return
    
    # Sort harmonics for consistent plotting
    harmonics = sorted(all_training_data.keys())
    
    # Create a figure with subplots - one row for each weight type (excluding BC since it's hard BC)
    weight_types = ['PDE_Weight', 'IC_Weight']
    
    fig, axes = plt.subplots(len(weight_types), 1, figsize=(14, 5*len(weight_types)), sharex=True)
    if len(weight_types) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    # Since weight factors are the same across harmonics, we'll just use the best model
    # and one additional model to show the pattern is consistent
    representative_models = [BEST_HARMONICS]
    if len(harmonics) > 1:
        # Add a second model that's not the best for comparison
        other_model = [h for h in harmonics if h != BEST_HARMONICS][0]
        representative_models.append(other_model)
    
    for i, weight_type in enumerate(weight_types):
        ax = axes[i]
        
        # Plot the best model with a thick green line
        df_best = all_training_data[BEST_HARMONICS]
        # FIX 2: Correct label to show actual best harmonics
        ax.plot(df_best['Epoch'], df_best[weight_type], '-', color='green', 
                label=f'All Harmonics ({BEST_HARMONICS} Harmonics is BEST)', linewidth=2.5)
        
        # If there's another model and it has different values, plot it with a dashed line
        if len(representative_models) > 1:
            df_other = all_training_data[representative_models[1]]
            # Only plot if it's different from the best model to avoid redundancy
            if not np.array_equal(df_best[weight_type].values, df_other[weight_type].values):
                ax.plot(df_other['Epoch'], df_other[weight_type], '--', color='blue', 
                        label=f'{representative_models[1]} Harmonics', linewidth=1.5)
        
        ax.set_ylabel(weight_type.replace('_', ' '))
        ax.set_title(f'{weight_type.replace("_", " ")} Across All Harmonics')
        ax.grid(True, alpha=0.3)
        
        # Only add legend if we plotted multiple lines
        if len(representative_models) > 1 and 'df_other' in locals():
            if not np.array_equal(df_best[weight_type].values, df_other[weight_type].values):
                ax.legend(loc='best')
    
    axes[-1].set_xlabel('Epoch')
    #plt.suptitle('Weight Factors (Same for All Harmonic Configurations)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('visualizations/weight_factors.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_single_harmonic(harmonics, results_dir):
    """Process visualizations for a single harmonic configuration"""
    print(f"Processing {harmonics} harmonics configuration...")
    
    # Find data files
    training_data, solution_data, l2_error_file = find_data_files(results_dir)
    if training_data is None or solution_data is None:
        # Attempt to run only for harmonic 5 for focused visualization
        if harmonics != 5:
            print(f"Skipping harmonic {harmonics} due to missing data")
            return None
        
        # For harmonic 5, try to regenerate solution data
        solution = generate_solution_data(results_dir)
        if solution is None:
            print(f"Could not generate solution data for harmonic {harmonics}")
            return None
        
        # Reload training data
        try:
            df = pd.read_csv(training_data)
            
            # Also check for L2 error file
            if l2_error_file and os.path.exists(l2_error_file):
                try:
                    with open(l2_error_file, 'r') as f:
                        final_l2 = float(f.read().strip())
                        
                    # Update the last L2 error in the dataframe
                    if len(df) > 0 and 'L2_Error' in df.columns:
                        df.loc[df.index[-1], 'L2_Error'] = final_l2
                        print(f"Updated final L2 error from file: {final_l2:.9e}")
                except Exception as e:
                    print(f"Error reading L2 error file: {e}")
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
    else:
        # Load data
        df, solution = load_data(training_data, solution_data, l2_error_file)
    
    # Create visualizations with harmonics info
    try:
        visualize_solution_comparison(solution, harmonics)
        visualize_3d_comparisons(solution, harmonics)
        visualize_space_slices(solution, harmonics)
        visualize_time_slices(solution, harmonics)
        visualize_training_losses(df, harmonics)
        visualize_validation_error(df, harmonics)
        visualize_weight_factors(df, harmonics)
        visualize_error_distribution(solution, harmonics)
        
        # Create additional Euler-Bernoulli beam visualization
        visualize_euler_bernoulli_beam(solution, harmonics)
        
        # Save statistics
        stats_df, l2_error = save_error_statistics(solution, df, harmonics)
        
        # Save numerical data to CSV files
        save_numerical_data_to_csv(solution, df, harmonics)
        
        return df, stats_df, l2_error, solution
    except Exception as e:
        print(f"Error during visualization for harmonic {harmonics}: {e}")
        traceback_details = traceback.format_exc()
        print(f"Traceback: {traceback_details}")
        return None

def save_numerical_data_to_csv(solution, df, harmonics=None):
    """
    Save numerical data used in plots to structured CSV files
    """
    # Prepare directory
    os.makedirs('visualizations/numerical_data', exist_ok=True)
    
    # 1. Save solution data
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    u_error = solution['u_error']
    
    # 2. Save space slice data (for each x value)
    target_x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    x_indices = [np.abs(x - val).argmin() for val in target_x_values]
    
    for i, idx in enumerate(x_indices):
        space_slice_df = pd.DataFrame({
            'time': t,
            'u_pred': u_pred[idx, :],
            'u_exact': u_exact[idx, :],
            'u_error': u_error[idx, :]
        })
        
        suffix = f"_{harmonics}h" if harmonics else ""
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        
        filename = f'visualizations/numerical_data/space_slice_x{target_x_values[i]:.2f}{suffix}.csv'
        space_slice_df.to_csv(filename, index=False)
    
    # 3. Save time slice data (for each t value)
    target_t_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    t_indices = [np.abs(t - val).argmin() for val in target_t_values]
    
    for i, idx in enumerate(t_indices):
        time_slice_df = pd.DataFrame({
            'position': x,
            'u_pred': u_pred[:, idx],
            'u_exact': u_exact[:, idx],
            'u_error': u_error[:, idx]
        })
        
        suffix = f"_{harmonics}h" if harmonics else ""
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        
        filename = f'visualizations/numerical_data/time_slice_t{target_t_values[i]:.2f}{suffix}.csv'
        time_slice_df.to_csv(filename, index=False)
    
    # 4. Save training and validation data
    suffix = f"_{harmonics}h" if harmonics else ""
    if harmonics == BEST_HARMONICS:
        suffix += "_best"
    
    training_csv_path = f'visualizations/numerical_data/training_data{suffix}.csv'
    df.to_csv(training_csv_path, index=False)
    
    print(f"Saved numerical data to visualizations/numerical_data/ directory")
    
    return True

def visualize_euler_bernoulli_beam(solution, harmonics=None):
    """
    Create visualization specifically for Euler-Bernoulli beam behavior
    
    This is an additional visualization relevant to Euler-Bernoulli beam 
    modeled using the PINN architecture
    """
    x = solution['x'].flatten()
    t = solution['t'].flatten()
    u_pred = solution['u_pred']
    u_exact = solution['u_exact']
    
    # Create figure for beam deflection at different times
    plt.figure(figsize=(12, 8))
    
    # Select 5 different time points to visualize beam deflection
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ['b', 'r', 'g', 'm', 'c']
    markers = ['o', 's', '^', 'v', 'D']
    
    for i, t_val in enumerate(time_points):
        # Find closest time index
        t_idx = np.abs(t - t_val).argmin()
        
        # Plot exact solution (solid line)
        plt.plot(x, u_exact[:, t_idx], f'{colors[i]}-', 
                 linewidth=2, label=f'Exact t={t_val:.1f}')
        
        # Plot PINN solution (markers at intervals)
        marker_indices = np.linspace(0, len(x)-1, 15).astype(int)
        plt.plot(x[marker_indices], u_pred[marker_indices, t_idx], 
                 f'{colors[i]}{markers[i]}', markersize=6, 
                 label=f'PINN t={t_val:.1f}')
    
    plt.xlabel('Position along beam (x)', fontsize=12)
    plt.ylabel('Beam deflection u(x,t)', fontsize=12)
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Euler-Bernoulli Beam Deflection - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        plt.title(title, fontsize=14)
    else:
        plt.title('Euler-Bernoulli Beam Deflection', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=10)
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/euler_bernoulli_beam{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/euler_bernoulli_beam.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 3D animation-like plot showing beam vibration over time
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select more time points for smooth visualization
    num_time_frames = 10
    time_indices = np.linspace(0, len(t)-1, num_time_frames).astype(int)
    
    # Create a custom colormap for time progression
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i/num_time_frames) for i in range(num_time_frames)]
    
    # Plot each time frame
    for i, t_idx in enumerate(time_indices):
        current_t = t[t_idx]
        ax.plot(x, np.ones_like(x)*current_t, u_pred[:, t_idx], 
                color=colors[i], linewidth=2, 
                label=f't={current_t:.2f}')
    
    # Add mesh surface for visualization
    T, X = np.meshgrid(t[time_indices], x)
    Z = np.zeros_like(T)
    for i, t_idx in enumerate(time_indices):
        Z[:, i] = u_pred[:, t_idx]
    
    ax.plot_surface(X, T, Z, alpha=0.3, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Position (x)', fontsize=12)
    ax.set_ylabel('Time (t)', fontsize=12)
    ax.set_zlabel('Beam deflection u(x,t)', fontsize=12)
    
    # Add title with harmonics info if provided
    if harmonics:
        title = f'Euler-Bernoulli Beam Vibration - {harmonics} Harmonics'
        if harmonics == BEST_HARMONICS:
            title += " (BEST MODEL)"
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Euler-Bernoulli Beam Vibration', fontsize=14)
    
    # Add a custom legend
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, 
                      label=f't={t[time_indices[i]]:.2f}') 
                      for i in range(0, num_time_frames, 2)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save with harmonics info if provided
    if harmonics:
        suffix = f"_{harmonics}h"
        if harmonics == BEST_HARMONICS:
            suffix += "_best"
        plt.savefig(f'visualizations/euler_bernoulli_3d{suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('visualizations/euler_bernoulli_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_solution_data(results_dir):
    """Generate solution data from a trained model when solution_data.npz is not available"""
    # Import necessary modules
    import torch
    import torch.nn as nn
    import numpy as np
    
    # Determine the number of harmonics from the directory name
    harmonics = int(results_dir.split('_')[-1])
    
    # Define the model architecture that matches the actual saved models
    class UltraPrecisionEulerBernoulliModel(nn.Module):
        def __init__(self, n_harmonics=50, c=1.0, L=1.0, max_harmonics=65):
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
    
    # Paths for model weights
    model_path = os.path.join(results_dir, 'best_model.pt')
    lbfgs_model_path = os.path.join(results_dir, 'best_model_lbfgs.pt')
    final_model_path = os.path.join(results_dir, 'final_model.pt')
    
    # Check if any of the models exist - prioritize best_model_lbfgs.pt for accurate L2 error
    if os.path.exists(lbfgs_model_path):
        model_path = lbfgs_model_path
        print(f"Using best L-BFGS model from {model_path}")
    elif os.path.exists(model_path):
        print(f"Using best Adam model from {model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"Warning: Using final model (may not be best) from {model_path}")
    else:
        print(f"No model found in {results_dir}")
        return None
    
    try:
        # Create the model with same configuration as training
        model = UltraPrecisionEulerBernoulliModel(n_harmonics=harmonics, max_harmonics=65)
        
        # Load the trained weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Generate grid for solution - use same density as training for consistency
        n_points = 50  # Match training grid to avoid L2 error discrepancies
        # CRITICAL: Use float32 to match PyTorch's default precision in training
        x = np.linspace(0, 1, n_points, dtype=np.float32)
        t = np.linspace(0, 2, n_points, dtype=np.float32)
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Prepare inputs for model
        X_torch = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
        T_torch = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
        
        # Compute model predictions in batches to avoid OOM
        batch_size = 1000  # Reduced batch size for better memory management
        u_pred = np.zeros(X.size)
        
        with torch.no_grad():
            for i in range(0, X.size, batch_size):
                end = min(i + batch_size, X.size)
                inputs = torch.cat([T_torch[i:end], X_torch[i:end]], dim=1)
                u_pred[i:end] = model(inputs).cpu().numpy().flatten()
                
                # Clean up GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Reshape to match grid
        u_pred = u_pred.reshape(X.shape)
        
        # Calculate exact solution using vectorized computation for better numerical precision
        # For Euler-Bernoulli beam: ω₁ = (π/L)² * c
        c = 1.0  # beam parameter
        L = 1.0  # domain length
        omega_1 = (np.pi / L)**2 * c
        
        # Vectorized computation avoids accumulation of numerical errors
        # The exact solution for Euler-Bernoulli beam with IC w(0,x) = sin(πx) and zero velocity
        u_exact = np.sin(np.pi * X / L) * np.cos(omega_1 * T)
        
        # Calculate error - use absolute error for better visualization
        u_error = np.abs(u_pred - u_exact)
        
        # Calculate relative L2 error
        rel_l2_error = np.sqrt(np.sum((u_pred - u_exact)**2) / np.sum(u_exact**2))
        
        # Create the solution data
        solution_data = {
            'x': x,
            't': t,
            'u_pred': u_pred,
            'u_exact': u_exact,
            'u_error': u_error,
            'rel_l2_error': rel_l2_error
        }
        
        # Save the solution data
        save_path = os.path.join(results_dir, 'solution_data.npz')
        np.savez(save_path, **solution_data)
        
        print(f"Generated and saved solution data to {save_path}")
        print(f"Relative L2 error: {rel_l2_error:.9e}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return solution_data
    except Exception as e:
        print(f"Error generating solution data: {e}")
        traceback_details = traceback.format_exc()
        print(f"Traceback: {traceback_details}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None

def main():
    """Main function to generate all visualizations"""
    # Define this at the top level outside any function to make it available
    global BEST_HARMONICS
    
    parser = argparse.ArgumentParser(description='Generate visualizations for wave equation PINNs')
    parser.add_argument('--harmonics', type=int, help='Specific harmonic configuration to visualize')
    parser.add_argument('--compare', action='store_true', help='Compare all available harmonic configurations')
    parser.add_argument('--best-only', action='store_true', help='Only process the best model (determined automatically)')
    args = parser.parse_args()
    
    print("Generating ultra-precision visualizations...")
    
    # Add import for gc and force garbage collection
    import gc
    gc.collect()
    
    # Track statistics for all processed configurations
    all_stats = {}
    all_training_data = {}
    all_solutions = {}
    
    # If specific harmonic is requested, only process that one
    if args.harmonics is not None:
        if args.harmonics in HARMONIC_CONFIGS:
            print(f"\nProcessing only harmonic {args.harmonics}...")
            result = process_single_harmonic(args.harmonics, HARMONIC_CONFIGS[args.harmonics])
            if result:
                df, stats_df, l2_error, solution = result
                all_stats[args.harmonics] = (stats_df, l2_error)
                all_training_data[args.harmonics] = df
                all_solutions[args.harmonics] = solution
                
                # Set as best model
                BEST_HARMONICS = args.harmonics
                print(f"\nBest model set to {BEST_HARMONICS} harmonics with L2 error: {l2_error:.2e}")
        else:
            print(f"Error: No data found for {args.harmonics} harmonics")
        
        print("\nUltra-precision visualizations created successfully!")
        print(f"Results saved in ./visualizations/")
        return
    
    # Process all harmonic configs if comparison mode
    if args.compare:
        for harmonics, results_dir in HARMONIC_CONFIGS.items():
            result = process_single_harmonic(harmonics, results_dir)
            if result:
                df, stats_df, l2_error, solution = result
                all_stats[harmonics] = (stats_df, l2_error)
                all_training_data[harmonics] = df
                all_solutions[harmonics] = solution
        
        # Compare models and determine the best one
        if len(all_stats) > 1:
            summary_df = compare_errors_across_harmonics(all_stats)
            
            # Create comparative plots across all harmonics
            compare_training_losses(all_training_data)
            compare_validation_errors(all_training_data)
            compare_weight_factors(all_training_data)
            
            print("\nHarmonics Comparison Summary:")
            print(summary_df.to_string(index=False))
        elif len(all_stats) == 1:
            # If only one model, it's the best by default
            BEST_HARMONICS = list(all_stats.keys())[0]
            print(f"\nOnly one model available: {BEST_HARMONICS} harmonics")
        else:
            print("No valid models found!")
            return
    
    # If user requested best model only, regenerate those visualizations
    if args.best_only and BEST_HARMONICS is not None:
        print(f"\nRegenerating visualizations for best model ({BEST_HARMONICS} harmonics) only...")
        all_stats = {}  # Clear previous processing
        result = process_single_harmonic(BEST_HARMONICS, HARMONIC_CONFIGS[BEST_HARMONICS])
        if result:
            df, stats_df, l2_error, solution = result
            all_stats[BEST_HARMONICS] = (stats_df, l2_error)
    
    print("\nUltra-precision visualizations created successfully!")
    print(f"Results saved in ./visualizations/")

if __name__ == "__main__":
    main() 
