#!/usr/bin/env python3
"""
Create an infographic for the Ultra-Precision Physics-Informed Neural Networks paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 16))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color scheme - professional academic colors
colors = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#3498DB',    # Bright blue
    'accent1': '#E74C3C',      # Red
    'accent2': '#F39C12',      # Orange
    'accent3': '#27AE60',      # Green
    'light': '#ECF0F1',        # Light gray
    'white': '#FFFFFF'
}

# Title
title_box = FancyBboxPatch((5, 88), 90, 10, 
                          boxstyle="round,pad=0.5",
                          facecolor=colors['primary'],
                          edgecolor='none')
ax.add_patch(title_box)
ax.text(50, 93, 'Ultra-Precision Physics-Informed Neural Networks', 
        ha='center', va='center', fontsize=24, fontweight='bold', color='white')
ax.text(50, 90, 'Solving the Euler-Bernoulli Beam Equation with L2 Error < 10⁻⁷', 
        ha='center', va='center', fontsize=16, color='white')

# Problem Statement Box
prob_box = FancyBboxPatch((5, 72), 42.5, 13,
                         boxstyle="round,pad=0.5",
                         facecolor=colors['light'],
                         edgecolor=colors['primary'],
                         linewidth=2)
ax.add_patch(prob_box)
ax.text(26.25, 81, 'CHALLENGE', ha='center', va='center', fontsize=14, fontweight='bold', color=colors['primary'])
ax.text(26.25, 79, 'Fourth-order PDEs are notoriously', ha='center', va='center', fontsize=11)
ax.text(26.25, 77.5, 'difficult for neural networks', ha='center', va='center', fontsize=11)
ax.text(26.25, 75.5, 'Standard PINNs achieve only', ha='center', va='center', fontsize=11)
ax.text(26.25, 74, '10⁻³ to 10⁻⁴ accuracy', ha='center', va='center', fontsize=11, color=colors['accent1'])

# Solution Box
sol_box = FancyBboxPatch((52.5, 72), 42.5, 13,
                        boxstyle="round,pad=0.5",
                        facecolor=colors['accent3'],
                        edgecolor=colors['primary'],
                        linewidth=2,
                        alpha=0.9)
ax.add_patch(sol_box)
ax.text(73.75, 81, 'BREAKTHROUGH', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(73.75, 79, 'Hybrid Fourier-Neural Architecture', ha='center', va='center', fontsize=11, color='white')
ax.text(73.75, 77.5, 'Achieves L2 error of', ha='center', va='center', fontsize=11, color='white')
ax.text(73.75, 75.5, '1.94 × 10⁻⁷', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(73.75, 73.5, '17× improvement!', ha='center', va='center', fontsize=12, color='white')

# Key Innovation Section
key_y = 65
ax.text(50, key_y, 'KEY INNOVATIONS', ha='center', va='center', fontsize=16, fontweight='bold', color=colors['primary'])

# Three innovation boxes
innovations = [
    ('Hybrid Architecture', 'Fourier basis for\nperiodic structure\n+\nNeural network for\nfine corrections'),
    ('Two-Phase Training', 'Adam optimizer\nfor rapid progress\n+\nL-BFGS for\nultra-precision'),
    ('Optimal Harmonics', '10 harmonics\noptimal balance\n\nMore ≠ Better!')
]

for i, (title, desc) in enumerate(innovations):
    x = 15 + i * 30
    y = key_y - 8
    
    # Box
    box = FancyBboxPatch((x-10, y-5), 20, 10,
                        boxstyle="round,pad=0.5",
                        facecolor=colors['secondary'],
                        edgecolor='none',
                        alpha=0.9)
    ax.add_patch(box)
    
    # Number circle
    circle = Circle((x, y+3.5), 1.5, facecolor=colors['white'], edgecolor='none')
    ax.add_patch(circle)
    ax.text(x, y+3.5, str(i+1), ha='center', va='center', fontsize=14, fontweight='bold', color=colors['secondary'])
    
    # Text
    ax.text(x, y+1, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(x, y-2, desc, ha='center', va='center', fontsize=9, color='white', multialignment='center')

# Architecture Diagram
arch_y = 42
ax.text(50, arch_y, 'ARCHITECTURE OVERVIEW', ha='center', va='center', fontsize=16, fontweight='bold', color=colors['primary'])

# Input
input_box = Rectangle((10, arch_y-8), 15, 6, facecolor=colors['light'], edgecolor=colors['primary'], linewidth=2)
ax.add_patch(input_box)
ax.text(17.5, arch_y-5, 'Input\n(x, t)', ha='center', va='center', fontsize=11, fontweight='bold')

# Fourier Component
fourier_box = Rectangle((30, arch_y-4), 18, 6, facecolor=colors['accent2'], edgecolor='none', alpha=0.9)
ax.add_patch(fourier_box)
ax.text(39, arch_y-1, 'Fourier Series', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(39, arch_y-3, '10 harmonics', ha='center', va='center', fontsize=9, color='white')

# Neural Network Component
nn_box = Rectangle((30, arch_y-12), 18, 6, facecolor=colors['secondary'], edgecolor='none', alpha=0.9)
ax.add_patch(nn_box)
ax.text(39, arch_y-9, 'Neural Network', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(39, arch_y-11, '2→128→64→1', ha='center', va='center', fontsize=9, color='white')

# Combination
comb_box = Rectangle((53, arch_y-8), 15, 6, facecolor=colors['accent3'], edgecolor='none', alpha=0.9)
ax.add_patch(comb_box)
ax.text(60.5, arch_y-5, 'Hybrid\nSolution', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Output
output_box = Rectangle((73, arch_y-8), 17, 6, facecolor=colors['primary'], edgecolor='none')
ax.add_patch(output_box)
ax.text(81.5, arch_y-5, 'u(x,t)\nL2 < 10⁻⁷', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Arrows
arrows = [
    ((25, arch_y-5), (30, arch_y-1)),
    ((25, arch_y-5), (30, arch_y-9)),
    ((48, arch_y-1), (53, arch_y-5)),
    ((48, arch_y-9), (53, arch_y-5)),
    ((68, arch_y-5), (73, arch_y-5))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.1", 
                           arrowstyle='->', linewidth=2, color=colors['primary'])
    ax.add_patch(arrow)

# Results Section
results_y = 22
ax.text(50, results_y, 'BREAKTHROUGH RESULTS', ha='center', va='center', fontsize=16, fontweight='bold', color=colors['primary'])

# Results boxes
result_box1 = FancyBboxPatch((10, results_y-10), 35, 8,
                            boxstyle="round,pad=0.5",
                            facecolor=colors['accent1'],
                            edgecolor='none',
                            alpha=0.9)
ax.add_patch(result_box1)
ax.text(27.5, results_y-4, 'Accuracy Achievement', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(27.5, results_y-6, 'L2 Error: 1.94 × 10⁻⁷', ha='center', va='center', fontsize=11, color='white')
ax.text(27.5, results_y-8, '17× better than standard PINNs', ha='center', va='center', fontsize=10, color='white')

result_box2 = FancyBboxPatch((55, results_y-10), 35, 8,
                            boxstyle="round,pad=0.5",
                            facecolor=colors['accent3'],
                            edgecolor='none',
                            alpha=0.9)
ax.add_patch(result_box2)
ax.text(72.5, results_y-4, 'Computational Efficiency', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(72.5, results_y-6, 'GPU-accelerated', ha='center', va='center', fontsize=11, color='white')
ax.text(72.5, results_y-8, '< 30 min training time', ha='center', va='center', fontsize=10, color='white')

# Applications
app_y = 8
ax.text(50, app_y, 'APPLICATIONS & IMPACT', ha='center', va='center', fontsize=16, fontweight='bold', color=colors['primary'])

applications = ['Structural Mechanics', 'Quantum Simulations', 'Precision Engineering', 'Wave Detection']
for i, app in enumerate(applications):
    x = 12.5 + i * 22.5
    app_circle = Circle((x, app_y-4), 4, facecolor=colors['light'], edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(app_circle)
    ax.text(x, app_y-4, app, ha='center', va='center', fontsize=9, multialignment='center')

# Footer
ax.text(50, 1, 'Ultra-Precision PINNs: Bridging Classical Methods with Modern AI', 
        ha='center', va='center', fontsize=12, style='italic', color=colors['primary'])

plt.tight_layout()
plt.savefig('output/figures/infographic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Infographic created successfully: output/figures/infographic.png")