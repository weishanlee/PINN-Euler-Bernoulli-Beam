#!/usr/bin/env python3
"""
Create a workflow diagram for the ultra-precision PINN methodology
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, ConnectionPatch
import matplotlib.lines as mlines
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color scheme
colors = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'accent1': '#2ca02c',      # Green
    'accent2': '#d62728',      # Red
    'accent3': '#9467bd',      # Purple
    'light': '#f0f0f0',        # Light gray
    'dark': '#333333'          # Dark gray
}

# Title
ax.text(50, 95, 'Ultra-Precision PINN Workflow', 
        ha='center', va='center', fontsize=20, fontweight='bold', color=colors['dark'])

# Main workflow boxes
y_positions = [82, 68, 54, 40, 26, 12]
main_steps = [
    ('Input Definition', 'Problem domain\n[0,T] × [0,L]\nInitial & boundary conditions'),
    ('Hybrid Architecture', 'Fourier series (10 harmonics)\n+\nNeural network (27,905 params)'),
    ('Loss Function', 'PDE residual + IC + BC\nAdaptive weight balancing'),
    ('Phase 1: Adam', '2000 iterations\nLR scheduling\nGradient clipping'),
    ('Phase 2: L-BFGS', '5000 iterations\nQuasi-Newton method\nTolerance: 10⁻⁹'),
    ('Ultra-Precision Solution', 'L2 error < 10⁻⁷\nValidation & analysis')
]

# Draw main workflow
for i, (y, (title, desc)) in enumerate(zip(y_positions, main_steps)):
    # Main box
    if i == 0:
        color = colors['primary']
    elif i == len(main_steps) - 1:
        color = colors['accent1']
    elif i in [3, 4]:
        color = colors['secondary']
    else:
        color = colors['accent3']
    
    box = FancyBboxPatch((25, y-4), 50, 8,
                        boxstyle="round,pad=0.5",
                        facecolor=color,
                        edgecolor='none',
                        alpha=0.8)
    ax.add_patch(box)
    
    # Title
    ax.text(50, y+1.5, title, ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white')
    
    # Description
    ax.text(50, y-2, desc, ha='center', va='center', 
            fontsize=10, color='white', multialignment='center')
    
    # Arrow to next step
    if i < len(main_steps) - 1:
        arrow = FancyArrowPatch((50, y-4.5), (50, y_positions[i+1]+4),
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', linewidth=2.5, 
                               color=colors['dark'], alpha=0.7)
        ax.add_patch(arrow)

# Side annotations
# Fourier component details
fourier_box = FancyBboxPatch((5, 64), 15, 8,
                            boxstyle="round,pad=0.3",
                            facecolor=colors['light'],
                            edgecolor=colors['primary'],
                            linewidth=1.5)
ax.add_patch(fourier_box)
ax.text(12.5, 69, 'Fourier Basis', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=colors['primary'])
ax.text(12.5, 67, 'sin(kₙx)', ha='center', va='center', fontsize=9)
ax.text(12.5, 65.5, 'cos(ωₙt)', ha='center', va='center', fontsize=9)

# Neural network details
nn_box = FancyBboxPatch((80, 64), 15, 8,
                       boxstyle="round,pad=0.3",
                       facecolor=colors['light'],
                       edgecolor=colors['accent3'],
                       linewidth=1.5)
ax.add_patch(nn_box)
ax.text(87.5, 69, 'Neural Net', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=colors['accent3'])
ax.text(87.5, 67, '7 layers', ha='center', va='center', fontsize=9)
ax.text(87.5, 65.5, 'tanh activation', ha='center', va='center', fontsize=9)

# Loss components
loss_y = 54
loss_components = ['PDE', 'IC', 'BC', 'Reg']
loss_x_positions = [10, 20, 85, 90]
for x, comp in zip(loss_x_positions, loss_components):
    circle = Circle((x, loss_y), 3, facecolor=colors['light'], 
                   edgecolor=colors['accent3'], linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, loss_y, comp, ha='center', va='center', fontsize=9, fontweight='bold')

# Optimization details
opt_y = 33
# Adam details
adam_box = Rectangle((5, opt_y-3), 15, 6, facecolor=colors['light'], 
                    edgecolor=colors['secondary'], linewidth=1.5)
ax.add_patch(adam_box)
ax.text(12.5, opt_y+1.5, 'Adam Details', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=colors['secondary'])
ax.text(12.5, opt_y, 'LR: 0.01→10⁻⁴', ha='center', va='center', fontsize=8)
ax.text(12.5, opt_y-1.5, 'Batch: dynamic', ha='center', va='center', fontsize=8)

# L-BFGS details
lbfgs_box = Rectangle((80, opt_y-3), 15, 6, facecolor=colors['light'], 
                     edgecolor=colors['secondary'], linewidth=1.5)
ax.add_patch(lbfgs_box)
ax.text(87.5, opt_y+1.5, 'L-BFGS Details', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=colors['secondary'])
ax.text(87.5, opt_y, 'Full batch', ha='center', va='center', fontsize=8)
ax.text(87.5, opt_y-1.5, 'Line search', ha='center', va='center', fontsize=8)

# Connect side boxes to main flow
# Fourier to architecture
arrow1 = FancyArrowPatch((20, 68), (25, 68),
                        connectionstyle="arc3,rad=0.2", 
                        arrowstyle='->', linewidth=1.5, 
                        color=colors['primary'], alpha=0.5)
ax.add_patch(arrow1)

# NN to architecture
arrow2 = FancyArrowPatch((80, 68), (75, 68),
                        connectionstyle="arc3,rad=-0.2", 
                        arrowstyle='->', linewidth=1.5, 
                        color=colors['accent3'], alpha=0.5)
ax.add_patch(arrow2)

# Loss components to loss
for x in loss_x_positions[:2]:
    arrow = FancyArrowPatch((x+3, loss_y), (25, loss_y),
                           connectionstyle="arc3,rad=0.1", 
                           arrowstyle='->', linewidth=1.5, 
                           color=colors['accent3'], alpha=0.3)
    ax.add_patch(arrow)

for x in loss_x_positions[2:]:
    arrow = FancyArrowPatch((x-3, loss_y), (75, loss_y),
                           connectionstyle="arc3,rad=-0.1", 
                           arrowstyle='->', linewidth=1.5, 
                           color=colors['accent3'], alpha=0.3)
    ax.add_patch(arrow)

# Adam details to Phase 1
arrow3 = FancyArrowPatch((20, opt_y), (25, 40),
                        connectionstyle="arc3,rad=0.3", 
                        arrowstyle='->', linewidth=1.5, 
                        color=colors['secondary'], alpha=0.5)
ax.add_patch(arrow3)

# L-BFGS details to Phase 2
arrow4 = FancyArrowPatch((80, opt_y), (75, 26),
                        connectionstyle="arc3,rad=-0.3", 
                        arrowstyle='->', linewidth=1.5, 
                        color=colors['secondary'], alpha=0.5)
ax.add_patch(arrow4)

# Key insights box
insights_box = FancyBboxPatch((25, 2), 50, 6,
                             boxstyle="round,pad=0.5",
                             facecolor=colors['accent1'],
                             edgecolor='none',
                             alpha=0.2)
ax.add_patch(insights_box)
ax.text(50, 6, 'Key Innovation: Optimal balance at 10 harmonics', 
        ha='center', va='center', fontsize=11, fontweight='bold', color=colors['accent1'])
ax.text(50, 4, 'Fourier captures physics → Neural network adds precision', 
        ha='center', va='center', fontsize=10, color=colors['dark'])

plt.tight_layout()
plt.savefig('output/figures/workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Workflow diagram created successfully: output/figures/workflow_diagram.png")