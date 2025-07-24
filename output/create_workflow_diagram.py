#!/usr/bin/env python3
"""
Create workflow diagram for the Ultra-Precision PINN implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define color scheme
colors = {
    'input': '#9FCDC9',        # Mint green
    'process': '#56AEDE',      # Light blue
    'output': '#EE7A5F',       # Coral
    'decision': '#FDD39F',     # Light orange
    'dark': '#2C3E50',         # Dark blue-gray
}

# Title
ax.text(7, 9.5, 'Ultra-Precision PINN Workflow', fontsize=20, weight='bold', ha='center')

# Step 1: Input/Initialization
init_box = FancyBboxPatch((0.5, 7.5), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['input'],
                         edgecolor=colors['dark'],
                         linewidth=2)
ax.add_patch(init_box)
ax.text(2, 8.3, 'Initialization', fontsize=11, weight='bold', ha='center')
ax.text(2, 7.9, 'Harmonics: N', fontsize=9, ha='center')
ax.text(2, 7.6, 'GPU Lock Acquired', fontsize=9, ha='center')

# Step 2: Memory Check
memory_box = FancyBboxPatch((4.5, 7.5), 3, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['decision'],
                           edgecolor=colors['dark'],
                           linewidth=2)
ax.add_patch(memory_box)
ax.text(6, 8.3, 'Memory Feasibility', fontsize=11, weight='bold', ha='center')
ax.text(6, 7.9, 'Estimate Requirements', fontsize=9, ha='center')
ax.text(6, 7.6, 'Adjust Batch Size', fontsize=9, ha='center')

# Step 3: Model Creation
model_box = FancyBboxPatch((8.5, 7.5), 3.5, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['process'],
                          edgecolor=colors['dark'],
                          linewidth=2)
ax.add_patch(model_box)
ax.text(10.25, 8.3, 'Hybrid Model', fontsize=11, weight='bold', ha='center')
ax.text(10.25, 7.9, 'Fourier (130 params)', fontsize=9, ha='center')
ax.text(10.25, 7.6, 'NN (27,905 params)', fontsize=9, ha='center')

# Step 4: Training Data
data_box = FancyBboxPatch((0.5, 5.5), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['input'],
                         edgecolor=colors['dark'],
                         linewidth=2)
ax.add_patch(data_box)
ax.text(2, 6.3, 'Training Data', fontsize=11, weight='bold', ha='center')
ax.text(2, 5.9, '50×50 Grid Points', fontsize=9, ha='center')
ax.text(2, 5.6, 'PDE, IC, BC Points', fontsize=9, ha='center')

# Step 5: Phase 1 - Adam
adam_box = FancyBboxPatch((4.5, 5.5), 3.5, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['process'],
                         edgecolor=colors['dark'],
                         linewidth=2)
ax.add_patch(adam_box)
ax.text(6.25, 6.3, 'Phase 1: Adam', fontsize=11, weight='bold', ha='center', color='white')
ax.text(6.25, 5.9, '2000 iterations', fontsize=9, ha='center', color='white')
ax.text(6.25, 5.6, 'LR: 0.01, Adaptive', fontsize=9, ha='center', color='white')

# Step 6: Phase 2 - L-BFGS
lbfgs_box = FancyBboxPatch((8.5, 5.5), 3.5, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['process'],
                          edgecolor=colors['dark'],
                          linewidth=2)
ax.add_patch(lbfgs_box)
ax.text(10.25, 6.3, 'Phase 2: L-BFGS', fontsize=11, weight='bold', ha='center', color='white')
ax.text(10.25, 5.9, '5000 iterations', fontsize=9, ha='center', color='white')
ax.text(10.25, 5.6, 'Quasi-Newton', fontsize=9, ha='center', color='white')

# Step 7: Loss Components
loss_box = FancyBboxPatch((2, 3.5), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['decision'],
                         edgecolor=colors['dark'],
                         linewidth=2)
ax.add_patch(loss_box)
ax.text(4, 4.3, 'Composite Loss', fontsize=11, weight='bold', ha='center')
ax.text(4, 3.9, 'PDE + IC + BC + Reg', fontsize=9, ha='center')
ax.text(4, 3.6, 'Adaptive Weighting', fontsize=9, ha='center')

# Step 8: Convergence Check
conv_box = FancyBboxPatch((7, 3.5), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['decision'],
                         edgecolor=colors['dark'],
                         linewidth=2)
ax.add_patch(conv_box)
ax.text(9, 4.3, 'Convergence?', fontsize=11, weight='bold', ha='center')
ax.text(9, 3.9, 'L2 Error < 10⁻⁷', fontsize=9, ha='center')
ax.text(9, 3.6, 'Gradient < 10⁻⁹', fontsize=9, ha='center')

# Step 9: Output
output_box = FancyBboxPatch((4.5, 1.5), 5, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['output'],
                           edgecolor=colors['dark'],
                           linewidth=2)
ax.add_patch(output_box)
ax.text(7, 2.3, 'Ultra-Precision Solution', fontsize=11, weight='bold', ha='center', color='white')
ax.text(7, 1.9, 'Best Model Saved', fontsize=9, ha='center', color='white')
ax.text(7, 1.6, 'L2 Error: 1.94×10⁻⁷', fontsize=9, ha='center', color='white')

# Add arrows
arrows = [
    # Horizontal flow
    ((3.5, 8.1), (4.5, 8.1)),    # Init -> Memory
    ((7.5, 8.1), (8.5, 8.1)),    # Memory -> Model
    ((3.5, 6.1), (4.5, 6.1)),    # Data -> Adam
    ((8, 6.1), (8.5, 6.1)),      # Adam -> L-BFGS
    
    # Vertical flow
    ((2, 7.5), (2, 6.7)),        # Init -> Data
    ((6.25, 5.5), (4, 4.7)),     # Adam -> Loss
    ((10.25, 5.5), (9, 4.7)),    # L-BFGS -> Loss
    ((6, 3.5), (7, 3.8)),        # Loss -> Conv
    ((9, 3.5), (7, 2.7)),        # Conv -> Output
    
    # Feedback loop
    ((9, 4.7), (6.25, 5.5)),     # Conv back to Adam (curved)
]

for start, end in arrows:
    if start[1] == end[1]:  # Horizontal arrow
        arrow = FancyArrowPatch(start, end,
                               connectionstyle="arc3,rad=0",
                               arrowstyle="->,head_width=0.3,head_length=0.2",
                               color=colors['dark'], linewidth=2)
    else:  # Vertical or diagonal arrow
        arrow = FancyArrowPatch(start, end,
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle="->,head_width=0.3,head_length=0.2",
                               color=colors['dark'], linewidth=1.5)
    ax.add_patch(arrow)

# Add legend
legend_elements = [
    mlines.Line2D([0], [0], marker='s', color='w', label='Input/Data',
                  markerfacecolor=colors['input'], markersize=10),
    mlines.Line2D([0], [0], marker='s', color='w', label='Processing',
                  markerfacecolor=colors['process'], markersize=10),
    mlines.Line2D([0], [0], marker='s', color='w', label='Decision',
                  markerfacecolor=colors['decision'], markersize=10),
    mlines.Line2D([0], [0], marker='s', color='w', label='Output',
                  markerfacecolor=colors['output'], markersize=10),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add annotations
ax.text(12.5, 3.8, 'No', fontsize=9, style='italic', color=colors['dark'])
ax.text(7.5, 4.5, 'Yes', fontsize=9, style='italic', color=colors['dark'])

# Save the figure
plt.tight_layout()
plt.savefig('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/workflow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Workflow diagram created successfully!")