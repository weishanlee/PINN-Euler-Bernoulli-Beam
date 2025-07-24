import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
primary_color = '#56AEDE'  # Blue from color set 1
secondary_color = '#9FCDC9'  # Green from color set 1
accent_color = '#EE7A5F'  # Orange from color set 1
combine_color = '#FDD39F'  # Yellow from color set 1
output_color = '#B6C4BA'  # Light green from color set 1

# Define box style
box_style = dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='black', linewidth=2)
highlight_box_style = dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor=accent_color, linewidth=3)

# Input
input_box = FancyBboxPatch((0.5, 4.5), 1.5, 1, **box_style)
ax.add_patch(input_box)
ax.text(1.25, 5, r'$(t, x)$', ha='center', va='center', fontsize=14, weight='bold')

# Fourier Series Branch
fourier_box = FancyBboxPatch((3, 6.5), 3, 2, **highlight_box_style)
ax.add_patch(fourier_box)
ax.text(4.5, 7.5, 'Fourier Series\nExpansion', ha='center', va='center', fontsize=12, weight='bold')

# Learnable parameters A_n, B_n
param_box = FancyBboxPatch((3.25, 5.8), 2.5, 0.5, boxstyle="round,pad=0.05", 
                          facecolor=secondary_color, edgecolor='black', linewidth=1.5)
ax.add_patch(param_box)
ax.text(4.5, 6.05, r'$A_n, B_n$ (Learnable)', ha='center', va='center', fontsize=11, style='italic')

# Deep Neural Network Branch
nn_box = FancyBboxPatch((3, 1.5), 3, 2, **highlight_box_style)
ax.add_patch(nn_box)
ax.text(4.5, 2.5, 'Deep Neural\nNetwork', ha='center', va='center', fontsize=12, weight='bold')

# Neural network layers
layer_widths = [0.3, 0.5, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
layer_x_positions = np.linspace(3.3, 5.7, 8)
for i, (x, w) in enumerate(zip(layer_x_positions, layer_widths)):
    layer = Rectangle((x-w/2, 1.8), w, 0.4, facecolor=primary_color, edgecolor='black')
    ax.add_patch(layer)

# Add enlarged gradient notation on the left side
ax.text(2.5, 3, r'$\frac{\partial \mathcal{L}}{\partial A_n}$', ha='center', va='center', 
        fontsize=14, weight='bold', color='red', bbox=dict(boxstyle="round,pad=0.1", 
        facecolor='yellow', alpha=0.3))

ax.text(2.5, 2, r'$\frac{\partial \mathcal{L}}{\partial W_{NN}}$', ha='center', va='center', 
        fontsize=14, weight='bold', color='red', bbox=dict(boxstyle="round,pad=0.1", 
        facecolor='yellow', alpha=0.3))

# Combine block
combine_box = FancyBboxPatch((7, 3.5), 2, 2, boxstyle="round,pad=0.1", 
                            facecolor=combine_color, edgecolor='black', linewidth=2)
ax.add_patch(combine_box)
ax.text(8, 4.5, 'Combine\n$w = w_{Fourier} + \lambda \cdot w_{NN}$', 
        ha='center', va='center', fontsize=11, weight='bold')

# Boundary Condition block
bc_box = FancyBboxPatch((10, 3.75), 2, 1.5, **box_style)
ax.add_patch(bc_box)
ax.text(11, 4.5, r'BC: $\sin(\pi x/L)$', ha='center', va='center', fontsize=11, weight='bold')

# Physics-Informed Loss
loss_box = FancyBboxPatch((12.5, 3.5), 1.5, 2, boxstyle="round,pad=0.1", 
                         facecolor=output_color, edgecolor='black', linewidth=2)
ax.add_patch(loss_box)
ax.text(13.25, 4.5, 'Physics\nInformed\nLoss', ha='center', va='center', fontsize=11, weight='bold')

# Forward arrows (solid)
# Input to branches
ax.arrow(2, 5, 0.9, 2, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.arrow(2, 5, 0.9, -2.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Branches to combine
ax.arrow(6, 7.5, 0.9, -2.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.arrow(6, 2.5, 0.9, 2, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Combine to BC
ax.arrow(9, 4.5, 0.9, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# BC to Loss
ax.arrow(12, 4.5, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Backward arrows (dashed) - CORRECTED DIRECTION
# From Loss backward through the network
arrow_props = dict(arrowstyle='->', connectionstyle='arc3', color='red', 
                  linewidth=2, linestyle='dashed')

# Loss to BC (backward)
ax.annotate('', xy=(12, 5.5), xytext=(12.5, 5.5),
            arrowprops=dict(arrowstyle='<-', color='red', linewidth=2, linestyle='dashed'))

# BC to Combine (backward)
ax.annotate('', xy=(10, 5.5), xytext=(12, 5.5),
            arrowprops=dict(arrowstyle='<-', color='red', linewidth=2, linestyle='dashed'))

# Combine to branches (backward)
ax.annotate('', xy=(7, 5.5), xytext=(10, 5.5),
            arrowprops=dict(arrowstyle='<-', color='red', linewidth=2, linestyle='dashed'))

# Split to Fourier branch (backward)
ax.annotate('', xy=(5.5, 7), xytext=(7, 5.5),
            arrowprops=dict(arrowstyle='<-', connectionstyle='arc3,rad=0.3', 
                          color='red', linewidth=2, linestyle='dashed'))

# Split to NN branch (backward)
ax.annotate('', xy=(5.5, 2), xytext=(7, 3.5),
            arrowprops=dict(arrowstyle='<-', connectionstyle='arc3,rad=-0.3', 
                          color='red', linewidth=2, linestyle='dashed'))

# Add enlarged gradient labels near the backward arrows
ax.text(6.2, 7.5, r'$\nabla_{A_n,B_n}\mathcal{L}$', ha='center', va='center', 
        fontsize=13, weight='bold', color='red', rotation=-30)

ax.text(6.2, 1.5, r'$\nabla_{W_{NN}}\mathcal{L}$', ha='center', va='center', 
        fontsize=13, weight='bold', color='red', rotation=30)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Forward Pass'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='dashed', label='Backward Pass (Gradients)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Add title
ax.text(7, 9, 'Hybrid Fourier-PINN Architecture', ha='center', va='center', 
        fontsize=16, weight='bold')

# Add annotations
ax.text(7, 0.5, 'Note: Fourier coefficients $A_n, B_n$ are learnable parameters, NOT outputs from the NN', 
        ha='center', va='center', fontsize=11, style='italic', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))

# Add performance metric
ax.text(1, 0.5, r'L2 Error: $1.94 \times 10^{-7}$', ha='left', va='center', 
        fontsize=12, weight='bold', color='green',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/figures/pinn_architecture_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Architecture diagram has been redrawn with corrected backward propagation arrows and enlarged gradient notations.")