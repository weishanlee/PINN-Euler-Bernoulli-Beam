"""
Generate PINN Architecture Diagram with Backward Propagation Arrows
Shows both forward pass (solid arrows) and backward propagation (dashed arrows)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np

# Set style and create figure
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 17)
ax.axis('off')

# Colors
primary_color = '#56AEDE'  # Blue
secondary_color = '#9FCDC9'  # Teal
accent_color = '#EE7A5F'    # Orange
highlight_color = '#FDD39F' # Light orange
support_color = '#B6C4BA'   # Gray-green
dark_color = '#2C3E50'      # Dark blue-gray
gradient_color = '#E74C3C'  # Red for gradients

# Title
ax.text(6.5, 16.5, 'Hybrid Fourier-PINN Architecture', 
        fontsize=24, weight='bold', ha='center')
ax.text(6.5, 15.9, 'Forward Pass and Backward Propagation Flow', 
        fontsize=18, ha='center')

# Function to draw a neuron
def draw_neuron(x, y, color='white', radius=0.18, edge_color='black', linewidth=2):
    circle = Circle((x, y), radius, facecolor=color, edgecolor=edge_color, linewidth=linewidth)
    ax.add_patch(circle)
    return circle

# Function to draw connections between layers
def draw_connections(layer1_positions, layer2_positions, color='gray', alpha=0.3, linewidth=1):
    for (x1, y1) in layer1_positions:
        for (x2, y2) in layer2_positions:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)

# 1. INPUT LAYER (Top)
input_y = 14.5
input_positions = [(5.5, input_y), (7.5, input_y)]
for pos in input_positions:
    draw_neuron(pos[0], pos[1], color=primary_color)
ax.text(6.5, input_y + 0.6, 'Input Layer', fontsize=18, ha='center', weight='bold')
ax.text(5.5, input_y - 0.5, 't', fontsize=14, ha='center', weight='bold')
ax.text(7.5, input_y - 0.5, 'x', fontsize=14, ha='center', weight='bold')

# 2. FOURIER SERIES BRANCH (Left side)
fourier_box = FancyBboxPatch((0.5, 10.5), 4, 2.7, 
                            boxstyle="round,pad=0.1",
                            facecolor=secondary_color,
                            edgecolor='black',
                            linewidth=2.5)
ax.add_patch(fourier_box)
ax.text(2.5, 12.6, 'Fourier Series', fontsize=16, ha='center', weight='bold')
ax.text(2.5, 12.1, 'Expansion', fontsize=16, ha='center', weight='bold')
ax.text(2.5, 11.5, r'$\sum_{n=1}^{10} [A_n\cos(\omega_n t)$', fontsize=13, ha='center')
ax.text(2.5, 11.1, r'$+ B_n\sin(\omega_n t)]$', fontsize=13, ha='center')
ax.text(2.5, 10.7, r'$\times \sin(k_n x)$', fontsize=13, ha='center')

# Add learnable parameters note
ax.text(2.5, 10.2, r'$A_n, B_n$ are learnable', fontsize=10, ha='center', style='italic', color='darkred')

# 3. NEURAL NETWORK (Right side, vertical layout)
# Define layer positions
nn_center_x = 9.5
layer_heights = [2, 128, 128, 64, 32, 16, 8, 1]
layer_y_positions = np.linspace(13, 5, len(layer_heights))

# Draw NN background box
nn_box = FancyBboxPatch((nn_center_x-2.8, 4.5), 5.6, 9, 
                        boxstyle="round,pad=0.1",
                        facecolor='#FFF5F0',
                        edgecolor=accent_color,
                        linewidth=3)
ax.add_patch(nn_box)
ax.text(nn_center_x, 13.8, 'Deep Neural Network', 
        fontsize=16, ha='center', weight='bold', color=accent_color)

# Draw neurons for each layer
layer_neurons = []
for i, (y, width) in enumerate(zip(layer_y_positions, layer_heights)):
    neurons = []
    
    if width <= 8:  # Show all neurons for small layers
        x_positions = np.linspace(nn_center_x - min(width*0.2, 1.2), 
                                 nn_center_x + min(width*0.2, 1.2), width)
        for x in x_positions:
            # First layer (input) should match the actual input neurons
            if i == 0:
                # Two input neurons for the neural network
                if len(x_positions) == 2:
                    neuron = draw_neuron(x, y, color=primary_color, radius=0.18)
                    neurons.append((x, y))
            else:
                neuron = draw_neuron(x, y, color='white', radius=0.18)
                neurons.append((x, y))
    else:  # Show subset for large layers
        # Show 5 representative neurons
        x_positions = np.linspace(nn_center_x - 1.5, nn_center_x + 1.5, 5)
        for j, x in enumerate(x_positions):
            if j == 2:  # Middle position
                ax.text(x, y, '···', fontsize=18, ha='center', va='center', weight='bold')
            else:
                draw_neuron(x, y, color='white', radius=0.18)
                neurons.append((x, y))
    
    # Add layer label
    ax.text(nn_center_x + 3.2, y, f'{width}', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))
    
    layer_neurons.append(neurons)

# Draw connections between layers
for i in range(len(layer_neurons) - 1):
    if len(layer_neurons[i]) <= 8 and len(layer_neurons[i+1]) <= 8:
        draw_connections(layer_neurons[i], layer_neurons[i+1], color=dark_color, alpha=0.4)
    else:
        # Show subset of connections
        sample_from = layer_neurons[i][::max(1, len(layer_neurons[i])//3)]
        sample_to = layer_neurons[i+1][::max(1, len(layer_neurons[i+1])//3)]
        draw_connections(sample_from, sample_to, color=dark_color, alpha=0.25)

# Add activation function label (right of output neuron)
ax.text(nn_center_x + 2.5, 5, 'Activation: tanh', fontsize=13, ha='left', 
        style='italic', color=accent_color, weight='bold')

# 4. COMBINATION (Middle, between Fourier and NN)
combine_y = 3.5
combine_x = 6.5
# Draw combination box
combine_box = FancyBboxPatch((combine_x-2, combine_y-1), 4, 2, 
                            boxstyle="round,pad=0.1",
                            facecolor=support_color,
                            edgecolor='black',
                            linewidth=2.5)
ax.add_patch(combine_box)
draw_neuron(combine_x, combine_y, color='white', radius=0.3)
ax.text(combine_x, combine_y, '+', fontsize=24, ha='center', va='center', weight='bold')
ax.text(combine_x, combine_y + 0.6, 'Combine', fontsize=16, ha='center', weight='bold')

# Add boundary enforcement text next to NN output
ax.text(nn_center_x, 4.2, r'$\times \sin(\pi x/L)$', fontsize=13, ha='center', weight='bold')

# 5. PHYSICS-INFORMED LOSS (Bottom)
loss_y = 1.5
loss_box = FancyBboxPatch((3.5, -0.5), 6, 3, 
                         boxstyle="round,pad=0.1",
                         facecolor='#F5E6E0',
                         edgecolor='black',
                         linewidth=2.5)
ax.add_patch(loss_box)
ax.text(6.5, 2.3, 'Physics-Informed Loss', fontsize=16, ha='center', weight='bold')

# Add equations in the loss box
ax.text(6.5, 1.7, r'$\mathcal{L} = w_{PDE}\mathcal{L}_{PDE} + w_{IC}\mathcal{L}_{IC} + w_{BC}\mathcal{L}_{BC} + w_{IC_v}\mathcal{L}_{IC_v}$', 
        fontsize=13, ha='center')
ax.text(6.5, 1.1, r'$\mathcal{L}_{PDE} = \frac{1}{N_{PDE}}\sum_{i} \left|\frac{\partial^2 u}{\partial t^2} + c^2\frac{\partial^4 u}{\partial x^4}\right|^2$', 
        fontsize=12, ha='center')
ax.text(6.5, 0.6, r'$\mathcal{L}_{IC} = \frac{1}{N_{IC}}\sum_{i} |u(0,x_i) - u_0(x_i)|^2$', 
        fontsize=12, ha='center')
ax.text(6.5, 0.1, r'$\mathcal{L}_{BC} = \frac{1}{N_{BC}}\sum_{i} |u(t_i,0)|^2 + |u(t_i,L)|^2$', 
        fontsize=12, ha='center')
ax.text(6.5, -0.3, 'Adaptive Weight Balancing', fontsize=11, ha='center', style='italic')

# FORWARD CONNECTIONS (Solid arrows)
# Both inputs to Fourier Series
arrow1a = FancyArrowPatch((5.5, input_y-0.2), (2.0, 13.2),
                        connectionstyle="arc3,rad=-0.3",
                        arrowstyle='-|>',
                        mutation_scale=25,
                        linewidth=2.5,
                        color=dark_color)
ax.add_patch(arrow1a)

arrow1b = FancyArrowPatch((7.5, input_y-0.2), (3.0, 13.2),
                        connectionstyle="arc3,rad=-0.3",
                        arrowstyle='-|>',
                        mutation_scale=25,
                        linewidth=2.5,
                        color=dark_color)
ax.add_patch(arrow1b)

# Both inputs to NN (connect to the input neurons of NN)
nn_input_neurons = layer_neurons[0]  # First layer of NN
if len(nn_input_neurons) >= 2:
    # Connect t to first NN input neuron
    arrow2a = FancyArrowPatch((5.5, input_y-0.2), nn_input_neurons[0],
                            connectionstyle="arc3,rad=0.3",
                            arrowstyle='-|>',
                            mutation_scale=25,
                            linewidth=2.5,
                            color=dark_color)
    ax.add_patch(arrow2a)
    
    # Connect x to second NN input neuron
    arrow2b = FancyArrowPatch((7.5, input_y-0.2), nn_input_neurons[1],
                            connectionstyle="arc3,rad=0.3",
                            arrowstyle='-|>',
                            mutation_scale=25,
                            linewidth=2.5,
                            color=dark_color)
    ax.add_patch(arrow2b)

# Fourier to Combine
arrow3 = FancyArrowPatch((2.5, 10.5), (combine_x-1.5, combine_y+0.8),
                        connectionstyle="arc3,rad=0.3",
                        arrowstyle='-|>',
                        mutation_scale=25,
                        linewidth=2.5,
                        color=dark_color)
ax.add_patch(arrow3)

# NN output neuron to Combine
# Get the position of the output neuron (last layer with 1 neuron)
output_neuron_pos = layer_neurons[-1][0] if layer_neurons[-1] else (nn_center_x, 5)
arrow4 = FancyArrowPatch(output_neuron_pos, (combine_x+1.5, combine_y+0.8),
                        connectionstyle="arc3,rad=-0.3",
                        arrowstyle='-|>',
                        mutation_scale=25,
                        linewidth=2.5,
                        color=dark_color)
ax.add_patch(arrow4)

# Combine to Loss
arrow5 = FancyArrowPatch((combine_x, combine_y-1), (6.5, 2.5),
                        arrowstyle='-|>',
                        mutation_scale=25,
                        linewidth=2.5,
                        color=dark_color)
ax.add_patch(arrow5)

# BACKWARD PROPAGATION ARROWS (Dashed arrows in gradient color)
# Loss to Combine
back_arrow1 = FancyArrowPatch((6.5, 2.5), (combine_x, combine_y-1),
                            arrowstyle='<|-',
                            mutation_scale=20,
                            linewidth=2.5,
                            color=gradient_color,
                            linestyle='dashed',
                            alpha=0.8)
ax.add_patch(back_arrow1)

# Combine to Fourier (gradients flow to A_n, B_n)
back_arrow2 = FancyArrowPatch((combine_x-1.5, combine_y+0.8), (2.5, 10.5),
                            connectionstyle="arc3,rad=0.3",
                            arrowstyle='<|-',
                            mutation_scale=20,
                            linewidth=2.5,
                            color=gradient_color,
                            linestyle='dashed',
                            alpha=0.8)
ax.add_patch(back_arrow2)

# Combine to NN output
back_arrow3 = FancyArrowPatch((combine_x+1.5, combine_y+0.8), output_neuron_pos,
                            connectionstyle="arc3,rad=-0.3",
                            arrowstyle='<|-',
                            mutation_scale=20,
                            linewidth=2.5,
                            color=gradient_color,
                            linestyle='dashed',
                            alpha=0.8)
ax.add_patch(back_arrow3)

# Add gradient flow annotations
ax.text(5.0, 1.8, r'$\nabla_\theta \mathcal{L}$', fontsize=12, ha='center', color=gradient_color, weight='bold')
ax.text(1.5, 8.5, r'$\frac{\partial \mathcal{L}}{\partial A_n}, \frac{\partial \mathcal{L}}{\partial B_n}$', 
        fontsize=11, ha='center', color=gradient_color, weight='bold', rotation=70)
ax.text(10.5, 6.5, r'$\frac{\partial \mathcal{L}}{\partial W_{NN}}$', 
        fontsize=11, ha='center', color=gradient_color, weight='bold', rotation=-70)

# Add legend for arrow types
legend_x = 0.5
legend_y = 15.5
ax.text(legend_x, legend_y, 'Legend:', fontsize=12, weight='bold')
# Forward pass
ax.plot([legend_x, legend_x+0.8], [legend_y-0.3, legend_y-0.3], 
        color=dark_color, linewidth=2.5)
ax.text(legend_x+1, legend_y-0.3, 'Forward Pass', fontsize=11, va='center')
# Backward pass
ax.plot([legend_x, legend_x+0.8], [legend_y-0.6, legend_y-0.6], 
        color=gradient_color, linewidth=2.5, linestyle='dashed')
ax.text(legend_x+1, legend_y-0.6, 'Backward Propagation', fontsize=11, va='center')

# Add optimization info (left side)
opt_box = FancyBboxPatch((0.2, -0.5), 3.5, 2.5, 
                        boxstyle="round,pad=0.1",
                        facecolor=primary_color,
                        edgecolor='black',
                        linewidth=2.5)
ax.add_patch(opt_box)
ax.text(1.95, 1.5, 'Two-Phase', fontsize=15, ha='center', weight='bold', color='white')
ax.text(1.95, 1, 'Optimization', fontsize=15, ha='center', weight='bold', color='white')
ax.text(1.95, 0.4, 'Adam (2000)', fontsize=13, ha='center', color='white')
ax.text(1.95, 0, 'L-BFGS (5000)', fontsize=13, ha='center', color='white')

# Add performance metric (right side)
perf_box = FancyBboxPatch((9.3, -0.5), 3.5, 2.5, 
                         boxstyle="round,pad=0.1",
                         facecolor='#E8F0E8',
                         edgecolor='green',
                         linewidth=2.5)
ax.add_patch(perf_box)
ax.text(11.05, 1.5, 'Achievement', fontsize=14, ha='center', weight='bold', color='darkgreen')
ax.text(11.05, 1, 'L2 Error:', fontsize=13, ha='center', weight='bold', color='darkgreen')
ax.text(11.05, 0.5, r'$1.94 \times 10^{-7}$', fontsize=13, ha='center', color='darkgreen')
ax.text(11.05, 0, '17× improvement', fontsize=12, ha='center', color='darkgreen', style='italic')

plt.tight_layout()
plt.savefig('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/figures/pinn_architecture_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("PINN architecture diagram with backward propagation saved!")