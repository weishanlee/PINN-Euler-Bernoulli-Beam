# Physics-Informed Neural Networks for Solving Euler-Bernoulli Beam Equation

Paper announced arXiv preprint:
arXiv:2507.20929 [cs.LG] 
(or arXiv:2507.20929v1 [cs.LG] for this version) https://doi.org/10.48550/arXiv.2507.20929

## Overview

This repository contains the implementation of Physics-Informed Neural Networks (PINNs) for solving the Euler-Bernoulli beam equation with ultra-high precision using GPU acceleration. The project demonstrates how PINNs can effectively solve fourth-order partial differential equations in structural mechanics with exceptional accuracy.

## Key Features

- **Ultra-High Precision**: Achieves L2 errors as low as 10^-6 through advanced techniques
- **GPU Acceleration**: Fully optimized for CUDA-enabled GPUs for efficient training
- **Adaptive Loss Weighting**: Dynamic adjustment of physics and boundary loss components
- **Comprehensive Visualization**: Extensive plotting capabilities for analysis and validation
- **Multiple Beam Configurations**: Supports various boundary conditions (simply supported, cantilever, etc.)

## Mathematical Formulation

The Euler-Bernoulli beam equation solved in this work:

```
EI ∂⁴w/∂x⁴ + μ ∂²w/∂t² = f(x,t)
```

Where:
- `w(x,t)` - transverse displacement
- `EI` - flexural rigidity
- `μ` - mass per unit length
- `f(x,t)` - external force

## Repository Structure

### `/codes`
- `ultra_precision_wave_pinn_GPU.py` - Main PINN implementation with GPU acceleration
- `ultra_visualizations_GPU.py` - Comprehensive visualization tools for results analysis
- `run_with_monitoring.py` - Training script with real-time monitoring and logging

### `/figures`
Contains all generated visualizations including:
- 3D surface plots of beam displacement
- Error distribution heatmaps
- Training loss convergence curves
- Validation error analysis
- Comparative plots between PINN and analytical solutions
- Architecture diagrams

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+ with CUDA support
- NumPy, Matplotlib, SciPy
- GPU with CUDA capability (recommended)

### Installation
```bash
git clone https://github.com/weishanlee/PINN-Euler-Bernoulli-Beam.git
cd PINN-Euler-Bernoulli-Beam
pip install -r requirements.txt  # requirements file to be added
```

### Running the Code
```bash
# Run the main training script
python codes/run_with_monitoring.py

# Generate visualizations
python codes/ultra_visualizations_GPU.py
```

## Results

The PINN model achieves:
- **L2 Error**: < 10^-6 for beam displacement
- **Training Time**: ~2 hours on NVIDIA GPU
- **Inference Speed**: Real-time predictions

## Methodology Highlights

1. **Network Architecture**: Deep neural network with specialized activation functions
2. **Loss Function**: Combined physics loss, boundary condition loss, and initial condition loss
3. **Optimization**: Adam optimizer with learning rate scheduling
4. **Sampling Strategy**: Adaptive collocation point sampling for improved accuracy

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lee2025physics,
  title={Physics-Informed Neural Networks for Ultra-High Precision Solution of Euler-Bernoulli Beam Problems},
  author={Lee, Wei Shan},
  journal={arXiv preprint arXiv:2507.20929},
  year={2025}
}
```

## Author

**Wei Shan Lee**  
Pui Ching Middle School Macau  
Email: wslee@g.puiching.edu.mo

## License

This project is open source and available under the MIT License.

## Acknowledgments

This work was supported by [funding/institution details if applicable].

---

For questions, issues, or contributions, please open an issue on GitHub or contact the author directly.