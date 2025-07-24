# Peer-Review Report: Methods Section

## Executive Summary
The Methods section presents a compelling hybrid Fourier-Neural PINN architecture that claims “ultra-precision” on a fourth-order Euler–Bernoulli beam PDE. It is well-written, logically ordered, and heavily cites current PINN literature. Major strengths are the clear gap-driven motivation, explicit mathematical formulation, and integration of GPU implementation details. Key areas needing improvement are (1) reproducibility (hyper-parameters & data sampling), (2) clearer distinction between **assumptions** and **findings**, and (3) tighter alignment among equations, algorithm listing, and code references.

---

## Major Concerns

| ID | Issue | Impact | Recommendation |
|----|-------|--------|----------------|
| **M1** | Reproducibility details are insufficient: batch-size function `f(N, M_max)`, collocation-point generation, and stopping criteria lack concrete numbers. | High | Provide exact values (e.g., `B = 16 384` for a 24 GB GPU) or a table mapping GPU memory → batch size; state collocation-grid density. |
| **M2** | Assumptions vs. empirical discoveries blurred. “Assumption 2” (10-harmonic truncation) is actually a result of sensitivity analysis. | Medium | Move this statement to Sensitivity subsection as **Finding 1**; reserve “Assumptions” for a-priori premises. |
| **M3** | Algorithm 1 and Eq. (1) inconsistent with gradient paths. Text says Fourier coefficients receive direct gradients yet Algorithm 1 treats them collectively with network weights. | Medium | Split update step into two parameter groups or annotate identical rule but separate tensors. |
| **M4** | Adaptive-weight formula uses unexplained constant `130`. | Medium | Cite tuning study or mark value as heuristic with brief justification. |
| **M5** | No code linkage for custom GPU kernels; 60 % memory-saving claim unverified. | Medium | Add pointer to `ultra_precision_wave_pinn_GPU.py` (lines X–Y) or include kernel code in appendix. |

---

## Minor Concerns
1. Notation duplication: `a_n` vs `A_n`.
2. Figure 2 caption too generic; emphasize novelty.
3. Equation numbering drift (Eq. 4 referenced as Eq. 5, etc.).
4. Add 2024 literature (*Lu et al., JCP 2024* on adaptive DGM).
5. Define FP16/FP32 on first use.

---

## Clarity & Writing Quality
• Sectioning is logical.  
• Tone occasionally over-assertive (“paradigm shift”, “fundamental breakthrough”).  
• Consider moving 10-harmonic discovery into a dedicated results subsection.

---

## Technical & Formatting Checks
- Required LaTeX packages: `amsmath`, `booktabs`, `graphicx`, `algorithm`, `algorithmic`, `setspace`, `caption`, `subcaption`.  
- Ensure `\graphicspath{{figures/}}` is present in wrapper/main preamble.  
- Run `detect_required_packages.py` before compiling `methods_v3.pdf`.

---

## Actionable Recommendations
1. **Enhance reproducibility** with precise hyper-parameters, grid description, and code link.  
2. **Separate assumptions from findings** for logical clarity.  
3. **Justify adaptive-weight heuristic** or label it as such.  
4. **Provide code/kernel references** to support GPU claims.  
5. **Fix equation numbering and symbol consistency**.

Implementing these changes will significantly improve rigor, transparency, and reader confidence in the proposed methodology. 