# Peer-Review Report: Methods Section (v6)

## Executive Summary
The revised Methods section (v6) demonstrates meaningful progress toward reproducibility and clarity compared with earlier drafts. The hybrid Fourier-Neural PINN architecture is well motivated, the two-phase optimisation scheme is detailed, and concrete hyper-parameters are now included. The manuscript convincingly addresses most prior major concerns, yet several issues remain—chiefly the need for empirical justification of key hyper-parameters (e.g., the 130 constant, batch-size formula), clearer linkage between code and equations, and minor consistency glitches.

---

## Major Concerns

| ID | Issue | Impact | Recommendation |
|----|-------|--------|----------------|
| **M1** | **Empirical justification for adaptive-weight constant (130) still weak.** The text now says “determined through hyper-parameter tuning” but provides neither tuning study nor sensitivity plot. | High | Add Appendix figure/table showing loss convergence for different constants; briefly summarise tuning protocol. |
| **M2** | **Batch-size formula lacks derivation & validation.** Equation includes memory factor and parameter count, but no citation or experiment verifies that $B=\min(16384, …)$ is optimal or even feasible across GPUs. | High | Supply a worked example (e.g., 24 GB GPU → B=16384) and a footnote citing prior large-batch PINN work or internal benchmark. |
| **M3** | **Algorithm 1 still blends Fourier and NN updates in same loop despite earlier clarification.** Lines 13–17 update $A_n$, $B_n$ using SGD-style step whereas weights use Adam; mixing optimisers can cause instability. | Medium | Separate loops or clearly state learning-rate schedule & optimiser for coefficients vs NN weights. Consider same optimiser for both or justify difference. |
| **M4** | **Code traceability only partial.** A file/line reference is provided (145–267) for GPU kernels, but no Git commit hash or public repository. Readers cannot reproduce custom CUDA kernels. | Medium | Publish kernels (e.g., GitHub repo) or include them in supplementary materials; reference commit hash for exact version. |
| **M5** | **Sensitivity study description qualitative only.** Claims “N=10 gives best L2 error” but does not show error curve, table or statistical test. | Medium | Add a figure or table summarising L2 errors for N=5–50; include CI or variance across seeds. |

---

## Minor Concerns
1. **Table 1**: `FP16`/`FP32` now defined—good—but alphabetical order is broken; consider re-ordering.
2. **Figure 1 caption**: Replace “achieves ultra-precision” with measurable claim (“L2 error < 2×10⁻⁷”).
3. **Equation numbering**: Eq. (5) (loss_pde) is referenced later as Eq. (4)—update cross-references.
4. **Hyper-parameter symbols**: `N_params` appears in batch-size formula but is never defined.
5. **Typos**: “non-modal” → “non-modal” appears twice; remove duplicate.

---

## Clarity & Writing Quality
• Tone improved—superlatives toned down.  
• Section flow logical (`Framework → Architecture → Loss → Optimisation → GPU → Sensitivity`).  
• Checklist markers remain inside `.tex`; ensure they are excised from clean version used in final PDF.

---

## Technical & Formatting Checks
- LaTeX packages now documented in symbol table comment; verify `algorithm2e` *not* required if using `algorithmic`.  
- Re-run `detect_required_packages.py`; previous warning about `setspace` unresolved.  
- Confirm `\graphicspath{{figures/}}` present in the wrapper.

---

## Actionable Recommendations
1. Provide empirical evidence for adaptive-weight constant (130).  
2. Justify batch-size formula with example and benchmark.  
3. Separate or justify mixed-optimiser strategy in Algorithm 1.  
4. Supply full code for GPU kernels or supplementary listing.  
5. Add quantitative sensitivity-analysis results (figure/table).  
6. Fix minor consistency and cross-reference issues.

Addressing these points will bring the Methods section to a publishable standard of rigour and reproducibility. 