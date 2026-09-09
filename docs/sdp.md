# `sdp` and `sdp_primitives` Module Documentation

Semidefinite programming problem definitions, KKT residual verification, PSD cone projection, and ADMM baseline solver.

## Overview

### Standard Primal-Dual SDP

$$\min_{X \succeq 0} \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i$$

### Solvers and Diagnostics

- `SdpProblem::solve`: ADMM baseline solver with deterministic convergence and residual metrics.
- `diagnose_psd`: Comprehensive spectral diagnostics ($\lambda_{\min}, \lambda_{\max}$, spectral gap, numerical rank, trace).
- `project_to_psd_cone`: Spectral projection onto the positive semidefinite cone $\mathcal{S}_+^n$.
