# `eigensystem_ext` Module Documentation

Extended eigensystem solvers.

## Overview

This module adds eigenvalue refinement, generalized symmetric eigen problems, and iterative techniques for dominant or nearby eigenpairs.

| Function | Problem | Method |
|---|---|---|
| [`generalized_eigen_sym`] | Ax = λBx, B SPD | Cholesky transform → standard |
| [`inverse_iteration`] | Refine eigenpair near σ | Shift-and-invert power iter |
| [`rayleigh_quotient_iter`] | Find eigenpair from hint x₀ | Cubic convergence |
| [`simultaneous_iteration`] | Dominant k eigenpairs | QR iteration on tall matrix |

## Notes

- Use these routines when the base eigensystem module is not enough.
- Iterative methods are especially useful for large or structured matrices.
