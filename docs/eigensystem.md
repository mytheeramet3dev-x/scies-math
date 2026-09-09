# `eigensystem` Module Documentation

Full eigensystem solvers returning eigenvalues **and** eigenvectors with convergence diagnostics.

## Overview

This module provides eigenvalue/eigenvector routines used by matrix analysis, spectral diagnostics, stability checks, and moment relaxations.

| Function | Matrix type | Method | Returns |
|---|---|---|---|
| [`jacobi_eigen`]     | Real symmetric | Classic Jacobi off-diagonal pivoting | [`Eigensystem`] (values, vectors, sweeps, residual, converged) |
| [`qr_eigen_general`] | General real   | QR iteration with Wilkinson shifts   | `(real_parts, imag_parts, eigenvector_matrix)` |

## Usage Examples

### Jacobi Decomposition for Symmetric Matrix

```rust
use scies_math_th::linear_algebra::DynamicMatrix;
use scies_math_th::eigensystem::jacobi_eigen;

let a = DynamicMatrix::new(2, 2, vec![
    4.0, 1.0,
    1.0, 2.0,
]).unwrap();

let eig = jacobi_eigen(&a, 1e-10, 100).unwrap();
println!("Eigenvalues: {:?}", eig.values); // [4.414..., 1.585...]
println!("Converged: {}, sweeps: {}, residual: {:e}", eig.converged, eig.sweeps, eig.residual);
```

## Numerical Precision & Proof Disclaimer

> **Mathematical Context:**
> Calculations in this module use 64-bit floating-point arithmetic (`f64`). Numerical eigenvalues and residuals are computational diagnostics, not exact mathematical certificates. In complexity theory or theoretical proofs (such as P vs NP investigations), numerical results must be verified through exact rational/interval certificates.

