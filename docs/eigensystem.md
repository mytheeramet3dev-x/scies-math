# `eigensystem` Module Documentation

Full eigensystem solvers returning eigenvalues **and** eigenvectors.

## Overview

This module provides the core eigenvalue/eigenvector routines used by matrix analysis, stability checks, modal analysis, and spectral methods.

| Function | Matrix type | Method |
|---|---|---|
| [`jacobi_eigen`]     | Real symmetric | Classic Jacobi off-diagonal pivoting |
| [`qr_eigen_general`] | General real   | QR iteration with Wilkinson shifts   |

## Usage

Use `jacobi_eigen` for symmetric matrices when you want a stable route to eigenpairs.
Use `qr_eigen_general` for general real matrices when you need a more universal solver.

## Notes

- Eigen problems are sensitive to conditioning; interpret near-repeated eigenvalues carefully.
- For advanced workflows, the extension module provides refinement and generalized solvers.
