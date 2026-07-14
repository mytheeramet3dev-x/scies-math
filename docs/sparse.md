# `sparse` Module Documentation

Sparse matrix algebra (CSR / COO) and iterative linear solvers.

## Overview

Sparse matrix support for memory-efficient linear algebra workflows.

## Data structures
- [`SparseMatrixCsr`] — Compressed Sparse Row storage (from COO or dense)
- [`IluPreconditioner`] — ILU(0) incomplete factorisation
- [`DiagPreconditioner`] — Jacobi (diagonal) preconditioner

## Solvers
| Method | System type | Preconditioned |
|---|---|---|
| `conjugate_gradient` | SPD | — |
| `preconditioned_cg` | SPD | Yes |
| `gmres` | General | — |
| `preconditioned_gmres` | General | Yes |
| `bicgstab` | General | — |
| `minres` | Symmetric (possibly indefinite) | — |

## Notes

- Use sparse storage when the matrix has many zeros and dense storage would waste memory.
- Pair iterative solvers with preconditioners when convergence is slow.
