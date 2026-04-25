# `sparse` Module Documentation

Sparse matrix algebra (CSR / COO) and iterative linear solvers.

# Data structures
- [`SparseMatrixCsr`] — Compressed Sparse Row storage (from COO or dense)
- [`IluPreconditioner`] — ILU(0) incomplete factorisation
- [`DiagPreconditioner`] — Jacobi (diagonal) preconditioner

# Solvers
| Method | System type | Preconditioned |
|---|---|---|
| `conjugate_gradient` | SPD | — |
| `preconditioned_cg` | SPD | Yes |
| `gmres` | General | — |
| `preconditioned_gmres` | General | Yes |
| `bicgstab` | General | — |
| `minres` | Symmetric (possibly indefinite) | — |