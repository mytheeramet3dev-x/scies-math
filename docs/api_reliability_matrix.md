# API Reliability & Mathematical Validity Matrix

This document provides a comprehensive classification of every module and subsystem in `scies-math-th` across the 5 Mathematical Validity Tiers.

---

## 1. Classification Tiers

| Tier | Status | Mathematical Guarantee | Verifier Status | Appropriate Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Experimental** | Exploratory heuristic; API subject to change | No formal verifier | Rapid prototyping, early exploration |
| **Tier 2** | **Numerical** | Standard floating-point IEEE 754 convergence | Internal stopping criteria only | Scientific simulation, engineering approximations |
| **Tier 3** | **Numerically Verified** | Rigorous a-posteriori KKT or residual enclosure | Independent residual verifier | High-assurance numerical optimization, SDP relaxations |
| **Tier 4** | **Exact** | Zero round-off algebraic representation ($p/q \in \mathbb{Q}$, symbolic AST) | Exact canonical reducer | Algebraic identity checking, exact polynomial division |
| **Tier 5** | **Formal** | Machine-checkable skeleton / interactive theorem prover export | External Lean 4 / proof assistant | Formalized mathematics, certified proof pipelines |

---

## 2. Subsystem Reliability Matrix

| Module | Core Types / Functions | Validity Tier | Preconditions & Validation | Error Taxonomy |
| :--- | :--- | :--- | :--- | :--- |
| `exact::rational` | [`Rational`] | **Exact** | Rejects denom = 0, checks 128-bit overflow | `DivisionByZero`, `ExactArithmeticOverflow` |
| `exact::interval` | [`Interval`] | **Numerical** | Rejects lower > upper, NaN bounds | `DomainError`, `InvalidParameter`, `DivisionByZero` |
| `symbolic` | [`Expr`], [`Polynomial`], `simplify`, `diff` | **Exact** | Canonical term ordering, degree checks | `SymbolicEvaluationError`, `DivisionByZero` |
| `units` | [`Dimension`], [`Quantity`] | **Exact** | SI 7-base dimensional consistency | `DimensionalMismatch` |
| `verification` | [`DerivationTree`], `export_lean4_theorem` | **Formal** | Step validation, premise tracking | `SymbolicEvaluationError` |
| `sdp` | [`SdpProblem`], [`SdpSolution`], ADMM | **Numerically Verified** | Validates matrix dimensions, Gram independence | `SingularMatrix`, `NonConvergence`, `InvalidParameter` |
| `sdp_ipm` | `solve_sdp_ipm` (Mehrotra IPM) | **Numerically Verified** | Independent KKT residual report ($r_p, v_p, R_d, v_d, \text{gap}$) | `SingularMatrix`, `NonConvergence`, `DomainError` |
| `sdp_primitives` | `diagnose_psd`, `project_to_psd_cone` | **Numerically Verified** | Rejects NaN/Inf/asymmetric, reports residual | `DomainError`, `InvalidParameter` |
| `moment` | [`MomentMatrixBuilder`], [`CnfFormula`] | **Numerically Verified** | Rejects insufficient relaxation degree | `InvalidParameter`, `UnsupportedFormat` |
| `moment::sdpa_io`| `export_sdpa_sparse`, `import_sdpa_sparse` | **Numerically Verified** | Rejects multi-block, validates 5-token entries | `InvalidParameter`, `ParseError`, `DomainError` |
| `linear_algebra` | [`DynamicMatrix`], LU, QR, Cholesky, SVD | **Numerical** | Shape compatibility, pivot > epsilon | `DimensionMismatch`, `SingularMatrix`, `DomainError` |
| `symmetric` | [`SymmetricMatrix`] | **Numerical** | Packed lower-triangular, finite entries | `InvalidParameter`, `DomainError` |
| `eigensystem` | `jacobi_eigen`, `qr_eigen_general` | **Numerically Verified** | Reports $\|A V - V \Lambda\|_F$ and $\|V^T V - I\|_F$ | `DomainError`, `InvalidParameter` |
| `sparse` | [`CsrMatrix`], `LinearOperator` | **Numerical** | Sorted column indices, valid row pointers | `DimensionMismatch`, `InvalidParameter` |
| `opt_multivar` | GD, Adam, BFGS, L-BFGS | **Numerical** | Finite gradient, Armijo line search | `NonConvergence`, `InvalidParameter` |
| `ode` / `pde` | RK4, ADI, Wave solvers | **Numerical** | Step size $h > 0$, Courant stability | `InvalidParameter`, `DimensionMismatch` |
| `autodiff` | Forward / Reverse AD | **Exact / Numerical** | Tape graph acyclicity | `InvalidParameter`, `DomainError` |
| `statistics` | Mean, variance, distributions | **Numerical** | Parameter positivity, non-empty slices | `EmptyInput`, `InvalidParameter`, `DomainError` |
