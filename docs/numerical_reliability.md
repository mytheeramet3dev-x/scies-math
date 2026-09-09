# Numerical Reliability, Verification, and Certificate Hierarchy

This document outlines the strict mathematical reliability framework, error semantics, and validity classification used across `scies-math-th` (v0.3.0+).

---

## 1. Four Levels of Mathematical Validity

To maintain scientific integrity and prevent overclaiming, every computational output in `scies-math-th` falls into one of four strictly delineated validity levels:

| Level | Classification | Representation | Description & Guarantees | Modules in `scies-math-th` |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **Numerical Result** | Standard IEEE 754 `f64` | Fast numerical heuristic or first-order convergence. Subject to floating-point round-off, conditioning issues, and termination heuristics. **Does NOT constitute a mathematical certificate.** | `opt_multivar`, `ode`, `pde`, `fitting`, `distributions` |
| **Level 2** | **Verified Numerical Result** | `f64` + [`SdpResidualReport`] | Floating-point candidate accompanied by independent, rigorous residual verification ($r_p, v_p, R_d, v_d, \text{gap} \le \epsilon$). Rejects NaN/Inf, checks symmetry, and validates PSD spectrum. | `sdp`, `sdp_ipm`, `sdp_primitives`, `eigensystem` |
| **Level 3** | **Exact Certificate** | [`Rational`] / Algebraic SOS | Arbitrary-precision exact arithmetic ($p/q \in \mathbb{Q}$ using 128-bit checked integers) and symbolic exact reductions. Guaranteed zero round-off error. | `exact::Rational`, `symbolic::Polynomial`, `symbolic::simplify` |
| **Level 4** | **Formal Proof** | Machine-checkable AST | Lean 4 derivation trees ([`DerivationTree`], [`export_lean4_theorem`]) exportable for formal machine-checked theorem verification. | `verification::derivation`, `verification::lean_export` |

---

## 2. Semidefinite Programming (SDP) KKT Diagnostics

In `scies-math-th`, the termination status `SdpStatus::Optimal` is **never** assigned based solely on internal step sizes or iteration counts. A solution is optimal if and only if all KKT conditions pass independent verification:

1. **Primal Equality Residual**:
   $$r_p = \max_{i=1..m} |\langle A_i, X \rangle - b_i| \le \epsilon_{\text{tol}}$$
2. **Primal Positive Semidefinite (PSD) Violation**:
   $$v_p = \max(0.0, -\lambda_{\min}(X)) \le \epsilon_{\text{tol}}$$
3. **Dual Equality Residual**:
   $$R_d = \|C - \sum_{i=1}^m y_i A_i - S\|_F \le \epsilon_{\text{tol}}$$
   *(Calculated explicitly from the dual difference matrix; never hard-coded to 0).*
4. **Dual Positive Semidefinite (PSD) Violation**:
   $$v_d = \max(0.0, -\lambda_{\min}(S)) \le \epsilon_{\text{tol}}$$
5. **Relative Duality Gap**:
   $$\text{gap} = \frac{|\langle C, X \rangle - b^T y|}{1 + |\langle C, X \rangle| + |b^T y|} \le \epsilon_{\text{tol}}$$
6. **Complementary Slackness Residual**:
   $$\text{comp} = \frac{|\text{Tr}(X S)|}{n}$$

---

## 3. Strict Failure and Error Handling Policies

1. **No Silent Fallbacks**:
   - The SDP solver does not silently substitute the identity matrix if matrix inversion fails.
   - Ill-conditioned or linearly dependent constraint matrices return `Err(SciError::SingularMatrix)`.
   - Non-convergent interior-point line searches return `Err(SciError::NonConvergent)`.
2. **Moment-SOS Degree Checks**:
   - If a CNF clause violation polynomial cannot be embedded in the moment matrix of degree $d$, [`CnfFormula::build_moment_sdp_relaxation`] returns `Err(SciError::InvalidParameter)` rather than silently skipping the clause.
3. **Exact Arithmetic Safety**:
   - Division by zero returns `Err(SciError::DivisionByZero)`.
   - Integer overflow in 128-bit exact rational arithmetic returns `Err(SciError::ExactArithmeticOverflow)`.
4. **SDPA Format I/O**:
   - Multi-block formats beyond single-block ($nblocks \ne 1$) are rejected with an explicit error.
   - Non-finite tokens (NaN/Inf) and out-of-bounds row/column indices are rejected.

---

## 4. API Migration Guide (v0.2.x $\to$ v0.3.0)

| Old API / Behavior | New API in v0.3.0 | Rationale & Migration Note |
| :--- | :--- | :--- |
| `sdp.solve()` returned unverified status | `problem.verify_solution(&x, &y, &s, tol)` + verified `SdpStatus::Optimal` | Independent KKT residual report [`SdpResidualReport`] is now populated accurately with real dual equality residuals. |
| Missing IPM interior-point solver | Added `sdp_ipm::solve_sdp_ipm` (Mehrotra Predictor-Corrector) | High-accuracy primal-dual interior-point solver with full step-damping and predictor-corrector directions. |
| Moment encoding silently skipped high-degree clauses | Returns `Err(SciError::InvalidParameter)` specifying degree | Use `build_moment_sdp_relaxation_with_report(d)` to inspect `MomentRelaxationReport`. |
| Float division panics on zero | `Rational::new(p, 0)` returns `Err(SciError::DivisionByZero)` | Use checked arithmetic (`checked_add`, `checked_mul`, `checked_div`). |
| SDPA parser skipped bad lines | `import_sdpa_sparse` returns `Err(SciError::InvalidParameter)` on token mismatch | Guarantees exact parsing of benchmark SDP datasets. |
