# Semidefinite Programming (`sdp`) Module Reference

The `sdp` and `sdp_primitives` modules provide semidefinite programming (SDP) problem abstractions, spectral diagnostics, PSD cone projection, and a first-order Alternating Direction Method of Multipliers (ADMM) solver.

---

## 1. Mathematical Formulation

Standard primal and dual semidefinite programs in `scies-math-th` are formulated over the cone of real symmetric positive semidefinite matrices $\mathcal{S}_+^n$:

### Primal SDP
$$\min_{X \in \mathcal{S}^n} \langle C, X \rangle \quad \text{subject to} \quad \langle A_i, X \rangle = b_i \quad (i = 1, \dots, m), \quad X \succeq 0$$

where $\langle A, B \rangle = \text{Tr}(A^T B) = \sum_{j,k} A_{jk} B_{jk}$ denotes the trace inner product (Frobenius inner product).

### Dual SDP
$$\max_{y \in \mathbb{R}^m, S \in \mathcal{S}^n} b^T y \quad \text{subject to} \quad \sum_{i=1}^m y_i A_i + S = C, \quad S \succeq 0$$

where $S \in \mathcal{S}_+^n$ is the dual slack matrix.

---

## 2. The Karush-Kuhn-Tucker (KKT) Optimality Conditions

A triplet $(X^*, y^*, S^*)$ constitutes a verified optimal primal-dual solution if and only if:

1. **Primal Feasibility**:
   $$\langle A_i, X \rangle = b_i \quad (\forall i \in \{1, \dots, m\}), \quad X \succeq 0$$
2. **Dual Feasibility**:
   $$\sum_{i=1}^m y_i A_i + S = C, \quad S \succeq 0$$
3. **Complementary Slackness**:
   $$\langle X, S \rangle = \text{Tr}(X S) = 0$$
4. **Zero Duality Gap**:
   $$\langle C, X \rangle - b^T y = 0$$

### Independent Verification (`verify_sdp_candidate`)

The function `verify_sdp_candidate` performs an independent a-posteriori check returning [`SdpResidualReport`]:

```rust
pub fn verify_sdp_candidate(
    problem: &SdpProblem,
    x: &SymmetricMatrix,
    y: &[f64],
    s: &SymmetricMatrix,
    eigen_tolerance: f64,
) -> SciResult<SdpResidualReport>
```

Metrics tracked in `SdpResidualReport`:
- `primal_equality_residual`: $r_p = \max_i |\langle A_i, X \rangle - b_i|$
- `primal_psd_violation`: $v_p = \max(0, -\lambda_{\min}(X))$
- `dual_equality_residual`: $R_d = \|C - \sum y_i A_i - S\|_F$
- `dual_psd_violation`: $v_d = \max(0, -\lambda_{\min}(S))$
- `relative_duality_gap`: $\frac{|\langle C, X \rangle - b^T y|}{1 + |\langle C, X \rangle| + |b^T y|}$
- `complementary_slackness_residual`: $\frac{|\text{Tr}(X S)|}{n}$

`is_optimal(tol)` evaluates to `true` if and only if $r_p \le \text{tol} \land v_p \le \text{tol} \land R_d \le \text{tol} \land v_d \le \text{tol} \land \text{gap} \le \text{tol}$.

---

## 3. ADMM Algorithm Architecture

`SdpProblem::solve` implements an operator-splitting ADMM algorithm:

1. **Formulate Augmented Lagrangian**:
   $$\mathcal{L}_\rho(X, Z, \Lambda) = \langle C, X \rangle + \langle \Lambda, X - Z \rangle + \frac{\rho}{2} \|X - Z\|_F^2 \quad \text{s.t.} \quad \mathcal{A}(X) = b, \; Z \in \mathcal{S}_+^n$$
2. **X-Update (Affine Subspace Projection)**:
   Solves a linear system using the unregularized Gram matrix $G_{ij} = \langle A_i, A_j \rangle$ via LU factorization. If $G$ is singular (linearly dependent constraints), returns `Err(SciError::SingularMatrix)`.
3. **Z-Update (PSD Cone Projection)**:
   $$Z^{k+1} = \Pi_{\mathcal{S}_+^n}\left(X^{k+1} + \frac{1}{\rho}\Lambda^k\right)$$
   Computes Jacobi spectral decomposition and clamps negative eigenvalues to 0.
4. **Multiplier Update**:
   $$\Lambda^{k+1} = \Lambda^k + \rho (X^{k+1} - Z^{k+1})$$

---

## 4. API Usage Example

```rust
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus, verify_sdp_candidate};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize Tr(C * X)
    // C = [1.0, 0.0; 0.0, 2.0]
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?;

    // A1 = [1.0, 0.0; 0.0, 0.0], b1 = 1.0 => X_00 = 1.0
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?;
    // A2 = [0.0, 0.0; 0.0, 1.0], b2 = 1.0 => X_11 = 1.0
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?;

    let problem = SdpProblem::new(c, vec![a1, a2], vec![1.0, 1.0])?;

    let config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 500,
        rho: 1.0,
        strict: true,
    };

    let solution = problem.solve(&config)?;
    assert_eq!(solution.status, SdpStatus::Optimal);

    println!("Primal Objective: {:.6}", solution.primal_objective); // 3.000000
    println!("Dual Objective:   {:.6}", solution.dual_objective);

    // Verify candidate
    let report = verify_sdp_candidate(&problem, &solution.x, &solution.y, &solution.s, 1e-4)?;
    assert!(report.is_optimal(1e-4));

    Ok(())
}
```

---

## 5. Spectral Diagnostics (`sdp_primitives`)

The `sdp_primitives` module provides cone projections and spectral properties:

- `diagnose_psd(&SymmetricMatrix, tol)`: Rejects non-finite values (NaN/Inf) and returns `PsdDiagnosis` containing:
  - `is_psd`: whether $\lambda_{\min} \ge -\text{tol}$
  - `is_strictly_pd`: whether $\lambda_{\min} > \text{tol}$
  - `min_eigenvalue`, `max_eigenvalue`
  - `condition_number`: $\lambda_{\max} / \lambda_{\min}$
  - `numerical_rank`: count of eigenvalues exceeding threshold
- `project_to_psd_cone(&SymmetricMatrix)`: Projects any symmetric matrix onto $\mathcal{S}_+^n$ with minimum Frobenius error:
  $$\Pi_{\mathcal{S}_+^n}(A) = \sum_{i: \lambda_i > 0} \lambda_i v_i v_i^T$$
