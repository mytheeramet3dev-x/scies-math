# Primal-Dual Interior-Point SDP Solver (`sdp_ipm`)

The `sdp_ipm` module provides a Mehrotra Predictor-Corrector Primal-Dual Interior-Point Method (IPM) for solving semidefinite programs with high numerical accuracy.

---

## 1. Algorithm Overview

Interior-Point Methods solve the perturbed KKT system along the central path parameterized by barrier parameter $\mu > 0$:

$$X S = \mu I$$

$$\langle A_i, X \rangle = b_i \quad (i = 1, \dots, m), \quad X \succ 0$$

$$\sum_{i=1}^m y_i A_i + S = C, \quad S \succ 0$$

As $\mu \to 0$, the sequence of solutions $(X(\mu), y(\mu), S(\mu))$ converges to the optimal primal-dual solution.

### Mehrotra Predictor-Corrector Scheme

Each IPM iteration consists of two phases:

1. **Affine-Scaling Predictor Step**:
   Solves the linearized Newton system with $\mu = 0$ to determine affine directions $(\Delta X_{\text{aff}}, \Delta y_{\text{aff}}, \Delta S_{\text{aff}})$.
2. **Centering and Corrector Step**:
   Evaluates the step sizes $\alpha_{\text{primal}}^{\text{aff}}, \alpha_{\text{dual}}^{\text{aff}}$ to estimate the centering parameter:
   $$\sigma = \left(\frac{\mu_{\text{aff}}}{\mu}\right)^3 \in [0, 1]$$
   Solves the system with right-hand side augmented by centering $\sigma \mu I$ and corrector $-\Delta X_{\text{aff}} \Delta S_{\text{aff}}$.
3. **Step-Length Computation with Safeguarded Damping**:
   $$\alpha_P = \min\left(1, \gamma \cdot \sup\{\alpha > 0 : X + \alpha \Delta X \succeq 0\}\right)$$
   $$\alpha_D = \min\left(1, \gamma \cdot \sup\{\alpha > 0 : S + \alpha \Delta S \succeq 0\}\right)$$
   where $\gamma \in (0, 1)$ is the step-damping factor (typically $0.95$).

---

## 2. Configuration & Parameter Validation

`SdpIpmConfig` controls the solver behavior:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tolerance` | `f64` | `1e-7` | Target relative duality gap and feasibility tolerance. Must be $> 0$. |
| `max_iterations` | `usize` | `100` | Maximum interior-point iterations. Must be $> 0$. |
| `step_damping` | `f64` | `0.95` | Fraction-to-the-boundary factor $\gamma \in (0, 1)$. |

If any parameter violates its domain, `solve_sdp_ipm` returns `Err(SciError::InvalidParameter)`.

---

## 3. Schur Complement Linear System

At each iteration, the dual direction $\Delta y$ is determined by solving the Schur complement system:

$$M \Delta y = r$$

where the $m \times m$ matrix $M$ is defined by:

$$M_{ij} = \text{Tr}\left(A_i S^{-1} A_j X\right)$$

- If $M$ is ill-conditioned or singular, the solver does not use silent fallbacks (such as replacing with the identity matrix); it strictly returns `Err(SciError::SingularMatrix)`.
- If matrix inversion of $S$ encounters zero eigenvalues, it returns `Err(SciError::DivisionByZero)`.

---

## 4. API Usage Example

```rust
use scies_math_th::sdp::SdpProblem;
use scies_math_th::sdp_ipm::{solve_sdp_ipm, SdpIpmConfig};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?;
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?;
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?;
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b)?;

    let config = SdpIpmConfig {
        tolerance: 1e-8,
        max_iterations: 60,
        step_damping: 0.95,
    };

    let solution = solve_sdp_ipm(&problem, &config)?;

    println!("Primal Objective: {:.8}", solution.primal_objective);
    println!("Dual Objective:   {:.8}", solution.dual_objective);
    println!("Iterations:       {}", solution.iterations);

    // Verify KKT residual report
    assert!(solution.residuals.is_optimal(1e-7));
    println!("Duality Gap:      {:.2e}", solution.residuals.relative_duality_gap);

    Ok(())
}
```

---

## 5. Failure Modes and Error Diagnostics

- `SciError::SingularMatrix`: The Schur complement $M$ or constraint Gram matrix is singular.
- `SciError::NonConvergence`: Solver reached `max_iterations` without satisfying `tolerance`.
- `SciError::DomainError`: Non-finite values (NaN / Inf) encountered during matrix operations.
