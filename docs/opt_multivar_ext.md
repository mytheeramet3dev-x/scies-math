# Extended Multivariate Optimization (`opt_multivar_ext`)

The `opt_multivar_ext` module provides second-order trust-region methods, first-order adaptive optimizers, constrained optimization algorithms, and global population-based metaheuristics.

---

## 1. Algorithm Overview

| Algorithm | Method Category | Gradient | Constraints Supported | Key Advantage |
| :--- | :--- | :--- | :--- | :--- |
| `trust_region_cg` | 2nd-Order Newton | Finite Difference | Unconstrained | Robust on indefinite Hessians / saddle points |
| `adam` | 1st-Order Adaptive | Numerical / User | Unconstrained | Deep learning, noisy stochastic gradients |
| `adagrad` | 1st-Order Adaptive | Numerical / User | Unconstrained | Sparse features, historical gradient scaling |
| `rmsprop` | 1st-Order Adaptive | Numerical / User | Unconstrained | Non-stationary loss surfaces |
| `projected_gradient` | 1st-Order Projected | Finite Difference | Box bounds $[l_i, u_i]$ | Exact bound projection at every iteration |
| `augmented_lagrangian` | Sequential Penalty | Finite Difference | Non-linear Equality + Inequality | General non-linear constrained optimization |
| `particle_swarm` | Metaheuristic (PSO) | None (0-th order) | Box bounds $[l_i, u_i]$ | Global multimodal optimization |
| `differential_evolution` | Evolutionary (DE) | None (0-th order) | Box bounds $[l_i, u_i]$ | Global search with mutation vectors |

---

## 2. Trust-Region with Steihaug-CG Subproblem (`trust_region_cg`)

Solves the quadratic subproblem within a dynamically adjusted trust radius $\Delta_k$:

$$\min_{p} m_k(p) = f(x_k) + g_k^T p + \frac{1}{2} p^T B_k p \quad \text{s.t.} \quad \|p\|_2 \le \Delta_k$$

The **Steihaug conjugate-gradient** algorithm terminates early if negative curvature ($p^T B_k p \le 0$) is encountered, safely following the direction of negative curvature to the boundary $\|p\| = \Delta_k$.

### Radius Adaptation
Based on the agreement ratio $\rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}$:
- If $\rho_k < 0.25$: Step rejected, shrink radius $\Delta_{k+1} = 0.25 \Delta_k$.
- If $\rho_k > 0.75$ and $\|p_k\| = \Delta_k$: Step accepted, expand radius $\Delta_{k+1} = \min(2 \Delta_k, \Delta_{\max})$.
- Otherwise: Step accepted, retain radius $\Delta_{k+1} = \Delta_k$.

---

## 3. Bound-Constrained Optimization (`projected_gradient`)

Minimizes $f(x)$ subject to box constraints $l_i \le x_i \le u_i$:

$$x_{k+1} = \Pi_{[l, u]}\left(x_k - \alpha_k \nabla f(x_k)\right)$$

where the projection operator $\Pi_{[l, u]}$ clamps coordinates:

$$\Pi_{[l, u]}(v)_i = \min\left(u_i, \max(l_i, v_i)\right)$$

```rust
use scies_math_th::opt_multivar_ext::projected_gradient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize f(x, y) = (x - 3)^2 + (y - 3)^2 inside [0, 2] x [0, 2]
    let f = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] - 3.0).powi(2);
    let lower = vec![0.0, 0.0];
    let upper = vec![2.0, 2.0];
    let x0 = vec![0.5, 0.5];

    let x_opt = projected_gradient(f, &x0, &lower, &upper, 1e-5, 200)?;
    println!("Constrained minimizer: {:?}", x_opt); // [2.0, 2.0]

    Ok(())
}
```

---

## 4. General Constrained Optimization (`augmented_lagrangian`)

Minimizes $f(x)$ subject to equality constraints $h_j(x) = 0$ ($j = 1, \dots, p$) and inequality constraints $g_k(x) \le 0$ ($k = 1, \dots, m$):

$$\mathcal{L}_A(x, \lambda, \mu, \rho) = f(x) + \sum_{j=1}^p \left(\lambda_j h_j(x) + \frac{\rho}{2} h_j(x)^2\right) + \sum_{k=1}^m \frac{1}{2\rho} \left(\max(0, \mu_k + \rho g_k(x))^2 - \mu_k^2\right)$$

Performs sequential unconstrained minimizations using BFGS or gradient descent, followed by dual multiplier updates:

$$\lambda_j \leftarrow \lambda_j + \rho h_j(x^*), \quad \mu_k \leftarrow \max(0, \mu_k + \rho g_k(x^*))$$

---

## 5. Global Search: Particle Swarm Optimization (`particle_swarm`)

Maintains a population of $N_p$ particles in $\mathbb{R}^n$. Each particle updates position $x_i$ and velocity $v_i$:

$$v_i^{(t+1)} = w v_i^{(t)} + c_1 r_1 (p_i^{\text{best}} - x_i^{(t)}) + c_2 r_2 (g^{\text{best}} - x_i^{(t)})$$

$$x_i^{(t+1)} = x_i^{(t)} + v_i^{(t+1)}$$

where $w \approx 0.729$ is the inertia weight, and $c_1 = c_2 \approx 1.494$ are the cognitive and social acceleration coefficients.
