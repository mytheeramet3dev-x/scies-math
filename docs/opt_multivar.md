# Multivariate Unconstrained Optimization (`opt_multivar`)

The `opt_multivar` module provides smooth gradient-based quasi-Newton methods, conjugate gradient schemes, and derivative-free simplex optimization for unconstrained objective functions $f: \mathbb{R}^n \to \mathbb{R}$.

---

## 1. Algorithm Overview

| Function | Algorithm | Gradients | Convergence Rate | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `bfgs` | Broyden-Fletcher-Goldfarb-Shanno | Exact (Dual AD) | Superlinear | Moderate dimensions ($n \le 1000$), smooth $C^2$ functions |
| `lbfgs` | Limited-Memory BFGS | Exact (Dual AD) | Linear / Superlinear | Large-scale problems ($n > 10^4$), memory-constrained |
| `conjugate_gradient` | Nonlinear CG (Polak-Ribière) | Exact (Dual AD) | Linear | Large-scale without storing Hessian approximations |
| `gradient_descent_armijo` | Gradient Descent + Armijo Line Search | Exact (Dual AD) | Linear | Robust baseline, convex problems |
| `gradient_descent` | Fixed-step Gradient Descent | Exact (Dual AD) | Linear | Simple test cases, machine learning iterations |
| `nelder_mead` | Downhill Simplex | None (0-th order) | Sublinear / Heuristic | Non-differentiable, noisy, or black-box objectives |

---

## 2. Quasi-Newton BFGS (`bfgs`)

The BFGS method iteratively updates an inverse Hessian approximation $H_k \approx \nabla^2 f(x_k)^{-1}$:

$$p_k = -H_k \nabla f(x_k)$$

$$x_{k+1} = x_k + \alpha_k p_k$$

where $\alpha_k$ is chosen via **Armijo backtracking line search**:

$$f(x_k + \alpha_k p_k) \le f(x_k) + c_1 \alpha_k \nabla f(x_k)^T p_k \quad (c_1 = 10^{-4})$$

The inverse Hessian is updated via the rank-2 formula:

$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$

where $s_k = x_{k+1} - x_k$, $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$, and $\rho_k = \frac{1}{y_k^T s_k}$.

### Automatic Gradients via `Dual`
Objective functions accept slices of forward-mode automatic differentiation variables `&[Dual]`, guaranteeing exact, machine-precision gradient vectors without finite-difference truncation error.

```rust
use scies_math_th::autodiff::Dual;
use scies_math_th::opt_multivar::bfgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2D Rosenbrock function: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    let rosenbrock = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        let one = Dual::from(1.0);
        let hundred = Dual::from(100.0);
        (one - x) * (one - x) + hundred * (y - x * x) * (y - x * x)
    };

    let x0 = [-1.2, 1.0];
    let (x_star, f_star) = bfgs(rosenbrock, &x0, 1e-6, 500)?;

    println!("Minimizer x*: [x = {:.6}, y = {:.6}]", x_star[0], x_star[1]); // [1.000000, 1.000000]
    println!("f(x*):        {:.2e}", f_star);

    Ok(())
}
```

---

## 3. Limited-Memory BFGS (`lbfgs`)

For high-dimensional problems where storing an $n \times n$ matrix $H_k$ exceeds RAM limits, `lbfgs` stores only the last $m$ displacement vectors $\{s_k, y_k\}_{k-m}^{k-1}$ (typically $m \in [5, 20]$).

Computes search directions $p_k = -r$ via the **two-loop recursion** in $O(m \cdot n)$ time and $O(m \cdot n)$ memory.

```rust
use scies_math_th::autodiff::Dual;
use scies_math_th::opt_multivar::lbfgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sphere = |v: &[Dual]| -> Dual {
        v.iter().map(|&xi| xi * xi).sum()
    };

    let x0 = vec![10.0; 100]; // 100-dimensional problem
    let memory_depth = 10;
    let (x_star, f_star) = lbfgs(sphere, &x0, memory_depth, 1e-6, 200)?;

    println!("Norm of x*: {:.2e}", x_star.iter().map(|x| x*x).sum::<f64>().sqrt());
    println!("f(x*):      {:.2e}", f_star);

    Ok(())
}
```

---

## 4. Derivative-Free Nelder-Mead Simplex (`nelder_mead`)

Constructs a non-degenerate geometric simplex of $n + 1$ vertices in $\mathbb{R}^n$. At each iteration, replaces the worst vertex via:
1. **Reflection**: $\bar{x} + \alpha (\bar{x} - x_{\text{worst}})$
2. **Expansion**: $\bar{x} + \gamma (x_{\text{refl}} - \bar{x})$
3. **Contraction**: $\bar{x} + \rho (x_{\text{worst}} - \bar{x})$
4. **Shrink**: $x_i \leftarrow x_{\text{best}} + \sigma (x_i - x_{\text{best}})$

Requires only floating-point scalar evaluation `Fn(&[f64]) -> f64`.
