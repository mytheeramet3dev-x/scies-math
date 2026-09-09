# Forward-Mode Automatic Differentiation (`autodiff`)

The `autodiff` module implements forward-mode automatic differentiation (AD) using dual numbers. It computes exact analytical derivatives and Jacobians without finite-difference truncation error and without symbolic expression expansion.

---

## 1. Mathematical Foundation: Dual Numbers

A dual number is an ordered pair $\langle u, u' \rangle$ formally represented as:

$$x = u + u' \varepsilon, \quad \varepsilon^2 = 0 \quad (\varepsilon \ne 0)$$

where $u \in \mathbb{R}$ is the primal value and $u' \in \mathbb{R}$ is the tangent (derivative).

Applying any differentiable function $f(x)$ through Taylor expansion yields:

$$f(u + u' \varepsilon) = f(u) + f'(u) u' \varepsilon + \frac{f''(u)}{2!} (u' \varepsilon)^2 + \dots = f(u) + [f'(u) u'] \varepsilon$$

By setting the seed $u' = 1$ for an independent variable $x$, evaluating $f(x + \varepsilon)$ directly computes the pair $\langle f(x), f'(x) \rangle$ in a single forward pass.

### Arithmetic Rules Enforced by `Dual`
- **Addition**: $\langle u, u' \rangle + \langle v, v' \rangle = \langle u + v, u' + v' \rangle$
- **Multiplication**: $\langle u, u' \rangle \cdot \langle v, v' \rangle = \langle u v, u' v + u v' \rangle$ (Product rule)
- **Division**: $\frac{\langle u, u' \rangle}{\langle v, v' \rangle} = \left\langle \frac{u}{v}, \frac{u' v - u v'}{v^2} \right\rangle$ (Quotient rule)
- **Elementary Functions**:
  - $\sin\langle u, u' \rangle = \langle \sin(u), u' \cos(u) \rangle$
  - $\cos\langle u, u' \rangle = \langle \cos(u), -u' \sin(u) \rangle$
  - $\exp\langle u, u' \rangle = \langle \exp(u), u' \exp(u) \rangle$
  - $\ln\langle u, u' \rangle = \left\langle \ln(u), \frac{u'}{u} \right\rangle$
  - $\sqrt{\langle u, u' \rangle} = \left\langle \sqrt{u}, \frac{u'}{2\sqrt{u}} \right\rangle$

---

## 2. API Reference and Complexity

| Function | Signature | Time Complexity | Return Type |
| :--- | :--- | :--- | :--- |
| `grad` | `grad<F: Fn(Dual) -> Dual>(f, x: f64)` | $O(1)$ forward passes | `f64` |
| `jacobian` | `jacobian<F: Fn(&[Dual]) -> Dual>(f, x: &[f64])` | $O(n)$ forward passes | `Vec<f64>` |
| `hessian` | `hessian<F: Fn(&[f64]) -> f64>(f, x: &[f64], h: f64)` | $O(n^2)$ evaluations | `Vec<Vec<f64>>` |

### When to Use Forward Mode
Forward-mode AD is optimal when the number of outputs $m$ is much larger than the number of inputs $n$, or for scalar-to-scalar derivatives ($f: \mathbb{R} \to \mathbb{R}$). For scalar loss functions with high-dimensional parameter vectors ($f: \mathbb{R}^n \to \mathbb{R}$ where $n \gg 1$), use [Reverse-Mode AD](reverse_ad.md).

---

## 3. Code Example

```rust
use scies_math_th::autodiff::{Dual, grad, jacobian};

fn main() {
    // 1. Scalar derivative: f(x) = x^3 + sin(x) at x = 2.0
    // Exact: f'(2.0) = 3*(2)^2 + cos(2.0) = 12.0 + cos(2.0)
    let df = grad(|x| x.powi(3) + x.sin(), 2.0);
    println!("df/dx at x=2.0: {:.8}", df);

    // 2. Multivariate gradient via Jacobian: f(x, y) = x^2 * y + exp(x + y)
    let f_multi = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        x * x * y + (x + y).exp()
    };

    let pt = [1.0, 2.0];
    let grad_vec = jacobian(f_multi, &pt);
    println!("∇f at (1.0, 2.0): [df/dx = {:.6}, df/dy = {:.6}]", grad_vec[0], grad_vec[1]);
}
```
