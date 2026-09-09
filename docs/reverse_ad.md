# Reverse-Mode Automatic Differentiation (`reverse_ad`)

The `reverse_ad` module implements reverse-mode automatic differentiation (also known as the adjoint state method or backpropagation). It records a dynamic computation DAG on a Wengert tape during the forward pass and accumulates adjoint gradients backwards in $O(1)$ reverse passes.

---

## 1. Mathematical Foundation: Adjoint Accumulation

For a scalar objective function $y = f(x_1, \dots, x_n)$ computed via intermediate variables $v_1, \dots, v_m$:

$$\bar{v}_i = \frac{\partial y}{\partial v_i} \quad (\text{Adjoint of variable } v_i)$$

The chain rule backwards accumulation states:

$$\bar{v}_i = \sum_{j \in \text{Children}(i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}$$

### Algorithmic Efficiency
- **Forward-mode AD**: Computing $\nabla f(x) \in \mathbb{R}^n$ requires $n$ separate passes ($O(n \cdot \text{Cost}(f))$).
- **Reverse-mode AD**: Evaluates the full gradient $\nabla f(x) \in \mathbb{R}^n$ in a **single backward pass**, with time bounded by $\le 4 \times \text{Cost}(f)$, completely independent of input dimension $n$.

---

## 2. The Wengert Tape Architecture

- `Tape`: Stores an execution graph of nodes, each recording parent indices and local partial derivatives $\frac{\partial v_j}{\partial v_i}$.
- `Var<'t>`: Represents an active scalar on the tape with operator overloading (`+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `ln`).
- `GradMap`: Key-value query structure returned by `tape.backward(&out_var)`.

---

## 3. Code Example

```rust
use scies_math_th::reverse_ad::Tape;

fn main() {
    let tape = Tape::new();

    // Independent variables
    let x = tape.var(3.0);
    let y = tape.var(2.0);

    // Compute z = x^2 * y + sin(x)
    let z = (x * x * y) + x.sin();

    // Reverse pass
    let grads = tape.backward(&z);

    // Query exact gradients
    // dz/dx = 2*x*y + cos(x) = 2*(3)*(2) + cos(3) = 12 + cos(3)
    let dz_dx = grads.of(&x);
    // dz/dy = x^2 = 9
    let dz_dy = grads.of(&y);

    println!("z value: {:.6}", z.val());
    println!("∂z/∂x:   {:.8} (Expected: {:.8})", dz_dx, 12.0 + 3.0_f64.cos());
    println!("∂z/∂y:   {:.8} (Expected: {:.8})", dz_dy, 9.0);
}
```
