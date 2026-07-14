# `autodiff` Module Documentation

Forward-mode automatic differentiation (AD).

Forward AD propagates dual numbers `(value, derivative)` through
arithmetic operations to compute exact derivatives — no finite differences,
no symbolic overhead.

# Dual number type

[`Dual`] represents  f + f'·ε  where ε² = 0.

```rust
use scies_math_th::autodiff::Dual;

let x = Dual::var(3.0);          // x = 3, dx = 1
let y = x * x + Dual::con(2.0);  // y = x² + 2
assert!((y.re - 11.0).abs() < 1e-12);  // value = 9 + 2
assert!((y.du - 6.0 ).abs() < 1e-12);  // derivative = 2x = 6
```

# Gradient and Jacobian

| Function | Output |
|---|---|
| `gradient(f, x)` | ∇f at x (one forward pass per input dim) |
| `jacobian(f, x)` | J_f at x (m×n matrix) |
| `directional_derivative(f, x, v)` | ∇f·v (one forward pass) |

```rust
use scies_math_th::autodiff::gradient;

// f(x, y) = x² + 3xy
let grad = gradient(
    |v| v[0]*v[0] + 3.0*v[0]*v[1],
    &[2.0, 1.0],
);
// grad ≈ [2x + 3y, 3x] = [7, 6]
```

For reverse-mode AD (more efficient when outputs << inputs) see [`crate::reverse_ad`].
