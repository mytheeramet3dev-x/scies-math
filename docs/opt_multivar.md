# `opt_multivar` Module Documentation

Multivariate unconstrained optimization.

# Methods

| Function | Method | Gradient? |
|---|---|---|
| `gradient_descent` | Fixed step-size GD | Yes |
| `gradient_descent_armijo` | GD + Armijo line search | Yes |
| `bfgs` | BFGS quasi-Newton | Yes |
| `nelder_mead` | Nelder-Mead simplex | No |
| `conjugate_gradient` | Polak-Ribière CG | Yes |

# Usage — BFGS

```rust
use scies_math::opt_multivar::bfgs;

// Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
let f = |v: &[f64]| (1.0-v[0]).powi(2) + 100.0*(v[1]-v[0]*v[0]).powi(2);
let g = |v: &[f64]| vec![
    -2.0*(1.0-v[0]) - 400.0*v[0]*(v[1]-v[0]*v[0]),
    200.0*(v[1]-v[0]*v[0]),
];

let result = bfgs(f, g, &[-1.0, 1.0], 1e-6, 1000).unwrap();
// result.x ≈ [1.0, 1.0]
```

For trust-region, augmented Lagrangian, PSO, and DE see [`crate::opt_multivar_ext`].