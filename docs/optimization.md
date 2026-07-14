# `optimization` Module Documentation

Scalar and univariate optimization.

# Functions

| Function | Method | Problem type |
|---|---|---|
| `golden_section` | Golden-section search | min f(x) on [a,b] |
| `brent` | Brent's method | min f(x), no derivatives |
| `bisection` | Bisection | root finding f(x)=0 |
| `newton_raphson` | Newton-Raphson | root finding with f' |
| `secant` | Secant method | root finding, no f' needed |
| `illinois` | Illinois/Regula Falsi | Bracketed, superlinear |

# Usage

```rust
use scies_math_th::optimization::golden_section;

// minimize f(x) = (x-2)² + 1 on [0, 4]
let (x_min, f_min) = golden_section(|x| (x-2.0).powi(2) + 1.0, 0.0, 4.0, 1e-8);
// x_min ≈ 2.0, f_min ≈ 1.0
```

For multivariate optimization (BFGS, L-BFGS, Trust-Region, PSO, DE)
see [`crate::opt_multivar`] and [`crate::opt_multivar_ext`].
