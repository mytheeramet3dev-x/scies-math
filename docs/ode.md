# `ode` Module Documentation

ODE solvers — fixed-step and adaptive methods for dy/dt = f(t, y).

# Methods

| Function | Type | Order | Notes |
|---|---|---|---|
| `euler` | Fixed | 1 | Simplest; large error accumulation |
| `rk4` | Fixed | 4 | Classical Runge-Kutta; workhorse for smooth problems |
| `rk4_system` | Fixed | 4 | Vector-state version of RK4 |
| `rk45` | Adaptive | 4/5 | Dormand-Prince; auto step-size via tolerance |
| `euler_system` | Fixed | 1 | Euler for vector-valued ODE |

# Usage

```rust
use scies_math_th::ode::rk4;

// dy/dt = -y, y(0) = 1  →  exact: e^{-t}
let sol = rk4(|_t, y| vec![-y[0]], 0.0, vec![1.0], 0.01, 1000);
// sol.y.last() ≈ e^{-10} ≈ 4.54e-5
```

# Adaptive step-size (RK45)

```rust
use scies_math_th::ode::rk45;

let sol = rk45(
    |_t, y| vec![-y[0]],   // f(t, y)
    0.0, 5.0,              // t_start, t_end
    vec![1.0],             // y0
    1e-6, 1e-9,            // rtol, atol
    0.1,                   // initial step
);
```

# Error handling

All solvers return `OdeSolution { t: Vec<f64>, y: Vec<Vec<f64>> }`.
Adaptive solvers may return fewer steps than fixed-step solvers.
