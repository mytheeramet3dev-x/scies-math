# `pde` Module Documentation

Partial differential equations — finite-difference solvers.

# Equations

| Solver | Equation |
|---|---|
| `heat_1d` | ∂u/∂t = α ∂²u/∂x² (1D heat / diffusion) |
| `wave_1d` | ∂²u/∂t² = c² ∂²u/∂x² (1D wave) |
| `laplace_2d` | ∇²u = 0 (2D Laplace / Poisson, Gauss-Seidel) |

# Usage — 1D Heat equation

```rust
use scies_math::pde::heat_1d;

// Rod of length 1, α = 0.01, initial temperature profile
let u0: Vec<f64> = (0..=100).map(|i| {
    let x = i as f64 / 100.0;
    (std::f64::consts::PI * x).sin()
}).collect();

let solution = heat_1d(&u0, 0.01, 0.01, 0.0001, 500);
// solution[k] = temperature profile at time step k
```

See [`crate::pde_ext`] for Navier-Stokes, Burgers' equation, and Crank-Nicolson.