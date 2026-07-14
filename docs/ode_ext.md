# `ode_ext` Module Documentation

Extended ODE solvers — high-order, stiff, and symplectic methods.

## Overview

This module expands the basic ODE layer with methods suited for higher accuracy, stiffness, and long-time Hamiltonian simulation.

## Methods

| Function | Type | Order | Best for |
|---|---|---|---|
| `dopri8` | Adaptive | 8 | High-accuracy smooth ODEs |
| `adams_bashforth4` | Fixed | 4 | Long integrations (multistep) |
| `backward_euler` | Implicit | 1 | Stiff systems |
| `bdf2` | Implicit | 2 | Moderately stiff |
| `rosenbrock2` | Semi-implicit | 2 | Stiff with Jacobian |
| `symplectic_euler` | Symplectic | 1 | Hamiltonian systems (energy-preserving) |
| `stormer_verlet` | Symplectic | 2 | N-body, molecular dynamics |
| `yoshida4` | Symplectic | 4 | High-accuracy Hamiltonian |

## DOP853 (8th order Dormand-Prince)

Preferred for high-accuracy scientific problems. Uses 13 function
evaluations per step (FSAL property saves one evaluation on accepted steps).

```rust
use scies_math_th::ode_ext::dopri8;

// Van der Pol oscillator (weakly nonlinear, mu=1)
let sol = dopri8(
    |_t, y| vec![y[1], (1.0 - y[0]*y[0])*y[1] - y[0]],
    0.0, 20.0,
    vec![2.0, 0.0],
    1e-8, 1e-10,
    0.1,
);
```

## Symplectic integrators

Symplectic methods preserve the phase-space volume of Hamiltonian systems,
giving bounded energy error over long simulations. They require the ODE to
be split into position (q) and momentum (p) parts:

```rust
use scies_math_th::ode_ext::stormer_verlet;

// Simple harmonic oscillator: H = p²/2 + q²/2
// dq/dt = p,  dp/dt = -q
let sol = stormer_verlet(
    |_q: &[f64]| vec![0.0_f64],   // grad_V(q) = q
    vec![1.0],                     // q0
    vec![0.0],                     // p0
    0.01,                          // dt
    10000,                         // steps
);
```

## Notes

- Use `dopri8` when accuracy is the primary goal.
- Use implicit or semi-implicit methods when stiffness is present.
- Use symplectic methods for Hamiltonian systems where long-term energy behavior matters.
