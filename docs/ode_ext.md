# Extended Ordinary Differential Equation Solvers (`ode_ext`)

The `ode_ext` module provides high-order adaptive integrators, implicit stiff solvers with Newton iteration, and geometric symplectic integrators for Hamiltonian dynamical systems.

---

## 1. Solver Selection Guide

| Solver | Type | Order | Adaptivity | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `dopri8` | Runge-Kutta (DOP853) | 8(5,3) | Adaptive | Extremely high precision ($10^{-10} - 10^{-14}$) on smooth celestial / trajectory models |
| `adams_bashforth4`| Linear Multistep | 4 | Fixed | Long-duration integration where function evaluations are computationally expensive |
| `backward_euler` | Fully Implicit | 1 | Fixed | Extremely stiff systems; unconditionally stable (A-stable, L-stable) |
| `bdf2` | Fully Implicit (BDF) | 2 | Fixed | Stiff chemical kinetics and heat conduction equations |
| `rosenbrock2` | Linearly Implicit | 2 | Fixed | Stiff ODEs avoiding non-linear Newton solves via exact Jacobian evaluation |
| `symplectic_euler`| Symplectic | 1 | Fixed | Long-term conservative Hamiltonian dynamics |
| `stormer_verlet` | Symplectic (Leapfrog)| 2 | Fixed | Molecular dynamics, planetary orbits, second-order $q'' = -\nabla V(q)$ |
| `yoshida4` | Symplectic Composition| 4 | Fixed | High-order energy-conserving orbital mechanics |

---

## 2. High-Order Adaptive DOP853 (`dopri8`)

Dormand-Prince 8th order method with embedded 5th and 3rd order error estimators. Uses 13 stages per step with the First-Same-As-Last (FSAL) property to minimize RHS function evaluations.

```rust
use scies_math_th::ode_ext::dopri8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Van der Pol Oscillator (stiff relaxation oscillations at mu = 5.0)
    let mu = 5.0;
    let vdp = move |_t: f64, y: &[f64]| vec![
        y[1],
        mu * (1.0 - y[0] * y[0]) * y[1] - y[0],
    ];

    let sol = dopri8(
        vdp,
        0.0, 20.0,      // [t_start, t_end]
        vec![2.0, 0.0], // Initial state [x, dx/dt]
        1e-8, 1e-10,    // rtol, atol
        0.05,           // Initial step size
    )?;

    println!("DOP853 completed in {} steps.", sol.t.len());
    println!("Final state at t=20: {:?}", sol.y.last().unwrap());
    Ok(())
}
```

---

## 3. Stiff System Solvers (`backward_euler`, `bdf2`, `rosenbrock2`)

For differential equations with widely separated time scales (e.g., chemical kinetics where eigenvalues $\text{Re}(\lambda_i) \ll 0$), explicit methods become unstable unless $\Delta t < 2 / |\lambda_{\max}|$.

### Backward Differentiation Formula 2 (`bdf2`)
Solves the non-linear algebraic system at each step via multi-dimensional Newton-Raphson:

$$y_{n+1} - \frac{4}{3} y_n + \frac{1}{3} y_{n-1} = \frac{2}{3} \Delta t f(t_{n+1}, y_{n+1})$$

A-stable for stiff transients without unphysical oscillations.

---

## 4. Geometric Symplectic Integrators (`stormer_verlet`, `yoshida4`)

Standard explicit Runge-Kutta methods introduce artificial numerical dissipation, causing planetary orbits to spiral inward or outward over long times. 

**Symplectic integrators** preserve the exact symplectic 2-form $d p \wedge d q$ and conserve a nearby "shadow Hamiltonian" $\tilde{H}(q, p)$, ensuring zero secular drift in energy over millions of orbits.

For separable Hamiltonian systems $H(q, p) = \frac{1}{2} p^T M^{-1} p + V(q)$:

### Störmer-Verlet (Leapfrog)
$$p_{n+1/2} = p_n - \frac{\Delta t}{2} \nabla V(q_n)$$

$$q_{n+1} = q_n + \Delta t M^{-1} p_{n+1/2}$$

$$p_{n+1} = p_{n+1/2} - \frac{\Delta t}{2} \nabla V(q_{n+1})$$

```rust
use scies_math_th::ode_ext::stormer_verlet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keplerian Orbit: V(q) = -1 / ||q||
    let grad_v = |q: &[f64]| {
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let r3 = r * r * r;
        vec![q[0] / r3, q[1] / r3]
    };

    let q0 = vec![1.0, 0.0]; // Initial position (perihelion)
    let p0 = vec![0.0, 1.0]; // Initial momentum (circular orbit speed)

    let sol = stormer_verlet(
        grad_v,
        q0, p0,
        0.001, // dt
        10000, // 10,000 steps
    )?;

    let final_q = &sol.y.last().unwrap()[0..2];
    println!("Position after 10,000 steps: {:?}", final_q); // Closes orbit at [1.0, 0.0]

    Ok(())
}
```
