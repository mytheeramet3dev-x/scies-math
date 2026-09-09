# Extended Partial Differential Equation Solvers (`pde_ext`)

The `pde_ext` module provides advanced partial differential equation (PDE) and boundary value problem (BVP) solvers, including implicit multi-dimensional schemes, non-linear advection-diffusion, reaction-diffusion wave fronts, shooting methods, and Chebyshev spectral collocation.

---

## 1. 2D Heat Equation — Alternating Direction Implicit (`heat_2d_adi`)

Solves the parabolic 2D heat diffusion equation on $[0, L_x] \times [0, L_y]$:

$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

with homogeneous Dirichlet boundary conditions $u|_{\partial \Omega} = 0$.

### The Peaceman-Rachford ADI Scheme
Standard explicit 2D schemes require $\Delta t \le \frac{\min(\Delta x^2, \Delta y^2)}{4 \alpha}$ for numerical stability. The Alternating Direction Implicit (ADI) method splits each time step into two half-steps:
1. **$x$-Sweep (Implicit in $x$, Explicit in $y$)**:
   $$\left(1 - \frac{r_x}{2} \delta_x^2\right) u^{n+1/2} = \left(1 + \frac{r_y}{2} \delta_y^2\right) u^n$$
2. **$y$-Sweep (Implicit in $y$, Explicit in $x$)**:
   $$\left(1 - \frac{r_y}{2} \delta_y^2\right) u^{n+1} = \left(1 + \frac{r_x}{2} \delta_x^2\right) u^{n+1/2}$$
where $r_x = \frac{\alpha \Delta t}{\Delta x^2}$ and $r_y = \frac{\alpha \Delta t}{\Delta y^2}$.

Both half-steps generate tridiagonal linear systems solved in $O(N)$ operations via the Thomas algorithm, achieving **unconditional stability** and second-order accuracy $O(\Delta t^2 + \Delta x^2 + \Delta y^2)$.

```rust
use scies_math_th::pde_ext::heat_2d_adi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx = 31;
    let ny = 31;
    let mut u0 = vec![0.0f64; nx * ny];
    u0[(ny / 2) * nx + (nx / 2)] = 100.0; // Central temperature spike

    let u_final = heat_2d_adi(
        &u0, nx, ny,
        0.05, 0.05, // dx, dy
        0.001,      // dt
        1.0,        // thermal diffusivity alpha
        100,        // n_steps
    )?;

    println!("Diffused central temp: {:.4}", u_final[(ny / 2) * nx + (nx / 2)]);
    Ok(())
}
```

---

## 2. 2D Hyperbolic Wave Equation (`wave_2d`)

Solves the second-order hyperbolic wave equation:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

with wave speed $c$ and Dirichlet boundary conditions $u|_{\partial \Omega} = 0$.

### Courant-Friedrichs-Lewy (CFL) Condition
Numerical stability requires the Courant number to satisfy:

$$C = c \Delta t \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}} \le 1$$

```rust
use scies_math_th::pde_ext::wave_2d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx = 21;
    let ny = 21;
    let mut u_prev = vec![0.0f64; nx * ny];
    let mut u_curr = vec![0.0f64; nx * ny];
    u_curr[(ny / 2) * nx + (nx / 2)] = 1.0; // Initial displacement pluck

    let u_waves = wave_2d(
        &u_prev, &u_curr,
        nx, ny,
        0.1, 0.1, // dx, dy
        0.01,     // dt
        1.0,      // wave speed c
        50,       // n_steps
    )?;

    println!("Center displacement after 50 steps: {:.4}", u_waves[(ny / 2) * nx + (nx / 2)]);
    Ok(())
}
```

---

## 3. 1D Viscous Burgers' Equation (`burgers_1d`)

Solves the non-linear advection-diffusion equation on a periodic domain $[0, L]$:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

### Numerical Discretization
- **Convective term $u \frac{\partial u}{\partial x}$**: Discretized using first-order **upwind differencing** depending on the sign of local velocity $u_i$:
  $$u_i \frac{\partial u}{\partial x} \approx \begin{cases} u_i \frac{u_i - u_{i-1}}{\Delta x} & \text{if } u_i \ge 0 \\ u_i \frac{u_{i+1} - u_i}{\Delta x} & \text{if } u_i < 0 \end{cases}$$
- **Viscous diffusion term $\nu \frac{\partial^2 u}{\partial x^2}$**: Discretized using second-order central differences $\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$.

```rust
use scies_math_th::pde_ext::burgers_1d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let dx = 2.0 * std::f64::consts::PI / n as f64;
    let dt = 0.001;
    let nu = 0.05; // Kinematic viscosity
    let t_end = 0.5;

    // Initial condition: u0(x) = sin(x)
    let u0: Vec<f64> = (0..n).map(|i| (i as f64 * dx).sin()).collect();
    let u_shock = burgers_1d(&u0, dx, dt, nu, t_end)?;

    println!("Burgers steepened peak at x=pi/2: {:.4}", u_shock[n / 4]);
    Ok(())
}
```

---

## 4. Fisher-KPP Reaction-Diffusion Equation (`fisher_kpp`)

Models traveling reaction wave fronts in ecology and chemical kinetics:

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r \cdot u (1 - u)$$

with zero-flux Neumann boundary conditions $\frac{\partial u}{\partial x}\Big|_{x=0} = \frac{\partial u}{\partial x}\Big|_{x=L} = 0$.

- **Stable front speed**: Minimum traveling wave speed $c^* = 2\sqrt{r D}$.
- **Stability condition**: $\Delta t \le \frac{\Delta x^2}{2 D}$.

```rust
use scies_math_th::pde_ext::fisher_kpp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 50;
    let dx = 0.2;
    let dt = 0.01;
    let mut u0 = vec![0.0f64; n];
    u0[0] = 1.0; // Population seed at left boundary

    let u_front = fisher_kpp(&u0, dx, dt, 0.1, 1.0, 5.0)?;
    println!("Front position at t=5.0: {:?}", &u_front[..10]);
    Ok(())
}
```

---

## 5. Two-Point Boundary Value Problems — Shooting Method (`bvp_shooting`)

Solves non-linear second-order two-point BVPs:

$$u'' = f(x, u, u'), \quad u(a) = y_a, \quad u(b) = y_b$$

### Algorithm
1. Formulates the IVP with initial slope parameter $s = u'(a)$:
   $$u_1' = u_2, \quad u_2' = f(x, u_1, u_2), \quad u_1(a) = y_a, \quad u_2(a) = s$$
2. Integrates using a 4th-order Runge-Kutta scheme (RK4) to evaluate the shooting error:
   $$\Phi(s) = u_1(b; s) - y_b = 0$$
3. Employs bisection root finding on $\Phi(s)$ to locate the exact slope $s^*$, then reconstructs the trajectory.

```rust
use scies_math_th::pde_ext::bvp_shooting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Solve u'' = -u with u(0) = 0, u(pi/2) = 1 (Exact: u(x) = sin(x))
    let f = |_x: f64, u: f64, _du: f64| -u;
    let (x_grid, u_sol) = bvp_shooting(
        f,
        0.0, std::f64::consts::FRAC_PI_2, // [a, b]
        0.0, 1.0,                         // ya, yb
        51,                               // n_points
        1e-6,                             // tolerance
        100,                              // max_iter
    )?;

    let mid_idx = 25;
    println!("Shooting solution at x=pi/4: {:.6} (Exact: {:.6})",
        u_sol[mid_idx],
        (std::f64::consts::FRAC_PI_4).sin()
    );
    Ok(())
}
```

---

## 6. Chebyshev Spectral Collocation (`chebyshev_bvp`)

Provides spectral (exponential) convergence for linear variable-coefficient BVPs on $[-1, 1]$:

$$a_2(x) u'' + a_1(x) u' + a_0(x) u = f(x), \quad u(-1) = u_{\text{left}}, \quad u(1) = u_{\text{right}}$$

### Chebyshev-Gauss-Lobatto Grid
Nodes are distributed quadratically denser near boundaries to eliminate Runge's phenomenon:

$$x_j = \cos\left(\frac{j \pi}{N}\right), \quad j = 0, \dots, N$$

Derivatives are discretized exactly using the Fornberg Chebyshev differentiation matrix $D_N$, yielding algebraic error decays $O(e^{-c N})$ for analytic solutions.

```rust
use scies_math_th::pde_ext::chebyshev_bvp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Solve u'' + u = 0 with u(-1) = sin(-1), u(1) = sin(1)
    let n_nodes = 16;
    let a2 = |_x: f64| 1.0;
    let a1 = |_x: f64| 0.0;
    let a0 = |_x: f64| 1.0;
    let rhs = |_x: f64| 0.0;

    let (nodes, u) = chebyshev_bvp(
        n_nodes,
        a2, a1, a0, rhs,
        (-1.0_f64).sin(), 1.0_f64.sin(),
    )?;

    for (node, val) in nodes.iter().zip(u.iter()).take(5) {
        println!("x = {:+.4}, u(x) = {:+.6} (Exact: {:+.6})", node, val, node.sin());
    }
    Ok(())
}
```

---

## 7. Method of Lines Discretization (`mol_diffusion_reaction`)

Transforms a 1D partial differential equation:

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(t, u)$$

into a coupled ODE system $\frac{d u_i}{dt} = F_i(t, \mathbf{u})$ with Dirichlet boundary conditions. The returned closure can be plugged directly into any ODE integrator in [`ode`](ode.md) (such as `rk4_system` or `rk45`).