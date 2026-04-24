//! Advanced PDE solvers beyond the basic heat-equation and Laplace relaxation.
//!
//! | Solver | PDE | Method |
//! |---|---|---|
//! | [`wave_equation_1d`]          | 1-D wave equation u_tt = c²u_xx | Explicit leapfrog |
//! | [`advection_1d`]              | 1-D linear advection u_t + a·u_x = 0 | Upwind scheme |
//! | [`heat_equation_1d_implicit`] | 1-D heat equation (unconditionally stable) | Crank–Nicolson / Thomas algorithm |
//! | [`poisson_2d`]                | 2-D Poisson ∇²u = f | Gauss–Seidel / SOR |

use crate::errors::{SciError, SciResult};

// ── 1-D Wave equation ─────────────────────────────────────────────────────────

/// Solve the 1-D wave equation  `u_tt = c²·u_xx`  with Dirichlet boundary
/// conditions using the explicit leapfrog (Verlet) scheme.
///
/// # Arguments
/// - `u0`   — initial displacement  u(x, 0)
/// - `u1`   — displacement after first time step  u(x, dt)  (or supply via IC velocities)
/// - `c`    — wave speed  (> 0)
/// - `dx`   — spatial step
/// - `dt`   — time step; must satisfy  c·dt/dx ≤ 1  (CFL condition)
/// - `steps`— number of time steps to evolve
pub fn wave_equation_1d(
    u0: &[f64],
    u1: &[f64],
    c: f64,
    dx: f64,
    dt: f64,
    steps: usize,
) -> SciResult<Vec<Vec<f64>>> {
    if u0.len() != u1.len() || u0.len() < 3 {
        return Err(SciError::InvalidParameter(
            "u0 and u1 must have the same length ≥ 3",
        ));
    }
    if c <= 0.0 || dx <= 0.0 || dt <= 0.0 {
        return Err(SciError::InvalidParameter("c, dx, dt must be positive"));
    }
    let r = c * dt / dx;
    if r > 1.0 {
        return Err(SciError::InvalidParameter(
            "CFL violation: c·dt/dx must be ≤ 1 for stability",
        ));
    }
    let r2 = r * r;
    let n = u0.len();
    let mut states = Vec::with_capacity(steps + 2);
    let mut prev = u0.to_vec();
    let mut curr = u1.to_vec();
    states.push(prev.clone());
    states.push(curr.clone());

    for _ in 0..steps {
        let mut next = vec![0.0f64; n];
        // Dirichlet BC: endpoints stay zero.
        for i in 1..(n - 1) {
            next[i] = 2.0 * curr[i] - prev[i] + r2 * (curr[i - 1] - 2.0 * curr[i] + curr[i + 1]);
        }
        states.push(next.clone());
        prev = curr;
        curr = next;
    }
    Ok(states)
}

// ── 1-D Advection ─────────────────────────────────────────────────────────────

/// First-order upwind scheme for `u_t + a·u_x = 0`.
///
/// Periodic boundary conditions.
/// CFL condition:  |a|·dt/dx ≤ 1.
pub fn advection_1d(
    initial: &[f64],
    velocity: f64,
    dx: f64,
    dt: f64,
    steps: usize,
) -> SciResult<Vec<Vec<f64>>> {
    if initial.len() < 2 {
        return Err(SciError::InvalidParameter(
            "initial condition requires at least 2 points",
        ));
    }
    if dx <= 0.0 || dt <= 0.0 {
        return Err(SciError::InvalidParameter("dx and dt must be positive"));
    }
    let cfl = velocity.abs() * dt / dx;
    if cfl > 1.0 {
        return Err(SciError::InvalidParameter(
            "CFL violation: |a|·dt/dx must be ≤ 1",
        ));
    }

    let n = initial.len();
    let r = velocity * dt / dx;
    let mut states = Vec::with_capacity(steps + 1);
    let mut current = initial.to_vec();
    states.push(current.clone());

    for _ in 0..steps {
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            let im1 = if i == 0 { n - 1 } else { i - 1 };
            let ip1 = (i + 1) % n;
            next[i] = if velocity >= 0.0 {
                current[i] - r * (current[i] - current[im1])
            } else {
                current[i] - r * (current[ip1] - current[i])
            };
        }
        states.push(next.clone());
        current = next;
    }
    Ok(states)
}

// ── 1-D Implicit Heat (Crank–Nicolson) ───────────────────────────────────────

/// Solve the 1-D heat equation `u_t = α·u_xx` with an **implicit**
/// Crank–Nicolson scheme (unconditionally stable).
///
/// Uses the Thomas (tridiagonal) algorithm for O(n) solves per step.
pub fn heat_equation_1d_implicit(
    initial: &[f64],
    diffusivity: f64,
    dx: f64,
    dt: f64,
    steps: usize,
) -> SciResult<Vec<Vec<f64>>> {
    if initial.len() < 3 {
        return Err(SciError::InvalidParameter(
            "heat equation requires at least 3 spatial samples",
        ));
    }
    if diffusivity <= 0.0 || dx <= 0.0 || dt <= 0.0 {
        return Err(SciError::InvalidParameter(
            "diffusivity, dx, and dt must be positive",
        ));
    }

    let n = initial.len();
    let interior = n - 2; // number of interior nodes
    let r = diffusivity * dt / (2.0 * dx * dx); // Crank–Nicolson factor

    // Tridiagonal system:  (-r)u_{i-1} + (1+2r)u_i + (-r)u_{i+1} = rhs_i
    let diag_main = 1.0 + 2.0 * r;
    let diag_off = -r;

    let mut states = Vec::with_capacity(steps + 1);
    let mut current = initial.to_vec();
    states.push(current.clone());

    for _ in 0..steps {
        // Build RHS from explicit half-step.
        let mut rhs: Vec<f64> = (1..=interior)
            .map(|i| r * current[i - 1] + (1.0 - 2.0 * r) * current[i] + r * current[i + 1])
            .collect();

        // Thomas forward sweep.
        let mut c_prime = vec![0.0f64; interior];
        let mut d_prime = vec![0.0f64; interior];
        c_prime[0] = diag_off / diag_main;
        d_prime[0] = rhs[0] / diag_main;
        for i in 1..interior {
            let denom = diag_main - diag_off * c_prime[i - 1];
            if denom.abs() < f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            c_prime[i] = diag_off / denom;
            d_prime[i] = (rhs[i] - diag_off * d_prime[i - 1]) / denom;
        }

        // Back substitution.
        let mut u_interior = vec![0.0f64; interior];
        u_interior[interior - 1] = d_prime[interior - 1];
        for i in (0..(interior - 1)).rev() {
            u_interior[i] = d_prime[i] - c_prime[i] * u_interior[i + 1];
        }

        let mut next = vec![0.0f64; n];
        next[0] = current[0]; // Dirichlet BC
        next[n - 1] = current[n - 1];
        for (i, &val) in u_interior.iter().enumerate() {
            next[i + 1] = val;
        }
        states.push(next.clone());
        current = next;
        let _ = rhs.as_mut_slice(); // suppress unused warning
    }
    Ok(states)
}

// ── 2-D Poisson SOR ──────────────────────────────────────────────────────────

/// Solve `∇²u = f` on a 2-D grid with Dirichlet boundary conditions
/// using Successive Over-Relaxation (SOR).
///
/// # Arguments
/// - `f`       — source term grid (rows × cols); boundary values in `u` are fixed
/// - `u`       — initial guess; boundary values must be set by the caller
/// - `dx`      — uniform spatial step (same in both directions)
/// - `omega`   — SOR relaxation factor (1 = Gauss–Seidel; 1.5–1.9 for acceleration)
/// - `tolerance` — convergence criterion on max-norm update
/// - `max_iter` — iteration limit
pub fn poisson_2d(
    f: &[Vec<f64>],
    initial_u: &[Vec<f64>],
    dx: f64,
    omega: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<Vec<f64>>> {
    let rows = f.len();
    if rows < 3 {
        return Err(SciError::InvalidParameter("grid must have at least 3 rows"));
    }
    let cols = f[0].len();
    if cols < 3 {
        return Err(SciError::InvalidParameter(
            "grid must have at least 3 columns",
        ));
    }
    if f.iter().any(|r| r.len() != cols) || initial_u.len() != rows {
        return Err(SciError::InvalidParameter(
            "f and u must have matching dimensions",
        ));
    }
    if initial_u.iter().any(|r| r.len() != cols) {
        return Err(SciError::InvalidParameter(
            "all rows of u must have the same length",
        ));
    }
    if dx <= 0.0 {
        return Err(SciError::InvalidParameter("dx must be positive"));
    }
    if !(1.0..2.0).contains(&omega) {
        return Err(SciError::InvalidParameter("omega must be in [1, 2)"));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let dx2 = dx * dx;
    let mut u = initial_u.to_vec();

    for _ in 0..max_iter {
        let mut max_change = 0.0_f64;
        for row in 1..(rows - 1) {
            for col in 1..(cols - 1) {
                let new_val = 0.25
                    * (u[row - 1][col] + u[row + 1][col] + u[row][col - 1] + u[row][col + 1]
                        - dx2 * f[row][col]);
                let update = omega * (new_val - u[row][col]);
                max_change = max_change.max(update.abs());
                u[row][col] += update;
            }
        }
        if max_change < tolerance {
            return Ok(u);
        }
    }
    Err(SciError::NonConvergent("Poisson SOR"))
}
