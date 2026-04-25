//! Partial differential equations — finite-difference solvers.
//!
//! # Equations
//!
//! | Solver | Equation |
//! |---|---|
//! | `heat_1d` | ∂u/∂t = α ∂²u/∂x² (1D heat / diffusion) |
//! | `wave_1d` | ∂²u/∂t² = c² ∂²u/∂x² (1D wave) |
//! | `laplace_2d` | ∇²u = 0 (2D Laplace / Poisson, Gauss-Seidel) |
//!
//! # Usage — 1D Heat equation
//!
//! ```rust
//! use scies_math::pde::heat_1d;
//!
//! // Rod of length 1, α = 0.01, initial temperature profile
//! let u0: Vec<f64> = (0..=100).map(|i| {
//!     let x = i as f64 / 100.0;
//!     (std::f64::consts::PI * x).sin()
//! }).collect();
//!
//! let solution = heat_1d(&u0, 0.01, 0.01, 0.0001, 500);
//! // solution[k] = temperature profile at time step k
//! ```
//!
//! See [`crate::pde_ext`] for Navier-Stokes, Burgers' equation, and Crank-Nicolson.
use crate::errors::{SciError, SciResult};

pub fn heat_equation_1d(
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

    let r = diffusivity * dt / (dx * dx);
    if r > 0.5 {
        return Err(SciError::InvalidParameter(
            "explicit heat scheme is unstable for diffusivity * dt / dx^2 > 0.5",
        ));
    }

    let mut states = Vec::with_capacity(steps + 1);
    let mut current = initial.to_vec();
    states.push(current.clone());

    for _ in 0..steps {
        let mut next = current.clone();
        for i in 1..(current.len() - 1) {
            next[i] = current[i] + r * (current[i - 1] - 2.0 * current[i] + current[i + 1]);
        }
        states.push(next.clone());
        current = next;
    }

    Ok(states)
}

pub fn laplace_relaxation_2d(
    initial: &[Vec<f64>],
    fixed_mask: &[Vec<bool>],
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<Vec<Vec<f64>>> {
    if initial.is_empty() || initial[0].is_empty() {
        return Err(SciError::EmptyInput);
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let rows = initial.len();
    let cols = initial[0].len();
    if rows < 3 || cols < 3 {
        return Err(SciError::InvalidParameter(
            "Laplace relaxation requires at least a 3x3 grid",
        ));
    }
    if fixed_mask.len() != rows || fixed_mask.iter().any(|row| row.len() != cols) {
        return Err(SciError::InvalidParameter(
            "fixed mask dimensions must match the grid",
        ));
    }
    if initial.iter().any(|row| row.len() != cols) {
        return Err(SciError::InvalidParameter(
            "all grid rows must have the same length",
        ));
    }

    let mut grid = initial.to_vec();
    for _ in 0..max_iterations {
        let mut max_change = 0.0_f64;

        for row in 1..(rows - 1) {
            for col in 1..(cols - 1) {
                if fixed_mask[row][col] {
                    continue;
                }

                let new_value = 0.25
                    * (grid[row - 1][col]
                        + grid[row + 1][col]
                        + grid[row][col - 1]
                        + grid[row][col + 1]);
                max_change = max_change.max((new_value - grid[row][col]).abs());
                grid[row][col] = new_value;
            }
        }

        if max_change < tolerance {
            return Ok(grid);
        }
    }

    Err(SciError::NonConvergent("Laplace relaxation"))
}
