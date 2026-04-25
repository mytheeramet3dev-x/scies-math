//! ODE solvers — fixed-step and adaptive methods for dy/dt = f(t, y).
//!
//! # Methods
//!
//! | Function | Type | Order | Notes |
//! |---|---|---|---|
//! | `euler` | Fixed | 1 | Simplest; large error accumulation |
//! | `rk4` | Fixed | 4 | Classical Runge-Kutta; workhorse for smooth problems |
//! | `rk4_system` | Fixed | 4 | Vector-state version of RK4 |
//! | `rk45` | Adaptive | 4/5 | Dormand-Prince; auto step-size via tolerance |
//! | `euler_system` | Fixed | 1 | Euler for vector-valued ODE |
//!
//! # Usage
//!
//! ```rust
//! use scies_math::ode::rk4;
//!
//! // dy/dt = -y, y(0) = 1  →  exact: e^{-t}
//! let sol = rk4(|_t, y| vec![-y[0]], 0.0, vec![1.0], 0.01, 1000);
//! // sol.y.last() ≈ e^{-10} ≈ 4.54e-5
//! ```
//!
//! # Adaptive step-size (RK45)
//!
//! ```rust
//! use scies_math::ode::rk45;
//!
//! let sol = rk45(
//!     |_t, y| vec![-y[0]],   // f(t, y)
//!     0.0, 5.0,              // t_start, t_end
//!     vec![1.0],             // y0
//!     1e-6, 1e-9,            // rtol, atol
//!     0.1,                   // initial step
//! );
//! ```
//!
//! # Error handling
//!
//! All solvers return `OdeSolution { t: Vec<f64>, y: Vec<Vec<f64>> }`.
//! Adaptive solvers may return fewer steps than fixed-step solvers.
use crate::errors::{SciError, SciResult};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Solution trajectory returned by all vector ODE solvers.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// Time stamps.
    pub t: Vec<f64>,
    /// `y[i]` is the state vector at time `t[i]`.
    pub y: Vec<Vec<f64>>,
    /// Number of function evaluations (adaptive solvers only; 0 otherwise).
    pub n_evals: usize,
}

impl OdeSolution {
    /// State at the final time point.
    pub fn final_state(&self) -> Option<&[f64]> {
        self.y.last().map(|v| v.as_slice())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-step — vector state
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-step **Euler** method for `dy/dt = f(t, y)`.
///
/// Integrates from `t0` to `t_end` using step `dt`.
pub fn euler_system<F>(f: F, y0: &[f64], t0: f64, t_end: f64, dt: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_ivp(y0, t0, t_end, dt)?;
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];

    while t < t_end - dt * 0.5 {
        let step = dt.min(t_end - t);
        let dy = f(t, &y);
        for (yi, dyi) in y.iter_mut().zip(dy.iter()) {
            *yi += step * dyi;
        }
        t += step;
        ts.push(t);
        ys.push(y.clone());
    }
    Ok(OdeSolution {
        t: ts,
        y: ys,
        n_evals: 0,
    })
}

/// Fixed-step **classical RK4** for `dy/dt = f(t, y)`.
pub fn rk4_system<F>(f: F, y0: &[f64], t0: f64, t_end: f64, dt: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_ivp(y0, t0, t_end, dt)?;
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];

    while t < t_end - dt * 0.5 {
        let h = dt.min(t_end - t);
        let k1 = f(t, &y);
        let yt = axpy_new(h / 2.0, &k1, &y, n);
        let k2 = f(t + h / 2.0, &yt);
        let yt = axpy_new(h / 2.0, &k2, &y, n);
        let k3 = f(t + h / 2.0, &yt);
        let yt = axpy_new(h, &k3, &y, n);
        let k4 = f(t + h, &yt);
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        ts.push(t);
        ys.push(y.clone());
    }
    Ok(OdeSolution {
        t: ts,
        y: ys,
        n_evals: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive RK45 — Dormand-Prince (DOPRI5)
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive **RK45 (Dormand-Prince)** solver for `dy/dt = f(t, y)`.
///
/// Automatically adjusts the step size to keep the local error below
/// `atol + rtol * |y|`.  Recommended for most problems.
///
/// # Parameters
/// - `rtol` — relative tolerance (e.g. `1e-6`)
/// - `atol` — absolute tolerance (e.g. `1e-9`)
/// - `max_steps` — hard limit on accepted steps
pub fn rk45<F>(
    f: F,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_tol(rtol, atol)?;
    if y0.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if t_end <= t0 {
        return Err(SciError::InvalidParameter("t_end must be > t0"));
    }

    // Dormand-Prince coefficients
    const C2: f64 = 1.0 / 5.0;
    const C3: f64 = 3.0 / 10.0;
    const C4: f64 = 4.0 / 5.0;
    const C5: f64 = 8.0 / 9.0;

    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19_372.0 / 6_561.0;
    const A52: f64 = -25_360.0 / 2_187.0;
    const A53: f64 = 64_448.0 / 6_561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9_017.0 / 3_168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46_732.0 / 5_247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5_103.0 / 18_656.0;

    // 5th-order weights
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1_113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2_187.0 / 6_784.0;
    const B6: f64 = 11.0 / 84.0;

    // Error coefficients (B5th - B4th)
    const E1: f64 = 71.0 / 57_600.0;
    const E3: f64 = -71.0 / 16_695.0;
    const E4: f64 = 71.0 / 1_920.0;
    const E5: f64 = -17_253.0 / 339_200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = initial_step(&f, t, &y, rtol, atol);
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    let mut n_evals = 1usize; // k1 is computed from y0

    let mut k1 = f(t, &y);

    for _ in 0..max_steps {
        if t >= t_end {
            break;
        }
        h = h.min(t_end - t);

        // Stage evaluations
        let yt = combine(&y, &[(h * A21, &k1)], n);
        let k2 = f(t + C2 * h, &yt);
        let yt = combine(&y, &[(h * A31, &k1), (h * A32, &k2)], n);
        let k3 = f(t + C3 * h, &yt);
        let yt = combine(&y, &[(h * A41, &k1), (h * A42, &k2), (h * A43, &k3)], n);
        let k4 = f(t + C4 * h, &yt);
        let yt = combine(
            &y,
            &[
                (h * A51, &k1),
                (h * A52, &k2),
                (h * A53, &k3),
                (h * A54, &k4),
            ],
            n,
        );
        let k5 = f(t + C5 * h, &yt);
        let yt = combine(
            &y,
            &[
                (h * A61, &k1),
                (h * A62, &k2),
                (h * A63, &k3),
                (h * A64, &k4),
                (h * A65, &k5),
            ],
            n,
        );
        let k6 = f(t + h, &yt);
        n_evals += 5;

        // 5th-order solution
        let y_new = combine(
            &y,
            &[
                (h * B1, &k1),
                (h * B3, &k3),
                (h * B4, &k4),
                (h * B5, &k5),
                (h * B6, &k6),
            ],
            n,
        );
        let k7 = f(t + h, &y_new);
        n_evals += 1;

        // Error estimate
        let err_norm = rms_norm(
            &(0..n)
                .map(|i| {
                    h * (E1 * k1[i]
                        + E3 * k3[i]
                        + E4 * k4[i]
                        + E5 * k5[i]
                        + E6 * k6[i]
                        + E7 * k7[i])
                })
                .collect::<Vec<_>>(),
            &y,
            &y_new,
            atol,
            rtol,
        );

        // Step-size control (PI controller)
        let factor = 0.9 * err_norm.powf(-0.2).clamp(0.1, 10.0);
        let h_new = h * factor;

        if err_norm <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            k1 = k7; // FSAL
            ts.push(t);
            ys.push(y.clone());
        }
        h = h_new;
    }

    if t < t_end - f64::EPSILON * t_end.abs() {
        return Err(SciError::NonConvergent("RK45: max_steps exceeded"));
    }
    Ok(OdeSolution {
        t: ts,
        y: ys,
        n_evals,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive RK23 — Bogacki-Shampine
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive **RK23 (Bogacki-Shampine)** solver — lightweight 3rd-order method.
///
/// Good for smooth, non-stiff problems where lower accuracy is acceptable.
/// Uses the same tolerance interface as [`rk45`].
pub fn rk23<F>(
    f: F,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_tol(rtol, atol)?;
    if y0.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if t_end <= t0 {
        return Err(SciError::InvalidParameter("t_end must be > t0"));
    }

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = initial_step(&f, t, &y, rtol, atol);
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    let mut n_evals = 1usize;

    let mut k1 = f(t, &y);

    for _ in 0..max_steps {
        if t >= t_end {
            break;
        }
        h = h.min(t_end - t);

        // Bogacki-Shampine stages
        let yt = axpy_new(h / 2.0, &k1, &y, n);
        let k2 = f(t + h / 2.0, &yt);
        let yt = axpy_new(3.0 * h / 4.0, &k2, &y, n);
        let k3 = f(t + 3.0 * h / 4.0, &yt);
        n_evals += 2;

        // 3rd-order solution
        let y_new = combine(
            &y,
            &[(2.0 * h / 9.0, &k1), (h / 3.0, &k2), (4.0 * h / 9.0, &k3)],
            n,
        );
        let k4 = f(t + h, &y_new);
        n_evals += 1;

        // Error: difference between 3rd and 2nd order
        let err: Vec<f64> = (0..n)
            .map(|i| {
                h * (5.0 / 72.0 * k1[i] - 1.0 / 12.0 * k2[i] - 1.0 / 9.0 * k3[i]
                    + 1.0 / 8.0 * k4[i])
            })
            .collect();
        let err_norm = rms_norm(&err, &y, &y_new, atol, rtol);

        let factor = 0.9 * err_norm.powf(-1.0 / 3.0).clamp(0.1, 10.0);
        let h_new = h * factor;

        if err_norm <= 1.0 {
            t += h;
            y = y_new;
            k1 = k4; // FSAL
            ts.push(t);
            ys.push(y.clone());
        }
        h = h_new;
    }

    if t < t_end - f64::EPSILON * t_end.abs() {
        return Err(SciError::NonConvergent("RK23: max_steps exceeded"));
    }
    Ok(OdeSolution {
        t: ts,
        y: ys,
        n_evals,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward-compatible scalar API
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar Euler solver — kept for backward compatibility.
/// Prefer [`euler_system`] for new code.
pub fn euler_solve<F>(
    derivative: F,
    initial_state: f64,
    initial_time: f64,
    dt: f64,
    steps: usize,
) -> SciResult<Vec<(f64, f64)>>
where
    F: Fn(f64, f64) -> f64,
{
    if dt <= 0.0 {
        return Err(SciError::InvalidParameter("dt must be positive"));
    }
    let mut time = initial_time;
    let mut state = initial_state;
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push((time, state));
    for _ in 0..steps {
        state += derivative(time, state) * dt;
        time += dt;
        trajectory.push((time, state));
    }
    Ok(trajectory)
}

/// Scalar RK4 solver — kept for backward compatibility.
/// Prefer [`rk4_system`] for new code.
pub fn rk4_solve<F>(
    derivative: F,
    initial_state: f64,
    initial_time: f64,
    dt: f64,
    steps: usize,
) -> SciResult<Vec<(f64, f64)>>
where
    F: Fn(f64, f64) -> f64,
{
    if dt <= 0.0 {
        return Err(SciError::InvalidParameter("dt must be positive"));
    }
    let mut time = initial_time;
    let mut state = initial_state;
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push((time, state));
    for _ in 0..steps {
        let k1 = derivative(time, state);
        let k2 = derivative(time + dt / 2.0, state + dt * k1 / 2.0);
        let k3 = derivative(time + dt / 2.0, state + dt * k2 / 2.0);
        let k4 = derivative(time + dt, state + dt * k3);
        state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        time += dt;
        trajectory.push((time, state));
    }
    Ok(trajectory)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// y + alpha * dx
fn axpy_new(alpha: f64, dx: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| y[i] + alpha * dx[i]).collect()
}

/// y + sum_j (alpha_j * k_j)
fn combine(y: &[f64], terms: &[(f64, &[f64])], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| y[i] + terms.iter().map(|(a, k)| a * k[i]).sum::<f64>())
        .collect()
}

/// Mixed-tolerance RMS error norm.
fn rms_norm(err: &[f64], y: &[f64], y_new: &[f64], atol: f64, rtol: f64) -> f64 {
    let n = err.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = err
        .iter()
        .zip(y.iter().zip(y_new.iter()))
        .map(|(&e, (&yi, &yni))| {
            let scale = atol + rtol * yi.abs().max(yni.abs());
            (e / scale).powi(2)
        })
        .sum();
    (sum / n as f64).sqrt()
}

/// Rough initial step size estimate (Hairer et al. §II.4).
fn initial_step<F>(f: &F, t0: f64, y0: &[f64], rtol: f64, atol: f64) -> f64
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let f0 = f(t0, y0);
    let d0 = rms_norm(y0, y0, y0, atol, rtol).max(f64::EPSILON);
    let d1 = rms_norm(&f0, y0, y0, atol, rtol).max(f64::EPSILON);
    let h0 = 0.01 * d0 / d1;

    let y1: Vec<f64> = y0
        .iter()
        .zip(f0.iter())
        .map(|(y, dy)| y + h0 * dy)
        .collect();
    let f1 = f(t0 + h0, &y1);
    let diff: Vec<f64> = f1
        .iter()
        .zip(f0.iter())
        .map(|(a, b)| (a - b) / h0)
        .collect();
    let d2 = rms_norm(&diff, y0, y0, atol, rtol).max(f64::EPSILON);

    let h1 = (0.01 / d2).powf(1.0 / 5.0).min(100.0 * h0);
    h1.min(1.0) // cap at 1.0 for safety
}

fn validate_ivp(y0: &[f64], t0: f64, t_end: f64, dt: f64) -> SciResult<()> {
    if y0.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(SciError::InvalidParameter("dt must be positive"));
    }
    if t_end <= t0 {
        return Err(SciError::InvalidParameter("t_end must be > t0"));
    }
    Ok(())
}

fn validate_tol(rtol: f64, atol: f64) -> SciResult<()> {
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(SciError::InvalidParameter("rtol and atol must be positive"));
    }
    Ok(())
}
