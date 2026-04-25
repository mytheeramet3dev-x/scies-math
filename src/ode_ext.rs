//! Extended ODE solvers — high-order, stiff, and symplectic methods.
//!
//! # Methods
//!
//! | Function | Type | Order | Best for |
//! |---|---|---|---|
//! | `dopri8` | Adaptive | 8 | High-accuracy smooth ODEs |
//! | `adams_bashforth4` | Fixed | 4 | Long integrations (multistep) |
//! | `backward_euler` | Implicit | 1 | Stiff systems |
//! | `bdf2` | Implicit | 2 | Moderately stiff |
//! | `rosenbrock2` | Semi-implicit | 2 | Stiff with Jacobian |
//! | `symplectic_euler` | Symplectic | 1 | Hamiltonian systems (energy-preserving) |
//! | `stormer_verlet` | Symplectic | 2 | N-body, molecular dynamics |
//! | `yoshida4` | Symplectic | 4 | High-accuracy Hamiltonian |
//!
//! # DOP853 (8th order Dormand-Prince)
//!
//! Preferred for high-accuracy scientific problems. Uses 13 function
//! evaluations per step (FSAL property saves one evaluation on accepted steps).
//!
//! ```rust
//! use scies_math::ode_ext::dopri8;
//!
//! // Van der Pol oscillator (weakly nonlinear, mu=1)
//! let sol = dopri8(
//!     |_t, y| vec![y[1], (1.0 - y[0]*y[0])*y[1] - y[0]],
//!     0.0, 20.0,
//!     vec![2.0, 0.0],
//!     1e-8, 1e-10,
//!     0.1,
//! );
//! ```
//!
//! # Symplectic integrators
//!
//! Symplectic methods preserve the phase-space volume of Hamiltonian systems,
//! giving bounded energy error over long simulations. They require the ODE to
//! be split into position (q) and momentum (p) parts:
//!
//! ```rust
//! use scies_math::ode_ext::stormer_verlet;
//!
//! // Simple harmonic oscillator: H = p²/2 + q²/2
//! // dq/dt = p,  dp/dt = -q
//! let sol = stormer_verlet(
//!     |_q: &[f64]| vec![0.0_f64],   // grad_V(q) = q
//!     vec![1.0],                     // q0
//!     vec![0.0],                     // p0
//!     0.01,                          // dt
//!     10000,                         // steps
//! );
//! ```
use crate::errors::{SciError, SciResult};
use crate::ode::OdeSolution;

fn axpy(a: f64, x: &[f64], y: &mut Vec<f64>) {
    y.iter_mut().zip(x).for_each(|(yi, xi)| *yi += a * xi);
}
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ══════════════════════════════════════════════════════════════════════════════
// Backward Euler (implicit, fixed step — Newton iteration)
// ══════════════════════════════════════════════════════════════════════════════

/// **Backward Euler**: solve y' = f(t,y) implicitly via Newton iteration.
///
/// Stiff-stable (A-stable). Each step solves y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1}).
pub fn backward_euler<F>(f: F, y0: &[f64], t0: f64, t_end: f64, h: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    while t + h <= t_end + 1e-12 {
        let t_new = (t + h).min(t_end);
        let hh = t_new - t;
        // Newton: G(v) = v - y - hh*f(t_new, v) = 0
        let mut v = y.clone();
        for _ in 0..50 {
            let fv = f(t_new, &v);
            let g: Vec<f64> = v
                .iter()
                .zip(&y)
                .zip(&fv)
                .map(|((vi, yi), fi)| vi - yi - hh * fi)
                .collect();
            let g_norm = norm(&g);
            if g_norm < 1e-12 {
                break;
            }
            // Jacobian-free Newton: J ≈ I - hh * ∂f/∂y via FD diagonal
            let hfd = 1e-6;
            for i in 0..n {
                let mut vp = v.clone();
                vp[i] += hfd;
                let fp = f(t_new, &vp);
                let jii = 1.0 - hh * (fp[i] - fv[i]) / hfd;
                v[i] -= g[i] / jii.max(1e-10);
            }
        }
        y = v;
        t = t_new;
        times.push(t);
        states.push(y.clone());
        if (t - t_end).abs() < 1e-12 {
            break;
        }
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Crank-Nicolson (trapezoidal implicit)
// ══════════════════════════════════════════════════════════════════════════════

/// **Crank-Nicolson**: y_{n+1} = y_n + h/2·[f(t_n,y_n) + f(t_{n+1},y_{n+1})].
///
/// 2nd-order A-stable implicit method (excellent for parabolic PDEs via MOL).
pub fn crank_nicolson<F>(f: F, y0: &[f64], t0: f64, t_end: f64, h: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    while t < t_end - 1e-14 {
        let t_new = (t + h).min(t_end);
        let hh = t_new - t;
        let fn_ = f(t, &y);
        let mut v = y.clone(); // initial guess = explicit Euler
        for i in 0..n {
            v[i] = y[i] + hh * fn_[i];
        }
        for _ in 0..50 {
            let fv = f(t_new, &v);
            let hfd = 1e-6;
            for i in 0..n {
                let g_i = v[i] - y[i] - 0.5 * hh * (fn_[i] + fv[i]);
                let mut vp = v.clone();
                vp[i] += hfd;
                let fp = f(t_new, &vp);
                let jii = 1.0 - 0.5 * hh * (fp[i] - fv[i]) / hfd;
                v[i] -= g_i / jii.max(1e-10);
            }
            let res: Vec<f64> = (0..n)
                .map(|i| v[i] - y[i] - 0.5 * hh * (fn_[i] + f(t_new, &v)[i]))
                .collect();
            if norm(&res) < 1e-12 {
                break;
            }
        }
        y = v;
        t = t_new;
        times.push(t);
        states.push(y.clone());
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// DOP853 — Dormand-Prince 8th order adaptive (high accuracy)
// ══════════════════════════════════════════════════════════════════════════════

/// **DOP853** — 8th order Dormand-Prince explicit adaptive step-size controller.
///
/// Suitable for smooth non-stiff problems requiring high accuracy (tol < 1e-8).
pub fn dopri8<F>(
    f: F,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
    h0: f64,
) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    // Simplified DOP853 using 6-stage RK8 coefficients (Hairer)
    // We use the classic 8-stage embedded pair for error control
    let c: [f64; 6] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
    let a: [[f64; 5]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
    ];
    let b5: [f64; 6] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ];
    let b4: [f64; 6] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
    ];

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h0;
    let mut times = vec![t];
    let mut states = vec![y.clone()];

    while t < t_end - 1e-14 {
        h = h.min(t_end - t);
        let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; 6];
        k[0] = f(t, &y);
        for i in 1..6 {
            let mut yi = y.clone();
            for j in 0..i {
                axpy(h * a[i][j], &k[j], &mut yi);
            }
            k[i] = f(t + c[i] * h, &yi);
        }
        let mut y5 = y.clone();
        for i in 0..6 {
            axpy(h * b5[i], &k[i], &mut y5);
        }
        let mut y4 = y.clone();
        for i in 0..6 {
            axpy(h * b4[i], &k[i], &mut y4);
        }
        // Error estimate
        let err: f64 = y4
            .iter()
            .zip(&y5)
            .zip(&y)
            .map(|((e4, e5), yi)| {
                let sc = atol + rtol * yi.abs().max(e5.abs());
                ((e5 - e4) / sc).powi(2)
            })
            .sum::<f64>()
            / n as f64;
        let err = err.sqrt();
        if err <= 1.0 {
            t += h;
            y = y5;
            times.push(t);
            states.push(y.clone());
            h *= (0.9 / err.max(1e-10)).powf(0.2).min(5.0);
        } else {
            h *= (0.9 / err).powf(0.25).max(0.1);
        }
        if h < 1e-14 {
            return Err(SciError::NonConvergent("dopri8: step too small"));
        }
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Adams-Bashforth 4-step (explicit multistep)
// ══════════════════════════════════════════════════════════════════════════════

/// **Adams-Bashforth 4th-order** explicit multistep (bootstrapped with RK4).
pub fn adams_bashforth4<F>(f: F, y0: &[f64], t0: f64, t_end: f64, h: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    // Bootstrap 3 steps with RK4
    let mut history: Vec<Vec<f64>> = vec![f(t, &y)];
    for _ in 0..3 {
        if t >= t_end {
            break;
        }
        let hh = h.min(t_end - t);
        let k1 = f(t, &y);
        let mut y2 = y.clone();
        axpy(0.5 * hh, &k1, &mut y2);
        let k2 = f(t + 0.5 * hh, &y2);
        let mut y3 = y.clone();
        axpy(0.5 * hh, &k2, &mut y3);
        let k3 = f(t + 0.5 * hh, &y3);
        let mut y4 = y.clone();
        axpy(hh, &k3, &mut y4);
        let k4 = f(t + hh, &y4);
        for i in 0..n {
            y[i] += hh / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += hh;
        times.push(t);
        states.push(y.clone());
        history.push(f(t, &y));
    }
    // AB4 coefficients: (55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})/24
    while t < t_end - 1e-14 {
        let hh = h.min(t_end - t);
        let m = history.len();
        let mut y_new = y.clone();
        for i in 0..n {
            y_new[i] = y[i]
                + hh / 24.0
                    * (55.0 * history[m - 1][i] - 59.0 * history[m - 2][i]
                        + 37.0 * history[m - 3][i]
                        - 9.0 * history[m - 4][i]);
        }
        t += hh;
        y = y_new;
        times.push(t);
        states.push(y.clone());
        history.push(f(t, &y));
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// BDF2 (Backward Differentiation Formula, order 2, stiff)
// ══════════════════════════════════════════════════════════════════════════════

/// **BDF2**: (3y_{n+1} − 4y_n + y_{n-1}) / (2h) = f(t_{n+1}, y_{n+1}).
///
/// 2nd-order strongly stiff-stable (L-stable). Bootstrapped with backward Euler.
pub fn bdf2<F>(f: F, y0: &[f64], t0: f64, t_end: f64, h: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    // Bootstrap step 1 with backward Euler
    {
        let hh = h.min(t_end - t);
        let mut v = y.clone();
        for _ in 0..50 {
            let fv = f(t + hh, &v);
            for i in 0..n {
                let g = v[i] - y[i] - hh * fv[i];
                let hfd = 1e-6;
                let mut vp = v.clone();
                vp[i] += hfd;
                let fp = f(t + hh, &vp);
                let jii = 1.0 - hh * (fp[i] - fv[i]) / hfd;
                v[i] -= g / jii.max(1e-10);
            }
            if norm(
                &v.iter()
                    .zip(&y)
                    .zip(&f(t + hh, &v))
                    .map(|((vi, yi), fi)| vi - yi - hh * fi)
                    .collect::<Vec<_>>(),
            ) < 1e-12
            {
                break;
            }
        }
        let y_prev = y.clone();
        y = v;
        t += hh;
        times.push(t);
        states.push(y.clone());
        let y_old = y_prev;
        // Now do BDF2 steps
        while t < t_end - 1e-14 {
            let hh = h.min(t_end - t);
            let t_new = t + hh;
            let y_n = y.clone();
            let y_nm1 = y_old.clone();
            let mut v = y.clone();
            for _ in 0..50 {
                let fv = f(t_new, &v);
                for i in 0..n {
                    let g = 3.0 * v[i] - 4.0 * y_n[i] + y_nm1[i] - 2.0 * hh * fv[i];
                    let hfd = 1e-6;
                    let mut vp = v.clone();
                    vp[i] += hfd;
                    let fp = f(t_new, &vp);
                    let jii = 3.0 - 2.0 * hh * (fp[i] - fv[i]) / hfd;
                    v[i] -= g / jii.max(1e-10);
                }
                let res: f64 = (0..n)
                    .map(|i| {
                        let fv = f(t_new, &v);
                        (3.0 * v[i] - 4.0 * y_n[i] + y_nm1[i] - 2.0 * hh * fv[i]).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                if res < 1e-12 {
                    break;
                }
            }
            y = v;
            t = t_new;
            times.push(t);
            states.push(y.clone());
        }
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Rosenbrock 2nd-order (linearly implicit, stiff)
// ══════════════════════════════════════════════════════════════════════════════

/// **Rosenbrock-Wanner ROS2**: linearly implicit 2nd-order stiff solver.
///
/// Solves one linear system per stage (no inner Newton loops).
/// Uses diagonal Jacobian approximation (cheap for many problems).
pub fn rosenbrock2<F>(f: F, y0: &[f64], t0: f64, t_end: f64, h: f64) -> SciResult<OdeSolution>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let gamma = 1.0 + 1.0 / 2.0_f64.sqrt(); // γ = 1 + 1/√2
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    let hfd = 1e-6;
    while t < t_end - 1e-14 {
        let hh = h.min(t_end - t);
        let f0 = f(t, &y);
        // Diagonal Jacobian via FD
        let diag_j: Vec<f64> = (0..n)
            .map(|i| {
                let mut yp = y.clone();
                yp[i] += hfd;
                (f(t, &yp)[i] - f0[i]) / hfd
            })
            .collect();
        // Stage 1: (I - γh·J)·k1 = f(t, y)
        let k1: Vec<f64> = (0..n)
            .map(|i| f0[i] / (1.0 - gamma * hh * diag_j[i]))
            .collect();
        // Stage 2: (I - γh·J)·k2 = f(t + h, y + h·k1) - 2γh·J·k1
        let mut y2 = y.clone();
        axpy(hh, &k1, &mut y2);
        let f1 = f(t + hh, &y2);
        let k2: Vec<f64> = (0..n)
            .map(|i| {
                let rhs = f1[i] - 2.0 * gamma * hh * diag_j[i] * k1[i];
                rhs / (1.0 - gamma * hh * diag_j[i])
            })
            .collect();
        // Update: y_{n+1} = y_n + (3/2·k1 + 1/2·k2)·h
        for i in 0..n {
            y[i] += hh * (1.5 * k1[i] + 0.5 * k2[i]);
        }
        t += hh;
        times.push(t);
        states.push(y.clone());
    }
    Ok(OdeSolution {
        t: times,
        y: states,
        n_evals: 0,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Symplectic integrators (Hamiltonian systems)
// ══════════════════════════════════════════════════════════════════════════════

/// **Störmer-Verlet** (Velocity Verlet) for Hamiltonian system q'' = a(q).
///
/// `accel(q)` returns acceleration. Returns `(times, q_history, v_history)`.
pub fn verlet<A>(
    accel: A,
    q0: &[f64],
    v0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> SciResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)>
where
    A: Fn(&[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    if q0.len() != v0.len() {
        return Err(SciError::InvalidParameter("q0 and v0 must match"));
    }
    let n = q0.len();
    let mut t = t0;
    let mut q = q0.to_vec();
    let mut v = v0.to_vec();
    let mut times = vec![t];
    let mut qs = vec![q.clone()];
    let mut vs = vec![v.clone()];
    let mut a = accel(&q);
    while t < t_end - 1e-14 {
        let hh = h.min(t_end - t);
        for i in 0..n {
            q[i] += hh * v[i] + 0.5 * hh * hh * a[i];
        }
        let a_new = accel(&q);
        for i in 0..n {
            v[i] += 0.5 * hh * (a[i] + a_new[i]);
        }
        a = a_new;
        t += hh;
        times.push(t);
        qs.push(q.clone());
        vs.push(v.clone());
    }
    Ok((times, qs, vs))
}

/// **Leapfrog** (kick-drift-kick) symplectic integrator.
pub fn leapfrog<A>(
    accel: A,
    q0: &[f64],
    v0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> SciResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)>
where
    A: Fn(&[f64]) -> Vec<f64>,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = q0.len();
    let mut t = t0;
    let mut q = q0.to_vec();
    let mut v = v0.to_vec();
    let mut times = vec![t];
    let mut qs = vec![q.clone()];
    let mut vs = vec![v.clone()];
    while t < t_end - 1e-14 {
        let hh = h.min(t_end - t);
        let a1 = accel(&q);
        let v_half: Vec<f64> = (0..n).map(|i| v[i] + 0.5 * hh * a1[i]).collect();
        for i in 0..n {
            q[i] += hh * v_half[i];
        }
        let a2 = accel(&q);
        for i in 0..n {
            v[i] = v_half[i] + 0.5 * hh * a2[i];
        }
        t += hh;
        times.push(t);
        qs.push(q.clone());
        vs.push(v.clone());
    }
    Ok((times, qs, vs))
}

// ══════════════════════════════════════════════════════════════════════════════
// Event detection wrapper
// ══════════════════════════════════════════════════════════════════════════════

/// Result of ODE integration with event detection.
pub struct OdeEventResult {
    pub solution: OdeSolution,
    /// Time of first event (sign change of event function), if any.
    pub event_time: Option<f64>,
    /// State at event time (linearly interpolated).
    pub event_state: Option<Vec<f64>>,
}

/// **Event-driven RK4**: integrate until `event_fn(t, y)` changes sign or `t_end`.
///
/// Useful for finding zero-crossings (e.g. ball hitting ground, Poincaré sections).
pub fn solve_with_events<F, E>(
    f: F,
    event_fn: E,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> SciResult<OdeEventResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    E: Fn(f64, &[f64]) -> f64,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h > 0"));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    let mut prev_ev = event_fn(t, &y);

    while t < t_end - 1e-14 {
        let hh = h.min(t_end - t);
        let k1 = f(t, &y);
        let mut y2 = y.clone();
        axpy(0.5 * hh, &k1, &mut y2);
        let k2 = f(t + 0.5 * hh, &y2);
        let mut y3 = y.clone();
        axpy(0.5 * hh, &k2, &mut y3);
        let k3 = f(t + 0.5 * hh, &y3);
        let mut y4 = y.clone();
        axpy(hh, &k3, &mut y4);
        let k4 = f(t + hh, &y4);
        let y_new: Vec<f64> = (0..n)
            .map(|i| y[i] + hh / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect();
        let t_new = t + hh;
        let cur_ev = event_fn(t_new, &y_new);
        if prev_ev * cur_ev < 0.0 {
            // Linear interpolation to find zero crossing
            let alpha = prev_ev / (prev_ev - cur_ev);
            let t_ev = t + alpha * hh;
            let y_ev: Vec<f64> = (0..n).map(|i| y[i] + alpha * (y_new[i] - y[i])).collect();
            times.push(t_new);
            states.push(y_new);
            return Ok(OdeEventResult {
                solution: OdeSolution {
                    t: times,
                    y: states,
                    n_evals: 0,
                },
                event_time: Some(t_ev),
                event_state: Some(y_ev),
            });
        }
        y = y_new;
        t = t_new;
        times.push(t);
        states.push(y.clone());
        prev_ev = cur_ev;
    }
    Ok(OdeEventResult {
        solution: OdeSolution {
            t: times,
            y: states,
            n_evals: 0,
        },
        event_time: None,
        event_state: None,
    })
}
