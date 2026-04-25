//! Multivariate unconstrained optimization.
//!
//! # Methods
//!
//! | Function | Method | Gradient? |
//! |---|---|---|
//! | `gradient_descent` | Fixed step-size GD | ✅ |
//! | `gradient_descent_armijo` | GD + Armijo line search | ✅ |
//! | `bfgs` | BFGS quasi-Newton | ✅ |
//! | `nelder_mead` | Nelder-Mead simplex | ❌ |
//! | `conjugate_gradient` | Polak-Ribière CG | ✅ |
//!
//! # Usage — BFGS
//!
//! ```rust
//! use scies_math::opt_multivar::bfgs;
//!
//! // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
//! let f = |v: &[f64]| (1.0-v[0]).powi(2) + 100.0*(v[1]-v[0]*v[0]).powi(2);
//! let g = |v: &[f64]| vec![
//!     -2.0*(1.0-v[0]) - 400.0*v[0]*(v[1]-v[0]*v[0]),
//!     200.0*(v[1]-v[0]*v[0]),
//! ];
//!
//! let result = bfgs(f, g, &[-1.0, 1.0], 1e-6, 1000).unwrap();
//! // result.x ≈ [1.0, 1.0]
//! ```
//!
//! For trust-region, augmented Lagrangian, PSO, and DE see [`crate::opt_multivar_ext`].
use crate::autodiff::{Dual, jacobian};
use crate::errors::{SciError, SciResult};

// ─────────────────────────────────────────────────────────────
// BFGS
// ─────────────────────────────────────────────────────────────

/// BFGS quasi-Newton minimiser for smooth unconstrained functions.
///
/// Uses forward-mode AD for exact gradients and an Armijo backtracking
/// line search.
///
/// Returns the minimiser `x*` and the function value at that point.
pub fn bfgs<F>(f: F, x0: &[f64], tolerance: f64, max_iter: usize) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[Dual]) -> Dual + Copy,
{
    if x0.is_empty() {
        return Err(SciError::InvalidParameter(
            "initial point must be non-empty",
        ));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let n = x0.len();
    let f_scalar = |x: &[f64]| {
        let duals: Vec<Dual> = x.iter().map(|&xi| Dual::from(xi)).collect();
        f(&duals).value
    };

    let mut x = x0.to_vec();
    // H_inv starts as identity.
    let mut h_inv = identity(n);
    let mut g = jacobian(f, &x);

    for _iter in 0..max_iter {
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tolerance {
            return Ok((x.clone(), f_scalar(&x)));
        }

        // Direction: p = -H_inv · g
        let p = mat_vec_mul(&h_inv, &g)
            .into_iter()
            .map(|v| -v)
            .collect::<Vec<_>>();

        // Armijo backtracking line search.
        let step = armijo_backtrack(&f_scalar, &x, &p, &g, 1.0, 0.5, 1e-4)?;

        // s = step · p
        let s: Vec<f64> = p.iter().map(|&pi| step * pi).collect();
        let x_new: Vec<f64> = x.iter().zip(s.iter()).map(|(xi, si)| xi + si).collect();
        let g_new = jacobian(f, &x_new);

        // y = g_new - g
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy.abs() > f64::EPSILON {
            // BFGS rank-2 update.
            h_inv = bfgs_update(&h_inv, &s, &y, sy, n);
        }

        x = x_new;
        g = g_new;
    }

    let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
    if g_norm < tolerance {
        Ok((x.clone(), f_scalar(&x)))
    } else {
        Err(SciError::NonConvergent("BFGS"))
    }
}

fn bfgs_update(h: &[Vec<f64>], s: &[f64], y: &[f64], sy: f64, n: usize) -> Vec<Vec<f64>> {
    let rho = 1.0 / sy;
    let hy: Vec<f64> = mat_vec_mul(h, y);
    let ythy: f64 = y.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
    let mut h_new = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            h_new[i][j] =
                h[i][j] + rho * ((1.0 + rho * ythy) * s[i] * s[j] - hy[i] * s[j] - s[i] * hy[j]);
        }
    }
    h_new
}

// ─────────────────────────────────────────────────────────────
// L-BFGS (limited-memory, m-step history)
// ─────────────────────────────────────────────────────────────

/// L-BFGS minimiser — memory-efficient for large problems.
///
/// `memory` is the number of past (s, y) pairs stored (typical: 5–20).
pub fn lbfgs<F>(
    f: F,
    x0: &[f64],
    memory: usize,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[Dual]) -> Dual + Copy,
{
    if x0.is_empty() {
        return Err(SciError::InvalidParameter(
            "initial point must be non-empty",
        ));
    }
    if memory == 0 {
        return Err(SciError::InvalidParameter("memory must be positive"));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let f_scalar = |x: &[f64]| {
        let duals: Vec<Dual> = x.iter().map(|&xi| Dual::from(xi)).collect();
        f(&duals).value
    };

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = jacobian(f, &x);
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();

    for _ in 0..max_iter {
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tolerance {
            return Ok((x.clone(), f_scalar(&x)));
        }

        // Two-loop L-BFGS direction.
        let p = lbfgs_direction(&g, &s_hist, &y_hist);
        let step = armijo_backtrack(&f_scalar, &x, &p, &g, 1.0, 0.5, 1e-4)?;

        let s: Vec<f64> = p.iter().map(|&pi| step * pi).collect();
        let x_new: Vec<f64> = x.iter().zip(s.iter()).map(|(xi, si)| xi + si).collect();
        let g_new = jacobian(f, &x_new);
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy > f64::EPSILON {
            if s_hist.len() == memory {
                s_hist.remove(0);
                y_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
        }
        x = x_new;
        g = g_new;
        let _ = n; // suppress warning
    }

    Err(SciError::NonConvergent("L-BFGS"))
}

fn lbfgs_direction(g: &[f64], s_hist: &[Vec<f64>], y_hist: &[Vec<f64>]) -> Vec<f64> {
    let mut q: Vec<f64> = g.to_vec();
    let m = s_hist.len();
    let mut alphas = vec![0.0f64; m];
    for i in (0..m).rev() {
        let rho = 1.0 / dot(&y_hist[i], &s_hist[i]).max(f64::EPSILON);
        alphas[i] = rho * dot(&s_hist[i], &q);
        for (qi, yi) in q.iter_mut().zip(y_hist[i].iter()) {
            *qi -= alphas[i] * yi;
        }
    }
    // Scale by H0 ≈ (s·y)/(y·y) of most recent pair.
    let scale = if m > 0 {
        dot(&s_hist[m - 1], &y_hist[m - 1]) / dot(&y_hist[m - 1], &y_hist[m - 1]).max(f64::EPSILON)
    } else {
        1.0
    };
    let mut r: Vec<f64> = q.iter().map(|qi| qi * scale).collect();
    for i in 0..m {
        let rho = 1.0 / dot(&y_hist[i], &s_hist[i]).max(f64::EPSILON);
        let beta = rho * dot(&y_hist[i], &r);
        for (ri, si) in r.iter_mut().zip(s_hist[i].iter()) {
            *ri += (alphas[i] - beta) * si;
        }
    }
    r.iter().map(|v| -v).collect()
}

// ─────────────────────────────────────────────────────────────
// Nelder–Mead Simplex
// ─────────────────────────────────────────────────────────────

/// Nelder–Mead simplex optimiser (derivative-free).
///
/// `step` controls the initial simplex size.
pub fn nelder_mead<F>(
    f: F,
    x0: &[f64],
    step: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(SciError::InvalidParameter(
            "initial point must be non-empty",
        ));
    }
    if step <= 0.0 || tolerance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "step and tolerance must be positive",
        ));
    }

    let n = x0.len();
    // Build initial simplex: n+1 vertices.
    let mut simplex: Vec<Vec<f64>> = (0..=n)
        .map(|i| {
            let mut v = x0.to_vec();
            if i > 0 {
                v[i - 1] += step;
            }
            v
        })
        .collect();
    let mut fval: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        // Sort by function value.
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fval[a].total_cmp(&fval[b]));

        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Convergence: std of fvals.
        let mean_f = fval.iter().sum::<f64>() / (n + 1) as f64;
        let std_f: f64 =
            (fval.iter().map(|v| (v - mean_f).powi(2)).sum::<f64>() / (n + 1) as f64).sqrt();
        if std_f < tolerance {
            return Ok((simplex[best].clone(), fval[best]));
        }

        // Centroid of all but worst.
        let centroid: Vec<f64> = (0..n)
            .map(|dim| order[..n].iter().map(|&idx| simplex[idx][dim]).sum::<f64>() / n as f64)
            .collect();

        // Reflection.
        let xr: Vec<f64> = centroid
            .iter()
            .zip(simplex[worst].iter())
            .map(|(c, w)| c + (c - w))
            .collect();
        let fr = f(&xr);

        if fr < fval[best] {
            // Expansion.
            let xe: Vec<f64> = centroid
                .iter()
                .zip(xr.iter())
                .map(|(c, r)| c + 2.0 * (r - c))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[worst] = xe;
                fval[worst] = fe;
            } else {
                simplex[worst] = xr;
                fval[worst] = fr;
            }
        } else if fr < fval[second_worst] {
            simplex[worst] = xr;
            fval[worst] = fr;
        } else {
            // Contraction.
            let xc: Vec<f64> = centroid
                .iter()
                .zip(simplex[worst].iter())
                .map(|(c, w)| c + 0.5 * (w - c))
                .collect();
            let fc = f(&xc);
            if fc < fval[worst] {
                simplex[worst] = xc;
                fval[worst] = fc;
            } else {
                // Shrink all but best.
                for i in 1..=n {
                    let idx = order[i];
                    for dim in 0..n {
                        simplex[idx][dim] =
                            simplex[best][dim] + 0.5 * (simplex[idx][dim] - simplex[best][dim]);
                    }
                    fval[idx] = f(&simplex[idx]);
                }
            }
        }
    }

    let _best_idx = (0..=n)
        .min_by(|&a, &b| fval[a].total_cmp(&fval[b]))
        .unwrap();
    Err(SciError::NonConvergent("Nelder-Mead"))
}

// ─────────────────────────────────────────────────────────────
// Simulated Annealing
// ─────────────────────────────────────────────────────────────

/// Simulated annealing global optimiser (Metropolis–Hastings).
///
/// `temp_start` / `temp_end` — initial and final temperature.
/// `step_size`               — proposal standard deviation per dimension.
/// `seed`                    — deterministic PRNG seed.
pub fn simulated_annealing<F>(
    f: F,
    x0: &[f64],
    temp_start: f64,
    temp_end: f64,
    step_size: f64,
    max_iter: usize,
    seed: u64,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if temp_start <= temp_end || temp_end <= 0.0 {
        return Err(SciError::InvalidParameter(
            "temp_start must be > temp_end > 0",
        ));
    }
    if step_size <= 0.0 {
        return Err(SciError::InvalidParameter("step_size must be positive"));
    }

    let n = x0.len();
    let mut rng = LcgRng::new(seed);
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut best_x = x.clone();
    let mut best_fx = fx;

    for iter in 0..max_iter {
        let t = temp_start * (temp_end / temp_start).powf(iter as f64 / max_iter as f64);

        // Gaussian proposal.
        let x_new: Vec<f64> = x.iter().map(|&xi| xi + step_size * rng.randn()).collect();
        let fx_new = f(&x_new);
        let delta = fx_new - fx;

        if delta < 0.0 || rng.rand01() < (-delta / t).exp() {
            x = x_new;
            fx = fx_new;
            if fx < best_fx {
                best_fx = fx;
                best_x = x.clone();
            }
        }
        let _ = n; // suppress warning
    }

    Ok((best_x, best_fx))
}

// ─────────────────────────────────────────────────────────────
// Fletcher–Reeves Conjugate Gradient (optimisation)
// ─────────────────────────────────────────────────────────────

/// Fletcher–Reeves conjugate gradient minimiser.
pub fn conjugate_gradient_opt<F>(
    f: F,
    x0: &[f64],
    tolerance: f64,
    max_iter: usize,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[Dual]) -> Dual + Copy,
{
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }
    let f_scalar = |x: &[f64]| {
        let duals: Vec<Dual> = x.iter().map(|&xi| Dual::from(xi)).collect();
        f(&duals).value
    };

    let mut x = x0.to_vec();
    let mut g = jacobian(f, &x);
    let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();

    for _ in 0..max_iter {
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tolerance {
            return Ok((x.clone(), f_scalar(&x)));
        }
        let step = armijo_backtrack(&f_scalar, &x, &d, &g, 1.0, 0.5, 1e-4)?;
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + step * di)
            .collect();
        let g_new = jacobian(f, &x_new);
        let beta = dot(&g_new, &g_new) / dot(&g, &g).max(f64::EPSILON);
        d = g_new
            .iter()
            .zip(d.iter())
            .map(|(gi, di)| -gi + beta * di)
            .collect();
        x = x_new;
        g = g_new;
    }
    Err(SciError::NonConvergent("conjugate gradient (optimisation)"))
}

// ─────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn armijo_backtrack<F>(
    f: F,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    alpha0: f64,
    rho: f64,
    c: f64,
) -> SciResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let fx = f(x);
    let slope = dot(g, d);
    let mut alpha = alpha0;
    for _ in 0..60 {
        let x_trial: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        if f(&x_trial) <= fx + c * alpha * slope {
            return Ok(alpha);
        }
        alpha *= rho;
    }
    Ok(alpha)
}

/// Minimal LCG random number generator (no external deps).
struct LcgRng {
    state: u64,
}
impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x123456789ABCDEF,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    fn rand01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Box–Muller standard normal sample.
    fn randn(&mut self) -> f64 {
        let u1 = self.rand01().max(f64::EPSILON);
        let u2 = self.rand01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}
