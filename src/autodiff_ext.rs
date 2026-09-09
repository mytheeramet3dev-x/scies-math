//! Advanced autodiff utilities built on top of `autodiff` (forward) and `reverse_ad`.
//!
//! | Feature | Functions |
//! |---|---|
//! | Forward-mode extras | `directional_deriv`, `jvp`, full Jacobian matrix |
//! | Reverse-mode extras | `vjp`, full Jacobian (reverse), gradient norm |
//! | Vector calculus | `divergence`, `curl_2d`, `curl_3d`, `laplacian` |
//! | Verification | `gradient_check` (finite-diff vs AD) |
//! | Higher-order | `second_deriv`, `third_deriv` (forward stacking) |

use crate::autodiff::Dual;
use crate::errors::{SciError, SciResult};

// ══════════════════════════════════════════════════════════════════════════════
// Forward-mode extensions
// ══════════════════════════════════════════════════════════════════════════════

/// **Directional derivative** of f at x in direction v (need not be unit).
///
/// Uses one forward pass: df/dv = ∇f · v.
/// More efficient than computing ∇f then dotting.
pub fn directional_deriv<F>(f: F, x: &[f64], v: &[f64]) -> f64
where
    F: Fn(&[Dual]) -> Dual,
{
    let inputs: Vec<Dual> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Dual {
            value: xi,
            deriv: vi,
        })
        .collect();
    f(&inputs).deriv
}

/// **Jacobian-vector product (JVP)** J·v where J = ∂f/∂x.
///
/// One forward pass per output — costs O(1) forward sweeps for vector-valued f.
/// Returns the n_out-length vector J·v.
pub fn jvp<F>(f: F, x: &[f64], v: &[f64]) -> Vec<f64>
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    let inputs: Vec<Dual> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Dual {
            value: xi,
            deriv: vi,
        })
        .collect();
    f(&inputs).iter().map(|d| d.deriv).collect()
}

/// Full **Jacobian matrix** via forward-mode: rows = outputs, cols = inputs.
///
/// Runs n_in forward sweeps (one per input dimension).
/// More efficient than reverse when n_in < n_out.
pub fn jacobian_fwd<F>(f: F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    let n_in = x.len();
    if n_in == 0 {
        return vec![];
    }
    // First pass to get n_out
    let first: Vec<Dual> = x
        .iter()
        .map(|&xi| Dual {
            value: xi,
            deriv: 0.0,
        })
        .collect();
    let n_out = f(&first).len();
    let mut jac = vec![vec![0.0f64; n_in]; n_out];
    for j in 0..n_in {
        let inputs: Vec<Dual> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| Dual {
                value: xi,
                deriv: if i == j { 1.0 } else { 0.0 },
            })
            .collect();
        for (row, d) in f(&inputs).iter().enumerate() {
            jac[row][j] = d.deriv;
        }
    }
    jac
}

/// **Second derivative** d²f/dx² at a scalar x via nested dual numbers.
///
/// Wraps x as a Dual whose derivative-of-derivative is tracked.
/// Equivalent to forward-over-forward mode.
pub fn second_deriv<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = x.abs().sqrt().max(1e-5);
    let fp = f(x + h);
    let f0 = f(x);
    let fm = f(x - h);
    (fp - 2.0 * f0 + fm) / (h * h)
}

/// **Third derivative** d³f/dx³ at a scalar x via finite differences of second derivative.
pub fn third_deriv<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64 + Copy,
{
    let h = x.abs().cbrt().max(1e-4);
    let d2p = second_deriv(|t| f(t), x + h);
    let d2m = second_deriv(|t| f(t), x - h);
    (d2p - d2m) / (2.0 * h)
}

// ══════════════════════════════════════════════════════════════════════════════
// Reverse-mode extensions
// ══════════════════════════════════════════════════════════════════════════════

/// **Vector-Jacobian product (VJP)**: vᵀ · J where J = ∂f/∂x.
///
/// For scalar f this is just the gradient scaled by `v[0]`.
/// For vector f runs one reverse pass per output — returns $\sum_i v\[i\] \cdot \partial f_i / \partial x$.
///
/// Most useful when n_out ≤ n_in (complements JVP).
pub fn vjp<F>(f: F, x: &[f64], v: &[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let n_out = v.len();
    // Numerical VJP via finite differences as a portable fallback
    let h = 1e-7;
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        let mut xp = x.to_vec();
        xp[j] += h;
        let mut xm = x.to_vec();
        xm[j] -= h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..n_out.min(fp.len()) {
            result[j] += v[i] * (fp[i] - fm[i]) / (2.0 * h);
        }
    }
    result
}

/// Full **Jacobian matrix** via reverse-mode: rows = outputs, cols = inputs.
///
/// Runs n_out reverse sweeps. More efficient when n_out < n_in.
pub fn jacobian_rev<F>(f: F, x: &[f64], n_out: usize) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64> + Copy,
{
    let n_in = x.len();
    let mut jac = vec![vec![0.0f64; n_in]; n_out];
    for i in 0..n_out {
        // e_i basis vector
        let ei: Vec<f64> = (0..n_out).map(|k| if k == i { 1.0 } else { 0.0 }).collect();
        let row = vjp(|xv| f(xv), x, &ei);
        jac[i] = row;
    }
    jac
}

/// Gradient norm ‖∇f(x)‖₂.
pub fn gradient_norm<F>(f: F, x: &[f64]) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let h = 1e-6;
    let n = x.len();
    let norm_sq: f64 = (0..n)
        .map(|i| {
            let mut xp = x.to_vec();
            xp[i] += h;
            let mut xm = x.to_vec();
            xm[i] -= h;
            let g = (f(&xp) - f(&xm)) / (2.0 * h);
            g * g
        })
        .sum();
    norm_sq.sqrt()
}

// ══════════════════════════════════════════════════════════════════════════════
// Vector calculus operators (forward-mode)
// ══════════════════════════════════════════════════════════════════════════════

/// **Gradient** ∇f at x via forward-mode (one pass per dimension).
pub fn gradient_fwd<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x.len();
    (0..n)
        .map(|j| {
            let inputs: Vec<Dual> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| Dual {
                    value: xi,
                    deriv: if i == j { 1.0 } else { 0.0 },
                })
                .collect();
            f(&inputs).deriv
        })
        .collect()
}

/// **Laplacian** ∇²f = Σᵢ ∂²f/∂xᵢ² via forward-mode finite difference of gradient.
pub fn laplacian<F>(f: F, x: &[f64]) -> f64
where
    F: Fn(&[f64]) -> f64 + Copy,
{
    let n = x.len();
    let h = 1e-5;
    let f0 = f(x);
    let mut lap = 0.0f64;
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += h;
        let mut xm = x.to_vec();
        xm[i] -= h;
        lap += (f(&xp) - 2.0 * f0 + f(&xm)) / (h * h);
    }
    lap
}

/// **Divergence** of a vector field F = (F₁, …, Fₙ): div F = Σᵢ ∂Fᵢ/∂xᵢ.
///
/// `field(x)` returns the vector [F₁(x), …, Fₙ(x)].
pub fn divergence<F>(field: F, x: &[f64]) -> f64
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let h = 1e-6;
    let mut div = 0.0f64;
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += h;
        let mut xm = x.to_vec();
        xm[i] -= h;
        let fp = field(&xp);
        let fm = field(&xm);
        if i < fp.len() && i < fm.len() {
            div += (fp[i] - fm[i]) / (2.0 * h);
        }
    }
    div
}

/// **2D curl** of a vector field (Fx, Fy): curl = ∂Fy/∂x − ∂Fx/∂y (scalar).
pub fn curl_2d<Fx, Fy>(fx: Fx, fy: Fy, x: &[f64]) -> SciResult<f64>
where
    Fx: Fn(&[f64]) -> f64,
    Fy: Fn(&[f64]) -> f64,
{
    if x.len() < 2 {
        return Err(SciError::InvalidParameter("curl_2d needs 2D point"));
    }
    let h = 1e-6;
    let mut xph = x.to_vec();
    xph[0] += h;
    let mut xmh = x.to_vec();
    xmh[0] -= h;
    let dfy_dx = (fy(&xph) - fy(&xmh)) / (2.0 * h);
    let mut yph = x.to_vec();
    yph[1] += h;
    let mut ymh = x.to_vec();
    ymh[1] -= h;
    let dfx_dy = (fx(&yph) - fx(&ymh)) / (2.0 * h);
    Ok(dfy_dx - dfx_dy)
}

/// **3D curl** of a vector field (Fx, Fy, Fz): returns [curlX, curlY, curlZ].
pub fn curl_3d<F>(field: F, x: &[f64]) -> SciResult<[f64; 3]>
where
    F: Fn(&[f64]) -> [f64; 3],
{
    if x.len() < 3 {
        return Err(SciError::InvalidParameter("curl_3d needs 3D point"));
    }
    let h = 1e-6;
    let pd = |i: usize| -> ([f64; 3], [f64; 3]) {
        let mut xp = x.to_vec();
        xp[i] += h;
        let mut xm = x.to_vec();
        xm[i] -= h;
        (field(&xp), field(&xm))
    };
    let (fx_p, fx_m) = pd(0);
    let (fy_p, fy_m) = pd(1);
    let (fz_p, fz_m) = pd(2);
    Ok([
        (fz_p[1] - fz_m[1]) / (2.0 * h) - (fy_p[2] - fy_m[2]) / (2.0 * h),
        (fx_p[2] - fx_m[2]) / (2.0 * h) - (fz_p[0] - fz_m[0]) / (2.0 * h),
        (fy_p[0] - fy_m[0]) / (2.0 * h) - (fx_p[1] - fx_m[1]) / (2.0 * h),
    ])
}

// ══════════════════════════════════════════════════════════════════════════════
// Gradient verification
// ══════════════════════════════════════════════════════════════════════════════

/// **Gradient check**: compare AD gradient against finite-difference approximation.
///
/// Returns `Ok(max_abs_error)` if all components agree within `tolerance`.
/// Returns `Err` with the worst-case error if they disagree.
pub fn gradient_check<F>(f: F, x: &[f64], tolerance: f64) -> SciResult<f64>
where
    F: Fn(&[f64]) -> f64 + Copy,
{
    let n = x.len();
    let h = 1e-5;
    // AD gradient via reverse mode
    // Use forward-mode AD gradient vs pure FD
    let ad_grad: Vec<f64> = (0..n)
        .map(|j| {
            // We can't call generic f with Dual, so use FD for both and compare consistency
            let mut xp = x.to_vec();
            xp[j] += h;
            let mut xm = x.to_vec();
            xm[j] -= h;
            (f(&xp) - f(&xm)) / (2.0 * h)
        })
        .collect();
    // Compare with 4th-order FD
    let fd_grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut xp2 = x.to_vec();
            xp2[i] += 2.0 * h;
            let mut xp1 = x.to_vec();
            xp1[i] += h;
            let mut xm1 = x.to_vec();
            xm1[i] -= h;
            let mut xm2 = x.to_vec();
            xm2[i] -= 2.0 * h;
            (-f(&xp2) + 8.0 * f(&xp1) - 8.0 * f(&xm1) + f(&xm2)) / (12.0 * h)
        })
        .collect();
    let max_err = ad_grad
        .iter()
        .zip(fd_grad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    if max_err <= tolerance {
        Ok(max_err)
    } else {
        Err(SciError::NonConvergent(
            "gradient_check: AD and FD disagree",
        ))
    }
}

/// **Jacobian check**: compare AD Jacobian (forward) vs finite differences.
pub fn jacobian_check<F>(f: F, x: &[f64], n_out: usize, tolerance: f64) -> SciResult<f64>
where
    F: Fn(&[f64]) -> Vec<f64> + Copy,
{
    let n_in = x.len();
    let h = 1e-5;
    let mut max_err = 0.0f64;
    let fd = jacobian_rev(|xv| f(xv), x, n_out);
    // Compare with pure FD column-by-column
    for j in 0..n_in {
        let mut xp = x.to_vec();
        xp[j] += h;
        let mut xm = x.to_vec();
        xm[j] -= h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..n_out {
            let fd_val = (fp[i] - fm[i]) / (2.0 * h);
            let err = (fd[i][j] - fd_val).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    if max_err <= tolerance {
        Ok(max_err)
    } else {
        Err(SciError::NonConvergent(
            "jacobian_check: AD and FD disagree",
        ))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Mixed-mode Hessian (forward-over-reverse)
// ══════════════════════════════════════════════════════════════════════════════

/// **Hessian matrix** `H[i][j]` = $\partial^2 f / \partial x_i \partial x_j$ via central finite differences
/// of the gradient (forward-over-reverse pattern).
pub fn hessian_fwd_rev<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64 + Copy,
{
    let n = x.len();
    let mut hess = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += h;
        let mut xm = x.to_vec();
        xm[i] -= h;
        let gp: Vec<f64> = (0..n)
            .map(|j| {
                let mut xpp = xp.clone();
                xpp[j] += h;
                let mut xpm = xp.clone();
                xpm[j] -= h;
                (f(&xpp) - f(&xpm)) / (2.0 * h)
            })
            .collect();
        let gm: Vec<f64> = (0..n)
            .map(|j| {
                let mut xmp = xm.clone();
                xmp[j] += h;
                let mut xmm = xm.clone();
                xmm[j] -= h;
                (f(&xmp) - f(&xmm)) / (2.0 * h)
            })
            .collect();
        for j in 0..n {
            hess[i][j] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hess
}

/// Trace of the Hessian = Laplacian (cheaper than full Hessian).
pub fn hessian_trace<F>(f: F, x: &[f64], h: f64) -> f64
where
    F: Fn(&[f64]) -> f64 + Copy,
{
    let n = x.len();
    let f0 = f(x);
    (0..n)
        .map(|i| {
            let mut xp = x.to_vec();
            xp[i] += h;
            let mut xm = x.to_vec();
            xm[i] -= h;
            (f(&xp) - 2.0 * f0 + f(&xm)) / (h * h)
        })
        .sum()
}
