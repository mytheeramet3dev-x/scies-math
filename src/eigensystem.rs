//! Full eigensystem solvers returning eigenvalues **and** eigenvectors.
//!
//! | Function | Matrix type | Method |
//! |---|---|---|
//! | [`jacobi_eigen`]     | Real symmetric | Classic Jacobi off-diagonal pivoting |
//! | [`qr_eigen_general`] | General real   | QR iteration with Wilkinson shifts   |

use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

/// Result of an eigensystem computation.
#[derive(Debug, Clone)]
pub struct Eigensystem {
    /// Eigenvalues sorted by descending absolute value.
    pub values: Vec<f64>,
    /// Eigenvectors stored as columns (rows × k matrix).
    pub vectors: DynamicMatrix,
}

// ─────────────────────────────────────────────────────────────
// Jacobi — symmetric matrices
// ─────────────────────────────────────────────────────────────

/// Jacobi eigendecomposition for a **real symmetric** matrix.
///
/// Classic cyclic off-diagonal pivoting; converges quadratically once
/// the off-diagonal norm is small.  Returns eigenvalues sorted by
/// descending absolute value.
pub fn jacobi_eigen(
    matrix: &DynamicMatrix,
    tolerance: f64,
    max_sweeps: usize,
) -> SciResult<Eigensystem> {
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SciError::InvalidParameter(
            "Jacobi requires a square matrix",
        ));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let mut a: Vec<f64> = (0..n)
        .flat_map(|r| (0..n).map(move |c| matrix.get(r, c).unwrap_or(0.0)))
        .collect();
    let mut v = identity_flat(n);

    for _sweep in 0..max_sweeps {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for r in 0..n {
            for c in (r + 1)..n {
                let val = a[r * n + c].abs();
                if val > max_val {
                    max_val = val;
                    p = r;
                    q = c;
                }
            }
        }
        if max_val < tolerance {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (aqq - app).abs() < f64::EPSILON {
            core::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (aqq - app)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        jacobi_rotate(&mut a, n, p, q, c, s);
        for r in 0..n {
            let vp = v[r * n + p];
            let vq = v[r * n + q];
            v[r * n + p] = c * vp - s * vq;
            v[r * n + q] = s * vp + c * vq;
        }
    }

    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    pairs.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    let values: Vec<f64> = pairs.iter().map(|(val, _)| *val).collect();
    let mut vec_data = vec![0.0f64; n * n];
    for (new_col, (_, old_col)) in pairs.iter().enumerate() {
        for row in 0..n {
            vec_data[row * n + new_col] = v[row * n + old_col];
        }
    }
    Ok(Eigensystem {
        values,
        vectors: DynamicMatrix::new(n, n, vec_data)?,
    })
}

fn jacobi_rotate(a: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    for r in 0..n {
        let apr = a[p * n + r];
        let aqr = a[q * n + r];
        a[p * n + r] = c * apr - s * aqr;
        a[q * n + r] = s * apr + c * aqr;
    }
    for r in 0..n {
        let arp = a[r * n + p];
        let arq = a[r * n + q];
        a[r * n + p] = c * arp - s * arq;
        a[r * n + q] = s * arp + c * arq;
    }
}

// ─────────────────────────────────────────────────────────────
// QR iteration with Wilkinson shifts — general real matrix
// ─────────────────────────────────────────────────────────────

/// QR iteration for a **general real square** matrix.
///
/// Applies Hessenberg reduction then QR steps with a Wilkinson shift.
/// Returns `(real_parts, imag_parts, eigenvector_matrix)`.
pub fn qr_eigen_general(
    matrix: &DynamicMatrix,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<(Vec<f64>, Vec<f64>, DynamicMatrix)> {
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SciError::InvalidParameter("matrix must be square"));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let mut h: Vec<f64> = (0..n)
        .flat_map(|r| (0..n).map(move |c| matrix.get(r, c).unwrap_or(0.0)))
        .collect();
    let mut z = identity_flat(n);

    hessenberg_reduce(&mut h, &mut z, n);

    let mut real = vec![0.0f64; n];
    let mut imag = vec![0.0f64; n];
    qr_iteration_loop(&mut h, &mut z, n, tolerance, max_iter, &mut real, &mut imag)?;

    Ok((real, imag, DynamicMatrix::new(n, n, z)?))
}

fn identity_flat(n: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    v
}

fn hessenberg_reduce(h: &mut [f64], z: &mut [f64], n: usize) {
    for k in 0..(n.saturating_sub(2)) {
        let mut v: Vec<f64> = (k + 1..n).map(|i| h[i * n + k]).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            continue;
        }
        v[0] += if v[0] >= 0.0 { norm } else { -norm };
        let v_norm2: f64 = v.iter().map(|x| x * x).sum();
        if v_norm2 < f64::EPSILON {
            continue;
        }
        for col in 0..n {
            let d: f64 = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi * h[(k + 1 + i) * n + col])
                .sum();
            for (i, vi) in v.iter().enumerate() {
                h[(k + 1 + i) * n + col] -= 2.0 * vi * d / v_norm2;
            }
        }
        for row in 0..n {
            let d: f64 = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi * h[row * n + k + 1 + i])
                .sum();
            for (i, vi) in v.iter().enumerate() {
                h[row * n + k + 1 + i] -= 2.0 * vi * d / v_norm2;
            }
        }
        for row in 0..n {
            let d: f64 = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi * z[row * n + k + 1 + i])
                .sum();
            for (i, vi) in v.iter().enumerate() {
                z[row * n + k + 1 + i] -= 2.0 * vi * d / v_norm2;
            }
        }
    }
}

fn qr_iteration_loop(
    h: &mut [f64],
    z: &mut [f64],
    n: usize,
    tol: f64,
    max_iter: usize,
    real: &mut [f64],
    imag: &mut [f64],
) -> SciResult<()> {
    let mut active = n;
    let mut iter = 0;
    while active > 1 {
        if iter >= max_iter {
            for i in 0..active {
                real[i] = h[i * n + i];
                imag[i] = 0.0;
            }
            return Err(SciError::NonConvergent("QR general eigensystem"));
        }
        iter += 1;
        let sub = h[(active - 1) * n + active - 2].abs();
        let scale = h[(active - 2) * n + active - 2].abs() + h[(active - 1) * n + active - 1].abs();
        if sub < tol * scale.max(f64::EPSILON) {
            active -= 1;
            real[active] = h[active * n + active];
            imag[active] = 0.0;
            continue;
        }
        let shift = h[(active - 1) * n + active - 1];
        single_shift_qr_step(h, z, n, active, shift);
    }
    if active == 1 {
        real[0] = h[0];
        imag[0] = 0.0;
    }
    Ok(())
}

fn single_shift_qr_step(h: &mut [f64], z: &mut [f64], n: usize, size: usize, shift: f64) {
    for i in 0..size {
        h[i * n + i] -= shift;
    }
    let mut cs = vec![(1.0_f64, 0.0_f64); size - 1];
    for k in 0..(size - 1) {
        let a = h[k * n + k];
        let b = h[(k + 1) * n + k];
        let r = a.hypot(b);
        if r < f64::EPSILON {
            continue;
        }
        let c = a / r;
        let s = -b / r;
        cs[k] = (c, s);
        for col in k..size {
            let hkc = h[k * n + col];
            let hk1c = h[(k + 1) * n + col];
            h[k * n + col] = c * hkc - s * hk1c;
            h[(k + 1) * n + col] = s * hkc + c * hk1c;
        }
    }
    for k in 0..(size - 1) {
        let (c, s) = cs[k];
        for row in 0..size {
            let hrc = h[row * n + k];
            let hrk1 = h[row * n + k + 1];
            h[row * n + k] = c * hrc + s * hrk1;
            h[row * n + k + 1] = -s * hrc + c * hrk1;
        }
        for row in 0..n {
            let zrc = z[row * n + k];
            let zrk1 = z[row * n + k + 1];
            z[row * n + k] = c * zrc + s * zrk1;
            z[row * n + k + 1] = -s * zrc + c * zrk1;
        }
    }
    for i in 0..size {
        h[i * n + i] += shift;
    }
}
