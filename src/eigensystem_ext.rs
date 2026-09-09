//! Extended eigensystem solvers.
//!
//! | Function | Problem | Method |
//! |---|---|---|
//! | [`generalized_eigen_sym`] | Ax = λBx, B SPD | Cholesky transform → standard |
//! | [`inverse_iteration`] | Refine eigenpair near σ | Shift-and-invert power iter |
//! | [`rayleigh_quotient_iter`] | Find eigenpair from hint x₀ | Cubic convergence |
//! | [`simultaneous_iteration`] | Dominant k eigenpairs | QR iteration on tall matrix |

use crate::eigensystem::{Eigensystem, jacobi_eigen};
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

// ─────────────────────────────────────────────────────────────────────────────
// Generalized eigenvalue problem  Ax = λBx  (B symmetric positive-definite)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the **generalized symmetric eigenvalue problem** Ax = λBx
/// where both A and B are real symmetric and B is positive-definite.
///
/// Algorithm:
/// 1. Cholesky factor B = L·Lᵀ.
/// 2. Transform to standard form C·y = λy where C = L⁻¹·A·L⁻ᵀ.
/// 3. Solve C with Jacobi.
/// 4. Back-transform eigenvectors: x = L⁻ᵀ·y.
///
/// Returns eigenvalues sorted ascending, with corresponding eigenvectors.
pub fn generalized_eigen_sym(
    a: &DynamicMatrix,
    b: &DynamicMatrix,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<Eigensystem> {
    let n = a.rows();
    if a.cols() != n || b.rows() != n || b.cols() != n {
        return Err(SciError::InvalidParameter(
            "A and B must be square and the same size",
        ));
    }

    // Step 1: Cholesky of B
    let chol = b.cholesky_decompose()?;
    let _l = &chol.l; // lower triangular (kept for documentation clarity)

    // Step 2: Compute L⁻¹ — solve L·X = I column by column
    let l_inv = {
        let id = DynamicMatrix::identity(n)?;
        chol.solve_multi(&id)?
    };

    // Step 3: C = L⁻¹ · A · (L⁻¹)ᵀ  (= L⁻¹ · A · L⁻ᵀ)
    let l_inv_t = l_inv.transpose();
    let c = l_inv.mul_matrix(a)?.mul_matrix(&l_inv_t)?;

    // Step 4: Jacobi eigendecomposition of (symmetric) C
    let eig = jacobi_eigen(&c, tolerance, max_iterations)?;

    // Step 5: Back-transform: x_i = L⁻ᵀ · y_i
    let nev = eig.values.len();
    let eig_vecs = &eig.vectors; // rows = components, cols = eigenvectors
    let back = l_inv_t.mul_matrix(eig_vecs)?;

    // Normalise each column so that xᵀ·B·x = 1
    let mut normed_data = back.raw_data().to_vec();
    for k in 0..nev {
        let col: Vec<f64> = (0..n).map(|r| normed_data[r * nev + k]).collect();
        let bx = b.mul_vector(&col)?;
        let norm = col
            .iter()
            .zip(bx.iter())
            .map(|(ci, bxi)| ci * bxi)
            .sum::<f64>()
            .sqrt();
        if norm > f64::EPSILON {
            for r in 0..n {
                normed_data[r * nev + k] /= norm;
            }
        }
    }

    Ok(Eigensystem {
        values: eig.values,
        vectors: DynamicMatrix::new(n, nev, normed_data)?,
        sweeps: eig.sweeps,
        residual: eig.residual,
        converged: eig.converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse (shift-and-invert) iteration to refine an eigenvector near `sigma`.
///
/// Solves `(A − σI)·x_{k+1} = x_k` iteratively.
/// Converges to the eigenvector whose eigenvalue is **closest** to `sigma`.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn inverse_iteration(
    a: &DynamicMatrix,
    sigma: f64,
    x0: Option<Vec<f64>>,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<(f64, Vec<f64>)> {
    let n = a.rows();
    if a.cols() != n {
        return Err(SciError::InvalidParameter("A must be square"));
    }

    // Shifted matrix A − σI
    let mut shifted = a.raw_data().to_vec();
    for i in 0..n {
        shifted[i * n + i] -= sigma;
    }
    let a_shifted = DynamicMatrix::new(n, n, shifted)?;
    let lu = a_shifted.lu_decompose()?;

    // Initial vector
    let mut x: Vec<f64> = match x0 {
        Some(v) if v.len() == n => v,
        _ => {
            let mut v = vec![1.0f64; n];
            let norm = (n as f64).sqrt();
            v.iter_mut().for_each(|xi| *xi /= norm);
            v
        }
    };
    normalise(&mut x);

    let mut eigenvalue = sigma;
    for _ in 0..max_iterations {
        let x_new = lu.solve(&x)?;
        let norm = l2_norm(&x_new);
        if norm < f64::EPSILON {
            return Err(SciError::NonConvergent("inverse_iteration: zero vector"));
        }
        // Rayleigh quotient of original A
        let ax = a.mul_vector(&x_new)?;
        let rq = dot(&x_new, &ax) / dot(&x_new, &x_new);
        let x_normed: Vec<f64> = x_new.iter().map(|v| v / norm).collect();
        // Convergence check
        let diff: Vec<f64> = x_normed.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let change = l2_norm(&diff);
        x = x_normed;
        eigenvalue = rq;
        if change < tolerance {
            break;
        }
    }
    Ok((eigenvalue, x))
}

// ─────────────────────────────────────────────────────────────────────────────
// Rayleigh quotient iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Rayleigh quotient iteration — **cubic convergence** near an eigenpair.
///
/// Starting from hint vector `x0`, adaptively shifts by the Rayleigh quotient
/// `ρ = xᵀAx / xᵀx` and applies inverse iteration with that shift.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn rayleigh_quotient_iter(
    a: &DynamicMatrix,
    x0: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<(f64, Vec<f64>)> {
    let n = a.rows();
    if a.cols() != n {
        return Err(SciError::InvalidParameter("A must be square"));
    }
    if x0.len() != n {
        return Err(SciError::InvalidParameter(
            "x0 length must match matrix size",
        ));
    }
    let mut x = x0;
    normalise(&mut x);

    let mut eigenvalue = rayleigh_quotient(a, &x)?;

    for _ in 0..max_iterations {
        // (A − ρI)·z = x
        let mut shifted = a.raw_data().to_vec();
        for i in 0..n {
            shifted[i * n + i] -= eigenvalue;
        }
        let a_shift = DynamicMatrix::new(n, n, shifted)?;
        let lu = match a_shift.lu_decompose() {
            Ok(lu) => lu,
            Err(_) => break, // singular → already at exact eigenpair
        };
        let z = lu.solve(&x)?;
        let norm = l2_norm(&z);
        if norm < f64::EPSILON {
            break;
        }
        let x_new: Vec<f64> = z.iter().map(|v| v / norm).collect();
        let rq = rayleigh_quotient(a, &x_new)?;
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let change = l2_norm(&diff).min(l2_norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
        ));
        x = x_new;
        eigenvalue = rq;
        if change < tolerance {
            break;
        }
    }
    Ok((eigenvalue, x))
}

// ─────────────────────────────────────────────────────────────────────────────
// Simultaneous iteration (subspace iteration) for dominant k eigenpairs
// ─────────────────────────────────────────────────────────────────────────────

/// Simultaneous (subspace) iteration for the **k largest-magnitude** eigenpairs.
///
/// Applies power iteration to a k-column matrix and uses QR to orthogonalise.
/// Converges to the eigenspace spanned by the k dominant eigenvectors.
///
/// # Parameters
/// - `k` — number of eigenpairs to compute
/// - `seed` — initial random subspace seed
pub fn simultaneous_iteration(
    a: &DynamicMatrix,
    k: usize,
    tolerance: f64,
    max_iterations: usize,
    seed: u64,
) -> SciResult<Eigensystem> {
    let n = a.rows();
    if a.cols() != n {
        return Err(SciError::InvalidParameter("A must be square"));
    }
    if k == 0 || k > n {
        return Err(SciError::InvalidParameter("k must be in [1, n]"));
    }

    // Random initial n×k matrix (seeded simple LCG)
    let mut q_data = vec![0.0f64; n * k];
    let mut lcg = seed.wrapping_add(6_364_136_223_846_793_005);
    for v in q_data.iter_mut() {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *v = (lcg >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
    }

    let mut q = DynamicMatrix::new(n, k, q_data)?;
    // QR orthogonalise initial Q
    let (qorth, _) = q.qr_decompose()?.into_qr();
    q = slice_cols(&qorth, k)?;

    let mut prev_eigenvalues = vec![f64::INFINITY; k];

    let mut final_iters = 0;
    let mut last_change = f64::INFINITY;
    let mut converged = false;

    for iter in 0..max_iterations {
        final_iters = iter + 1;
        // Z = A · Q
        let z = a.mul_matrix(&q)?;
        // QR of Z
        let (qnew, r) = z.qr_decompose()?.into_qr();
        q = slice_cols(&qnew, k)?;

        // Rayleigh quotients (diagonal of Rᵀ projected)
        let eigenvalues: Vec<f64> = (0..k).map(|i| r.get(i, i).unwrap_or(0.0)).collect();

        // Convergence
        let max_change = eigenvalues
            .iter()
            .zip(prev_eigenvalues.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        prev_eigenvalues = eigenvalues.clone();
        last_change = max_change;
        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    // Refine eigenvalues as Rayleigh quotients
    let eigenvalues: Vec<f64> = (0..k)
        .map(|i| {
            let col: Vec<f64> = (0..n).map(|r| q.get(r, i).unwrap_or(0.0)).collect();
            rayleigh_quotient(a, &col).unwrap_or(0.0)
        })
        .collect();

    Ok(Eigensystem {
        values: eigenvalues,
        vectors: q,
        sweeps: final_iters,
        residual: last_change,
        converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}
fn normalise(v: &mut Vec<f64>) {
    let n = l2_norm(v);
    if n > f64::EPSILON {
        v.iter_mut().for_each(|x| *x /= n);
    }
}
fn rayleigh_quotient(a: &DynamicMatrix, x: &[f64]) -> SciResult<f64> {
    let ax = a.mul_vector(x)?;
    Ok(dot(x, &ax) / dot(x, x))
}

/// Extract first `k` columns of an n×m matrix (k ≤ m).
fn slice_cols(a: &DynamicMatrix, k: usize) -> SciResult<DynamicMatrix> {
    let n = a.rows();
    let m = a.cols();
    if k > m {
        return Err(SciError::InvalidParameter("k exceeds column count"));
    }
    let data: Vec<f64> = (0..n)
        .flat_map(|r| (0..k).map(move |c| a.get(r, c).unwrap_or(0.0)))
        .collect();
    DynamicMatrix::new(n, k, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// QrDecomposition helper  (expose Q, R)
// ─────────────────────────────────────────────────────────────────────────────

trait IntoQR {
    fn into_qr(self) -> (DynamicMatrix, DynamicMatrix);
}
impl IntoQR for crate::linear_algebra::QrDecomposition {
    fn into_qr(self) -> (DynamicMatrix, DynamicMatrix) {
        (self.q, self.r)
    }
}
