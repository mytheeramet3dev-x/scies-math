//! Positive Semidefinite (PSD) cone diagnostics, spectral projection, and metrics.
//!
//! # Diagnostic vs Proof Distinction
//!
//! > **Important Warning:**
//! > Functions in this module operate on floating-point matrices.
//! > The [`PsdDiagnostic`] report and [`project_to_psd_cone`] spectral projection
//! > provide numerical heuristics and diagnostic bounds.
//! > They do **not** constitute an exact algebraic certificate or an SDP solver.

use crate::eigensystem::jacobi_eigen;
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

/// Comprehensive diagnostic summary of a symmetric matrix's spectrum on the PSD cone.
#[derive(Debug, Clone, PartialEq)]
pub struct PsdDiagnostic {
    /// Smallest (minimum) eigenvalue $\lambda_{\min}$.
    pub min_eigenvalue: f64,
    /// Largest (maximum) eigenvalue $\lambda_{\max}$.
    pub max_eigenvalue: f64,
    /// Matrix trace $\sum_i \lambda_i = \text{Tr}(A)$.
    pub trace: f64,
    /// Frobenius norm $\|A\|_F = \sqrt{\sum \lambda_i^2}$.
    pub frobenius_norm: f64,
    /// Spectral gap between top two eigenvalues $\lambda_1 - \lambda_2$.
    pub spectral_gap: f64,
    /// Estimated numerical rank (number of eigenvalues strictly greater than tolerance).
    pub numerical_rank: usize,
    /// Whether all eigenvalues satisfy $\lambda_i \ge -\text{tolerance}$.
    pub is_psd: bool,
    /// Whether all eigenvalues satisfy $\lambda_i > \text{tolerance}$.
    pub is_strictly_pd: bool,
    /// Tolerance used for the diagnostic check.
    pub tolerance: f64,
}

/// Diagnoses the spectral and positive semidefinite properties of a symmetric matrix.
pub fn diagnose_psd(matrix: &DynamicMatrix, tolerance: f64) -> SciResult<PsdDiagnostic> {
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let eig = jacobi_eigen(matrix, tolerance.min(1e-12), 200)?;
    if eig.values.is_empty() {
        return Err(SciError::EmptyInput);
    }

    let mut sorted_vals = eig.values.clone();
    sorted_vals.sort_by(|a, b| a.total_cmp(b)); // Ascending

    let min_val = sorted_vals[0];
    let max_val = sorted_vals[sorted_vals.len() - 1];

    let trace: f64 = sorted_vals.iter().sum();
    let frobenius_norm: f64 = sorted_vals.iter().map(|v| v * v).sum::<f64>().sqrt();

    let spectral_gap = if sorted_vals.len() >= 2 {
        sorted_vals[sorted_vals.len() - 1] - sorted_vals[sorted_vals.len() - 2]
    } else {
        0.0
    };

    let numerical_rank = sorted_vals.iter().filter(|&&v| v > tolerance).count();
    let is_psd = min_val >= -tolerance;
    let is_strictly_pd = min_val > tolerance;

    Ok(PsdDiagnostic {
        min_eigenvalue: min_val,
        max_eigenvalue: max_val,
        trace,
        frobenius_norm,
        spectral_gap,
        numerical_rank,
        is_psd,
        is_strictly_pd,
        tolerance,
    })
}

/// Spectrally projects a symmetric matrix onto the positive semidefinite cone $\mathcal{S}_+^n$.
///
/// Truncates negative eigenvalues to `min_eigenvalue_clamp` (default `0.0`) via spectral
/// reconstruction:
/// $$A_+ = V \max(\Lambda, \text{clamp}) V^T$$
///
/// Returns `(projected_matrix, projection_residual_frobenius_norm)`.
pub fn project_to_psd_cone(
    matrix: &DynamicMatrix,
    min_eigenvalue_clamp: f64,
) -> SciResult<(DynamicMatrix, f64)> {
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SciError::InvalidParameter("matrix must be square"));
    }

    let eig = jacobi_eigen(matrix, 1e-12, 200)?;
    let mut clamped_vals = eig.values.clone();
    let mut proj_res_sq = 0.0_f64;

    for val in clamped_vals.iter_mut() {
        if *val < min_eigenvalue_clamp {
            let diff = *val - min_eigenvalue_clamp;
            proj_res_sq += diff * diff;
            *val = min_eigenvalue_clamp;
        }
    }

    // Reconstruct A_+ = V * diag(clamped_vals) * V^T
    let mut proj_data = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for (k, &cv) in clamped_vals.iter().enumerate().take(n) {
                let v_rk = eig.vectors.get(r, k)?;
                let v_ck = eig.vectors.get(c, k)?;
                sum += cv * v_rk * v_ck;
            }
            proj_data[r * n + c] = sum;
        }
    }

    let projected = DynamicMatrix::new(n, n, proj_data)?;
    Ok((projected, proj_res_sq.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnose_psd_known_matrix() {
        // PD matrix [[2, -1], [-1, 2]] with eigenvalues 1 and 3
        let mat = DynamicMatrix::new(2, 2, vec![2.0, -1.0, -1.0, 2.0]).unwrap();
        let diag = diagnose_psd(&mat, 1e-8).unwrap();
        assert!(diag.is_psd);
        assert!(diag.is_strictly_pd);
        assert!((diag.min_eigenvalue - 1.0).abs() < 1e-6);
        assert!((diag.max_eigenvalue - 3.0).abs() < 1e-6);
        assert_eq!(diag.numerical_rank, 2);
    }

    #[test]
    fn test_project_to_psd_cone() {
        // Indefinite matrix [[1, 2], [2, 1]] with eigenvalues 3 and -1
        let mat = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let (proj, residual) = project_to_psd_cone(&mat, 0.0).unwrap();

        // Residual should equal |-1| = 1.0
        assert!((residual - 1.0).abs() < 1e-6);

        // Projected matrix should be PSD
        let diag = diagnose_psd(&proj, 1e-8).unwrap();
        assert!(diag.is_psd);
        assert_eq!(diag.numerical_rank, 1);
    }
}
