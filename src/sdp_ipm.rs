//! Numerical Primal-Dual Interior-Point Method (Mehrotra Predictor-Corrector) for Semidefinite Programs.
//!
//! Solves standard SDPs with independent KKT residual verification:
//!
//! $$\min \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i \; (i = 1, \dots, m), \; X \succeq 0$$
//!
//! # Numerical Reliability Notice
//! Solutions produced by this solver are floating-point approximations. Always inspect
//! the returned [`crate::sdp::SdpResidualReport`] to verify primal/dual feasibility and duality gap.

use crate::eigensystem::jacobi_eigen;
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use crate::sdp::{SdpProblem, SdpSolution, SdpStatus};
use crate::symmetric::SymmetricMatrix;

/// Configuration parameters for the Interior-Point SDP Solver.
#[derive(Debug, Clone)]
pub struct SdpIpmConfig {
    /// Target duality gap and KKT tolerance (default 1e-6).
    pub tolerance: f64,
    /// Maximum allowed interior-point iterations (default 100).
    pub max_iterations: usize,
    /// Step-size damping factor $\gamma \in (0, 1)$ to stay strictly in the interior (default 0.95).
    pub step_damping: f64,
}

impl Default for SdpIpmConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 100,
            step_damping: 0.95,
        }
    }
}

/// Solves an SDP problem using the Primal-Dual Interior-Point Method (Mehrotra Predictor-Corrector).
///
/// Returns an error if the problem is malformed, singular, or if the numerical step computation fails.
pub fn solve_sdp_ipm(problem: &SdpProblem, config: &SdpIpmConfig) -> SciResult<SdpSolution> {
    let n = problem.matrix_dim();
    let m = problem.num_constraints();
    let tol = config.tolerance;
    let gamma = config.step_damping;

    if tol <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }
    if config.max_iterations == 0 {
        return Err(SciError::InvalidParameter("max_iterations must be > 0"));
    }
    if gamma <= 0.0 || gamma >= 1.0 {
        return Err(SciError::InvalidParameter(
            "step_damping parameter must be strictly in (0.0, 1.0)",
        ));
    }

    if m == 0 {
        return problem.solve(&crate::sdp::SdpSolverConfig {
            tolerance: tol,
            max_iterations: config.max_iterations,
            rho: 1.0,
            strict: false,
        });
    }

    // Initialize strictly positive definite variables X0 = I, S0 = I, y0 = 0
    let mut x_dense = DynamicMatrix::identity(n)?;
    let mut s_dense = DynamicMatrix::identity(n)?;
    let mut y = vec![0.0; m];

    let a_dense: Vec<DynamicMatrix> = problem
        .a_constraints
        .iter()
        .map(|a| a.to_dense())
        .collect::<SciResult<Vec<_>>>()?;
    let c_dense = problem.c.to_dense()?;

    let mut final_iter = 0;

    for iter in 0..config.max_iterations {
        final_iter = iter + 1;

        // Current duality measure mu = Tr(X * S) / n
        let xs = x_dense.mul_matrix(&s_dense)?;
        let mut tr_xs = 0.0;
        for i in 0..n {
            tr_xs += xs.get(i, i)?;
        }
        let mu = (tr_xs / n as f64).max(1e-16);

        // Primal residual r_p[i] = b[i] - <A_i, X>
        let mut r_p = vec![0.0; m];
        let mut max_rp = 0.0_f64;
        for (i, item) in r_p.iter_mut().enumerate().take(m) {
            let mut axi = 0.0;
            for r in 0..n {
                for c in 0..n {
                    axi += a_dense[i].get(r, c)? * x_dense.get(r, c)?;
                }
            }
            let diff = problem.b[i] - axi;
            *item = diff;
            if diff.abs() > max_rp {
                max_rp = diff.abs();
            }
        }

        // Dual residual R_d = C - sum_i y_i A_i - S
        let mut r_d_data = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut sum_ya = 0.0;
                for i in 0..m {
                    sum_ya += y[i] * a_dense[i].get(r, c)?;
                }
                r_d_data[r * n + c] = c_dense.get(r, c)? - sum_ya - s_dense.get(r, c)?;
            }
        }
        let r_d = DynamicMatrix::new(n, n, r_d_data)?;
        let mut rd_norm_sq = 0.0;
        for r in 0..n {
            for c in 0..n {
                let v = r_d.get(r, c)?;
                rd_norm_sq += v * v;
            }
        }
        let rd_norm = rd_norm_sq.sqrt();

        // Duality Gap
        let mut p_obj = 0.0;
        for r in 0..n {
            for c in 0..n {
                p_obj += c_dense.get(r, c)? * x_dense.get(r, c)?;
            }
        }
        let d_obj: f64 = y.iter().zip(problem.b.iter()).map(|(yi, bi)| yi * bi).sum();
        let gap = (p_obj - d_obj).abs() / (1.0 + p_obj.abs() + d_obj.abs());

        if max_rp < tol && rd_norm < tol && gap < tol {
            break;
        }

        // Invert S (Strict - error if singular)
        let s_inv = s_dense.inverse().map_err(|_| {
            SciError::SingularMatrix("dual slack matrix S is singular during IPM iteration")
        })?;

        // Build Schur complement matrix M_ij = Tr(A_i * S^{-1} * A_j * X)
        let mut schur_data = vec![0.0; m * m];
        let mut s_inv_a: Vec<DynamicMatrix> = Vec::with_capacity(m);
        for item in a_dense.iter().take(m) {
            s_inv_a.push(s_inv.mul_matrix(item)?);
        }

        for i in 0..m {
            for j in 0..m {
                let a_j_x = a_dense[j].mul_matrix(&x_dense)?;
                let prod = s_inv_a[i].mul_matrix(&a_j_x)?;
                let mut tr = 0.0;
                for k in 0..n {
                    tr += prod.get(k, k)?;
                }
                schur_data[i * m + j] = tr;
            }
            schur_data[i * m + i] += 1e-10; // Regularization
        }
        let schur = DynamicMatrix::new(m, m, schur_data)?;
        let schur_lu = schur.lu_decompose().map_err(|_| {
            SciError::SingularMatrix("Schur complement matrix is singular in IPM iteration")
        })?;

        // --- Step 1: Predictor Step (Affine Scaling Direction with mu = 0) ---
        let r_c_aff = xs.scale(-1.0)?;
        let t_aff = s_inv.mul_matrix(&r_c_aff.sub_matrix(&x_dense.mul_matrix(&r_d)?)?)?;

        let mut h_aff = vec![0.0; m];
        for i in 0..m {
            let at = a_dense[i].mul_matrix(&t_aff)?;
            let mut tr = 0.0;
            for k in 0..n {
                tr += at.get(k, k)?;
            }
            h_aff[i] = r_p[i] - tr;
        }

        let dy_aff = schur_lu.solve(&h_aff)?;

        // dS_aff = R_d - sum_i dy_aff[i] * A_i
        let mut ds_aff_data = r_d.raw_data().to_vec();
        for (i, &dyi) in dy_aff.iter().enumerate().take(m) {
            let ai_data = a_dense[i].raw_data();
            for (idx, item) in ds_aff_data.iter_mut().enumerate().take(ai_data.len()) {
                *item -= dyi * ai_data[idx];
            }
        }
        let ds_aff = DynamicMatrix::new(n, n, ds_aff_data)?;

        // dX_aff = S^{-1} * (R_c_aff - X * dS_aff)
        let dx_aff = s_inv.mul_matrix(&r_c_aff.sub_matrix(&x_dense.mul_matrix(&ds_aff)?)?)?;
        let dx_aff = dx_aff.add_matrix(&dx_aff.transpose())?.scale(0.5)?;

        // Step-length computation for affine step
        let alpha_p_aff = compute_max_step_dense(&x_dense, &dx_aff, 1.0)?;
        let alpha_d_aff = compute_max_step_dense(&s_dense, &ds_aff, 1.0)?;

        let x_aff = x_dense.add_matrix(&dx_aff.scale(alpha_p_aff)?)?;
        let s_aff = s_dense.add_matrix(&ds_aff.scale(alpha_d_aff)?)?;
        let xs_aff = x_aff.mul_matrix(&s_aff)?;
        let mut tr_xs_aff = 0.0;
        for i in 0..n {
            tr_xs_aff += xs_aff.get(i, i)?;
        }
        let mu_aff = (tr_xs_aff / n as f64).max(1e-16);

        // Centering parameter sigma
        let sigma = (mu_aff / mu).powi(3).clamp(1e-4, 0.95);

        // --- Step 2: Combined Corrector Step ---
        let mut rc_comb_data = vec![0.0; n * n];
        let dx_ds_aff = dx_aff.mul_matrix(&ds_aff)?;
        for r in 0..n {
            for c in 0..n {
                let diag = if r == c { sigma * mu } else { 0.0 };
                rc_comb_data[r * n + c] = diag - xs.get(r, c)? - dx_ds_aff.get(r, c)?;
            }
        }
        let r_c_comb = DynamicMatrix::new(n, n, rc_comb_data)?;
        let t_comb = s_inv.mul_matrix(&r_c_comb.sub_matrix(&x_dense.mul_matrix(&r_d)?)?)?;

        let mut h_comb = vec![0.0; m];
        for i in 0..m {
            let at = a_dense[i].mul_matrix(&t_comb)?;
            let mut tr = 0.0;
            for k in 0..n {
                tr += at.get(k, k)?;
            }
            h_comb[i] = r_p[i] - tr;
        }

        let dy_comb = schur_lu.solve(&h_comb)?;

        let mut ds_comb_data = r_d.raw_data().to_vec();
        for (i, &dyi) in dy_comb.iter().enumerate().take(m) {
            let ai_data = a_dense[i].raw_data();
            for (idx, item) in ds_comb_data.iter_mut().enumerate().take(ai_data.len()) {
                *item -= dyi * ai_data[idx];
            }
        }
        let ds_comb = DynamicMatrix::new(n, n, ds_comb_data)?;
        let ds_comb = ds_comb.add_matrix(&ds_comb.transpose())?.scale(0.5)?;

        let dx_comb = s_inv.mul_matrix(&r_c_comb.sub_matrix(&x_dense.mul_matrix(&ds_comb)?)?)?;
        let dx_comb = dx_comb.add_matrix(&dx_comb.transpose())?.scale(0.5)?;

        // Damped step sizes
        let alpha_p = compute_max_step_dense(&x_dense, &dx_comb, gamma)?;
        let alpha_d = compute_max_step_dense(&s_dense, &ds_comb, gamma)?;
        let alpha = alpha_p.min(alpha_d).max(1e-4);

        x_dense = x_dense.add_matrix(&dx_comb.scale(alpha)?)?;
        x_dense = x_dense.add_matrix(&x_dense.transpose())?.scale(0.5)?;

        s_dense = s_dense.add_matrix(&ds_comb.scale(alpha)?)?;
        s_dense = s_dense.add_matrix(&s_dense.transpose())?.scale(0.5)?;

        for (i, &dyi) in dy_comb.iter().enumerate().take(m) {
            y[i] += alpha * dyi;
        }
    }

    let x_sym = SymmetricMatrix::from_dense(&x_dense)?;
    let s_sym = SymmetricMatrix::from_dense(&s_dense)?;

    // Rigorous independent verification
    let residuals = problem.verify_solution(&x_sym, &y, &s_sym, tol)?;
    let status = if residuals.is_optimal(tol) {
        SdpStatus::Optimal
    } else {
        SdpStatus::MaxIterationsReached
    };

    let p_obj = problem.c.trace_inner_product(&x_sym)?;
    let d_obj: f64 = y.iter().zip(problem.b.iter()).map(|(yi, bi)| yi * bi).sum();

    Ok(SdpSolution {
        x: x_sym,
        y,
        s: s_sym,
        primal_objective: p_obj,
        dual_objective: d_obj,
        residuals,
        iterations: final_iter,
        status,
    })
}

fn compute_max_step_dense(x: &DynamicMatrix, dx: &DynamicMatrix, max_step: f64) -> SciResult<f64> {
    let mut step = max_step;
    for _ in 0..25 {
        let valid = dx
            .scale(step)
            .and_then(|scaled| x.add_matrix(&scaled))
            .and_then(|cand| jacobi_eigen(&cand, 1e-10, 50))
            .map(|eig| {
                let min_ev = *eig
                    .values
                    .iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&0.0);
                min_ev > 1e-9
            })
            .unwrap_or(false);

        if valid {
            return Ok(step);
        }
        step *= 0.7;
    }
    if step < 1e-12 {
        return Err(SciError::NonConvergent(
            "IPM line search failed to maintain positive definiteness",
        ));
    }
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdp_ipm_solver() {
        // Min Tr(C * X) s.t. X_00 = 1.0, X_11 = 2.0, X >= 0
        let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 3.0]).unwrap();
        let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
        let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0]).unwrap();

        let problem = SdpProblem::new(c, vec![a1, a2], vec![1.0, 2.0]).unwrap();
        let config = SdpIpmConfig {
            tolerance: 1e-3,
            max_iterations: 50,
            step_damping: 0.9,
        };

        let sol = solve_sdp_ipm(&problem, &config).unwrap();
        assert_eq!(sol.status, SdpStatus::Optimal);
        assert!((sol.primal_objective - 7.0).abs() < 0.05);
    }
}
