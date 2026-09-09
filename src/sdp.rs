//! Semidefinite Programming (SDP) problem formulation, KKT diagnostics, and first-order baseline solver.
//!
//! Solves standard primal-dual semidefinite programs of the form:
//!
//! **Primal:**
//! $$\min_{X \in \mathcal{S}^n} \langle C, X \rangle \quad \text{subject to} \quad \langle A_i, X \rangle = b_i \; (i = 1, \dots, m), \; X \succeq 0$$
//!
//! **Dual:**
//! $$\max_{y \in \mathbb{R}^m, S \in \mathcal{S}^n} b^T y \quad \text{subject to} \quad \sum_{i=1}^m y_i A_i + S = C, \; S \succeq 0$$

use crate::eigensystem::jacobi_eigen;
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use crate::symmetric::SymmetricMatrix;

/// Status of the SDP solver termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpStatus {
    /// Converged to verified primal and dual optimality within requested tolerance.
    Optimal,
    /// Converged to verified primal and dual optimality within numerical tolerance.
    OptimalNumerical,
    /// Maximum iteration limit reached before full convergence.
    MaxIterationsReached,
    /// Numerical divergence, ill-conditioned problem, or infeasibility detected.
    InfeasibleOrDiverged,
    /// Unbounded objective detected along an extreme ray.
    UnboundedOrDiverged,
    /// Numerical breakdown (e.g. singular matrix, NaN/Inf detected).
    NumericalFailure,
}

/// Configuration parameters for the SDP solver.
#[derive(Debug, Clone)]
pub struct SdpSolverConfig {
    /// Target KKT tolerance for primal/dual feasibility and duality gap (default 1e-4).
    pub tolerance: f64,
    /// Maximum allowed iterations (default 500).
    pub max_iterations: usize,
    /// ADMM penalty parameter $\rho > 0$ (default 1.0).
    pub rho: f64,
    /// Whether to enforce strict tolerance checks (returns Err if not Optimal).
    pub strict: bool,
}

impl Default for SdpSolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-4,
            max_iterations: 500,
            rho: 1.0,
            strict: false,
        }
    }
}

/// Detailed KKT residual and spectrum report for an SDP candidate solution $(X, y, S)$.
#[derive(Debug, Clone, PartialEq)]
pub struct SdpResidualReport {
    /// Primal equality residual $\max_i |\langle A_i, X \rangle - b_i|$.
    pub primal_equality_residual: f64,
    /// Primal PSD cone violation $|\min(0, \lambda_{\min}(X))|$.
    pub primal_psd_violation: f64,
    /// Dual equality residual $\|\sum y_i A_i + S - C\|_F$.
    pub dual_equality_residual: f64,
    /// Dual PSD cone violation $|\min(0, \lambda_{\min}(S))|$.
    pub dual_psd_violation: f64,
    /// Relative duality gap $\frac{|\langle C, X \rangle - b^T y|}{1 + |\langle C, X \rangle| + |b^T y|}$.
    pub relative_duality_gap: f64,
    /// Complementary slackness residual $\frac{|\langle X, S \rangle|}{n}$.
    pub complementary_slackness_residual: f64,
    /// Minimum eigenvalue of primal solution $X$.
    pub min_eigenvalue_x: f64,
    /// Minimum eigenvalue of dual slack $S$.
    pub min_eigenvalue_s: f64,
}

impl SdpResidualReport {
    /// Returns true if all primal and dual feasibility conditions are satisfied within `tolerance`.
    pub fn is_feasible(&self, tolerance: f64) -> bool {
        self.primal_equality_residual <= tolerance
            && self.primal_psd_violation <= tolerance
            && self.dual_equality_residual <= tolerance
            && self.dual_psd_violation <= tolerance
    }

    /// Returns true if the solution is feasible and the relative duality gap is within `tolerance`.
    pub fn is_optimal(&self, tolerance: f64) -> bool {
        self.is_feasible(tolerance) && self.relative_duality_gap <= tolerance
    }

    /// Maximum KKT violation across all primal, dual, and complementarity conditions.
    pub fn max_kkt_violation(&self) -> f64 {
        self.primal_equality_residual
            .max(self.primal_psd_violation)
            .max(self.dual_equality_residual)
            .max(self.dual_psd_violation)
            .max(self.relative_duality_gap)
            .max(self.complementary_slackness_residual)
    }
}

/// Output of the SDP solver.
#[derive(Debug, Clone)]
pub struct SdpSolution {
    /// Primal matrix variable $X \succeq 0$.
    pub x: SymmetricMatrix,
    /// Dual vector $y \in \mathbb{R}^m$.
    pub y: Vec<f64>,
    /// Dual slack matrix $S \succeq 0$.
    pub s: SymmetricMatrix,
    /// Primal objective value $\langle C, X \rangle$.
    pub primal_objective: f64,
    /// Dual objective value $b^T y$.
    pub dual_objective: f64,
    /// KKT residual and diagnostic report.
    pub residuals: SdpResidualReport,
    /// Total iterations performed.
    pub iterations: usize,
    /// Termination status.
    pub status: SdpStatus,
}

/// Standard Semidefinite Program definition.
#[derive(Debug, Clone)]
pub struct SdpProblem {
    /// Cost matrix $C \in \mathcal{S}^n$.
    pub c: SymmetricMatrix,
    /// Linear constraint matrices $A_1, \dots, A_m \in \mathcal{S}^n$.
    pub a_constraints: Vec<SymmetricMatrix>,
    /// Right-hand side constraint values $b \in \mathbb{R}^m$.
    pub b: Vec<f64>,
}

impl SdpProblem {
    /// Creates a new SDP problem with input validation.
    pub fn new(
        c: SymmetricMatrix,
        a_constraints: Vec<SymmetricMatrix>,
        b: Vec<f64>,
    ) -> SciResult<Self> {
        let n = c.size();
        let m = a_constraints.len();
        if b.len() != m {
            return Err(SciError::InvalidParameter(
                "number of constraint matrices must match length of vector b",
            ));
        }
        for (i, &val) in b.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(SciError::DomainError(
                    "vector b entries must be finite (not NaN or Inf)",
                ));
            }
            let _ = i;
        }
        for a in &a_constraints {
            if a.size() != n {
                return Err(SciError::InvalidParameter(
                    "constraint matrix size does not match cost matrix C",
                ));
            }
            if !a.is_finite() {
                return Err(SciError::DomainError(
                    "constraint matrix entries must be finite",
                ));
            }
        }
        if !c.is_finite() {
            return Err(SciError::DomainError(
                "cost matrix C entries must be finite",
            ));
        }
        Ok(Self {
            c,
            a_constraints,
            b,
        })
    }

    /// Dimension $n$ of the matrix variable $X \in \mathcal{S}^n$.
    pub fn matrix_dim(&self) -> usize {
        self.c.size()
    }

    /// Number of linear equality constraints $m$.
    pub fn num_constraints(&self) -> usize {
        self.b.len()
    }

    /// Independently verifies a candidate primal-dual triplet $(X, y, S)$ against the SDP problem.
    ///
    /// Evaluates:
    /// - Primal equality residual $\max_i |\langle A_i, X \rangle - b_i|$
    /// - Primal PSD violation $|\min(0, \lambda_{\min}(X))|$
    /// - Dual equality residual $\|\sum y_i A_i + S - C\|_F$
    /// - Dual PSD violation $|\min(0, \lambda_{\min}(S))|$
    /// - Relative duality gap
    /// - Complementary slackness
    pub fn verify_solution(
        &self,
        x: &SymmetricMatrix,
        y: &[f64],
        s: &SymmetricMatrix,
        eigen_tol: f64,
    ) -> SciResult<SdpResidualReport> {
        let n = self.matrix_dim();
        let m = self.num_constraints();

        if x.size() != n || s.size() != n || y.len() != m {
            return Err(SciError::InvalidParameter(
                "dimension mismatch in candidate solution verification",
            ));
        }

        if !x.is_finite() || !s.is_finite() || y.iter().any(|v| !v.is_finite()) {
            return Err(SciError::DomainError(
                "solution candidate contains non-finite values (NaN or Inf)",
            ));
        }

        // 1. Primal equality residual: max_i |<A_i, X> - b_i|
        let mut max_p_eq = 0.0_f64;
        for i in 0..m {
            let axi = self.a_constraints[i].trace_inner_product(x)?;
            let diff = (axi - self.b[i]).abs();
            if diff > max_p_eq {
                max_p_eq = diff;
            }
        }

        // 2. Primal PSD cone eigenvalue: lambda_min(X)
        let x_dense = x.to_dense()?;
        let eig_x = jacobi_eigen(&x_dense, eigen_tol.min(1e-12), 200)?;
        let min_x = *eig_x
            .values
            .iter()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(&0.0);
        let p_psd_viol = (-min_x).max(0.0);

        // 3. Dual equality residual: || sum_i y_i A_i + S - C ||_F
        let mut sum_ya = SymmetricMatrix::zeros(n)?;
        for (i, &yi) in y.iter().enumerate().take(m) {
            let scaled_ai = self.a_constraints[i].scale(yi);
            sum_ya = sum_ya.add(&scaled_ai)?;
        }
        let dual_lhs = sum_ya.add(s)?;
        let dual_diff = dual_lhs.sub(&self.c)?;
        let d_eq_res = dual_diff.frobenius_norm();

        // 4. Dual PSD cone eigenvalue: lambda_min(S)
        let s_dense = s.to_dense()?;
        let eig_s = jacobi_eigen(&s_dense, eigen_tol.min(1e-12), 200)?;
        let min_s = *eig_s
            .values
            .iter()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(&0.0);
        let d_psd_viol = (-min_s).max(0.0);

        // 5. Duality Gap
        let p_obj = self.c.trace_inner_product(x)?;
        let d_obj: f64 = y.iter().zip(self.b.iter()).map(|(yi, bi)| yi * bi).sum();
        let gap = (p_obj - d_obj).abs() / (1.0 + p_obj.abs() + d_obj.abs());

        // 6. Complementary Slackness: |<X, S>| / n
        let xs_trace = x.trace_inner_product(s)?.abs();
        let comp_slack = xs_trace / n as f64;

        Ok(SdpResidualReport {
            primal_equality_residual: max_p_eq,
            primal_psd_violation: p_psd_viol,
            dual_equality_residual: d_eq_res,
            dual_psd_violation: d_psd_viol,
            relative_duality_gap: gap,
            complementary_slackness_residual: comp_slack,
            min_eigenvalue_x: min_x,
            min_eigenvalue_s: min_s,
        })
    }
}

/// Independently verifies an SDP candidate solution $(X, y, S)$ against KKT conditions.
pub fn verify_sdp_candidate(
    problem: &SdpProblem,
    x: &SymmetricMatrix,
    y: &[f64],
    s: &SymmetricMatrix,
    eigen_tolerance: f64,
) -> SciResult<SdpResidualReport> {
    problem.verify_solution(x, y, s, eigen_tolerance)
}

impl SdpProblem {
    /// Solves the SDP problem using an ADMM (Alternating Direction Method of Multipliers) solver.
    pub fn solve(&self, config: &SdpSolverConfig) -> SciResult<SdpSolution> {
        let n = self.matrix_dim();
        let m = self.num_constraints();
        let rho = if config.rho > 0.0 { config.rho } else { 1.0 };
        let tol = config.tolerance;

        if tol <= 0.0 {
            return Err(SciError::InvalidParameter("tolerance must be positive"));
        }
        if config.max_iterations == 0 {
            return Err(SciError::InvalidParameter("max_iterations must be > 0"));
        }

        if m == 0 {
            // Unconstrained SDP: min <C, X> s.t. X >= 0
            let x = SymmetricMatrix::zeros(n)?;
            let s = self.c.clone();
            let y = vec![];
            let residuals = self.verify_solution(&x, &y, &s, tol)?;
            let status = if residuals.is_optimal(tol) {
                SdpStatus::Optimal
            } else {
                SdpStatus::InfeasibleOrDiverged
            };

            if config.strict && status != SdpStatus::Optimal {
                return Err(SciError::NonConvergent(
                    "unconstrained SDP is unbounded or non-optimal under strict check",
                ));
            }

            return Ok(SdpSolution {
                x,
                y,
                s,
                primal_objective: 0.0,
                dual_objective: 0.0,
                residuals,
                iterations: 1,
                status,
            });
        }

        // Build Gram matrix G_ij = <A_i, A_j> for constraint projection
        let mut gram_data = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                gram_data[i * m + j] =
                    self.a_constraints[i].trace_inner_product(&self.a_constraints[j])?;
            }
        }

        let raw_gram = DynamicMatrix::new(m, m, gram_data.clone())?;
        if raw_gram.lu_decompose().is_err() {
            return Err(SciError::SingularMatrix(
                "constraint matrices are linearly dependent (singular Gram matrix)",
            ));
        }

        for i in 0..m {
            gram_data[i * m + i] += 1e-10; // Regularization
        }
        let gram = DynamicMatrix::new(m, m, gram_data)?;
        let gram_lu = gram.lu_decompose().map_err(|_| {
            SciError::SingularMatrix(
                "constraint matrices are linearly dependent / singular Gram matrix",
            )
        })?;

        // Initialize variables
        let mut x = SymmetricMatrix::identity(n)?;
        let mut z = SymmetricMatrix::identity(n)?;
        let mut lambda = SymmetricMatrix::zeros(n)?; // Multiplier for X - Z = 0
        let mut y = vec![0.0; m];

        let mut final_iter = 0;

        for iter in 0..config.max_iterations {
            final_iter = iter + 1;

            // --- Step 1: Update X by projecting V = Z - (C + Lambda)/rho onto { <A_i, X> = b_i } ---
            let mut v_data = vec![0.0; z.packed_len()];
            for (idx, item) in v_data.iter_mut().enumerate().take(z.packed_len()) {
                *item = z.raw_data()[idx] - (self.c.raw_data()[idx] + lambda.raw_data()[idx]) / rho;
            }
            let v = SymmetricMatrix::new(n, v_data)?;

            // Compute rhs_nu = b - A(V)
            let mut rhs_nu = vec![0.0; m];
            for (i, item) in rhs_nu.iter_mut().enumerate().take(m) {
                let avi = self.a_constraints[i].trace_inner_product(&v)?;
                *item = self.b[i] - avi;
            }

            // Solve G * nu = rhs_nu
            let nu = gram_lu.solve(&rhs_nu)?;
            y = nu.iter().map(|&nui| rho * nui).collect();

            // X = V + sum_i nu_i * A_i
            let mut x_new_data = v.raw_data().to_vec();
            for (i, &nui) in nu.iter().enumerate().take(m) {
                let ai_data = self.a_constraints[i].raw_data();
                for (idx, item) in x_new_data.iter_mut().enumerate().take(ai_data.len()) {
                    *item += nui * ai_data[idx];
                }
            }
            x = SymmetricMatrix::new(n, x_new_data)?;

            // --- Step 2: Update Z by projecting (X + Lambda/rho) onto PSD cone S_+^n ---
            let mut w_data = vec![0.0; x.packed_len()];
            for (idx, item) in w_data.iter_mut().enumerate().take(x.packed_len()) {
                *item = x.raw_data()[idx] + lambda.raw_data()[idx] / rho;
            }
            let w = SymmetricMatrix::new(n, w_data)?;
            let w_dense = w.to_dense()?;

            let eig_w = jacobi_eigen(&w_dense, 1e-12, 100)?;
            let mut z_dense_data = vec![0.0; n * n];
            for r in 0..n {
                for c in 0..n {
                    let mut sum = 0.0;
                    for (k, &lam_k) in eig_w.values.iter().enumerate().take(n) {
                        let clamped_lam = lam_k.max(0.0);
                        let vrk = eig_w.vectors.get(r, k)?;
                        let vck = eig_w.vectors.get(c, k)?;
                        sum += clamped_lam * vrk * vck;
                    }
                    z_dense_data[r * n + c] = sum;
                }
            }
            let z_dense = DynamicMatrix::new(n, n, z_dense_data)?;
            z = SymmetricMatrix::from_dense(&z_dense)?;

            // --- Step 3: Update Lambda multiplier ---
            for idx in 0..lambda.packed_len() {
                let diff = x.raw_data()[idx] - z.raw_data()[idx];
                lambda.raw_data_mut()[idx] += rho * diff;
            }

            // --- Step 4: KKT Convergence Check ---
            let mut s_data = self.c.raw_data().to_vec();
            for (i, &yi) in y.iter().enumerate().take(m) {
                let ai_data = self.a_constraints[i].raw_data();
                for (idx, item) in s_data.iter_mut().enumerate().take(ai_data.len()) {
                    *item -= yi * ai_data[idx];
                }
            }
            let s = SymmetricMatrix::new(n, s_data)?;

            if let Ok(res) = self.verify_solution(&x, &y, &s, tol) {
                if res.is_optimal(tol) {
                    break;
                }
            }
        }

        // Dual slack matrix S = C - sum_i y_i A_i
        let mut s_data = self.c.raw_data().to_vec();
        for (i, &yi) in y.iter().enumerate().take(m) {
            let ai_data = self.a_constraints[i].raw_data();
            for (idx, item) in s_data.iter_mut().enumerate().take(ai_data.len()) {
                *item -= yi * ai_data[idx];
            }
        }
        let s = SymmetricMatrix::new(n, s_data)?;

        // Rigorous independent verification of candidate (X, y, S)
        let residuals = self.verify_solution(&x, &y, &s, tol)?;
        let status = if residuals.is_optimal(tol) {
            SdpStatus::Optimal
        } else {
            SdpStatus::MaxIterationsReached
        };

        if config.strict && status != SdpStatus::Optimal {
            return Err(SciError::NonConvergent(
                "ADMM SDP solver did not converge to verified optimal solution within strict tolerance",
            ));
        }

        let p_obj = self.c.trace_inner_product(&x)?;
        let d_obj: f64 = y.iter().zip(self.b.iter()).map(|(yi, bi)| yi * bi).sum();

        Ok(SdpSolution {
            x,
            y,
            s,
            primal_objective: p_obj,
            dual_objective: d_obj,
            residuals,
            iterations: final_iter,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdp_simple_1d_problem() {
        // Min x_11 s.t. x_11 = 2.0, X >= 0
        let c = SymmetricMatrix::new(1, vec![1.0]).unwrap();
        let a1 = SymmetricMatrix::new(1, vec![1.0]).unwrap();
        let b = vec![2.0];

        let problem = SdpProblem::new(c, vec![a1], b).unwrap();
        let config = SdpSolverConfig {
            tolerance: 1e-3,
            max_iterations: 100,
            rho: 1.0,
            strict: false,
        };

        let sol = problem.solve(&config).unwrap();
        assert_eq!(sol.status, SdpStatus::Optimal);
        assert!((sol.primal_objective - 2.0).abs() < 1e-2);
        assert!((sol.x.get(0, 0).unwrap() - 2.0).abs() < 1e-2);
        assert!(sol.residuals.primal_equality_residual < 1e-2);
    }
}
