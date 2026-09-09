//! Standardized experiment and calculation logging for reproducibility.
//!
//! Captures complete provenance metadata (crate version, tolerance, seed,
//! matrix dimension, residual, sweeps/iterations, and spectrum diagnostics)
//! to ensure scientific rigor in relaxation experiments.

/// A structured record of a numerical or spectral calculation run.
#[derive(Debug, Clone, PartialEq)]
pub struct CalculationLog {
    /// Version of `scies-math-th` used for the run.
    pub crate_version: &'static str,
    /// Name or identifier of the experiment / instance.
    pub experiment_name: String,
    /// Dimension $n$ of the principal matrix variable.
    pub dimension: usize,
    /// Number of constraints in the problem.
    pub num_constraints: usize,
    /// Optional RNG seed used for stochastic heuristics.
    pub seed: Option<u64>,
    /// Target numerical tolerance.
    pub tolerance: f64,
    /// Total iterations or sweeps performed.
    pub iterations: usize,
    /// Primal equality residual $\|\mathcal{A}(X) - b\|_\infty$.
    pub primal_residual: f64,
    /// Dual equality residual.
    pub dual_residual: f64,
    /// Relative duality gap.
    pub duality_gap: f64,
    /// Smallest eigenvalue of the solution matrix.
    pub min_eigenvalue: f64,
    /// Largest eigenvalue of the solution matrix.
    pub max_eigenvalue: f64,
    /// Termination status descriptor (e.g. "Optimal", "MaxIterationsReached").
    pub status: String,
}

impl CalculationLog {
    /// Creates a new calculation log initialized with the current crate version.
    pub fn new(experiment_name: impl Into<String>, dimension: usize) -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION"),
            experiment_name: experiment_name.into(),
            dimension,
            num_constraints: 0,
            seed: None,
            tolerance: 1e-6,
            iterations: 0,
            primal_residual: 0.0,
            dual_residual: 0.0,
            duality_gap: 0.0,
            min_eigenvalue: 0.0,
            max_eigenvalue: 0.0,
            status: "Unknown".to_string(),
        }
    }

    /// Formats a human-readable diagnostic summary.
    pub fn summary(&self) -> String {
        format!(
            "[{}] v{} | dim={} | constr={} | tol={:e} | iters={} | pres={:e} | dres={:e} | gap={:e} | eig_min={:.4e} | status={}",
            self.experiment_name,
            self.crate_version,
            self.dimension,
            self.num_constraints,
            self.tolerance,
            self.iterations,
            self.primal_residual,
            self.dual_residual,
            self.duality_gap,
            self.min_eigenvalue,
            self.status
        )
    }
}
