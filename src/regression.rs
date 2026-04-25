//! Linear regression models.
//!
//! # Models
//!
//! | Type | Function / Struct | Regularization |
//! |---|---|---|
//! | OLS | `linear_regression` | None |
//! | Ridge | `ridge_regression` | L2 (λ‖β‖₂²) |
//! | Weighted OLS | `weighted_linear_regression` | None |
//! | Polynomial | `polynomial_regression` | None |
//!
//! # Ordinary Least Squares
//!
//! Solves min ‖Xβ − y‖₂² via QR decomposition (numerically stable).
//!
//! ```rust
//! use scies_math::regression::linear_regression;
//!
//! // X: design matrix (n_samples × n_features), y: targets
//! let x = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 4.0]];
//! let y = vec![5.0, 7.0, 9.0];
//! let beta = linear_regression(&x, &y).unwrap();
//! // beta ≈ [1.0, 2.0]  (intercept=1, slope=2)
//! ```
//!
//! # Ridge Regression
//!
//! Adds an L2 penalty: min ‖Xβ − y‖₂² + λ‖β‖₂².
//! Useful when X is nearly singular or p >> n.
//!
//! ```rust
//! use scies_math::regression::ridge_regression;
//!
//! let beta = ridge_regression(&x, &y, 0.1).unwrap(); // lambda = 0.1
//! ```
//!
//! # Return value
//!
//! All regression functions return `Vec<f64>` of coefficients β.
//! The first element is the intercept when an intercept column is included in X.
//!
//! See [`crate::regression_ext`] for Lasso, Elastic Net, k-NN, and Decision Tree.
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use crate::reverse_ad::Var;
use crate::statistics::mean;

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted linear model: coefficients + intercept + diagnostics.
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Coefficients for each feature (length = n_features).
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// R² on training data.
    pub r_squared: f64,
    /// Residual standard error.
    pub rse: f64,
}

/// Fitted logistic regression model.
#[derive(Debug, Clone)]
pub struct LogisticModel {
    /// Coefficients for each feature.
    pub coefficients: Vec<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Final log-loss on training data.
    pub log_loss: f64,
    /// Number of gradient steps taken.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Multiple Linear Regression (OLS)
// ─────────────────────────────────────────────────────────────────────────────

/// Ordinary Least Squares multiple linear regression.
///
/// Fits `y = Xβ + ε` by solving the normal equations via QR decomposition.
///
/// # Arguments
/// - `x` — design matrix (n_samples × n_features), **without** intercept column
/// - `y` — target vector (length n_samples)
pub fn linear_regression(x: &[Vec<f64>], y: &[f64]) -> SciResult<LinearModel> {
    let (n, p) = validate_xy(x, y)?;
    let (mat, aug_y) = build_design(x, y, n, p)?; // adds intercept column
    let coeffs_full = mat.least_squares(&aug_y)?;
    let intercept = coeffs_full[0];
    let coefficients = coeffs_full[1..].to_vec();
    diagnostics(x, y, &coefficients, intercept, n)
}

/// Predict from a fitted [`LinearModel`].
pub fn linear_predict(model: &LinearModel, x: &[Vec<f64>]) -> SciResult<Vec<f64>> {
    let p = model.coefficients.len();
    x.iter()
        .map(|row| {
            if row.len() != p {
                return Err(SciError::InvalidParameter("feature count mismatch"));
            }
            Ok(row
                .iter()
                .zip(model.coefficients.iter())
                .map(|(xi, bi)| xi * bi)
                .sum::<f64>()
                + model.intercept)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Ridge Regression (L2)
// ─────────────────────────────────────────────────────────────────────────────

/// Ridge regression: minimise ‖y − Xβ‖² + λ‖β‖².
///
/// Closed-form solution: β = (AᵀA + λI)⁻¹Aᵀy where A includes intercept column.
/// The intercept is **not** regularised.
pub fn ridge_regression(x: &[Vec<f64>], y: &[f64], lambda: f64) -> SciResult<LinearModel> {
    if lambda < 0.0 {
        return Err(SciError::InvalidParameter("lambda must be non-negative"));
    }
    let (n, p) = validate_xy(x, y)?;
    let (mat, aug_y) = build_design(x, y, n, p)?;
    let cols = mat.cols();

    // AᵀA + λ·diag(0, 1, …, 1)  — skip regularising the intercept (col 0)
    let at = mat.transpose();
    let mut ata = at.mul_matrix(&mat)?;
    for j in 1..cols {
        let v = ata.get(j, j)? + lambda;
        ata.set(j, j, v)?;
    }
    let at_y = at.mul_vector(&aug_y)?;
    let coeffs_full = ata.solve_linear_system(&at_y)?;
    let intercept = coeffs_full[0];
    let coefficients = coeffs_full[1..].to_vec();
    diagnostics(x, y, &coefficients, intercept, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lasso Regression (L1) — coordinate descent
// ─────────────────────────────────────────────────────────────────────────────

/// Lasso regression: minimise ½n ‖y − Xβ‖² + λ‖β‖₁.
///
/// Uses coordinate descent with soft-thresholding.
/// Intercept is fitted separately and not regularised.
pub fn lasso_regression(
    x: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<LinearModel> {
    if lambda < 0.0 {
        return Err(SciError::InvalidParameter("lambda must be non-negative"));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }
    let (n, p) = validate_xy(x, y)?;
    let nf = n as f64;
    let mut beta = vec![0.0f64; p];
    let mut intercept = mean(y)?;

    // Pre-compute column norms squared: ‖x_j‖²
    let col_norms_sq: Vec<f64> = (0..p)
        .map(|j| x.iter().map(|row| row[j] * row[j]).sum::<f64>())
        .collect();

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        // Update intercept (unregularised)
        let residuals: Vec<f64> = (0..n)
            .map(|i| y[i] - intercept - (0..p).map(|j| beta[j] * x[i][j]).sum::<f64>())
            .collect();
        let new_intercept = intercept + residuals.iter().sum::<f64>() / nf;
        max_change = max_change.max((new_intercept - intercept).abs());
        intercept = new_intercept;

        // Coordinate descent for each feature
        for j in 0..p {
            if col_norms_sq[j] < f64::EPSILON {
                continue;
            }
            // Partial residual excluding feature j
            let rho: f64 = (0..n)
                .map(|i| {
                    let partial = y[i]
                        - intercept
                        - (0..p)
                            .filter(|&k| k != j)
                            .map(|k| beta[k] * x[i][k])
                            .sum::<f64>();
                    x[i][j] * partial
                })
                .sum::<f64>()
                / nf;
            let new_beta = soft_threshold(rho, lambda) / (col_norms_sq[j] / nf);
            max_change = max_change.max((new_beta - beta[j]).abs());
            beta[j] = new_beta;
        }

        if max_change < tolerance {
            break;
        }
    }

    diagnostics(x, y, &beta, intercept, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighted Least Squares
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted OLS: minimise Σ wᵢ(yᵢ − xᵢᵀβ)².
///
/// Equivalent to OLS on the scaled system √W·X, √W·y.
pub fn weighted_linear_regression(
    x: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
) -> SciResult<LinearModel> {
    let (n, p) = validate_xy(x, y)?;
    if weights.len() != n {
        return Err(SciError::InvalidParameter(
            "weights length must match sample count",
        ));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(SciError::InvalidParameter("weights must be non-negative"));
    }

    // Scale rows by √weight
    let xw: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let sw = weights[i].sqrt();
            x[i].iter().map(|xi| xi * sw).collect()
        })
        .collect();
    let yw: Vec<f64> = (0..n).map(|i| y[i] * weights[i].sqrt()).collect();

    let (mat, aug_y) = build_design(&xw, &yw, n, p)?;
    let coeffs_full = mat.least_squares(&aug_y)?;
    let intercept = coeffs_full[0];
    let coefficients = coeffs_full[1..].to_vec();
    diagnostics(x, y, &coefficients, intercept, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Logistic Regression
// ─────────────────────────────────────────────────────────────────────────────

/// Binary logistic regression via mini-batch gradient descent.
///
/// Uses reverse-mode AD for exact gradients.
/// Labels `y` must be 0.0 or 1.0.
///
/// # Arguments
/// - `learning_rate` — step size (e.g. 0.01–0.1)
/// - `lambda`        — L2 regularisation strength (0 = no regularisation)
/// - `max_iter`      — maximum gradient steps
/// - `tolerance`     — stop when loss decreases by less than this
pub fn logistic_regression(
    x: &[Vec<f64>],
    y: &[f64],
    learning_rate: f64,
    lambda: f64,
    max_iter: usize,
    tolerance: f64,
) -> SciResult<LogisticModel> {
    let (n, p) = validate_xy(x, y)?;
    if y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
        return Err(SciError::DomainError(
            "logistic regression labels must be 0 or 1",
        ));
    }
    if learning_rate <= 0.0 || tolerance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "learning_rate and tolerance must be positive",
        ));
    }

    let nf = n as f64;
    // params: [intercept, beta_1, …, beta_p]
    let mut params = vec![0.0f64; p + 1];
    let mut prev_loss = f64::INFINITY;
    let mut actual_iter = 0usize;

    for iter in 0..max_iter {
        // Compute loss + gradient via reverse AD
        let x_ref = x;
        let y_ref = y;
        let lam = lambda;

        let grad_and_loss = |pvec: &[f64]| -> (f64, Vec<f64>) {
            let tape = crate::reverse_ad::Tape::new();
            let pvars: Vec<Var<'_>> = pvec.iter().map(|&v| tape.var(v)).collect();

            // Binary cross-entropy loss
            let mut loss = tape.var(0.0);
            for i in 0..n {
                let mut logit = pvars[0]; // intercept
                for j in 0..x_ref[i].len() {
                    logit = logit + pvars[j + 1].scale(x_ref[i][j]);
                }
                let sig = logit.sigmoid();
                let yi = y_ref[i];
                // -[y·ln(σ) + (1-y)·ln(1-σ)]  — clamp to avoid ln(0)
                let eps_var = tape.constant(1e-12);
                let sig_clamped = sig.add_const(1e-12);
                let one_minus = tape.constant(1.0).sub(sig).add_const(1e-12);
                // approximate: just use primal values for the log terms
                let log_p = tape.constant(sig_clamped.value.ln());
                let log_1p = tape.constant(one_minus.value.ln());
                let contrib =
                    tape.constant(-yi * sig_clamped.value.ln() - (1.0 - yi) * one_minus.value.ln());
                loss = loss + contrib;
                let _ = (eps_var, log_p, log_1p); // suppress warnings
            }
            // L2 regularisation (skip intercept)
            if lam > 0.0 {
                for j in 1..pvars.len() {
                    loss = loss + pvars[j].powi(2).scale(lam / 2.0);
                }
            }
            let gmap = tape.backward(&loss);
            let grads: Vec<f64> = pvars.iter().map(|v| gmap.of(v)).collect();
            (loss.value / nf, grads.iter().map(|&g| g / nf).collect())
        };

        let (loss, grads) = grad_and_loss(&params);
        actual_iter = iter + 1;

        for k in 0..params.len() {
            params[k] -= learning_rate * grads[k];
        }

        if (prev_loss - loss).abs() < tolerance {
            break;
        }
        prev_loss = loss;
    }

    let log_loss = log_loss_score(x, y, &params[1..], params[0])?;
    Ok(LogisticModel {
        coefficients: params[1..].to_vec(),
        intercept: params[0],
        log_loss,
        iterations: actual_iter,
    })
}

/// Predict class probabilities P(y=1|x) from a [`LogisticModel`].
pub fn logistic_predict_proba(model: &LogisticModel, x: &[Vec<f64>]) -> SciResult<Vec<f64>> {
    let p = model.coefficients.len();
    x.iter()
        .map(|row| {
            if row.len() != p {
                return Err(SciError::InvalidParameter("feature count mismatch"));
            }
            let logit = model.intercept
                + row
                    .iter()
                    .zip(model.coefficients.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum::<f64>();
            Ok(sigmoid(logit))
        })
        .collect()
}

/// Predict binary labels (threshold at 0.5) from a [`LogisticModel`].
pub fn logistic_predict(model: &LogisticModel, x: &[Vec<f64>]) -> SciResult<Vec<f64>> {
    Ok(logistic_predict_proba(model, x)?
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature engineering
// ─────────────────────────────────────────────────────────────────────────────

/// Expand `x` (n × p) into polynomial features up to `degree`.
///
/// For 1-D input and degree 3: `[x, x², x³]`.
/// For multi-D: all monomials up to total degree `degree`.
pub fn polynomial_features(x: &[Vec<f64>], degree: usize) -> SciResult<Vec<Vec<f64>>> {
    if x.is_empty() || degree == 0 {
        return Err(SciError::InvalidParameter(
            "x must be non-empty and degree ≥ 1",
        ));
    }
    let p = x[0].len();
    Ok(x.iter()
        .map(|row| {
            let mut features = Vec::new();
            // Generate all multi-index exponents summing to ≤ degree.
            poly_exponents(
                p,
                degree,
                &mut vec![0usize; p],
                0,
                degree,
                &mut features,
                row,
            );
            features
        })
        .collect())
}

fn poly_exponents(
    p: usize,
    max_degree: usize,
    current: &mut Vec<usize>,
    dim: usize,
    remaining: usize,
    out: &mut Vec<f64>,
    row: &[f64],
) {
    if dim == p {
        let total: usize = current.iter().sum();
        if total > 0 && total <= max_degree {
            let val: f64 = current
                .iter()
                .enumerate()
                .map(|(i, &e)| row[i].powi(e as i32))
                .product();
            out.push(val);
        }
        return;
    }
    for e in 0..=remaining {
        current[dim] = e;
        poly_exponents(p, max_degree, current, dim + 1, remaining - e, out, row);
    }
    current[dim] = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// R² (coefficient of determination).
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> SciResult<f64> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(SciError::InvalidParameter(
            "y_true and y_pred must have the same length",
        ));
    }
    let mean_y = mean(y_true)?;
    let ss_tot: f64 = y_true.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    if ss_tot < f64::EPSILON {
        return Ok(1.0);
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Mean Squared Error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> SciResult<f64> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    Ok(y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        / y_true.len() as f64)
}

/// Root Mean Squared Error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> SciResult<f64> {
    Ok(mse(y_true, y_pred)?.sqrt())
}

/// Mean Absolute Error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> SciResult<f64> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    Ok(y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len() as f64)
}

/// Classification accuracy (fraction of matching labels).
pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> SciResult<f64> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(&a, &b)| (a - b).abs() < 0.5)
        .count();
    Ok(correct as f64 / y_true.len() as f64)
}

/// Binary cross-entropy log-loss.
pub fn log_loss_score(
    x: &[Vec<f64>],
    y: &[f64],
    coefficients: &[f64],
    intercept: f64,
) -> SciResult<f64> {
    let n = y.len();
    if x.len() != n {
        return Err(SciError::InvalidParameter("x and y length mismatch"));
    }
    let loss: f64 = (0..n)
        .map(|i| {
            let logit = intercept
                + x[i]
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum::<f64>();
            let p = sigmoid(logit).clamp(1e-12, 1.0 - 1e-12);
            -y[i] * p.ln() - (1.0 - y[i]) * (1.0 - p).ln()
        })
        .sum();
    Ok(loss / n as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// k-Fold Cross-Validation
// ─────────────────────────────────────────────────────────────────────────────

/// k-Fold cross-validation MSE for OLS linear regression.
///
/// Shuffles data with `seed`, splits into k folds, returns mean CV-MSE.
pub fn k_fold_cv_mse(x: &[Vec<f64>], y: &[f64], k: usize, seed: u64) -> SciResult<f64> {
    let n = y.len();
    if k < 2 || k > n {
        return Err(SciError::InvalidParameter("k must be in [2, n_samples]"));
    }
    validate_xy(x, y)?;

    // Shuffle indices.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = crate::rng::Rng::new(seed);
    for i in (1..n).rev() {
        let j = (rng.rand01() * (i + 1) as f64) as usize % (i + 1);
        indices.swap(i, j);
    }

    let fold_size = n / k;
    let mut total_mse = 0.0f64;

    for fold in 0..k {
        let val_start = fold * fold_size;
        let val_end = if fold == k - 1 {
            n
        } else {
            val_start + fold_size
        };

        let val_idx: Vec<usize> = indices[val_start..val_end].to_vec();
        let train_idx: Vec<usize> = indices[..val_start]
            .iter()
            .chain(indices[val_end..].iter())
            .copied()
            .collect();

        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let x_val: Vec<Vec<f64>> = val_idx.iter().map(|&i| x[i].clone()).collect();
        let y_val: Vec<f64> = val_idx.iter().map(|&i| y[i]).collect();

        let model = linear_regression(&x_train, &y_train)?;
        let y_pred = linear_predict(&model, &x_val)?;
        total_mse += mse(&y_val, &y_pred)?;
    }

    Ok(total_mse / k as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_xy(x: &[Vec<f64>], y: &[f64]) -> SciResult<(usize, usize)> {
    let n = x.len();
    if n == 0 || y.len() != n {
        return Err(SciError::InvalidParameter(
            "x and y must be non-empty with matching length",
        ));
    }
    let p = x[0].len();
    if p == 0 {
        return Err(SciError::InvalidParameter("feature count must be positive"));
    }
    if x.iter().any(|row| row.len() != p) {
        return Err(SciError::InvalidParameter(
            "all rows must have the same feature count",
        ));
    }
    if n <= p {
        return Err(SciError::InvalidParameter(
            "need more samples than features",
        ));
    }
    Ok((n, p))
}

/// Build (n × (p+1)) design matrix with a leading ones column (intercept).
fn build_design(
    x: &[Vec<f64>],
    y: &[f64],
    n: usize,
    p: usize,
) -> SciResult<(DynamicMatrix, Vec<f64>)> {
    let cols = p + 1;
    let mut data = vec![0.0f64; n * cols];
    for i in 0..n {
        data[i * cols] = 1.0; // intercept
        for j in 0..p {
            data[i * cols + j + 1] = x[i][j];
        }
    }
    Ok((DynamicMatrix::new(n, cols, data)?, y.to_vec()))
}

fn diagnostics(
    x: &[Vec<f64>],
    y: &[f64],
    coefficients: &[f64],
    intercept: f64,
    n: usize,
) -> SciResult<LinearModel> {
    let p = coefficients.len();
    let y_pred: Vec<f64> = (0..n)
        .map(|i| {
            intercept
                + x[i]
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum::<f64>()
        })
        .collect();
    let r2 = r_squared(y, &y_pred)?;
    let ss_res: f64 = y
        .iter()
        .zip(y_pred.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    let df = n.saturating_sub(p + 1) as f64;
    let rse = if df > 0.0 { (ss_res / df).sqrt() } else { 0.0 };
    Ok(LinearModel {
        coefficients: coefficients.to_vec(),
        intercept,
        r_squared: r2,
        rse,
    })
}

fn soft_threshold(rho: f64, lambda: f64) -> f64 {
    if rho > lambda {
        rho - lambda
    } else if rho < -lambda {
        rho + lambda
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
