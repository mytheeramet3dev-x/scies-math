//! Extended probability: multivariate distributions, information theory,
//! extreme value, Bayesian conjugates, copulas, and missing inverse CDFs.

use crate::distributions::{ln_gamma, reg_inc_beta, reg_inc_gamma};
use crate::errors::{SciError, SciResult};
use crate::probability::normal_cdf;

// ══════════════════════════════════════════════════════════════════════════════
// Missing inverse CDFs
// ══════════════════════════════════════════════════════════════════════════════

/// Normal quantile / probit: Φ⁻¹(p) via rational Beasley-Springer-Moro approximation.
pub fn normal_icdf(p: f64) -> SciResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(SciError::DomainError("p must be in (0, 1)"));
    }
    // Rational approximation (Abramowitz & Stegun 26.2.17)
    let q = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();
    let c = [2.515_517, 0.802_853, 0.010_328_f64];
    let d = [1.432_788, 0.189_269, 0.001_308_f64];
    let num = c[0] + t * (c[1] + t * c[2]);
    let den = 1.0 + t * (d[0] + t * (d[1] + t * d[2]));
    let x = t - num / den;
    Ok(if p < 0.5 { -x } else { x })
}

/// Gamma inverse CDF via Newton-Raphson on the regularised incomplete gamma.
pub fn gamma_icdf(p: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(SciError::DomainError("p must be in (0, 1)"));
    }
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter("alpha, beta > 0"));
    }
    let mut x = alpha * beta; // initial guess
    for _ in 0..50 {
        let cdf = reg_inc_gamma(alpha, x / beta);
        let pdf = ((alpha - 1.0) * x.ln() - x / beta - ln_gamma(alpha) - alpha * beta.ln()).exp();
        if pdf.abs() < f64::EPSILON {
            break;
        }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.max(1e-300);
        if dx.abs() < 1e-10 {
            break;
        }
    }
    Ok(x)
}

/// Beta inverse CDF via Newton-Raphson on the regularised incomplete beta.
pub fn beta_icdf(p: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(SciError::DomainError("p must be in (0, 1)"));
    }
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter("alpha, beta > 0"));
    }
    let mut x = alpha / (alpha + beta);
    let ln_b = ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta);
    for _ in 0..50 {
        let cdf = reg_inc_beta(alpha, beta, x);
        let pdf = ((alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_b).exp();
        if pdf.abs() < f64::EPSILON {
            break;
        }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.clamp(1e-300, 1.0 - 1e-15);
        if dx.abs() < 1e-10 {
            break;
        }
    }
    Ok(x)
}

// ══════════════════════════════════════════════════════════════════════════════
// Multivariate Normal
// ══════════════════════════════════════════════════════════════════════════════

/// Multivariate Normal log-PDF: log N(x | μ, Σ).
///
/// `cov_chol` is the lower-triangular Cholesky factor L of Σ (L·Lᵀ = Σ).
pub fn mvnormal_log_pdf(x: &[f64], mean: &[f64], cov_chol: &[Vec<f64>]) -> SciResult<f64> {
    let n = x.len();
    if mean.len() != n || cov_chol.len() != n {
        return Err(SciError::InvalidParameter("dimension mismatch"));
    }
    // Solve L·z = x - μ  (forward substitution)
    let diff: Vec<f64> = x.iter().zip(mean).map(|(xi, mi)| xi - mi).collect();
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut s = diff[i];
        for j in 0..i {
            s -= cov_chol[i][j] * z[j];
        }
        let d = cov_chol[i][i];
        if d.abs() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        z[i] = s / d;
    }
    let maha_sq: f64 = z.iter().map(|v| v * v).sum();
    let log_det: f64 = cov_chol
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].abs().ln())
        .sum::<f64>()
        * 2.0;
    let pi = core::f64::consts::PI;
    Ok(-0.5 * (n as f64 * (2.0 * pi).ln() + log_det + maha_sq))
}

/// Multivariate Normal PDF.
pub fn mvnormal_pdf(x: &[f64], mean: &[f64], cov_chol: &[Vec<f64>]) -> SciResult<f64> {
    Ok(mvnormal_log_pdf(x, mean, cov_chol)?.exp())
}

// ══════════════════════════════════════════════════════════════════════════════
// Dirichlet distribution
// ══════════════════════════════════════════════════════════════════════════════

/// Dirichlet PDF: p(x | α) over the simplex Σxᵢ = 1, xᵢ > 0.
pub fn dirichlet_pdf(x: &[f64], alpha: &[f64]) -> SciResult<f64> {
    Ok(dirichlet_log_pdf(x, alpha)?.exp())
}

/// Dirichlet log-PDF.
pub fn dirichlet_log_pdf(x: &[f64], alpha: &[f64]) -> SciResult<f64> {
    let k = x.len();
    if alpha.len() != k {
        return Err(SciError::InvalidParameter(
            "x and alpha must have same length",
        ));
    }
    if x.iter().any(|&xi| xi <= 0.0) || alpha.iter().any(|&ai| ai <= 0.0) {
        return Err(SciError::DomainError("all x > 0 and alpha > 0 required"));
    }
    let alpha_sum: f64 = alpha.iter().sum();
    let log_beta: f64 = alpha.iter().map(|&ai| ln_gamma(ai)).sum::<f64>() - ln_gamma(alpha_sum);
    let log_p: f64 = alpha
        .iter()
        .zip(x)
        .map(|(&ai, &xi)| (ai - 1.0) * xi.ln())
        .sum::<f64>();
    Ok(log_p - log_beta)
}

/// Dirichlet mean: $\mathbb{E}\[x_i\] = \alpha_i / \sum \alpha$.
pub fn dirichlet_mean(alpha: &[f64]) -> Vec<f64> {
    let s: f64 = alpha.iter().sum();
    alpha.iter().map(|&a| a / s).collect()
}

/// Dirichlet variance: $\text{Var}\[x_i\] = \alpha_i(\sum \alpha - \alpha_i) / ((\sum \alpha)^2(\sum \alpha + 1))$.
pub fn dirichlet_variance(alpha: &[f64]) -> Vec<f64> {
    let s: f64 = alpha.iter().sum();
    let denom = s * s * (s + 1.0);
    alpha.iter().map(|&a| a * (s - a) / denom).collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Gaussian Mixture Model
// ══════════════════════════════════════════════════════════════════════════════

/// Evaluate a 1D **Gaussian Mixture Model** PDF at x.
///
/// `weights` must sum to 1; `means` and `stds` define each component.
pub fn gmm_pdf(x: f64, weights: &[f64], means: &[f64], stds: &[f64]) -> SciResult<f64> {
    let k = weights.len();
    if means.len() != k || stds.len() != k {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    let pi = core::f64::consts::PI;
    let mut sum = 0.0f64;
    for i in 0..k {
        if stds[i] <= 0.0 {
            return Err(SciError::InvalidParameter("std must be positive"));
        }
        let z = (x - means[i]) / stds[i];
        sum += weights[i] * (-0.5 * z * z).exp() / (stds[i] * (2.0 * pi).sqrt());
    }
    Ok(sum)
}

/// GMM log-likelihood over a dataset.
pub fn gmm_log_likelihood(
    data: &[f64],
    weights: &[f64],
    means: &[f64],
    stds: &[f64],
) -> SciResult<f64> {
    data.iter()
        .map(|&x| {
            Ok(gmm_pdf(x, weights, means, stds)?
                .max(f64::MIN_POSITIVE)
                .ln())
        })
        .try_fold(0.0f64, |a, r: SciResult<f64>| Ok(a + r?))
}

// ══════════════════════════════════════════════════════════════════════════════
// Information theory
// ══════════════════════════════════════════════════════════════════════════════

/// Shannon entropy H(p) = -Σ pᵢ log pᵢ (nats).
pub fn entropy(p: &[f64]) -> SciResult<f64> {
    if p.iter().any(|&v| v < 0.0) {
        return Err(SciError::DomainError("probabilities must be non-negative"));
    }
    Ok(p.iter()
        .map(|&pi| if pi > 0.0 { -pi * pi.ln() } else { 0.0 })
        .sum())
}

/// Cross-entropy H(p, q) = -Σ pᵢ log qᵢ.
pub fn cross_entropy(p: &[f64], q: &[f64]) -> SciResult<f64> {
    if p.len() != q.len() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    Ok(p.iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            if pi > 0.0 && qi > 0.0 {
                -pi * qi.ln()
            } else if pi == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .sum())
}

/// KL divergence D_KL(p ‖ q) = Σ pᵢ log(pᵢ/qᵢ).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> SciResult<f64> {
    if p.len() != q.len() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    Ok(p.iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            if pi > 0.0 && qi > 0.0 {
                pi * (pi / qi).ln()
            } else if pi == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .sum())
}

/// Jensen-Shannon divergence (symmetric, bounded [0, ln2]).
pub fn js_divergence(p: &[f64], q: &[f64]) -> SciResult<f64> {
    let m: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    Ok(0.5 * kl_divergence(p, &m)? + 0.5 * kl_divergence(q, &m)?)
}

/// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) from joint distribution matrix.
/// `joint[i*cols + j]` = P(X=i, Y=j).
pub fn mutual_information(joint: &[f64], rows: usize, cols: usize) -> SciResult<f64> {
    if joint.len() != rows * cols {
        return Err(SciError::InvalidParameter("joint length mismatch"));
    }
    let px: Vec<f64> = (0..rows)
        .map(|i| (0..cols).map(|j| joint[i * cols + j]).sum())
        .collect();
    let py: Vec<f64> = (0..cols)
        .map(|j| (0..rows).map(|i| joint[i * cols + j]).sum())
        .collect();
    let hx = entropy(&px)?;
    let hy = entropy(&py)?;
    let hxy = entropy(joint)?;
    Ok(hx + hy - hxy)
}

// ══════════════════════════════════════════════════════════════════════════════
// Extreme Value distributions
// ══════════════════════════════════════════════════════════════════════════════

/// Generalized Extreme Value (GEV) PDF.  ξ=0 → Gumbel, ξ>0 → Fréchet, ξ<0 → Weibull.
pub fn gev_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    Ok(gev_log_pdf(x, mu, sigma, xi)?.exp())
}

pub fn gev_log_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        // Gumbel limit
        Ok(-z - (-z).exp() - sigma.ln())
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 {
            return Err(SciError::DomainError("x outside GEV support"));
        }
        Ok(-(1.0 + 1.0 / xi) * t.ln() - t.powf(-1.0 / xi) - sigma.ln())
    }
}

pub fn gev_cdf(x: f64, mu: f64, sigma: f64, xi: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        Ok((-(-z).exp()).exp())
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 {
            return Err(SciError::DomainError("x outside GEV support"));
        }
        Ok((-t.powf(-1.0 / xi)).exp())
    }
}

/// Generalized Pareto Distribution (GPD) PDF — exceedance over threshold.
pub fn gpd_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        if z < 0.0 {
            return Err(SciError::DomainError("x < mu"));
        }
        Ok((-z).exp() / sigma)
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 {
            return Err(SciError::DomainError("x outside GPD support"));
        }
        Ok(t.powf(-1.0 / xi - 1.0) / sigma)
    }
}

pub fn gpd_cdf(x: f64, mu: f64, sigma: f64, xi: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-10 {
        if z < 0.0 {
            return Err(SciError::DomainError("x < mu"));
        }
        Ok(1.0 - (-z).exp())
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 {
            return Err(SciError::DomainError("x outside GPD support"));
        }
        Ok(1.0 - t.powf(-1.0 / xi))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Bayesian conjugate updates
// ══════════════════════════════════════════════════════════════════════════════

/// **Beta-Binomial conjugate update**.
///
/// Prior: Beta(α, β).  Observe k successes in n trials.
/// Posterior: Beta(α + k, β + n − k).
pub fn beta_binomial_update(
    alpha_prior: f64,
    beta_prior: f64,
    k: u64,
    n: u64,
) -> SciResult<(f64, f64)> {
    if k > n {
        return Err(SciError::InvalidParameter("k > n"));
    }
    Ok((alpha_prior + k as f64, beta_prior + (n - k) as f64))
}

/// **Normal-Normal conjugate update** (known variance σ²).
///
/// Prior: N(μ₀, τ₀²).  Observe n samples with mean x̄ and known variance σ².
/// Returns posterior (μₙ, τₙ²).
pub fn normal_normal_update(
    mu0: f64,
    tau0_sq: f64,
    x_bar: f64,
    sigma_sq: f64,
    n: usize,
) -> SciResult<(f64, f64)> {
    if tau0_sq <= 0.0 || sigma_sq <= 0.0 {
        return Err(SciError::InvalidParameter("variances must be positive"));
    }
    let tau_n_sq = 1.0 / (1.0 / tau0_sq + n as f64 / sigma_sq);
    let mu_n = tau_n_sq * (mu0 / tau0_sq + n as f64 * x_bar / sigma_sq);
    Ok((mu_n, tau_n_sq))
}

/// **Gamma-Poisson conjugate update** (Poisson rate, Gamma prior).
///
/// Prior: Gamma(α, β).  Observe sum of counts Σk over n intervals.
/// Posterior: Gamma(α + Σk, β / (1 + n·β)).
pub fn gamma_poisson_update(
    alpha_prior: f64,
    beta_prior: f64,
    sum_k: u64,
    n: usize,
) -> SciResult<(f64, f64)> {
    if alpha_prior <= 0.0 || beta_prior <= 0.0 {
        return Err(SciError::InvalidParameter("alpha, beta > 0"));
    }
    let alpha_post = alpha_prior + sum_k as f64;
    let beta_post = beta_prior / (1.0 + n as f64 * beta_prior);
    Ok((alpha_post, beta_post))
}

// ══════════════════════════════════════════════════════════════════════════════
// Gaussian Copula
// ══════════════════════════════════════════════════════════════════════════════

/// **Gaussian copula** density C(u, v | ρ) for bivariate case.
///
/// u, v ∈ (0,1); ρ ∈ (−1, 1) is the correlation.
/// Transforms marginals via Φ⁻¹, then evaluates bivariate normal.
pub fn gaussian_copula_pdf(u: f64, v: f64, rho: f64) -> SciResult<f64> {
    if rho.abs() >= 1.0 {
        return Err(SciError::InvalidParameter("|rho| < 1"));
    }
    let x = normal_icdf(u)?;
    let y = normal_icdf(v)?;
    let r2 = 1.0 - rho * rho;
    let exponent = -(rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * r2);
    Ok(exponent.exp() / r2.sqrt())
}

/// Gaussian copula CDF approximated numerically (bivariate normal CDF).
pub fn gaussian_copula_cdf(u: f64, v: f64, rho: f64) -> SciResult<f64> {
    if rho.abs() >= 1.0 {
        return Err(SciError::InvalidParameter("|rho| < 1"));
    }
    let x = normal_icdf(u)?;
    let y = normal_icdf(v)?;
    // Numerical bivariate normal CDF via product if ρ ≈ 0, else Simpson integration
    if rho.abs() < 1e-10 {
        return Ok(normal_cdf(x, 0.0, 1.0)? * normal_cdf(y, 0.0, 1.0)?);
    }
    // Integrate Φ₂ via Gauss-Legendre on the joint PDF
    let n = 32usize;
    let bvn = bivariate_normal_cdf(x, y, rho, n);
    Ok(bvn)
}

/// Bivariate standard normal CDF Φ₂(x, y; ρ) via 2D quadrature (Simpson).
fn bivariate_normal_cdf(xb: f64, yb: f64, rho: f64, n: usize) -> f64 {
    let xlo = -6.0_f64;
    let ylo = -6.0_f64;
    let hx = (xb - xlo) / n as f64;
    let hy = (yb - ylo) / n as f64;
    let pi = core::f64::consts::PI;
    let r2 = 1.0 - rho * rho;
    let kern = |x: f64, y: f64| -> f64 {
        let e = -(x * x - 2.0 * rho * x * y + y * y) / (2.0 * r2);
        e.exp() / (2.0 * pi * r2.sqrt())
    };
    // Simple trapezoidal 2D
    let mut sum = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let x = xlo + (i as f64 + 0.5) * hx;
            let y = ylo + (j as f64 + 0.5) * hy;
            sum += kern(x, y);
        }
    }
    (sum * hx * hy).clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// Discrete distributions (additional)
// ══════════════════════════════════════════════════════════════════════════════

/// Hypergeometric PMF: P(X=k | N, K, n) — drawing without replacement.
pub fn hypergeometric_pmf(k: u64, n_pop: u64, k_pop: u64, n_draw: u64) -> SciResult<f64> {
    if k_pop > n_pop || n_draw > n_pop {
        return Err(SciError::InvalidParameter("invalid hypergeometric params"));
    }
    let log_p = ln_binom(k_pop, k) + ln_binom(n_pop - k_pop, n_draw - k) - ln_binom(n_pop, n_draw);
    Ok(log_p.exp())
}

fn ln_binom(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

/// Zipf (Zeta) PMF: P(X=k) ∝ k^{−s},  k = 1, 2, …, n_max.
pub fn zipf_pmf(k: u64, s: f64, n_max: u64) -> SciResult<f64> {
    if k == 0 || k > n_max {
        return Err(SciError::DomainError("k must be in [1, n_max]"));
    }
    if s <= 0.0 {
        return Err(SciError::InvalidParameter("s > 0"));
    }
    let norm: f64 = (1..=n_max).map(|i| (i as f64).powf(-s)).sum();
    Ok((k as f64).powf(-s) / norm)
}
