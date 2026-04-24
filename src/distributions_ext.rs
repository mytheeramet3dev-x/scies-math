//! Extended distributions: Pareto, Laplace, Logistic, Gumbel, Rayleigh,
//! Truncated Normal, Triangular, Inverse Gaussian, Von Mises, Maxwell-Boltzmann,
//! Chi, Multinomial, Beta-Binomial, plus missing inverse CDFs.

use crate::distributions::{ln_gamma, reg_inc_gamma};
use crate::errors::{SciError, SciResult};
use crate::probability_ext::normal_icdf;

// ══════════════════════════════════════════════════════════════════════════════
// Missing inverse CDFs for base distributions
// ══════════════════════════════════════════════════════════════════════════════

/// Log-Normal inverse CDF: icdf(p) = exp(μ + σ·Φ⁻¹(p)).
pub fn lognormal_icdf(p: f64, mu: f64, sigma: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    Ok((mu + sigma * normal_icdf(p)?).exp())
}

/// Gamma inverse CDF via Newton-Raphson (convenience re-export in this module).
pub fn gamma_icdf(p: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    crate::probability_ext::gamma_icdf(p, alpha, beta)
}

/// Beta inverse CDF via Newton-Raphson.
pub fn beta_icdf(p: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    crate::probability_ext::beta_icdf(p, alpha, beta)
}

// ══════════════════════════════════════════════════════════════════════════════
// Pareto (Type I)
// ══════════════════════════════════════════════════════════════════════════════

/// Pareto PDF: f(x) = α·xₘᵅ / xᵅ⁺¹  for x ≥ xₘ.
pub fn pareto_pdf(x: f64, x_min: f64, alpha: f64) -> SciResult<f64> {
    if x_min <= 0.0 || alpha <= 0.0 {
        return Err(SciError::InvalidParameter("x_min > 0, alpha > 0"));
    }
    if x < x_min {
        return Ok(0.0);
    }
    Ok(alpha * x_min.powf(alpha) / x.powf(alpha + 1.0))
}

pub fn pareto_cdf(x: f64, x_min: f64, alpha: f64) -> SciResult<f64> {
    if x_min <= 0.0 || alpha <= 0.0 {
        return Err(SciError::InvalidParameter("x_min > 0, alpha > 0"));
    }
    if x < x_min {
        return Ok(0.0);
    }
    Ok(1.0 - (x_min / x).powf(alpha))
}

pub fn pareto_icdf(p: f64, x_min: f64, alpha: f64) -> SciResult<f64> {
    if !(0.0..1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ [0, 1)"));
    }
    if x_min <= 0.0 || alpha <= 0.0 {
        return Err(SciError::InvalidParameter("x_min > 0, alpha > 0"));
    }
    Ok(x_min / (1.0 - p).powf(1.0 / alpha))
}

pub fn pareto_mean(x_min: f64, alpha: f64) -> SciResult<f64> {
    if alpha <= 1.0 {
        return Err(SciError::DomainError("mean undefined for alpha <= 1"));
    }
    Ok(alpha * x_min / (alpha - 1.0))
}

// ══════════════════════════════════════════════════════════════════════════════
// Laplace (Double Exponential)
// ══════════════════════════════════════════════════════════════════════════════

/// Laplace PDF: f(x) = (1/2b) exp(−|x − μ|/b).
pub fn laplace_pdf(x: f64, mu: f64, b: f64) -> SciResult<f64> {
    if b <= 0.0 {
        return Err(SciError::InvalidParameter("b > 0"));
    }
    Ok((-(x - mu).abs() / b).exp() / (2.0 * b))
}

pub fn laplace_cdf(x: f64, mu: f64, b: f64) -> SciResult<f64> {
    if b <= 0.0 {
        return Err(SciError::InvalidParameter("b > 0"));
    }
    if x < mu {
        Ok(0.5 * ((x - mu) / b).exp())
    } else {
        Ok(1.0 - 0.5 * (-(x - mu) / b).exp())
    }
}

pub fn laplace_icdf(p: f64, mu: f64, b: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ [0, 1]"));
    }
    if b <= 0.0 {
        return Err(SciError::InvalidParameter("b > 0"));
    }
    Ok(if p < 0.5 {
        mu + b * (2.0 * p).ln()
    } else {
        mu - b * (2.0 * (1.0 - p)).ln()
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Logistic
// ══════════════════════════════════════════════════════════════════════════════

/// Logistic PDF: f(x) = e^{−z} / (s·(1 + e^{−z})²),  z = (x−μ)/s.
pub fn logistic_pdf(x: f64, mu: f64, s: f64) -> SciResult<f64> {
    if s <= 0.0 {
        return Err(SciError::InvalidParameter("s > 0"));
    }
    let e = (-(x - mu) / s).exp();
    Ok(e / (s * (1.0 + e) * (1.0 + e)))
}

pub fn logistic_cdf(x: f64, mu: f64, s: f64) -> SciResult<f64> {
    if s <= 0.0 {
        return Err(SciError::InvalidParameter("s > 0"));
    }
    Ok(1.0 / (1.0 + (-(x - mu) / s).exp()))
}

pub fn logistic_icdf(p: f64, mu: f64, s: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ (0, 1)"));
    }
    if s <= 0.0 {
        return Err(SciError::InvalidParameter("s > 0"));
    }
    Ok(mu + s * (p / (1.0 - p)).ln())
}

// ══════════════════════════════════════════════════════════════════════════════
// Gumbel (Type I Extreme Value)
// ══════════════════════════════════════════════════════════════════════════════

/// Gumbel PDF: f(x) = (1/β) exp(−z − e^{−z}),  z = (x−μ)/β.
pub fn gumbel_pdf(x: f64, mu: f64, beta: f64) -> SciResult<f64> {
    if beta <= 0.0 {
        return Err(SciError::InvalidParameter("beta > 0"));
    }
    let z = (x - mu) / beta;
    Ok((-z - (-z).exp()).exp() / beta)
}

pub fn gumbel_cdf(x: f64, mu: f64, beta: f64) -> SciResult<f64> {
    if beta <= 0.0 {
        return Err(SciError::InvalidParameter("beta > 0"));
    }
    Ok((-(-(x - mu) / beta).exp()).exp())
}

pub fn gumbel_icdf(p: f64, mu: f64, beta: f64) -> SciResult<f64> {
    if !(0.0..1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ (0, 1)"));
    }
    if beta <= 0.0 {
        return Err(SciError::InvalidParameter("beta > 0"));
    }
    Ok(mu - beta * (-p.ln()).ln())
}

// ══════════════════════════════════════════════════════════════════════════════
// Rayleigh
// ══════════════════════════════════════════════════════════════════════════════

/// Rayleigh PDF: f(x) = (x/σ²) exp(−x²/(2σ²))  for x ≥ 0.
pub fn rayleigh_pdf(x: f64, sigma: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    if x < 0.0 {
        return Ok(0.0);
    }
    Ok(x / (sigma * sigma) * (-x * x / (2.0 * sigma * sigma)).exp())
}

pub fn rayleigh_cdf(x: f64, sigma: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    if x < 0.0 {
        return Ok(0.0);
    }
    Ok(1.0 - (-x * x / (2.0 * sigma * sigma)).exp())
}

pub fn rayleigh_icdf(p: f64, sigma: f64) -> SciResult<f64> {
    if !(0.0..1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ [0, 1)"));
    }
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    Ok(sigma * (-2.0 * (1.0 - p).ln()).sqrt())
}

// ══════════════════════════════════════════════════════════════════════════════
// Triangular
// ══════════════════════════════════════════════════════════════════════════════

/// Triangular PDF on [a, b] with mode c.
pub fn triangular_pdf(x: f64, a: f64, b: f64, c: f64) -> SciResult<f64> {
    if !(a <= c && c <= b) || a == b {
        return Err(SciError::InvalidParameter("need a ≤ c ≤ b, a ≠ b"));
    }
    let range = b - a;
    if x < a || x > b {
        return Ok(0.0);
    }
    if x <= c {
        Ok(2.0 * (x - a) / (range * (c - a)))
    } else {
        Ok(2.0 * (b - x) / (range * (b - c)))
    }
}

pub fn triangular_cdf(x: f64, a: f64, b: f64, c: f64) -> SciResult<f64> {
    if !(a <= c && c <= b) || a == b {
        return Err(SciError::InvalidParameter("need a ≤ c ≤ b, a ≠ b"));
    }
    if x <= a {
        return Ok(0.0);
    }
    if x >= b {
        return Ok(1.0);
    }
    let range = b - a;
    if x <= c {
        Ok((x - a) * (x - a) / (range * (c - a)))
    } else {
        Ok(1.0 - (b - x) * (b - x) / (range * (b - c)))
    }
}

pub fn triangular_icdf(p: f64, a: f64, b: f64, c: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ [0, 1]"));
    }
    if !(a <= c && c <= b) || a == b {
        return Err(SciError::InvalidParameter("need a ≤ c ≤ b"));
    }
    let fc = (c - a) / (b - a);
    if p < fc {
        Ok(a + ((b - a) * (c - a) * p).sqrt())
    } else {
        Ok(b - ((b - a) * (b - c) * (1.0 - p)).sqrt())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Inverse Gaussian (Wald)
// ══════════════════════════════════════════════════════════════════════════════

/// Inverse Gaussian PDF: f(x) = sqrt(λ/(2πx³)) exp(−λ(x−μ)²/(2μ²x))  x > 0.
pub fn wald_pdf(x: f64, mu: f64, lambda: f64) -> SciResult<f64> {
    if mu <= 0.0 || lambda <= 0.0 {
        return Err(SciError::InvalidParameter("mu, lambda > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    let pi = core::f64::consts::PI;
    let coeff = (lambda / (2.0 * pi * x * x * x)).sqrt();
    let exponent = -lambda * (x - mu) * (x - mu) / (2.0 * mu * mu * x);
    Ok(coeff * exponent.exp())
}

pub fn wald_cdf(x: f64, mu: f64, lambda: f64) -> SciResult<f64> {
    if mu <= 0.0 || lambda <= 0.0 {
        return Err(SciError::InvalidParameter("mu, lambda > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    let sq = (lambda / x).sqrt();
    let t1 = sq * (x / mu - 1.0);
    let t2 = -sq * (x / mu + 1.0);
    // Φ(t1) + exp(2λ/μ)·Φ(t2)
    let p1 = normal_cdf_std(t1);
    let p2 = normal_cdf_std(t2);
    Ok(p1 + (2.0 * lambda / mu).exp() * p2)
}

// ══════════════════════════════════════════════════════════════════════════════
// Truncated Normal
// ══════════════════════════════════════════════════════════════════════════════

/// Truncated Normal PDF on [a, b].
pub fn truncnorm_pdf(x: f64, mu: f64, sigma: f64, a: f64, b: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    if x < a || x > b {
        return Ok(0.0);
    }
    let alpha = (a - mu) / sigma;
    let beta = (b - mu) / sigma;
    let big_phi_a = normal_cdf_std(alpha);
    let big_phi_b = normal_cdf_std(beta);
    let denom = big_phi_b - big_phi_a;
    if denom.abs() < f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    let z = (x - mu) / sigma;
    Ok(normal_pdf_std(z) / (sigma * denom))
}

pub fn truncnorm_cdf(x: f64, mu: f64, sigma: f64, a: f64, b: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma > 0"));
    }
    if x <= a {
        return Ok(0.0);
    }
    if x >= b {
        return Ok(1.0);
    }
    let alpha = (a - mu) / sigma;
    let beta_v = (b - mu) / sigma;
    let z = (x - mu) / sigma;
    let denom = normal_cdf_std(beta_v) - normal_cdf_std(alpha);
    if denom.abs() < f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    Ok((normal_cdf_std(z) - normal_cdf_std(alpha)) / denom)
}

pub fn truncnorm_icdf(p: f64, mu: f64, sigma: f64, a: f64, b: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p ∈ [0, 1]"));
    }
    let alpha = (a - mu) / sigma;
    let beta_v = (b - mu) / sigma;
    let phi_a = normal_cdf_std(alpha);
    let phi_b = normal_cdf_std(beta_v);
    let q = phi_a + p * (phi_b - phi_a);
    let z = normal_icdf(q)?;
    Ok(mu + sigma * z)
}

// ══════════════════════════════════════════════════════════════════════════════
// Von Mises (circular Normal)
// ══════════════════════════════════════════════════════════════════════════════

/// Von Mises PDF: f(θ) = exp(κ·cos(θ−μ)) / (2π·I₀(κ)),  θ ∈ [−π, π].
pub fn von_mises_pdf(theta: f64, mu: f64, kappa: f64) -> SciResult<f64> {
    if kappa < 0.0 {
        return Err(SciError::InvalidParameter("kappa >= 0"));
    }
    let pi = core::f64::consts::PI;
    Ok((kappa * (theta - mu).cos()).exp() / (2.0 * pi * bessel_i0(kappa)))
}

/// Von Mises log-normalisation constant: log(2π·I₀(κ)).
pub fn von_mises_log_norm(kappa: f64) -> f64 {
    let pi = core::f64::consts::PI;
    (2.0 * pi * bessel_i0(kappa)).ln()
}

// ══════════════════════════════════════════════════════════════════════════════
// Maxwell-Boltzmann
// ══════════════════════════════════════════════════════════════════════════════

/// Maxwell-Boltzmann PDF: f(x) = sqrt(2/π) x² exp(−x²/(2a²)) / a³,  x ≥ 0.
pub fn maxwell_pdf(x: f64, a: f64) -> SciResult<f64> {
    if a <= 0.0 {
        return Err(SciError::InvalidParameter("a > 0"));
    }
    if x < 0.0 {
        return Ok(0.0);
    }
    let pi = core::f64::consts::PI;
    Ok((2.0 / pi).sqrt() * x * x * (-x * x / (2.0 * a * a)).exp() / (a * a * a))
}

pub fn maxwell_cdf(x: f64, a: f64) -> SciResult<f64> {
    if a <= 0.0 {
        return Err(SciError::InvalidParameter("a > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    Ok(reg_inc_gamma(1.5, x * x / (2.0 * a * a)))
}

// ══════════════════════════════════════════════════════════════════════════════
// Chi distribution (not chi-squared)
// ══════════════════════════════════════════════════════════════════════════════

/// Chi PDF: f(x) = 2^{1−k/2} x^{k−1} e^{−x²/2} / Γ(k/2),  x ≥ 0.
pub fn chi_pdf(x: f64, k: f64) -> SciResult<f64> {
    if k <= 0.0 {
        return Err(SciError::InvalidParameter("k > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    let log_p =
        (1.0 - k / 2.0) * 2.0_f64.ln() + (k - 1.0) * x.ln() - x * x / 2.0 - ln_gamma(k / 2.0);
    Ok(log_p.exp())
}

pub fn chi_cdf(x: f64, k: f64) -> SciResult<f64> {
    if k <= 0.0 {
        return Err(SciError::InvalidParameter("k > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    Ok(reg_inc_gamma(k / 2.0, x * x / 2.0))
}

// ══════════════════════════════════════════════════════════════════════════════
// Multinomial PMF
// ══════════════════════════════════════════════════════════════════════════════

/// Multinomial PMF: P(X₁=k₁,…,Xₘ=kₘ) = n! / (k₁!…kₘ!) · p₁^k₁…pₘ^kₘ.
pub fn multinomial_pmf(n: u64, counts: &[u64], probs: &[f64]) -> SciResult<f64> {
    let m = counts.len();
    if probs.len() != m {
        return Err(SciError::InvalidParameter("counts and probs must match"));
    }
    if counts.iter().sum::<u64>() != n {
        return Err(SciError::InvalidParameter("counts must sum to n"));
    }
    let log_coeff = ln_gamma((n + 1) as f64)
        - counts
            .iter()
            .map(|&k| ln_gamma((k + 1) as f64))
            .sum::<f64>();
    let log_probs: f64 = counts
        .iter()
        .zip(probs)
        .map(|(&k, &p)| {
            if k == 0 {
                0.0
            } else if p <= 0.0 {
                f64::NEG_INFINITY
            } else {
                k as f64 * p.ln()
            }
        })
        .sum();
    Ok((log_coeff + log_probs).exp())
}

// ══════════════════════════════════════════════════════════════════════════════
// Beta-Binomial PMF
// ══════════════════════════════════════════════════════════════════════════════

/// Beta-Binomial PMF: marginalises Binomial over Beta(α, β) prior.
///
/// P(X=k | n, α, β) = C(n,k) B(k+α, n−k+β) / B(α, β).
pub fn beta_binomial_pmf(k: u64, n: u64, alpha: f64, beta: f64) -> SciResult<f64> {
    if k > n {
        return Ok(0.0);
    }
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter("alpha, beta > 0"));
    }
    let log_binom =
        ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64);
    let log_num = ln_gamma(k as f64 + alpha) + ln_gamma((n - k) as f64 + beta)
        - ln_gamma(n as f64 + alpha + beta);
    let log_den = ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta);
    Ok((log_binom + log_num - log_den).exp())
}

// ══════════════════════════════════════════════════════════════════════════════
// Private helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Standard normal PDF φ(x).
fn normal_pdf_std(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    (-0.5 * x * x).exp() / (2.0 * pi).sqrt()
}

/// Standard normal CDF Φ(x) via error function.
fn normal_cdf_std(x: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / core::f64::consts::SQRT_2))
}

/// Error function approximation (Horner, max error ~1.5e-7).
fn libm_erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

/// Modified Bessel function I₀(x).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let x2 = (x / 2.0) * (x / 2.0);
    for k in 1..30 {
        term *= x2 / (k * k) as f64;
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}
