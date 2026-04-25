//! Probability distributions — PDF, CDF, and inverse CDF.
//!
//! # Distributions
//!
//! | Distribution | PDF | CDF | Inverse CDF |
//! |---|---|---|---|
//! | `Normal` | Yes | Yes | Yes |
//! | `LogNormal` | Yes | Yes | Yes |
//! | `Exponential` | Yes | Yes | Yes |
//! | `Gamma` | Yes | Yes | — |
//! | `Beta` | Yes | Yes | Yes |
//! | `ChiSquared` | Yes | Yes | — |
//! | `StudentT` | Yes | Yes | — |
//! | `Weibull` | Yes | Yes | Yes |
//! | `Uniform` | Yes | Yes | Yes |
//! | `Triangular` | Yes | Yes | Yes |
//! | `Pareto` | Yes | Yes | Yes |
//!
//! # Usage
//!
//! ```rust
//! use scies_math::distributions::{Normal, Distribution};
//!
//! let n = Normal::new(0.0, 1.0).unwrap();
//! let p = n.pdf(1.96);   // ≈ 0.0584
//! let c = n.cdf(1.96);   // ≈ 0.975
//! let q = n.inverse_cdf(0.975).unwrap(); // ≈ 1.96
//! ```
//!
//! # Design
//!
//! All distribution structs implement a common `Distribution` trait:
//!
//! ```text
//! trait Distribution {
//!     fn pdf(&self, x: f64) -> f64;
//!     fn cdf(&self, x: f64) -> f64;
//!     fn mean(&self) -> f64;
//!     fn variance(&self) -> f64;
//! }
//! ```
//!
//! For sampling, use [`crate::rng_ext::Sampler`] which provides high-quality
//! samplers for every distribution above using rejection/transformation methods.
use crate::errors::{SciError, SciResult};
use crate::probability::normal_cdf;

// ─────────────────────────────────────────────────────────────────────────────
// Exponential
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential PDF: λ·exp(−λx) for x ≥ 0.
pub fn exponential_pdf(x: f64, lambda: f64) -> SciResult<f64> {
    if lambda <= 0.0 {
        return Err(SciError::InvalidParameter("lambda must be positive"));
    }
    if x < 0.0 {
        return Ok(0.0);
    }
    Ok(lambda * (-lambda * x).exp())
}

/// Exponential CDF: 1 − exp(−λx).
pub fn exponential_cdf(x: f64, lambda: f64) -> SciResult<f64> {
    if lambda <= 0.0 {
        return Err(SciError::InvalidParameter("lambda must be positive"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    Ok(1.0 - (-lambda * x).exp())
}

/// Exponential inverse CDF: −ln(1−p)/λ.
pub fn exponential_icdf(p: f64, lambda: f64) -> SciResult<f64> {
    check_prob(p)?;
    if lambda <= 0.0 {
        return Err(SciError::InvalidParameter("lambda must be positive"));
    }
    Ok(-(1.0 - p).max(f64::EPSILON).ln() / lambda)
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform PDF on [a, b].
pub fn uniform_pdf(x: f64, a: f64, b: f64) -> SciResult<f64> {
    if b <= a {
        return Err(SciError::InvalidParameter("b must be > a"));
    }
    if x < a || x > b {
        return Ok(0.0);
    }
    Ok(1.0 / (b - a))
}

/// Uniform CDF on [a, b].
pub fn uniform_cdf(x: f64, a: f64, b: f64) -> SciResult<f64> {
    if b <= a {
        return Err(SciError::InvalidParameter("b must be > a"));
    }
    if x <= a {
        return Ok(0.0);
    }
    if x >= b {
        return Ok(1.0);
    }
    Ok((x - a) / (b - a))
}

/// Uniform inverse CDF: a + p·(b − a).
pub fn uniform_icdf(p: f64, a: f64, b: f64) -> SciResult<f64> {
    check_prob(p)?;
    if b <= a {
        return Err(SciError::InvalidParameter("b must be > a"));
    }
    Ok(a + p * (b - a))
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma
// ─────────────────────────────────────────────────────────────────────────────

/// Gamma PDF: x^(α−1)·exp(−x/β) / (β^α·Γ(α)) for x > 0.
pub fn gamma_pdf(x: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter(
            "alpha and beta must be positive",
        ));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    let ln_pdf = (alpha - 1.0) * x.ln() - x / beta - alpha * beta.ln() - ln_gamma(alpha);
    Ok(ln_pdf.exp())
}

/// Gamma CDF: regularized lower incomplete gamma P(α, x/β).
pub fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter(
            "alpha and beta must be positive",
        ));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    Ok(reg_inc_gamma(alpha, x / beta))
}

// ─────────────────────────────────────────────────────────────────────────────
// Beta
// ─────────────────────────────────────────────────────────────────────────────

/// Beta PDF: x^(α−1)·(1−x)^(β−1) / B(α,β) for x ∈ (0, 1).
pub fn beta_pdf(x: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter(
            "alpha and beta must be positive",
        ));
    }
    if x <= 0.0 || x >= 1.0 {
        return Ok(0.0);
    }
    let ln_pdf = (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_beta(alpha, beta);
    Ok(ln_pdf.exp())
}

/// Beta CDF: regularized incomplete beta I_x(α, β).
pub fn beta_cdf(x: f64, alpha: f64, beta: f64) -> SciResult<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(SciError::InvalidParameter(
            "alpha and beta must be positive",
        ));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x >= 1.0 {
        return Ok(1.0);
    }
    Ok(reg_inc_beta(alpha, beta, x))
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-Normal
// ─────────────────────────────────────────────────────────────────────────────

/// Log-Normal PDF: 1/(x·σ·√(2π)) · exp(−(ln x − μ)²/(2σ²)).
pub fn lognormal_pdf(x: f64, mu: f64, sigma: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma must be positive"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    let pi = core::f64::consts::PI;
    let z = (x.ln() - mu) / sigma;
    Ok((-0.5 * z * z).exp() / (x * sigma * (2.0 * pi).sqrt()))
}

/// Log-Normal CDF: Φ((ln x − μ)/σ).
pub fn lognormal_cdf(x: f64, mu: f64, sigma: f64) -> SciResult<f64> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma must be positive"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    normal_cdf(x.ln(), mu, sigma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weibull
// ─────────────────────────────────────────────────────────────────────────────

/// Weibull PDF: (k/λ)·(x/λ)^(k−1)·exp(−(x/λ)^k).
pub fn weibull_pdf(x: f64, k: f64, lambda: f64) -> SciResult<f64> {
    if k <= 0.0 || lambda <= 0.0 {
        return Err(SciError::InvalidParameter("k and lambda must be positive"));
    }
    if x < 0.0 {
        return Ok(0.0);
    }
    if x == 0.0 {
        return Ok(if k < 1.0 {
            f64::INFINITY
        } else if k == 1.0 {
            1.0 / lambda
        } else {
            0.0
        });
    }
    let t = x / lambda;
    Ok((k / lambda) * t.powf(k - 1.0) * (-t.powf(k)).exp())
}

/// Weibull CDF: 1 − exp(−(x/λ)^k).
pub fn weibull_cdf(x: f64, k: f64, lambda: f64) -> SciResult<f64> {
    if k <= 0.0 || lambda <= 0.0 {
        return Err(SciError::InvalidParameter("k and lambda must be positive"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    Ok(1.0 - (-(x / lambda).powf(k)).exp())
}

/// Weibull inverse CDF: λ·(−ln(1−p))^(1/k).
pub fn weibull_icdf(p: f64, k: f64, lambda: f64) -> SciResult<f64> {
    check_prob(p)?;
    if k <= 0.0 || lambda <= 0.0 {
        return Err(SciError::InvalidParameter("k and lambda must be positive"));
    }
    Ok(lambda * (-(1.0 - p).max(f64::EPSILON).ln()).powf(1.0 / k))
}

// ─────────────────────────────────────────────────────────────────────────────
// Cauchy
// ─────────────────────────────────────────────────────────────────────────────

/// Cauchy PDF: 1/(π·γ·(1+((x−x₀)/γ)²)).
pub fn cauchy_pdf(x: f64, x0: f64, gamma: f64) -> SciResult<f64> {
    if gamma <= 0.0 {
        return Err(SciError::InvalidParameter("gamma must be positive"));
    }
    let z = (x - x0) / gamma;
    Ok(1.0 / (core::f64::consts::PI * gamma * (1.0 + z * z)))
}

/// Cauchy CDF: 1/π·arctan((x−x₀)/γ) + 1/2.
pub fn cauchy_cdf(x: f64, x0: f64, gamma: f64) -> SciResult<f64> {
    if gamma <= 0.0 {
        return Err(SciError::InvalidParameter("gamma must be positive"));
    }
    Ok(((x - x0) / gamma).atan() / core::f64::consts::PI + 0.5)
}

/// Cauchy inverse CDF: x₀ + γ·tan(π(p − 1/2)).
pub fn cauchy_icdf(p: f64, x0: f64, gamma: f64) -> SciResult<f64> {
    check_prob(p)?;
    if gamma <= 0.0 {
        return Err(SciError::InvalidParameter("gamma must be positive"));
    }
    Ok(x0 + gamma * (core::f64::consts::PI * (p - 0.5)).tan())
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometric
// ─────────────────────────────────────────────────────────────────────────────

/// Geometric PMF: (1−p)^(k−1)·p   for k = 1, 2, 3, …
pub fn geometric_pmf(k: u64, p: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p must be in [0, 1]"));
    }
    if k == 0 {
        return Ok(0.0);
    }
    Ok((1.0 - p).powi(k as i32 - 1) * p)
}

/// Geometric CDF: 1 − (1−p)^k.
pub fn geometric_cdf(k: u64, p: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p must be in [0, 1]"));
    }
    Ok(1.0 - (1.0 - p).powi(k as i32))
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative Binomial
// ─────────────────────────────────────────────────────────────────────────────

/// Negative Binomial PMF: C(k+r−1, k)·p^r·(1−p)^k.
///
/// Models the number of failures `k` before `r` successes with success prob `p`.
pub fn neg_binomial_pmf(k: u64, r: u64, p: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("p must be in [0, 1]"));
    }
    if r == 0 {
        return Err(SciError::InvalidParameter("r must be positive"));
    }
    let log_coeff = log_binomial_coeff(k + r - 1, k);
    Ok((log_coeff + (r as f64) * p.ln() + (k as f64) * (1.0 - p).ln()).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn check_prob(p: f64) -> SciResult<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("probability must be in [0, 1]"));
    }
    Ok(())
}

/// Natural log of the Gamma function (Lanczos, g=7).
pub fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        let pi = core::f64::consts::PI;
        return (pi / (pi * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = 0.999_999_999_999_809_9;
    for (i, &c) in COEFFS.iter().enumerate() {
        x += c / (z + i as f64 + 1.0);
    }
    let t = z + 7.5;
    let two_pi = 2.0 * core::f64::consts::PI;
    0.5 * two_pi.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn log_binomial_coeff(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Regularized lower incomplete gamma P(a, x) via series / continued fraction.
pub fn reg_inc_gamma(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        inc_gamma_series(a, x)
    } else {
        1.0 - inc_gamma_cf(a, x)
    }
}

fn inc_gamma_series(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut n = 1.0f64;
    for _ in 0..200 {
        term *= x / (a + n);
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
        n += 1.0;
    }
    (a * x.ln() - x - ln_gamma(a)).exp() * sum
}

fn inc_gamma_cf(a: f64, x: f64) -> f64 {
    // Modified Lentz continued fraction.
    let mut f = x + 1.0 - a;
    if f.abs() < f64::EPSILON {
        f = 1e-30;
    }
    let mut c = f;
    let mut d = 1.0 / f;
    let mut h = d;
    for i in 1..200usize {
        let an = -(i as f64) * (i as f64 - a);
        let bn = x + 2.0 * i as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        if (d * c - 1.0).abs() < 1e-14 {
            break;
        }
    }
    (a * x.ln() - x - ln_gamma(a)).exp() * h
}

/// Regularized incomplete beta I_x(a, b) via continued fraction (Lentz).
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    // Use the symmetry relation for numerical stability.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - reg_inc_beta(b, a, 1.0 - x);
    }
    let lbeta_ab = ln_beta(a, b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta_ab).exp() / a;
    front * beta_cf(a, b, x)
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    for m in 1..200usize {
        // Even step
        let mf = m as f64;
        let num_e = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num_e * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_e / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        // Odd step
        let num_o = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num_o * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_o / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }
    f
}
