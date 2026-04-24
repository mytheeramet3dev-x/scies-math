//! Xoshiro256** pseudo-random number generator + distribution samplers.
//!
//! # Usage
//! ```rust
//! use sciesrust::math::rng::Rng;
//!
//! let mut rng = Rng::new(42);
//! let u  = rng.rand01();          // Uniform [0, 1)
//! let z  = rng.randn();           // Standard Normal
//! let g  = rng.sample_gamma(2.0, 1.0); // Gamma(α=2, β=1)
//! ```

use core::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Xoshiro256** generator
// ─────────────────────────────────────────────────────────────────────────────

/// Xoshiro256** pseudo-random number generator.
///
/// Passes all known statistical tests, period = 2²⁵⁶ − 1.
/// Seeded via SplitMix64 so any `u64` seed (including 0) is valid.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next_sm = || {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next_sm(), next_sm(), next_sm(), next_sm()],
        }
    }

    /// Raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in `[0, 1)`.
    pub fn rand01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform float in `[lo, hi)`.
    pub fn rand_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.rand01() * (hi - lo)
    }

    // ── Standard distributions ────────────────────────────────────────────────

    /// Standard Normal via Box-Muller transform.
    pub fn randn(&mut self) -> f64 {
        let u1 = self.rand01().max(f64::EPSILON);
        let u2 = self.rand01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Normal N(mean, std_dev).
    pub fn sample_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        mean + std_dev * self.randn()
    }

    /// Exponential Exp(λ) via inverse CDF: −ln(U)/λ.
    pub fn sample_exponential(&mut self, lambda: f64) -> f64 {
        -self.rand01().max(f64::EPSILON).ln() / lambda
    }

    /// Gamma(α, β) using Marsaglia-Tsang "squeeze" method.
    ///
    /// - α (shape) > 0, β (scale) > 0.
    /// - For α < 1, uses the relation: Gamma(α) = Gamma(α+1) · U^(1/α).
    pub fn sample_gamma(&mut self, alpha: f64, beta: f64) -> f64 {
        beta * self.gamma_standard(alpha)
    }

    /// Beta(α, β) via ratio of two Gammas.
    pub fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.gamma_standard(alpha);
        let y = self.gamma_standard(beta);
        x / (x + y)
    }

    /// Log-Normal LN(μ, σ): exp(N(μ, σ)).
    pub fn sample_lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        self.sample_normal(mu, sigma).exp()
    }

    /// Weibull(k, λ) via inverse CDF: λ·(−ln U)^(1/k).
    pub fn sample_weibull(&mut self, k: f64, lambda: f64) -> f64 {
        lambda * (-self.rand01().max(f64::EPSILON).ln()).powf(1.0 / k)
    }

    /// Chi-squared χ²(df) = Gamma(df/2, 2).
    pub fn sample_chi_squared(&mut self, df: f64) -> f64 {
        self.sample_gamma(df / 2.0, 2.0)
    }

    /// Student-t via Normal / sqrt(Chi²/df).
    pub fn sample_student_t(&mut self, df: f64) -> f64 {
        self.randn() / (self.sample_chi_squared(df) / df).sqrt()
    }

    /// Binomial B(n, p) via sum of Bernoullis.
    pub fn sample_binomial(&mut self, n: u64, p: f64) -> u64 {
        (0..n).filter(|_| self.rand01() < p).count() as u64
    }

    /// Poisson(λ) via Knuth's algorithm (valid for moderate λ).
    pub fn sample_poisson(&mut self, lambda: f64) -> u64 {
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut prod = 1.0f64;
        loop {
            prod *= self.rand01();
            if prod <= l {
                break;
            }
            k += 1;
        }
        k
    }

    /// Draw `n` samples from any closure `sampler(rng)`.
    pub fn sample_n<F>(&mut self, mut sampler: F, n: usize) -> Vec<f64>
    where
        F: FnMut(&mut Self) -> f64,
    {
        (0..n).map(|_| sampler(self)).collect()
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Standard Gamma(α, 1) — Marsaglia-Tsang.
    fn gamma_standard(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            // Boost: Gamma(α) = Gamma(α+1) · U^(1/α)
            let u = self.rand01().max(f64::EPSILON);
            return self.gamma_standard(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.randn();
            let v_raw = 1.0 + c * x;
            if v_raw <= 0.0 {
                continue;
            }
            let v = v_raw * v_raw * v_raw;
            let u = self.rand01().max(f64::EPSILON);
            // Squeeze test (avoids ln most of the time).
            if u < 1.0 - 0.331 * x * x * x * x {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}
