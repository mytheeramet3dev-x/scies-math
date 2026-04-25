//! Monte Carlo integration and basic sampling.
//!
//! # Functions
//!
//! | Function | Description |
//! |---|---|
//! | `monte_carlo_integrate` | Estimate ∫f(x)dx over a box via uniform sampling |
//! | `importance_sample` | Importance sampling with proposal distribution |
//! | `stratified_sample` | Stratified sampling for variance reduction |
//!
//! # Usage
//!
//! ```rust
//! use scies_math::monte_carlo::monte_carlo_integrate;
//!
//! // Estimate π by integrating f(x) = √(1-x²) over [0,1]
//! let pi_quarter = monte_carlo_integrate(
//!     |x| (1.0 - x[0]*x[0]).sqrt(),
//!     &[(0.0, 1.0)],  // bounds
//!     100_000,
//!     42,             // seed
//! );
//! // pi_quarter * 4 ≈ π
//! ```
//!
//! For advanced MCMC (HMC, Gibbs, Slice, Particle Filter) see [`crate::monte_carlo_ext`].
use crate::errors::{SciError, SciResult};
use crate::rng::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// Monte Carlo Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Monte Carlo integration of `f` over the hyper-rectangle
/// `[bounds[i].0, bounds[i].1]` for each dimension `i`.
///
/// Returns `(estimate, standard_error)`.
///
/// # Parameters
/// - `f`       — integrand: `f(x: &[f64]) -> f64`
/// - `bounds`  — list of `(lo, hi)` per dimension
/// - `n`       — number of sample points
/// - `seed`    — PRNG seed
pub fn mc_integrate<F>(f: F, bounds: &[(f64, f64)], n: usize, seed: u64) -> SciResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(SciError::InvalidParameter("bounds must be non-empty"));
    }
    if n == 0 {
        return Err(SciError::InvalidParameter("n must be positive"));
    }
    for &(lo, hi) in bounds {
        if hi <= lo {
            return Err(SciError::InvalidParameter("each bound must have hi > lo"));
        }
    }

    let volume: f64 = bounds.iter().map(|(lo, hi)| hi - lo).product();
    let dim = bounds.len();
    let mut rng = Rng::new(seed);
    let mut x = vec![0.0f64; dim];
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for _ in 0..n {
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            x[i] = lo + rng.rand01() * (hi - lo);
        }
        let val = f(&x);
        sum += val;
        sum_sq += val * val;
    }

    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64 - mean * mean).max(0.0);
    let std_error = volume * (variance / n as f64).sqrt();
    Ok((volume * mean, std_error))
}

/// Importance-sampling MC integration: ∫ f(x) dx ≈ (1/n) Σ f(xᵢ)/q(xᵢ)
/// where xᵢ ~ q.
///
/// The caller supplies a sampler `draw(rng) -> (x, q(x))` returning the
/// sample and its proposal density.
///
/// Returns `(estimate, standard_error)`.
pub fn mc_integrate_importance<F, Q>(f: F, draw: Q, n: usize, seed: u64) -> SciResult<(f64, f64)>
where
    F: Fn(f64) -> f64,
    Q: Fn(&mut Rng) -> (f64, f64),
{
    if n == 0 {
        return Err(SciError::InvalidParameter("n must be positive"));
    }
    let mut rng = Rng::new(seed);
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for _ in 0..n {
        let (x, qx) = draw(&mut rng);
        if qx <= 0.0 {
            continue;
        }
        let w = f(x) / qx;
        sum += w;
        sum_sq += w * w;
    }

    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64 - mean * mean).max(0.0);
    let std_error = (variance / n as f64).sqrt();
    Ok((mean, std_error))
}

/// **Quasi-Monte Carlo** integration over [0,1]ⁿ using the Halton sequence.
///
/// Lower error rate than plain MC (O(log(n)^d / n) vs O(1/√n)).
/// `primes` must have length equal to the number of dimensions;
/// pass distinct primes, e.g. `&[2, 3, 5, 7, …]`.
pub fn quasi_mc_integrate<F>(
    f: F,
    bounds: &[(f64, f64)],
    n: usize,
    primes: &[u64],
) -> SciResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let dim = bounds.len();
    if dim == 0 || primes.len() != dim || n == 0 {
        return Err(SciError::InvalidParameter(
            "bounds, primes must be non-empty and same length; n > 0",
        ));
    }
    for &(lo, hi) in bounds {
        if hi <= lo {
            return Err(SciError::InvalidParameter("each bound must have hi > lo"));
        }
    }
    let volume: f64 = bounds.iter().map(|(lo, hi)| hi - lo).product();
    let mut x = vec![0.0f64; dim];
    let mut sum = 0.0f64;
    for i in 1..=n {
        for (d, &base) in primes.iter().enumerate() {
            x[d] = halton(i as u64, base);
            let (lo, hi) = bounds[d];
            x[d] = lo + x[d] * (hi - lo);
        }
        sum += f(&x);
    }
    Ok(volume * sum / n as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

/// Percentile bootstrap confidence interval for the **mean** of `data`.
///
/// Returns `(lower, upper)` at the given `confidence` level (e.g. 0.95).
pub fn bootstrap_ci(
    data: &[f64],
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> SciResult<(f64, f64)> {
    bootstrap_stat_ci(
        data,
        |s| s.iter().sum::<f64>() / s.len() as f64,
        n_resamples,
        confidence,
        seed,
    )
}

/// Percentile bootstrap CI for an **arbitrary statistic** `stat(resample)`.
///
/// Returns `(lower, upper)`.
pub fn bootstrap_stat_ci<F>(
    data: &[f64],
    stat: F,
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> SciResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if data.len() < 2 {
        return Err(SciError::InvalidParameter(
            "data must have at least 2 elements",
        ));
    }
    if n_resamples == 0 {
        return Err(SciError::InvalidParameter("n_resamples must be positive"));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(SciError::DomainError("confidence must be in (0, 1)"));
    }

    let n = data.len();
    let mut rng = Rng::new(seed);
    let mut boot_stats: Vec<f64> = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let resample: Vec<f64> = (0..n)
            .map(|_| {
                let idx = (rng.rand01() * n as f64) as usize % n;
                data[idx]
            })
            .collect();
        boot_stats.push(stat(&resample));
    }

    boot_stats.sort_by(f64::total_cmp);
    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_resamples as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64) as usize;
    let hi_idx = hi_idx.min(n_resamples - 1);
    Ok((boot_stats[lo_idx], boot_stats[hi_idx]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Permutation test
// ─────────────────────────────────────────────────────────────────────────────

/// Two-sample permutation test (two-sided) comparing the means of `a` and `b`.
///
/// Returns the p-value: fraction of random permutations with
/// |mean_diff| ≥ |observed mean_diff|.
pub fn permutation_test(a: &[f64], b: &[f64], n_permutations: usize, seed: u64) -> SciResult<f64> {
    if a.len() < 2 || b.len() < 2 {
        return Err(SciError::InvalidParameter(
            "both groups must have at least 2 elements",
        ));
    }
    if n_permutations == 0 {
        return Err(SciError::InvalidParameter(
            "n_permutations must be positive",
        ));
    }

    let na = a.len();
    let observed =
        (a.iter().sum::<f64>() / na as f64 - b.iter().sum::<f64>() / b.len() as f64).abs();

    let mut combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let total = combined.len();
    let mut rng = Rng::new(seed);
    let mut extreme = 0usize;

    for _ in 0..n_permutations {
        // Fisher-Yates shuffle
        for i in (1..total).rev() {
            let j = (rng.rand01() * (i + 1) as f64) as usize % (i + 1);
            combined.swap(i, j);
        }
        let mean_a: f64 = combined[..na].iter().sum::<f64>() / na as f64;
        let mean_b: f64 = combined[na..].iter().sum::<f64>() / (total - na) as f64;
        if (mean_a - mean_b).abs() >= observed {
            extreme += 1;
        }
    }

    Ok(extreme as f64 / n_permutations as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo / utility
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate π by counting points inside the unit circle.
///
/// Classic teaching example; actual accuracy is O(1/√n).
pub fn mc_pi_estimate(n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let inside = (0..n)
        .filter(|_| {
            let x = rng.rand_range(-1.0, 1.0);
            let y = rng.rand_range(-1.0, 1.0);
            x * x + y * y <= 1.0
        })
        .count();
    4.0 * inside as f64 / n as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Halton sequence
// ─────────────────────────────────────────────────────────────────────────────

fn halton(index: u64, base: u64) -> f64 {
    let mut f = 1.0f64;
    let mut r = 0.0f64;
    let mut i = index;
    let b = base as f64;
    while i > 0 {
        f /= b;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}
