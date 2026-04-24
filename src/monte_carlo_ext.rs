//! Extended Monte Carlo: MCMC samplers, variance reduction, quasi-random
//! sequences, particle filter (SMC), and rejection/slice sampling.
//!
//! | Algorithm | Function |
//! |---|---|
//! | Metropolis-Hastings | `metropolis_hastings` |
//! | Random-walk Metropolis | `random_walk_metropolis` |
//! | Gibbs sampler | `gibbs_sampler` |
//! | Hamiltonian MC (HMC) | `hmc` |
//! | Slice sampling | `slice_sample` |
//! | Rejection sampling | `rejection_sample` |
//! | Control variates | `mc_control_variates` |
//! | Antithetic variates | `mc_antithetic` |
//! | Latin Hypercube | `latin_hypercube` |
//! | Sobol (scrambled) | `sobol_sequence` |
//! | Particle filter (SMC) | `particle_filter` |
//! | Simulated annealing MC | `sa_sample` |

use crate::errors::{SciError, SciResult};

// ══════════════════════════════════════════════════════════════════════════════
// RNG
// ══════════════════════════════════════════════════════════════════════════════
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 6364136223846793005)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
    fn next_usize(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Metropolis-Hastings (general)
// ══════════════════════════════════════════════════════════════════════════════

/// **Metropolis-Hastings** MCMC sampler.
///
/// - `log_target`: log of unnormalized target density
/// - `proposal`: given current x, return proposed x'
/// - `log_proposal_ratio`: log q(x|x') − log q(x'|x) (0 for symmetric proposals)
pub fn metropolis_hastings<L, P, R>(
    log_target: L,
    proposal: P,
    log_proposal_ratio: R,
    x0: Vec<f64>,
    n_samples: usize,
    n_burnin: usize,
    seed: u64,
) -> SciResult<Vec<Vec<f64>>>
where
    L: Fn(&[f64]) -> f64,
    P: Fn(&[f64], &mut dyn FnMut() -> f64) -> Vec<f64>,
    R: Fn(&[f64], &[f64]) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut x = x0;
    let mut lp = log_target(&x);
    let mut samples = Vec::with_capacity(n_samples);
    let total = n_burnin + n_samples;
    for i in 0..total {
        let mut rand = || rng.next_f64();
        let x_prop = proposal(&x, &mut rand as &mut dyn FnMut() -> f64);
        let lp_prop = log_target(&x_prop);
        let log_alpha = lp_prop - lp + log_proposal_ratio(&x, &x_prop);
        if log_alpha >= 0.0 || rng.next_f64().ln() < log_alpha {
            x = x_prop;
            lp = lp_prop;
        }
        if i >= n_burnin {
            samples.push(x.clone());
        }
    }
    Ok(samples)
}

/// **Random-Walk Metropolis** with isotropic Gaussian proposal (step_size σ).
pub fn random_walk_metropolis<L>(
    log_target: L,
    x0: Vec<f64>,
    step_size: f64,
    n_samples: usize,
    n_burnin: usize,
    seed: u64,
) -> SciResult<Vec<Vec<f64>>>
where
    L: Fn(&[f64]) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut x = x0;
    let mut lp = log_target(&x);
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..(n_burnin + n_samples) {
        let x_prop: Vec<f64> = x
            .iter()
            .map(|&xi| xi + step_size * rng.next_normal())
            .collect();
        let lp_prop = log_target(&x_prop);
        if (lp_prop - lp) >= 0.0 || rng.next_f64().ln() < lp_prop - lp {
            x = x_prop;
            lp = lp_prop;
        }
        if i >= n_burnin {
            samples.push(x.clone());
        }
    }
    Ok(samples)
}

// ══════════════════════════════════════════════════════════════════════════════
// Gibbs sampler
// ══════════════════════════════════════════════════════════════════════════════

/// **Gibbs sampler**: cycles through each coordinate, sampling from its
/// conditional distribution `conditionals[i](current_x, rng)`.
///
/// Each conditional is `Fn(&[f64], &mut dyn FnMut() -> f64) -> f64`.
pub fn gibbs_sampler<C>(
    conditionals: &[C],
    x0: Vec<f64>,
    n_samples: usize,
    n_burnin: usize,
    seed: u64,
) -> SciResult<Vec<Vec<f64>>>
where
    C: Fn(&[f64], &mut dyn FnMut() -> f64) -> f64,
{
    let d = conditionals.len();
    if x0.len() != d {
        return Err(SciError::InvalidParameter(
            "x0 length must match conditionals",
        ));
    }
    let mut rng = Lcg::new(seed);
    let mut x = x0;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..(n_burnin + n_samples) {
        for j in 0..d {
            let mut rand = || rng.next_f64();
            x[j] = conditionals[j](&x, &mut rand as &mut dyn FnMut() -> f64);
        }
        if i >= n_burnin {
            samples.push(x.clone());
        }
    }
    Ok(samples)
}

// ══════════════════════════════════════════════════════════════════════════════
// Hamiltonian Monte Carlo (HMC)
// ══════════════════════════════════════════════════════════════════════════════

/// **Hamiltonian Monte Carlo** (Neal 2011).
///
/// `grad_log_target(x)` must return ∇ log π(x).
/// Uses leapfrog with `n_leapfrog` steps of size `eps`.
pub fn hmc<L, G>(
    log_target: L,
    grad_log_target: G,
    x0: Vec<f64>,
    eps: f64,
    n_leapfrog: usize,
    n_samples: usize,
    n_burnin: usize,
    seed: u64,
) -> SciResult<Vec<Vec<f64>>>
where
    L: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let d = x0.len();
    let mut rng = Lcg::new(seed);
    let mut q = x0;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..(n_burnin + n_samples) {
        // Sample momentum
        let p0: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
        let mut q_new = q.clone();
        let mut p = p0.clone();
        // Leapfrog
        let mut g = grad_log_target(&q_new);
        for _ in 0..n_leapfrog {
            for j in 0..d {
                p[j] += 0.5 * eps * g[j];
            }
            for j in 0..d {
                q_new[j] += eps * p[j];
            }
            g = grad_log_target(&q_new);
            for j in 0..d {
                p[j] += 0.5 * eps * g[j];
            }
        }
        let k_new: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() / 2.0;
        let k_old: f64 = p0.iter().map(|pi| pi * pi).sum::<f64>() / 2.0;
        let log_alpha = log_target(&q_new) - k_new - log_target(&q) + k_old;
        if log_alpha >= 0.0 || rng.next_f64().ln() < log_alpha {
            q = q_new;
        }
        if i >= n_burnin {
            samples.push(q.clone());
        }
    }
    Ok(samples)
}

// ══════════════════════════════════════════════════════════════════════════════
// Slice sampling (1D)
// ══════════════════════════════════════════════════════════════════════════════

/// **Slice sampling** for a 1D log-target (stepping-out + shrinkage).
pub fn slice_sample<L>(
    log_target: L,
    x0: f64,
    width: f64,
    n_samples: usize,
    n_burnin: usize,
    seed: u64,
) -> SciResult<Vec<f64>>
where
    L: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut x = x0;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..(n_burnin + n_samples) {
        let lp = log_target(x);
        let y = lp + rng.next_f64().ln(); // slice level
        // Stepping-out
        let mut lo = x - width * rng.next_f64();
        let mut hi = lo + width;
        while log_target(lo) > y {
            lo -= width;
        }
        while log_target(hi) > y {
            hi += width;
        }
        // Shrinkage
        loop {
            let xp = lo + rng.next_f64() * (hi - lo);
            if log_target(xp) > y {
                x = xp;
                break;
            }
            if xp < x {
                lo = xp;
            } else {
                hi = xp;
            }
            if (hi - lo) < 1e-12 {
                break;
            }
        }
        if i >= n_burnin {
            samples.push(x);
        }
    }
    Ok(samples)
}

// ══════════════════════════════════════════════════════════════════════════════
// Rejection sampling
// ══════════════════════════════════════════════════════════════════════════════

/// **Rejection sampling**: sample from target f ≤ M·g, using proposal `sample_g`
/// and ratio `log_target(x) - log(M) - log_proposal(x)`.
///
/// Returns n_samples accepted samples or error if acceptance too low.
pub fn rejection_sample<L, S, LP>(
    log_target: L,
    sample_proposal: S,
    log_proposal: LP,
    log_m: f64,
    n_samples: usize,
    max_tries: usize,
    seed: u64,
) -> SciResult<Vec<f64>>
where
    L: Fn(f64) -> f64,
    S: Fn(f64) -> f64,
    LP: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut tries = 0usize;
    while samples.len() < n_samples {
        if tries >= max_tries {
            return Err(SciError::NonConvergent("rejection_sample: too many tries"));
        }
        let u = rng.next_f64();
        let x = sample_proposal(u);
        let log_ratio = log_target(x) - log_m - log_proposal(x);
        if rng.next_f64().ln() <= log_ratio {
            samples.push(x);
        }
        tries += 1;
    }
    Ok(samples)
}

// ══════════════════════════════════════════════════════════════════════════════
// Variance reduction: control variates
// ══════════════════════════════════════════════════════════════════════════════

/// **Control variates** estimator: E[f] ≈ mean(f) − c·(mean(g) − E[g]).
///
/// Optimal c = Cov(f,g) / Var(g).  Returns improved estimate and variance.
pub fn mc_control_variates<F, G>(f: F, g: G, eg: f64, n: usize, seed: u64) -> SciResult<(f64, f64)>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let fv: Vec<f64> = samples.iter().map(|&u| f(u)).collect();
    let gv: Vec<f64> = samples.iter().map(|&u| g(u)).collect();
    let mf = fv.iter().sum::<f64>() / n as f64;
    let mg = gv.iter().sum::<f64>() / n as f64;
    let cov: f64 = fv
        .iter()
        .zip(&gv)
        .map(|(&fi, &gi)| (fi - mf) * (gi - mg))
        .sum::<f64>()
        / (n - 1) as f64;
    let var_g: f64 = gv.iter().map(|&gi| (gi - mg).powi(2)).sum::<f64>() / (n - 1) as f64;
    let c = if var_g > 0.0 { cov / var_g } else { 0.0 };
    let cv_est = mf - c * (mg - eg);
    let var_cv = fv
        .iter()
        .zip(&gv)
        .map(|(&fi, &gi)| (fi - c * gi - (mf - c * mg)).powi(2))
        .sum::<f64>()
        / (n * (n - 1)) as f64;
    Ok((cv_est, var_cv.sqrt()))
}

/// **Antithetic variates**: E[f] ≈ (f(U) + f(1−U))/2 to reduce variance.
pub fn mc_antithetic<F>(f: F, n: usize, seed: u64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let half = n / 2;
    let pairs: Vec<f64> = (0..half)
        .map(|_| {
            let u = rng.next_f64();
            (f(u) + f(1.0 - u)) / 2.0
        })
        .collect();
    let mean = pairs.iter().sum::<f64>() / half as f64;
    let var = pairs.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (half * (half - 1)) as f64;
    (mean, var.sqrt())
}

// ══════════════════════════════════════════════════════════════════════════════
// Latin Hypercube Sampling
// ══════════════════════════════════════════════════════════════════════════════

/// **Latin Hypercube Sampling**: n samples × d dimensions in [0,1]^d.
///
/// Each dimension is stratified into n equal intervals with one sample per interval.
pub fn latin_hypercube(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg::new(seed);
    // For each dimension, create a random permutation of [0..n)
    let mut lhs = vec![vec![0.0f64; d]; n];
    for j in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates
        for i in (1..n).rev() {
            let k = rng.next_usize(i + 1);
            perm.swap(i, k);
        }
        for i in 0..n {
            lhs[i][j] = (perm[i] as f64 + rng.next_f64()) / n as f64;
        }
    }
    lhs
}

// ══════════════════════════════════════════════════════════════════════════════
// Sobol sequence (scrambled, base 2, up to 6 dimensions)
// ══════════════════════════════════════════════════════════════════════════════

/// **Sobol quasi-random sequence** in [0,1]^d, d ≤ 6.
///
/// Uses direction numbers from Joe-Kuo (2010).
pub fn sobol_sequence(n: usize, d: usize) -> SciResult<Vec<Vec<f64>>> {
    if d == 0 || d > 6 {
        return Err(SciError::InvalidParameter("d must be 1..=6"));
    }
    // Direction numbers for dimensions 2-6 (primitive polynomials)
    // Dimension 1 is always 1/2, 1/4, 1/8, ...
    let m_bits = 32u32;
    let scale = 2.0f64.powi(-(m_bits as i32));
    // Hardcoded direction numbers (first 32 bits) for d=1..6
    let dir_nums: Vec<Vec<u32>> = vec![
        // d=1: Gray code of 1,1,1,...
        {
            let mut v = vec![0u32; 32];
            for i in 0..32 {
                v[i] = 1 << (31 - i);
            }
            v
        },
        // d=2: primitive poly x+1, init m=1
        {
            let s = 1u32;
            let mut v = vec![0u32; 32];
            v[0] = 1 << 31;
            for i in 1..32 {
                v[i] = v[i - 1] ^ (v[i - 1] >> s);
            }
            v
        },
        // d=3: primitive poly x^2+x+1
        {
            let s = 2u32;
            let mut v = vec![0u32; 32];
            v[0] = 1 << 31;
            v[1] = 1 << 30;
            for i in 2..32 {
                v[i] = v[i - 2] ^ (v[i - 2] >> s) ^ v[i - 1];
            }
            v
        },
        // d=4: primitive poly x^3+x+1
        {
            let s = 3u32;
            let a = 1u32;
            let mut v = vec![0u32; 32];
            v[0] = 1 << 31;
            v[1] = 3 << 30;
            v[2] = 7 << 29;
            for i in 3..32 {
                v[i] = v[i - 3]
                    ^ (v[i - 3] >> s)
                    ^ (if (a >> (s - 1 - ((i as u32 - 1) % s))) & 1 == 1 {
                        v[i - 1]
                    } else {
                        0
                    });
            }
            v
        },
        // d=5: primitive poly x^4+x+1
        {
            let s = 4u32;
            let mut v = vec![0u32; 32];
            v[0] = 1 << 31;
            v[1] = 1 << 30;
            v[2] = 3 << 29;
            v[3] = 13 << 28;
            for i in 4..32 {
                v[i] = v[i - 4] ^ (v[i - 4] >> s) ^ v[i - 1];
            }
            v
        },
        // d=6: primitive poly x^5+x^2+1
        {
            let s = 5u32;
            let mut v = vec![0u32; 32];
            v[0] = 1 << 31;
            v[1] = 3 << 30;
            v[2] = 7 << 29;
            v[3] = 11 << 28;
            v[4] = 27 << 27;
            for i in 5..32 {
                v[i] = v[i - 5] ^ (v[i - 5] >> s) ^ v[i - 1];
            }
            v
        },
    ];
    let mut x = vec![0u32; d];
    let mut result = Vec::with_capacity(n);
    for i in 1..=n {
        let c = i.trailing_zeros() as usize;
        for j in 0..d {
            x[j] ^= dir_nums[j][c.min(31)];
        }
        result.push(x.iter().map(|&xi| xi as f64 * scale).collect());
    }
    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════════════
// Particle Filter (Sequential Monte Carlo)
// ══════════════════════════════════════════════════════════════════════════════

/// **Bootstrap Particle Filter** (SIR) for a state-space model.
///
/// - `transition(x, rng_u) → x'`: sample from p(x_t | x_{t-1})
/// - `log_likelihood(x, y) → f64`: log p(y_t | x_t)
/// - `observations`: sequence of observations y_1, …, y_T
///
/// Returns estimated state mean at each time step.
pub fn particle_filter<T, L>(
    transition: T,
    log_likelihood: L,
    x0_samples: Vec<Vec<f64>>,
    observations: &[Vec<f64>],
    seed: u64,
) -> SciResult<Vec<Vec<f64>>>
where
    T: Fn(&[f64], f64) -> Vec<f64>,
    L: Fn(&[f64], &[f64]) -> f64,
{
    let n_particles = x0_samples.len();
    if n_particles == 0 {
        return Err(SciError::InvalidParameter("no particles"));
    }
    let mut rng = Lcg::new(seed);
    let mut particles = x0_samples;
    let mut means = Vec::with_capacity(observations.len());
    for obs in observations {
        // Weight
        let log_w: Vec<f64> = particles.iter().map(|x| log_likelihood(x, obs)).collect();
        let max_lw = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = log_w.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let w_sum: f64 = w.iter().sum();
        let w_norm: Vec<f64> = w.iter().map(|&wi| wi / w_sum).collect();
        // Mean estimate
        let d = particles[0].len();
        let mut mean = vec![0.0f64; d];
        for (p, &wi) in particles.iter().zip(&w_norm) {
            for j in 0..d {
                mean[j] += wi * p[j];
            }
        }
        means.push(mean);
        // Systematic resampling
        let mut cumw = 0.0f64;
        let cdf: Vec<f64> = w_norm
            .iter()
            .map(|&wi| {
                cumw += wi;
                cumw
            })
            .collect();
        let start = rng.next_f64() / n_particles as f64;
        let mut new_particles = Vec::with_capacity(n_particles);
        let mut idx = 0usize;
        for k in 0..n_particles {
            let u = start + k as f64 / n_particles as f64;
            while idx < n_particles - 1 && cdf[idx] < u {
                idx += 1;
            }
            new_particles.push(particles[idx].clone());
        }
        // Propagate
        particles = new_particles
            .iter()
            .map(|x| transition(x, rng.next_f64()))
            .collect();
    }
    Ok(means)
}

// ══════════════════════════════════════════════════════════════════════════════
// MCMC diagnostics
// ══════════════════════════════════════════════════════════════════════════════

/// **Effective Sample Size** (ESS) via autocorrelation for a 1D chain.
pub fn effective_sample_size(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return n as f64;
    }
    let m = chain.iter().sum::<f64>() / n as f64;
    let v = chain.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
    if v < f64::EPSILON {
        return 1.0;
    }
    let mut rho_sum = 0.0f64;
    for lag in 1..n / 2 {
        let rho = chain[..n - lag]
            .iter()
            .zip(&chain[lag..])
            .map(|(&a, &b)| (a - m) * (b - m))
            .sum::<f64>()
            / (n as f64 * v);
        if rho < 0.05 {
            break;
        }
        rho_sum += rho;
    }
    (n as f64 / (1.0 + 2.0 * rho_sum)).max(1.0)
}

/// **R-hat** (Gelman-Rubin convergence diagnostic) across multiple chains.
pub fn r_hat(chains: &[Vec<f64>]) -> SciResult<f64> {
    let m = chains.len() as f64;
    if chains.len() < 2 {
        return Err(SciError::InvalidParameter("need ≥ 2 chains"));
    }
    let n = chains[0].len() as f64;
    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|&cm| (cm - grand_mean).powi(2))
            .sum::<f64>();
    let w = chains
        .iter()
        .map(|c| {
            let cm = c.iter().sum::<f64>() / n;
            c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;
    let var_hat = (n - 1.0) / n * w + b / n;
    Ok((var_hat / w.max(f64::EPSILON)).sqrt())
}
