//! Phase 4 — High-quality RNG engines + rich sampling API.
//!
//! | Engine | Period | Speed | Quality | Use-case |
//! |---|---|---|---|---|
//! | `Xoshiro256ss` | 2²⁵⁶ |  | BigCrush Yes | General purpose |
//! | `Pcg64` | 2¹²⁸ |  | BigCrush Yes | Multiple streams |
//! | `Wyrand` | 2⁶⁴  |  | PractRand Yes | Ultra-fast |
//! | `Mt19937` | 2¹⁹⁹³⁷ |  | Good | Legacy / ML compat |
//!
//! All engines implement the `RngCore` trait, so they can be used uniformly
//! with `Sampler` and all distribution helpers.

use crate::errors::{SciError, SciResult};

// ══════════════════════════════════════════════════════════════════════════════
// RngCore trait
// ══════════════════════════════════════════════════════════════════════════════

/// Common interface for all RNG engines.
pub trait RngCore: Sized {
    fn next_u64(&mut self) -> u64;

    #[inline]
    fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }

    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform usize in 0..n (unbiased, rejection sampling).
    fn next_usize(&mut self, n: usize) -> usize {
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u64() as usize;
            if r >= threshold { return r % n; }
        }
    }

    /// Standard normal variate via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// Fill a `&mut [u8]` with random bytes.
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut chunks = buf.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v);
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let v = self.next_u64().to_le_bytes();
            rem.iter_mut().zip(v.iter()).for_each(|(b, r)| *b = *r);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Engine 1: Xoshiro256** (same as perf.rs but now in rng_ext)
// ══════════════════════════════════════════════════════════════════════════════

/// **Xoshiro256\*\*** — 256-bit state, BigCrush certified. Default choice.
#[derive(Clone, Debug)]
pub struct Xoshiro256ss { s: [u64; 4] }

impl Xoshiro256ss {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    /// Jump ahead 2¹²⁸ steps — use to create independent parallel streams.
    pub fn jump(&mut self) {
        const J: [u64; 4] = [
            0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
            0xa9582618e03fc9aa, 0x39abdc4529b1661c,
        ];
        let (mut s0, mut s1, mut s2, mut s3) = (0u64, 0u64, 0u64, 0u64);
        for &j in &J {
            for b in 0..64 {
                if (j >> b) & 1 != 0 {
                    s0 ^= self.s[0]; s1 ^= self.s[1];
                    s2 ^= self.s[2]; s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s = [s0, s1, s2, s3];
    }

    /// Create `n` independent streams via successive jumps.
    pub fn parallel_streams(seed: u64, n: usize) -> Vec<Self> {
        let mut base = Self::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(base.clone());
            base.jump();
        }
        out
    }
}

impl RngCore for Xoshiro256ss {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let res = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        res
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Engine 2: PCG64 (Permuted Congruential Generator)
// ══════════════════════════════════════════════════════════════════════════════

/// **PCG64** — 128-bit LCG with permutation output. Excellent statistical quality.
/// Supports independent streams via `stream` parameter.
#[derive(Clone, Debug)]
pub struct Pcg64 { state: u128, inc: u128 }

impl Pcg64 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let inc = ((stream as u128) << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.state = rng.state.wrapping_add(seed as u128);
        rng.next_u64(); // Warm up
        rng
    }

    pub fn from_seed(seed: u64) -> Self { Self::new(seed, 1) }
}

impl RngCore for Pcg64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        const MUL: u128 = 2549297995355413924u128 | (4865540595714422341u128 << 64);
        let old = self.state;
        self.state = old.wrapping_mul(MUL).wrapping_add(self.inc);
        // Output permutation: XSL-RR
        let xsl = ((old >> 64) as u64) ^ (old as u64);
        xsl.rotate_right((old >> 122) as u32)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Engine 3: Wyrand (ultra-fast 64-bit)
// ══════════════════════════════════════════════════════════════════════════════

/// **Wyrand** — 64-bit state, fastest engine, passes PractRand.
/// Best for non-cryptographic, high-throughput applications.
#[derive(Clone, Debug)]
pub struct Wyrand { state: u64 }

impl Wyrand {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
}

impl RngCore for Wyrand {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0xa0761d6478bd642f);
        let t = (self.state as u128).wrapping_mul(
            (self.state ^ 0xe7037ed1a0b428db) as u128
        );
        (t ^ (t >> 64)) as u64
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Engine 4: Mersenne Twister MT19937 (legacy compatibility)
// ══════════════════════════════════════════════════════════════════════════════

const MT_N: usize = 624;
const MT_M: usize = 397;

/// **MT19937** — Classic Mersenne Twister. Period 2¹⁹⁹³⁷.
/// Not recommended for new code; included for compatibility with NumPy/sklearn seeds.
#[derive(Clone)]
pub struct Mt19937 { mt: Box<[u32; MT_N]>, index: usize }

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = Box::new([0u32; MT_N]);
        mt[0] = seed;
        for i in 1..MT_N {
            mt[i] = 1812433253u32.wrapping_mul(mt[i-1] ^ (mt[i-1] >> 30)).wrapping_add(i as u32);
        }
        Self { mt, index: MT_N }
    }

    fn generate(&mut self) {
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER_MASK: u32 = 0x80000000;
        const LOWER_MASK: u32 = 0x7fffffff;
        for i in 0..MT_N {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[(i+1) % MT_N] & LOWER_MASK);
            self.mt[i] = self.mt[(i + MT_M) % MT_N] ^ (x >> 1);
            if x & 1 != 0 { self.mt[i] ^= MATRIX_A; }
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N { self.generate(); }
        let mut y = self.mt[self.index]; self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
}

impl RngCore for Mt19937 {
    fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }
}

impl core::fmt::Debug for Mt19937 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mt19937 {{ index: {} }}", self.index)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Sampler — rich distribution API on top of any RngCore
// ══════════════════════════════════════════════════════════════════════════════

/// Wraps any `RngCore` and exposes a full distribution sampling API.
pub struct Sampler<R: RngCore> { pub rng: R }

impl<R: RngCore> Sampler<R> {
    pub fn new(rng: R) -> Self { Self { rng } }

    // ── Basic ─────────────────────────────────────────────────────────────

    pub fn f64(&mut self) -> f64 { self.rng.next_f64() }
    pub fn f32(&mut self) -> f32 { self.rng.next_f32() }
    pub fn u64(&mut self) -> u64 { self.rng.next_u64() }
    pub fn bool(&mut self, p: f64) -> bool { self.rng.next_f64() < p }
    pub fn usize_range(&mut self, lo: usize, hi: usize) -> usize { lo + self.rng.next_usize(hi - lo) }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.rng.next_f64() * (hi - lo) }

    // ── Continuous distributions ──────────────────────────────────────────

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 { mean + std * self.rng.next_normal() }

    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.rng.next_f64().max(f64::MIN_POSITIVE).ln() / lambda
    }

    pub fn gamma(&mut self, alpha: f64, beta: f64) -> f64 {
        if alpha < 1.0 {
            return self.gamma(1.0 + alpha, beta) * self.rng.next_f64().powf(1.0 / alpha);
        }
        // Marsaglia-Tsang
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = self.rng.next_normal();
            let v = (1.0 + c * z).powi(3);
            if v > 0.0 {
                let u = self.rng.next_f64();
                if u < 1.0 - 0.0331 * (z * z) * (z * z) { return d * v / beta; }
                if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) { return d * v / beta; }
            }
        }
    }

    pub fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.gamma(alpha, 1.0);
        let y = self.gamma(beta, 1.0);
        x / (x + y)
    }

    pub fn chi_squared(&mut self, df: f64) -> f64 { self.gamma(df / 2.0, 0.5) }

    pub fn student_t(&mut self, df: f64) -> f64 {
        self.rng.next_normal() / (self.chi_squared(df) / df).sqrt()
    }

    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.rng.next_normal()).exp()
    }

    pub fn weibull(&mut self, k: f64, lambda: f64) -> f64 {
        lambda * (-self.rng.next_f64().max(f64::MIN_POSITIVE).ln()).powf(1.0 / k)
    }

    pub fn cauchy(&mut self, x0: f64, gamma: f64) -> f64 {
        x0 + gamma * (core::f64::consts::PI * (self.rng.next_f64() - 0.5)).tan()
    }

    pub fn laplace(&mut self, mu: f64, b: f64) -> f64 {
        let u = self.rng.next_f64() - 0.5;
        mu - b * u.signum() * (1.0 - 2.0 * u.abs()).max(f64::MIN_POSITIVE).ln()
    }

    pub fn logistic(&mut self, mu: f64, s: f64) -> f64 {
        let u = self.rng.next_f64().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        mu + s * (u / (1.0 - u)).ln()
    }

    pub fn pareto(&mut self, alpha: f64, x_m: f64) -> f64 {
        x_m / (1.0 - self.rng.next_f64()).powf(1.0 / alpha)
    }

    pub fn uniform_sphere(&mut self) -> [f64; 3] {
        let z = self.range(-1.0, 1.0);
        let r = (1.0 - z * z).sqrt();
        let phi = self.range(0.0, core::f64::consts::TAU);
        [r * phi.cos(), r * phi.sin(), z]
    }

    pub fn uniform_disk(&mut self) -> [f64; 2] {
        loop {
            let x = self.range(-1.0, 1.0);
            let y = self.range(-1.0, 1.0);
            if x * x + y * y <= 1.0 { return [x, y]; }
        }
    }

    // ── Discrete distributions ────────────────────────────────────────────

    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        (0..n).filter(|_| self.rng.next_f64() < p).count() as u64
    }

    pub fn poisson(&mut self, lambda: f64) -> u64 {
        // Knuth direct method (good for small lambda)
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut prod = 1.0f64;
        loop {
            prod *= self.rng.next_f64();
            if prod <= l { return k; }
            k += 1;
        }
    }

    pub fn geometric(&mut self, p: f64) -> u64 {
        (self.rng.next_f64().max(f64::MIN_POSITIVE).ln() / (1.0 - p).ln()).ceil() as u64
    }

    pub fn negative_binomial(&mut self, r: f64, p: f64) -> u64 {
        let lambda = self.gamma(r, p / (1.0 - p));
        self.poisson(lambda)
    }

    pub fn hypergeometric(&mut self, n_pop: u64, n_good: u64, n_draw: u64) -> u64 {
        (0..n_draw).filter(|i| {
            let remaining_good = n_good.saturating_sub(*i);
            let remaining_total = n_pop.saturating_sub(*i);
            if remaining_total == 0 { return false; }
            self.rng.next_f64() < remaining_good as f64 / remaining_total as f64
        }).count() as u64
    }

    // ── Multivariate ──────────────────────────────────────────────────────

    pub fn multivariate_normal(&mut self, mean: &[f64], chol_l: &[Vec<f64>]) -> Vec<f64> {
        let n = mean.len();
        let z: Vec<f64> = (0..n).map(|_| self.rng.next_normal()).collect();
        (0..n).map(|i| {
            mean[i] + (0..=i).map(|j| chol_l[i][j] * z[j]).sum::<f64>()
        }).collect()
    }

    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let gammas: Vec<f64> = alpha.iter().map(|&a| self.gamma(a, 1.0)).collect();
        let s: f64 = gammas.iter().sum();
        gammas.iter().map(|&g| g / s).collect()
    }

    // ── Sampling / Shuffling ─────────────────────────────────────────────

    /// Fisher-Yates shuffle in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.rng.next_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Sample `k` elements without replacement (reservoir sampling).
    pub fn sample_without_replacement<T: Clone>(&mut self, pool: &[T], k: usize) -> SciResult<Vec<T>> {
        if k > pool.len() { return Err(SciError::InvalidParameter("k > pool size")); }
        let mut reservoir: Vec<T> = pool[..k].to_vec();
        for i in k..pool.len() {
            let j = self.rng.next_usize(i + 1);
            if j < k { reservoir[j] = pool[i].clone(); }
        }
        Ok(reservoir)
    }

    /// Weighted random choice (roulette wheel).
    pub fn weighted_choice(&mut self, weights: &[f64]) -> SciResult<usize> {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 { return Err(SciError::InvalidParameter("weights must sum > 0")); }
        let u = self.rng.next_f64() * total;
        let mut cum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if u < cum { return Ok(i); }
        }
        Ok(weights.len() - 1)
    }

    /// Alias method for O(1) weighted sampling after O(n) setup.
    pub fn build_alias_table(weights: &[f64]) -> SciResult<AliasTable> {
        AliasTable::new(weights)
    }

    /// Generate `n` i.i.d. samples from a function `f(&mut R) -> T`.
    pub fn sample_n<T, F: FnMut(&mut R) -> T>(&mut self, mut f: F, n: usize) -> Vec<T> {
        (0..n).map(|_| f(&mut self.rng)).collect()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Alias table — O(1) weighted sampling
// ══════════════════════════════════════════════════════════════════════════════

/// Vose's alias method — O(1) sampling after O(n) setup.
pub struct AliasTable { prob: Vec<f64>, alias: Vec<usize> }

impl AliasTable {
    pub fn new(weights: &[f64]) -> SciResult<Self> {
        let n = weights.len();
        if n == 0 { return Err(SciError::InvalidParameter("empty weights")); }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 { return Err(SciError::InvalidParameter("zero total weight")); }
        let mut prob: Vec<f64> = weights.iter().map(|&w| w * n as f64 / total).collect();
        let mut alias = vec![0usize; n];
        let mut small: Vec<usize> = (0..n).filter(|&i| prob[i] < 1.0).collect();
        let mut large: Vec<usize> = (0..n).filter(|&i| prob[i] >= 1.0).collect();
        while let (Some(s), Some(l)) = (small.pop(), large.last().copied()) {
            alias[s] = l;
            prob[l] -= 1.0 - prob[s];
            if prob[l] < 1.0 { large.pop(); small.push(l); } else { /* l stays */ }
        }
        Ok(Self { prob, alias })
    }

    pub fn sample<R: RngCore>(&self, rng: &mut R) -> usize {
        let i = rng.next_usize(self.prob.len());
        if rng.next_f64() < self.prob[i] { i } else { self.alias[i] }
    }

    pub fn sample_n<R: RngCore>(&self, rng: &mut R, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Convenient type aliases
// ══════════════════════════════════════════════════════════════════════════════

/// Default high-quality sampler (Xoshiro256**).
pub type DefaultSampler = Sampler<Xoshiro256ss>;

/// PCG64-backed sampler (good for parallel streams).
pub type PcgSampler = Sampler<Pcg64>;

/// Ultra-fast sampler (Wyrand).
pub type FastSampler = Sampler<Wyrand>;

/// Legacy MT19937 sampler.
pub type MtSampler = Sampler<Mt19937>;

impl DefaultSampler {
    pub fn seeded(seed: u64) -> Self { Sampler::new(Xoshiro256ss::new(seed)) }
}

impl PcgSampler {
    pub fn seeded(seed: u64) -> Self { Sampler::new(Pcg64::from_seed(seed)) }
    /// Independent stream sampler.
    pub fn stream(seed: u64, stream: u64) -> Self { Sampler::new(Pcg64::new(seed, stream)) }
}

impl FastSampler {
    pub fn seeded(seed: u64) -> Self { Sampler::new(Wyrand::new(seed)) }
}
