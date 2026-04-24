//! Extended statistical inference: non-parametric tests, multiple comparison
//! corrections, bootstrap, permutation tests, additional correlations.
//!
//! | Test | Function |
//! |---|---|
//! | Mann-Whitney U | `mann_whitney_u` |
//! | Wilcoxon signed-rank | `wilcoxon_signed_rank` |
//! | Kruskal-Wallis | `kruskal_wallis` |
//! | Kolmogorov-Smirnov 1-sample | `ks_test_1sample` |
//! | Kolmogorov-Smirnov 2-sample | `ks_test_2sample` |
//! | Fisher's exact | `fisher_exact_2x2` |
//! | Levene's test | `levene_test` |
//! | Spearman correlation | `spearman_correlation` |
//! | Kendall's τ | `kendall_tau` |
//! | Bootstrap CI | `bootstrap_ci` |
//! | Permutation test | `permutation_test` |
//! | Bonferroni / Holm / BH | `p_adjust` |

use crate::distributions::ln_gamma;
use crate::errors::{SciError, SciResult};
use crate::inference::HypothesisTestResult;
use crate::probability::{chi_square_cdf, student_t_cdf};

// ══════════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════════

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
/// Assign average ranks (handles ties) to a sorted list.
fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(f64, usize)> = data
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (indexed[j + 1].0 - indexed[j].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for k in i..=j {
            r[indexed[k].1] = avg;
        }
        i = j + 1;
    }
    r
}

/// Normal CDF approximation (used for large-sample p-values).
fn normal_cdf_approx(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = 1.0 - poly * (-0.5 * z * z).exp() / (2.0 * core::f64::consts::PI).sqrt();
    if z >= 0.0 { phi } else { 1.0 - phi }
}

// ══════════════════════════════════════════════════════════════════════════════
// Mann-Whitney U test (Wilcoxon rank-sum)
// ══════════════════════════════════════════════════════════════════════════════

/// **Mann-Whitney U** test for two independent samples.
///
/// H₀: the two populations have the same distribution.
/// Returns two-sided p-value via normal approximation (valid for n₁, n₂ > 8).
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> SciResult<HypothesisTestResult> {
    if a.len() < 2 || b.len() < 2 {
        return Err(SciError::InvalidParameter("need n >= 2"));
    }
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    // Pool and rank
    let mut combined: Vec<(f64, u8)> = a
        .iter()
        .map(|&v| (v, 0u8))
        .chain(b.iter().map(|&v| (v, 1u8)))
        .collect();
    combined.sort_by(|x, y| x.0.total_cmp(&y.0));
    let n_total = combined.len();
    // Average rank ties
    let mut ranked = vec![0.0f64; n_total];
    let mut i = 0;
    while i < n_total {
        let mut j = i;
        while j + 1 < n_total && (combined[j + 1].0 - combined[j].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for k in i..=j {
            ranked[k] = avg;
        }
        i = j + 1;
    }
    let r1: f64 = combined
        .iter()
        .zip(&ranked)
        .filter(|(c, _)| c.1 == 0)
        .map(|(_, &r)| r)
        .sum();
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let u = u1.min(n1 * n2 - u1);
    let mu = n1 * n2 / 2.0;
    let sigma = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p = 2.0 * normal_cdf_approx(z.min(-z.abs()));
    Ok(HypothesisTestResult {
        statistic: u1,
        p_value: p,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Wilcoxon signed-rank test (paired)
// ══════════════════════════════════════════════════════════════════════════════

/// **Wilcoxon signed-rank test** for matched pairs or one-sample vs. median.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> SciResult<HypothesisTestResult> {
    if a.len() != b.len() {
        return Err(SciError::InvalidParameter("samples must be same length"));
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(ai, bi)| ai - bi)
        .filter(|&d| d.abs() > f64::EPSILON)
        .collect();
    let n = diffs.len() as f64;
    if n < 1.0 {
        return Err(SciError::InvalidParameter("no non-zero differences"));
    }
    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let r = ranks(&abs_diffs);
    let w_plus: f64 = diffs
        .iter()
        .zip(&r)
        .map(|(&d, &ri)| if d > 0.0 { ri } else { 0.0 })
        .sum();
    let w_minus: f64 = diffs
        .iter()
        .zip(&r)
        .map(|(&d, &ri)| if d < 0.0 { ri } else { 0.0 })
        .sum();
    let w = w_plus.min(w_minus);
    // Normal approximation
    let mu = n * (n + 1.0) / 4.0;
    let sigma = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0).sqrt();
    let z = (w - mu) / sigma;
    let p = 2.0 * normal_cdf_approx(z.min(-z.abs()));
    Ok(HypothesisTestResult {
        statistic: w,
        p_value: p,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Kruskal-Wallis H test (non-parametric ANOVA)
// ══════════════════════════════════════════════════════════════════════════════

/// **Kruskal-Wallis H test** for k independent groups.
///
/// H₀: all groups come from the same distribution.
/// Statistic H ~ χ²(k-1) under H₀.
pub fn kruskal_wallis(groups: &[&[f64]]) -> SciResult<HypothesisTestResult> {
    let k = groups.len();
    if k < 2 {
        return Err(SciError::InvalidParameter("need at least 2 groups"));
    }
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().cloned()).collect();
    let n = all.len() as f64;
    // Pool ranks
    let mut combined: Vec<(f64, usize)> = all
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut global_ranks = vec![0.0f64; all.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j + 1 < combined.len() && (combined[j + 1].0 - combined[j].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for l in i..=j {
            global_ranks[combined[l].1] = avg;
        }
        i = j + 1;
    }
    // Compute H
    let mut offset = 0;
    let mut h_sum = 0.0f64;
    for g in groups {
        let ng = g.len() as f64;
        let rg: f64 = global_ranks[offset..offset + g.len()].iter().sum();
        h_sum += rg * rg / ng;
        offset += g.len();
    }
    let h = 12.0 / (n * (n + 1.0)) * h_sum - 3.0 * (n + 1.0);
    let df = (k - 1) as f64;
    let p = 1.0 - chi_square_cdf(h, df)?;
    Ok(HypothesisTestResult {
        statistic: h,
        p_value: p,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Kolmogorov-Smirnov tests
// ══════════════════════════════════════════════════════════════════════════════

/// **KS 1-sample test**: compare empirical CDF to a theoretical CDF `cdf_fn`.
pub fn ks_test_1sample<F: Fn(f64) -> f64>(
    data: &[f64],
    cdf_fn: F,
) -> SciResult<HypothesisTestResult> {
    if data.is_empty() {
        return Err(SciError::InvalidParameter("empty data"));
    }
    let n = data.len() as f64;
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let d = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = cdf_fn(x);
            let ecdf_after = (i + 1) as f64 / n;
            let ecdf_before = i as f64 / n;
            (f - ecdf_before).abs().max((ecdf_after - f).abs())
        })
        .fold(0.0f64, f64::max);
    // Approximate p-value (Marsaglia)
    let p = ks_p_value(d, data.len());
    Ok(HypothesisTestResult {
        statistic: d,
        p_value: p,
    })
}

/// **KS 2-sample test**: test whether two samples come from the same distribution.
pub fn ks_test_2sample(a: &[f64], b: &[f64]) -> SciResult<HypothesisTestResult> {
    if a.is_empty() || b.is_empty() {
        return Err(SciError::InvalidParameter("empty sample"));
    }
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let mut sa = a.to_vec();
    sa.sort_by(|x, y| x.total_cmp(y));
    let mut sb = b.to_vec();
    sb.sort_by(|x, y| x.total_cmp(y));
    let mut d = 0.0f64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < sa.len() && j < sb.len() {
        let f1 = (i + 1) as f64 / n1;
        let f2 = (j + 1) as f64 / n2;
        if sa[i] <= sb[j] {
            i += 1;
        } else {
            j += 1;
        }
        d = d.max((f1 - f2).abs());
    }
    let n_eff = (n1 * n2 / (n1 + n2)).sqrt();
    let p = ks_p_value(d, (n_eff * n_eff) as usize);
    Ok(HypothesisTestResult {
        statistic: d,
        p_value: p,
    })
}

fn ks_p_value(d: f64, n: usize) -> f64 {
    let lambda = ((n as f64).sqrt() + 0.12 + 0.11 / (n as f64).sqrt()) * d;
    // Kolmogorov distribution CDF complement
    let mut sum = 0.0f64;
    for k in 1..=20i32 {
        let term = 2.0
            * (if k % 2 == 0 { 1.0 } else { -1.0 })
            * (-2.0 * k as f64 * k as f64 * lambda * lambda).exp();
        sum += term;
    }
    sum.clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// Fisher's exact test (2×2)
// ══════════════════════════════════════════════════════════════════════════════

/// **Fisher's exact test** for a 2×2 contingency table [[a, b], [c, d]].
///
/// Returns two-sided p-value.
pub fn fisher_exact_2x2(a: u64, b: u64, c: u64, d: u64) -> SciResult<HypothesisTestResult> {
    let n = a + b + c + d;
    let r1 = a + b;
    let r2 = c + d;
    let c1 = a + c;
    // P(table) = C(r1,a)·C(r2,c) / C(n, c1)
    let p_obs = hypergeom_log_pmf(a, n, r1, c1).exp();
    // Sum all tables with P ≤ P_obs
    let lo = c1.saturating_sub(r2);
    let hi = r1.min(c1);
    let mut p_total = 0.0f64;
    for k in lo..=hi {
        let pk = hypergeom_log_pmf(k, n, r1, c1).exp();
        if pk <= p_obs + 1e-10 {
            p_total += pk;
        }
    }
    let p = p_total.min(1.0);
    Ok(HypothesisTestResult {
        statistic: p_obs,
        p_value: p,
    })
}

fn hypergeom_log_pmf(k: u64, n: u64, k_pop: u64, n_draw: u64) -> f64 {
    ln_binom(k_pop, k) + ln_binom(n - k_pop, n_draw.saturating_sub(k)) - ln_binom(n, n_draw)
}
fn ln_binom(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

// ══════════════════════════════════════════════════════════════════════════════
// Levene's test for equality of variances
// ══════════════════════════════════════════════════════════════════════════════

/// **Levene's test** (using group means) for H₀: all group variances are equal.
pub fn levene_test(groups: &[&[f64]]) -> SciResult<HypothesisTestResult> {
    let k = groups.len();
    if k < 2 {
        return Err(SciError::InvalidParameter("need at least 2 groups"));
    }
    let z: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let m = mean(g);
            g.iter().map(|&xi| (xi - m).abs()).collect()
        })
        .collect();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let all_z: Vec<f64> = z.iter().flat_map(|zi| zi.iter().cloned()).collect();
    let z_grand = mean(&all_z);
    let mut between = 0.0f64;
    let mut within = 0.0f64;
    for (zi, g) in z.iter().zip(groups.iter()) {
        let ni = g.len() as f64;
        let zi_mean = mean(zi);
        between += ni * (zi_mean - z_grand) * (zi_mean - z_grand);
        within += zi
            .iter()
            .map(|&v| (v - zi_mean) * (v - zi_mean))
            .sum::<f64>();
    }
    let f = (between / (k - 1) as f64) / (within / (n_total - k) as f64);
    // F distribution p-value approximation via chi-sq with df1 dof
    let p = 1.0 - chi_square_cdf(f, (k - 1) as f64)?; // rough
    Ok(HypothesisTestResult {
        statistic: f,
        p_value: p,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Spearman & Kendall correlations
// ══════════════════════════════════════════════════════════════════════════════

/// **Spearman rank correlation** ρ and its t-statistic p-value.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> SciResult<(f64, f64)> {
    if x.len() != y.len() || x.len() < 3 {
        return Err(SciError::InvalidParameter("need n ≥ 3 matched values"));
    }
    let n = x.len() as f64;
    let rx = ranks(x);
    let ry = ranks(y);
    let mx = mean(&rx);
    let my = mean(&ry);
    let cov: f64 = rx
        .iter()
        .zip(&ry)
        .map(|(&a, &b)| (a - mx) * (b - my))
        .sum::<f64>()
        / (n - 1.0);
    let sx = rx.iter().map(|&a| (a - mx) * (a - mx)).sum::<f64>().sqrt() / (n - 1.0).sqrt();
    let sy = ry.iter().map(|&b| (b - my) * (b - my)).sum::<f64>().sqrt() / (n - 1.0).sqrt();
    if sx < f64::EPSILON || sy < f64::EPSILON {
        return Err(SciError::InvalidParameter("zero variance in ranks"));
    }
    let rho = cov / (sx * sy);
    let t = rho * (n - 2.0).sqrt() / (1.0 - rho * rho).max(f64::EPSILON).sqrt();
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), n - 2.0)?);
    Ok((rho, p))
}

/// **Kendall's τ-b** rank correlation and approximate p-value.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> SciResult<(f64, f64)> {
    if x.len() != y.len() || x.len() < 3 {
        return Err(SciError::InvalidParameter("need n ≥ 3 matched values"));
    }
    let n = x.len();
    let (mut conc, mut disc, mut tx, mut ty) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (x[i] - x[j]).signum() as i64;
            let dy = (y[i] - y[j]).signum() as i64;
            match dx * dy {
                1 => conc += 1,
                -1 => disc += 1,
                _ => {
                    if dx == 0 {
                        tx += 1;
                    }
                    if dy == 0 {
                        ty += 1;
                    }
                }
            }
        }
    }
    let nn = n as f64 * (n as f64 - 1.0) / 2.0;
    let denom = ((nn - tx as f64) * (nn - ty as f64)).sqrt();
    let tau = if denom < f64::EPSILON {
        0.0
    } else {
        (conc - disc) as f64 / denom
    };
    let sigma = ((2.0 * (2.0 * n as f64 + 5.0)) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let z = tau / sigma;
    let p = 2.0 * normal_cdf_approx(-z.abs());
    Ok((tau, p))
}

// ══════════════════════════════════════════════════════════════════════════════
// Bootstrap confidence interval
// ══════════════════════════════════════════════════════════════════════════════

/// **Bootstrap percentile CI** for a statistic (e.g. mean, median).
///
/// `stat_fn` takes a bootstrap resample and returns the statistic.
/// Returns `(lower, upper)` for the given `confidence` level (e.g. 0.95).
pub fn bootstrap_ci<F>(
    data: &[f64],
    stat_fn: F,
    n_boot: usize,
    confidence: f64,
    seed: u64,
) -> SciResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if data.is_empty() {
        return Err(SciError::InvalidParameter("empty data"));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(SciError::InvalidParameter("confidence ∈ (0,1)"));
    }
    let n = data.len();
    let mut rng = Lcg64(seed.wrapping_add(777));
    let mut stats: Vec<f64> = (0..n_boot)
        .map(|_| {
            let sample: Vec<f64> = (0..n).map(|_| data[rng.next_usize() % n]).collect();
            stat_fn(&sample)
        })
        .collect();
    stats.sort_by(|a, b| a.total_cmp(b));
    let alpha = 1.0 - confidence;
    let lo = (alpha / 2.0 * n_boot as f64) as usize;
    let hi = ((1.0 - alpha / 2.0) * n_boot as f64) as usize;
    Ok((stats[lo.min(n_boot - 1)], stats[hi.min(n_boot - 1)]))
}

// ══════════════════════════════════════════════════════════════════════════════
// Permutation test
// ══════════════════════════════════════════════════════════════════════════════

/// **Permutation test** for difference of means between two groups.
///
/// Returns two-sided p-value (proportion of permutations with |Δ| ≥ observed).
pub fn permutation_test(a: &[f64], b: &[f64], n_perm: usize, seed: u64) -> SciResult<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(SciError::InvalidParameter("empty sample"));
    }
    let obs = (mean(a) - mean(b)).abs();
    let mut combined: Vec<f64> = a.iter().chain(b.iter()).cloned().collect();
    let n_a = a.len();
    let mut rng = Lcg64(seed.wrapping_add(999));
    let mut count = 0usize;
    for _ in 0..n_perm {
        // Fisher-Yates shuffle
        let n = combined.len();
        for i in (1..n).rev() {
            let j = rng.next_usize() % (i + 1);
            combined.swap(i, j);
        }
        let diff = (mean(&combined[..n_a]) - mean(&combined[n_a..])).abs();
        if diff >= obs {
            count += 1;
        }
    }
    Ok(count as f64 / n_perm as f64)
}

// ══════════════════════════════════════════════════════════════════════════════
// Multiple testing correction
// ══════════════════════════════════════════════════════════════════════════════

/// Method for p-value adjustment.
#[derive(Debug, Clone, Copy)]
pub enum PValueAdjustMethod {
    /// Bonferroni: multiply each p by m.
    Bonferroni,
    /// Holm (step-down Bonferroni): less conservative.
    Holm,
    /// Benjamini-Hochberg: controls false discovery rate.
    BenjaminiHochberg,
}

/// **Multiple comparison p-value adjustment**.
///
/// Returns adjusted p-values (same order as input).
pub fn p_adjust(p_values: &[f64], method: PValueAdjustMethod) -> SciResult<Vec<f64>> {
    let m = p_values.len();
    if m == 0 {
        return Err(SciError::InvalidParameter("empty p-values"));
    }
    match method {
        PValueAdjustMethod::Bonferroni => {
            Ok(p_values.iter().map(|&p| (p * m as f64).min(1.0)).collect())
        }
        PValueAdjustMethod::Holm => {
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&i, &j| p_values[i].total_cmp(&p_values[j]));
            let mut adj = vec![0.0f64; m];
            let mut running_max = 0.0f64;
            for (rank, &idx) in order.iter().enumerate() {
                let a = (p_values[idx] * (m - rank) as f64).min(1.0);
                running_max = running_max.max(a);
                adj[idx] = running_max;
            }
            Ok(adj)
        }
        PValueAdjustMethod::BenjaminiHochberg => {
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&i, &j| p_values[j].total_cmp(&p_values[i])); // descending
            let mut adj = vec![0.0f64; m];
            let mut running_min = 1.0f64;
            for (rank, &idx) in order.iter().enumerate() {
                let k = m - rank; // rank in ascending order (1-indexed)
                let a = (p_values[idx] * m as f64 / k as f64).min(1.0);
                running_min = running_min.min(a);
                adj[idx] = running_min;
            }
            Ok(adj)
        }
    }
}

// ─── Minimal LCG RNG ─────────────────────────────────────────────────────────
struct Lcg64(u64);
impl Lcg64 {
    fn next_usize(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize
    }
}
