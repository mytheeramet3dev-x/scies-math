//! Statistical inference — hypothesis tests and confidence intervals.
//!
//! # Tests
//!
//! | Function | Test | H₀ |
//! |---|---|---|
//! | `one_sample_t_test` | One-sample t | μ = μ₀ |
//! | `two_sample_t_test` | Welch t-test | μ₁ = μ₂ |
//! | `paired_t_test` | Paired t | μ_diff = 0 |
//! | `one_way_anova` | One-way ANOVA | All group means equal |
//! | `chi_square_gof` | Goodness-of-fit χ² | Observed = Expected |
//! | `chi_square_independence` | χ² independence | Rows ⊥ Columns |
//! | `ks_test` | Kolmogorov-Smirnov | CDF = reference CDF |
//! | `mann_whitney_u` | Mann-Whitney U | Identical distributions |
//! | `wilcoxon_signed_rank` | Wilcoxon | Symmetric about 0 |
//!
//! # Return type — `TestResult`
//!
//! ```text
//! pub struct TestResult {
//!     pub statistic: f64,
//!     pub p_value:   f64,
//!     pub reject_h0: bool,   // true when p_value < alpha
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use scies_math_th::inference::two_sample_t_test;
//!
//! let a = vec![5.1, 4.9, 5.0, 5.2, 4.8];
//! let b = vec![5.5, 5.3, 5.4, 5.6, 5.2];
//! let result = two_sample_t_test(&a, &b, 0.05).unwrap();
//! println!("p = {:.4}, reject H0: {}", result.p_value, result.reject_h0);
//! ```
//!
//! For bootstrap CI, permutation tests, and FDR correction see [`crate::inference_ext`].
use crate::errors::{SciError, SciResult};
use crate::probability::{
    chi_square_cdf, confidence_interval_mean_unknown_sigma, f_cdf, normal_cdf, student_t_cdf,
};
use crate::statistics::{mean, variance};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HypothesisTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRegressionResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnovaResult {
    pub f_statistic: f64,
    pub degrees_of_freedom_between: usize,
    pub degrees_of_freedom_within: usize,
    pub p_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairwiseComparisonResult {
    pub group_a: usize,
    pub group_b: usize,
    pub mean_difference: f64,
    pub statistic: f64,
    pub p_value: f64,
}

pub fn covariance(x: &[f64], y: &[f64], sample: bool) -> SciResult<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(SciError::InvalidParameter(
            "x and y must be non-empty and have the same length",
        ));
    }
    if sample && x.len() < 2 {
        return Err(SciError::InvalidParameter(
            "sample covariance requires at least 2 observations",
        ));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let denominator = if sample {
        (x.len() - 1) as f64
    } else {
        x.len() as f64
    };
    Ok(numerator / denominator)
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> SciResult<f64> {
    let cov = covariance(x, y, true)?;
    let std_x = variance(x, true)?.sqrt();
    let std_y = variance(y, true)?.sqrt();
    if std_x <= f64::EPSILON || std_y <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    Ok(cov / (std_x * std_y))
}

pub fn simple_linear_regression(x: &[f64], y: &[f64]) -> SciResult<LinearRegressionResult> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(SciError::InvalidParameter(
            "x and y must have the same length and contain at least 2 points",
        ));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let ss_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let ss_xx: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    if ss_xx <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let prediction = slope * xi + intercept;
            (yi - prediction).powi(2)
        })
        .sum();
    let r_squared = if ss_tot <= f64::EPSILON {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    Ok(LinearRegressionResult {
        slope,
        intercept,
        r_squared,
    })
}

pub fn z_test_mean_known_sigma(
    sample_mean: f64,
    population_mean: f64,
    sigma: f64,
    sample_size: usize,
) -> SciResult<HypothesisTestResult> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma must be positive"));
    }
    if sample_size == 0 {
        return Err(SciError::InvalidParameter("sample size must be positive"));
    }

    let z = (sample_mean - population_mean) / (sigma / (sample_size as f64).sqrt());
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs(), 0.0, 1.0)?);
    Ok(HypothesisTestResult {
        statistic: z,
        p_value,
    })
}

pub fn chi_square_goodness_of_fit(
    observed: &[f64],
    expected: &[f64],
) -> SciResult<HypothesisTestResult> {
    if observed.len() != expected.len() || observed.is_empty() {
        return Err(SciError::InvalidParameter(
            "observed and expected must be non-empty and have the same length",
        ));
    }
    if expected.iter().any(|value| *value <= 0.0) {
        return Err(SciError::InvalidParameter(
            "expected frequencies must be positive",
        ));
    }

    let statistic: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(obs, exp)| (obs - exp).powi(2) / exp)
        .sum();

    let degrees_of_freedom = observed.len() as f64 - 1.0;
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "chi-square test requires at least 2 categories",
        ));
    }

    let p_value = 1.0 - chi_square_cdf(statistic, degrees_of_freedom)?;

    Ok(HypothesisTestResult { statistic, p_value })
}

pub fn one_sample_t_test(sample: &[f64], population_mean: f64) -> SciResult<HypothesisTestResult> {
    if sample.len() < 2 {
        return Err(SciError::InvalidParameter(
            "one-sample t-test requires at least 2 observations",
        ));
    }

    let sample_mean = mean(sample)?;
    let sample_variance = variance(sample, true)?;
    let standard_error = (sample_variance / sample.len() as f64).sqrt();
    if standard_error <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let t = (sample_mean - population_mean) / standard_error;
    let degrees_of_freedom = (sample.len() - 1) as f64;
    let p_value = 2.0 * (1.0 - student_t_cdf(t.abs(), degrees_of_freedom)?);
    Ok(HypothesisTestResult {
        statistic: t,
        p_value,
    })
}

pub fn two_sample_t_test(sample_a: &[f64], sample_b: &[f64]) -> SciResult<HypothesisTestResult> {
    validate_two_samples(sample_a, sample_b)?;

    let mean_a = mean(sample_a)?;
    let mean_b = mean(sample_b)?;
    let var_a = variance(sample_a, true)?;
    let var_b = variance(sample_b, true)?;
    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;

    let pooled_variance = (((n_a - 1.0) * var_a) + ((n_b - 1.0) * var_b)) / (n_a + n_b - 2.0);
    let standard_error = (pooled_variance * (1.0 / n_a + 1.0 / n_b)).sqrt();
    if standard_error <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let t = (mean_a - mean_b) / standard_error;
    let degrees_of_freedom = n_a + n_b - 2.0;
    let p_value = 2.0 * (1.0 - student_t_cdf(t.abs(), degrees_of_freedom)?);
    Ok(HypothesisTestResult {
        statistic: t,
        p_value,
    })
}

pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64]) -> SciResult<HypothesisTestResult> {
    validate_two_samples(sample_a, sample_b)?;

    let mean_a = mean(sample_a)?;
    let mean_b = mean(sample_b)?;
    let var_a = variance(sample_a, true)?;
    let var_b = variance(sample_b, true)?;
    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;

    let term_a = var_a / n_a;
    let term_b = var_b / n_b;
    let standard_error = (term_a + term_b).sqrt();
    if standard_error <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let t = (mean_a - mean_b) / standard_error;
    let numerator = (term_a + term_b).powi(2);
    let denominator = term_a.powi(2) / (n_a - 1.0).max(f64::EPSILON)
        + term_b.powi(2) / (n_b - 1.0).max(f64::EPSILON);
    if denominator <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    let degrees_of_freedom = numerator / denominator;
    let p_value = 2.0 * (1.0 - student_t_cdf(t.abs(), degrees_of_freedom)?);
    Ok(HypothesisTestResult {
        statistic: t,
        p_value,
    })
}

pub fn paired_t_test(sample_a: &[f64], sample_b: &[f64]) -> SciResult<HypothesisTestResult> {
    validate_two_samples(sample_a, sample_b)?;

    let differences = sample_a
        .iter()
        .zip(sample_b.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    one_sample_t_test(&differences, 0.0)
}

pub fn two_proportion_z_test(
    successes_a: usize,
    trials_a: usize,
    successes_b: usize,
    trials_b: usize,
) -> SciResult<HypothesisTestResult> {
    if trials_a == 0 || trials_b == 0 {
        return Err(SciError::InvalidParameter("trial counts must be positive"));
    }
    if successes_a > trials_a || successes_b > trials_b {
        return Err(SciError::InvalidParameter(
            "successes must not exceed trials",
        ));
    }

    let p_a = successes_a as f64 / trials_a as f64;
    let p_b = successes_b as f64 / trials_b as f64;
    let pooled = (successes_a + successes_b) as f64 / (trials_a + trials_b) as f64;
    let standard_error =
        (pooled * (1.0 - pooled) * (1.0 / trials_a as f64 + 1.0 / trials_b as f64)).sqrt();
    if standard_error <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let z = (p_a - p_b) / standard_error;
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs(), 0.0, 1.0)?);
    Ok(HypothesisTestResult {
        statistic: z,
        p_value,
    })
}

pub fn one_way_anova(groups: &[&[f64]]) -> SciResult<AnovaResult> {
    if groups.len() < 2 {
        return Err(SciError::InvalidParameter(
            "one-way ANOVA requires at least 2 groups",
        ));
    }
    if groups.iter().any(|group| group.is_empty()) {
        return Err(SciError::InvalidParameter("ANOVA groups must be non-empty"));
    }

    let total_count: usize = groups.iter().map(|group| group.len()).sum();
    if total_count <= groups.len() {
        return Err(SciError::InvalidParameter(
            "ANOVA requires at least one degree of freedom within groups",
        ));
    }

    let grand_mean = groups
        .iter()
        .flat_map(|group| group.iter())
        .copied()
        .sum::<f64>()
        / total_count as f64;

    let ss_between: f64 = groups
        .iter()
        .map(|group| {
            let group_mean = mean(group).unwrap_or(0.0);
            group.len() as f64 * (group_mean - grand_mean).powi(2)
        })
        .sum();

    let ss_within: f64 = groups
        .iter()
        .map(|group| {
            let group_mean = mean(group).unwrap_or(0.0);
            group
                .iter()
                .map(|value| (value - group_mean).powi(2))
                .sum::<f64>()
        })
        .sum();

    let df_between = groups.len() - 1;
    let df_within = total_count - groups.len();
    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;
    if ms_within <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let f_statistic = ms_between / ms_within;
    let p_value = 1.0 - f_cdf(f_statistic, df_between as f64, df_within as f64)?;

    Ok(AnovaResult {
        f_statistic,
        degrees_of_freedom_between: df_between,
        degrees_of_freedom_within: df_within,
        p_value,
    })
}

pub fn pairwise_tukey_like_comparisons(
    groups: &[&[f64]],
) -> SciResult<Vec<PairwiseComparisonResult>> {
    let anova = one_way_anova(groups)?;
    let ms_within = within_group_mean_square(groups)?;

    let mut results = Vec::new();
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let mean_i = mean(groups[i])?;
            let mean_j = mean(groups[j])?;
            let n_i = groups[i].len() as f64;
            let n_j = groups[j].len() as f64;
            let standard_error = (ms_within * 0.5 * (1.0 / n_i + 1.0 / n_j)).sqrt();
            if standard_error <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }

            let statistic = (mean_i - mean_j).abs() / standard_error;
            let p_value = 2.0
                * (1.0
                    - student_t_cdf(
                        statistic / 2.0_f64.sqrt(),
                        anova.degrees_of_freedom_within as f64,
                    )?);

            results.push(PairwiseComparisonResult {
                group_a: i,
                group_b: j,
                mean_difference: mean_i - mean_j,
                statistic,
                p_value,
            });
        }
    }

    Ok(results)
}

pub fn confidence_interval_from_sample(
    sample: &[f64],
    confidence_level: f64,
) -> SciResult<(f64, f64)> {
    if sample.len() < 2 {
        return Err(SciError::InvalidParameter(
            "sample confidence interval requires at least 2 observations",
        ));
    }
    let sample_mean = mean(sample)?;
    let sample_std_dev = variance(sample, true)?.sqrt();
    confidence_interval_mean_unknown_sigma(
        sample_mean,
        sample_std_dev,
        sample.len(),
        confidence_level,
    )
}

fn validate_two_samples(sample_a: &[f64], sample_b: &[f64]) -> SciResult<()> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err(SciError::InvalidParameter(
            "both samples must contain at least 2 observations",
        ));
    }
    Ok(())
}

fn within_group_mean_square(groups: &[&[f64]]) -> SciResult<f64> {
    let total_count: usize = groups.iter().map(|group| group.len()).sum();
    let ss_within: f64 = groups
        .iter()
        .map(|group| {
            let group_mean = mean(group).unwrap_or(0.0);
            group
                .iter()
                .map(|value| (value - group_mean).powi(2))
                .sum::<f64>()
        })
        .sum();
    let df_within = total_count - groups.len();
    if df_within == 0 {
        return Err(SciError::DivisionByZero);
    }
    Ok(ss_within / df_within as f64)
}
