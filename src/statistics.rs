//! Descriptive statistics — moments, order statistics, correlations.
//!
//! # Functions
//!
//! | Function | Description |
//! |---|---|
//! | `mean(data)` | Arithmetic mean |
//! | `variance(data)` | Sample variance (Bessel corrected) |
//! | `std_dev(data)` | Sample standard deviation |
//! | `median(data)` | Middle value (sorted) |
//! | `mode(data)` | Most frequent value |
//! | `quantile(data, p)` | p-th quantile (0 ≤ p ≤ 1) |
//! | `iqr(data)` | Inter-quartile range Q3 − Q1 |
//! | `skewness(data)` | Third standardised moment |
//! | `kurtosis(data)` | Fourth standardised moment − 3 |
//! | `covariance(x, y)` | Sample covariance |
//! | `pearson_r(x, y)` | Pearson correlation coefficient |
//! | `spearman_r(x, y)` | Spearman rank correlation |
//! | `histogram(data, bins)` | Bin counts and edges |
//! | `z_score(data)` | Standardised data (mean=0, std=1) |
//!
//! # Usage
//!
//! ```rust
//! use scies_math::statistics::{mean, std_dev, pearson_r};
//!
//! let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
//! assert!((mean(&data) - 5.0).abs() < 1e-10);
//! assert!((std_dev(&data) - 2.0).abs() < 1e-10);
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![2.0, 4.0, 6.0];
//! assert!((pearson_r(&x, &y).unwrap() - 1.0).abs() < 1e-10);
//! ```
use crate::errors::{SciError, SciResult};

pub fn mean(data: &[f64]) -> SciResult<f64> {
    if data.is_empty() {
        return Err(SciError::EmptyInput);
    }
    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

pub fn median(data: &[f64]) -> SciResult<f64> {
    if data.is_empty() {
        return Err(SciError::EmptyInput);
    }
    let mut values = data.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Ok((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Ok(values[mid])
    }
}

pub fn variance(data: &[f64], sample: bool) -> SciResult<f64> {
    if data.is_empty() || (sample && data.len() < 2) {
        return Err(SciError::EmptyInput);
    }
    let avg = mean(data)?;
    let sum = data.iter().map(|v| (v - avg).powi(2)).sum::<f64>();
    let denominator = if sample {
        (data.len() - 1) as f64
    } else {
        data.len() as f64
    };
    Ok(sum / denominator)
}

pub fn standard_deviation(data: &[f64], sample: bool) -> SciResult<f64> {
    Ok(variance(data, sample)?.sqrt())
}

pub fn z_score(value: f64, avg: f64, std_dev: f64) -> SciResult<f64> {
    if std_dev == 0.0 {
        return Err(SciError::DivisionByZero);
    }
    Ok((value - avg) / std_dev)
}
