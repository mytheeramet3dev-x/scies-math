use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use crate::probability::f_cdf;
use crate::statistics::mean;

#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeriesDecomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastMetrics {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArModel {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub noise_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArOrderCandidate {
    pub order: usize,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArOrderSelection {
    pub best_order: usize,
    pub candidates: Vec<ArOrderCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssRegression {
    Level,
    Trend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeriesTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub lags_used: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArmaModel {
    pub ar_coefficients: Vec<f64>,
    pub ma_window: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilterResult {
    pub filtered_state: Vec<f64>,
    pub predicted_state: Vec<f64>,
    pub innovations: Vec<f64>,
    pub innovation_variance: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChangePoint {
    pub index: usize,
    pub score: f64,
    pub mean_before: f64,
    pub mean_after: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyPoint {
    pub index: usize,
    pub value: f64,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Garch11Model {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub conditional_variance: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarModel {
    pub intercept: Vec<f64>,
    pub coefficients: DynamicMatrix,
    pub residual_covariance: DynamicMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CointegrationResult {
    pub intercept: f64,
    pub hedge_ratio: f64,
    pub residuals: Vec<f64>,
    pub adf_statistic: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrangerCausalityResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub restricted_rss: f64,
    pub unrestricted_rss: f64,
    pub lag_order: usize,
}

pub fn simple_moving_average(series: &[f64], window: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if window == 0 || window > series.len() {
        return Err(SciError::InvalidParameter(
            "window must be in 1..=series.len()",
        ));
    }

    let mut smoothed = Vec::with_capacity(series.len() - window + 1);
    for start in 0..=(series.len() - window) {
        smoothed.push(mean(&series[start..start + window])?);
    }
    Ok(smoothed)
}

pub fn exponential_moving_average(series: &[f64], alpha: f64) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    validate_alpha(alpha)?;

    let mut ema = Vec::with_capacity(series.len());
    let mut current = series[0];
    ema.push(current);

    for &value in &series[1..] {
        current = alpha * value + (1.0 - alpha) * current;
        ema.push(current);
    }

    Ok(ema)
}

pub fn simple_exponential_smoothing(series: &[f64], alpha: f64) -> SciResult<Vec<f64>> {
    exponential_moving_average(series, alpha)
}

pub fn difference(series: &[f64], lag: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if lag == 0 || lag >= series.len() {
        return Err(SciError::InvalidParameter("lag must be in 1..series.len()"));
    }

    Ok((lag..series.len())
        .map(|index| series[index] - series[index - lag])
        .collect())
}

pub fn autocorrelation(series: &[f64], lag: usize) -> SciResult<f64> {
    validate_series(series)?;
    if lag >= series.len() {
        return Err(SciError::InvalidParameter(
            "lag must be smaller than series length",
        ));
    }

    let mu = mean(series)?;
    let denominator: f64 = series.iter().map(|x| (x - mu).powi(2)).sum();
    if denominator.abs() <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let numerator: f64 = (lag..series.len())
        .map(|index| (series[index] - mu) * (series[index - lag] - mu))
        .sum();

    Ok(numerator / denominator)
}

pub fn partial_autocorrelation(series: &[f64], lag: usize) -> SciResult<f64> {
    validate_series(series)?;
    if lag == 0 || lag >= series.len() {
        return Err(SciError::InvalidParameter("lag must be in 1..series.len()"));
    }

    let mut autocovariances = Vec::with_capacity(lag + 1);
    autocovariances.push(1.0);
    for current_lag in 1..=lag {
        autocovariances.push(autocorrelation(series, current_lag)?);
    }

    let mut phi = vec![vec![0.0; lag + 1]; lag + 1];
    let mut variance = vec![0.0; lag + 1];
    variance[0] = 1.0;
    phi[1][1] = autocovariances[1];
    variance[1] = 1.0 - phi[1][1] * phi[1][1];

    for k in 2..=lag {
        let mut numerator = autocovariances[k];
        for j in 1..k {
            numerator -= phi[k - 1][j] * autocovariances[k - j];
        }
        if variance[k - 1].abs() <= f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        phi[k][k] = numerator / variance[k - 1];
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
        variance[k] = variance[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
    }

    Ok(phi[lag][lag])
}

pub fn seasonal_difference(series: &[f64], period: usize) -> SciResult<Vec<f64>> {
    difference(series, period)
}

pub fn cross_correlation(x: &[f64], y: &[f64], lag: isize) -> SciResult<f64> {
    validate_series(x)?;
    validate_series(y)?;
    if x.len() != y.len() {
        return Err(SciError::InvalidParameter("series length mismatch"));
    }
    let n = x.len();
    let shift = lag.unsigned_abs();
    if shift >= n {
        return Err(SciError::InvalidParameter(
            "absolute lag must be smaller than series length",
        ));
    }

    let x_mean = mean(x)?;
    let y_mean = mean(y)?;
    let x_var: f64 = x.iter().map(|value| (value - x_mean).powi(2)).sum();
    let y_var: f64 = y.iter().map(|value| (value - y_mean).powi(2)).sum();
    if x_var.abs() <= f64::EPSILON || y_var.abs() <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let numerator: f64 = if lag >= 0 {
        (shift..n)
            .map(|index| (x[index] - x_mean) * (y[index - shift] - y_mean))
            .sum()
    } else {
        (shift..n)
            .map(|index| (x[index - shift] - x_mean) * (y[index] - y_mean))
            .sum()
    };

    Ok(numerator / (x_var.sqrt() * y_var.sqrt()))
}

pub fn forecast_metrics(actual: &[f64], predicted: &[f64]) -> SciResult<ForecastMetrics> {
    validate_series(actual)?;
    validate_series(predicted)?;
    if actual.len() != predicted.len() {
        return Err(SciError::InvalidParameter(
            "actual/predicted length mismatch",
        ));
    }

    let n = actual.len() as f64;
    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut mape = 0.0;

    for (&actual_value, &predicted_value) in actual.iter().zip(predicted.iter()) {
        let error = actual_value - predicted_value;
        mae += error.abs();
        mse += error * error;
        if actual_value.abs() <= f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        mape += (error / actual_value).abs();
    }

    mae /= n;
    mse /= n;
    mape = 100.0 * mape / n;

    Ok(ForecastMetrics {
        mae,
        mse,
        rmse: mse.sqrt(),
        mape,
    })
}

pub fn moving_average_residuals(series: &[f64], window: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if window == 0 || window > series.len() {
        return Err(SciError::InvalidParameter(
            "window must be in 1..=series.len()",
        ));
    }

    let mut residuals = Vec::with_capacity(series.len());
    for index in 0..series.len() {
        let start = (index + 1).saturating_sub(window);
        let baseline = mean(&series[start..=index])?;
        residuals.push(series[index] - baseline);
    }
    Ok(residuals)
}

pub fn fit_ar_yule_walker(series: &[f64], order: usize) -> SciResult<ArModel> {
    validate_series(series)?;
    if order == 0 || order >= series.len() {
        return Err(SciError::InvalidParameter(
            "order must be in 1..series.len()",
        ));
    }

    let series_mean = mean(series)?;
    let centered: Vec<f64> = series.iter().map(|value| value - series_mean).collect();
    let gamma0: f64 =
        centered.iter().map(|value| value * value).sum::<f64>() / centered.len() as f64;

    if gamma0.abs() <= f64::EPSILON {
        return Ok(ArModel {
            intercept: series_mean,
            coefficients: vec![0.0; order],
            noise_variance: 0.0,
        });
    }

    let mut autocovariances = Vec::with_capacity(order + 1);
    for lag in 0..=order {
        let covariance = (lag..centered.len())
            .map(|index| centered[index] * centered[index - lag])
            .sum::<f64>()
            / centered.len() as f64;
        autocovariances.push(covariance);
    }

    let mut toeplitz = Vec::with_capacity(order * order);
    for row in 0..order {
        for col in 0..order {
            toeplitz.push(autocovariances[row.abs_diff(col)]);
        }
    }

    let matrix = DynamicMatrix::new(order, order, toeplitz)?;
    let rhs = autocovariances[1..].to_vec();
    let coefficients = matrix.solve_linear_system(&rhs)?;
    let intercept = series_mean * (1.0 - coefficients.iter().sum::<f64>());
    let explained: f64 = coefficients
        .iter()
        .zip(autocovariances.iter().skip(1))
        .map(|(phi, gamma)| phi * gamma)
        .sum();

    Ok(ArModel {
        intercept,
        coefficients,
        noise_variance: (gamma0 - explained).max(0.0),
    })
}

pub fn gaussian_log_likelihood_from_variance(sample_size: usize, variance: f64) -> SciResult<f64> {
    if sample_size == 0 {
        return Err(SciError::InvalidParameter("sample_size must be positive"));
    }
    if variance < 0.0 {
        return Err(SciError::InvalidParameter("variance must be non-negative"));
    }
    let safe_variance = variance.max(1e-12);
    let n = sample_size as f64;
    Ok(-0.5 * n * ((2.0 * core::f64::consts::PI).ln() + safe_variance.ln() + 1.0))
}

pub fn aic(log_likelihood: f64, parameter_count: usize) -> f64 {
    2.0 * parameter_count as f64 - 2.0 * log_likelihood
}

pub fn bic(log_likelihood: f64, parameter_count: usize, sample_size: usize) -> SciResult<f64> {
    if sample_size == 0 {
        return Err(SciError::InvalidParameter("sample_size must be positive"));
    }
    Ok((parameter_count as f64) * (sample_size as f64).ln() - 2.0 * log_likelihood)
}

pub fn select_best_ar_order(series: &[f64], max_order: usize) -> SciResult<ArOrderSelection> {
    validate_series(series)?;
    if max_order == 0 || max_order >= series.len() {
        return Err(SciError::InvalidParameter(
            "max_order must be in 1..series.len()",
        ));
    }

    let mut candidates = Vec::with_capacity(max_order);
    for order in 1..=max_order {
        let model = fit_ar_yule_walker(series, order)?;
        let residuals = model.in_sample_residuals(series)?;
        let variance =
            residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64;
        let log_likelihood = gaussian_log_likelihood_from_variance(residuals.len(), variance)?;
        let parameter_count = order + 2;
        candidates.push(ArOrderCandidate {
            order,
            aic: aic(log_likelihood, parameter_count),
            bic: bic(log_likelihood, parameter_count, residuals.len())?,
            log_likelihood,
        });
    }

    let best = candidates
        .iter()
        .min_by(|left, right| left.aic.total_cmp(&right.aic))
        .ok_or(SciError::EmptyInput)?;

    Ok(ArOrderSelection {
        best_order: best.order,
        candidates,
    })
}

pub fn fit_arma_like(series: &[f64], ar_order: usize, ma_window: usize) -> SciResult<ArmaModel> {
    validate_series(series)?;
    if ar_order == 0 || ar_order >= series.len() {
        return Err(SciError::InvalidParameter(
            "ar_order must be in 1..series.len()",
        ));
    }
    if ma_window == 0 || ma_window > series.len() {
        return Err(SciError::InvalidParameter(
            "ma_window must be in 1..=series.len()",
        ));
    }

    let ar_model = fit_ar_yule_walker(series, ar_order)?;
    Ok(ArmaModel {
        ar_coefficients: ar_model.coefficients,
        ma_window,
    })
}

pub fn forecast_arma_like(
    series: &[f64],
    ar_order: usize,
    ma_window: usize,
    horizon: usize,
) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    let model = fit_arma_like(series, ar_order, ma_window)?;
    model.forecast(series, horizon)
}

pub fn adf_test(series: &[f64], lag_order: usize) -> SciResult<TimeSeriesTestResult> {
    validate_series(series)?;
    if series.len() < lag_order + 4 {
        return Err(SciError::InvalidParameter(
            "series is too short for the requested ADF lag order",
        ));
    }

    let diffed = difference(series, 1)?;
    let observations = diffed.len() - lag_order;
    let predictors = 2 + lag_order;
    let mut design = Vec::with_capacity(observations * predictors);
    let mut target = Vec::with_capacity(observations);

    for t in lag_order..diffed.len() {
        target.push(diffed[t]);
        design.push(1.0);
        design.push(series[t]);
        for lag in 1..=lag_order {
            design.push(diffed[t - lag]);
        }
    }

    let x = DynamicMatrix::new(observations, predictors, design)?;
    let x_t = x.transpose();
    let xtx = x_t.mul_matrix(&x)?;
    let xtx_inv = xtx.inverse()?;
    let xty = x_t.mul_vector(&target)?;
    let beta = xtx_inv.mul_vector(&xty)?;
    let fitted = x.mul_vector(&beta)?;

    let rss: f64 = target
        .iter()
        .zip(fitted.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum();
    let degrees_of_freedom = observations as f64 - predictors as f64;
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::DivisionByZero);
    }
    let sigma2 = rss / degrees_of_freedom;
    let variance_beta = sigma2 * xtx_inv.get(1, 1)?;
    if variance_beta <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let statistic = beta[1] / variance_beta.sqrt();
    let p_value = approximate_adf_p_value(statistic);

    Ok(TimeSeriesTestResult {
        statistic,
        p_value,
        lags_used: lag_order,
    })
}

pub fn kpss_test(
    series: &[f64],
    regression: KpssRegression,
    truncation_lag: Option<usize>,
) -> SciResult<TimeSeriesTestResult> {
    validate_series(series)?;
    if series.len() < 3 {
        return Err(SciError::InvalidParameter(
            "KPSS test requires at least 3 observations",
        ));
    }

    let residuals = kpss_residuals(series, regression)?;
    let n = residuals.len();
    let bandwidth = truncation_lag
        .unwrap_or_else(|| (n as f64).sqrt().floor() as usize)
        .max(1);

    let mut cumulative = 0.0;
    let eta: f64 = residuals
        .iter()
        .map(|residual| {
            cumulative += residual;
            cumulative * cumulative
        })
        .sum::<f64>()
        / (n as f64 * n as f64);

    let gamma0: f64 = residuals
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        / n as f64;
    let mut long_run_variance = gamma0;
    for lag in 1..=bandwidth.min(n - 1) {
        let weight = 1.0 - lag as f64 / (bandwidth + 1) as f64;
        let gamma: f64 = (lag..n)
            .map(|index| residuals[index] * residuals[index - lag])
            .sum::<f64>()
            / n as f64;
        long_run_variance += 2.0 * weight * gamma;
    }
    if long_run_variance <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let statistic = eta / long_run_variance;
    let p_value = approximate_kpss_p_value(statistic, regression);

    Ok(TimeSeriesTestResult {
        statistic,
        p_value,
        lags_used: bandwidth,
    })
}

pub fn ljung_box_test(series: &[f64], max_lag: usize) -> SciResult<TimeSeriesTestResult> {
    validate_series(series)?;
    if max_lag == 0 || max_lag >= series.len() {
        return Err(SciError::InvalidParameter(
            "max_lag must be in 1..series.len()",
        ));
    }

    let n = series.len() as f64;
    let mut statistic = 0.0;
    for lag in 1..=max_lag {
        let rho = autocorrelation(series, lag)?;
        statistic += rho * rho / (n - lag as f64);
    }
    statistic *= n * (n + 2.0);

    let p_value = 1.0 - approximate_chi_square_cdf(statistic, max_lag as f64);
    Ok(TimeSeriesTestResult {
        statistic,
        p_value,
        lags_used: max_lag,
    })
}

pub fn kalman_filter_local_level(
    observations: &[f64],
    initial_state: f64,
    initial_variance: f64,
    process_variance: f64,
    measurement_variance: f64,
) -> SciResult<KalmanFilterResult> {
    validate_series(observations)?;
    if initial_variance < 0.0 || process_variance < 0.0 || measurement_variance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "variances must be non-negative and measurement_variance positive",
        ));
    }

    let mut state = initial_state;
    let mut variance = initial_variance;
    let mut filtered_state = Vec::with_capacity(observations.len());
    let mut predicted_state = Vec::with_capacity(observations.len());
    let mut innovations = Vec::with_capacity(observations.len());
    let mut innovation_variance = Vec::with_capacity(observations.len());

    for &observation in observations {
        let predicted = state;
        let predicted_variance = variance + process_variance;
        let innovation = observation - predicted;
        let innovation_var = predicted_variance + measurement_variance;
        let kalman_gain = predicted_variance / innovation_var;

        state = predicted + kalman_gain * innovation;
        variance = (1.0 - kalman_gain) * predicted_variance;

        predicted_state.push(predicted);
        filtered_state.push(state);
        innovations.push(innovation);
        innovation_variance.push(innovation_var);
    }

    Ok(KalmanFilterResult {
        filtered_state,
        predicted_state,
        innovations,
        innovation_variance,
    })
}

pub fn kalman_smooth_local_level(
    observations: &[f64],
    initial_state: f64,
    initial_variance: f64,
    process_variance: f64,
    measurement_variance: f64,
) -> SciResult<Vec<f64>> {
    validate_series(observations)?;
    if initial_variance < 0.0 || process_variance < 0.0 || measurement_variance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "variances must be non-negative and measurement_variance positive",
        ));
    }

    let mut filtered_state = Vec::with_capacity(observations.len());
    let mut filtered_variance = Vec::with_capacity(observations.len());
    let mut predicted_variance = Vec::with_capacity(observations.len());

    let mut state = initial_state;
    let mut variance = initial_variance;
    for &observation in observations {
        let predicted_variance_t = variance + process_variance;
        let innovation_variance = predicted_variance_t + measurement_variance;
        let kalman_gain = predicted_variance_t / innovation_variance;
        state = state + kalman_gain * (observation - state);
        variance = (1.0 - kalman_gain) * predicted_variance_t;

        filtered_state.push(state);
        filtered_variance.push(variance);
        predicted_variance.push(predicted_variance_t);
    }

    let mut smoothed = filtered_state.clone();
    if smoothed.len() > 1 {
        for index in (0..smoothed.len() - 1).rev() {
            let smoother_gain = filtered_variance[index] / predicted_variance[index + 1];
            smoothed[index] += smoother_gain * (smoothed[index + 1] - filtered_state[index]);
        }
    }

    Ok(smoothed)
}

pub fn detect_mean_shift_change_point(
    series: &[f64],
    min_segment_size: usize,
) -> SciResult<ChangePoint> {
    validate_series(series)?;
    if min_segment_size == 0 || 2 * min_segment_size >= series.len() {
        return Err(SciError::InvalidParameter(
            "min_segment_size must allow at least two non-empty segments",
        ));
    }

    let mut best: Option<ChangePoint> = None;
    for split in min_segment_size..=(series.len() - min_segment_size) {
        let before = &series[..split];
        let after = &series[split..];
        let mean_before = mean(before)?;
        let mean_after = mean(after)?;
        let score = (mean_after - mean_before).abs()
            * ((before.len() * after.len()) as f64 / series.len() as f64).sqrt();

        let candidate = ChangePoint {
            index: split,
            score,
            mean_before,
            mean_after,
        };

        if best
            .as_ref()
            .is_none_or(|current| candidate.score > current.score)
        {
            best = Some(candidate);
        }
    }

    best.ok_or(SciError::EmptyInput)
}

pub fn rolling_z_score_anomalies(
    series: &[f64],
    window: usize,
    z_threshold: f64,
) -> SciResult<Vec<AnomalyPoint>> {
    validate_series(series)?;
    if window < 2 || window > series.len() {
        return Err(SciError::InvalidParameter(
            "window must be in 2..=series.len()",
        ));
    }
    if z_threshold <= 0.0 {
        return Err(SciError::InvalidParameter("z_threshold must be positive"));
    }

    let mut anomalies = Vec::new();
    for index in (window - 1)..series.len() {
        let window_slice = &series[index + 1 - window..=index];
        let mu = mean(window_slice)?;
        let variance = window_slice
            .iter()
            .map(|value| (value - mu).powi(2))
            .sum::<f64>()
            / window_slice.len() as f64;
        let sigma = variance.sqrt();
        if sigma <= f64::EPSILON {
            continue;
        }
        let score = (series[index] - mu) / sigma;
        if score.abs() >= z_threshold {
            anomalies.push(AnomalyPoint {
                index,
                value: series[index],
                score,
            });
        }
    }
    Ok(anomalies)
}

pub fn ewma_anomalies(
    series: &[f64],
    lambda: f64,
    sigma_multiplier: f64,
) -> SciResult<Vec<AnomalyPoint>> {
    validate_series(series)?;
    validate_alpha(lambda)?;
    if sigma_multiplier <= 0.0 {
        return Err(SciError::InvalidParameter(
            "sigma_multiplier must be positive",
        ));
    }

    let mu = mean(series)?;
    let variance =
        series.iter().map(|value| (value - mu).powi(2)).sum::<f64>() / series.len() as f64;
    let sigma = variance.sqrt();
    if sigma <= f64::EPSILON {
        return Ok(Vec::new());
    }

    let mut ewma = series[0];
    let mut anomalies = Vec::new();
    for (index, &value) in series.iter().enumerate() {
        ewma = lambda * value + (1.0 - lambda) * ewma;
        let sigma_ewma = sigma
            * (lambda / (2.0 - lambda) * (1.0 - (1.0 - lambda).powi(2 * (index as i32 + 1))))
                .sqrt();
        if sigma_ewma <= f64::EPSILON {
            continue;
        }
        let score = (value - ewma) / sigma_ewma;
        if score.abs() >= sigma_multiplier {
            anomalies.push(AnomalyPoint {
                index,
                value,
                score,
            });
        }
    }
    Ok(anomalies)
}

pub fn fit_garch11(returns: &[f64]) -> SciResult<Garch11Model> {
    validate_series(returns)?;
    if returns.len() < 3 {
        return Err(SciError::InvalidParameter(
            "GARCH(1,1) requires at least 3 observations",
        ));
    }

    let mu = mean(returns)?;
    let centered: Vec<f64> = returns.iter().map(|value| value - mu).collect();
    let variance = centered.iter().map(|value| value * value).sum::<f64>() / centered.len() as f64;
    if variance <= f64::EPSILON {
        return Ok(Garch11Model {
            omega: 0.0,
            alpha: 0.0,
            beta: 0.0,
            conditional_variance: vec![0.0; returns.len()],
        });
    }

    let squared: Vec<f64> = centered.iter().map(|value| value * value).collect();
    let lag1_corr = if squared.len() > 1 {
        autocorrelation(&squared, 1).unwrap_or(0.0).max(0.0)
    } else {
        0.0
    };
    let alpha = (0.15 + 0.35 * lag1_corr).clamp(0.05, 0.3);
    let beta = (0.75 - 0.25 * lag1_corr).clamp(0.5, 0.93);
    let persistence = (alpha + beta).min(0.98);
    let adjusted_beta = persistence - alpha;
    let omega = variance * (1.0 - persistence).max(1e-6);

    let mut conditional_variance = Vec::with_capacity(returns.len());
    conditional_variance.push(variance);
    for t in 1..returns.len() {
        let next = omega + alpha * squared[t - 1] + adjusted_beta * conditional_variance[t - 1];
        conditional_variance.push(next.max(1e-12));
    }

    Ok(Garch11Model {
        omega,
        alpha,
        beta: adjusted_beta,
        conditional_variance,
    })
}

pub fn forecast_garch_variance(returns: &[f64], horizon: usize) -> SciResult<Vec<f64>> {
    validate_series(returns)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    let model = fit_garch11(returns)?;
    if model.conditional_variance.is_empty() {
        return Ok(vec![0.0; horizon]);
    }

    let long_run = if (1.0 - model.alpha - model.beta).abs() <= f64::EPSILON {
        *model.conditional_variance.last().unwrap_or(&0.0)
    } else {
        model.omega / (1.0 - model.alpha - model.beta)
    };
    let mut current = *model.conditional_variance.last().unwrap_or(&0.0);
    let mut forecast = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        current = model.omega + (model.alpha + model.beta) * current;
        current = 0.5 * current + 0.5 * long_run;
        forecast.push(current.max(1e-12));
    }
    Ok(forecast)
}

pub fn covariance_matrix(series_set: &[Vec<f64>]) -> SciResult<DynamicMatrix> {
    validate_multivariate_series(series_set)?;
    let dimension = series_set.len();
    let length = series_set[0].len();
    let means: Vec<f64> = series_set
        .iter()
        .map(|series| mean(series))
        .collect::<SciResult<Vec<_>>>()?;

    let mut values = Vec::with_capacity(dimension * dimension);
    for i in 0..dimension {
        for j in 0..dimension {
            let covariance = series_set[i]
                .iter()
                .zip(series_set[j].iter())
                .map(|(a, b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>()
                / (length - 1) as f64;
            values.push(covariance);
        }
    }

    DynamicMatrix::new(dimension, dimension, values)
}

pub fn correlation_matrix(series_set: &[Vec<f64>]) -> SciResult<DynamicMatrix> {
    let covariance = covariance_matrix(series_set)?;
    let dimension = covariance.rows();
    let mut values = Vec::with_capacity(dimension * dimension);
    let mut std = Vec::with_capacity(dimension);
    for i in 0..dimension {
        std.push(covariance.get(i, i)?.sqrt());
    }
    for i in 0..dimension {
        for j in 0..dimension {
            if std[i] <= f64::EPSILON || std[j] <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            values.push(covariance.get(i, j)? / (std[i] * std[j]));
        }
    }
    DynamicMatrix::new(dimension, dimension, values)
}

pub fn fit_var1(series_set: &[Vec<f64>]) -> SciResult<VarModel> {
    validate_multivariate_series(series_set)?;
    let dimension = series_set.len();
    let length = series_set[0].len();
    if length < 3 {
        return Err(SciError::InvalidParameter(
            "VAR(1) requires at least 3 observations per series",
        ));
    }

    let rows = length - 1;
    let predictors = dimension + 1;
    let mut design = Vec::with_capacity(rows * predictors);
    for t in 1..length {
        design.push(1.0);
        for series in series_set {
            design.push(series[t - 1]);
        }
    }
    let x = DynamicMatrix::new(rows, predictors, design)?;
    let x_t = x.transpose();
    let xtx = x_t.mul_matrix(&x)?;
    let xtx_inv = xtx.inverse()?;

    let mut intercept = Vec::with_capacity(dimension);
    let mut coefficient_values = vec![0.0; dimension * dimension];
    let mut residual_series = Vec::with_capacity(dimension);

    for (target_index, target_series) in series_set.iter().enumerate() {
        let y = target_series[1..].to_vec();
        let xty = x_t.mul_vector(&y)?;
        let beta = xtx_inv.mul_vector(&xty)?;
        intercept.push(beta[0]);
        for source_index in 0..dimension {
            coefficient_values[target_index * dimension + source_index] = beta[source_index + 1];
        }

        let fitted = x.mul_vector(&beta)?;
        residual_series.push(
            y.iter()
                .zip(fitted.iter())
                .map(|(actual, predicted)| actual - predicted)
                .collect::<Vec<_>>(),
        );
    }

    let residual_covariance = covariance_matrix(&residual_series)?;
    Ok(VarModel {
        intercept,
        coefficients: DynamicMatrix::new(dimension, dimension, coefficient_values)?,
        residual_covariance,
    })
}

pub fn forecast_var1(series_set: &[Vec<f64>], horizon: usize) -> SciResult<Vec<Vec<f64>>> {
    validate_multivariate_series(series_set)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    let model = fit_var1(series_set)?;
    model.forecast(series_set, horizon)
}

pub fn engle_granger_cointegration_test(x: &[f64], y: &[f64]) -> SciResult<CointegrationResult> {
    validate_series(x)?;
    validate_series(y)?;
    if x.len() != y.len() || x.len() < 4 {
        return Err(SciError::InvalidParameter(
            "x and y must have the same length and at least 4 observations",
        ));
    }

    let rows = x.len();
    let design = x.iter().flat_map(|value| [1.0, *value]).collect::<Vec<_>>();
    let x_matrix = DynamicMatrix::new(rows, 2, design)?;
    let x_t = x_matrix.transpose();
    let beta = x_t
        .mul_matrix(&x_matrix)?
        .inverse()?
        .mul_vector(&x_t.mul_vector(y)?)?;

    let intercept = beta[0];
    let hedge_ratio = beta[1];
    let residuals = x
        .iter()
        .zip(y.iter())
        .map(|(xv, yv)| yv - (intercept + hedge_ratio * xv))
        .collect::<Vec<_>>();
    let adf = if residuals.iter().all(|value| value.abs() <= 1e-12) {
        TimeSeriesTestResult {
            statistic: f64::NEG_INFINITY,
            p_value: 0.0,
            lags_used: 1,
        }
    } else {
        adf_test(&residuals, 1)?
    };

    Ok(CointegrationResult {
        intercept,
        hedge_ratio,
        residuals,
        adf_statistic: adf.statistic,
        p_value: adf.p_value,
    })
}

pub fn granger_causality_test(
    cause: &[f64],
    effect: &[f64],
    lag_order: usize,
) -> SciResult<GrangerCausalityResult> {
    validate_series(cause)?;
    validate_series(effect)?;
    if cause.len() != effect.len() || cause.len() <= 2 * lag_order + 1 || lag_order == 0 {
        return Err(SciError::InvalidParameter(
            "cause/effect must have the same length and be long enough for lag_order",
        ));
    }

    let observations = effect.len() - lag_order;
    let unrestricted_predictors = 1 + 2 * lag_order;
    let restricted_predictors = 1 + lag_order;
    let target = effect[lag_order..].to_vec();

    let mut unrestricted_design = Vec::with_capacity(observations * unrestricted_predictors);
    let mut restricted_design = Vec::with_capacity(observations * restricted_predictors);
    for t in lag_order..effect.len() {
        unrestricted_design.push(1.0);
        restricted_design.push(1.0);
        for lag in 1..=lag_order {
            unrestricted_design.push(effect[t - lag]);
            restricted_design.push(effect[t - lag]);
        }
        for lag in 1..=lag_order {
            unrestricted_design.push(cause[t - lag]);
        }
    }

    let unrestricted_x =
        DynamicMatrix::new(observations, unrestricted_predictors, unrestricted_design)?;
    let restricted_x = DynamicMatrix::new(observations, restricted_predictors, restricted_design)?;
    let unrestricted_rss = ols_rss(&unrestricted_x, &target)?;
    let restricted_rss = ols_rss(&restricted_x, &target)?;

    let restrictions = lag_order as f64;
    let denominator_df = observations as f64 - unrestricted_predictors as f64;
    if denominator_df <= 0.0 || unrestricted_rss <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let f_statistic =
        ((restricted_rss - unrestricted_rss) / restrictions) / (unrestricted_rss / denominator_df);
    let p_value = 1.0 - f_cdf(f_statistic.max(0.0), restrictions, denominator_df)?;

    Ok(GrangerCausalityResult {
        f_statistic,
        p_value,
        restricted_rss,
        unrestricted_rss,
        lag_order,
    })
}

fn ols_rss(design: &DynamicMatrix, target: &[f64]) -> SciResult<f64> {
    let x_t = design.transpose();
    let beta = x_t
        .mul_matrix(design)?
        .inverse()?
        .mul_vector(&x_t.mul_vector(target)?)?;
    let fitted = design.mul_vector(&beta)?;
    Ok(target
        .iter()
        .zip(fitted.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum())
}

pub fn forecast_ar(series: &[f64], order: usize, horizon: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    let model = fit_ar_yule_walker(series, order)?;
    model.forecast(series, horizon)
}

pub fn forecast_arima(
    series: &[f64],
    ar_order: usize,
    difference_order: usize,
    horizon: usize,
) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }
    if difference_order >= series.len() {
        return Err(SciError::InvalidParameter(
            "difference_order must be smaller than series length",
        ));
    }

    let mut differenced = series.to_vec();
    let mut histories = Vec::with_capacity(difference_order);
    for _ in 0..difference_order {
        histories.push(differenced.clone());
        differenced = difference(&differenced, 1)?;
    }

    let differenced_forecast = if difference_order == 0 {
        forecast_ar(&differenced, ar_order, horizon)?
    } else if differenced
        .iter()
        .all(|value| (*value - differenced[0]).abs() < 1e-12)
    {
        vec![differenced[0]; horizon]
    } else {
        forecast_ar(&differenced, ar_order, horizon)?
    };

    let mut reconstructed = differenced_forecast;
    for history in histories.iter().rev() {
        let mut last = *history.last().ok_or(SciError::EmptyInput)?;
        let mut integrated = Vec::with_capacity(horizon);
        for &step in &reconstructed {
            last += step;
            integrated.push(last);
        }
        reconstructed = integrated;
    }

    Ok(reconstructed)
}

pub fn forecast_sarima(
    series: &[f64],
    ar_order: usize,
    difference_order: usize,
    seasonal_ar_order: usize,
    seasonal_period: usize,
    seasonal_difference_order: usize,
    horizon: usize,
) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }
    if seasonal_difference_order > 0 && seasonal_period < 2 {
        return Err(SciError::InvalidParameter(
            "seasonal_period must be at least 2 when seasonal differencing is used",
        ));
    }
    if ar_order > 0 && seasonal_ar_order > 0 {
        return Err(SciError::InvalidParameter(
            "combined seasonal and non-seasonal AR order fitting is not yet supported",
        ));
    }

    let total_order = ar_order.max(seasonal_ar_order);
    let mut transformed = series.to_vec();
    let mut seasonal_histories = Vec::with_capacity(seasonal_difference_order);
    for _ in 0..seasonal_difference_order {
        seasonal_histories.push(transformed.clone());
        transformed = seasonal_difference(&transformed, seasonal_period)?;
    }

    let mut regular_histories = Vec::with_capacity(difference_order);
    for _ in 0..difference_order {
        regular_histories.push(transformed.clone());
        transformed = difference(&transformed, 1)?;
    }

    let transformed_forecast = if total_order == 0 {
        vec![*transformed.last().ok_or(SciError::EmptyInput)?; horizon]
    } else if transformed
        .iter()
        .all(|value| (*value - transformed[0]).abs() < 1e-12)
    {
        vec![transformed[0]; horizon]
    } else {
        forecast_ar(&transformed, total_order, horizon)?
    };

    let mut reconstructed = transformed_forecast;
    for history in regular_histories.iter().rev() {
        let mut last = *history.last().ok_or(SciError::EmptyInput)?;
        let mut integrated = Vec::with_capacity(horizon);
        for &step in &reconstructed {
            last += step;
            integrated.push(last);
        }
        reconstructed = integrated;
    }

    for history in seasonal_histories.iter().rev() {
        let mut extended = history.clone();
        let mut integrated = Vec::with_capacity(horizon);
        for &step in &reconstructed {
            let seasonal_base = extended[extended.len() - seasonal_period];
            let next = seasonal_base + step;
            extended.push(next);
            integrated.push(next);
        }
        reconstructed = integrated;
    }

    Ok(reconstructed)
}

pub fn naive_forecast(series: &[f64], horizon: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    Ok(vec![series[series.len() - 1]; horizon])
}

pub fn drift_forecast(series: &[f64], horizon: usize) -> SciResult<Vec<f64>> {
    validate_series(series)?;
    if series.len() < 2 {
        return Err(SciError::InvalidParameter(
            "drift forecast requires at least 2 observations",
        ));
    }
    if horizon == 0 {
        return Err(SciError::InvalidParameter("horizon must be positive"));
    }

    let drift = (series[series.len() - 1] - series[0]) / (series.len() - 1) as f64;
    let last = series[series.len() - 1];
    Ok((1..=horizon)
        .map(|step| last + drift * step as f64)
        .collect())
}

pub fn holt_linear_trend(
    series: &[f64],
    alpha: f64,
    beta: f64,
    forecast_horizon: usize,
) -> SciResult<(Vec<f64>, Vec<f64>)> {
    validate_series(series)?;
    if series.len() < 2 {
        return Err(SciError::InvalidParameter(
            "Holt linear trend requires at least 2 observations",
        ));
    }
    validate_alpha(alpha)?;
    validate_alpha(beta)?;
    if forecast_horizon == 0 {
        return Err(SciError::InvalidParameter(
            "forecast_horizon must be positive",
        ));
    }

    let mut level = series[0];
    let mut trend = series[1] - series[0];
    let mut fitted = Vec::with_capacity(series.len());
    fitted.push(level);

    for &value in &series[1..] {
        let previous_level = level;
        level = alpha * value + (1.0 - alpha) * (level + trend);
        trend = beta * (level - previous_level) + (1.0 - beta) * trend;
        fitted.push(level + trend);
    }

    let forecast = (1..=forecast_horizon)
        .map(|step| level + step as f64 * trend)
        .collect();
    Ok((fitted, forecast))
}

pub fn seasonal_decompose_additive(
    series: &[f64],
    period: usize,
) -> SciResult<TimeSeriesDecomposition> {
    validate_series(series)?;
    if period < 2 {
        return Err(SciError::InvalidParameter("period must be at least 2"));
    }
    if series.len() < period {
        return Err(SciError::InvalidParameter(
            "series length must be at least one full period",
        ));
    }

    let trend = centered_moving_average(series, period)?;
    let detrended: Vec<f64> = series
        .iter()
        .zip(trend.iter())
        .map(|(value, trend_value)| value - trend_value)
        .collect();

    let mut seasonal_pattern = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for (index, &value) in detrended.iter().enumerate() {
        seasonal_pattern[index % period] += value;
        counts[index % period] += 1;
    }
    for index in 0..period {
        seasonal_pattern[index] /= counts[index] as f64;
    }

    let seasonal_mean = mean(&seasonal_pattern)?;
    for value in &mut seasonal_pattern {
        *value -= seasonal_mean;
    }

    let seasonal: Vec<f64> = (0..series.len())
        .map(|index| seasonal_pattern[index % period])
        .collect();
    let residual: Vec<f64> = series
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((value, trend_value), seasonal_value)| value - trend_value - seasonal_value)
        .collect();

    Ok(TimeSeriesDecomposition {
        trend,
        seasonal,
        residual,
    })
}

fn centered_moving_average(series: &[f64], period: usize) -> SciResult<Vec<f64>> {
    let base = simple_moving_average(series, period)?;
    if period % 2 == 1 {
        let offset = period / 2;
        let mut trend = vec![base[0]; series.len()];
        for (index, &value) in base.iter().enumerate() {
            trend[index + offset] = value;
        }
        for index in 0..offset {
            trend[index] = trend[offset];
        }
        for index in (series.len() - offset)..series.len() {
            trend[index] = trend[series.len() - offset - 1];
        }
        return Ok(trend);
    }

    let centered: Vec<f64> = base
        .windows(2)
        .map(|window| 0.5 * (window[0] + window[1]))
        .collect();
    let left_pad = period / 2;
    let mut trend = vec![centered[0]; series.len()];
    for (index, &value) in centered.iter().enumerate() {
        trend[index + left_pad] = value;
    }
    for index in 0..left_pad {
        trend[index] = trend[left_pad];
    }
    for index in (left_pad + centered.len())..series.len() {
        trend[index] = trend[left_pad + centered.len() - 1];
    }
    Ok(trend)
}

fn validate_series(series: &[f64]) -> SciResult<()> {
    if series.is_empty() {
        return Err(SciError::EmptyInput);
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> SciResult<()> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(SciError::InvalidParameter("alpha must be in [0, 1]"));
    }
    Ok(())
}

fn validate_multivariate_series(series_set: &[Vec<f64>]) -> SciResult<()> {
    if series_set.is_empty() {
        return Err(SciError::EmptyInput);
    }
    let length = series_set[0].len();
    if length < 2 {
        return Err(SciError::InvalidParameter(
            "each series must contain at least 2 observations",
        ));
    }
    if series_set
        .iter()
        .any(|series| series.len() != length || series.is_empty())
    {
        return Err(SciError::InvalidParameter(
            "all series must be non-empty and have the same length",
        ));
    }
    Ok(())
}

fn kpss_residuals(series: &[f64], regression: KpssRegression) -> SciResult<Vec<f64>> {
    match regression {
        KpssRegression::Level => {
            let mu = mean(series)?;
            Ok(series.iter().map(|value| value - mu).collect())
        }
        KpssRegression::Trend => {
            let n = series.len();
            let x: Vec<f64> = (0..n).map(|index| index as f64).collect();
            let x_mean = mean(&x)?;
            let y_mean = mean(series)?;
            let ss_xx: f64 = x.iter().map(|value| (value - x_mean).powi(2)).sum();
            if ss_xx <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            let ss_xy: f64 = x
                .iter()
                .zip(series.iter())
                .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
                .sum();
            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;
            Ok(x.iter()
                .zip(series.iter())
                .map(|(xi, yi)| yi - (intercept + slope * xi))
                .collect())
        }
    }
}

fn approximate_adf_p_value(statistic: f64) -> f64 {
    let criticals = [
        (-4.0, 0.01),
        (-3.43, 0.05),
        (-2.86, 0.10),
        (-2.57, 0.20),
        (-1.95, 0.50),
    ];
    interpolate_p_value_from_criticals(statistic, &criticals, true)
}

fn approximate_kpss_p_value(statistic: f64, regression: KpssRegression) -> f64 {
    let criticals = match regression {
        KpssRegression::Level => [(0.739, 0.01), (0.574, 0.025), (0.463, 0.05), (0.347, 0.10)],
        KpssRegression::Trend => [(0.216, 0.01), (0.176, 0.025), (0.146, 0.05), (0.119, 0.10)],
    };
    interpolate_p_value_from_criticals(statistic, &criticals, false)
}

fn approximate_chi_square_cdf(x: f64, degrees_of_freedom: f64) -> f64 {
    if x <= 0.0 || degrees_of_freedom <= 0.0 {
        return 0.0;
    }
    let z = ((x / degrees_of_freedom).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * degrees_of_freedom)))
        / (2.0 / (9.0 * degrees_of_freedom)).sqrt();
    0.5 * (1.0 + erf_approx(z / 2.0_f64.sqrt()))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t)
            * (-ax * ax).exp();
    sign * y
}

fn interpolate_p_value_from_criticals(
    statistic: f64,
    criticals: &[(f64, f64)],
    lower_is_stronger: bool,
) -> f64 {
    if lower_is_stronger {
        if statistic <= criticals[0].0 {
            return criticals[0].1;
        }
        for window in criticals.windows(2) {
            let (x0, p0) = window[0];
            let (x1, p1) = window[1];
            if statistic <= x1 {
                let weight = (statistic - x0) / (x1 - x0);
                return p0 + weight * (p1 - p0);
            }
        }
        0.5
    } else {
        if statistic >= criticals[0].0 {
            return criticals[0].1;
        }
        for window in criticals.windows(2) {
            let (x0, p0) = window[0];
            let (x1, p1) = window[1];
            if statistic >= x1 {
                let weight = (statistic - x0) / (x1 - x0);
                return p0 + weight * (p1 - p0);
            }
        }
        0.2
    }
}

impl ArModel {
    pub fn in_sample_residuals(&self, series: &[f64]) -> SciResult<Vec<f64>> {
        let order = self.coefficients.len();
        if series.len() <= order {
            return Err(SciError::InvalidParameter(
                "series must be longer than model order",
            ));
        }

        let mut residuals = Vec::with_capacity(series.len() - order);
        for index in order..series.len() {
            let prediction = self.predict_next(&series[..index])?;
            residuals.push(series[index] - prediction);
        }
        Ok(residuals)
    }

    pub fn predict_next(&self, history: &[f64]) -> SciResult<f64> {
        if history.len() < self.coefficients.len() {
            return Err(SciError::InvalidParameter(
                "history is shorter than model order",
            ));
        }

        let mut prediction = self.intercept;
        for (index, coefficient) in self.coefficients.iter().enumerate() {
            let value = history[history.len() - 1 - index];
            prediction += coefficient * value;
        }
        Ok(prediction)
    }

    pub fn forecast(&self, history: &[f64], horizon: usize) -> SciResult<Vec<f64>> {
        if horizon == 0 {
            return Err(SciError::InvalidParameter("horizon must be positive"));
        }
        if history.len() < self.coefficients.len() {
            return Err(SciError::InvalidParameter(
                "history is shorter than model order",
            ));
        }

        let mut values = history.to_vec();
        let mut forecast = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next = self.predict_next(&values)?;
            values.push(next);
            forecast.push(next);
        }
        Ok(forecast)
    }
}

impl ArmaModel {
    pub fn forecast(&self, history: &[f64], horizon: usize) -> SciResult<Vec<f64>> {
        if horizon == 0 {
            return Err(SciError::InvalidParameter("horizon must be positive"));
        }
        if history.len() < self.ar_coefficients.len() {
            return Err(SciError::InvalidParameter(
                "history is shorter than AR order",
            ));
        }

        let mut values = history.to_vec();
        let mut forecast = Vec::with_capacity(horizon);
        let mut residuals = moving_average_residuals(history, self.ma_window)?;
        for _ in 0..horizon {
            let mut next = 0.0;
            for (index, coefficient) in self.ar_coefficients.iter().enumerate() {
                next += coefficient * values[values.len() - 1 - index];
            }
            let ma_adjustment = if residuals.is_empty() {
                0.0
            } else {
                let start = residuals.len().saturating_sub(self.ma_window);
                mean(&residuals[start..])?
            };
            next += ma_adjustment;
            values.push(next);
            residuals.push(0.0);
            forecast.push(next);
        }
        Ok(forecast)
    }
}

impl VarModel {
    pub fn forecast(&self, history: &[Vec<f64>], horizon: usize) -> SciResult<Vec<Vec<f64>>> {
        validate_multivariate_series(history)?;
        if horizon == 0 {
            return Err(SciError::InvalidParameter("horizon must be positive"));
        }
        let dimension = self.intercept.len();
        if history.len() != dimension || self.coefficients.rows() != dimension {
            return Err(SciError::InvalidParameter(
                "history dimension does not match VAR model",
            ));
        }

        let mut current = history
            .iter()
            .map(|series| series.last().copied().ok_or(SciError::EmptyInput))
            .collect::<SciResult<Vec<_>>>()?;
        let mut outputs = vec![Vec::with_capacity(horizon); dimension];

        for _ in 0..horizon {
            let mut next = self.intercept.clone();
            let contribution = self.coefficients.mul_vector(&current)?;
            for index in 0..dimension {
                next[index] += contribution[index];
                outputs[index].push(next[index]);
            }
            current = next;
        }

        Ok(outputs)
    }

    pub fn impulse_response(&self, shock: &[f64], steps: usize) -> SciResult<Vec<Vec<f64>>> {
        if steps == 0 {
            return Err(SciError::InvalidParameter("steps must be positive"));
        }
        if shock.len() != self.intercept.len() {
            return Err(SciError::InvalidParameter(
                "shock dimension does not match VAR model",
            ));
        }

        let dimension = shock.len();
        let mut current = shock.to_vec();
        let mut responses = vec![Vec::with_capacity(steps); dimension];
        for _ in 0..steps {
            for index in 0..dimension {
                responses[index].push(current[index]);
            }
            current = self.coefficients.mul_vector(&current)?;
        }
        Ok(responses)
    }
}
