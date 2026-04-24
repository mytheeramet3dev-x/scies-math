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
