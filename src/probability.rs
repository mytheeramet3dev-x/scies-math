use crate::errors::{SciError, SciResult};

pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> SciResult<f64> {
    if std_dev <= 0.0 {
        return Err(SciError::InvalidParameter(
            "standard deviation must be positive",
        ));
    }

    let variance = std_dev * std_dev;
    let z = x - mean;
    Ok((1.0 / (std_dev * (2.0 * core::f64::consts::PI).sqrt()))
        * (-(z * z) / (2.0 * variance)).exp())
}

pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> SciResult<f64> {
    if std_dev <= 0.0 {
        return Err(SciError::InvalidParameter(
            "standard deviation must be positive",
        ));
    }

    let z = (x - mean) / (std_dev * 2.0_f64.sqrt());
    Ok(0.5 * (1.0 + erf(z)))
}

pub fn student_t_pdf(x: f64, degrees_of_freedom: f64) -> SciResult<f64> {
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }

    let numerator = gamma((degrees_of_freedom + 1.0) / 2.0);
    let denominator =
        (degrees_of_freedom * core::f64::consts::PI).sqrt() * gamma(degrees_of_freedom / 2.0);
    let shape = (1.0 + x * x / degrees_of_freedom).powf(-(degrees_of_freedom + 1.0) / 2.0);
    Ok((numerator / denominator) * shape)
}

pub fn student_t_cdf(x: f64, degrees_of_freedom: f64) -> SciResult<f64> {
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }
    if x == 0.0 {
        return Ok(0.5);
    }

    let upper = x.abs();
    let area = integrate_simpson(
        |value| student_t_pdf(value, degrees_of_freedom).unwrap_or(0.0),
        0.0,
        upper,
        800,
    )?;
    if x > 0.0 {
        Ok((0.5 + area).min(1.0))
    } else {
        Ok((0.5 - area).max(0.0))
    }
}

pub fn student_t_inverse_cdf(probability: f64, degrees_of_freedom: f64) -> SciResult<f64> {
    if !(0.0..1.0).contains(&probability) {
        return Err(SciError::DomainError(
            "probability must be between 0 and 1 exclusive",
        ));
    }
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }

    let mut left = -20.0;
    let mut right = 20.0;
    for _ in 0..120 {
        let midpoint = (left + right) / 2.0;
        let cdf = student_t_cdf(midpoint, degrees_of_freedom)?;
        if (cdf - probability).abs() < 1e-8 {
            return Ok(midpoint);
        }
        if cdf < probability {
            left = midpoint;
        } else {
            right = midpoint;
        }
    }

    Ok((left + right) / 2.0)
}

pub fn chi_square_pdf(x: f64, degrees_of_freedom: f64) -> SciResult<f64> {
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }
    if x < 0.0 {
        return Ok(0.0);
    }

    let k = degrees_of_freedom;
    let coefficient = 1.0 / (2.0_f64.powf(k / 2.0) * gamma(k / 2.0));
    Ok(coefficient * x.powf(k / 2.0 - 1.0) * (-x / 2.0).exp())
}

pub fn chi_square_cdf(x: f64, degrees_of_freedom: f64) -> SciResult<f64> {
    if degrees_of_freedom <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }

    let area = integrate_simpson(
        |value| chi_square_pdf(value, degrees_of_freedom).unwrap_or(0.0),
        0.0,
        x,
        2_000,
    )?;
    Ok(area.clamp(0.0, 1.0))
}

pub fn f_pdf(x: f64, df1: f64, df2: f64) -> SciResult<f64> {
    if df1 <= 0.0 || df2 <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }
    if x < 0.0 {
        return Ok(0.0);
    }

    let numerator = (df1 / df2).powf(df1 / 2.0) * x.powf(df1 / 2.0 - 1.0);
    let denominator = beta(df1 / 2.0, df2 / 2.0) * (1.0 + (df1 / df2) * x).powf((df1 + df2) / 2.0);
    Ok(numerator / denominator)
}

pub fn f_cdf(x: f64, df1: f64, df2: f64) -> SciResult<f64> {
    if df1 <= 0.0 || df2 <= 0.0 {
        return Err(SciError::InvalidParameter(
            "degrees of freedom must be positive",
        ));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }

    let area = integrate_simpson(|value| f_pdf(value, df1, df2).unwrap_or(0.0), 0.0, x, 2_000)?;
    Ok(area.clamp(0.0, 1.0))
}

pub fn binomial_pmf(n: u64, k: u64, p: f64) -> SciResult<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SciError::DomainError("probability must be between 0 and 1"));
    }
    if k > n {
        return Err(SciError::InvalidParameter("k must be <= n"));
    }

    let coefficient = binomial_coefficient(n, k) as f64;
    Ok(coefficient * p.powf(k as f64) * (1.0 - p).powf((n - k) as f64))
}

pub fn poisson_pmf(lambda: f64, k: u64) -> SciResult<f64> {
    if lambda <= 0.0 {
        return Err(SciError::InvalidParameter("lambda must be positive"));
    }

    Ok(lambda.powf(k as f64) * (-lambda).exp() / factorial(k) as f64)
}

pub fn confidence_interval_mean_known_sigma(
    sample_mean: f64,
    sigma: f64,
    sample_size: usize,
    z_score: f64,
) -> SciResult<(f64, f64)> {
    if sigma <= 0.0 {
        return Err(SciError::InvalidParameter("sigma must be positive"));
    }
    if sample_size == 0 {
        return Err(SciError::InvalidParameter("sample size must be positive"));
    }
    if z_score <= 0.0 {
        return Err(SciError::InvalidParameter("z-score must be positive"));
    }

    let margin = z_score * sigma / (sample_size as f64).sqrt();
    Ok((sample_mean - margin, sample_mean + margin))
}

pub fn confidence_interval_mean_unknown_sigma(
    sample_mean: f64,
    sample_std_dev: f64,
    sample_size: usize,
    confidence_level: f64,
) -> SciResult<(f64, f64)> {
    if sample_std_dev <= 0.0 {
        return Err(SciError::InvalidParameter(
            "sample standard deviation must be positive",
        ));
    }
    if sample_size < 2 {
        return Err(SciError::InvalidParameter("sample size must be at least 2"));
    }
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(SciError::DomainError(
            "confidence level must be between 0 and 1",
        ));
    }

    let alpha = 1.0 - confidence_level;
    let critical = student_t_inverse_cdf(1.0 - alpha / 2.0, (sample_size - 1) as f64)?;
    let margin = critical * sample_std_dev / (sample_size as f64).sqrt();
    Ok((sample_mean - margin, sample_mean + margin))
}

fn factorial(n: u64) -> u128 {
    (1..=n).fold(1_u128, |acc, value| acc * u128::from(value))
}

fn binomial_coefficient(n: u64, k: u64) -> u128 {
    let k = k.min(n - k);
    let numerator = (0..k).fold(1_u128, |acc, i| acc * u128::from(n - i));
    numerator / factorial(k)
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26 approximation.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let polynomial =
        (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t).mul_add(-(-x * x).exp(), 1.0);
    sign * polynomial
}

fn integrate_simpson<F>(f: F, a: f64, b: f64, n: usize) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 || n % 2 == 1 {
        return Err(SciError::InvalidParameter(
            "n must be a positive even number",
        ));
    }

    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    Ok(sum * h / 3.0)
}

fn gamma(z: f64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
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
        core::f64::consts::PI / ((core::f64::consts::PI * z).sin() * gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = 0.999_999_999_999_809_9;
        for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
            x += coefficient / (z + index as f64 + 1.0);
        }
        let t = z + 7.5;
        (2.0 * core::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

fn beta(a: f64, b: f64) -> f64 {
    gamma(a) * gamma(b) / gamma(a + b)
}
