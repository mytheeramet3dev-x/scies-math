use crate::errors::{SciError, SciResult};

pub fn factorial(n: u64) -> u128 {
    (1..=n).fold(1_u128, |acc, value| acc * u128::from(value))
}

pub fn permutations(n: u64, r: u64) -> SciResult<u128> {
    if r > n {
        return Err(SciError::InvalidParameter("r must be <= n"));
    }
    Ok((0..r).fold(1_u128, |acc, i| acc * u128::from(n - i)))
}

pub fn combinations(n: u64, r: u64) -> SciResult<u128> {
    if r > n {
        return Err(SciError::InvalidParameter("r must be <= n"));
    }
    let r = r.min(n - r);
    let numerator = (0..r).fold(1_u128, |acc, i| acc * u128::from(n - i));
    Ok(numerator / factorial(r))
}

pub fn arithmetic_mean(a: f64, b: f64) -> f64 {
    (a + b) / 2.0
}

pub fn geometric_mean(a: f64, b: f64) -> SciResult<f64> {
    if a < 0.0 || b < 0.0 {
        return Err(SciError::DomainError(
            "geometric mean requires non-negative inputs",
        ));
    }
    Ok((a * b).sqrt())
}

pub fn harmonic_mean(a: f64, b: f64) -> SciResult<f64> {
    if a == 0.0 || b == 0.0 {
        return Err(SciError::DivisionByZero);
    }
    Ok(2.0 * a * b / (a + b))
}

pub fn quadratic_roots(a: f64, b: f64, c: f64) -> SciResult<(f64, f64)> {
    if a == 0.0 {
        return Err(SciError::InvalidParameter("a must not be zero"));
    }
    let discriminant = (b * b) - (4.0 * a * c);
    if discriminant < 0.0 {
        return Err(SciError::DomainError("quadratic roots are complex"));
    }
    let sqrt_disc = discriminant.sqrt();
    let denom = 2.0 * a;
    Ok(((-b + sqrt_disc) / denom, (-b - sqrt_disc) / denom))
}

pub fn solve_linear(a: f64, b: f64) -> SciResult<f64> {
    if a == 0.0 {
        return Err(SciError::DivisionByZero);
    }
    Ok(-b / a)
}
