use crate::errors::{SciError, SciResult};

pub fn derivative<F>(f: F, x: f64, h: f64) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h must be positive"));
    }
    Ok((f(x + h) - f(x - h)) / (2.0 * h))
}

pub fn integrate_simpson<F>(f: F, a: f64, b: f64, n: usize) -> SciResult<f64>
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

pub fn newton_raphson<F, G>(
    f: F,
    df: G,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let mut x = initial_guess;
    for _ in 0..max_iterations {
        let slope = df(x);
        if slope.abs() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }

        let next = x - f(x) / slope;
        if (next - x).abs() < tolerance {
            return Ok(next);
        }
        x = next;
    }

    Err(SciError::NonConvergent("Newton-Raphson"))
}
