use crate::errors::{SciError, SciResult};

pub fn integrate_trapezoidal<F>(f: F, a: f64, b: f64, n: usize) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Err(SciError::InvalidParameter("n must be positive"));
    }

    let h = (b - a) / n as f64;
    let interior_sum = (1..n)
        .map(|i| {
            let x = a + i as f64 * h;
            f(x)
        })
        .sum::<f64>();

    Ok(h * ((f(a) + f(b)) / 2.0 + interior_sum))
}

pub fn bisection<F>(
    f: F,
    mut left: f64,
    mut right: f64,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
{
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }

    let mut f_left = f(left);
    let f_right = f(right);
    if f_left * f_right > 0.0 {
        return Err(SciError::DomainError(
            "interval must bracket a root with opposite signs",
        ));
    }

    for _ in 0..max_iterations {
        let midpoint = (left + right) / 2.0;
        let f_mid = f(midpoint);

        if f_mid.abs() < tolerance || ((right - left) / 2.0).abs() < tolerance {
            return Ok(midpoint);
        }

        if f_left * f_mid < 0.0 {
            right = midpoint;
        } else {
            left = midpoint;
            f_left = f_mid;
        }
    }

    Err(SciError::NonConvergent("bisection"))
}

pub fn euler_step<F>(state: f64, time: f64, dt: f64, derivative: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    state + derivative(time, state) * dt
}

pub fn rk4_step<F>(state: f64, time: f64, dt: f64, derivative: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let k1 = derivative(time, state);
    let k2 = derivative(time + dt / 2.0, state + dt * k1 / 2.0);
    let k3 = derivative(time + dt / 2.0, state + dt * k2 / 2.0);
    let k4 = derivative(time + dt, state + dt * k3);

    state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}
