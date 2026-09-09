//! Scalar and univariate optimization.
//!
//! # Functions
//!
//! | Function | Method | Problem type |
//! |---|---|---|
//! | `golden_section` | Golden-section search | min f(x) on $[a, b]$ |
//! | `brent` | Brent's method | min f(x), no derivatives |
//! | `bisection` | Bisection | root finding f(x)=0 |
//! | `newton_raphson` | Newton-Raphson | root finding with f' |
//! | `secant` | Secant method | root finding, no f' needed |
//! | `illinois` | Illinois/Regula Falsi | Bracketed, superlinear |
//!
//! # Usage
//!
//! ```ignore
//! use scies_math_th::optimization::golden_section;
//!
//! // minimize f(x) = (x-2)² + 1 on [0, 4]
//! let (x_min, f_min) = golden_section(|x| (x-2.0).powi(2) + 1.0, 0.0, 4.0, 1e-8);
//! // x_min ≈ 2.0, f_min ≈ 1.0
//! ```
//!
//! For multivariate optimization (BFGS, L-BFGS, Trust-Region, PSO, DE)
//! see [`crate::opt_multivar`] and [`crate::opt_multivar_ext`].
use crate::errors::{SciError, SciResult};

pub fn gradient_descent<F, G>(
    objective: F,
    gradient: G,
    mut initial_guess: f64,
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    if learning_rate <= 0.0 || tolerance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "learning rate and tolerance must be positive",
        ));
    }

    let mut previous_value = objective(initial_guess);
    for _ in 0..max_iterations {
        let slope = gradient(initial_guess);
        let next = initial_guess - learning_rate * slope;
        let value = objective(next);
        if (value - previous_value).abs() < tolerance && (next - initial_guess).abs() < tolerance {
            return Ok(next);
        }
        initial_guess = next;
        previous_value = value;
    }

    Err(SciError::NonConvergent("gradient descent"))
}

pub fn golden_section_search<F>(
    f: F,
    mut left: f64,
    mut right: f64,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<f64>
where
    F: Fn(f64) -> f64,
{
    if right <= left || tolerance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "search interval must be valid and tolerance positive",
        ));
    }

    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = right - inv_phi * (right - left);
    let mut d = left + inv_phi * (right - left);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..max_iterations {
        if (right - left).abs() < tolerance {
            return Ok((left + right) / 2.0);
        }

        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - inv_phi * (right - left);
            fc = f(c);
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + inv_phi * (right - left);
            fd = f(d);
        }
    }

    Err(SciError::NonConvergent("golden section search"))
}
