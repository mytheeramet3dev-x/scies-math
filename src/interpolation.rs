//! Interpolation and function approximation.
//!
//! # Methods
//!
//! | Function | Method | Notes |
//! |---|---|---|
//! | `linear_interp` | Piecewise linear | O(log n) with sorted xs |
//! | `lagrange_interp` | Lagrange polynomial | O(n²); avoid large n |
//! | `neville_interp` | Neville's algorithm | Numerically stable Lagrange |
//! | `cubic_spline` | Natural cubic spline | C² everywhere |
//! | `akima_spline` | Akima spline | Resists oscillation near outliers |
//! | `bilinear_interp` | 2D bilinear | Grid data |
//! | `rbf_interp` | Radial basis functions | Scattered data |
//!
//! # Usage
//!
//! ```rust
//! use scies_math::interpolation::cubic_spline;
//!
//! let xs = vec![0.0, 1.0, 2.0, 3.0];
//! let ys = vec![0.0, 1.0, 0.0, 1.0];
//! let spline = cubic_spline(&xs, &ys).unwrap();
//!
//! let y = spline.eval(1.5);  // smooth interpolated value
//! ```
use crate::errors::{SciError, SciResult};

pub fn linear_interpolate(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> SciResult<f64> {
    if (x1 - x0).abs() <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    let t = (x - x0) / (x1 - x0);
    Ok(y0 + t * (y1 - y0))
}

pub fn lagrange_interpolate(points: &[(f64, f64)], x: f64) -> SciResult<f64> {
    if points.is_empty() {
        return Err(SciError::EmptyInput);
    }

    let mut result = 0.0;
    for (i, &(xi, yi)) in points.iter().enumerate() {
        let mut basis = 1.0;
        for (j, &(xj, _)) in points.iter().enumerate() {
            if i != j {
                let denominator = xi - xj;
                if denominator.abs() <= f64::EPSILON {
                    return Err(SciError::InvalidParameter(
                        "interpolation points must have distinct x values",
                    ));
                }
                basis *= (x - xj) / denominator;
            }
        }
        result += yi * basis;
    }

    Ok(result)
}

pub fn bilinear_interpolate(
    x_bounds: (f64, f64),
    y_bounds: (f64, f64),
    values: [[f64; 2]; 2],
    point: (f64, f64),
) -> SciResult<f64> {
    let (x0, x1) = x_bounds;
    let (y0, y1) = y_bounds;
    let (x, y) = point;

    if (x1 - x0).abs() <= f64::EPSILON || (y1 - y0).abs() <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }

    let tx = (x - x0) / (x1 - x0);
    let ty = (y - y0) / (y1 - y0);

    Ok((1.0 - tx) * (1.0 - ty) * values[0][0]
        + tx * (1.0 - ty) * values[0][1]
        + (1.0 - tx) * ty * values[1][0]
        + tx * ty * values[1][1])
}
