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
//! use scies_math_th::interpolation::cubic_spline;
//!
//! let xs = vec![0.0, 1.0, 2.0, 3.0];
//! let ys = vec![0.0, 1.0, 0.0, 1.0];
//! let spline = cubic_spline(&xs, &ys).unwrap();
//!
//! let y = spline.eval(1.5);  // smooth interpolated value
//! ```
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

#[derive(Debug, Clone, PartialEq)]
pub struct CubicSpline {
    xs: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    pub fn eval(&self, x: f64) -> f64 {
        let index = find_interval(&self.xs, x);
        let dx = x - self.xs[index];
        self.a[index]
            + self.b[index] * dx
            + self.c[index] * dx * dx
            + self.d[index] * dx * dx * dx
    }

    pub fn derivative(&self, x: f64) -> f64 {
        let index = find_interval(&self.xs, x);
        let dx = x - self.xs[index];
        self.b[index] + 2.0 * self.c[index] * dx + 3.0 * self.d[index] * dx * dx
    }

    pub fn knots(&self) -> &[f64] {
        &self.xs
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AkimaSpline {
    xs: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl AkimaSpline {
    pub fn eval(&self, x: f64) -> f64 {
        let index = find_interval(&self.xs, x);
        let dx = x - self.xs[index];
        self.a[index]
            + self.b[index] * dx
            + self.c[index] * dx * dx
            + self.d[index] * dx * dx * dx
    }

    pub fn derivative(&self, x: f64) -> f64 {
        let index = find_interval(&self.xs, x);
        let dx = x - self.xs[index];
        self.b[index] + 2.0 * self.c[index] * dx + 3.0 * self.d[index] * dx * dx
    }

    pub fn knots(&self) -> &[f64] {
        &self.xs
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
    Linear,
    Cubic,
    ThinPlate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RbfInterpolator {
    points: Vec<f64>,
    weights: Vec<f64>,
    epsilon: f64,
    kernel: RbfKernel,
}

impl RbfInterpolator {
    pub fn eval(&self, x: f64) -> f64 {
        self.points
            .iter()
            .zip(self.weights.iter())
            .map(|(&point, &weight)| {
                weight * rbf_value(self.kernel, (x - point).abs(), self.epsilon)
            })
            .sum()
    }

    pub fn points(&self) -> &[f64] {
        &self.points
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

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

pub fn neville_interpolate(points: &[(f64, f64)], x: f64) -> SciResult<f64> {
    if points.is_empty() {
        return Err(SciError::EmptyInput);
    }
    validate_distinct_points(points)?;

    let n = points.len();
    let mut values = points.iter().map(|&(_, y)| y).collect::<Vec<_>>();
    for width in 1..n {
        for i in 0..(n - width) {
            let xi = points[i].0;
            let xj = points[i + width].0;
            let denominator = xi - xj;
            if denominator.abs() <= f64::EPSILON {
                return Err(SciError::InvalidParameter(
                    "interpolation points must have distinct x values",
                ));
            }
            values[i] =
                ((x - xj) * values[i] + (xi - x) * values[i + 1]) / denominator;
        }
    }

    Ok(values[0])
}

pub fn cubic_spline(xs: &[f64], ys: &[f64]) -> SciResult<CubicSpline> {
    validate_xy(xs, ys, 2)?;

    let n = xs.len();
    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = xs[i + 1] - xs[i];
    }

    let mut alpha = vec![0.0; n];
    for i in 1..n - 1 {
        alpha[i] =
            3.0 * (ys[i + 1] - ys[i]) / h[i] - 3.0 * (ys[i] - ys[i - 1]) / h[i - 1];
    }

    let mut l = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];
    l[0] = 1.0;

    for i in 1..n - 1 {
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
        if l[i].abs() <= f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1.0;

    let mut a = vec![0.0; n - 1];
    let mut b = vec![0.0; n - 1];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n - 1];

    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] =
            (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        a[j] = ys[j];
    }

    c.truncate(n - 1);
    Ok(CubicSpline {
        xs: xs.to_vec(),
        a,
        b,
        c,
        d,
    })
}

pub fn akima_spline(xs: &[f64], ys: &[f64]) -> SciResult<AkimaSpline> {
    validate_xy(xs, ys, 2)?;

    let n = xs.len();
    if n == 2 {
        let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return Ok(AkimaSpline {
            xs: xs.to_vec(),
            a: vec![ys[0]],
            b: vec![slope],
            c: vec![0.0],
            d: vec![0.0],
        });
    }

    let mut slopes = vec![0.0; n - 1];
    for i in 0..n - 1 {
        slopes[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }

    let mut extended = vec![0.0; n + 3];
    extended[0] = 3.0 * slopes[0] - 2.0 * slopes[1];
    extended[1] = 2.0 * slopes[0] - slopes[1];
    extended[2..2 + slopes.len()].copy_from_slice(&slopes);
    extended[n + 1] = 2.0 * slopes[n - 2] - slopes[n - 3];
    extended[n + 2] = 3.0 * slopes[n - 2] - 2.0 * slopes[n - 3];

    let mut derivatives = vec![0.0; n];
    for i in 0..n {
        let w1 = (extended[i + 3] - extended[i + 2]).abs();
        let w2 = (extended[i + 1] - extended[i]).abs();
        derivatives[i] = if (w1 + w2) <= f64::EPSILON {
            0.5 * (extended[i + 1] + extended[i + 2])
        } else {
            (w1 * extended[i + 1] + w2 * extended[i + 2]) / (w1 + w2)
        };
    }

    let mut a = vec![0.0; n - 1];
    let mut b = vec![0.0; n - 1];
    let mut c = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];
    for i in 0..n - 1 {
        let h = xs[i + 1] - xs[i];
        let slope = slopes[i];
        a[i] = ys[i];
        b[i] = derivatives[i];
        c[i] = (3.0 * slope - 2.0 * derivatives[i] - derivatives[i + 1]) / h;
        d[i] = (derivatives[i] + derivatives[i + 1] - 2.0 * slope) / (h * h);
    }

    Ok(AkimaSpline {
        xs: xs.to_vec(),
        a,
        b,
        c,
        d,
    })
}

pub fn rbf_interp(
    xs: &[f64],
    ys: &[f64],
    epsilon: f64,
    kernel: RbfKernel,
) -> SciResult<RbfInterpolator> {
    validate_xy_unordered(xs, ys, 1)?;
    validate_distinct_xs(xs)?;
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(SciError::InvalidParameter(
            "epsilon must be positive and finite",
        ));
    }

    let n = xs.len();
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            matrix[row * n + col] = rbf_value(kernel, (xs[row] - xs[col]).abs(), epsilon);
        }
    }

    let system = DynamicMatrix::new(n, n, matrix)?;
    let weights = system.solve_linear_system(ys)?;
    Ok(RbfInterpolator {
        points: xs.to_vec(),
        weights,
        epsilon,
        kernel,
    })
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

pub fn linear_interp(xs: &[f64], ys: &[f64], x: f64) -> SciResult<f64> {
    validate_xy(xs, ys, 2)?;
    let index = find_interval(xs, x);
    linear_interpolate(xs[index], ys[index], xs[index + 1], ys[index + 1], x)
}

pub fn lagrange_interp(points: &[(f64, f64)], x: f64) -> SciResult<f64> {
    lagrange_interpolate(points, x)
}

pub fn neville_interp(points: &[(f64, f64)], x: f64) -> SciResult<f64> {
    neville_interpolate(points, x)
}

pub fn bilinear_interp(
    x_bounds: (f64, f64),
    y_bounds: (f64, f64),
    values: [[f64; 2]; 2],
    point: (f64, f64),
) -> SciResult<f64> {
    bilinear_interpolate(x_bounds, y_bounds, values, point)
}

fn validate_xy(xs: &[f64], ys: &[f64], min_len: usize) -> SciResult<()> {
    validate_xy_unordered(xs, ys, min_len)?;
    for pair in xs.windows(2) {
        if pair[1] <= pair[0] {
            return Err(SciError::InvalidParameter(
                "x values must be strictly increasing",
            ));
        }
    }
    Ok(())
}

fn validate_xy_unordered(xs: &[f64], ys: &[f64], min_len: usize) -> SciResult<()> {
    if xs.is_empty() || ys.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if xs.len() != ys.len() {
        return Err(SciError::InvalidParameter(
            "x and y data must have the same length",
        ));
    }
    if xs.len() < min_len {
        return Err(SciError::InvalidParameter("not enough interpolation points"));
    }
    if xs.iter().chain(ys.iter()).any(|value| !value.is_finite()) {
        return Err(SciError::InvalidParameter("values must be finite"));
    }
    Ok(())
}

fn validate_distinct_points(points: &[(f64, f64)]) -> SciResult<()> {
    if points.iter().any(|&(x, y)| !x.is_finite() || !y.is_finite()) {
        return Err(SciError::InvalidParameter("values must be finite"));
    }
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            if (points[i].0 - points[j].0).abs() <= f64::EPSILON {
                return Err(SciError::InvalidParameter(
                    "interpolation points must have distinct x values",
                ));
            }
        }
    }
    Ok(())
}

fn validate_distinct_xs(xs: &[f64]) -> SciResult<()> {
    for i in 0..xs.len() {
        for j in i + 1..xs.len() {
            if (xs[i] - xs[j]).abs() <= f64::EPSILON {
                return Err(SciError::InvalidParameter("x values must be distinct"));
            }
        }
    }
    Ok(())
}

fn find_interval(xs: &[f64], x: f64) -> usize {
    if x <= xs[0] {
        return 0;
    }
    let last_interval = xs.len() - 2;
    if x >= xs[xs.len() - 1] {
        return last_interval;
    }

    match xs.binary_search_by(|probe| probe.total_cmp(&x)) {
        Ok(index) => index.min(last_interval),
        Err(index) => (index - 1).min(last_interval),
    }
}

fn rbf_value(kernel: RbfKernel, radius: f64, epsilon: f64) -> f64 {
    let scaled = epsilon * radius;
    match kernel {
        RbfKernel::Gaussian => (-scaled * scaled).exp(),
        RbfKernel::Multiquadric => (1.0 + scaled * scaled).sqrt(),
        RbfKernel::InverseMultiquadric => 1.0 / (1.0 + scaled * scaled).sqrt(),
        RbfKernel::Linear => radius,
        RbfKernel::Cubic => radius * radius * radius,
        RbfKernel::ThinPlate => {
            if radius <= f64::EPSILON {
                0.0
            } else {
                let r2 = radius * radius;
                r2 * radius.ln()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn neville_matches_quadratic() {
        let points = [(0.0, 1.0), (1.0, 4.0), (2.0, 9.0)];
        let y = neville_interpolate(&points, 1.5).unwrap();
        assert!(approx_eq(y, 6.25, 1e-12));
    }

    #[test]
    fn cubic_spline_interpolates_knots() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 1.0, 0.0, 1.0];
        let spline = cubic_spline(&xs, &ys).unwrap();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!(approx_eq(spline.eval(x), y, 1e-12));
        }
    }

    #[test]
    fn akima_spline_interpolates_knots() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 0.0, 1.0, 0.5];
        let spline = akima_spline(&xs, &ys).unwrap();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!(approx_eq(spline.eval(x), y, 1e-12));
        }
    }

    #[test]
    fn rbf_interpolates_training_points() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 0.0];
        let rbf = rbf_interp(&xs, &ys, 1.0, RbfKernel::Gaussian).unwrap();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!(approx_eq(rbf.eval(x), y, 1e-9));
        }
    }

    #[test]
    fn linear_interp_uses_sorted_knots() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 2.0, 4.0];
        assert!(approx_eq(
            linear_interp(&xs, &ys, 1.25).unwrap(),
            2.5,
            1e-12
        ));
    }
}
