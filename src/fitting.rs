use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

pub fn polynomial_fit_least_squares(
    x_values: &[f64],
    y_values: &[f64],
    degree: usize,
) -> SciResult<Vec<f64>> {
    if x_values.is_empty() || y_values.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if x_values.len() != y_values.len() {
        return Err(SciError::InvalidParameter(
            "x and y data must have the same length",
        ));
    }
    if x_values.len() < degree + 1 {
        return Err(SciError::InvalidParameter(
            "not enough samples for requested polynomial degree",
        ));
    }

    let cols = degree + 1;
    let mut vandermonde = Vec::with_capacity(x_values.len() * cols);
    for &x in x_values {
        let mut value = 1.0;
        for _ in 0..cols {
            vandermonde.push(value);
            value *= x;
        }
    }

    let matrix = DynamicMatrix::new(x_values.len(), cols, vandermonde)?;
    matrix.least_squares(y_values)
}

pub fn polynomial_evaluate(coefficients: &[f64], x: f64) -> SciResult<f64> {
    if coefficients.is_empty() {
        return Err(SciError::EmptyInput);
    }
    Ok(coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, coefficient| acc * x + coefficient))
}
