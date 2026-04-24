use crate::errors::{SciError, SciResult};

pub fn numerical_gradient<F>(f: F, point: &[f64], h: f64) -> SciResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if point.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("step size must be positive"));
    }

    let mut gradient = vec![0.0; point.len()];
    let mut plus = point.to_vec();
    let mut minus = point.to_vec();

    for index in 0..point.len() {
        plus[index] += h;
        minus[index] -= h;
        gradient[index] = (f(&plus) - f(&minus)) / (2.0 * h);
        plus[index] = point[index];
        minus[index] = point[index];
    }

    Ok(gradient)
}

pub fn gradient_descent<F>(
    f: F,
    initial_point: Vec<f64>,
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
) -> SciResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if initial_point.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if learning_rate <= 0.0 || tolerance <= 0.0 {
        return Err(SciError::InvalidParameter(
            "learning rate and tolerance must be positive",
        ));
    }

    let mut point = initial_point;

    for _ in 0..max_iterations {
        let gradient = numerical_gradient(&f, &point, 1e-6)?;
        let gradient_norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if gradient_norm < tolerance {
            return Ok(point);
        }

        let previous_point = point.clone();
        for (coordinate, slope) in point.iter_mut().zip(gradient.iter()) {
            *coordinate -= learning_rate * slope;
        }

        let step_norm = point
            .iter()
            .zip(previous_point.iter())
            .map(|(next, prev)| (next - prev).powi(2))
            .sum::<f64>()
            .sqrt();
        if step_norm < tolerance {
            return Ok(point);
        }
    }

    Err(SciError::NonConvergent("multivariate gradient descent"))
}
