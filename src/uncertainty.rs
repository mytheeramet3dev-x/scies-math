//! Uncertainty propagation for experimental and scientific measurement values ($x \pm \sigma$).
//!
//! Uses first-order Gaussian error propagation:
//! $$\sigma_f = \sqrt{ \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_{x_i}^2 }$$

use crate::errors::{SciError, SciResult};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A physical or numerical value with standard Gaussian uncertainty ($x \pm \sigma_x$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UncertainValue {
    /// Nominal or expected mean value $x$.
    pub value: f64,
    /// 1-sigma standard uncertainty $\sigma_x \ge 0$.
    pub std_dev: f64,
}

impl UncertainValue {
    /// Creates a new value with standard uncertainty.
    pub fn new(value: f64, std_dev: f64) -> Self {
        Self {
            value,
            std_dev: std_dev.abs(),
        }
    }

    /// Creates an exact value without uncertainty ($\sigma = 0$).
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            std_dev: 0.0,
        }
    }

    /// Relative uncertainty $\frac{\sigma_x}{|x|}$.
    pub fn relative_uncertainty(&self) -> f64 {
        if self.value.abs() < f64::EPSILON {
            0.0
        } else {
            self.std_dev / self.value.abs()
        }
    }

    /// Computes $x^p$ with uncertainty propagation.
    pub fn powf(&self, p: f64) -> Self {
        let val = self.value.powf(p);
        let grad = p * self.value.powf(p - 1.0);
        let std_dev = (grad * self.std_dev).abs();
        Self::new(val, std_dev)
    }

    /// Computes $\sqrt{x}$ with uncertainty propagation.
    pub fn sqrt(&self) -> SciResult<Self> {
        if self.value < 0.0 {
            return Err(SciError::DomainError("sqrt of negative value"));
        }
        let val = self.value.sqrt();
        let grad = if val > 0.0 { 0.5 / val } else { 0.0 };
        Ok(Self::new(val, (grad * self.std_dev).abs()))
    }

    /// Computes $\exp(x)$ with uncertainty propagation.
    pub fn exp(&self) -> Self {
        let val = self.value.exp();
        Self::new(val, (val * self.std_dev).abs())
    }

    /// Computes $\ln(x)$ with uncertainty propagation.
    pub fn ln(&self) -> SciResult<Self> {
        if self.value <= 0.0 {
            return Err(SciError::DomainError("ln of non-positive value"));
        }
        let val = self.value.ln();
        let grad = 1.0 / self.value;
        Ok(Self::new(val, (grad * self.std_dev).abs()))
    }

    /// Computes $\sin(x)$ with uncertainty propagation.
    pub fn sin(&self) -> Self {
        Self::new(self.value.sin(), (self.value.cos() * self.std_dev).abs())
    }

    /// Computes $\cos(x)$ with uncertainty propagation.
    pub fn cos(&self) -> Self {
        Self::new(self.value.cos(), (self.value.sin() * self.std_dev).abs())
    }
}

impl Add for UncertainValue {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let val = self.value + rhs.value;
        let std = (self.std_dev * self.std_dev + rhs.std_dev * rhs.std_dev).sqrt();
        Self::new(val, std)
    }
}

impl Sub for UncertainValue {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let val = self.value - rhs.value;
        let std = (self.std_dev * self.std_dev + rhs.std_dev * rhs.std_dev).sqrt();
        Self::new(val, std)
    }
}

impl Mul for UncertainValue {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let val = self.value * rhs.value;
        let var = (rhs.value * self.std_dev).powi(2) + (self.value * rhs.std_dev).powi(2);
        Self::new(val, var.sqrt())
    }
}

impl Div for UncertainValue {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let val = self.value / rhs.value;
        let denom_sq = rhs.value * rhs.value;
        let var =
            (self.std_dev / rhs.value).powi(2) + (self.value * rhs.std_dev / denom_sq).powi(2);
        Self::new(val, var.sqrt())
    }
}

impl Neg for UncertainValue {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.value, self.std_dev)
    }
}

impl fmt::Display for UncertainValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} ± {:.4}", self.value, self.std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_arithmetic() {
        let a = UncertainValue::new(10.0, 0.3);
        let b = UncertainValue::new(5.0, 0.4);

        let sum = a + b;
        assert_eq!(sum.value, 15.0);
        assert!((sum.std_dev - 0.5).abs() < 1e-10); // sqrt(0.09 + 0.16) = 0.5

        let prod = a * b;
        assert_eq!(prod.value, 50.0);
        // sqrt((5 * 0.3)^2 + (10 * 0.4)^2) = sqrt(2.25 + 16) = sqrt(18.25) approx 4.272
        assert!((prod.std_dev - 18.25_f64.sqrt()).abs() < 1e-10);
    }
}
