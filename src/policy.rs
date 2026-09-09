//! Centralized tolerance policies, floating-point validation, and stability contracts.

use crate::errors::{SciError, SciResult};

/// Numerical tolerance and convergence policy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TolerancePolicy {
    /// Strict tolerance ($10^{-12}$) for high-precision and verification workflows.
    Strict,
    /// Standard engineering tolerance ($10^{-7}$) for everyday scientific computing.
    #[default]
    Standard,
    /// Relaxed tolerance ($10^{-4}$) for ill-conditioned or coarse exploratory heuristics.
    Relaxed,
    /// Custom configuration with user-specified absolute tolerance and condition limit.
    Custom {
        tolerance: f64,
        max_iterations: usize,
        max_condition: f64,
    },
}

impl TolerancePolicy {
    /// Absolute epsilon threshold for this policy.
    pub fn eps(&self) -> f64 {
        match self {
            Self::Strict => 1e-12,
            Self::Standard => 1e-7,
            Self::Relaxed => 1e-4,
            Self::Custom { tolerance, .. } => *tolerance,
        }
    }

    /// Maximum recommended iterations for this policy.
    pub fn max_iterations(&self) -> usize {
        match self {
            Self::Strict => 1000,
            Self::Standard => 500,
            Self::Relaxed => 200,
            Self::Custom { max_iterations, .. } => *max_iterations,
        }
    }

    /// Maximum allowable condition number before warning or error.
    pub fn max_condition(&self) -> f64 {
        match self {
            Self::Strict => 1e10,
            Self::Standard => 1e14,
            Self::Relaxed => 1e16,
            Self::Custom { max_condition, .. } => *max_condition,
        }
    }

    /// Checks whether $|a| \le \text{tolerance}$.
    pub fn is_zero(&self, val: f64) -> bool {
        val.abs() <= self.eps()
    }

    /// Checks whether $|a - b| \le \text{tolerance} \cdot (1 + |a| + |b|)$.
    pub fn is_equal(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.eps() * (1.0 + a.abs() + b.abs())
    }

    /// Validates that a floating-point value is neither NaN nor Infinite.
    pub fn validate_finite(&self, val: f64) -> SciResult<()> {
        if val.is_nan() || val.is_infinite() {
            return Err(SciError::DomainError(
                "value must be finite (not NaN or Inf)",
            ));
        }
        Ok(())
    }

    /// Validates that all elements of a slice are finite.
    pub fn validate_slice_finite(&self, slice: &[f64]) -> SciResult<()> {
        for &val in slice {
            self.validate_finite(val)?;
        }
        Ok(())
    }

    /// Checks if a matrix condition number is within acceptable bounds.
    pub fn check_condition(&self, condition: f64) -> SciResult<()> {
        if condition > self.max_condition() {
            return Err(SciError::IllConditioned {
                condition_number: condition,
                threshold: self.max_condition(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_policy_equality() {
        let policy = TolerancePolicy::Standard;
        assert!(policy.is_zero(1e-9));
        assert!(!policy.is_zero(1e-5));
        assert!(policy.is_equal(1.0, 1.0 + 1e-8));
    }

    #[test]
    fn test_finite_validation() {
        let policy = TolerancePolicy::Strict;
        assert!(policy.validate_finite(1.234).is_ok());
        assert!(policy.validate_finite(f64::NAN).is_err());
        assert!(policy.validate_finite(f64::INFINITY).is_err());
    }
}
