//! Floating-point interval arithmetic (Moore algorithm).
//!
//! Represents a closed real interval $[a, b] = \{x \in \mathbb{R} \mid a \le x \le b\}$.
//!
//! # Precision Notice
//! Operations in this module use standard 64-bit floating-point (`f64`) arithmetic.
//! While interval arithmetic tracks algebraic intervals across operations, exact rigorous
//! mathematical containment bounds at the sub-epsilon level require directed outward hardware
//! rounding or exact rational intervals ([`crate::exact::Rational`]).

use crate::errors::{SciError, SciResult};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A closed real interval $[a, b]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    /// Creates a new interval $[a, b]$ where $a \le b$.
    pub fn new(lower: f64, upper: f64) -> SciResult<Self> {
        if lower.is_nan() || upper.is_nan() {
            return Err(SciError::DomainError("interval bounds cannot be NaN"));
        }
        if lower > upper {
            return Err(SciError::InvalidParameter(
                "lower bound must be <= upper bound",
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Creates a point interval $[x, x]$.
    pub fn point(val: f64) -> Self {
        Self {
            lower: val,
            upper: val,
        }
    }

    /// Lower bound $a$.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Upper bound $b$.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Width of the interval $b - a$.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Midpoint of the interval $\frac{a + b}{2}$.
    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    /// Radius of the interval $\frac{b - a}{2}$.
    pub fn radius(&self) -> f64 {
        (self.upper - self.lower) / 2.0
    }

    /// Checks if a scalar value $x$ is contained within $[a, b]$.
    pub fn contains(&self, val: f64) -> bool {
        self.lower <= val && val <= self.upper
    }

    /// Checks if two intervals overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }

    /// Computes the intersection of two intervals, returning `None` if disjoint.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let low = self.lower.max(other.lower);
        let high = self.upper.min(other.upper);
        if low <= high {
            Some(Self {
                lower: low,
                upper: high,
            })
        } else {
            None
        }
    }

    /// Computes the hull (smallest interval enclosing both intervals).
    pub fn hull(&self, other: &Self) -> Self {
        Self {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }

    /// Computes $\sqrt{[a, b]} = [\sqrt{a}, \sqrt{b}]$.
    pub fn sqrt(&self) -> SciResult<Self> {
        if self.lower < 0.0 {
            return Err(SciError::DomainError(
                "sqrt of interval containing negative numbers",
            ));
        }
        Ok(Self {
            lower: self.lower.sqrt(),
            upper: self.upper.sqrt(),
        })
    }
}

impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            lower: self.lower + rhs.lower,
            upper: self.upper + rhs.upper,
        }
    }
}

impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            lower: self.lower - rhs.upper,
            upper: self.upper - rhs.lower,
        }
    }
}

impl Mul for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let p1 = self.lower * rhs.lower;
        let p2 = self.lower * rhs.upper;
        let p3 = self.upper * rhs.lower;
        let p4 = self.upper * rhs.upper;

        let low = p1.min(p2).min(p3.min(p4));
        let high = p1.max(p2).max(p3.max(p4));

        Self {
            lower: low,
            upper: high,
        }
    }
}

impl Div for Interval {
    type Output = SciResult<Self>;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.contains(0.0) {
            return Err(SciError::DivisionByZero);
        }
        let inv = Self {
            lower: 1.0 / rhs.upper,
            upper: 1.0 / rhs.lower,
        };
        Ok(self * inv)
    }
}

impl Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            lower: -self.upper,
            upper: -self.lower,
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lower, self.upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_arithmetic() {
        let i1 = Interval::new(1.0, 2.0).unwrap();
        let i2 = Interval::new(3.0, 4.0).unwrap();

        let sum = i1 + i2;
        assert_eq!(sum.lower(), 4.0);
        assert_eq!(sum.upper(), 6.0);

        let diff = i1 - i2;
        assert_eq!(diff.lower(), -3.0);
        assert_eq!(diff.upper(), -1.0);

        let prod = i1 * i2;
        assert_eq!(prod.lower(), 3.0);
        assert_eq!(prod.upper(), 8.0);
    }

    #[test]
    fn test_interval_containment_and_intersection() {
        let i1 = Interval::new(0.0, 5.0).unwrap();
        let i2 = Interval::new(3.0, 8.0).unwrap();

        assert!(i1.contains(3.5));
        assert!(i1.overlaps(&i2));

        let inter = i1.intersection(&i2).unwrap();
        assert_eq!(inter.lower(), 3.0);
        assert_eq!(inter.upper(), 5.0);
    }
}
