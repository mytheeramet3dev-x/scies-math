//! Exact rational number arithmetic backed by 128-bit signed integers.
//!
//! Stored in canonical irreducible form:
//! $$r = \frac{p}{q} \quad \text{where} \quad q > 0, \; \gcd(|p|, q) = 1$$
//!
//! # Numerical & Overflow Safety
//! Standard arithmetic operators (`+`, `-`, `*`, `/`) perform checked operations
//! and panic on integer overflow with descriptive diagnostics. For non-panicking
//! workflows, use [`Rational::checked_add`], [`Rational::checked_sub`],
//! [`Rational::checked_mul`], and [`Rational::checked_div`].

use crate::errors::{SciError, SciResult};
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// An exact rational number $p/q$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    numer: i128,
    denom: i128,
}

impl Rational {
    /// Creates a new rational number $p/q$, automatically reduced to canonical form.
    ///
    /// # Errors
    /// Returns [`SciError::DivisionByZero`] if `denom == 0`.
    pub fn new(numer: i128, denom: i128) -> SciResult<Self> {
        if denom == 0 {
            return Err(SciError::DivisionByZero);
        }

        if numer == 0 {
            return Ok(Self { numer: 0, denom: 1 });
        }

        let sign = if denom < 0 { -1 } else { 1 };
        let n = numer
            .checked_mul(sign)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow when adjusting sign of numerator",
            ))?;
        let d = denom.unsigned_abs();

        let g = gcd(n.unsigned_abs(), d) as i128;
        Ok(Self {
            numer: n / g,
            denom: (d as i128) / g,
        })
    }

    /// Creates a rational from an integer $n = n/1$.
    pub const fn from_integer(n: i128) -> Self {
        Self { numer: n, denom: 1 }
    }

    /// Rational zero $0/1$.
    pub const fn zero() -> Self {
        Self { numer: 0, denom: 1 }
    }

    /// Rational one $1/1$.
    pub const fn one() -> Self {
        Self { numer: 1, denom: 1 }
    }

    /// Numerator $p$.
    pub const fn numer(&self) -> i128 {
        self.numer
    }

    /// Denominator $q > 0$.
    pub const fn denom(&self) -> i128 {
        self.denom
    }

    /// Converts this exact rational to 64-bit floating point `f64`.
    pub fn to_f64(&self) -> f64 {
        self.numer as f64 / self.denom as f64
    }

    /// Multiplicative inverse $\frac{q}{p}$.
    ///
    /// # Errors
    /// Returns [`SciError::DivisionByZero`] if `self.numer == 0`.
    pub fn recip(&self) -> SciResult<Self> {
        Self::new(self.denom, self.numer)
    }

    /// Absolute value $|p|/q$.
    pub fn abs(&self) -> Self {
        Self {
            numer: self.numer.abs(),
            denom: self.denom,
        }
    }

    /// Checked addition returning [`SciResult`].
    pub fn checked_add(&self, rhs: &Self) -> SciResult<Self> {
        let n1 = self
            .numer
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational addition (left cross product)",
            ))?;
        let n2 = rhs
            .numer
            .checked_mul(self.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational addition (right cross product)",
            ))?;
        let n = n1.checked_add(n2).ok_or(SciError::ExactArithmeticOverflow(
            "overflow in rational addition (numerator sum)",
        ))?;
        let d = self
            .denom
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational addition (denominator product)",
            ))?;
        Self::new(n, d)
    }

    /// Checked subtraction returning [`SciResult`].
    pub fn checked_sub(&self, rhs: &Self) -> SciResult<Self> {
        let n1 = self
            .numer
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational subtraction (left cross product)",
            ))?;
        let n2 = rhs
            .numer
            .checked_mul(self.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational subtraction (right cross product)",
            ))?;
        let n = n1.checked_sub(n2).ok_or(SciError::ExactArithmeticOverflow(
            "overflow in rational subtraction (numerator difference)",
        ))?;
        let d = self
            .denom
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational subtraction (denominator product)",
            ))?;
        Self::new(n, d)
    }

    /// Checked multiplication returning [`SciResult`].
    pub fn checked_mul(&self, rhs: &Self) -> SciResult<Self> {
        let n = self
            .numer
            .checked_mul(rhs.numer)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational multiplication (numerator product)",
            ))?;
        let d = self
            .denom
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational multiplication (denominator product)",
            ))?;
        Self::new(n, d)
    }

    /// Checked division returning [`SciResult`].
    pub fn checked_div(&self, rhs: &Self) -> SciResult<Self> {
        if rhs.numer == 0 {
            return Err(SciError::DivisionByZero);
        }
        let n = self
            .numer
            .checked_mul(rhs.denom)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational division (numerator product)",
            ))?;
        let d = self
            .denom
            .checked_mul(rhs.numer)
            .ok_or(SciError::ExactArithmeticOverflow(
                "overflow in rational division (denominator product)",
            ))?;
        Self::new(n, d)
    }

    /// Integer power $r^k$ with checked arithmetic.
    pub fn powi(&self, exp: i32) -> SciResult<Self> {
        if exp == 0 {
            return Ok(Self::one());
        }
        if exp > 0 {
            let n = self
                .numer
                .checked_pow(exp as u32)
                .ok_or(SciError::ExactArithmeticOverflow(
                    "rational numerator power overflow",
                ))?;
            let d = self
                .denom
                .checked_pow(exp as u32)
                .ok_or(SciError::ExactArithmeticOverflow(
                    "rational denominator power overflow",
                ))?;
            Self::new(n, d)
        } else {
            let inv = self.recip()?;
            inv.powi(-exp)
        }
    }

    /// Approximates a floating-point value with an exact rational using continued fractions.
    pub fn from_f64_approx(val: f64, max_denom: i128) -> SciResult<Self> {
        if val.is_nan() || val.is_infinite() {
            return Err(SciError::DomainError(
                "cannot convert NaN or Inf to Rational",
            ));
        }

        let sign = if val < 0.0 { -1 } else { 1 };
        let mut x = val.abs();

        let mut m00 = 1_i128;
        let mut m01 = 0_i128;
        let mut m10 = 0_i128;
        let mut m11 = 1_i128;

        while m10 * (x.floor() as i128) + m11 <= max_denom {
            let a = x.floor() as i128;
            let t0 = m00 * a + m01;
            let t1 = m10 * a + m11;
            m01 = m00;
            m00 = t0;
            m11 = m10;
            m10 = t1;

            let frac = x - x.floor();
            if frac < 1e-15 {
                break;
            }
            x = 1.0 / frac;
        }

        Self::new(sign * m00, m10)
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
            .expect("rational addition integer overflow")
    }
}

impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
            .expect("rational subtraction integer overflow")
    }
}

impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
            .expect("rational multiplication integer overflow")
    }
}

impl Div for Rational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(&rhs)
            .expect("rational division by zero or integer overflow")
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            numer: -self.numer,
            denom: self.denom,
        }
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.numer * other.denom).cmp(&(other.numer * self.denom))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "{}/{}", self.numer, self.denom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_canonical_reduction() {
        let r1 = Rational::new(6, 8).unwrap();
        assert_eq!(r1.numer(), 3);
        assert_eq!(r1.denom(), 4);

        let r2 = Rational::new(-4, -6).unwrap();
        assert_eq!(r2.numer(), 2);
        assert_eq!(r2.denom(), 3);
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::new(1, 3).unwrap();
        let b = Rational::new(1, 6).unwrap();

        assert_eq!(a + b, Rational::new(1, 2).unwrap());
        assert_eq!(a - b, Rational::new(1, 6).unwrap());
        assert_eq!(a * b, Rational::new(1, 18).unwrap());
        assert_eq!(a / b, Rational::from_integer(2));
    }

    #[test]
    fn test_rational_division_by_zero_error() {
        assert_eq!(Rational::new(1, 0), Err(SciError::DivisionByZero));
        let a = Rational::new(1, 2).unwrap();
        let zero = Rational::zero();
        assert_eq!(a.checked_div(&zero), Err(SciError::DivisionByZero));
    }

    #[test]
    fn test_rational_checked_overflow() {
        let huge = Rational::new(i128::MAX, 1).unwrap();
        let two = Rational::from_integer(2);
        assert!(huge.checked_mul(&two).is_err());
    }

    #[test]
    fn test_from_f64_approx() {
        let r = Rational::from_f64_approx(0.75, 1000).unwrap();
        assert_eq!(r, Rational::new(3, 4).unwrap());

        let pi_approx = Rational::from_f64_approx(std::f64::consts::PI, 1000).unwrap();
        assert_eq!(pi_approx, Rational::new(355, 113).unwrap());
    }
}
