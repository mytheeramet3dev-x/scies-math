//! Univariate polynomial algebra and Euclidean operations.
//!
//! Represents a polynomial $P(x) = \sum_{i=0}^d a_i x^i$ with coefficients stored in ascending degree.

use super::expr::Expr;
use crate::errors::{SciError, SciResult};
use core::fmt;

/// A univariate polynomial with floating-point coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients in ascending degree order: `coeffs[i]` is coefficient of $x^i$.
    coeffs: Vec<f64>,
}

impl Polynomial {
    /// Creates a polynomial, automatically trimming trailing zero coefficients.
    pub fn new(mut coeffs: Vec<f64>) -> Self {
        while coeffs.len() > 1 && coeffs.last().is_some_and(|&c| c.abs() < 1e-15) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(0.0);
        }
        Self { coeffs }
    }

    /// Polynomial zero $P(x) = 0$.
    pub fn zero() -> Self {
        Self { coeffs: vec![0.0] }
    }

    /// Constant polynomial $P(x) = c$.
    pub fn constant(c: f64) -> Self {
        Self::new(vec![c])
    }

    /// Monomial polynomial $P(x) = x$.
    pub fn x() -> Self {
        Self::new(vec![0.0, 1.0])
    }

    /// Slice of coefficients.
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// Degree of the polynomial (0 for constant or zero polynomial).
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Leading coefficient $a_d$.
    pub fn leading_coeff(&self) -> f64 {
        *self.coeffs.last().unwrap_or(&0.0)
    }

    /// Checks whether $P(x) = 0$.
    pub fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].abs() < 1e-15
    }

    /// Evaluates $P(x)$ at given value using Horner's method.
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for &coeff in self.coeffs.iter().rev() {
            result = result * x + coeff;
        }
        result
    }

    /// Adds two polynomials $P(x) + Q(x)$.
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut sum_coeffs = vec![0.0; max_len];

        for (i, &c) in self.coeffs.iter().enumerate() {
            sum_coeffs[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            sum_coeffs[i] += c;
        }

        Self::new(sum_coeffs)
    }

    /// Subtracts two polynomials $P(x) - Q(x)$.
    pub fn sub(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut diff_coeffs = vec![0.0; max_len];

        for (i, &c) in self.coeffs.iter().enumerate() {
            diff_coeffs[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            diff_coeffs[i] -= c;
        }

        Self::new(diff_coeffs)
    }

    /// Multiplies two polynomials $P(x) \cdot Q(x)$ using Cauchy convolution.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut prod = vec![0.0; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                prod[i + j] += a * b;
            }
        }
        Self::new(prod)
    }

    /// Euclidean polynomial division $A(x) = B(x) Q(x) + R(x)$ where $\deg(R) < \deg(B)$.
    ///
    /// Returns `(Quotient, Remainder)`.
    pub fn div_rem(&self, divisor: &Self) -> SciResult<(Self, Self)> {
        if divisor.is_zero() {
            return Err(SciError::DivisionByZero);
        }

        if self.degree() < divisor.degree() {
            return Ok((Self::zero(), self.clone()));
        }

        let mut remainder = self.coeffs.clone();
        let divisor_deg = divisor.degree();
        let divisor_lead = divisor.leading_coeff();

        let quotient_deg = self.degree() - divisor_deg;
        let mut quotient = vec![0.0; quotient_deg + 1];

        for k in (0..=quotient_deg).rev() {
            let cur_deg = divisor_deg + k;
            let q_coeff = remainder[cur_deg] / divisor_lead;
            quotient[k] = q_coeff;

            for (j, &d_coeff) in divisor.coeffs.iter().enumerate() {
                remainder[k + j] -= q_coeff * d_coeff;
            }
        }

        Ok((Self::new(quotient), Self::new(remainder)))
    }

    /// Computes the formal derivative $P'(x) = \sum_{i=1}^d i a_i x^{i-1}$.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let diff_coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| i as f64 * c)
            .collect();
        Self::new(diff_coeffs)
    }

    /// Converts this polynomial into a symbolic expression [`Expr`].
    pub fn to_expr(&self, var: &str) -> Expr {
        let x = Expr::sym(var);
        let mut terms = Vec::new();

        for (deg, &coeff) in self.coeffs.iter().enumerate() {
            if coeff.abs() < 1e-15 {
                continue;
            }
            let c_expr = Expr::Rational(
                crate::exact::Rational::from_f64_approx(coeff, 10000)
                    .unwrap_or(crate::exact::Rational::zero()),
            );

            if deg == 0 {
                terms.push(c_expr);
            } else if deg == 1 {
                terms.push(Expr::Mul(vec![c_expr, x.clone()]));
            } else {
                terms.push(Expr::Mul(vec![
                    c_expr,
                    Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(deg as i128))),
                ]));
            }
        }

        if terms.is_empty() {
            Expr::int(0)
        } else if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            Expr::Add(terms)
        }
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-12 && self.coeffs.len() > 1 {
                continue;
            }
            if !first && c >= 0.0 {
                write!(f, " + ")?;
            } else if !first {
                write!(f, " - ")?;
            }
            let val = if first { c } else { c.abs() };
            if i == 0 {
                write!(f, "{:.2}", val)?;
            } else if i == 1 {
                write!(f, "{:.2}x", val)?;
            } else {
                write!(f, "{:.2}x^{i}", val)?;
            }
            first = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_arithmetic_and_horner() {
        // P(x) = 2x^2 + 3x + 1
        let p = Polynomial::new(vec![1.0, 3.0, 2.0]);
        // P(2) = 2(4) + 3(2) + 1 = 15
        assert_eq!(p.eval(2.0), 15.0);

        // P'(x) = 4x + 3
        let dp = p.derivative();
        assert_eq!(dp.eval(2.0), 11.0);
    }

    #[test]
    fn test_polynomial_euclidean_division() {
        // A(x) = (x + 1)(x + 2) = x^2 + 3x + 2
        let a = Polynomial::new(vec![2.0, 3.0, 1.0]);
        // B(x) = x + 1
        let b = Polynomial::new(vec![1.0, 1.0]);

        let (q, r) = a.div_rem(&b).unwrap();
        // Q(x) = x + 2, R(x) = 0
        assert_eq!(q.coeffs(), &[2.0, 1.0]);
        assert!(r.is_zero());
    }
}
