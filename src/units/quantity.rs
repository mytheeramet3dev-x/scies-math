//! Physical quantities with dimension-checked arithmetic and unit conversions.

use super::dimension::Dimension;
use crate::errors::{SciError, SciResult};
use core::fmt;
use core::ops::{Div, Mul, Neg};

/// A physical quantity consisting of a magnitude and an SI dimension vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// Numerical value in standard SI base units.
    pub value: f64,
    /// Physical dimension.
    pub dimension: Dimension,
}

impl Quantity {
    /// Creates a quantity with given value and dimension.
    pub const fn new(value: f64, dimension: Dimension) -> Self {
        Self { value, dimension }
    }

    /// Dimensionless scalar.
    pub const fn dimensionless(val: f64) -> Self {
        Self::new(val, Dimension::DIMENSIONLESS)
    }

    /// Length in meters ($\text{m}$).
    pub const fn meters(val: f64) -> Self {
        Self::new(val, Dimension::LENGTH)
    }

    /// Length in kilometers ($\text{km}$).
    pub fn kilometers(val: f64) -> Self {
        Self::new(val * 1000.0, Dimension::LENGTH)
    }

    /// Mass in kilograms ($\text{kg}$).
    pub const fn kilograms(val: f64) -> Self {
        Self::new(val, Dimension::MASS)
    }

    /// Mass in grams ($\text{g}$).
    pub fn grams(val: f64) -> Self {
        Self::new(val * 0.001, Dimension::MASS)
    }

    /// Time in seconds ($\text{s}$).
    pub const fn seconds(val: f64) -> Self {
        Self::new(val, Dimension::TIME)
    }

    /// Time in hours ($\text{h}$).
    pub fn hours(val: f64) -> Self {
        Self::new(val * 3600.0, Dimension::TIME)
    }

    /// Velocity in meters per second ($\text{m/s}$).
    pub const fn meters_per_second(val: f64) -> Self {
        Self::new(val, Dimension::VELOCITY)
    }

    /// Force in newtons ($\text{N} = \text{kg}\cdot\text{m}/\text{s}^2$).
    pub const fn newtons(val: f64) -> Self {
        Self::new(val, Dimension::FORCE)
    }

    /// Energy in joules ($\text{J} = \text{kg}\cdot\text{m}^2/\text{s}^2$).
    pub const fn joules(val: f64) -> Self {
        Self::new(val, Dimension::ENERGY)
    }

    /// Pressure in pascals ($\text{Pa} = \text{N}/\text{m}^2$).
    pub const fn pascals(val: f64) -> Self {
        Self::new(val, Dimension::PRESSURE)
    }

    /// Power in watts ($\text{W} = \text{J}/\text{s}$).
    pub const fn watts(val: f64) -> Self {
        Self::new(val, Dimension::POWER)
    }

    /// Adds two quantities, returning error if physical dimensions mismatch.
    pub fn add(&self, other: &Self) -> SciResult<Self> {
        if self.dimension != other.dimension {
            return Err(SciError::DimensionalMismatch {
                expected: format!("{}", self.dimension),
                found: format!("{}", other.dimension),
            });
        }
        Ok(Self::new(self.value + other.value, self.dimension))
    }

    /// Subtracts two quantities, returning error if physical dimensions mismatch.
    pub fn sub(&self, other: &Self) -> SciResult<Self> {
        if self.dimension != other.dimension {
            return Err(SciError::DimensionalMismatch {
                expected: format!("{}", self.dimension),
                found: format!("{}", other.dimension),
            });
        }
        Ok(Self::new(self.value - other.value, self.dimension))
    }

    /// Scales quantity by scalar $s$.
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.value * s, self.dimension)
    }

    /// Raises quantity to integer power $Q^k$.
    pub fn powi(&self, k: i8) -> Self {
        Self::new(self.value.powi(k as i32), self.dimension.powi(k))
    }

    /// Computes square root $\sqrt{Q}$, returning error if dimension exponents are not even.
    pub fn sqrt(&self) -> SciResult<Self> {
        if self.value < 0.0 {
            return Err(SciError::DomainError("sqrt of negative quantity"));
        }
        let d = self.dimension;
        if d.length % 2 != 0
            || d.mass % 2 != 0
            || d.time % 2 != 0
            || d.current % 2 != 0
            || d.temperature % 2 != 0
            || d.amount % 2 != 0
            || d.luminous % 2 != 0
        {
            return Err(SciError::InvalidParameter(
                "cannot compute square root of quantity with fractional dimension",
            ));
        }

        let half_dim = Dimension {
            length: d.length / 2,
            mass: d.mass / 2,
            time: d.time / 2,
            current: d.current / 2,
            temperature: d.temperature / 2,
            amount: d.amount / 2,
            luminous: d.luminous / 2,
        };

        Ok(Self::new(self.value.sqrt(), half_dim))
    }
}

impl Mul for Quantity {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.value * rhs.value, self.dimension * rhs.dimension)
    }
}

impl Div for Quantity {
    type Output = SciResult<Self>;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.value.abs() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        Ok(Self::new(
            self.value / rhs.value,
            self.dimension / rhs.dimension,
        ))
    }
}

impl Neg for Quantity {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.value, self.dimension)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity_operations() {
        // F = m * a
        let mass = Quantity::kilograms(2.0);
        let acc = Quantity::new(9.8, Dimension::ACCELERATION);
        let force = mass * acc;

        assert_eq!(force.dimension, Dimension::FORCE);
        assert!((force.value - 19.6).abs() < 1e-10);

        // Kinetic Energy = 0.5 * m * v^2
        let vel = Quantity::meters_per_second(10.0);
        let ke = mass * vel.powi(2);
        let ke_half = ke.scale(0.5);

        assert_eq!(ke_half.dimension, Dimension::ENERGY);
        assert_eq!(ke_half.value, 100.0);
    }

    #[test]
    fn test_dimensional_mismatch_error() {
        let length = Quantity::meters(5.0);
        let time = Quantity::seconds(2.0);

        // Adding length + time should fail
        assert!(length.add(&time).is_err());
    }
}
