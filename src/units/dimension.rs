//! SI 7-base dimensional analysis engine.
//!
//! Dimension vector $[L^\alpha, M^\beta, T^\gamma, I^\delta, \Theta^\epsilon, N^\zeta, J^\eta]$
//! representing the 7 fundamental physical dimensions of the International System of Units (SI).

use core::fmt;
use core::ops::{Div, Mul};

/// 7-base SI physical dimension vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dimension {
    /// Length ($L$) — SI base unit: meter ($\text{m}$).
    pub length: i8,
    /// Mass ($M$) — SI base unit: kilogram ($\text{kg}$).
    pub mass: i8,
    /// Time ($T$) — SI base unit: second ($\text{s}$).
    pub time: i8,
    /// Electric Current ($I$) — SI base unit: ampere ($\text{A}$).
    pub current: i8,
    /// Thermodynamic Temperature ($\Theta$) — SI base unit: kelvin ($\text{K}$).
    pub temperature: i8,
    /// Amount of Substance ($N$) — SI base unit: mole ($\text{mol}$).
    pub amount: i8,
    /// Luminous Intensity ($J$) — SI base unit: candela ($\text{cd}$).
    pub luminous: i8,
}

impl Dimension {
    /// Dimensionless scalar $[0, 0, 0, 0, 0, 0, 0]$.
    pub const DIMENSIONLESS: Self = Self {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };

    /// Length ($L$).
    pub const LENGTH: Self = Self {
        length: 1,
        ..Self::DIMENSIONLESS
    };
    /// Mass ($M$).
    pub const MASS: Self = Self {
        mass: 1,
        ..Self::DIMENSIONLESS
    };
    /// Time ($T$).
    pub const TIME: Self = Self {
        time: 1,
        ..Self::DIMENSIONLESS
    };
    /// Electric Current ($I$).
    pub const CURRENT: Self = Self {
        current: 1,
        ..Self::DIMENSIONLESS
    };
    /// Temperature ($\Theta$).
    pub const TEMPERATURE: Self = Self {
        temperature: 1,
        ..Self::DIMENSIONLESS
    };
    /// Amount of Substance ($N$).
    pub const AMOUNT: Self = Self {
        amount: 1,
        ..Self::DIMENSIONLESS
    };
    /// Luminous Intensity ($J$).
    pub const LUMINOUS: Self = Self {
        luminous: 1,
        ..Self::DIMENSIONLESS
    };

    // Derived Dimensions
    /// Velocity ($L T^{-1}$).
    pub const VELOCITY: Self = Self {
        length: 1,
        time: -1,
        ..Self::DIMENSIONLESS
    };
    /// Acceleration ($L T^{-2}$).
    pub const ACCELERATION: Self = Self {
        length: 1,
        time: -2,
        ..Self::DIMENSIONLESS
    };
    /// Force ($M L T^{-2}$, Newton).
    pub const FORCE: Self = Self {
        mass: 1,
        length: 1,
        time: -2,
        ..Self::DIMENSIONLESS
    };
    /// Energy / Work ($M L^2 T^{-2}$, Joule).
    pub const ENERGY: Self = Self {
        mass: 1,
        length: 2,
        time: -2,
        ..Self::DIMENSIONLESS
    };
    /// Pressure ($M L^{-1} T^{-2}$, Pascal).
    pub const PRESSURE: Self = Self {
        mass: 1,
        length: -1,
        time: -2,
        ..Self::DIMENSIONLESS
    };
    /// Power ($M L^2 T^{-3}$, Watt).
    pub const POWER: Self = Self {
        mass: 1,
        length: 2,
        time: -3,
        ..Self::DIMENSIONLESS
    };
    /// Frequency ($T^{-1}$, Hertz).
    pub const FREQUENCY: Self = Self {
        time: -1,
        ..Self::DIMENSIONLESS
    };

    /// Whether this dimension is purely dimensionless.
    pub fn is_dimensionless(&self) -> bool {
        *self == Self::DIMENSIONLESS
    }

    /// Raises dimension to integer power $D^k$.
    pub fn powi(&self, k: i8) -> Self {
        Self {
            length: self.length * k,
            mass: self.mass * k,
            time: self.time * k,
            current: self.current * k,
            temperature: self.temperature * k,
            amount: self.amount * k,
            luminous: self.luminous * k,
        }
    }
}

impl Mul for Dimension {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            length: self.length + rhs.length,
            mass: self.mass + rhs.mass,
            time: self.time + rhs.time,
            current: self.current + rhs.current,
            temperature: self.temperature + rhs.temperature,
            amount: self.amount + rhs.amount,
            luminous: self.luminous + rhs.luminous,
        }
    }
}

impl Div for Dimension {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            length: self.length - rhs.length,
            mass: self.mass - rhs.mass,
            time: self.time - rhs.time,
            current: self.current - rhs.current,
            temperature: self.temperature - rhs.temperature,
            amount: self.amount - rhs.amount,
            luminous: self.luminous - rhs.luminous,
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "dimensionless");
        }

        let mut num = Vec::new();
        let mut den = Vec::new();

        let bases = [
            ("m", self.length),
            ("kg", self.mass),
            ("s", self.time),
            ("A", self.current),
            ("K", self.temperature),
            ("mol", self.amount),
            ("cd", self.luminous),
        ];

        for (sym, exp) in bases {
            if exp > 1 {
                num.push(format!("{sym}^{exp}"));
            } else if exp == 1 {
                num.push(sym.to_string());
            } else if exp < -1 {
                den.push(format!("{sym}^{}", -exp));
            } else if exp == -1 {
                den.push(sym.to_string());
            }
        }

        match (num.is_empty(), den.is_empty()) {
            (false, true) => write!(f, "{}", num.join("·")),
            (false, false) => write!(f, "{}/{}", num.join("·"), den.join("·")),
            (true, false) => write!(f, "1/{}", den.join("·")),
            (true, true) => write!(f, "1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensional_arithmetic() {
        // Velocity = Length / Time
        assert_eq!(Dimension::LENGTH / Dimension::TIME, Dimension::VELOCITY);

        // Force = Mass * Acceleration
        assert_eq!(Dimension::MASS * Dimension::ACCELERATION, Dimension::FORCE);

        // Energy = Force * Length
        assert_eq!(Dimension::FORCE * Dimension::LENGTH, Dimension::ENERGY);

        // Pressure = Force / Area (Length^2)
        assert_eq!(
            Dimension::FORCE / Dimension::LENGTH.powi(2),
            Dimension::PRESSURE
        );
    }
}
