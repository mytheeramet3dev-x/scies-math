//! Exact mathematics core: arbitrary-precision rational arithmetic and interval bounds.

pub mod interval;
pub mod rational;

pub use interval::Interval;
pub use rational::Rational;
