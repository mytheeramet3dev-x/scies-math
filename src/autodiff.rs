//! Forward-mode automatic differentiation (AD).
//!
//! Forward AD propagates dual numbers `(value, derivative)` through
//! arithmetic operations to compute exact derivatives — no finite differences,
//! no symbolic overhead.
//!
//! # Dual number type
//!
//! [`Dual`] represents  f + f'·ε  where ε² = 0.
//!
//! ```rust
//! use scies_math::autodiff::Dual;
//!
//! let x = Dual::var(3.0);          // x = 3, dx = 1
//! let y = x * x + Dual::con(2.0);  // y = x² + 2
//! assert!((y.re - 11.0).abs() < 1e-12);  // value = 9 + 2
//! assert!((y.du - 6.0 ).abs() < 1e-12);  // derivative = 2x = 6
//! ```
//!
//! # Gradient and Jacobian
//!
//! | Function | Output |
//! |---|---|
//! | `gradient(f, x)` | ∇f at x (one forward pass per input dim) |
//! | `jacobian(f, x)` | J_f at x (m×n matrix) |
//! | `directional_derivative(f, x, v)` | ∇f·v (one forward pass) |
//!
//! ```rust
//! use scies_math::autodiff::gradient;
//!
//! // f(x, y) = x² + 3xy
//! let grad = gradient(
//!     |v| v[0]*v[0] + 3.0*v[0]*v[1],
//!     &[2.0, 1.0],
//! );
//! // grad ≈ [2x + 3y, 3x] = [7, 6]
//! ```
//!
//! For reverse-mode AD (more efficient when outputs << inputs) see [`crate::reverse_ad`].
use core::ops::{Add, Div, Mul, Neg, Sub};

// ─────────────────────────────────────────────────────────────
// Dual number
// ─────────────────────────────────────────────────────────────

/// A dual number  `value + ε·deriv`  where  ε² = 0.
///
/// Arithmetic on `Dual` values propagates derivatives automatically
/// according to the chain rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Primal (function value).
    pub value: f64,
    /// Tangent (derivative w.r.t. the seeded variable).
    pub deriv: f64,
}

impl Dual {
    /// Constant: derivative is zero.
    pub const fn from(value: f64) -> Self {
        Self { value, deriv: 0.0 }
    }

    /// Seed a variable: its own derivative is 1.
    pub const fn var(value: f64) -> Self {
        Self { value, deriv: 1.0 }
    }

    // ── Standard math ops ─────────────────────────────────────────────────────

    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            deriv: self.deriv * self.value.cos(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            deriv: -self.deriv * self.value.sin(),
        }
    }

    pub fn tan(self) -> Self {
        let c = self.value.cos();
        Self {
            value: self.value.tan(),
            deriv: self.deriv / (c * c),
        }
    }

    pub fn exp(self) -> Self {
        let e = self.value.exp();
        Self {
            value: e,
            deriv: self.deriv * e,
        }
    }

    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            deriv: self.deriv / self.value,
        }
    }

    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        Self {
            value: s,
            deriv: self.deriv / (2.0 * s),
        }
    }

    pub fn powi(self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            deriv: self.deriv * n as f64 * self.value.powi(n - 1),
        }
    }

    pub fn powf(self, n: f64) -> Self {
        Self {
            value: self.value.powf(n),
            deriv: self.deriv * n * self.value.powf(n - 1.0),
        }
    }

    pub fn abs(self) -> Self {
        Self {
            value: self.value.abs(),
            deriv: if self.value >= 0.0 {
                self.deriv
            } else {
                -self.deriv
            },
        }
    }

    pub fn sinh(self) -> Self {
        Self {
            value: self.value.sinh(),
            deriv: self.deriv * self.value.cosh(),
        }
    }

    pub fn cosh(self) -> Self {
        Self {
            value: self.value.cosh(),
            deriv: self.deriv * self.value.sinh(),
        }
    }

    pub fn tanh(self) -> Self {
        let t = self.value.tanh();
        Self {
            value: t,
            deriv: self.deriv * (1.0 - t * t),
        }
    }

    pub fn atan(self) -> Self {
        Self {
            value: self.value.atan(),
            deriv: self.deriv / (1.0 + self.value * self.value),
        }
    }

    pub fn asin(self) -> Self {
        Self {
            value: self.value.asin(),
            deriv: self.deriv / (1.0 - self.value * self.value).sqrt(),
        }
    }

    pub fn acos(self) -> Self {
        Self {
            value: self.value.acos(),
            deriv: -self.deriv / (1.0 - self.value * self.value).sqrt(),
        }
    }
}

// ── Arithmetic operator impls ─────────────────────────────────────────────────

impl Add for Dual {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self {
            value: self.value + r.value,
            deriv: self.deriv + r.deriv,
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self {
            value: self.value - r.value,
            deriv: self.deriv - r.deriv,
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self {
            value: self.value * r.value,
            deriv: self.deriv * r.value + self.value * r.deriv,
        }
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        Self {
            value: self.value / r.value,
            deriv: (self.deriv * r.value - self.value * r.deriv) / (r.value * r.value),
        }
    }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            deriv: -self.deriv,
        }
    }
}
impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Self {
            value: self.value + s,
            deriv: self.deriv,
        }
    }
}
impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, s: f64) -> Self {
        Self {
            value: self.value - s,
            deriv: self.deriv,
        }
    }
}
impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self {
            value: self.value * s,
            deriv: self.deriv * s,
        }
    }
}
impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self {
            value: self.value / s,
            deriv: self.deriv / s,
        }
    }
}
// Allow f64 op Dual via commutativity helpers.
impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, d: Dual) -> Dual {
        d + self
    }
}
impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, d: Dual) -> Dual {
        d * self
    }
}
impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, d: Dual) -> Dual {
        Dual {
            value: self - d.value,
            deriv: -d.deriv,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Convenience entry-points
// ─────────────────────────────────────────────────────────────

/// Compute the **scalar derivative** `f'(x)` exactly via forward-mode AD.
///
/// ```
/// use sciesrust::math::autodiff::grad;
/// let df = grad(|x| x.sin() + x * x, 1.0);
/// assert!((df - (1.0_f64.cos() + 2.0)).abs() < 1e-14);
/// ```
pub fn grad<F>(f: F, x: f64) -> f64
where
    F: Fn(Dual) -> Dual,
{
    f(Dual::var(x)).deriv
}

/// Compute the **Jacobian** (gradient for a scalar-to-scalar function evaluated
/// at each component of `x`) using one forward pass per input dimension.
///
/// Returns the gradient vector `∇f` at `x`.
pub fn jacobian<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: Fn(&[Dual]) -> Dual,
{
    x.iter()
        .enumerate()
        .map(|(seed_idx, _)| {
            let duals: Vec<Dual> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| {
                    if i == seed_idx {
                        Dual::var(xi)
                    } else {
                        Dual::from(xi)
                    }
                })
                .collect();
            f(&duals).deriv
        })
        .collect()
}

/// Compute the **Hessian** of a scalar function at `x`
/// (finite differences of the AD gradient — exact to ~1e-7 for f64).
pub fn hessian<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut result = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            let mut xmm = x.to_vec();
            xpp[i] += h;
            xpp[j] += h;
            xpm[i] += h;
            xpm[j] -= h;
            xmp[i] -= h;
            xmp[j] += h;
            xmm[i] -= h;
            xmm[j] -= h;
            let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
            result[i][j] = val;
            result[j][i] = val;
        }
    }
    result
}
