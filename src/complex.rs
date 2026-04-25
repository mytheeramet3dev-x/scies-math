//! Complex numbers — arithmetic, polar form, and elementary functions.
//!
//! [`Complex`] is a `Copy` struct `{ re: f64, im: f64 }`.
//!
//! # Arithmetic
//!
//! ```rust
//! use scies_math::complex::Complex;
//!
//! let a = Complex::new(3.0, 4.0);
//! let b = Complex::new(1.0, -2.0);
//!
//! let sum  = a + b;             // 4 + 2i
//! let prod = a * b;             // 3·1 - 4·(-2) + (3·(-2) + 4·1)i = 11 - 2i
//! let mag  = a.modulus();       // 5.0
//! let arg  = a.argument();      // atan2(4,3) ≈ 0.9273 rad
//! let conj = a.conjugate();     // 3 - 4i
//! let inv  = a.inverse().unwrap();
//! ```
//!
//! # Polar form
//!
//! ```rust
//! use scies_math::complex::Complex;
//!
//! let c = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_4);
//! // c ≈ √2 + √2·i
//! let (r, θ) = c.to_polar();
//! ```
//!
//! # Elementary functions
//!
//! | Method | Description |
//! |---|---|
//! | `exp()` | eᶻ = eˣ(cos y + i sin y) |
//! | `ln()` | Natural logarithm |
//! | `sqrt()` | Principal square root |
//! | `pow(n)` | Integer power via De Moivre |
//! | `sin()`, `cos()`, `tan()` | Trigonometric |
//! | `sinh()`, `cosh()` | Hyperbolic |
use crate::errors::{SciError, SciResult};
use core::ops::{Add, Div, Mul, Neg, Sub};

// ─────────────────────────────────────────────────────────────
// C64 — complex number
// ─────────────────────────────────────────────────────────────

/// A complex number `re + i·im` with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }
    pub const fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }
    pub fn from_real(r: f64) -> Self {
        Self { re: r, im: 0.0 }
    }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
    pub fn modulus(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn argument(self) -> f64 {
        self.im.atan2(self.re)
    }
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// e^z
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// ln z  (principal branch, argument in (-π, π])
    pub fn ln(self) -> SciResult<Self> {
        let m = self.modulus();
        if m < f64::EPSILON {
            return Err(SciError::DomainError("ln(0) is undefined"));
        }
        Ok(Self {
            re: m.ln(),
            im: self.argument(),
        })
    }

    /// Principal square root.
    pub fn sqrt(self) -> Self {
        let r = self.modulus().sqrt();
        let theta = self.argument() / 2.0;
        Self::from_polar(r, theta)
    }

    /// z^n for integer n.
    pub fn powi(self, n: i32) -> Self {
        let r = self.modulus().powi(n);
        let theta = self.argument() * n as f64;
        Self::from_polar(r, theta)
    }

    /// z^w (complex power, principal branch).
    pub fn pow(self, w: Self) -> SciResult<Self> {
        if self.modulus() < f64::EPSILON {
            if w.re > 0.0 {
                return Ok(Self::zero());
            }
            return Err(SciError::DomainError("0^w with re(w) <= 0"));
        }
        Ok((w * self.ln()?).exp())
    }

    /// sin z
    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// cos z
    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    /// tan z = sin z / cos z
    pub fn tan(self) -> SciResult<Self> {
        let c = self.cos();
        if c.modulus() < f64::EPSILON {
            return Err(SciError::DomainError("tan is undefined at this point"));
        }
        Ok(self.sin() / c)
    }

    /// sinh z
    pub fn sinh(self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    /// cosh z
    pub fn cosh(self) -> Self {
        Self {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }
}

// ── Operator impls ────────────────────────────────────────────────────────────

impl Add for C64 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.re + r.re, self.im + r.im)
    }
}
impl Sub for C64 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.re - r.re, self.im - r.im)
    }
}
impl Mul for C64 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.re * r.re - self.im * r.im,
            self.re * r.im + self.im * r.re,
        )
    }
}
impl Div for C64 {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        let denom = r.norm_sq();
        Self::new(
            (self.re * r.re + self.im * r.im) / denom,
            (self.im * r.re - self.re * r.im) / denom,
        )
    }
}
impl Neg for C64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Mul<f64> for C64 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}
impl Add<f64> for C64 {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Self::new(self.re + s, self.im)
    }
}
impl Sub<f64> for C64 {
    type Output = Self;
    fn sub(self, s: f64) -> Self {
        Self::new(self.re - s, self.im)
    }
}

// ── Root finding ──────────────────────────────────────────────────────────────

/// Exact roots of the quadratic `a·z² + b·z + c = 0` over ℂ.
///
/// Returns `(z1, z2)` where |z1| ≥ |z2|.
pub fn complex_roots_quadratic(a: f64, b: f64, c: f64) -> SciResult<(C64, C64)> {
    if a.abs() < f64::EPSILON {
        return Err(SciError::InvalidParameter(
            "leading coefficient must be non-zero",
        ));
    }
    let disc = C64::new(b * b - 4.0 * a * c, 0.0).sqrt();
    let two_a = 2.0 * a;
    let z1 = C64::new(-b, 0.0) + disc;
    let z2 = C64::new(-b, 0.0) - disc;
    let r1 = C64::new(z1.re / two_a, z1.im / two_a);
    let r2 = C64::new(z2.re / two_a, z2.im / two_a);
    if r1.modulus() >= r2.modulus() {
        Ok((r1, r2))
    } else {
        Ok((r2, r1))
    }
}

/// Newton's method for finding a complex root of `f(z) = 0`.
///
/// Requires both `f` and its derivative `df`.
pub fn complex_newton<F, DF>(
    f: F,
    df: DF,
    initial: C64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
    DF: Fn(C64) -> C64,
{
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }
    let mut z = initial;
    for _ in 0..max_iter {
        let fz = f(z);
        if fz.modulus() < tolerance {
            return Ok(z);
        }
        let dfz = df(z);
        if dfz.modulus() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        z = z - fz / dfz;
    }
    if f(z).modulus() < tolerance {
        Ok(z)
    } else {
        Err(SciError::NonConvergent("complex Newton"))
    }
}

// ── Contour integration ───────────────────────────────────────────────────────

/// Numerical integration of `f(t)·γ'(t)` along a circular contour
/// `γ(t) = centre + radius·e^{i·t}`, t ∈ [0, 2π], using the composite
/// Simpson rule with `n_points` sample points (must be even).
///
/// Returns the complex value of `∮_γ f(z) dz`.
pub fn cauchy_contour_integral<F>(f: F, centre: C64, radius: f64, n_points: usize) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
{
    if radius <= 0.0 {
        return Err(SciError::InvalidParameter("radius must be positive"));
    }
    if n_points < 4 || n_points % 2 != 0 {
        return Err(SciError::InvalidParameter(
            "n_points must be even and at least 4",
        ));
    }

    let pi2 = 2.0 * core::f64::consts::PI;
    let h = pi2 / n_points as f64;

    // γ(t) = centre + radius·e^{it},  γ'(t) = i·radius·e^{it}
    let sample = |k: usize| {
        let t = k as f64 * h;
        let e_it = C64::from_polar(1.0, t);
        let gamma_t = centre + e_it * radius;
        let gamma_prime_t = C64::i() * e_it * radius;
        f(gamma_t) * gamma_prime_t
    };

    let mut sum = sample(0) + sample(n_points); // endpoints
    for k in 1..n_points {
        let w = if k % 2 == 0 { 2.0 } else { 4.0 };
        sum = sum + sample(k) * w;
    }
    Ok(sum * (h / 3.0))
}

/// Estimate the **residue** of `f` at a simple pole `z0` via the limit
/// `lim_{z→z0} (z − z0)·f(z)`, evaluated numerically with step `h`.
pub fn residue_simple_pole<F>(f: F, z0: C64, h: f64) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
{
    if h <= 0.0 {
        return Err(SciError::InvalidParameter("h must be positive"));
    }
    // Average over 4 directions to reduce directional bias.
    let dirs = [
        C64::new(h, 0.0),
        C64::new(0.0, h),
        C64::new(-h, 0.0),
        C64::new(0.0, -h),
    ];
    let mut sum = C64::zero();
    for &d in &dirs {
        let z = z0 + d;
        sum = sum + d * f(z);
    }
    Ok(sum * 0.25)
}
