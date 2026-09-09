//! Special Mathematical Functions
//!
//! This module provides high-precision approximations for special functions
//! commonly used in physics, statistics, and engineering.
//!
//! # Functions
//!
//! | Function | Description |
//! |---|---|
//! | `erf(x)` | Error function |
//! | `erfc(x)` | Complementary error function |
//! | `gamma(x)` | Gamma function $\Gamma(x)$ |
//! | `ln_gamma(x)` | Natural logarithm of the Gamma function $\ln\Gamma(x)$ |
//! | `beta(x, y)` | Beta function $B(x, y)$ |
//! | `bessel_j0(x)` | Bessel function of the first kind, order 0 |
//! | `bessel_j1(x)` | Bessel function of the first kind, order 1 |

/// Error function $\operatorname{erf}(x)$
///
/// Uses fractional approximation from Abramowitz and Stegun (Eq. 7.1.26).
/// Maximum error: $1.5 \times 10^{-7}$.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Complementary error function $\operatorname{erfc}(x) = 1 - \operatorname{erf}(x)$
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Natural logarithm of the Gamma function $\ln \Gamma(x)$
///
/// Uses the Lanczos approximation.
/// Highly accurate for all $x > 0$.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN; // Undefined for non-positive values
    }

    #[allow(clippy::excessive_precision)]
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut y = x;
    let mut tmp = x + 7.5;
    tmp = (x + 0.5) * tmp.ln() - tmp;

    let mut ser = p[0];
    for &pi in p.iter().skip(1) {
        y += 1.0;
        ser += pi / y;
    }

    let sqrt_2pi = 2.5066282746310005; // sqrt(2 * PI)
    tmp + (sqrt_2pi * ser / x).ln()
}

/// Gamma function $\Gamma(x)$
pub fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

/// Beta function $B(x, y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$
pub fn beta(x: f64, y: f64) -> f64 {
    (ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)).exp()
}

/// Bessel function of the first kind, order 0 ($J_0(x)$)
///
/// Polynomial approximation from Abramowitz and Stegun.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0
            + y * (1029532985.0
                + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7))));
        (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

/// Bessel function of the first kind, order 1 ($J_1(x)$)
///
/// Polynomial approximation from Abramowitz and Stegun.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let y;
    let ans1;
    let ans2;

    if ax < 8.0 {
        y = x * x;
        ans1 = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        ans2 = 144725228442.0
            + y * (2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        y = z * z;
        let xx = ax - 2.356194491;
        ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let val =
            (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 { -val } else { val }
    }
}
