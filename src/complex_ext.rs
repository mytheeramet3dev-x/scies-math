//! Extended complex analysis: conformal maps, Laurent series, residues, polynomial roots,
//! complex integration, Blaschke products, multi-branch logarithm.
//!
//! All functions use [`crate::complex::C64`].

use crate::complex::C64;
use crate::errors::{SciError, SciResult};

// ══════════════════════════════════════════════════════════════════════════════
// Complex polynomial operations
// ══════════════════════════════════════════════════════════════════════════════

/// Evaluate a complex polynomial p(z) = a[0] + a[1]z + … + a[n]zⁿ via Horner.
pub fn poly_eval(coeffs: &[C64], z: C64) -> C64 {
    coeffs
        .iter()
        .rev()
        .fold(C64::from_real(0.0), |acc, &c| acc * z + c)
}

/// Derivative of a complex polynomial (coefficients of p'(z)).
pub fn poly_deriv(coeffs: &[C64]) -> Vec<C64> {
    if coeffs.len() <= 1 {
        return vec![C64::from_real(0.0)];
    }
    coeffs[1..]
        .iter()
        .enumerate()
        .map(|(i, &c)| c * C64::from_real((i + 1) as f64))
        .collect()
}

/// Find all roots of a complex polynomial using **Durand-Kerner** iteration.
///
/// Starting from a spread of initial guesses, converges to all `n` roots simultaneously.
/// `coeffs` = [a₀, a₁, …, aₙ] with aₙ ≠ 0 (leading coefficient).
pub fn poly_roots(coeffs: &[C64], tolerance: f64, max_iter: usize) -> SciResult<Vec<C64>> {
    let n = coeffs.len();
    if n < 2 {
        return Err(SciError::InvalidParameter(
            "polynomial must have degree ≥ 1",
        ));
    }
    if coeffs[n - 1].modulus() < f64::EPSILON {
        return Err(SciError::InvalidParameter(
            "leading coefficient must be non-zero",
        ));
    }
    let degree = n - 1;
    // Monic polynomial: divide by leading coefficient
    let lead = coeffs[n - 1];
    let monic: Vec<C64> = coeffs
        .iter()
        .map(|&c| {
            let r = c / lead;
            r
        })
        .collect();

    // Initial guesses: z_k = 0.4 + 0.9j)^k
    let mut roots: Vec<C64> = (0..degree)
        .map(|k| C64::from_polar(0.4, 0.9 * k as f64))
        .collect();

    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..degree {
            let pz = poly_eval(&monic, roots[i]);
            let denom: C64 = (0..degree)
                .filter(|&j| j != i)
                .map(|j| roots[i] - roots[j])
                .fold(C64::from_real(1.0), |acc, d| acc * d);
            if denom.modulus() < f64::EPSILON {
                continue;
            }
            let step = pz / denom;
            roots[i] = roots[i] - step;
            if step.modulus() > tolerance {
                converged = false;
            }
        }
        if converged {
            break;
        }
    }
    Ok(roots)
}

/// Multiply two polynomials given as coefficient vectors.
pub fn poly_mul(a: &[C64], b: &[C64]) -> Vec<C64> {
    let na = a.len();
    let nb = b.len();
    let mut c = vec![C64::from_real(0.0); na + nb - 1];
    for i in 0..na {
        for j in 0..nb {
            c[i + j] = c[i + j] + a[i] * b[j];
        }
    }
    c
}

// ══════════════════════════════════════════════════════════════════════════════
// Multi-branch complex functions
// ══════════════════════════════════════════════════════════════════════════════

/// Principal value logarithm Log(z) = ln|z| + i·Arg(z), Arg ∈ (−π, π].
pub fn log_principal(z: C64) -> SciResult<C64> {
    if z.modulus() < f64::EPSILON {
        return Err(SciError::DomainError("log(0) undefined"));
    }
    Ok(C64 {
        re: z.modulus().ln(),
        im: z.argument(),
    })
}

/// Generalised logarithm with branch `k`: Log_k(z) = Log(z) + 2πki.
pub fn log_branch(z: C64, k: i32) -> SciResult<C64> {
    let p = log_principal(z)?;
    Ok(C64 {
        re: p.re,
        im: p.im + 2.0 * core::f64::consts::PI * k as f64,
    })
}

/// Complex arc-sine: arcsin(z) = −i · Log(iz + √(1 − z²)).
pub fn asin_c(z: C64) -> SciResult<C64> {
    let one = C64::from_real(1.0);
    let iz = z * C64 { re: 0.0, im: 1.0 };
    let inner = (one - z * z).sqrt();
    let r = log_principal(iz + inner)?;
    Ok(r * C64 { re: 0.0, im: -1.0 })
}

/// Complex arc-cosine: arccos(z) = −i · Log(z + i√(1 − z²)).
pub fn acos_c(z: C64) -> SciResult<C64> {
    let one = C64::from_real(1.0);
    let i = C64 { re: 0.0, im: 1.0 };
    let inner = i * (one - z * z).sqrt();
    let r = log_principal(z + inner)?;
    Ok(r * C64 { re: 0.0, im: -1.0 })
}

/// Complex arc-tangent: arctan(z) = (i/2) · Log((i+z)/(i−z)).
pub fn atan_c(z: C64) -> SciResult<C64> {
    let i = C64 { re: 0.0, im: 1.0 };
    let num = i + z;
    let den = i - z;
    if den.modulus() < f64::EPSILON {
        return Err(SciError::DomainError("atan(±i) undefined"));
    }
    let r = log_principal(num / den)?;
    Ok(r * C64 { re: 0.0, im: 0.5 })
}

/// Complex hyperbolic arc-sine: arcsinh(z) = Log(z + √(z²+1)).
pub fn asinh_c(z: C64) -> SciResult<C64> {
    let one = C64::from_real(1.0);
    log_principal(z + (z * z + one).sqrt())
}

/// Complex hyperbolic arc-cosine: arccosh(z) = Log(z + √(z²−1)).
pub fn acosh_c(z: C64) -> SciResult<C64> {
    let one = C64::from_real(1.0);
    log_principal(z + (z * z - one).sqrt())
}

// ══════════════════════════════════════════════════════════════════════════════
// Conformal maps
// ══════════════════════════════════════════════════════════════════════════════

/// **Möbius (bilinear) transformation**: f(z) = (az + b)/(cz + d).
///
/// Maps circles/lines to circles/lines. Requires ad − bc ≠ 0.
pub fn mobius(z: C64, a: C64, b: C64, c: C64, d: C64) -> SciResult<C64> {
    let det = a * d - b * c;
    if det.modulus() < f64::EPSILON {
        return Err(SciError::InvalidParameter("Möbius: ad−bc must be non-zero"));
    }
    let denom = c * z + d;
    if denom.modulus() < f64::EPSILON {
        return Err(SciError::DomainError("Möbius: pole at this z"));
    }
    Ok((a * z + b) / denom)
}

/// **Joukowski transform**: w = z + c²/z  (aerodynamic airfoil mapping).
pub fn joukowski(z: C64, c: f64) -> SciResult<C64> {
    if z.modulus() < f64::EPSILON {
        return Err(SciError::DomainError("Joukowski: z ≠ 0"));
    }
    Ok(z + C64::from_real(c * c) / z)
}

/// **Schwarz-Christoffel-like**: exponential map w = exp(z) (strip → half-plane).
pub fn exp_map(z: C64) -> C64 {
    z.exp()
}

/// **Square root map** w = √z (half-plane → quarter-plane).
pub fn sqrt_map(z: C64) -> C64 {
    z.sqrt()
}

/// **Inverse Joukowski**: find preimage z given w on the airfoil (principal branch).
///
/// Solves z² − w·z + c² = 0.
pub fn joukowski_inv(w: C64, c: f64) -> SciResult<(C64, C64)> {
    let disc = (w * w - C64::from_real(4.0 * c * c)).sqrt();
    Ok((
        (w + disc) * C64::from_real(0.5),
        (w - disc) * C64::from_real(0.5),
    ))
}

// ══════════════════════════════════════════════════════════════════════════════
// Residues and Laurent series
// ══════════════════════════════════════════════════════════════════════════════

/// Residue at a **double pole** z₀ of f(z) = lim_{z→z₀} d/dz[(z−z₀)²·f(z)].
///
/// Computed numerically via central difference.
pub fn residue_double_pole<F>(f: F, z0: C64, h: f64) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
{
    // d/dz[(z-z0)²f(z)] at z=z0 via finite difference
    let dh = C64 { re: h, im: 0.0 };
    let fp = {
        let z = z0 + dh;
        f(z) * ((z - z0) * (z - z0))
    };
    let fm = {
        let z = z0 - dh;
        f(z) * ((z - z0) * (z - z0))
    };
    Ok((fp - fm) * C64::from_real(1.0 / (2.0 * h)))
}

/// Compute the **winding number** of a closed path around `z0`.
///
/// `path` is a sequence of points on the closed contour (last → first closes it).
/// Returns an integer (as f64) indicating how many times the path winds around z0.
pub fn winding_number(path: &[C64], z0: C64) -> f64 {
    let n = path.len();
    if n < 2 {
        return 0.0;
    }
    let mut total_angle = 0.0f64;
    for i in 0..n {
        let p1 = path[i] - z0;
        let p2 = path[(i + 1) % n] - z0;
        let angle = (p1.conjugate() * p2).argument();
        total_angle += angle;
    }
    total_angle / (2.0 * core::f64::consts::PI)
}

/// **Argument principle**: count zeros minus poles inside a contour.
///
/// Numerically computes (1/2πi) ∮ f'(z)/f(z) dz.
pub fn argument_principle<F>(f: F, path: &[C64], h: f64) -> SciResult<f64>
where
    F: Fn(C64) -> C64,
{
    let n = path.len();
    if n < 3 {
        return Err(SciError::InvalidParameter("path must have ≥ 3 points"));
    }
    // Numerical derivative f'(z) via central difference, then integrate f'/f
    let mut integral = C64::from_real(0.0);
    let dh = C64 { re: h, im: 0.0 };
    for i in 0..n {
        let z0 = path[i];
        let z1 = path[(i + 1) % n];
        let dz = z1 - z0;
        let zm = (z0 + z1) * C64::from_real(0.5); // midpoint
        let fp = (f(zm + dh) - f(zm - dh)) * C64::from_real(1.0 / (2.0 * h));
        let fv = f(zm);
        if fv.modulus() > f64::EPSILON {
            integral = integral + fp / fv * dz;
        }
    }
    let two_pi_i = C64 {
        re: 0.0,
        im: 2.0 * core::f64::consts::PI,
    };
    let result = integral / two_pi_i;
    Ok(result.re.round())
}

// ══════════════════════════════════════════════════════════════════════════════
// Blaschke product
// ══════════════════════════════════════════════════════════════════════════════

/// Evaluate a **Blaschke product** B(z) = ∏ (z − aₖ)/(1 − ā_k·z) for |aₖ| < 1.
///
/// Each factor maps the unit disk to itself; |B(z)| = 1 on the unit circle.
pub fn blaschke_product(z: C64, zeros: &[C64]) -> SciResult<C64> {
    for &a in zeros {
        if a.modulus() >= 1.0 {
            return Err(SciError::DomainError("Blaschke zeros must satisfy |a| < 1"));
        }
    }
    let mut b = C64::from_real(1.0);
    for &a in zeros {
        let num = z - a;
        let den = C64::from_real(1.0) - a.conjugate() * z;
        if den.modulus() < f64::EPSILON {
            return Err(SciError::DomainError("Blaschke: denominator zero"));
        }
        b = b * (num / den);
    }
    Ok(b)
}

// ══════════════════════════════════════════════════════════════════════════════
// High-precision complex integration (Gauss-Legendre quadrature)
// ══════════════════════════════════════════════════════════════════════════════

/// Integrate f(z) along a straight-line segment from `z_start` to `z_end`
/// using **5-point Gauss-Legendre** quadrature (exact for polynomials of degree ≤ 9).
pub fn gauss_legendre_segment<F>(f: F, z_start: C64, z_end: C64) -> C64
where
    F: Fn(C64) -> C64,
{
    // Nodes and weights for 5-point GL on [-1, 1]
    let nodes = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683,
        0.0,
        0.538_469_310_105_683,
        0.906_179_845_938_664,
    ];
    let weights = [
        0.236_926_885_056_189,
        0.478_628_670_499_366,
        0.568_888_888_888_889,
        0.478_628_670_499_366,
        0.236_926_885_056_189,
    ];
    let mid = (z_start + z_end) * C64::from_real(0.5);
    let half = (z_end - z_start) * C64::from_real(0.5);
    nodes
        .iter()
        .zip(weights.iter())
        .map(|(&t, &w)| {
            let z = mid + half * C64::from_real(t);
            f(z) * C64::from_real(w)
        })
        .fold(C64::from_real(0.0), |a, b| a + b)
        * half
}

/// Integrate f(z) along a **polygonal path** (sequence of line segments)
/// using Gauss-Legendre on each segment.
pub fn path_integral<F>(f: F, path: &[C64]) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
{
    if path.len() < 2 {
        return Err(SciError::InvalidParameter("path must have ≥ 2 points"));
    }
    let mut result = C64::from_real(0.0);
    for i in 0..(path.len() - 1) {
        result = result + gauss_legendre_segment(&f, path[i], path[i + 1]);
    }
    Ok(result)
}

/// Compute a **circular contour integral** ∮_C f(z) dz where C is the circle |z − z0| = r,
/// using N-point trapezoidal rule (exponentially accurate for analytic f).
pub fn circular_integral<F>(f: F, z0: C64, r: f64, n_points: usize) -> SciResult<C64>
where
    F: Fn(C64) -> C64,
{
    if r <= 0.0 {
        return Err(SciError::InvalidParameter("radius must be positive"));
    }
    if n_points < 4 {
        return Err(SciError::InvalidParameter("n_points >= 4"));
    }
    let pi = core::f64::consts::PI;
    let dt = 2.0 * pi / n_points as f64;
    let mut sum = C64::from_real(0.0);
    for k in 0..n_points {
        let t = k as f64 * dt;
        let z = z0
            + C64 {
                re: r * t.cos(),
                im: r * t.sin(),
            };
        let dz = C64 {
            re: -r * t.sin(),
            im: r * t.cos(),
        } * C64::from_real(dt);
        sum = sum + f(z) * dz;
    }
    Ok(sum)
}

// ══════════════════════════════════════════════════════════════════════════════
// Partial fraction decomposition (over simple poles)
// ══════════════════════════════════════════════════════════════════════════════

/// Partial fraction decomposition of N(z)/D(z) over **simple poles** (roots of D).
///
/// Returns `(residues, poles)` where N/D ≈ Σ residues[k] / (z − poles[k]).
///
/// Requires `deg(N) < deg(D)` and D having no repeated roots.
pub fn partial_fractions(
    num: &[C64],
    den: &[C64],
    tol: f64,
    max_iter: usize,
) -> SciResult<(Vec<C64>, Vec<C64>)> {
    let poles = poly_roots(den, tol, max_iter)?;
    let den_deriv = poly_deriv(den);
    let residues: Vec<C64> = poles
        .iter()
        .map(|&p| {
            let n = poly_eval(num, p);
            let d = poly_eval(&den_deriv, p);
            if d.modulus() < f64::EPSILON {
                C64::from_real(0.0)
            } else {
                n / d
            }
        })
        .collect();
    Ok((residues, poles))
}

// ══════════════════════════════════════════════════════════════════════════════
// Operator overloading helpers (if not already in C64)
// ══════════════════════════════════════════════════════════════════════════════

/// Add f64 to C64.
pub fn c(re: f64, im: f64) -> C64 {
    C64 { re, im }
}
