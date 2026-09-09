# Exact Arithmetic and Interval Bounds (`exact`)

The `exact` module provides arbitrary-precision-safe rational arithmetic (`Rational`) and interval arithmetic (`Interval`) for high-assurance calculations where floating-point approximation error is unacceptable.

---

## 1. 128-Bit Checked Rational Arithmetic (`Rational`)

A rational number is stored in canonical form as a numerator $p \in \mathbb{Z}$ and denominator $q \in \mathbb{N}^+$:

$$r = \frac{p}{q}, \quad \gcd(|p|, q) = 1, \quad q > 0$$

Stored using 128-bit signed integers (`i128`), allowing exact integer operations up to $\pm (2^{127} - 1) \approx \pm 1.7 \times 10^{38}$.

### Invariants Enforced
- **Zero Denominator Rejection**: `Rational::new(p, 0)` returns `Err(SciError::DivisionByZero)`.
- **Automatic Canonical Reduction**: Simplifies via Euclidean Greatest Common Divisor ($\gcd$) upon instantiation.
- **Checked Arithmetic**: To prevent silent integer overflow, all operations offer checked variants:
  - `checked_add(&self, other: &Self) -> SciResult<Self>`
  - `checked_sub(&self, other: &Self) -> SciResult<Self>`
  - `checked_mul(&self, other: &Self) -> SciResult<Self>`
  - `checked_div(&self, other: &Self) -> SciResult<Self>`
  - `checked_powi(&self, exponent: i32) -> SciResult<Self>`
- Overflow errors return `Err(SciError::ExactArithmeticOverflow)`.

### Stern-Brocot Rational Reconstruction

Reconstructs the closest irreducible rational approximation of a floating-point number within a bounded denominator $q \le \text{max\_denom}$:

```rust
use scies_math_th::exact::Rational;

let pi_approx = Rational::from_f64_approx(3.141592653589793, 1000)?;
assert_eq!(pi_approx, Rational::new(355, 113)?);
```

---

## 2. Interval Arithmetic (`Interval`)

Represents a real scalar as a closed bounded interval $[a, b] = \{x \in \mathbb{R} : a \le x \le b\}$.

### Arithmetic Operations
- **Addition**: $[a, b] + [c, d] = [a + c, b + d]$
- **Subtraction**: $[a, b] - [c, d] = [a - d, b - c]$
- **Multiplication**: $[a, b] \cdot [c, d] = [\min(ac, ad, bc, bd), \max(ac, ad, bc, bd)]$
- **Division**: $[a, b] / [c, d] = [a, b] \cdot [1/d, 1/c]$ (Requires $0 \notin [c, d]$)

### Numerical Limitations Notice
In version 0.3.0, `Interval` uses IEEE 754 64-bit floating-point numbers without hardware-directed outward rounding. While it captures interval spread under standard rounding, sub-epsilon rigorous enclosures require exact arithmetic (`Rational`) or outward rounding modes.

---

## 3. Code Example

```rust
use scies_math_th::exact::{Interval, Rational};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Exact rational computation
    let r1 = Rational::new(1, 3)?;
    let r2 = Rational::new(5, 6)?;

    let sum = r1.checked_add(&r2)?;
    assert_eq!(sum, Rational::new(7, 6)?);

    let prod = r1.checked_mul(&r2)?;
    assert_eq!(prod, Rational::new(5, 18)?);

    // Interval bounding
    let i1 = Interval::new(1.0, 2.0)?;
    let i2 = Interval::new(3.0, 4.0)?;

    let i_sum = (i1 + i2)?;
    assert_eq!(i_sum.lower(), 4.0);
    assert_eq!(i_sum.upper(), 6.0);

    assert!(i_sum.contains(5.0));

    Ok(())
}
```
