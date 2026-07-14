# `complex` Module Documentation

Complex numbers for arithmetic, polar form, and elementary functions.

## Overview

[`Complex`] is a `Copy` struct `{ re: f64, im: f64 }` used throughout the crate whenever complex-valued numerics are required.

## Arithmetic

```rust
use scies_math_th::complex::Complex;

let a = Complex::new(3.0, 4.0);
let b = Complex::new(1.0, -2.0);

let sum  = a + b;             // 4 + 2i
let prod = a * b;             // 3·1 - 4·(-2) + (3·(-2) + 4·1)i = 11 - 2i
let mag  = a.modulus();       // 5.0
let arg  = a.argument();      // atan2(4,3) ≈ 0.9273 rad
let conj = a.conjugate();     // 3 - 4i
let inv  = a.inverse().unwrap();
```

## Polar form

```rust
use scies_math_th::complex::Complex;

let c = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_4);
// c ≈ √2 + √2·i
let (r, θ) = c.to_polar();
```

## Elementary functions

| Method | Description |
|---|---|
| `exp()` | eᶻ = eˣ(cos y + i sin y) |
| `ln()` | Natural logarithm |
| `sqrt()` | Principal square root |
| `pow(n)` | Integer power via De Moivre |
| `sin()`, `cos()`, `tan()` | Trigonometric |
| `sinh()`, `cosh()` | Hyperbolic |

## Notes

- Complex numbers are useful in signal processing, transforms, and spectral workflows.
- Prefer the module-level APIs when you need more than scalar operations.
