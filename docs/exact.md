# `exact` Module Documentation

Exact mathematical core: canonical rational numbers and interval arithmetic error bounding.

## Overview

The `exact` module eliminates round-off errors and precision loss by providing exact algebraic arithmetic and rigorous error bounds.

| Type | Description |
|---|---|
| [`Rational`] | Irreducible fraction $p/q$ backed by `i128` with canonical GCD reduction |
| [`Interval`] | Real interval $[a, b]$ for Moore/Kahan interval arithmetic |

## Usage Examples

### Exact Rational Arithmetic

```rust
use scies_math_th::exact::Rational;

let a = Rational::new(1, 3).unwrap();
let b = Rational::new(1, 6).unwrap();
let sum = a + b; // Exactly 1/2
assert_eq!(sum, Rational::new(1, 2).unwrap());
```

### Interval Error Bounds

```rust
use scies_math_th::exact::Interval;

let x = Interval::new(1.99, 2.01).unwrap();
let y = Interval::new(2.99, 3.01).unwrap();
let prod = x * y;
println!("Bounded product: {}", prod); // [5.9501, 6.0501]
```
