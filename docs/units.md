# `units` Module Documentation

Physical quantities with dimension-checked arithmetic (SI 7-base dimensions).

## Overview

Prevents physical modeling and unit mismatch bugs by validating dimensions at calculation time.

| Type | Description |
|---|---|
| [`Dimension`] | 7-base SI dimension vector $[L, M, T, I, \Theta, N, J]$ |
| [`Quantity`] | Magnitude with physical dimension (supports meters, kg, seconds, newtons, joules, pascals, etc.) |

## Usage Examples

```rust
use scies_math_th::units::{Quantity, Dimension};

// Force = mass * acceleration
let m = Quantity::kilograms(5.0);
let a = Quantity::new(9.8, Dimension::ACCELERATION);
let f = m * a; // 49 N (kg*m/s^2)
println!("Force: {}", f);
```
