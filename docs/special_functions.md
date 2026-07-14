# `special_functions` Module Documentation

Special mathematical functions for scientific and engineering workflows.

## Overview

This module provides high-precision approximations for special functions commonly used in physics, statistics, and engineering.

## Functions

| Function | Description |
|---|---|
| `erf(x)` | Error function |
| `erfc(x)` | Complementary error function |
| `gamma(x)` | Gamma function $\Gamma(x)$ |
| `ln_gamma(x)` | Natural logarithm of the Gamma function $\ln\Gamma(x)$ |
| `beta(x, y)` | Beta function $B(x, y)$ |
| `bessel_j0(x)` | Bessel function of the first kind, order 0 |
| `bessel_j1(x)` | Bessel function of the first kind, order 1 |

## Notes

- These functions are useful in probability, diffusion models, and numerical integration.
- Prefer the dedicated API instead of approximating these manually in user code.
