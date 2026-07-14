# `monte_carlo` Module Documentation

Monte Carlo integration and basic sampling.

## Overview

Monte Carlo utilities for stochastic estimation and simple variance-reduction workflows.

## Functions

| Function | Description |
|---|---|
| `monte_carlo_integrate` | Estimate ∫f(x)dx over a box via uniform sampling |
| `importance_sample` | Importance sampling with proposal distribution |
| `stratified_sample` | Stratified sampling for variance reduction |

## Usage

```rust
use scies_math_th::monte_carlo::monte_carlo_integrate;

// Estimate π by integrating f(x) = √(1-x²) over [0,1]
let pi_quarter = monte_carlo_integrate(
    |x| (1.0 - x[0]*x[0]).sqrt(),
    &[(0.0, 1.0)],  // bounds
    100_000,
    42,             // seed
);
// pi_quarter * 4 ≈ π
```

For advanced MCMC (HMC, Gibbs, Slice, Particle Filter) see [`crate::monte_carlo_ext`].

## Notes

- Monte Carlo methods trade deterministic error guarantees for flexibility.
- Use many samples and a stable seed when you want reproducible estimates.
