# `rng` Module Documentation

Xoshiro256** pseudo-random number generator + distribution samplers.

## Overview

Core random-number utilities and sampler helpers.

## Usage
```rust
use scies_math_th::rng::Rng;

let mut rng = Rng::new(42);
let u  = rng.rand01();          // Uniform [0, 1)
let z  = rng.randn();           // Standard Normal
let g  = rng.sample_gamma(2.0, 1.0); // Gamma(α=2, β=1)
```

## Notes

- Keep seeds explicit when you need reproducible experiments.
- For direct distribution work, prefer the distribution and sampler modules when possible.
