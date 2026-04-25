# `rng` Module Documentation

Xoshiro256** pseudo-random number generator + distribution samplers.

# Usage
```rust
use sciesrust::math::rng::Rng;

let mut rng = Rng::new(42);
let u  = rng.rand01();          // Uniform [0, 1)
let z  = rng.randn();           // Standard Normal
let g  = rng.sample_gamma(2.0, 1.0); // Gamma(α=2, β=1)
```