# `rng_ext` Module Documentation

Phase 4 — High-quality RNG engines + rich sampling API.

| Engine | Period | Speed | Quality | Use-case |
|---|---|---|---|---|
| `Xoshiro256ss` | 2²⁵⁶ |  | BigCrush Yes | General purpose |
| `Pcg64` | 2¹²⁸ |  | BigCrush Yes | Multiple streams |
| `Wyrand` | 2⁶⁴  |  | PractRand Yes | Ultra-fast |
| `Mt19937` | 2¹⁹⁹³⁷ |  | Good | Legacy / ML compat |

All engines implement the `RngCore` trait, so they can be used uniformly
with `Sampler` and all distribution helpers.