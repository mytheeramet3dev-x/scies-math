# `moment` Module Documentation

Moment-SOS hierarchy, canonical monomial indexing, Boolean quotient reductions, 3-SAT clause encodings, and SDPA format I/O.

## Submodules

- `monomial`: Canonical DegRevLex ordered monomials, Boolean reductions ($x_i^2 = x_i$ for $\{0, 1\}$ or $x_i^2 = 1$ for $\{-1, +1\}$).
- `builder`: Moment matrix $M_d(y)$ and localizing matrix builders.
- `sat_encoding`: CNF and 3-SAT clause polynomial representation and SDP relaxation compilation.
- `sdpa_io`: Export and import of universal SDPA sparse format (`.dat-s`).
- `log`: Standardized calculation log (`CalculationLog`) recording crate version, seed, tolerance, dimension, residuals, and spectrum.
