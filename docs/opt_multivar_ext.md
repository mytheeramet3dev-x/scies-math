# `opt_multivar_ext` Module Documentation

Extended multivariate optimisation: trust-region, adaptive gradient methods,
constrained optimisation, and global population-based algorithms.

## Overview

This module extends the multivariate optimization layer into constrained and global-search methods.

| Algorithm | Type | Best for |
|---|---|---|
| `trust_region_cg` | 2nd-order unconstrained | General smooth f |
| `adam` | 1st-order adaptive | Large-scale / ML |
| `adagrad` | 1st-order adaptive | Sparse gradients |
| `rmsprop` | 1st-order adaptive | Non-stationary |
| `projected_gradient` | 1st-order box-constrained | Simple bound constraints |
| `augmented_lagrangian` | Constrained (eq+ineq) | General constrained |
| `particle_swarm` | Global / derivative-free | Multimodal global |
| `differential_evolution` | Global / derivative-free | Continuous global |

## Notes

- Use the constrained routines when feasibility matters as much as objective value.
- Use global methods when local methods are likely to get trapped in poor minima.
