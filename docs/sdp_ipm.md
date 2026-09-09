# `sdp_ipm` Module Documentation

High-accuracy Primal-Dual Interior-Point Method (Mehrotra Predictor-Corrector) for Semidefinite Programming.

## Overview

Solves standard semidefinite programs to proof-grade precision ($10^{-8}$ to $10^{-12}$):

$$\min \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i, \; X \succeq 0$$

## Usage Examples

```rust
use scies_math_th::symmetric::SymmetricMatrix;
use scies_math_th::sdp::SdpProblem;
use scies_math_th::sdp_ipm::{solve_sdp_ipm, SdpIpmConfig};

let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0]).unwrap();
let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
let problem = SdpProblem::new(c, vec![a1], vec![3.0]).unwrap();

let config = SdpIpmConfig::default();
let solution = solve_sdp_ipm(&problem, &config).unwrap();
println!("Optimal Primal Obj: {}", solution.primal_objective);
```
