# `autodiff_ext` Module Documentation

Advanced autodiff utilities built on top of `autodiff` (forward) and `reverse_ad`.

## Overview

This module extends the core autodiff layers with vector-calculus helpers, gradient checks, and higher-order derivative routines.

| Feature | Functions |
|---|---|
| Forward-mode extras | `directional_deriv`, `jvp`, full Jacobian matrix |
| Reverse-mode extras | `vjp`, full Jacobian (reverse), gradient norm |
| Vector calculus | `divergence`, `curl_2d`, `curl_3d`, `laplacian` |
| Verification | `gradient_check` (finite-diff vs AD) |
| Higher-order | `second_deriv`, `third_deriv` (forward stacking) |

## Usage

Use this module when you need derivative helpers beyond the basic forward- or reverse-mode APIs.

## Notes

- The goal is convenience and correctness checks, not replacing a full symbolic algebra system.
- Prefer the core autodiff modules when you only need gradients or Jacobians.
