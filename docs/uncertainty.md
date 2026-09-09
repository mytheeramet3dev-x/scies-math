# `uncertainty` and `policy` Module Documentation

Uncertainty propagation and centralized numerical tolerance policies.

## Overview

- [`UncertainValue`]: Values with standard Gaussian uncertainty ($x \pm \sigma$) with first-order error propagation.
- [`TolerancePolicy`]: Centralized convergence policies (Strict $10^{-12}$, Standard $10^{-7}$, Relaxed $10^{-4}$) and floating-point safety checks.
