# `pde_advanced` Module Documentation

Advanced PDE solvers beyond the basic heat-equation and Laplace relaxation.

## Overview

This module expands the core PDE layer with solvers that are useful for wave propagation, advection, diffusion, and elliptic problems.

| Solver | PDE | Method |
|---|---|---|
| [`wave_equation_1d`]          | 1-D wave equation u_tt = c²u_xx | Explicit leapfrog |
| [`advection_1d`]              | 1-D linear advection u_t + a·u_x = 0 | Upwind scheme |
| [`heat_equation_1d_implicit`] | 1-D heat equation (unconditionally stable) | Crank–Nicolson / Thomas algorithm |
| [`poisson_2d`]                | 2-D Poisson ∇²u = f | Gauss–Seidel / SOR |

## Notes

- Use the implicit heat solver when stability is more important than a simple explicit update.
- Advection and wave problems are sensitive to numerical scheme choice and time-step size.
