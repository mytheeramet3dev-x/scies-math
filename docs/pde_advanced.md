# `pde_advanced` Module Documentation

Advanced PDE solvers beyond the basic heat-equation and Laplace relaxation.

| Solver | PDE | Method |
|---|---|---|
| [`wave_equation_1d`]          | 1-D wave equation u_tt = c²u_xx | Explicit leapfrog |
| [`advection_1d`]              | 1-D linear advection u_t + a·u_x = 0 | Upwind scheme |
| [`heat_equation_1d_implicit`] | 1-D heat equation (unconditionally stable) | Crank–Nicolson / Thomas algorithm |
| [`poisson_2d`]                | 2-D Poisson ∇²u = f | Gauss–Seidel / SOR |