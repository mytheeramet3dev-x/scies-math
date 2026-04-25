# `eigensystem_ext` Module Documentation

Extended eigensystem solvers.

| Function | Problem | Method |
|---|---|---|
| [`generalized_eigen_sym`] | Ax = λBx, B SPD | Cholesky transform → standard |
| [`inverse_iteration`] | Refine eigenpair near σ | Shift-and-invert power iter |
| [`rayleigh_quotient_iter`] | Find eigenpair from hint x₀ | Cubic convergence |
| [`simultaneous_iteration`] | Dominant k eigenpairs | QR iteration on tall matrix |