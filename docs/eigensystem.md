# `eigensystem` Module Documentation

Full eigensystem solvers returning eigenvalues **and** eigenvectors.

| Function | Matrix type | Method |
|---|---|---|
| [`jacobi_eigen`]     | Real symmetric | Classic Jacobi off-diagonal pivoting |
| [`qr_eigen_general`] | General real   | QR iteration with Wilkinson shifts   |