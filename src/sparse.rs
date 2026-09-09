//! Sparse matrix algebra (CSR / COO) and iterative linear solvers.
//!
//! # Data structures
//! - [`SparseMatrixCsr`] — Compressed Sparse Row storage (from COO or dense)
//! - [`IluPreconditioner`] — ILU(0) incomplete factorisation
//! - [`DiagPreconditioner`] — Jacobi (diagonal) preconditioner
//!
//! # Solvers
//! | Method | System type | Preconditioned |
//! |---|---|---|
//! | `conjugate_gradient` | SPD | — |
//! | `preconditioned_cg` | SPD | Yes |
//! | `gmres` | General | — |
//! | `preconditioned_gmres` | General | Yes |
//! | `bicgstab` | General | — |
//! | `minres` | Symmetric (possibly indefinite) | — |

use crate::errors::{SciError, SciResult};

// ──────────────────────────────────────────────────────────────────────────────
// CSR Sparse Matrix
// ──────────────────────────────────────────────────────────────────────────────

/// Compressed Sparse Row matrix.
///
/// - `values[k]`      — non-zero value at position `k`
/// - `col_indices[k]` — column index of `values[k]`
/// - `row_pointers[i]..row_pointers[i+1]` — range of non-zeros in row `i`
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrixCsr {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    col_indices: Vec<usize>,
    row_pointers: Vec<usize>,
}

impl SparseMatrixCsr {
    /// Build from coordinate (COO) triplets `(row, col, value)`.
    /// Duplicate `(row, col)` entries are summed.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> SciResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SciError::InvalidParameter("dimensions must be positive"));
        }
        for &(r, c, _) in triplets {
            if r >= rows || c >= cols {
                return Err(SciError::InvalidParameter(
                    "triplet index out of matrix bounds",
                ));
            }
        }

        // Sort by (row, col) then accumulate duplicates.
        let mut sorted = triplets.to_vec();
        sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut values: Vec<f64> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut row_pointers = vec![0usize; rows + 1];

        for (r, c, v) in sorted {
            if let (Some(&last_c), Some(last_v)) = (col_indices.last(), values.last_mut()) {
                if col_indices.len() > row_pointers[r] && last_c == c {
                    *last_v += v;
                    continue;
                }
            }
            col_indices.push(c);
            values.push(v);
            row_pointers[r + 1] += 1;
        }

        // Convert counts → cumulative pointers.
        for i in 0..rows {
            row_pointers[i + 1] += row_pointers[i];
        }

        Ok(Self {
            rows,
            cols,
            values,
            col_indices,
            row_pointers,
        })
    }

    /// Number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Validates all internal CSR structure invariants.
    ///
    /// # Invariants checked:
    /// - `row_pointers.len() == rows + 1`
    /// - `row_pointers[0] == 0`
    /// - `row_pointers` is monotonically non-decreasing
    /// - `values.len() == col_indices.len() == row_pointers[rows]`
    /// - `col_indices[k] < cols` for all non-zeros
    /// - Column indices are strictly increasing within each row
    /// - All stored values are finite (not NaN or Inf)
    pub fn validate_invariants(&self) -> SciResult<()> {
        if self.row_pointers.len() != self.rows + 1 {
            return Err(SciError::InvalidParameter(
                "CSR row_pointers length must equal rows + 1",
            ));
        }
        if self.row_pointers[0] != 0 {
            return Err(SciError::InvalidParameter(
                "CSR row_pointers[0] must be exactly 0",
            ));
        }
        if self.values.len() != self.col_indices.len() {
            return Err(SciError::InvalidParameter(
                "CSR values and col_indices lengths must match",
            ));
        }
        if self.values.len() != self.row_pointers[self.rows] {
            return Err(SciError::InvalidParameter(
                "CSR total non-zeros must match row_pointers[rows]",
            ));
        }

        for r in 0..self.rows {
            let start = self.row_pointers[r];
            let end = self.row_pointers[r + 1];
            if start > end {
                return Err(SciError::InvalidParameter(
                    "CSR row_pointers must be monotonically non-decreasing",
                ));
            }
            let mut last_col: Option<usize> = None;
            for k in start..end {
                let col = self.col_indices[k];
                if col >= self.cols {
                    return Err(SciError::InvalidParameter("CSR column index out of bounds"));
                }
                if let Some(prev) = last_col {
                    if col <= prev {
                        return Err(SciError::InvalidParameter(
                            "CSR column indices must be strictly increasing within each row",
                        ));
                    }
                }
                last_col = Some(col);
                if !self.values[k].is_finite() {
                    return Err(SciError::DomainError("CSR values must be finite"));
                }
            }
        }
        Ok(())
    }

    /// Checks if the sparse matrix is numerically symmetric $A = A^T$ within `tolerance`.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for k in self.row_pointers[r]..self.row_pointers[r + 1] {
                let c = self.col_indices[k];
                let v1 = self.values[k];
                let v2 = csr_find(&self.values, &self.col_indices, &self.row_pointers, c, r);
                if (v1 - v2).abs() > tolerance * (v1.abs() + v2.abs() + 1.0) {
                    return false;
                }
            }
        }
        true
    }

    /// Sparse matrix–vector product  y = A·x.
    pub fn mul_vec(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        if x.len() != self.cols {
            return Err(SciError::InvalidParameter(
                "vector length must match number of columns",
            ));
        }
        let mut y = vec![0.0; self.rows];
        for row in 0..self.rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];
            for k in start..end {
                y[row] += self.values[k] * x[self.col_indices[k]];
            }
        }
        Ok(y)
    }

    // ── Iterative Solvers ──────────────────────────────────────────────────────

    /// **Conjugate Gradient** — solves A·x = b for symmetric positive-definite A.
    ///
    /// Converges in at most `max_iter` steps; terminates early when the
    /// residual Euclidean norm drops below `tolerance`.
    pub fn conjugate_gradient(
        &self,
        b: &[f64],
        tolerance: f64,
        max_iter: usize,
    ) -> SciResult<Vec<f64>> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter("CG requires a square matrix"));
        }
        if b.len() != self.rows {
            return Err(SciError::InvalidParameter(
                "right-hand side length must match number of rows",
            ));
        }
        if tolerance <= 0.0 {
            return Err(SciError::InvalidParameter("tolerance must be positive"));
        }

        let n = self.rows;
        let mut x = vec![0.0_f64; n];
        let mut r = b.to_vec(); // r = b - A·x₀ = b
        let mut p = r.clone();
        let mut rs_old = dot(&r, &r);

        for _ in 0..max_iter {
            if rs_old.sqrt() < tolerance {
                return Ok(x);
            }
            let ap = self.mul_vec(&p)?;
            let denom = dot(&p, &ap);
            if denom.abs() <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            let alpha = rs_old / denom;
            axpy(alpha, &p, &mut x); // x = x + α·p
            axpy(-alpha, &ap, &mut r); // r = r - α·A·p
            let rs_new = dot(&r, &r);
            let beta = rs_new / rs_old;
            // p = r + β·p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        if rs_old.sqrt() < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("conjugate gradient"))
        }
    }

    /// **GMRES(restart)** — Generalised Minimal Residual method.
    ///
    /// `restart` controls the Krylov subspace dimension between restarts
    /// (classic value: 30–50).  For small systems you can set `restart == n`.
    pub fn gmres(
        &self,
        b: &[f64],
        restart: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> SciResult<Vec<f64>> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter("GMRES requires a square matrix"));
        }
        if b.len() != self.rows {
            return Err(SciError::InvalidParameter(
                "right-hand side length must match number of rows",
            ));
        }
        if tolerance <= 0.0 || restart == 0 {
            return Err(SciError::InvalidParameter(
                "tolerance must be positive and restart > 0",
            ));
        }

        let n = self.rows;
        let mut x = vec![0.0_f64; n];

        for _outer in 0..max_iter {
            // r = b - A·x
            let ax = self.mul_vec(&x)?;
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
            let beta = norm(&r);
            if beta < tolerance {
                return Ok(x);
            }

            // Arnoldi basis  Q (columns) and upper Hessenberg H
            let m = restart.min(n);
            let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
            q.push(r.iter().map(|v| v / beta).collect());

            // H stored as (m+1) × m in column-major flattened vec
            let mut h = vec![0.0_f64; (m + 1) * m];

            // Givens rotation cosines/sines accumulated per column
            let mut cs = vec![0.0_f64; m];
            let mut sn = vec![0.0_f64; m];
            let mut e1 = vec![0.0_f64; m + 1];
            e1[0] = beta;

            let mut j_end = 0usize;

            'inner: for j in 0..m {
                j_end = j + 1;
                let mut w = self.mul_vec(&q[j])?;
                // Modified Gram-Schmidt
                for i in 0..=j {
                    let h_ij = dot(&q[i], &w);
                    h[i * m + j] = h_ij;
                    axpy(-h_ij, &q[i], &mut w);
                }
                let w_norm = norm(&w);
                h[(j + 1) * m + j] = w_norm;

                if w_norm > f64::EPSILON {
                    q.push(w.iter().map(|v| v / w_norm).collect());
                }

                // Apply previous Givens rotations to column j of H
                for i in 0..j {
                    let temp = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
                    h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                    h[i * m + j] = temp;
                }

                // Compute new Givens rotation
                let (c, s) = givens_rotation(h[j * m + j], h[(j + 1) * m + j]);
                cs[j] = c;
                sn[j] = s;
                h[j * m + j] = c * h[j * m + j] + s * h[(j + 1) * m + j];
                h[(j + 1) * m + j] = 0.0;

                e1[j + 1] = -s * e1[j];
                e1[j] *= c;

                if e1[j + 1].abs() < tolerance {
                    break 'inner;
                }
            }

            // Back-substitution on the upper triangular (j_end × j_end) system
            let y = back_substitute_upper(&h, &e1, m, j_end)?;

            // x = x + Q_m · y
            for (i, yi) in y.iter().enumerate() {
                axpy(*yi, &q[i], &mut x);
            }
        }

        // Final residual check
        let ax = self.mul_vec(&x)?;
        let res: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, ai)| (bi - ai).powi(2))
            .sum::<f64>()
            .sqrt();
        if res < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("GMRES"))
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// y += α·x  (BLAS-like axpy)
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Givens rotation: returns (cos θ, sin θ) such that [c s; -s c]·[a; b] = [r; 0].
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < f64::EPSILON {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        (s * t, s)
    } else {
        let t = -b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    }
}

/// Back-substitution on upper-triangular portion of H (stored column-major in h).
fn back_substitute_upper(h: &[f64], rhs: &[f64], m: usize, size: usize) -> SciResult<Vec<f64>> {
    let mut y = rhs[..size].to_vec();
    for i in (0..size).rev() {
        let diag = h[i * m + i];
        if diag.abs() <= f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        y[i] /= diag;
        for k in 0..i {
            y[k] -= h[k * m + i] * y[i];
        }
    }
    Ok(y)
}

// ══════════════════════════════════════════════════════════════════════════════
// SparseMatrixCsr — additional construction & matrix operations
// ══════════════════════════════════════════════════════════════════════════════

impl SparseMatrixCsr {
    /// Build from a dense row-major slice (zeros are dropped below `drop_tol`).
    pub fn from_dense(rows: usize, cols: usize, data: &[f64], drop_tol: f64) -> SciResult<Self> {
        if data.len() != rows * cols {
            return Err(SciError::InvalidParameter(
                "data length must equal rows*cols",
            ));
        }
        let triplets: Vec<(usize, usize, f64)> = (0..rows)
            .flat_map(|r| {
                (0..cols).filter_map(move |c| {
                    let v = data[r * cols + c];
                    if v.abs() > drop_tol {
                        Some((r, c, v))
                    } else {
                        None
                    }
                })
            })
            .collect();
        Self::from_triplets(rows, cols, &triplets)
    }

    /// Get the value at (row, col); returns 0.0 if not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        csr_find(
            &self.values,
            &self.col_indices,
            &self.row_pointers,
            row,
            col,
        )
    }

    /// Scalar multiply: returns A·s.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            values: self.values.iter().map(|v| v * s).collect(),
            col_indices: self.col_indices.clone(),
            row_pointers: self.row_pointers.clone(),
        }
    }

    /// Sparse matrix addition A + B (must have same shape).
    pub fn add_sparse(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for sparse add"));
        }
        // Merge COO lists then re-build
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for r in 0..self.rows {
            for k in self.row_pointers[r]..self.row_pointers[r + 1] {
                triplets.push((r, self.col_indices[k], self.values[k]));
            }
            for k in other.row_pointers[r]..other.row_pointers[r + 1] {
                triplets.push((r, other.col_indices[k], other.values[k]));
            }
        }
        Self::from_triplets(self.rows, self.cols, &triplets)
    }

    /// CSR transpose: returns Aᵀ.
    pub fn transpose(&self) -> Self {
        let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(self.values.len());
        for r in 0..self.rows {
            for k in self.row_pointers[r]..self.row_pointers[r + 1] {
                triplets.push((self.col_indices[k], r, self.values[k]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets).unwrap()
    }

    /// Convert to dense row-major Vec.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut d = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for k in self.row_pointers[r]..self.row_pointers[r + 1] {
                d[r * self.cols + self.col_indices[k]] = self.values[k];
            }
        }
        d
    }

    // ── Preconditioners ───────────────────────────────────────────────────────

    /// Jacobi (diagonal) preconditioner — M = diag(A).
    pub fn diag_preconditioner(&self) -> SciResult<DiagPreconditioner> {
        let mut inv_diag = vec![1.0f64; self.rows];
        for r in 0..self.rows {
            let d = csr_find(&self.values, &self.col_indices, &self.row_pointers, r, r);
            if d.abs() < f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            inv_diag[r] = 1.0 / d;
        }
        Ok(DiagPreconditioner { inv_diag })
    }

    /// ILU(0) incomplete LU factorisation — keeps sparsity pattern of A.
    ///
    /// Works in-place on a copy of the values; returns an [`IluPreconditioner`]
    /// that can be used in `preconditioned_gmres` or `preconditioned_cg`.
    pub fn ilu0(&self) -> SciResult<IluPreconditioner> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter("ILU(0) requires square matrix"));
        }
        let n = self.rows;
        let mut vals = self.values.clone();
        // Find diagonal position in each row (cache for speed).
        let mut diag_pos = vec![usize::MAX; n];
        for r in 0..n {
            for k in self.row_pointers[r]..self.row_pointers[r + 1] {
                if self.col_indices[k] == r {
                    diag_pos[r] = k;
                    break;
                }
            }
            if diag_pos[r] == usize::MAX {
                return Err(SciError::InvalidParameter("ILU(0): zero diagonal"));
            }
        }

        for i in 1..n {
            // Walk lower triangular entries of row i
            for ki in self.row_pointers[i]..self.row_pointers[i + 1] {
                let k = self.col_indices[ki];
                if k >= i {
                    break;
                }
                let a_kk = vals[diag_pos[k]];
                if a_kk.abs() < f64::EPSILON {
                    return Err(SciError::DivisionByZero);
                }
                let m = vals[ki] / a_kk; // multiplier
                vals[ki] = m;
                // Update remaining entries of row i that exist in row k
                for ji in (ki + 1)..self.row_pointers[i + 1] {
                    let j = self.col_indices[ji];
                    // Find (k, j) in row k
                    let a_kj = csr_find(&vals, &self.col_indices, &self.row_pointers, k, j);
                    vals[ji] -= m * a_kj;
                }
            }
        }

        Ok(IluPreconditioner {
            values: vals,
            col_indices: self.col_indices.clone(),
            row_pointers: self.row_pointers.clone(),
            diag_pos,
            n,
        })
    }

    // ── Additional Solvers ────────────────────────────────────────────────────

    /// **Preconditioned CG** — solves A·x = b for SPD A with preconditioner M.
    pub fn preconditioned_cg(
        &self,
        b: &[f64],
        precond: &dyn Preconditioner,
        tolerance: f64,
        max_iter: usize,
    ) -> SciResult<Vec<f64>> {
        if self.rows != self.cols || b.len() != self.rows {
            return Err(SciError::InvalidParameter("PCG: shape mismatch"));
        }
        let n = self.rows;
        let mut x = vec![0.0f64; n];
        let mut r = b.to_vec();
        let mut z = precond.solve(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);

        for _ in 0..max_iter {
            if norm(&r) < tolerance {
                return Ok(x);
            }
            let ap = self.mul_vec(&p)?;
            let denom = dot(&p, &ap);
            if denom.abs() < f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            let alpha = rz / denom;
            axpy(alpha, &p, &mut x);
            axpy(-alpha, &ap, &mut r);
            z = precond.solve(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        if norm(&r) < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("PCG"))
        }
    }

    /// **Preconditioned GMRES(restart)** with right-preconditioning: A·M⁻¹·y = b.
    pub fn preconditioned_gmres(
        &self,
        b: &[f64],
        precond: &dyn Preconditioner,
        restart: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> SciResult<Vec<f64>> {
        if self.rows != self.cols || b.len() != self.rows {
            return Err(SciError::InvalidParameter("P-GMRES: shape mismatch"));
        }
        let n = self.rows;
        let mut x = vec![0.0f64; n];

        for _outer in 0..max_iter {
            let ax = self.mul_vec(&x)?;
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
            let beta = norm(&r);
            if beta < tolerance {
                return Ok(x);
            }
            let m = restart.min(n);
            let mut q: Vec<Vec<f64>> = vec![r.iter().map(|v| v / beta).collect()];
            let mut h = vec![0.0f64; (m + 1) * m];
            let mut cs = vec![0.0f64; m];
            let mut sn = vec![0.0f64; m];
            let mut e1 = vec![0.0f64; m + 1];
            e1[0] = beta;
            let mut j_end = 0usize;

            'inner: for j in 0..m {
                j_end = j + 1;
                let z = precond.solve(&q[j]);
                let mut w = self.mul_vec(&z)?;
                for i in 0..=j {
                    let h_ij = dot(&q[i], &w);
                    h[i * m + j] = h_ij;
                    axpy(-h_ij, &q[i], &mut w);
                }
                let w_norm = norm(&w);
                h[(j + 1) * m + j] = w_norm;
                if w_norm > f64::EPSILON {
                    q.push(w.iter().map(|v| v / w_norm).collect());
                }
                for i in 0..j {
                    let tmp = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
                    h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                    h[i * m + j] = tmp;
                }
                let (c, s) = givens_rotation(h[j * m + j], h[(j + 1) * m + j]);
                cs[j] = c;
                sn[j] = s;
                h[j * m + j] = c * h[j * m + j] + s * h[(j + 1) * m + j];
                h[(j + 1) * m + j] = 0.0;
                e1[j + 1] = -s * e1[j];
                e1[j] *= c;
                if e1[j + 1].abs() < tolerance {
                    break 'inner;
                }
            }
            let y = back_substitute_upper(&h, &e1, m, j_end)?;
            // Right-preconditioned update: x += M⁻¹ · (Q · y)
            for (i, yi) in y.iter().enumerate() {
                let zy = precond.solve(&q[i]);
                axpy(*yi, &zy, &mut x);
            }
        }
        let ax = self.mul_vec(&x)?;
        let res = norm(
            &b.iter()
                .zip(ax.iter())
                .map(|(bi, ai)| bi - ai)
                .collect::<Vec<_>>(),
        );
        if res < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("P-GMRES"))
        }
    }

    /// **BiCGSTAB** — Biconjugate Gradient Stabilised; works for general non-symmetric systems.
    pub fn bicgstab(&self, b: &[f64], tolerance: f64, max_iter: usize) -> SciResult<Vec<f64>> {
        if self.rows != self.cols || b.len() != self.rows {
            return Err(SciError::InvalidParameter("BiCGSTAB: shape mismatch"));
        }
        let n = self.rows;
        let mut x = vec![0.0f64; n];
        let mut r: Vec<f64> = {
            let ax = self.mul_vec(&x)?;
            b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect()
        };
        let r_hat = r.clone(); // shadow residual (fixed)
        let mut rho_old = 1.0f64;
        let mut alpha = 1.0f64;
        let mut omega = 1.0f64;
        let mut v = vec![0.0f64; n];
        let mut p = vec![0.0f64; n];

        for _ in 0..max_iter {
            if norm(&r) < tolerance {
                return Ok(x);
            }
            let rho = dot(&r_hat, &r);
            if rho.abs() < f64::EPSILON {
                break;
            }
            let beta = (rho / rho_old) * (alpha / omega);
            // p = r + beta*(p - omega*v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            v = self.mul_vec(&p)?;
            let rtv = dot(&r_hat, &v);
            if rtv.abs() < f64::EPSILON {
                break;
            }
            alpha = rho / rtv;
            // s = r - alpha*v
            let s: Vec<f64> = r
                .iter()
                .zip(v.iter())
                .map(|(ri, vi)| ri - alpha * vi)
                .collect();
            if norm(&s) < tolerance {
                axpy(alpha, &p, &mut x);
                return Ok(x);
            }
            let t = self.mul_vec(&s)?;
            let tt = dot(&t, &t);
            omega = if tt < f64::EPSILON {
                0.0
            } else {
                dot(&t, &s) / tt
            };
            axpy(alpha, &p, &mut x);
            axpy(omega, &s, &mut x);
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }
            rho_old = rho;
        }
        if norm(&r) < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("BiCGSTAB"))
        }
    }

    /// **MINRES** — Paige-Saunders method for symmetric (possibly indefinite) systems.
    ///
    /// More stable than CG for indefinite A (where CG can break down).
    pub fn minres(&self, b: &[f64], tolerance: f64, max_iter: usize) -> SciResult<Vec<f64>> {
        if self.rows != self.cols || b.len() != self.rows {
            return Err(SciError::InvalidParameter("MINRES: shape mismatch"));
        }
        let n = self.rows;
        let mut x = vec![0.0f64; n];

        let mut v_prev = vec![0.0f64; n];
        let mut v_cur = b.to_vec();
        let mut beta_cur = norm(&v_cur);
        if beta_cur < f64::EPSILON {
            return Ok(x);
        }
        for vi in v_cur.iter_mut() {
            *vi /= beta_cur;
        }

        let mut d_cur = vec![0.0f64; n];
        let mut d_prev = vec![0.0f64; n];

        let mut phi_bar = beta_cur;
        let mut c_old = -1.0f64;
        let mut s_old = 0.0f64;
        let mut c_cur = -1.0f64;
        let mut s_cur = 0.0f64;

        for _ in 0..max_iter {
            // Lanczos step
            let mut w = self.mul_vec(&v_cur)?;
            let alpha = dot(&v_cur, &w);
            axpy(-alpha, &v_cur, &mut w);
            axpy(-beta_cur, &v_prev, &mut w);
            let beta_next = norm(&w);

            // QR via Givens on tri-diagonal
            let eps = s_old * beta_cur;
            let delta_bar = -c_old * beta_cur;
            let delta = c_cur * delta_bar + s_cur * alpha;
            let phi = s_cur * delta_bar - c_cur * alpha;
            let (c_new, s_new) = givens_rotation(phi, beta_next);

            // Update x
            // d = (v_cur - eps*d_prev - delta*d_cur) / phi
            // x += phi_bar * c_new * d
            let phi_bar_new = phi_bar * s_new;
            let phi_new = phi_bar * c_new;

            let mut d_next = v_cur.clone();
            for i in 0..n {
                d_next[i] = (d_next[i] - eps * d_prev[i] - delta * d_cur[i]) / phi;
            }
            for i in 0..n {
                x[i] += phi_new * d_next[i];
            }

            // Convergence estimate
            if phi_bar_new.abs() < tolerance {
                return Ok(x);
            }

            // Advance
            v_prev = v_cur.clone();
            if beta_next > f64::EPSILON {
                v_cur = w.iter().map(|wi| wi / beta_next).collect();
            } else {
                return Ok(x);
            }
            d_prev = d_cur;
            d_cur = d_next;
            beta_cur = beta_next;
            phi_bar = phi_bar_new;
            c_old = c_cur;
            s_old = s_cur;
            c_cur = c_new;
            s_cur = s_new;
        }

        let ax = self.mul_vec(&x)?;
        let res = norm(
            &b.iter()
                .zip(ax.iter())
                .map(|(bi, ai)| bi - ai)
                .collect::<Vec<_>>(),
        );
        if res < tolerance {
            Ok(x)
        } else {
            Err(SciError::NonConvergent("MINRES"))
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Preconditioner trait + implementations
// ══════════════════════════════════════════════════════════════════════════════

/// Common interface for linear preconditioners.
pub trait Preconditioner {
    /// Solve M·z = r approximately.
    fn solve(&self, r: &[f64]) -> Vec<f64>;
}

/// Jacobi (diagonal) preconditioner: M = diag(A).
pub struct DiagPreconditioner {
    inv_diag: Vec<f64>,
}

impl Preconditioner for DiagPreconditioner {
    fn solve(&self, r: &[f64]) -> Vec<f64> {
        r.iter()
            .zip(self.inv_diag.iter())
            .map(|(ri, di)| ri * di)
            .collect()
    }
}

/// ILU(0) preconditioner — solves (LU)·z = r via forward + backward substitution.
pub struct IluPreconditioner {
    values: Vec<f64>,
    col_indices: Vec<usize>,
    row_pointers: Vec<usize>,
    diag_pos: Vec<usize>,
    n: usize,
}

impl Preconditioner for IluPreconditioner {
    fn solve(&self, r: &[f64]) -> Vec<f64> {
        let n = self.n;
        // Forward substitution: L·y = r  (L has unit diagonal, stored below diag)
        let mut y = r.to_vec();
        for i in 0..n {
            for k in self.row_pointers[i]..self.diag_pos[i] {
                y[i] -= self.values[k] * y[self.col_indices[k]];
            }
        }
        // Backward substitution: U·z = y
        let mut z = y;
        for i in (0..n).rev() {
            for k in (self.diag_pos[i] + 1)..self.row_pointers[i + 1] {
                z[i] -= self.values[k] * z[self.col_indices[k]];
            }
            let d = self.values[self.diag_pos[i]];
            if d.abs() > f64::EPSILON {
                z[i] /= d;
            }
        }
        z
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private CSR helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Linear-scan lookup of value at (row, col) in CSR arrays.
fn csr_find(
    values: &[f64],
    col_indices: &[usize],
    row_pointers: &[usize],
    row: usize,
    col: usize,
) -> f64 {
    for k in row_pointers[row]..row_pointers[row + 1] {
        if col_indices[k] == col {
            return values[k];
        }
    }
    0.0
}

use crate::linear_algebra::DynamicMatrix;

/// Result of the iterative symmetric Lanczos eigensolver.
#[derive(Debug, Clone, PartialEq)]
pub struct LanczosEigenResult {
    /// Approximate eigenvalues in descending order.
    pub eigenvalues: Vec<f64>,
    /// Approximate Ritz eigenvectors corresponding to eigenvalues.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Individual eigenpair residuals $\|A v_i - \lambda_i v_i\|_2$.
    pub residuals: Vec<f64>,
    /// Total Lanczos iterations performed.
    pub iterations: usize,
    /// Whether all requested eigenpair residuals satisfy `tolerance`.
    pub converged: bool,
}

/// Deterministic iterative symmetric eigensolver (Lanczos algorithm with full reorthogonalization).
///
/// Approximates the extreme eigenvalues and eigenvectors of a large symmetric sparse matrix.
///
/// # Preconditions
/// - Matrix must be square ($n \times n$) and symmetric within numerical tolerance.
/// - `k_eigenvalues` must satisfy $1 \le k \le n$.
/// - `max_iterations` must satisfy $k \le \text{max\_iterations} \le n$.
///
/// # Numerical Limitations
/// This routine produces numerical floating-point approximations (Ritz pairs) and does not
/// constitute an exact algebraic spectrum certificate.
pub fn lanczos_eigen(
    matrix: &SparseMatrixCsr,
    k_eigenvalues: usize,
    max_iterations: usize,
    tolerance: f64,
) -> SciResult<LanczosEigenResult> {
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(SciError::InvalidParameter(
            "Lanczos eigensolver requires a square matrix",
        ));
    }
    if k_eigenvalues == 0 || k_eigenvalues > n {
        return Err(SciError::InvalidParameter(
            "k_eigenvalues must be between 1 and matrix dimension",
        ));
    }
    if tolerance <= 0.0 {
        return Err(SciError::InvalidParameter("tolerance must be positive"));
    }
    if !matrix.is_symmetric(1e-6) {
        return Err(SciError::InvalidParameter(
            "Lanczos eigensolver requires a symmetric matrix",
        ));
    }

    let m_steps = max_iterations.max(k_eigenvalues).min(n);

    // Initial deterministic unit vector q1 = [1/sqrt(n), ..., 1/sqrt(n)]^T
    let mut q_basis: Vec<Vec<f64>> = Vec::with_capacity(m_steps);
    let mut alphas: Vec<f64> = Vec::with_capacity(m_steps);
    let mut betas: Vec<f64> = Vec::with_capacity(m_steps);

    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let q1 = vec![inv_sqrt_n; n];
    q_basis.push(q1);

    let mut actual_m = 0;

    for j in 0..m_steps {
        actual_m = j + 1;
        let q_j = &q_basis[j];
        let mut v = matrix.mul_vec(q_j)?;

        let alpha = dot_product(q_j, &v);
        alphas.push(alpha);

        // v = v - alpha * q_j
        for (idx, item) in v.iter_mut().enumerate().take(n) {
            *item -= alpha * q_j[idx];
        }

        if j > 0 {
            let beta_prev = betas[j - 1];
            let q_prev = &q_basis[j - 1];
            for (idx, item) in v.iter_mut().enumerate().take(n) {
                *item -= beta_prev * q_prev[idx];
            }
        }

        // Full Gram-Schmidt reorthogonalization for numerical stability
        for q_prev in &q_basis {
            let proj = dot_product(q_prev, &v);
            for (idx, item) in v.iter_mut().enumerate().take(n) {
                *item -= proj * q_prev[idx];
            }
        }

        let beta = vector_norm(&v);
        if beta < 1e-13 || j + 1 == m_steps {
            break;
        }

        betas.push(beta);
        let q_next: Vec<f64> = v.iter().map(|&x| x / beta).collect();
        q_basis.push(q_next);
    }

    // Build m x m tridiagonal matrix T
    let m = alphas.len();
    let mut t_dense_data = vec![0.0; m * m];
    for i in 0..m {
        t_dense_data[i * m + i] = alphas[i];
        if i + 1 < m && i < betas.len() {
            t_dense_data[i * m + i + 1] = betas[i];
            t_dense_data[(i + 1) * m + i] = betas[i];
        }
    }
    let t_mat = DynamicMatrix::new(m, m, t_dense_data)?;
    let eig_t = crate::eigensystem::jacobi_eigen(&t_mat, tolerance.min(1e-12), 200)?;

    // Map Ritz values and vectors
    let mut indexed_eigs: Vec<(usize, f64)> = eig_t.values.iter().copied().enumerate().collect();
    indexed_eigs.sort_by(|a, b| b.1.total_cmp(&a.1)); // Descending order

    let selected_count = k_eigenvalues.min(m);
    let mut eigenvalues = Vec::with_capacity(selected_count);
    let mut eigenvectors = Vec::with_capacity(selected_count);
    let mut residuals = Vec::with_capacity(selected_count);

    for (t_idx, val) in indexed_eigs.into_iter().take(selected_count) {
        eigenvalues.push(val);

        // Ritz vector v = sum_{j=0..m-1} s_{j, t_idx} * q_j
        let mut ritz_v = vec![0.0; n];
        for j in 0..m {
            let s_j = eig_t.vectors.get(j, t_idx)?;
            let q_j = &q_basis[j];
            for (idx, item) in ritz_v.iter_mut().enumerate().take(n) {
                *item += s_j * q_j[idx];
            }
        }

        // Normalize and enforce deterministic sign convention (first non-zero entry > 0)
        let norm_v = vector_norm(&ritz_v);
        if norm_v > 1e-14 {
            for item in ritz_v.iter_mut() {
                *item /= norm_v;
            }
        }
        if let Some(&first_significant) = ritz_v.iter().find(|&&x| x.abs() > 1e-6) {
            if first_significant < 0.0 {
                for item in ritz_v.iter_mut() {
                    *item = -*item;
                }
            }
        }

        // Compute eigenpair residual ||A v - lambda v||_2
        let av = matrix.mul_vec(&ritz_v)?;
        let mut res_sq = 0.0;
        for (idx, &avi) in av.iter().enumerate().take(n) {
            let diff = avi - val * ritz_v[idx];
            res_sq += diff * diff;
        }
        let res = res_sq.sqrt();
        residuals.push(res);
        eigenvectors.push(ritz_v);
    }

    let converged = residuals.iter().all(|&r| r <= tolerance);

    Ok(LanczosEigenResult {
        eigenvalues,
        eigenvectors,
        residuals,
        iterations: actual_m,
        converged,
    })
}

#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn vector_norm(a: &[f64]) -> f64 {
    dot_product(a, a).sqrt()
}
