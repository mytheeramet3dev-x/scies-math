use core::ops::{Add, Mul, Sub};

use crate::errors::{SciError, SciResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

impl Vector2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix2 {
    pub data: [[f64; 2]; 2],
}

impl Matrix2 {
    pub const fn new(data: [[f64; 2]; 2]) -> Self {
        Self { data }
    }

    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0], [0.0, 1.0]])
    }

    pub fn determinant(self) -> f64 {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }

    pub fn transpose(self) -> Self {
        Self::new([
            [self.data[0][0], self.data[1][0]],
            [self.data[0][1], self.data[1][1]],
        ])
    }

    pub fn mul_vector(self, vector: Vector2) -> Vector2 {
        Vector2::new(
            self.data[0][0] * vector.x + self.data[0][1] * vector.y,
            self.data[1][0] * vector.x + self.data[1][1] * vector.y,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    pub data: [[f64; 3]; 3],
}

impl Matrix3 {
    pub const fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    pub fn determinant(self) -> f64 {
        let m = self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn transpose(self) -> Self {
        let m = self.data;
        Self::new([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    pub fn mul_vector(self, vector: Vector3) -> Vector3 {
        let m = self.data;
        Vector3::new(
            m[0][0] * vector.x + m[0][1] * vector.y + m[0][2] * vector.z,
            m[1][0] * vector.x + m[1][1] * vector.y + m[1][2] * vector.z,
            m[2][0] * vector.x + m[2][1] * vector.y + m[2][2] * vector.z,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tensor2 {
    pub xx: f64,
    pub xy: f64,
    pub yx: f64,
    pub yy: f64,
}

impl Tensor2 {
    pub const fn new(xx: f64, xy: f64, yx: f64, yy: f64) -> Self {
        Self { xx, xy, yx, yy }
    }

    pub fn trace(self) -> f64 {
        self.xx + self.yy
    }

    pub fn determinant(self) -> f64 {
        self.xx * self.yy - self.xy * self.yx
    }

    pub fn as_matrix(self) -> Matrix2 {
        Matrix2::new([[self.xx, self.xy], [self.yx, self.yy]])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DynamicMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LuDecomposition {
    pub l: DynamicMatrix,
    pub u: DynamicMatrix,
    pub permutation: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QrDecomposition {
    pub q: DynamicMatrix,
    pub r: DynamicMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SingularValueDecomposition {
    pub u: DynamicMatrix,
    pub singular_values: Vec<f64>,
    pub v_t: DynamicMatrix,
}

/// Lower-triangular Cholesky factor L such that A = L · L^T.
#[derive(Debug, Clone, PartialEq)]
pub struct CholeskyDecomposition {
    pub l: DynamicMatrix,
}

impl DynamicMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> SciResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(SciError::InvalidParameter(
                "matrix dimensions must be positive",
            ));
        }
        if data.len() != rows * cols {
            return Err(SciError::InvalidParameter(
                "matrix data length does not match dimensions",
            ));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> SciResult<Self> {
        Self::new(rows, cols, vec![0.0; rows * cols])
    }

    pub fn identity(size: usize) -> SciResult<Self> {
        let mut matrix = Self::zeros(size, size)?;
        for index in 0..size {
            matrix.data[index * size + index] = 1.0;
        }
        Ok(matrix)
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Raw row-major data slice — for use by extension modules.
    pub fn raw_data(&self) -> &[f64] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> SciResult<f64> {
        if row >= self.rows || col >= self.cols {
            return Err(SciError::InvalidParameter("matrix index out of bounds"));
        }
        Ok(self.data[row * self.cols + col])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> SciResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(SciError::InvalidParameter("matrix index out of bounds"));
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                data[col * self.rows + row] = self.data[row * self.cols + col];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn mul_matrix(&self, other: &Self) -> SciResult<Self> {
        if self.cols != other.rows {
            return Err(SciError::InvalidParameter(
                "matrix dimensions are not compatible for multiplication",
            ));
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = vec![0.0f64; m * n];
        crate::perf::matmul(&self.data, &other.data, &mut result, m, k, n);

        Ok(Self {
            rows: m,
            cols: n,
            data: result,
        })
    }

    /// Multithreaded matrix multiply using `n_threads` OS threads.
    ///
    /// Falls back to `mul_matrix` for small inputs.  
    /// Prefer for square matrices where `n ≥ 256`.
    pub fn mul_matrix_par(&self, other: &Self, n_threads: usize) -> SciResult<Self> {
        if self.cols != other.rows {
            return Err(SciError::InvalidParameter(
                "matrix dimensions are not compatible for multiplication",
            ));
        }
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut result = vec![0.0f64; m * n];
        crate::perf::matmul_threaded(&self.data, &other.data, &mut result, m, k, n, n_threads);
        Ok(Self { rows: m, cols: n, data: result })
    }

    pub fn mul_vector(&self, vector: &[f64]) -> SciResult<Vec<f64>> {
        if self.cols != vector.len() {
            return Err(SciError::InvalidParameter(
                "matrix columns must match vector length",
            ));
        }

        let mut result = vec![0.0; self.rows];
        for (row, output) in result.iter_mut().enumerate() {
            *output = (0..self.cols)
                .map(|col| self.data[row * self.cols + col] * vector[col])
                .sum();
        }
        Ok(result)
    }

    pub fn trace(&self) -> SciResult<f64> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter("trace requires a square matrix"));
        }
        Ok((0..self.rows)
            .map(|index| self.data[index * self.cols + index])
            .sum())
    }

    pub fn determinant(&self) -> SciResult<f64> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "determinant requires a square matrix",
            ));
        }

        let n = self.rows;
        let mut matrix = self.data.clone();
        let mut determinant = 1.0;
        let mut sign = 1.0;

        for pivot in 0..n {
            let mut max_row = pivot;
            let mut max_value = matrix[pivot * n + pivot].abs();
            for row in (pivot + 1)..n {
                let value = matrix[row * n + pivot].abs();
                if value > max_value {
                    max_value = value;
                    max_row = row;
                }
            }

            if max_value <= f64::EPSILON {
                return Ok(0.0);
            }

            if max_row != pivot {
                for col in 0..n {
                    matrix.swap(pivot * n + col, max_row * n + col);
                }
                sign *= -1.0;
            }

            let pivot_value = matrix[pivot * n + pivot];
            determinant *= pivot_value;

            for row in (pivot + 1)..n {
                let factor = matrix[row * n + pivot] / pivot_value;
                for col in pivot..n {
                    matrix[row * n + col] -= factor * matrix[pivot * n + col];
                }
            }
        }

        Ok(determinant * sign)
    }

    pub fn solve_linear_system(&self, rhs: &[f64]) -> SciResult<Vec<f64>> {
        let lu = self.lu_decompose()?;
        lu.solve(rhs)
    }

    pub fn lu_decompose(&self) -> SciResult<LuDecomposition> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "LU decomposition requires a square matrix",
            ));
        }

        let n = self.rows;
        let mut u = self.data.clone();
        let mut l = vec![0.0; n * n];
        let mut permutation = (0..n).collect::<Vec<_>>();

        for pivot in 0..n {
            let mut max_row = pivot;
            let mut max_value = u[pivot * n + pivot].abs();
            for row in (pivot + 1)..n {
                let value = u[row * n + pivot].abs();
                if value > max_value {
                    max_value = value;
                    max_row = row;
                }
            }

            if max_value <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }

            if max_row != pivot {
                for col in 0..n {
                    u.swap(pivot * n + col, max_row * n + col);
                    if col < pivot {
                        l.swap(pivot * n + col, max_row * n + col);
                    }
                }
                permutation.swap(pivot, max_row);
            }

            l[pivot * n + pivot] = 1.0;
            for row in (pivot + 1)..n {
                let factor = u[row * n + pivot] / u[pivot * n + pivot];
                l[row * n + pivot] = factor;
                for col in pivot..n {
                    u[row * n + col] -= factor * u[pivot * n + col];
                }
            }
        }

        Ok(LuDecomposition {
            l: DynamicMatrix::new(n, n, l)?,
            u: DynamicMatrix::new(n, n, u)?,
            permutation,
        })
    }

    pub fn qr_decompose(&self) -> SciResult<QrDecomposition> {
        let m = self.rows;
        let n = self.cols;
        if m < n {
            return Err(SciError::InvalidParameter(
                "QR decomposition currently requires rows >= cols",
            ));
        }

        let mut q_columns = vec![vec![0.0; m]; n];
        let mut r = vec![0.0; n * n];

        for j in 0..n {
            let mut v = self.column(j);
            for i in 0..j {
                let coefficient = dot(&q_columns[i], &v);
                r[i * n + j] = coefficient;
                for (vk, qk) in v.iter_mut().zip(q_columns[i].iter()) {
                    *vk -= coefficient * qk;
                }
            }

            let norm = vector_norm(&v);
            if norm <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            r[j * n + j] = norm;
            for value in &mut v {
                *value /= norm;
            }
            q_columns[j] = v;
        }

        let mut q_data = vec![0.0; m * n];
        for col in 0..n {
            for row in 0..m {
                q_data[row * n + col] = q_columns[col][row];
            }
        }

        Ok(QrDecomposition {
            q: DynamicMatrix::new(m, n, q_data)?,
            r: DynamicMatrix::new(n, n, r)?,
        })
    }

    pub fn inverse(&self) -> SciResult<Self> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "inverse requires a square matrix",
            ));
        }

        let lu = self.lu_decompose()?;
        let n = self.rows;
        let mut inverse = Self::zeros(n, n)?;

        for col in 0..n {
            let mut basis = vec![0.0; n];
            basis[col] = 1.0;
            let solution = lu.solve(&basis)?;
            for (row, value) in solution.iter().enumerate() {
                inverse.data[row * n + col] = *value;
            }
        }

        Ok(inverse)
    }

    pub fn least_squares(&self, rhs: &[f64]) -> SciResult<Vec<f64>> {
        if self.rows < self.cols {
            return Err(SciError::InvalidParameter(
                "least squares requires rows >= cols",
            ));
        }
        if rhs.len() != self.rows {
            return Err(SciError::InvalidParameter(
                "right-hand side length must match matrix rows",
            ));
        }

        let qr = self.qr_decompose()?;
        let n = self.cols;
        let mut qt_b = vec![0.0; n];
        for (col, value) in qt_b.iter_mut().enumerate() {
            *value =
                qr.q.column(col)
                    .iter()
                    .zip(rhs.iter())
                    .map(|(q, b)| q * b)
                    .sum();
        }

        solve_upper_triangular(&qr.r, &qt_b)
    }

    pub fn qr_eigenvalues(&self, tolerance: f64, max_iterations: usize) -> SciResult<Vec<f64>> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "QR eigenvalue algorithm requires a square matrix",
            ));
        }
        if tolerance <= 0.0 {
            return Err(SciError::InvalidParameter("tolerance must be positive"));
        }

        let mut current = self.clone();
        for _ in 0..max_iterations {
            let qr = current.qr_decompose()?;
            current = qr.r.mul_matrix(&qr.q)?;

            let off_diagonal = current.off_diagonal_norm();
            if off_diagonal < tolerance {
                return Ok((0..current.rows)
                    .map(|index| current.data[index * current.cols + index])
                    .collect());
            }
        }

        Err(SciError::NonConvergent("QR eigenvalue algorithm"))
    }

    pub fn singular_value_decompose(
        &self,
        tolerance: f64,
        max_iterations: usize,
    ) -> SciResult<SingularValueDecomposition> {
        if tolerance <= 0.0 {
            return Err(SciError::InvalidParameter("tolerance must be positive"));
        }

        let ata = self.transpose().mul_matrix(self)?;
        let (eigenvalues, eigenvectors) =
            ata.symmetric_qr_eigendecomposition(tolerance, max_iterations)?;

        let mut ordering = (0..eigenvalues.len()).collect::<Vec<_>>();
        ordering.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

        let singular_values = ordering
            .iter()
            .map(|&index| eigenvalues[index].max(0.0).sqrt())
            .collect::<Vec<_>>();

        let mut v_data = vec![0.0; eigenvectors.rows * eigenvectors.cols];
        for (new_col, &old_col) in ordering.iter().enumerate() {
            for row in 0..eigenvectors.rows {
                v_data[row * eigenvectors.cols + new_col] =
                    eigenvectors.data[row * eigenvectors.cols + old_col];
            }
        }
        let v = DynamicMatrix::new(eigenvectors.rows, eigenvectors.cols, v_data)?;

        let mut u_data = vec![0.0; self.rows * v.cols];
        for (col, sigma) in singular_values.iter().enumerate() {
            let v_col = v.column(col);
            let av = self.mul_vector(&v_col)?;
            if sigma.abs() > tolerance {
                for row in 0..self.rows {
                    u_data[row * v.cols + col] = av[row] / sigma;
                }
            }
        }
        let u = DynamicMatrix::new(self.rows, v.cols, u_data)?;

        Ok(SingularValueDecomposition {
            u,
            singular_values,
            v_t: v.transpose(),
        })
    }

    pub fn power_iteration(
        &self,
        initial_vector: &[f64],
        tolerance: f64,
        max_iterations: usize,
    ) -> SciResult<(f64, Vec<f64>)> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "power iteration requires a square matrix",
            ));
        }
        if initial_vector.len() != self.cols {
            return Err(SciError::InvalidParameter(
                "initial vector length must match matrix size",
            ));
        }
        if tolerance <= 0.0 {
            return Err(SciError::InvalidParameter("tolerance must be positive"));
        }

        let mut vector = initial_vector.to_vec();
        normalize_vector(&mut vector)?;
        let mut eigenvalue = 0.0;

        for _ in 0..max_iterations {
            let next = self.mul_vector(&vector)?;
            let next_norm = vector_norm(&next);
            if next_norm <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }

            let mut normalized = next
                .iter()
                .map(|value| value / next_norm)
                .collect::<Vec<_>>();
            let candidate = rayleigh_quotient(self, &normalized)?;

            let diff = normalized
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            vector = core::mem::take(&mut normalized);
            if (candidate - eigenvalue).abs() < tolerance && diff < tolerance {
                return Ok((candidate, vector));
            }
            eigenvalue = candidate;
        }

        Err(SciError::NonConvergent("power iteration"))
    }

    // ── Cholesky ──────────────────────────────────────────────────────────────

    /// Cholesky-Banachiewicz decomposition for symmetric positive-definite A.
    ///
    /// Returns L such that `A = L · L^T`.
    /// Errors with `DomainError` if A is not positive-definite.
    pub fn cholesky_decompose(&self) -> SciResult<CholeskyDecomposition> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "Cholesky requires a square matrix",
            ));
        }
        let n = self.rows;
        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let sum: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
                if i == j {
                    let diag = self.data[i * n + i] - sum;
                    if diag <= 0.0 {
                        return Err(SciError::DomainError("matrix is not positive-definite"));
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    let ljj = l[j * n + j];
                    if ljj.abs() <= f64::EPSILON {
                        return Err(SciError::DivisionByZero);
                    }
                    l[i * n + j] = (self.data[i * n + j] - sum) / ljj;
                }
            }
        }
        Ok(CholeskyDecomposition {
            l: DynamicMatrix::new(n, n, l)?,
        })
    }

    // ── Pseudoinverse ─────────────────────────────────────────────────────────

    /// Moore-Penrose pseudoinverse via SVD: A⁺ = V · Σ⁺ · U^T.
    ///
    /// Singular values below `tolerance * σ_max` are treated as zero,
    /// giving a numerically rank-safe result for any matrix shape.
    pub fn pseudoinverse(&self, tolerance: f64, max_iterations: usize) -> SciResult<Self> {
        let svd = self.singular_value_decompose(tolerance, max_iterations)?;
        let sigma_max = svd.singular_values.first().copied().unwrap_or(0.0);
        let cutoff = tolerance * sigma_max;
        let sigma_plus: Vec<f64> = svd
            .singular_values
            .iter()
            .map(|&s| if s > cutoff { 1.0 / s } else { 0.0 })
            .collect();

        // V = v_t^T  (n_cols × k),  U^T  (k × n_rows)
        let v = svd.v_t.transpose();
        let u_t = svd.u.transpose();
        let k = sigma_plus.len();
        let out_rows = v.rows;
        let out_cols = u_t.cols;

        let mut data = vec![0.0f64; out_rows * out_cols];
        for i in 0..out_rows {
            for j in 0..out_cols {
                data[i * out_cols + j] = (0..k)
                    .map(|l| v.data[i * v.cols + l] * sigma_plus[l] * u_t.data[l * u_t.cols + j])
                    .sum();
            }
        }
        DynamicMatrix::new(out_rows, out_cols, data)
    }

    // ── Rank ──────────────────────────────────────────────────────────────────

    /// Numerical rank: number of singular values above `tolerance * σ_max`.
    pub fn rank_estimate(&self, tolerance: f64, max_iterations: usize) -> SciResult<usize> {
        let svd = self.singular_value_decompose(tolerance, max_iterations)?;
        let sigma_max = svd.singular_values.first().copied().unwrap_or(0.0);
        let cutoff = tolerance * sigma_max;
        Ok(svd.singular_values.iter().filter(|&&s| s > cutoff).count())
    }

    // ── Condition Number ──────────────────────────────────────────────────────

    /// Spectral (2-norm) condition number κ₂ = σ_max / σ_min.
    ///
    /// Returns `f64::INFINITY` for rank-deficient matrices.
    pub fn condition_number_2(&self, tolerance: f64, max_iterations: usize) -> SciResult<f64> {
        let svd = self.singular_value_decompose(tolerance, max_iterations)?;
        let sigma_max = svd.singular_values.first().copied().unwrap_or(0.0);
        if sigma_max < f64::EPSILON {
            return Ok(0.0); // zero matrix
        }
        let cutoff = tolerance * sigma_max;
        let sigma_min = svd
            .singular_values
            .iter()
            .copied()
            .filter(|&s| s > cutoff)
            .last()
            .unwrap_or(0.0);
        if sigma_min < f64::EPSILON {
            return Ok(f64::INFINITY);
        }
        Ok(sigma_max / sigma_min)
    }

    /// 1-norm condition number κ₁ = ‖A‖₁ · ‖A⁻¹‖₁  (square matrices only).
    pub fn condition_number_1(&self) -> SciResult<f64> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "1-norm condition number requires a square matrix",
            ));
        }
        let norm_a = self.norm_1();
        let inv = self.inverse()?;
        Ok(norm_a * inv.norm_1())
    }

    // ── Matrix Norms ──────────────────────────────────────────────────────────

    /// 1-norm: maximum absolute column sum.
    pub fn norm_1(&self) -> f64 {
        (0..self.cols)
            .map(|c| {
                (0..self.rows)
                    .map(|r| self.data[r * self.cols + c].abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max)
    }

    /// Infinity-norm: maximum absolute row sum.
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| self.data[r * self.cols + c].abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max)
    }

    /// Frobenius norm: √(Σ aᵢⱼ²).
    pub fn norm_frobenius(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn column(&self, col: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|row| self.data[row * self.cols + col])
            .collect()
    }

    fn off_diagonal_norm(&self) -> f64 {
        let mut sum = 0.0;
        for row in 0..self.rows {
            for col in 0..self.cols {
                if row != col {
                    let value = self.data[row * self.cols + col];
                    sum += value * value;
                }
            }
        }
        sum.sqrt()
    }

    fn symmetric_qr_eigendecomposition(
        &self,
        tolerance: f64,
        max_iterations: usize,
    ) -> SciResult<(Vec<f64>, DynamicMatrix)> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "symmetric eigendecomposition requires a square matrix",
            ));
        }

        let mut current = self.clone();
        let mut eigenvectors = DynamicMatrix::identity(self.rows)?;

        for _ in 0..max_iterations {
            let qr = current.qr_decompose()?;
            current = qr.r.mul_matrix(&qr.q)?;
            eigenvectors = eigenvectors.mul_matrix(&qr.q)?;

            if current.off_diagonal_norm() < tolerance {
                let eigenvalues = (0..current.rows)
                    .map(|index| current.data[index * current.cols + index])
                    .collect();
                return Ok((eigenvalues, eigenvectors));
            }
        }

        Err(SciError::NonConvergent("symmetric QR eigendecomposition"))
    }
}

impl LuDecomposition {
    pub fn solve(&self, rhs: &[f64]) -> SciResult<Vec<f64>> {
        if rhs.len() != self.l.rows {
            return Err(SciError::InvalidParameter(
                "right-hand side length must match matrix size",
            ));
        }

        let n = self.l.rows;
        let mut permuted_rhs = vec![0.0; n];
        for (row, &source) in self.permutation.iter().enumerate() {
            permuted_rhs[row] = rhs[source];
        }

        let mut y = vec![0.0; n];
        for row in 0..n {
            let sum: f64 = (0..row)
                .map(|col| self.l.data[row * n + col] * y[col])
                .sum();
            y[row] = permuted_rhs[row] - sum;
        }

        solve_upper_triangular(&self.u, &y)
    }
}

impl CholeskyDecomposition {
    /// Solve A·x = b via forward/backward substitution on L and L^T.
    pub fn solve(&self, rhs: &[f64]) -> SciResult<Vec<f64>> {
        let n = self.l.rows;
        if rhs.len() != n {
            return Err(SciError::InvalidParameter(
                "rhs length must match system size",
            ));
        }
        // Forward substitution: L · y = b
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let sum: f64 = (0..i).map(|k| self.l.data[i * n + k] * y[k]).sum();
            let lii = self.l.data[i * n + i];
            if lii.abs() <= f64::EPSILON {
                return Err(SciError::DivisionByZero);
            }
            y[i] = (rhs[i] - sum) / lii;
        }
        // Backward substitution: L^T · x = y
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let sum: f64 = ((i + 1)..n).map(|k| self.l.data[k * n + i] * x[k]).sum();
            x[i] = (y[i] - sum) / self.l.data[i * n + i];
        }
        Ok(x)
    }

    /// Solve for multiple right-hand sides simultaneously.
    pub fn solve_multi(&self, rhs: &DynamicMatrix) -> SciResult<DynamicMatrix> {
        let n = self.l.rows;
        if rhs.rows != n {
            return Err(SciError::InvalidParameter(
                "rhs rows must match system size",
            ));
        }
        let k = rhs.cols;
        let mut data = vec![0.0f64; n * k];
        for col in 0..k {
            let col_rhs: Vec<f64> = (0..n).map(|r| rhs.data[r * k + col]).collect();
            let sol = self.solve(&col_rhs)?;
            for (r, &v) in sol.iter().enumerate() {
                data[r * k + col] = v;
            }
        }
        DynamicMatrix::new(n, k, data)
    }

    /// Determinant of A: det(A) = (∏ Lᵢᵢ)².
    pub fn determinant(&self) -> f64 {
        let n = self.l.rows;
        let prod: f64 = (0..n).map(|i| self.l.data[i * n + i]).product();
        prod * prod
    }
}

fn vector_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_vector(vector: &mut [f64]) -> SciResult<()> {
    let norm = vector_norm(vector);
    if norm <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn rayleigh_quotient(matrix: &DynamicMatrix, vector: &[f64]) -> SciResult<f64> {
    let av = matrix.mul_vector(vector)?;
    let numerator: f64 = vector.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
    let denominator: f64 = vector.iter().map(|value| value * value).sum();
    if denominator <= f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    Ok(numerator / denominator)
}

fn solve_upper_triangular(matrix: &DynamicMatrix, rhs: &[f64]) -> SciResult<Vec<f64>> {
    if matrix.rows != matrix.cols || rhs.len() != matrix.rows {
        return Err(SciError::InvalidParameter(
            "upper-triangular solve requires square matrix and matching rhs",
        ));
    }

    let n = matrix.rows;
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let diagonal = matrix.data[row * n + row];
        if diagonal.abs() <= f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        let sum: f64 = ((row + 1)..n)
            .map(|col| matrix.data[row * n + col] * solution[col])
            .sum();
        solution[row] = (rhs[row] - sum) / diagonal;
    }
    Ok(solution)
}

// ══════════════════════════════════════════════════════════════════════════════
// DynamicMatrix — advanced methods: element-wise ops, matrix exp, Schur, RRQR, robust SVD
// ══════════════════════════════════════════════════════════════════════════════

impl DynamicMatrix {
    // ── Element-wise / scalar helpers ─────────────────────────────────────────

    pub fn add_matrix(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for add_matrix"));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.rows, self.cols, data)
    }

    pub fn sub_matrix(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for sub_matrix"));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(self.rows, self.cols, data)
    }

    pub fn scale(&self, c: f64) -> SciResult<Self> {
        let data = self.data.iter().map(|v| v * c).collect();
        Self::new(self.rows, self.cols, data)
    }

    // ── Matrix exponential — Padé [3/3] + scaling & squaring ─────────────────

    /// exp(A) via Padé [3/3] approximant with scaling & squaring.
    pub fn matrix_exp(&self) -> SciResult<Self> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter(
                "matrix_exp requires square matrix",
            ));
        }
        let n = self.rows;
        let norm = self.norm_1();
        let s = (norm.log2().ceil() as i64).max(0) as u32;
        let a = self.scale(1.0 / 2.0_f64.powi(s as i32))?;

        // Padé [3/3]: p = 120I + 60A + 12A² + A³,  q same with sign flip on odd
        let id = Self::identity(n)?;
        let a2 = a.mul_matrix(&a)?;
        let a3 = a2.mul_matrix(&a)?;
        let p = id
            .scale(120.0)?
            .add_matrix(&a.scale(60.0)?)?
            .add_matrix(&a2.scale(12.0)?)?
            .add_matrix(&a3)?;
        let q = id
            .scale(120.0)?
            .sub_matrix(&a.scale(60.0)?)?
            .add_matrix(&a2.scale(12.0)?)?
            .sub_matrix(&a3)?;

        // exp(a) ≈ q⁻¹ · p — solve q·X = p column by column via LU
        let lu = q.lu_decompose()?;
        let mut x_data = vec![0.0f64; n * n];
        for col in 0..n {
            let rhs: Vec<f64> = (0..n).map(|r| p.data[r * n + col]).collect();
            let sol = lu.solve(&rhs)?;
            for (r, &v) in sol.iter().enumerate() {
                x_data[r * n + col] = v;
            }
        }
        let mut r = Self::new(n, n, x_data)?;
        for _ in 0..s {
            r = r.mul_matrix(&r)?;
        }
        Ok(r)
    }

    // ── Rank-revealing QR with column pivoting ────────────────────────────────

    /// QR decomposition with column pivoting: A·P = Q·R.
    /// Returns `(Q, R, perm)` where `perm` records column permutations.
    pub fn qr_column_pivot(&self) -> SciResult<(Self, Self, Vec<usize>)> {
        let m = self.rows;
        let n = self.cols;
        let k = m.min(n);
        let mut a = self.data.clone();
        let mut q_data = {
            let mut v = vec![0.0f64; m * m];
            for i in 0..m {
                v[i * m + i] = 1.0;
            }
            v
        };
        let mut perm: Vec<usize> = (0..n).collect();
        let mut col_norms: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j] * a[i * n + j]).sum::<f64>())
            .collect();

        for j in 0..k {
            let pivot = (j..n)
                .max_by(|&c1, &c2| col_norms[c1].total_cmp(&col_norms[c2]))
                .unwrap_or(j);
            if pivot != j {
                for i in 0..m {
                    a.swap(i * n + j, i * n + pivot);
                }
                col_norms.swap(j, pivot);
                perm.swap(j, pivot);
            }
            let x: Vec<f64> = (j..m).map(|i| a[i * n + j]).collect();
            let sigma = la_norm(&x);
            if sigma < f64::EPSILON {
                continue;
            }
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            let mut v = x.clone();
            v[0] += sign * sigma;
            let nv2: f64 = v.iter().map(|vi| vi * vi).sum();
            for jj in j..n {
                let dot: f64 = (0..v.len()).map(|i| v[i] * a[(j + i) * n + jj]).sum();
                let f = 2.0 * dot / nv2;
                for i in 0..v.len() {
                    a[(j + i) * n + jj] -= f * v[i];
                }
            }
            for jj in 0..m {
                let dot: f64 = (0..v.len()).map(|i| v[i] * q_data[(j + i) * m + jj]).sum();
                let f = 2.0 * dot / nv2;
                for i in 0..v.len() {
                    q_data[(j + i) * m + jj] -= f * v[i];
                }
            }
            for jj in (j + 1)..n {
                col_norms[jj] -= a[j * n + jj] * a[j * n + jj];
                col_norms[jj] = col_norms[jj].max(0.0);
            }
        }
        let mut r_data = vec![0.0f64; k * n];
        for i in 0..k {
            for jj in i..n {
                r_data[i * n + jj] = a[i * n + jj];
            }
        }
        let q = Self::new(m, m, q_data)?.transpose();
        let r = Self::new(k, n, r_data)?;
        Ok((q, r, perm))
    }

    // ── Schur decomposition ───────────────────────────────────────────────────

    /// Real Schur decomposition: A = Q·T·Qᵀ.
    /// T is quasi-upper-triangular (2×2 blocks for complex pairs).
    pub fn schur_decompose(&self, tol: f64, max_iter: usize) -> SciResult<(Self, Self)> {
        if self.rows != self.cols {
            return Err(SciError::InvalidParameter("Schur requires square matrix"));
        }
        let n = self.rows;
        let (mut h, mut q) = la_hessenberg(self)?;
        let mut p = n;
        for _ in 0..max_iter {
            while p > 1 {
                let sub = h.data[(p - 1) * n + (p - 2)].abs();
                let scl = h.data[(p - 2) * n + (p - 2)].abs() + h.data[(p - 1) * n + (p - 1)].abs();
                if sub <= tol * scl {
                    h.data[(p - 1) * n + (p - 2)] = 0.0;
                    p -= 1;
                } else {
                    break;
                }
            }
            if p <= 1 {
                break;
            }
            let a11 = h.data[(p - 2) * n + (p - 2)];
            let a12 = h.data[(p - 2) * n + (p - 1)];
            let a21 = h.data[(p - 1) * n + (p - 2)];
            let a22 = h.data[(p - 1) * n + (p - 1)];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let d = ((tr * tr / 4.0 - det).max(0.0)).sqrt();
            let mu1 = tr / 2.0 + d;
            let mu2 = tr / 2.0 - d;
            let shift = if (mu1 - a22).abs() < (mu2 - a22).abs() {
                mu1
            } else {
                mu2
            };
            let (nh, g) = la_qr_step(&h, shift, n)?;
            h = nh;
            q = q.mul_matrix(&g)?;
        }
        Ok((q, h))
    }

    // ── Robust SVD — Golub-Reinsch bidiagonalization ──────────────────────────

    /// SVD via Golub-Reinsch bidiagonalization.
    /// More numerically stable than `singular_value_decompose` for ill-conditioned A.
    pub fn svd_golub_reinsch(
        &self,
        tol: f64,
        max_iter: usize,
    ) -> SciResult<SingularValueDecomposition> {
        let m = self.rows;
        let n = self.cols;
        if m < n {
            let svd = self.transpose().svd_golub_reinsch(tol, max_iter)?;
            return Ok(SingularValueDecomposition {
                u: svd.v_t.transpose(),
                singular_values: svd.singular_values,
                v_t: svd.u.transpose(),
            });
        }
        let (mut u_data, mut d, mut e, mut vt_data) = la_bidiag(&self.data, m, n);
        let mut kk = n;
        for _ in 0..max_iter {
            // Zero small superdiagonals
            for i in 0..kk.saturating_sub(1) {
                if e[i].abs() <= tol * (d[i].abs() + d[i + 1].abs()) {
                    e[i] = 0.0;
                }
            }
            while kk > 1 && e[kk - 2].abs() <= tol * (d[kk - 2].abs() + d[kk - 1].abs()) {
                kk -= 1;
            }
            if kk <= 1 {
                break;
            }

            // Wilkinson shift from BᵀB bottom-right 2×2
            let t11 = d[kk - 2] * d[kk - 2] + (if kk >= 3 { e[kk - 3] * e[kk - 3] } else { 0.0 });
            let t12 = d[kk - 2] * e[kk - 2];
            let t22 = d[kk - 1] * d[kk - 1] + e[kk - 2] * e[kk - 2];
            let mu = la_wilkinson(t11, t12, t22);
            let mut f = (d[0] * d[0] - mu) / d[0].abs().max(f64::EPSILON);
            let mut g = d[0] * e[0];

            for i in 0..(kk - 1) {
                let (c, s) = la_givens(f, g);
                la_givens_cols(&mut vt_data, n, n, i, i + 1, c, s);
                let new_f = c * d[i] + s * e[i];
                e[i] = -s * d[i] + c * e[i];
                g = s * d[i + 1];
                d[i + 1] *= c;
                let (c2, s2) = la_givens(new_f, g);
                la_givens_rows(&mut u_data, m, m, i, i + 1, c2, s2);
                d[i] = c2 * new_f + s2 * g;
                let new_e = c2 * e[i] + s2 * d[i + 1];
                d[i + 1] = -s2 * e[i] + c2 * d[i + 1];
                e[i] = new_e;
                if i + 1 < kk - 1 {
                    f = e[i];
                    g = s2 * e[i + 1];
                    e[i + 1] *= c2;
                }
            }
            for i in 0..n {
                if d[i] < 0.0 {
                    d[i] = -d[i];
                    for j in 0..n {
                        vt_data[i * n + j] = -vt_data[i * n + j];
                    }
                }
            }
        }
        // Sort descending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
        let sv: Vec<f64> = order.iter().map(|&i| d[i]).collect();
        let mut us = vec![0.0f64; m * n];
        let mut vts = vec![0.0f64; n * n];
        for (nc, &old) in order.iter().enumerate() {
            for r in 0..m {
                us[r * n + nc] = u_data[r * m + old];
            }
            for c in 0..n {
                vts[nc * n + c] = vt_data[old * n + c];
            }
        }
        Ok(SingularValueDecomposition {
            u: Self::new(m, n, us)?,
            singular_values: sv,
            v_t: Self::new(n, n, vts)?,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-private helpers for advanced methods
// ─────────────────────────────────────────────────────────────────────────────

fn la_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn la_hessenberg(a: &DynamicMatrix) -> SciResult<(DynamicMatrix, DynamicMatrix)> {
    let n = a.rows;
    let mut h = a.data.to_vec();
    let mut q = {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        v
    };
    for j in 0..n.saturating_sub(2) {
        let x: Vec<f64> = ((j + 1)..n).map(|i| h[i * n + j]).collect();
        let sigma = la_norm(&x);
        if sigma < f64::EPSILON {
            continue;
        }
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = x.clone();
        v[0] += sign * sigma;
        let nv2: f64 = v.iter().map(|vi| vi * vi).sum();
        for jj in j..n {
            let dot: f64 = (0..v.len()).map(|i| v[i] * h[(j + 1 + i) * n + jj]).sum();
            let f = 2.0 * dot / nv2;
            for i in 0..v.len() {
                h[(j + 1 + i) * n + jj] -= f * v[i];
            }
        }
        for ii in 0..n {
            let dot: f64 = (0..v.len()).map(|i| v[i] * h[ii * n + j + 1 + i]).sum();
            let f = 2.0 * dot / nv2;
            for i in 0..v.len() {
                h[ii * n + j + 1 + i] -= f * v[i];
            }
        }
        for ii in 0..n {
            let dot: f64 = (0..v.len()).map(|i| v[i] * q[ii * n + j + 1 + i]).sum();
            let f = 2.0 * dot / nv2;
            for i in 0..v.len() {
                q[ii * n + j + 1 + i] -= f * v[i];
            }
        }
    }
    Ok((
        DynamicMatrix::new(n, n, h)?,
        DynamicMatrix::new(n, n, q)?.transpose(),
    ))
}

fn la_qr_step(
    h: &DynamicMatrix,
    shift: f64,
    n: usize,
) -> SciResult<(DynamicMatrix, DynamicMatrix)> {
    let mut hd = h.data.to_vec();
    let mut g = {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        v
    };
    let mut x = hd[0] - shift;
    let mut y = hd[n];
    for k in 0..(n - 1) {
        let (c, s) = la_givens(x, y);
        for j in k..n {
            let a = hd[k * n + j];
            let b = hd[(k + 1) * n + j];
            hd[k * n + j] = c * a + s * b;
            hd[(k + 1) * n + j] = -s * a + c * b;
        }
        let end = (k + 2).min(n);
        for i in 0..end {
            let a = hd[i * n + k];
            let b = hd[i * n + k + 1];
            hd[i * n + k] = c * a + s * b;
            hd[i * n + k + 1] = -s * a + c * b;
        }
        for i in 0..n {
            let a = g[i * n + k];
            let b = g[i * n + k + 1];
            g[i * n + k] = c * a + s * b;
            g[i * n + k + 1] = -s * a + c * b;
        }
        if k + 2 < n {
            x = hd[(k + 1) * n + k];
            y = hd[(k + 2) * n + k];
        }
    }
    Ok((DynamicMatrix::new(n, n, hd)?, DynamicMatrix::new(n, n, g)?))
}

fn la_bidiag(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut b = a.to_vec();
    let mut u = {
        let mut v = vec![0.0f64; m * m];
        for i in 0..m {
            v[i * m + i] = 1.0;
        }
        v
    };
    let mut vt = {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        v
    };
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n.saturating_sub(1)];
    for j in 0..n {
        let x: Vec<f64> = (j..m).map(|i| b[i * n + j]).collect();
        let sigma = la_norm(&x);
        if sigma > f64::EPSILON {
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            let mut v = x.clone();
            v[0] += sign * sigma;
            let nv2: f64 = v.iter().map(|vi| vi * vi).sum();
            for jj in j..n {
                let dot: f64 = (0..v.len()).map(|i| v[i] * b[(j + i) * n + jj]).sum();
                let f = 2.0 * dot / nv2;
                for i in 0..v.len() {
                    b[(j + i) * n + jj] -= f * v[i];
                }
            }
            for jj in 0..m {
                let dot: f64 = (0..v.len()).map(|i| v[i] * u[jj * m + j + i]).sum();
                let f = 2.0 * dot / nv2;
                for i in 0..v.len() {
                    u[jj * m + j + i] -= f * v[i];
                }
            }
        }
        d[j] = b[j * n + j];
        if j + 1 < n {
            let x: Vec<f64> = ((j + 1)..n).map(|jj| b[j * n + jj]).collect();
            let sigma = la_norm(&x);
            if sigma > f64::EPSILON {
                let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
                let mut v = x.clone();
                v[0] += sign * sigma;
                let nv2: f64 = v.iter().map(|vi| vi * vi).sum();
                for ii in j..m {
                    let dot: f64 = (0..v.len()).map(|i| v[i] * b[ii * n + j + 1 + i]).sum();
                    let f = 2.0 * dot / nv2;
                    for i in 0..v.len() {
                        b[ii * n + j + 1 + i] -= f * v[i];
                    }
                }
                for ii in 0..n {
                    let dot: f64 = (0..v.len()).map(|i| v[i] * vt[ii * n + j + 1 + i]).sum();
                    let f = 2.0 * dot / nv2;
                    for i in 0..v.len() {
                        vt[ii * n + j + 1 + i] -= f * v[i];
                    }
                }
            }
            if j + 1 < n {
                e[j] = b[j * n + j + 1];
            }
        }
    }
    (u, d, e, vt)
}

fn la_givens(f: f64, g: f64) -> (f64, f64) {
    let r = f.hypot(g);
    if r < f64::EPSILON {
        return (1.0, 0.0);
    }
    (f / r, g / r)
}

fn la_givens_rows(a: &mut [f64], _rows: usize, cols: usize, i: usize, j: usize, c: f64, s: f64) {
    for k in 0..cols {
        let ai = a[i * cols + k];
        let aj = a[j * cols + k];
        a[i * cols + k] = c * ai + s * aj;
        a[j * cols + k] = -s * ai + c * aj;
    }
}

fn la_givens_cols(a: &mut [f64], rows: usize, cols: usize, i: usize, j: usize, c: f64, s: f64) {
    for k in 0..rows {
        let ai = a[k * cols + i];
        let aj = a[k * cols + j];
        a[k * cols + i] = c * ai + s * aj;
        a[k * cols + j] = -s * ai + c * aj;
    }
}

fn la_wilkinson(t11: f64, t12: f64, t22: f64) -> f64 {
    let d = (t11 - t22) / 2.0;
    let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
    t22 - sign_d * t12 * t12 / (d.abs() + (d * d + t12 * t12).sqrt())
}
