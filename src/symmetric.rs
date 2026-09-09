//! Packed symmetric matrix storage and operations.
//!
//! An $n \times n$ symmetric matrix satisfies $A_{ij} = A_{ji}$ and can be
//! stored using only $n(n+1)/2$ entries in lower-triangular packed format.
//! This reduces memory footprint by 50% compared to full dense storage,
//! which is essential for larger moment matrices and SDP relaxations.

use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

/// Symmetric matrix using packed lower-triangular storage.
///
/// Element $(i, j)$ where $i \ge j$ is stored at index $i(i+1)/2 + j$.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricMatrix {
    size: usize,
    data: Vec<f64>,
}

impl SymmetricMatrix {
    /// Creates a symmetric matrix from packed lower-triangular data.
    ///
    /// The length of `data` must equal $n(n+1)/2$ where $n = \text{size}$.
    pub fn new(size: usize, data: Vec<f64>) -> SciResult<Self> {
        let expected_len = size * (size + 1) / 2;
        if data.len() != expected_len {
            return Err(SciError::InvalidParameter(
                "packed data length does not match n(n+1)/2",
            ));
        }
        for &val in &data {
            if val.is_nan() || val.is_infinite() {
                return Err(SciError::DomainError(
                    "packed data entries must be finite (not NaN or Inf)",
                ));
            }
        }
        Ok(Self { size, data })
    }

    /// Creates an $n \times n$ zero symmetric matrix.
    pub fn zeros(size: usize) -> SciResult<Self> {
        let len = size * (size + 1) / 2;
        Ok(Self {
            size,
            data: vec![0.0; len],
        })
    }

    /// Creates an $n \times n$ identity symmetric matrix.
    pub fn identity(size: usize) -> SciResult<Self> {
        let mut mat = Self::zeros(size)?;
        for i in 0..size {
            mat.set(i, i, 1.0)?;
        }
        Ok(mat)
    }

    /// Builds a `SymmetricMatrix` from a dense `DynamicMatrix`.
    ///
    /// Validates that the input matrix is square and symmetric within tolerance.
    pub fn from_dense(dense: &DynamicMatrix) -> SciResult<Self> {
        let n = dense.rows();
        if n != dense.cols() {
            return Err(SciError::InvalidParameter("matrix must be square"));
        }

        let mut data = Vec::with_capacity(n * (n + 1) / 2);
        for r in 0..n {
            for c in 0..=r {
                let v1 = dense.get(r, c)?;
                let v2 = dense.get(c, r)?;
                if v1.is_nan() || v1.is_infinite() || v2.is_nan() || v2.is_infinite() {
                    return Err(SciError::DomainError("dense matrix entries must be finite"));
                }
                if (v1 - v2).abs() > 1e-7 * (v1.abs() + v2.abs() + 1.0) {
                    return Err(SciError::InvalidParameter(
                        "input matrix is not symmetric within tolerance",
                    ));
                }
                data.push(0.5 * (v1 + v2));
            }
        }
        Ok(Self { size: n, data })
    }

    /// Converts this packed symmetric matrix into a full dense `DynamicMatrix`.
    pub fn to_dense(&self) -> SciResult<DynamicMatrix> {
        let n = self.size;
        let mut full_data = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                full_data[r * n + c] = self.get(r, c)?;
            }
        }
        DynamicMatrix::new(n, n, full_data)
    }

    /// Dimension $n$ of the $n \times n$ matrix.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of packed elements ($n(n+1)/2$).
    pub fn packed_len(&self) -> usize {
        self.data.len()
    }

    /// Checks whether all entries are finite (not NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Slice of packed lower-triangular values.
    pub fn raw_data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable slice of packed lower-triangular values.
    pub fn raw_data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    #[inline]
    fn packed_index(r: usize, c: usize) -> usize {
        let (i, j) = if r >= c { (r, c) } else { (c, r) };
        i * (i + 1) / 2 + j
    }

    /// Gets entry $A_{r, c}$.
    pub fn get(&self, row: usize, col: usize) -> SciResult<f64> {
        if row >= self.size || col >= self.size {
            return Err(SciError::InvalidParameter("matrix index out of bounds"));
        }
        Ok(self.data[Self::packed_index(row, col)])
    }

    /// Sets entry $A_{r, c}$ (which simultaneously sets $A_{c, r}$).
    pub fn set(&mut self, row: usize, col: usize, val: f64) -> SciResult<()> {
        if row >= self.size || col >= self.size {
            return Err(SciError::InvalidParameter("matrix index out of bounds"));
        }
        if val.is_nan() || val.is_infinite() {
            return Err(SciError::DomainError(
                "cannot set NaN or Inf in SymmetricMatrix",
            ));
        }
        let idx = Self::packed_index(row, col);
        self.data[idx] = val;
        Ok(())
    }

    /// Matrix-vector multiplication $y = A x$.
    pub fn mul_vector(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        let n = self.size;
        if x.len() != n {
            return Err(SciError::InvalidParameter(
                "vector dimension does not match matrix size",
            ));
        }

        let mut y = vec![0.0; n];
        for (i, item) in y.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            for (j, &xj) in x.iter().enumerate().take(n) {
                sum += self.data[Self::packed_index(i, j)] * xj;
            }
            *item = sum;
        }
        Ok(y)
    }

    /// Quadratic form $x^T A x$.
    pub fn quadratic_form(&self, x: &[f64]) -> SciResult<f64> {
        let n = self.size;
        if x.len() != n {
            return Err(SciError::InvalidParameter(
                "vector dimension does not match matrix size",
            ));
        }

        let mut sum = 0.0;
        for (i, &xi) in x.iter().enumerate().take(n) {
            // Diagonal entry
            sum += self.get(i, i)? * xi * xi;
            // Off-diagonal entries
            for (j, &xj) in x.iter().enumerate().take(i) {
                sum += 2.0 * self.get(i, j)? * xi * xj;
            }
        }
        Ok(sum)
    }

    /// Trace inner product (Frobenius inner product) $\langle A, B \rangle = \text{Tr}(A B) = \sum_{i,j} A_{ij} B_{ij}$.
    pub fn trace_inner_product(&self, other: &Self) -> SciResult<f64> {
        if self.size != other.size {
            return Err(SciError::InvalidParameter("matrix dimensions must match"));
        }

        let n = self.size;
        let mut sum = 0.0;
        for i in 0..n {
            // Diagonal terms A_ii * B_ii
            let diag_idx = Self::packed_index(i, i);
            sum += self.data[diag_idx] * other.data[diag_idx];

            // Off-diagonal terms (counted twice due to symmetry)
            for j in 0..i {
                let off_idx = Self::packed_index(i, j);
                sum += 2.0 * self.data[off_idx] * other.data[off_idx];
            }
        }
        Ok(sum)
    }

    /// Frobenius norm $\|A\|_F = \sqrt{\langle A, A \rangle}$.
    pub fn frobenius_norm(&self) -> f64 {
        self.trace_inner_product(self)
            .unwrap_or(0.0)
            .max(0.0)
            .sqrt()
    }

    /// Adds another symmetric matrix of the same size.
    pub fn add(&self, other: &Self) -> SciResult<Self> {
        if self.size != other.size {
            return Err(SciError::InvalidParameter("matrix dimensions must match"));
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Self {
            size: self.size,
            data,
        })
    }

    /// Subtracts another symmetric matrix of the same size.
    pub fn sub(&self, other: &Self) -> SciResult<Self> {
        if self.size != other.size {
            return Err(SciError::InvalidParameter("matrix dimensions must match"));
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(Self {
            size: self.size,
            data,
        })
    }

    /// Scales the matrix by scalar $\alpha$.
    pub fn scale(&self, alpha: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&a| a * alpha).collect();
        Self {
            size: self.size,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_indexing_and_dense_conversion() {
        let dense =
            DynamicMatrix::new(3, 3, vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0]).unwrap();

        let sym = SymmetricMatrix::from_dense(&dense).unwrap();
        assert_eq!(sym.size(), 3);
        assert_eq!(sym.packed_len(), 6); // 3*4/2 = 6

        assert_eq!(sym.get(0, 0).unwrap(), 1.0);
        assert_eq!(sym.get(0, 1).unwrap(), 2.0);
        assert_eq!(sym.get(1, 0).unwrap(), 2.0);
        assert_eq!(sym.get(2, 2).unwrap(), 9.0);

        let roundtrip = sym.to_dense().unwrap();
        assert_eq!(roundtrip.raw_data(), dense.raw_data());
    }

    #[test]
    fn test_quadratic_form_and_trace_product() {
        let sym = SymmetricMatrix::from_dense(
            &DynamicMatrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap(),
        )
        .unwrap();

        let x = vec![1.0, 2.0];
        // x^T A x = 2*(1)^2 + 2*1*(1*2) + 3*(2)^2 = 2 + 4 + 12 = 18
        let qf = sym.quadratic_form(&x).unwrap();
        assert!((qf - 18.0).abs() < 1e-12);

        let id = SymmetricMatrix::identity(2).unwrap();
        // Tr(A * I) = Tr(A) = 2 + 3 = 5
        let tr = sym.trace_inner_product(&id).unwrap();
        assert!((tr - 5.0).abs() < 1e-12);
    }
}
