//! Linear operator abstractions for matrix-free iterative algorithms.
//!
//! Enables defining matrix-vector operations $y = A x$ without explicitly
//! instantiating full dense matrices, which is crucial for large-scale
//! spectral methods and relaxation operators.

use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use crate::sparse::SparseMatrixCsr;
use crate::symmetric::SymmetricMatrix;

/// Abstract linear operator mapping $\mathbb{R}^n \to \mathbb{R}^m$.
pub trait LinearOperator {
    /// Dimension of the output space (number of rows).
    fn rows(&self) -> usize;

    /// Dimension of the input space (number of columns).
    fn cols(&self) -> usize;

    /// Evaluates the forward action $y = A x$.
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>>;

    /// Evaluates the adjoint action $y = A^T x$. Default implementation calls `apply(x)` (self-adjoint).
    fn apply_adjoint(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.apply(x)
    }
}

// Implement for DynamicMatrix
impl LinearOperator for DynamicMatrix {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.mul_vector(x)
    }
    fn apply_adjoint(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.transpose().mul_vector(x)
    }
}

// Implement for SymmetricMatrix
impl LinearOperator for SymmetricMatrix {
    fn rows(&self) -> usize {
        self.size()
    }
    fn cols(&self) -> usize {
        self.size()
    }
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.mul_vector(x)
    }
    fn apply_adjoint(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.mul_vector(x)
    }
}

// Implement for SparseMatrixCsr
impl LinearOperator for SparseMatrixCsr {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.mul_vec(x)
    }
    fn apply_adjoint(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.transpose().mul_vec(x)
    }
}

/// Composite linear operator representing the sum $(A + B) x$.
pub struct SumOperator<'a, Op1: LinearOperator, Op2: LinearOperator> {
    pub op1: &'a Op1,
    pub op2: &'a Op2,
}

impl<'a, Op1: LinearOperator, Op2: LinearOperator> LinearOperator for SumOperator<'a, Op1, Op2> {
    fn rows(&self) -> usize {
        self.op1.rows()
    }
    fn cols(&self) -> usize {
        self.op1.cols()
    }
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        if self.op1.rows() != self.op2.rows() || self.op1.cols() != self.op2.cols() {
            return Err(SciError::InvalidParameter(
                "operator dimensions must match for sum",
            ));
        }
        let y1 = self.op1.apply(x)?;
        let y2 = self.op2.apply(x)?;
        Ok(y1.iter().zip(y2.iter()).map(|(a, b)| a + b).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_matrix_operator() {
        let mat = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = vec![1.0, 0.0, -1.0];
        let y = mat.apply(&x).unwrap();
        assert_eq!(y, vec![-2.0, -2.0]);

        let x_adj = vec![1.0, 2.0];
        let y_adj = mat.apply_adjoint(&x_adj).unwrap();
        // [1 4; 2 5; 3 6] * [1; 2] = [9; 12; 15]
        assert_eq!(y_adj, vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_symmetric_operator() {
        let sym = SymmetricMatrix::from_dense(
            &DynamicMatrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap(),
        )
        .unwrap();

        let x = vec![1.0, 2.0];
        let y = sym.apply(&x).unwrap();
        assert_eq!(y, vec![4.0, 7.0]);
    }
}
