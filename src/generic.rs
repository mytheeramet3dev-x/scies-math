//! Phase 2 — Generic scalar types: `Scalar` trait, `Mat<T>`, `SMatrix<T,R,C>`.
//!
//! # Overview
//!
//! | Type | Dims | Scalar | Use case |
//! |---|---|---|---|
//! | `Mat<T>` | Runtime | `T: Scalar` | General-purpose dynamic matrix |
//! | `SMatrix<T,R,C>` | Compile-time | `T: Scalar` | Zero-overhead fixed-size matrix |
//! | `Vec1<T>` | Runtime | `T: Scalar` | Dense column vector |
//!
//! # Quick start
//! ```rust
//! use scies_math::generic::{Mat, SMatrix, Scalar};
//!
//! // Dynamic f32 matrix
//! let a: Mat<f32> = Mat::zeros(3, 3);
//!
//! // Compile-time 4×4 f64 matrix (zero-cost, no heap alloc)
//! let eye: SMatrix<f64, 4, 4> = SMatrix::identity();
//! ```

use crate::errors::{SciError, SciResult};
use core::ops::{Add, Div, Mul, Neg, Sub};

// ══════════════════════════════════════════════════════════════════════════════
// Scalar trait — abstraction over f32 / f64 (and Complex in a later phase)
// ══════════════════════════════════════════════════════════════════════════════

/// Minimal numeric scalar trait, automatically implemented for `f32` and `f64`.
pub trait Scalar:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + core::fmt::Debug
    + core::fmt::Display
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_finite(self) -> bool;
    fn min_positive() -> Self;
}

impl Scalar for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
    #[inline] fn min_positive() -> Self { f64::MIN_POSITIVE }
}

impl Scalar for f32 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
    #[inline] fn min_positive() -> Self { f32::MIN_POSITIVE }
}

// ══════════════════════════════════════════════════════════════════════════════
// Mat<T> — heap-allocated dynamic matrix, generic over any Scalar
// ══════════════════════════════════════════════════════════════════════════════

/// Dynamic m×n matrix with elements of type `T: Scalar`.
///
/// Row-major storage (C order).  Compatible with `DynamicMatrix` via
/// `Mat::<f64>::from_dynamic` / `.into_dynamic()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> Mat<T> {
    // ── Construction ────────────────────────────────────────────────────────

    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> SciResult<Self> {
        if data.len() != rows * cols {
            return Err(SciError::InvalidParameter("data length ≠ rows × cols"));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![T::zero(); rows * cols] }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![T::one(); rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.data[i * n + i] = T::one(); }
        m
    }

    pub fn from_fn<F: Fn(usize, usize) -> T>(rows: usize, cols: usize, f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows { for c in 0..cols { data.push(f(r, c)); } }
        Self { rows, cols, data }
    }

    // ── Interop with f64 DynamicMatrix ────────────────────────────────────

    /// Convert a `DynamicMatrix` (f64) into `Mat<T>` by casting each element.
    pub fn from_dynamic(m: &crate::linear_algebra::DynamicMatrix) -> Self {
        Self {
            rows: m.rows(),
            cols: m.cols(),
            data: m.raw_data().iter().map(|&v| T::from_f64(v)).collect(),
        }
    }

    /// Cast this matrix to `Mat<f64>` (always lossless from f32, near-lossless vice versa).
    pub fn cast<U: Scalar>(&self) -> Mat<U> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| U::from_f64(v.to_f64())).collect(),
        }
    }

    // ── Indexing ─────────────────────────────────────────────────────────

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> T { self.data[r * self.cols + c] }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: T) { self.data[r * self.cols + c] = v; }

    /// Non-copying row view as a slice.
    #[inline]
    pub fn row_slice(&self, r: usize) -> &[T] {
        &self.data[r * self.cols .. (r + 1) * self.cols]
    }

    /// Mutable row view.
    #[inline]
    pub fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        &mut self.data[r * self.cols .. (r + 1) * self.cols]
    }

    /// Extract a sub-matrix view (copies data — use sparingly on hot paths).
    pub fn submatrix(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> SciResult<Self> {
        if r0 + rows > self.rows || c0 + cols > self.cols {
            return Err(SciError::InvalidParameter("submatrix out of bounds"));
        }
        let data = (r0..r0+rows).flat_map(|r| (c0..c0+cols).map(move |c| self.get(r, c))).collect();
        Ok(Self { rows, cols, data })
    }

    // ── Elementwise ops ───────────────────────────────────────────────────

    pub fn add(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for add"));
        }
        Ok(Self {
            rows: self.rows, cols: self.cols,
            data: self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect(),
        })
    }

    pub fn sub(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for sub"));
        }
        Ok(Self {
            rows: self.rows, cols: self.cols,
            data: self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect(),
        })
    }

    pub fn scale(&self, s: T) -> Self {
        Self {
            rows: self.rows, cols: self.cols,
            data: self.data.iter().map(|&v| v * s).collect(),
        }
    }

    pub fn neg(&self) -> Self { self.scale(-T::one()) }

    pub fn hadamard(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SciError::InvalidParameter("shape mismatch for hadamard"));
        }
        Ok(Self {
            rows: self.rows, cols: self.cols,
            data: self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect(),
        })
    }

    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Self {
        Self { rows: self.rows, cols: self.cols, data: self.data.iter().map(|&v| f(v)).collect() }
    }

    // ── Matrix ops ────────────────────────────────────────────────────────

    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows { for c in 0..self.cols { out.set(c, r, self.get(r, c)); } }
        out
    }

    /// Matrix multiply using the auto-dispatching backend (tiled / Strassen).
    /// Internally converts via f64 — future versions will add a native generic kernel.
    pub fn matmul(&self, other: &Self) -> SciResult<Self> {
        if self.cols != other.rows {
            return Err(SciError::InvalidParameter("incompatible dims for matmul"));
        }
        let (m, k, n) = (self.rows, self.cols, other.cols);
        // Convert to f64 for the perf kernel.
        let af: Vec<f64> = self.data.iter().map(|v| v.to_f64()).collect();
        let bf: Vec<f64> = other.data.iter().map(|v| v.to_f64()).collect();
        let mut cf = vec![0.0f64; m * n];
        crate::perf::matmul(&af, &bf, &mut cf, m, k, n);
        Ok(Self { rows: m, cols: n, data: cf.iter().map(|&v| T::from_f64(v)).collect() })
    }

    /// Mat-vec: self (m×n) * rhs (n) → Vec of length m.
    pub fn matvec(&self, rhs: &[T]) -> SciResult<Vec<T>> {
        if self.cols != rhs.len() { return Err(SciError::InvalidParameter("matvec dim mismatch")); }
        Ok((0..self.rows).map(|r| {
            self.row_slice(r).iter().zip(rhs).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
        }).collect())
    }

    // ── Norms ─────────────────────────────────────────────────────────────

    /// Frobenius norm ‖A‖_F = √Σ a²ᵢⱼ.
    pub fn norm_fro(&self) -> T {
        T::sqrt(self.data.iter().fold(T::zero(), |acc, &v| acc + v * v))
    }

    /// Max absolute value (∞-norm of vec(A)).
    pub fn norm_max(&self) -> T {
        self.data.iter().map(|v| v.abs()).fold(T::zero(), |acc, v| if v > acc { v } else { acc })
    }

    // ── Reductions ────────────────────────────────────────────────────────

    pub fn sum(&self) -> T { self.data.iter().copied().fold(T::zero(), |a, v| a + v) }

    pub fn mean(&self) -> T {
        self.sum() * T::from_f64(1.0 / (self.rows * self.cols) as f64)
    }

    pub fn trace(&self) -> SciResult<T> {
        if self.rows != self.cols { return Err(SciError::InvalidParameter("trace needs square")); }
        Ok((0..self.rows).fold(T::zero(), |acc, i| acc + self.get(i, i)))
    }

    // ── Stacking ──────────────────────────────────────────────────────────

    /// Stack `other` below `self` (same cols required).
    pub fn vstack(&self, other: &Self) -> SciResult<Self> {
        if self.cols != other.cols { return Err(SciError::InvalidParameter("vstack col mismatch")); }
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Ok(Self { rows: self.rows + other.rows, cols: self.cols, data })
    }

    /// Stack `other` to the right of `self` (same rows required).
    pub fn hstack(&self, other: &Self) -> SciResult<Self> {
        if self.rows != other.rows { return Err(SciError::InvalidParameter("hstack row mismatch")); }
        let cols = self.cols + other.cols;
        let mut data = Vec::with_capacity(self.rows * cols);
        for r in 0..self.rows {
            data.extend_from_slice(self.row_slice(r));
            data.extend_from_slice(other.row_slice(r));
        }
        Ok(Self { rows: self.rows, cols, data })
    }
}

// ── Operator overloads for Mat<T> ─────────────────────────────────────────────

impl<T: Scalar> core::ops::Add for &Mat<T> {
    type Output = Mat<T>;
    fn add(self, rhs: Self) -> Mat<T> { self.add(rhs).expect("shape mismatch") }
}

impl<T: Scalar> core::ops::Sub for &Mat<T> {
    type Output = Mat<T>;
    fn sub(self, rhs: Self) -> Mat<T> { self.sub(rhs).expect("shape mismatch") }
}

impl<T: Scalar> core::ops::Mul<T> for &Mat<T> {
    type Output = Mat<T>;
    fn mul(self, rhs: T) -> Mat<T> { self.scale(rhs) }
}

impl<T: Scalar> core::ops::Mul for &Mat<T> {
    type Output = Mat<T>;
    fn mul(self, rhs: Self) -> Mat<T> { self.matmul(rhs).expect("shape mismatch") }
}

impl<T: Scalar> core::fmt::Display for Mat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for r in 0..self.rows {
            write!(f, "[")?;
            for c in 0..self.cols {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{:.6}", self.get(r, c))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SMatrix<T, R, C> — compile-time fixed-size matrix (stack-allocated)
// ══════════════════════════════════════════════════════════════════════════════

/// Fixed-size R×C matrix, stored on the stack.
///
/// Dimension errors are caught **at compile time**.
/// No heap allocation for any operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SMatrix<T: Scalar, const R: usize, const C: usize> {
    pub data: [[T; C]; R],
}

impl<T: Scalar, const R: usize, const C: usize> SMatrix<T, R, C> {
    pub fn zeros() -> Self { Self { data: [[T::zero(); C]; R] } }

    pub fn from_array(data: [[T; C]; R]) -> Self { Self { data } }

    pub fn from_fn<F: Fn(usize, usize) -> T>(f: F) -> Self {
        let mut m = Self::zeros();
        for r in 0..R { for c in 0..C { m.data[r][c] = f(r, c); } }
        m
    }

    #[inline] pub fn get(&self, r: usize, c: usize) -> T { self.data[r][c] }
    #[inline] pub fn set(&mut self, r: usize, c: usize, v: T) { self.data[r][c] = v; }

    pub fn transpose(&self) -> SMatrix<T, C, R> {
        SMatrix::<T, C, R>::from_fn(|r, c| self.data[c][r])
    }

    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Self {
        Self::from_fn(|r, c| f(self.data[r][c]))
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::from_fn(|r, c| self.data[r][c] + other.data[r][c])
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::from_fn(|r, c| self.data[r][c] - other.data[r][c])
    }

    pub fn scale(&self, s: T) -> Self { self.map(|v| v * s) }

    pub fn norm_fro(&self) -> T {
        T::sqrt((0..R).flat_map(|r| (0..C).map(move |c| self.data[r][c]))
            .fold(T::zero(), |acc, v| acc + v * v))
    }

    pub fn norm_max(&self) -> T {
        (0..R).flat_map(|r| (0..C).map(move |c| self.data[r][c]))
            .map(|v| v.abs())
            .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
    }

    /// Convert to heap-allocated `Mat<T>`.
    pub fn to_mat(&self) -> Mat<T> {
        let data: Vec<T> = (0..R).flat_map(|r| self.data[r].iter().copied()).collect();
        Mat { rows: R, cols: C, data }
    }

    /// Multiply two static matrices — dimension checked at compile time.
    pub fn matmul<const K: usize>(&self, other: &SMatrix<T, C, K>) -> SMatrix<T, R, K> {
        SMatrix::<T, R, K>::from_fn(|i, j| {
            (0..C).fold(T::zero(), |acc, k| acc + self.data[i][k] * other.data[k][j])
        })
    }

    /// Matrix-vector product: `self` (R×C) × `v` (C) → `[T; R]`.
    pub fn matvec(&self, v: &[T; C]) -> [T; R] {
        core::array::from_fn(|i| {
            (0..C).fold(T::zero(), |acc, j| acc + self.data[i][j] * v[j])
        })
    }
}

impl<T: Scalar, const N: usize> SMatrix<T, N, N> {
    /// N×N identity matrix.
    pub fn identity() -> Self {
        Self::from_fn(|r, c| if r == c { T::one() } else { T::zero() })
    }

    pub fn trace(&self) -> T {
        (0..N).fold(T::zero(), |acc, i| acc + self.data[i][i])
    }
}

// Operator overloads for SMatrix
impl<T: Scalar, const R: usize, const C: usize> core::ops::Add for SMatrix<T, R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { SMatrix::add(&self, &rhs) }
}

impl<T: Scalar, const R: usize, const C: usize> core::ops::Sub for SMatrix<T, R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { SMatrix::sub(&self, &rhs) }
}

impl<T: Scalar, const R: usize, const C: usize> core::ops::Mul<T> for SMatrix<T, R, C> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self { self.scale(rhs) }
}

impl<T: Scalar, const R: usize, const C: usize> core::ops::Neg for SMatrix<T, R, C> {
    type Output = Self;
    fn neg(self) -> Self { self.scale(-T::one()) }
}

impl<T: Scalar, const R: usize, const C: usize> core::fmt::Display for SMatrix<T, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for r in 0..R {
            write!(f, "[")?;
            for c in 0..C {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{:.6}", self.data[r][c])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Vec1<T> — generic dense column vector
// ══════════════════════════════════════════════════════════════════════════════

/// Dense column vector with `T: Scalar` elements.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec1<T: Scalar> {
    pub data: Vec<T>,
}

impl<T: Scalar> Vec1<T> {
    pub fn new(data: Vec<T>) -> Self { Self { data } }
    pub fn zeros(n: usize) -> Self { Self { data: vec![T::zero(); n] } }
    pub fn ones(n: usize) -> Self { Self { data: vec![T::one(); n] } }
    pub fn from_fn<F: Fn(usize) -> T>(n: usize, f: F) -> Self {
        Self { data: (0..n).map(|i| f(i)).collect() }
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    pub fn dot(&self, other: &Self) -> T {
        self.data.iter().zip(&other.data).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    pub fn norm(&self) -> T { T::sqrt(self.dot(self)) }

    pub fn normalize(&self) -> SciResult<Self> {
        let n = self.norm();
        if n < T::min_positive() { return Err(SciError::DivisionByZero); }
        Ok(Self::new(self.data.iter().map(|&v| v * (T::one() / n)).collect()))
    }

    pub fn add(&self, other: &Self) -> SciResult<Self> {
        if self.data.len() != other.data.len() { return Err(SciError::InvalidParameter("length mismatch")); }
        Ok(Self::new(self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect()))
    }

    pub fn sub(&self, other: &Self) -> SciResult<Self> {
        if self.data.len() != other.data.len() { return Err(SciError::InvalidParameter("length mismatch")); }
        Ok(Self::new(self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect()))
    }

    pub fn scale(&self, s: T) -> Self { Self::new(self.data.iter().map(|&v| v * s).collect()) }

    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Self { Self::new(self.data.iter().map(|&v| f(v)).collect()) }

    pub fn sum(&self) -> T { self.data.iter().copied().fold(T::zero(), |a, v| a + v) }

    pub fn outer(&self, other: &Self) -> Mat<T> {
        let (m, n) = (self.len(), other.len());
        let mut data = Vec::with_capacity(m * n);
        for i in 0..m { for &b in &other.data { data.push(self.data[i] * b); } }
        Mat { rows: m, cols: n, data }
    }

    pub fn as_mat_col(&self) -> Mat<T> {
        Mat { rows: self.len(), cols: 1, data: self.data.clone() }
    }

    pub fn as_mat_row(&self) -> Mat<T> {
        Mat { rows: 1, cols: self.len(), data: self.data.clone() }
    }
}

impl<T: Scalar> core::ops::Index<usize> for Vec1<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.data[i] }
}

impl<T: Scalar> core::ops::IndexMut<usize> for Vec1<T> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.data[i] }
}

impl<T: Scalar> core::fmt::Display for Vec1<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, &v) in self.data.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.6}", v)?;
        }
        write!(f, "]")
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Type aliases for common use-cases
// ══════════════════════════════════════════════════════════════════════════════

/// Convenience: `MatF64` = `Mat<f64>`.
pub type MatF64 = Mat<f64>;
/// Convenience: `MatF32` = `Mat<f32>`.
pub type MatF32 = Mat<f32>;
/// 2×2 f64 matrix on the stack.
pub type SMatrix2f64 = SMatrix<f64, 2, 2>;
/// 3×3 f64 matrix on the stack.
pub type SMatrix3f64 = SMatrix<f64, 3, 3>;
/// 4×4 f64 matrix on the stack (MVP / transforms).
pub type SMatrix4f64 = SMatrix<f64, 4, 4>;
/// 4×4 f32 matrix on the stack (GPU-friendly).
pub type SMatrix4f32 = SMatrix<f32, 4, 4>;
