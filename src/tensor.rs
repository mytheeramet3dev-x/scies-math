//! N-dimensional tensors — creation, indexing, decomposition.
//!
//! # Overview
//!
//! [`Tensor`] stores data in a flat `Vec<f64>` with a shape `Vec<usize>`.
//! Indices follow C (row-major) order.
//!
//! # Construction
//!
//! ```ignore
//! use scies_math_th::tensor::Tensor;
//!
//! let t = Tensor::zeros(&[3, 4, 5]);   // 3×4×5 tensor, all zeros
//! let t = Tensor::ones(&[2, 3]);       // 2×3 matrix
//! let t = Tensor::from_fn(&[4, 4], |idx| idx[0] as f64 + idx[1] as f64);
//! ```
//!
//! # Indexing and slicing
//!
//! ```ignore
//! # use scies_math_th::tensor::Tensor;
//! let mut t = Tensor::zeros(&[3, 3]);
//! t.set(&[0, 0], 1.0);
//! let v = t.get(&[0, 0]);  // 1.0
//! ```
//!
//! # Operations
//!
//! | Method | Description |
//! |---|---|
//! | `reshape(shape)` | Change shape (total elements unchanged) |
//! | `transpose_axes(perm)` | Permute axes |
//! | `add`, `sub`, `scale` | Elementwise arithmetic |
//! | `matmul_2d` | Matrix product on last two axes |
//! | `contract(other, axes)` | Einstein summation (tensor contraction) |
//! | `outer(other)` | Outer product |
//!
//! # Decompositions
//!
//! | Function | Description |
//! |---|---|
//! | `hosvd(tensor)` | Higher-order SVD (Tucker decomposition) |
//! | `cp_als(tensor, rank, iters)` | CP decomposition via alternating least squares |
//! | `kronecker_product(a, b)` | Kronecker (tensor) product |
use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;

// ─────────────────────────────────────────────────────────────
// Tensor
// ─────────────────────────────────────────────────────────────

/// Dense N-dimensional tensor stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create from shape and flat row-major data.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> SciResult<Self> {
        if shape.is_empty() {
            return Err(SciError::InvalidParameter("shape must be non-empty"));
        }
        if shape.iter().any(|&d| d == 0) {
            return Err(SciError::InvalidParameter(
                "all dimensions must be positive",
            ));
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(SciError::InvalidParameter(
                "data length does not match shape product",
            ));
        }
        Ok(Self { shape, data })
    }

    /// All-zeros tensor.
    pub fn zeros(shape: Vec<usize>) -> SciResult<Self> {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![0.0; n])
    }

    /// All-ones tensor.
    pub fn ones(shape: Vec<usize>) -> SciResult<Self> {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![1.0; n])
    }

    /// Wrap a 2-D `DynamicMatrix` as a rank-2 tensor.
    pub fn from_matrix(m: &DynamicMatrix) -> SciResult<Self> {
        let rows = m.rows();
        let cols = m.cols();
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(m.get(r, c)?);
            }
        }
        Self::new(vec![rows, cols], data)
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    // ── Element access ────────────────────────────────────────────────────────

    /// Get value at multi-index (length must match rank).
    pub fn get(&self, idx: &[usize]) -> SciResult<f64> {
        let flat = self.flat_index(idx)?;
        Ok(self.data[flat])
    }

    /// Set value at multi-index.
    pub fn set(&mut self, idx: &[usize], value: f64) -> SciResult<()> {
        let flat = self.flat_index(idx)?;
        self.data[flat] = value;
        Ok(())
    }

    // ── Shape manipulation ────────────────────────────────────────────────────

    /// Reshape without copying data; total elements must be preserved.
    pub fn reshape(self, new_shape: Vec<usize>) -> SciResult<Self> {
        let new_n: usize = new_shape.iter().product();
        if new_n != self.data.len() {
            return Err(SciError::InvalidParameter(
                "reshape: product of new shape must equal current number of elements",
            ));
        }
        Ok(Self {
            shape: new_shape,
            data: self.data,
        })
    }

    /// Permute axes.  `axes` must be a permutation of `0..rank`.
    pub fn transpose_axes(&self, axes: &[usize]) -> SciResult<Self> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(SciError::InvalidParameter(
                "axes length must match tensor rank",
            ));
        }
        let mut seen = vec![false; rank];
        for &ax in axes {
            if ax >= rank {
                return Err(SciError::InvalidParameter("axis index out of range"));
            }
            if seen[ax] {
                return Err(SciError::InvalidParameter("duplicate axis in permutation"));
            }
            seen[ax] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&ax| self.shape[ax]).collect();
        let new_strides = strides_of(&new_shape);
        let _old_strides = strides_of(&self.shape);
        let mut data = vec![0.0f64; self.data.len()];

        for flat in 0..self.data.len() {
            let old_idx = flat_to_multi(flat, &self.shape);
            let new_idx: Vec<usize> = axes.iter().map(|&ax| old_idx[ax]).collect();
            let new_flat: usize = new_idx
                .iter()
                .zip(new_strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            data[new_flat] = self.data[flat];
        }
        Self::new(new_shape, data)
    }

    // ── Element-wise ops ──────────────────────────────────────────────────────

    pub fn add(&self, other: &Self) -> SciResult<Self> {
        if self.shape != other.shape {
            return Err(SciError::InvalidParameter(
                "element-wise add requires matching shapes",
            ));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.shape.clone(), data)
    }

    pub fn sub(&self, other: &Self) -> SciResult<Self> {
        if self.shape != other.shape {
            return Err(SciError::InvalidParameter(
                "element-wise sub requires matching shapes",
            ));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(self.shape.clone(), data)
    }

    pub fn scale(&self, s: f64) -> SciResult<Self> {
        let data = self.data.iter().map(|&v| v * s).collect();
        Self::new(self.shape.clone(), data)
    }

    pub fn element_mul(&self, other: &Self) -> SciResult<Self> {
        if self.shape != other.shape {
            return Err(SciError::InvalidParameter(
                "element-wise mul requires matching shapes",
            ));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::new(self.shape.clone(), data)
    }

    // ── Norms ─────────────────────────────────────────────────────────────────

    /// Frobenius norm (√Σ aᵢⱼ…²).
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    // ── Outer product ─────────────────────────────────────────────────────────

    /// Outer product: result shape = `self.shape ++ other.shape`.
    pub fn outer(&self, other: &Self) -> SciResult<Self> {
        let mut shape = self.shape.clone();
        shape.extend_from_slice(&other.shape);
        let data: Vec<f64> = self
            .data
            .iter()
            .flat_map(|&a| other.data.iter().map(move |&b| a * b))
            .collect();
        Self::new(shape, data)
    }

    // ── Contraction ───────────────────────────────────────────────────────────

    /// Contract `self` over axis `axis_a` with `other` over axis `axis_b`.
    ///
    /// The contracted dimension must match.  The result rank = rank(A)+rank(B)-2.
    pub fn contract(&self, axis_a: usize, other: &Self, axis_b: usize) -> SciResult<Self> {
        if axis_a >= self.rank() || axis_b >= other.rank() {
            return Err(SciError::InvalidParameter("contraction axis out of range"));
        }
        let dim = self.shape[axis_a];
        if other.shape[axis_b] != dim {
            return Err(SciError::InvalidParameter(
                "contracted dimensions must match",
            ));
        }

        // Result shape: all dims of self except axis_a, then all of other except axis_b.
        let shape_a: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis_a)
            .map(|(_, &d)| d)
            .collect();
        let shape_b: Vec<usize> = other
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis_b)
            .map(|(_, &d)| d)
            .collect();
        let mut result_shape = shape_a.clone();
        result_shape.extend_from_slice(&shape_b);

        let result_len: usize = if result_shape.is_empty() {
            1
        } else {
            result_shape.iter().product()
        };
        let mut result_data = vec![0.0f64; result_len];

        // Iterate over all multi-indices in result.
        let strides_a = strides_of(&self.shape);
        let strides_b = strides_of(&other.shape);
        let out_strides = strides_of(&result_shape);

        for out_flat in 0..result_len {
            let out_idx = flat_to_multi(out_flat, &result_shape);
            let na = self.rank() - 1; // free dims from A
            let (out_a, out_b) = out_idx.split_at(na);

            let mut sum = 0.0f64;
            for k in 0..dim {
                // Re-assemble multi-index into A: insert k at axis_a.
                let idx_a: Vec<usize> = (0..self.rank())
                    .map(|i| {
                        if i < axis_a {
                            out_a[i]
                        } else if i == axis_a {
                            k
                        } else {
                            out_a[i - 1]
                        }
                    })
                    .collect();
                let idx_b: Vec<usize> = (0..other.rank())
                    .map(|i| {
                        if i < axis_b {
                            out_b[i]
                        } else if i == axis_b {
                            k
                        } else {
                            out_b[i - 1]
                        }
                    })
                    .collect();
                let flat_a: usize = idx_a.iter().zip(strides_a.iter()).map(|(i, s)| i * s).sum();
                let flat_b: usize = idx_b.iter().zip(strides_b.iter()).map(|(i, s)| i * s).sum();
                sum += self.data[flat_a] * other.data[flat_b];
            }
            let out_flat2: usize = out_idx
                .iter()
                .zip(out_strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            result_data[out_flat2] = sum;
        }

        Self::new(result_shape, result_data)
    }

    // ── Mode-n matricisation ──────────────────────────────────────────────────

    /// Unfold the tensor along mode `n` into a `DynamicMatrix`.
    ///
    /// The mode-n unfolding arranges the mode-n fibers as columns.
    pub fn matricize(&self, n: usize) -> SciResult<DynamicMatrix> {
        if n >= self.rank() {
            return Err(SciError::InvalidParameter("mode n out of range"));
        }
        let rows = self.shape[n];
        let cols = self.data.len() / rows;
        let mut data = vec![0.0f64; rows * cols];

        for flat in 0..self.data.len() {
            let idx = flat_to_multi(flat, &self.shape);
            let row = idx[n];
            // Column index: combine all other mode indices in canonical order.
            let mut col = 0usize;
            let mut stride = 1usize;
            for (mode, &dim) in self.shape.iter().enumerate().rev() {
                if mode == n {
                    continue;
                }
                col += idx[mode] * stride;
                stride *= dim;
            }
            data[row * cols + col] = self.data[flat];
        }
        DynamicMatrix::new(rows, cols, data)
    }

    // ── Slicing ───────────────────────────────────────────────────────────────

    /// Fix axis `axis` at index `idx`, returning a tensor of rank-1.
    pub fn slice(&self, axis: usize, idx: usize) -> SciResult<Self> {
        if axis >= self.rank() {
            return Err(SciError::InvalidParameter("axis out of range"));
        }
        if idx >= self.shape[axis] {
            return Err(SciError::InvalidParameter("slice index out of bounds"));
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        let new_n: usize = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        let mut data = vec![0.0f64; new_n];

        for new_flat in 0..new_n {
            let new_idx = flat_to_multi(new_flat, &new_shape);
            let old_idx: Vec<usize> = (0..self.rank())
                .map(|i| {
                    if i < axis {
                        new_idx[i]
                    } else if i == axis {
                        idx
                    } else {
                        new_idx[i - 1]
                    }
                })
                .collect();
            data[new_flat] = self.get(&old_idx)?;
        }

        if new_shape.is_empty() {
            Self::new(vec![1], data)
        } else {
            Self::new(new_shape, data)
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn flat_index(&self, idx: &[usize]) -> SciResult<usize> {
        if idx.len() != self.rank() {
            return Err(SciError::InvalidParameter(
                "index length must match tensor rank",
            ));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (dim, (&i, &size)) in idx.iter().zip(self.shape.iter()).enumerate().rev() {
            if i >= size {
                return Err(SciError::InvalidParameter("index out of bounds"));
            }
            flat += i * stride;
            stride *= size;
            let _ = dim;
        }
        Ok(flat)
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    let mut rem = flat;
    for i in (0..shape.len()).rev() {
        idx[i] = rem % shape[i];
        rem /= shape[i];
    }
    idx
}

// ══════════════════════════════════════════════════════════════════════════════
// Tensor — additional operations (100% coverage extension)
// ══════════════════════════════════════════════════════════════════════════════

impl Tensor {
    // ── Constructors (additional) ─────────────────────────────────────────────

    /// Identity tensor of rank 2 (Kronecker delta δᵢⱼ), shape [n, n].
    pub fn eye(n: usize) -> SciResult<Self> {
        let mut t = Self::zeros(vec![n, n])?;
        for i in 0..n {
            t.set(&[i, i], 1.0)?;
        }
        Ok(t)
    }

    /// Fill tensor with a scalar value.
    pub fn fill(shape: Vec<usize>, value: f64) -> SciResult<Self> {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![value; n])
    }

    /// Create from a closure: `data[i] = f(multi_index)`.
    pub fn from_fn<F: Fn(&[usize]) -> f64>(shape: Vec<usize>, f: F) -> SciResult<Self> {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|flat| {
                let idx = flat_to_multi(flat, &shape);
                f(&idx)
            })
            .collect();
        Self::new(shape, data)
    }

    // ── Unary element-wise ────────────────────────────────────────────────────

    /// Apply f(x) element-wise.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> SciResult<Self> {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Self::new(self.shape.clone(), data)
    }

    /// Negate all elements.
    pub fn neg(&self) -> SciResult<Self> {
        self.map(|v| -v)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> SciResult<Self> {
        self.map(|v| v.abs())
    }

    /// Softmax along `axis` (numerically stable).
    pub fn softmax(&self, axis: usize) -> SciResult<Self> {
        if axis >= self.rank() {
            return Err(SciError::InvalidParameter("axis out of range"));
        }
        let mut out = self.data.clone();
        let ax_size = self.shape[axis];
        let strides = strides_of(&self.shape);
        let ax_stride = strides[axis];
        let n = self.data.len();
        // Iterate over all "lanes" along axis
        let lane_count = n / ax_size;
        for lane in 0..lane_count {
            // Compute lane indices (skip the axis dimension)
            let flat_start = lane_flat(lane, axis, &self.shape, &strides);
            let indices: Vec<usize> = (0..ax_size).map(|k| flat_start + k * ax_stride).collect();
            let max_v = indices
                .iter()
                .map(|&i| out[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = indices.iter().map(|&i| (out[i] - max_v).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (k, &i) in indices.iter().enumerate() {
                out[i] = exps[k] / sum;
            }
        }
        Self::new(self.shape.clone(), out)
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    /// Sum of all elements.
    pub fn sum_all(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Product of all elements.
    pub fn prod_all(&self) -> f64 {
        self.data.iter().product()
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> f64 {
        self.sum_all() / self.data.len() as f64
    }

    /// Maximum element value.
    pub fn max_all(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum element value.
    pub fn min_all(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Flat index of the maximum element.
    pub fn argmax_all(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Flat index of the minimum element.
    pub fn argmin_all(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Sum along `axis` — reduces that dimension to 1 then squeezes.
    pub fn sum_axis(&self, axis: usize) -> SciResult<Self> {
        self.reduce_axis(axis, 0.0, |acc, v| acc + v)
    }

    /// Max along `axis`.
    pub fn max_axis(&self, axis: usize) -> SciResult<Self> {
        self.reduce_axis(
            axis,
            f64::NEG_INFINITY,
            |acc, v| if v > acc { v } else { acc },
        )
    }

    fn reduce_axis<F: Fn(f64, f64) -> f64>(&self, axis: usize, init: f64, f: F) -> SciResult<Self> {
        if axis >= self.rank() {
            return Err(SciError::InvalidParameter("axis out of range"));
        }
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        let n_out: usize = new_shape.iter().product();
        let mut out = vec![init; n_out];
        let out_strides = strides_of(&new_shape);
        for flat in 0..self.data.len() {
            let idx = flat_to_multi(flat, &self.shape);
            let out_idx: Vec<usize> = idx
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != axis)
                .map(|(_, &v)| v)
                .collect();
            let out_flat: usize = if out_idx.is_empty() {
                0
            } else {
                out_idx
                    .iter()
                    .zip(out_strides.iter())
                    .map(|(i, s)| i * s)
                    .sum()
            };
            out[out_flat] = f(out[out_flat], self.data[flat]);
        }
        Self::new(new_shape, out)
    }

    // ── Shape utilities ───────────────────────────────────────────────────────

    /// Concatenate tensors along `axis` (all other dims must match).
    pub fn concatenate(tensors: &[&Self], axis: usize) -> SciResult<Self> {
        if tensors.is_empty() {
            return Err(SciError::InvalidParameter("no tensors"));
        }
        let rank = tensors[0].rank();
        if tensors.iter().any(|t| t.rank() != rank) {
            return Err(SciError::InvalidParameter(
                "all tensors must have same rank",
            ));
        }
        if axis >= rank {
            return Err(SciError::InvalidParameter("axis out of range"));
        }
        let total_ax: usize = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut new_shape = tensors[0].shape.clone();
        new_shape[axis] = total_ax;
        let n_out: usize = new_shape.iter().product();
        let out_strides = strides_of(&new_shape);
        let mut out = vec![0.0f64; n_out];
        let mut ax_offset = 0;
        for t in tensors.iter() {
            for flat in 0..t.data.len() {
                let mut idx = flat_to_multi(flat, &t.shape);
                idx[axis] += ax_offset;
                let out_flat: usize = idx.iter().zip(out_strides.iter()).map(|(i, s)| i * s).sum();
                out[out_flat] = t.data[flat];
            }
            ax_offset += t.shape[axis];
        }
        Self::new(new_shape, out)
    }

    /// Stack tensors along a new axis at position `axis`.
    pub fn stack(tensors: &[&Self], axis: usize) -> SciResult<Self> {
        if tensors.is_empty() {
            return Err(SciError::InvalidParameter("no tensors"));
        }
        let shape = tensors[0].shape.clone();
        if tensors.iter().any(|t| t.shape != shape) {
            return Err(SciError::InvalidParameter(
                "all tensors must have same shape",
            ));
        }
        // Wrap each tensor with a new dimension of size 1 at `axis`, then concatenate
        let expanded: Vec<Tensor> = tensors
            .iter()
            .map(|t| {
                let mut sh = t.shape.clone();
                sh.insert(axis, 1);
                Tensor::new(sh, t.data.clone()).unwrap()
            })
            .collect();
        let refs: Vec<&Tensor> = expanded.iter().collect();
        Self::concatenate(&refs, axis)
    }

    /// Add a dimension of size 1 at `axis` (numpy `expand_dims`).
    pub fn unsqueeze(&self, axis: usize) -> SciResult<Self> {
        if axis > self.rank() {
            return Err(SciError::InvalidParameter("axis out of range"));
        }
        let mut sh = self.shape.clone();
        sh.insert(axis, 1);
        Self::new(sh, self.data.clone())
    }

    /// Remove all dimensions of size 1.
    pub fn squeeze(&self) -> SciResult<Self> {
        let sh: Vec<usize> = self.shape.iter().copied().filter(|&d| d != 1).collect();
        let sh = if sh.is_empty() { vec![1] } else { sh };
        Self::new(sh, self.data.clone())
    }

    // ── Advanced norms ────────────────────────────────────────────────────────

    /// p-norm of all elements: (Σ|xᵢ|^p)^(1/p).
    pub fn norm_p(&self, p: f64) -> f64 {
        self.data
            .iter()
            .map(|v| v.abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p)
    }

    /// Max (infinity) norm.
    pub fn norm_inf(&self) -> f64 {
        self.max_all().abs()
    }

    // ── Kronecker product ─────────────────────────────────────────────────────

    /// Kronecker product of two rank-2 tensors (matrices): A ⊗ B.
    pub fn kron(&self, other: &Self) -> SciResult<Self> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(SciError::InvalidParameter("kron requires rank-2 tensors"));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let (p, q) = (other.shape[0], other.shape[1]);
        let mut data = vec![0.0f64; m * p * n * q];
        for i in 0..m {
            for j in 0..n {
                let a = self.data[i * n + j];
                for k in 0..p {
                    for l in 0..q {
                        data[(i * p + k) * (n * q) + j * q + l] = a * other.data[k * q + l];
                    }
                }
            }
        }
        Self::new(vec![m * p, n * q], data)
    }

    // ── HOSVD (Higher-Order SVD / Tucker decomposition) ───────────────────────

    /// **HOSVD** (de Lathauwer 2000): truncated Tucker decomposition.
    ///
    /// Computes mode-n SVD for each mode and projects onto the leading `ranks[n]`
    /// singular vectors, returning the core tensor G and factor matrices `U[n]`.
    ///
    /// `ranks[n]` must be ≤ `shape[n]` for each mode n.
    pub fn hosvd(&self, ranks: &[usize]) -> SciResult<(Self, Vec<DynamicMatrix>)> {
        let nd = self.rank();
        if ranks.len() != nd {
            return Err(SciError::InvalidParameter(
                "ranks length must equal tensor rank",
            ));
        }
        for (n, &r) in ranks.iter().enumerate() {
            if r == 0 || r > self.shape[n] {
                return Err(SciError::InvalidParameter("rank out of bounds"));
            }
        }

        let mut factors: Vec<DynamicMatrix> = Vec::with_capacity(nd);

        // For each mode: unfold → thin SVD → take leading r columns of U
        for n in 0..nd {
            let mat = self.matricize(n)?;
            let svd = mat.svd_golub_reinsch(1e-14, 1000)?;
            // U is m×min(m,k), take first ranks[n] columns
            let r = ranks[n];
            let m = svd.u.rows();
            let k = svd.u.cols().min(r);
            let mut u_data = vec![0.0f64; m * k];
            for row in 0..m {
                for col in 0..k {
                    u_data[row * k + col] = svd.u.get(row, col)?;
                }
            }
            factors.push(DynamicMatrix::new(m, k, u_data)?);
        }

        // Core tensor G = T ×₁ U₁ᵀ ×₂ U₂ᵀ … ×ₙ Uₙᵀ
        // For each mode n: multiply unfolded tensor by Uₙᵀ then re-fold
        let mut core = self.clone();
        for n in 0..nd {
            let un = &factors[n]; // shape[n] × ranks[n]
            let mat = core.matricize(n)?; // shape[n] × J
            // Uₙᵀ · mat (ranks[n] × J)
            let utn = un.transpose();
            let proj = utn.mul_matrix(&mat)?;
            // Rebuild tensor shape
            let mut new_shape = core.shape.clone();
            new_shape[n] = ranks[n];
            let pr = proj.rows();
            let pc = proj.cols();
            let data: Vec<f64> = (0..pr)
                .flat_map(|r| {
                    let pref = &proj;
                    (0..pc).map(move |c| pref.get(r, c).unwrap_or(0.0))
                })
                .collect();
            // Re-fold: matricize gives (shape[n]) × Π(other dims)
            // We need to invert the matricization — simplest: store as rank-2 then reshape
            // The mode-n unfolding is shape[n] × (product of other modes in order)
            // After projection: ranks[n] × (same product)
            // Reshape back by keeping new_shape
            core = Self::new(new_shape, data)?;
        }
        Ok((core, factors))
    }

    // ── CP rank-1 approximation (HOPM / alternating least squares) ─────────────

    /// **CP decomposition** rank-1 approximation via **Higher-Order Power Method**.
    ///
    /// Returns factor vectors [u₁, u₂, …, uₙ] and scaling λ such that
    /// T ≈ λ · u₁ ⊗ u₂ ⊗ … ⊗ uₙ.
    pub fn cp_rank1(&self, tolerance: f64, max_iter: usize) -> SciResult<(f64, Vec<Vec<f64>>)> {
        let nd = self.rank();
        // Initialise factor vectors to all-ones (normalised)
        let mut u: Vec<Vec<f64>> = self
            .shape
            .iter()
            .map(|&d| vec![1.0f64 / (d as f64).sqrt(); d])
            .collect();
        let mut lam = 1.0f64;

        for _ in 0..max_iter {
            let old_lam = lam;
            for n in 0..nd {
                // Contract T with all factor vectors except mode n
                let contracted = self.mode_product_except(n, &u)?;
                // contracted is a vector of shape[n] elements
                let norm = contracted.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm < f64::EPSILON {
                    break;
                }
                lam = norm;
                u[n] = contracted.iter().map(|v| v / norm).collect();
            }
            if (lam - old_lam).abs() < tolerance {
                break;
            }
        }
        Ok((lam, u))
    }

    /// Contract tensor with all factor vectors except mode `skip`.
    fn mode_product_except(&self, skip: usize, u: &[Vec<f64>]) -> SciResult<Vec<f64>> {
        // Result is a vector of length self.shape[skip]
        let n_skip = self.shape[skip];
        let mut result = vec![0.0f64; n_skip];
        for flat in 0..self.data.len() {
            let idx = flat_to_multi(flat, &self.shape);
            let mut prod = self.data[flat];
            for n in 0..self.rank() {
                if n != skip {
                    prod *= u[n][idx[n]];
                }
            }
            result[idx[skip]] += prod;
        }
        Ok(result)
    }
}

// ── Private helper for softmax ────────────────────────────────────────────────

/// Compute the flat index of the start of a "lane" along `axis`.
fn lane_flat(lane: usize, axis: usize, shape: &[usize], strides: &[usize]) -> usize {
    // Decompose `lane` using all dimensions except `axis`
    let reduced_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .collect();
    if reduced_shape.is_empty() {
        return 0;
    }
    let reduced_strides: Vec<usize> = reduced_shape
        .iter()
        .rev()
        .scan(1usize, |s, &d| {
            let old = *s;
            *s *= d;
            Some(old)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let multi: Vec<usize> = {
        let mut rem = lane;
        reduced_strides
            .iter()
            .map(|&s| {
                let i = rem / s;
                rem %= s;
                i
            })
            .collect()
    };
    // Map back to original strides
    let mut flat = 0;
    let mut ri = 0;
    for i in 0..shape.len() {
        if i == axis {
            continue;
        }
        flat += multi[ri] * strides[i];
        ri += 1;
    }
    flat
}
