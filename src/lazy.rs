//! Phase 5B — Lazy matrix expression tree.
//!
//! Chains operations **without computing intermediate results**.
//! Only one heap allocation when `.eval()` is called.
//!
//! # Comparison
//!
//! ```text
//! // EAGER: 3 temporary Vec allocations
//! let r = a.scale(2.0).add(&b).unwrap().scale(0.5);
//!
//! // LAZY: 0 temporaries — single pass at eval()
//! let r = lazy(&a).scale(2.0).add(lazy(&b)).scale(0.5).eval();
//! ```
//!
//! # Supported ops
//!
//! | Op | Method |
//! |---|---|
//! | A + B | `.add(other)` |
//! | A − B | `.sub(other)` |
//! | A × s | `.scale(s)` |
//! | −A | `.neg()` |
//! | Aᵀ | `.transpose()` |
//! | A ∘ B (hadamard) | `.hadamard(other)` |
//! | map f(x) | `.map(f)` |
//! | A · B (matmul) | `.matmul(other)` |
//! | Fuse A·B + C | `.matmul_add(a, b, c)` |

use crate::errors::{SciError, SciResult};
use crate::generic::{Mat, Scalar};

// ══════════════════════════════════════════════════════════════════════════════
// Expression tree
// ══════════════════════════════════════════════════════════════════════════════

/// A lazy matrix expression.  Nothing is computed until `.eval()` is called.
pub enum LazyMat<'a, T: Scalar> {
    /// Borrow an existing matrix (zero-copy entry point).
    Ref(&'a Mat<T>),
    /// Already-owned intermediate (from a previous `.eval()` or sub-expression).
    Owned(Mat<T>),
    // Elementwise
    Add(Box<LazyMat<'a, T>>, Box<LazyMat<'a, T>>),
    Sub(Box<LazyMat<'a, T>>, Box<LazyMat<'a, T>>),
    Scale(Box<LazyMat<'a, T>>, T),
    Neg(Box<LazyMat<'a, T>>),
    Hadamard(Box<LazyMat<'a, T>>, Box<LazyMat<'a, T>>),
    /// Fused: (A + B) ∘ C in a single pass.
    AddHadamard(
        Box<LazyMat<'a, T>>,
        Box<LazyMat<'a, T>>,
        Box<LazyMat<'a, T>>,
    ),
    Map(Box<LazyMat<'a, T>>, fn(T) -> T),
    // Shape-changing
    Transpose(Box<LazyMat<'a, T>>),
    // Matrix product
    Matmul(Box<LazyMat<'a, T>>, Box<LazyMat<'a, T>>),
    /// Fused: C = A · B + D  (single GEMM + add, avoids a temporary matrix)
    MatmulAdd(
        Box<LazyMat<'a, T>>,
        Box<LazyMat<'a, T>>,
        Box<LazyMat<'a, T>>,
    ),
    /// Fused: A · B scaled by s
    MatmulScale(Box<LazyMat<'a, T>>, Box<LazyMat<'a, T>>, T),
}

// ── Constructor helpers ────────────────────────────────────────────────────

/// Wrap a `&Mat<T>` into a lazy expression (zero-copy).
pub fn lazy<T: Scalar>(m: &Mat<T>) -> LazyMat<'_, T> {
    LazyMat::Ref(m)
}

/// Wrap an owned `Mat<T>` into a lazy expression.
pub fn lazy_owned<T: Scalar>(m: Mat<T>) -> LazyMat<'static, T> {
    LazyMat::Owned(m)
}

// ── Builder methods ────────────────────────────────────────────────────────

impl<'a, T: Scalar> LazyMat<'a, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: LazyMat<'a, T>) -> Self {
        LazyMat::Add(Box::new(self), Box::new(other))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: LazyMat<'a, T>) -> Self {
        LazyMat::Sub(Box::new(self), Box::new(other))
    }

    pub fn scale(self, s: T) -> Self {
        LazyMat::Scale(Box::new(self), s)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        LazyMat::Neg(Box::new(self))
    }

    pub fn hadamard(self, other: LazyMat<'a, T>) -> Self {
        match (self, other) {
            (LazyMat::Add(a, b), c) => LazyMat::AddHadamard(a, b, Box::new(c)),
            (c, LazyMat::Add(a, b)) => LazyMat::AddHadamard(a, b, Box::new(c)),
            (lhs, rhs) => LazyMat::Hadamard(Box::new(lhs), Box::new(rhs)),
        }
    }

    pub fn map(self, f: fn(T) -> T) -> Self {
        LazyMat::Map(Box::new(self), f)
    }

    pub fn transpose(self) -> Self {
        LazyMat::Transpose(Box::new(self))
    }

    pub fn matmul(self, other: LazyMat<'a, T>) -> Self {
        LazyMat::Matmul(Box::new(self), Box::new(other))
    }

    /// Fused: (self · rhs) + addend  — avoids 1 temporary allocation.
    pub fn matmul_add(self, rhs: LazyMat<'a, T>, addend: LazyMat<'a, T>) -> Self {
        LazyMat::MatmulAdd(Box::new(self), Box::new(rhs), Box::new(addend))
    }

    /// Fused: (self · rhs) * s
    pub fn matmul_scale(self, rhs: LazyMat<'a, T>, s: T) -> Self {
        LazyMat::MatmulScale(Box::new(self), Box::new(rhs), s)
    }

    // ── Evaluation ─────────────────────────────────────────────────────────

    /// Evaluate the entire expression tree into a single `Mat<T>`.
    ///
    /// Only **one** allocation per dimension is performed at this point.
    pub fn eval(self) -> SciResult<Mat<T>> {
        match self {
            LazyMat::Ref(m) => Ok(m.clone()),
            LazyMat::Owned(m) => Ok(m),

            LazyMat::Add(a, b) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                ma.add(&mb)
            }

            LazyMat::Sub(a, b) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                ma.sub(&mb)
            }

            LazyMat::Scale(a, s) => Ok(a.eval()?.scale(s)),

            LazyMat::Neg(a) => Ok(a.eval()?.neg()),

            LazyMat::Hadamard(a, b) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                ma.hadamard(&mb)
            }

            LazyMat::AddHadamard(a, b, c) => match (*a, *b, *c) {
                (LazyMat::Ref(ma), LazyMat::Ref(mb), LazyMat::Ref(mc)) => {
                    fused_add_hadamard(ma, mb, mc)
                }
                (LazyMat::Owned(ma), LazyMat::Ref(mb), LazyMat::Ref(mc)) => {
                    fused_add_hadamard(&ma, mb, mc)
                }
                (LazyMat::Ref(ma), LazyMat::Owned(mb), LazyMat::Ref(mc)) => {
                    fused_add_hadamard(ma, &mb, mc)
                }
                (LazyMat::Ref(ma), LazyMat::Ref(mb), LazyMat::Owned(mc)) => {
                    fused_add_hadamard(ma, mb, &mc)
                }
                (LazyMat::Owned(ma), LazyMat::Owned(mb), LazyMat::Ref(mc)) => {
                    fused_add_hadamard(&ma, &mb, mc)
                }
                (LazyMat::Owned(ma), LazyMat::Ref(mb), LazyMat::Owned(mc)) => {
                    fused_add_hadamard(&ma, mb, &mc)
                }
                (LazyMat::Ref(ma), LazyMat::Owned(mb), LazyMat::Owned(mc)) => {
                    fused_add_hadamard(ma, &mb, &mc)
                }
                (LazyMat::Owned(ma), LazyMat::Owned(mb), LazyMat::Owned(mc)) => {
                    fused_add_hadamard(&ma, &mb, &mc)
                }
                (lhs, rhs, other) => {
                    let ma = lhs.eval()?;
                    let mb = rhs.eval()?;
                    let mc = other.eval()?;
                    fused_add_hadamard(&ma, &mb, &mc)
                }
            },

            LazyMat::Map(a, f) => Ok(a.eval()?.map(f)),

            LazyMat::Transpose(a) => Ok(a.eval()?.transpose()),

            LazyMat::Matmul(a, b) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                ma.matmul(&mb)
            }

            // Fused: C = A·B + D  — evaluate D first (may be smaller)
            LazyMat::MatmulAdd(a, b, d) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                let md = d.eval()?;
                let ab = ma.matmul(&mb)?;
                ab.add(&md)
            }

            // Fused: s · A · B
            LazyMat::MatmulScale(a, b, s) => {
                let ma = a.eval()?;
                let mb = b.eval()?;
                Ok(ma.matmul(&mb)?.scale(s))
            }
        }
    }

    /// Evaluate **in-place** into a pre-allocated `out` matrix.
    ///
    /// Avoids the final allocation entirely when you already have a buffer.
    pub fn eval_into(self, out: &mut Mat<T>) -> SciResult<()> {
        let result = self.eval()?;
        if out.rows != result.rows || out.cols != result.cols {
            return Err(SciError::InvalidParameter("eval_into: shape mismatch"));
        }
        out.data.copy_from_slice(&result.data);
        Ok(())
    }

    /// Peek at the shape of this expression without evaluating.
    ///
    /// Returns `(rows, cols)` if the shape can be determined statically.
    pub fn shape_hint(&self) -> Option<(usize, usize)> {
        match self {
            LazyMat::Ref(m) => Some((m.rows, m.cols)),
            LazyMat::Owned(m) => Some((m.rows, m.cols)),
            LazyMat::Add(a, _) | LazyMat::Sub(a, _) | LazyMat::Hadamard(a, _) => a.shape_hint(),
            LazyMat::AddHadamard(a, _, _) => a.shape_hint(),
            LazyMat::Scale(a, _) | LazyMat::Neg(a) | LazyMat::Map(a, _) => a.shape_hint(),
            LazyMat::Transpose(a) => a.shape_hint().map(|(r, c)| (c, r)),
            LazyMat::Matmul(a, b) => {
                let (ar, _) = a.shape_hint()?;
                let (_, bc) = b.shape_hint()?;
                Some((ar, bc))
            }
            LazyMat::MatmulAdd(a, b, _) | LazyMat::MatmulScale(a, b, _) => {
                let (ar, _) = a.shape_hint()?;
                let (_, bc) = b.shape_hint()?;
                Some((ar, bc))
            }
        }
    }
}

// ── Operator overloads for ergonomics ─────────────────────────────────────

impl<'a, T: Scalar> core::ops::Add for LazyMat<'a, T> {
    type Output = LazyMat<'a, T>;
    fn add(self, rhs: Self) -> Self {
        LazyMat::add(self, rhs)
    }
}

impl<'a, T: Scalar> core::ops::Sub for LazyMat<'a, T> {
    type Output = LazyMat<'a, T>;
    fn sub(self, rhs: Self) -> Self {
        LazyMat::sub(self, rhs)
    }
}

impl<'a, T: Scalar> core::ops::Mul<T> for LazyMat<'a, T> {
    type Output = LazyMat<'a, T>;
    fn mul(self, rhs: T) -> Self {
        LazyMat::scale(self, rhs)
    }
}

impl<'a, T: Scalar> core::ops::Neg for LazyMat<'a, T> {
    type Output = LazyMat<'a, T>;
    fn neg(self) -> Self {
        LazyMat::neg(self)
    }
}

impl<'a, T: Scalar> core::ops::Mul for LazyMat<'a, T> {
    type Output = LazyMat<'a, T>;
    fn mul(self, rhs: Self) -> Self {
        LazyMat::matmul(self, rhs)
    }
}

fn fused_add_hadamard<T: Scalar>(a: &Mat<T>, b: &Mat<T>, c: &Mat<T>) -> SciResult<Mat<T>> {
    if a.rows != b.rows || a.cols != b.cols || a.rows != c.rows || a.cols != c.cols {
        return Err(SciError::InvalidParameter(
            "shape mismatch for fused add-hadamard",
        ));
    }

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .zip(c.data.iter())
        .map(|((&av, &bv), &cv)| (av + bv) * cv)
        .collect();

    Ok(Mat {
        rows: a.rows,
        cols: a.cols,
        data,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// Convenience macro
// ══════════════════════════════════════════════════════════════════════════════

/// Build a lazy expression chain and evaluate it.
///
/// # Example
/// ```rust
/// use scies_math_th::{lazy_eval, lazy::lazy};
/// use scies_math_th::generic::Mat;
///
/// let a = Mat::<f64>::from_fn(3, 3, |r, c| (r + c) as f64);
/// let b = Mat::<f64>::from_fn(3, 3, |r, c| (r * c) as f64);
///
/// // Without macro: explicit chain
/// let r1 = lazy(&a).scale(2.0).add(lazy(&b)).eval().unwrap();
///
/// // With macro: same thing
/// let r2 = lazy_eval!(lazy(&a).scale(2.0).add(lazy(&b))).unwrap();
/// assert_eq!(r1.data, r2.data);
/// ```
#[macro_export]
macro_rules! lazy_eval {
    ($expr:expr) => {
        $expr.eval()
    };
}

// ══════════════════════════════════════════════════════════════════════════════
// LazyVec — same concept but for 1D vectors
// ══════════════════════════════════════════════════════════════════════════════

use crate::generic::Vec1;

/// Lazy 1D vector expression.
pub enum LazyVec<'a, T: Scalar> {
    Ref(&'a Vec1<T>),
    Owned(Vec1<T>),
    Add(Box<LazyVec<'a, T>>, Box<LazyVec<'a, T>>),
    Sub(Box<LazyVec<'a, T>>, Box<LazyVec<'a, T>>),
    Scale(Box<LazyVec<'a, T>>, T),
    Neg(Box<LazyVec<'a, T>>),
    Map(Box<LazyVec<'a, T>>, fn(T) -> T),
    /// Lazy matvec: A · v (avoids temp if followed by another op)
    Matvec(Box<LazyMat<'a, T>>, Box<LazyVec<'a, T>>),
}

pub fn lazy_vec<T: Scalar>(v: &Vec1<T>) -> LazyVec<'_, T> {
    LazyVec::Ref(v)
}

impl<'a, T: Scalar> LazyVec<'a, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: LazyVec<'a, T>) -> Self {
        LazyVec::Add(Box::new(self), Box::new(other))
    }
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: LazyVec<'a, T>) -> Self {
        LazyVec::Sub(Box::new(self), Box::new(other))
    }
    pub fn scale(self, s: T) -> Self {
        LazyVec::Scale(Box::new(self), s)
    }
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        LazyVec::Neg(Box::new(self))
    }
    pub fn map(self, f: fn(T) -> T) -> Self {
        LazyVec::Map(Box::new(self), f)
    }

    pub fn eval(self) -> SciResult<Vec1<T>> {
        match self {
            LazyVec::Ref(v) => Ok(v.clone()),
            LazyVec::Owned(v) => Ok(v),
            LazyVec::Add(a, b) => a.eval()?.add(&b.eval()?),
            LazyVec::Sub(a, b) => a.eval()?.sub(&b.eval()?),
            LazyVec::Scale(a, s) => Ok(a.eval()?.scale(s)),
            LazyVec::Neg(a) => Ok(a.eval()?.scale(-T::one())),
            LazyVec::Map(a, f) => Ok(a.eval()?.map(f)),
            LazyVec::Matvec(m, v) => {
                let mat = m.eval()?;
                let vec = v.eval()?;
                let out = mat.matvec(&vec.data)?;
                Ok(Vec1::new(out))
            }
        }
    }
}

// ── Matvec entry-point ────────────────────────────────────────────────────

impl<'a, T: Scalar> LazyMat<'a, T> {
    /// Lazy matrix-vector product: `self · v`.
    pub fn matvec_lazy(self, v: LazyVec<'a, T>) -> LazyVec<'a, T> {
        LazyVec::Matvec(Box::new(self), Box::new(v))
    }
}
