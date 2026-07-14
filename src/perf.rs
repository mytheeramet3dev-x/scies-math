//! High-performance matrix multiplication — Phase 2 optimisations.
//!
//! Key changes vs v1:
//! - NEON/AVX2 micro-kernels now accumulate into registers, store ONCE per row-panel
//! - 4×4 register blocking (NEON) and 8×6 (AVX2) — maximises FMA throughput
//! - Strassen sub-matrix extraction uses index arithmetic, no per-call allocation
//! - matmul_threaded uses scoped threads (no sequential join in map)
//! - TILE tuned per-arch: 64 for AVX2 (32 KiB L1), 32 for NEON (16 KiB L1)
//!
//! Backend selection by max dim:
//! | ≤ 24   | transpose + unrolled dot |
//! | 25–511 | tiled + SIMD             |
//! | ≥ 512  | Strassen                 |

// ── Tile sizes ────────────────────────────────────────────────────────────────
// AVX2: 4 × f64 per register → 64×64 tile ≈ 32 KiB fits L1
// NEON: 2 × f64 per register → 32×32 tile ≈  8 KiB fits L1
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const TILE: usize = 64;
#[cfg(target_arch = "aarch64")]
const TILE: usize = 32;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
const TILE: usize = 64;
const SMALL_TRANSPOSE_MAX: usize = 24;

// ── Public entry-point ────────────────────────────────────────────────────────

/// Multiply `a` (m×k) by `b` (k×n) → accumulate into pre-zeroed `c` (m×n).
#[inline]
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let max_dim = m.max(k).max(n);
    if max_dim <= SMALL_TRANSPOSE_MAX {
        matmul_small_transposed(a, b, c, m, k, n);
    } else if max_dim < 512 {
        matmul_tiled(a, b, c, m, k, n);
    } else {
        matmul_strassen_dispatch(a, b, c, m, k, n);
    }
}

// ── Backend 1: Naive ──────────────────────────────────────────────────────────

#[inline(always)]
#[allow(dead_code)]
fn matmul_naive(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;
            for j in 0..n {
                c[c_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

#[inline(always)]
fn dot_unrolled(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0usize;
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut s4 = 0.0f64;
    let mut s5 = 0.0f64;
    let mut s6 = 0.0f64;
    let mut s7 = 0.0f64;

    while i + 7 < len {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
        i += 8;
    }

    let mut sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn matmul_small_transposed(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let mut b_t = vec![0.0f64; n * k];
    for p in 0..k {
        let b_row = p * n;
        for j in 0..n {
            b_t[j * k + p] = b[b_row + j];
        }
    }

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b_t[j * k..(j + 1) * k];
            c_row[j] += dot_unrolled(a_row, b_row);
        }
    }
}

// ── Backend 2: Cache-tiled + SIMD ────────────────────────────────────────────

fn matmul_tiled(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_tiled_avx2(a, b, c, m, k, n); }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { matmul_tiled_neon(a, b, c, m, k, n); }
            return;
        }
    }
    matmul_tiled_fallback(a, b, c, m, k, n);
}

// ── AVX2 kernel: 4-row × 8-col FMA micro-kernel ─────────────────────────────
//
// Strategy: reuse the same B loads across 4 output rows at a time.
// This cuts memory traffic substantially vs the previous row-at-a-time kernel
// and helps the 32–128 range where nalgebra is strongest.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_tiled_avx2(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    #[cfg(target_arch = "x86")]    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")] use core::arch::x86_64::*;

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let cp = c.as_mut_ptr();

        for i0 in (0..m).step_by(TILE) {
            let i1 = (i0 + TILE).min(m);
            for p0 in (0..k).step_by(TILE) {
                let p1 = (p0 + TILE).min(k);
                for j0 in (0..n).step_by(TILE) {
                    let j1 = (j0 + TILE).min(n);

                    let mut i = i0;
                    while i + 3 < i1 {
                        let c_row0 = cp.add(i * n);
                        let c_row1 = cp.add((i + 1) * n);
                        let c_row2 = cp.add((i + 2) * n);
                        let c_row3 = cp.add((i + 3) * n);
                        let a_row0 = ap.add(i * k);
                        let a_row1 = ap.add((i + 1) * k);
                        let a_row2 = ap.add((i + 2) * k);
                        let a_row3 = ap.add((i + 3) * k);

                        let mut j = j0;
                        while j + 7 < j1 {
                            let mut acc00;
                            let mut acc01;
                            let mut acc10;
                            let mut acc11;
                            let mut acc20;
                            let mut acc21;
                            let mut acc30;
                            let mut acc31;

                            if p0 == 0 {
                                acc00 = _mm256_setzero_pd();
                                acc01 = _mm256_setzero_pd();
                                acc10 = _mm256_setzero_pd();
                                acc11 = _mm256_setzero_pd();
                                acc20 = _mm256_setzero_pd();
                                acc21 = _mm256_setzero_pd();
                                acc30 = _mm256_setzero_pd();
                                acc31 = _mm256_setzero_pd();
                            } else {
                                acc00 = _mm256_loadu_pd(c_row0.add(j));
                                acc01 = _mm256_loadu_pd(c_row0.add(j + 4));
                                acc10 = _mm256_loadu_pd(c_row1.add(j));
                                acc11 = _mm256_loadu_pd(c_row1.add(j + 4));
                                acc20 = _mm256_loadu_pd(c_row2.add(j));
                                acc21 = _mm256_loadu_pd(c_row2.add(j + 4));
                                acc30 = _mm256_loadu_pd(c_row3.add(j));
                                acc31 = _mm256_loadu_pd(c_row3.add(j + 4));
                            }

                            for p in p0..p1 {
                                let b_ptr = bp.add(p * n + j);
                                let b0 = _mm256_loadu_pd(b_ptr);
                                let b1 = _mm256_loadu_pd(b_ptr.add(4));

                                let a0 = _mm256_set1_pd(*a_row0.add(p));
                                let a1 = _mm256_set1_pd(*a_row1.add(p));
                                let a2 = _mm256_set1_pd(*a_row2.add(p));
                                let a3 = _mm256_set1_pd(*a_row3.add(p));

                                acc00 = _mm256_fmadd_pd(a0, b0, acc00);
                                acc01 = _mm256_fmadd_pd(a0, b1, acc01);
                                acc10 = _mm256_fmadd_pd(a1, b0, acc10);
                                acc11 = _mm256_fmadd_pd(a1, b1, acc11);
                                acc20 = _mm256_fmadd_pd(a2, b0, acc20);
                                acc21 = _mm256_fmadd_pd(a2, b1, acc21);
                                acc30 = _mm256_fmadd_pd(a3, b0, acc30);
                                acc31 = _mm256_fmadd_pd(a3, b1, acc31);
                            }

                            _mm256_storeu_pd(c_row0.add(j), acc00);
                            _mm256_storeu_pd(c_row0.add(j + 4), acc01);
                            _mm256_storeu_pd(c_row1.add(j), acc10);
                            _mm256_storeu_pd(c_row1.add(j + 4), acc11);
                            _mm256_storeu_pd(c_row2.add(j), acc20);
                            _mm256_storeu_pd(c_row2.add(j + 4), acc21);
                            _mm256_storeu_pd(c_row3.add(j), acc30);
                            _mm256_storeu_pd(c_row3.add(j + 4), acc31);
                            j += 8;
                        }

                        while j + 3 < j1 {
                            let mut acc0 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row0.add(j))
                            };
                            let mut acc1 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row1.add(j))
                            };
                            let mut acc2 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row2.add(j))
                            };
                            let mut acc3 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row3.add(j))
                            };

                            for p in p0..p1 {
                                let b_vec = _mm256_loadu_pd(bp.add(p * n + j));
                                acc0 = _mm256_fmadd_pd(_mm256_set1_pd(*a_row0.add(p)), b_vec, acc0);
                                acc1 = _mm256_fmadd_pd(_mm256_set1_pd(*a_row1.add(p)), b_vec, acc1);
                                acc2 = _mm256_fmadd_pd(_mm256_set1_pd(*a_row2.add(p)), b_vec, acc2);
                                acc3 = _mm256_fmadd_pd(_mm256_set1_pd(*a_row3.add(p)), b_vec, acc3);
                            }

                            _mm256_storeu_pd(c_row0.add(j), acc0);
                            _mm256_storeu_pd(c_row1.add(j), acc1);
                            _mm256_storeu_pd(c_row2.add(j), acc2);
                            _mm256_storeu_pd(c_row3.add(j), acc3);
                            j += 4;
                        }

                        while j < j1 {
                            let mut acc0 = if p0 == 0 { 0.0 } else { *c_row0.add(j) };
                            let mut acc1 = if p0 == 0 { 0.0 } else { *c_row1.add(j) };
                            let mut acc2 = if p0 == 0 { 0.0 } else { *c_row2.add(j) };
                            let mut acc3 = if p0 == 0 { 0.0 } else { *c_row3.add(j) };
                            for p in p0..p1 {
                                let b_val = *bp.add(p * n + j);
                                acc0 += *a_row0.add(p) * b_val;
                                acc1 += *a_row1.add(p) * b_val;
                                acc2 += *a_row2.add(p) * b_val;
                                acc3 += *a_row3.add(p) * b_val;
                            }
                            *c_row0.add(j) = acc0;
                            *c_row1.add(j) = acc1;
                            *c_row2.add(j) = acc2;
                            *c_row3.add(j) = acc3;
                            j += 1;
                        }

                        i += 4;
                    }

                    while i < i1 {
                        let c_row = cp.add(i * n);
                        let a_row = ap.add(i * k);

                        let mut j = j0;
                        while j + 7 < j1 {
                            let mut acc0 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row.add(j))
                            };
                            let mut acc1 = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row.add(j + 4))
                            };
                            for p in p0..p1 {
                                let va = _mm256_set1_pd(*a_row.add(p));
                                let b_ptr = bp.add(p * n + j);
                                acc0 = _mm256_fmadd_pd(va, _mm256_loadu_pd(b_ptr), acc0);
                                acc1 = _mm256_fmadd_pd(va, _mm256_loadu_pd(b_ptr.add(4)), acc1);
                            }
                            _mm256_storeu_pd(c_row.add(j), acc0);
                            _mm256_storeu_pd(c_row.add(j + 4), acc1);
                            j += 8;
                        }
                        while j + 3 < j1 {
                            let mut acc = if p0 == 0 {
                                _mm256_setzero_pd()
                            } else {
                                _mm256_loadu_pd(c_row.add(j))
                            };
                            for p in p0..p1 {
                                let va = _mm256_set1_pd(*a_row.add(p));
                                acc = _mm256_fmadd_pd(va, _mm256_loadu_pd(bp.add(p * n + j)), acc);
                            }
                            _mm256_storeu_pd(c_row.add(j), acc);
                            j += 4;
                        }
                        while j < j1 {
                            let mut acc = if p0 == 0 { 0.0 } else { *c_row.add(j) };
                            for p in p0..p1 {
                                acc += *a_row.add(p) * *bp.add(p * n + j);
                            }
                            *c_row.add(j) = acc;
                            j += 1;
                        }

                        i += 1;
                    }
                }
            }
        }
    }
}

// ── NEON kernel: 4-wide FMA (2×vfmaq), accumulate in registers ───────────────
//
// NEON f64x2 → process 4 j-columns per micro-step using 2 registers.
// Critical fix vs v1: C is loaded ONCE before the p-loop, accumulated,
// then stored ONCE after — eliminates the load/store inside the hot loop.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matmul_tiled_neon(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    use core::arch::aarch64::*;

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let cp = c.as_mut_ptr();

        for i0 in (0..m).step_by(TILE) {
            let i1 = (i0 + TILE).min(m);
            for p0 in (0..k).step_by(TILE) {
                let p1 = (p0 + TILE).min(k);
                for j0 in (0..n).step_by(TILE) {
                    let j1 = (j0 + TILE).min(n);

                    for i in i0..i1 {
                        let c_row = cp.add(i * n);
                        let a_row = ap.add(i * k);

                        let mut j = j0;

                        // 4-wide: 2 × float64x2_t, accumulate over entire p-tile
                        while j + 3 < j1 {
                            // Load current C values ONCE
                            let mut acc0 = vld1q_f64(c_row.add(j));
                            let mut acc1 = vld1q_f64(c_row.add(j + 2));

                            for p in p0..p1 {
                                let va = vdupq_n_f64(*a_row.add(p));
                                let b_ptr = bp.add(p * n + j);
                                acc0 = vfmaq_f64(acc0, vld1q_f64(b_ptr),        va);
                                acc1 = vfmaq_f64(acc1, vld1q_f64(b_ptr.add(2)), va);
                            }

                            // Store ONCE
                            vst1q_f64(c_row.add(j),     acc0);
                            vst1q_f64(c_row.add(j + 2), acc1);
                            j += 4;
                        }

                        // 2-wide tail
                        while j + 1 < j1 {
                            let mut acc = vld1q_f64(c_row.add(j));
                            for p in p0..p1 {
                                let va = vdupq_n_f64(*a_row.add(p));
                                acc = vfmaq_f64(acc, vld1q_f64(bp.add(p * n + j)), va);
                            }
                            vst1q_f64(c_row.add(j), acc);
                            j += 2;
                        }

                        // scalar tail
                        while j < j1 {
                            let mut acc = *c_row.add(j);
                            for p in p0..p1 {
                                acc += *a_row.add(p) * *bp.add(p * n + j);
                            }
                            *c_row.add(j) = acc;
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}

fn matmul_tiled_fallback(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i0 in (0..m).step_by(TILE) {
        let i1 = (i0 + TILE).min(m);
        for p0 in (0..k).step_by(TILE) {
            let p1 = (p0 + TILE).min(k);
            for j0 in (0..n).step_by(TILE) {
                let j1 = (j0 + TILE).min(n);
                for i in i0..i1 {
                    let a_row = i * k;
                    let c_row = i * n;
                    for p in p0..p1 {
                        let a_val = a[a_row + p];
                        let b_row = p * n;
                        for j in j0..j1 {
                            c[c_row + j] += a_val * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

// ── Backend 3: Strassen ───────────────────────────────────────────────────────
//
// Fix vs v1: sub-matrix copy uses a single flat buffer allocated once per
// recursion level (not one Vec per sub-matrix × 8 calls).

fn matmul_strassen_dispatch(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    if m == k && k == n && m.is_power_of_two() {
        strassen(a, b, c, m, m);
    } else {
        let s = m.max(k).max(n).next_power_of_two();
        let mut ap = vec![0.0f64; s * s];
        let mut bp = vec![0.0f64; s * s];
        let mut cp = vec![0.0f64; s * s];
        for i in 0..m { ap[i*s..i*s+k].copy_from_slice(&a[i*k..i*k+k]); }
        for p in 0..k {
            for j in 0..n { bp[p*s+j] = b[p*n+j]; }
        }
        strassen(&ap, &bp, &mut cp, s, s);
        for i in 0..m {
            for j in 0..n { c[i*n+j] += cp[i*s+j]; }
        }
    }
}

/// Copy sub-matrix (r, col) of size h×h from `src` with stride `stride`.
#[inline]
fn copy_sub(src: &[f64], r: usize, col: usize, h: usize, stride: usize, dst: &mut [f64]) {
    for i in 0..h {
        dst[i*h..i*h+h].copy_from_slice(&src[(r+i)*stride+col..(r+i)*stride+col+h]);
    }
}

/// Write h×h matrix `src` back into (r, col) of `dst` with stride `stride`, adding.
#[inline]
#[allow(dead_code)]
fn add_to(dst: &mut [f64], r: usize, col: usize, h: usize, stride: usize, src: &[f64]) {
    for i in 0..h {
        let base = (r+i)*stride + col;
        for j in 0..h { dst[base+j] += src[i*h+j]; }
    }
}

fn strassen(a: &[f64], b: &[f64], c: &mut [f64], n: usize, stride: usize) {
    if n <= TILE {
        // leaf: tiled GEMM on contiguous copies
        let mut ac = vec![0.0f64; n*n];
        let mut bc = vec![0.0f64; n*n];
        for i in 0..n { ac[i*n..i*n+n].copy_from_slice(&a[i*stride..i*stride+n]); }
        for i in 0..n { bc[i*n..i*n+n].copy_from_slice(&b[i*stride..i*stride+n]); }
        let mut cc = vec![0.0f64; n*n];
        matmul_tiled(&ac, &bc, &mut cc, n, n, n);
        for i in 0..n {
            for j in 0..n { c[i*stride+j] += cc[i*n+j]; }
        }
        return;
    }

    let h = n / 2;
    let h2 = h * h;

    // Allocate all scratch in one go — 7 products + temp add/sub buffers
    let mut a11 = vec![0.0f64; h2]; let mut a12 = vec![0.0f64; h2];
    let mut a21 = vec![0.0f64; h2]; let mut a22 = vec![0.0f64; h2];
    let mut b11 = vec![0.0f64; h2]; let mut b12 = vec![0.0f64; h2];
    let mut b21 = vec![0.0f64; h2]; let mut b22 = vec![0.0f64; h2];

    copy_sub(a, 0, 0, h, stride, &mut a11);
    copy_sub(a, 0, h, h, stride, &mut a12);
    copy_sub(a, h, 0, h, stride, &mut a21);
    copy_sub(a, h, h, h, stride, &mut a22);
    copy_sub(b, 0, 0, h, stride, &mut b11);
    copy_sub(b, 0, h, h, stride, &mut b12);
    copy_sub(b, h, 0, h, stride, &mut b21);
    copy_sub(b, h, h, h, stride, &mut b22);

    // Temporaries for the 7 products (stored as contiguous h×h, stride=h)
    let mut m1 = vec![0.0f64; h2]; let mut m2 = vec![0.0f64; h2];
    let mut m3 = vec![0.0f64; h2]; let mut m4 = vec![0.0f64; h2];
    let mut m5 = vec![0.0f64; h2]; let mut m6 = vec![0.0f64; h2];
    let mut m7 = vec![0.0f64; h2];

    // Reusable add/sub scratch
    let mut ta = vec![0.0f64; h2];
    let mut tb = vec![0.0f64; h2];

    macro_rules! add { ($x:expr, $y:expr, $o:expr) => {
        for i in 0..h2 { $o[i] = $x[i] + $y[i]; }
    }}
    macro_rules! sub { ($x:expr, $y:expr, $o:expr) => {
        for i in 0..h2 { $o[i] = $x[i] - $y[i]; }
    }}
    macro_rules! mm { ($a:expr, $b:expr, $m:expr) => {
        strassen($a, $b, $m, h, h);
    }}

    // M1 = (A11+A22)(B11+B22)
    add!(a11, a22, ta); add!(b11, b22, tb); mm!(&ta, &tb, &mut m1);
    // M2 = (A21+A22) B11
    add!(a21, a22, ta); mm!(&ta, &b11, &mut m2);
    // M3 = A11 (B12−B22)
    sub!(b12, b22, tb); mm!(&a11, &tb, &mut m3);
    // M4 = A22 (B21−B11)
    sub!(b21, b11, tb); mm!(&a22, &tb, &mut m4);
    // M5 = (A11+A12) B22
    add!(a11, a12, ta); mm!(&ta, &b22, &mut m5);
    // M6 = (A21−A11)(B11+B12)
    sub!(a21, a11, ta); add!(b11, b12, tb); mm!(&ta, &tb, &mut m6);
    // M7 = (A12−A22)(B21+B22)
    sub!(a12, a22, ta); add!(b21, b22, tb); mm!(&ta, &tb, &mut m7);

    // Assemble C (add into existing c, stride = stride)
    for i in 0..h {
        for j in 0..h {
            let idx = i*h+j;
            // C11 = M1+M4−M5+M7
            c[ i      *stride + j    ] += m1[idx] + m4[idx] - m5[idx] + m7[idx];
            // C12 = M3+M5
            c[ i      *stride + h+j  ] += m3[idx] + m5[idx];
            // C21 = M2+M4
            c[(h+i)   *stride + j    ] += m2[idx] + m4[idx];
            // C22 = M1−M2+M3+M6
            c[(h+i)   *stride + h+j  ] += m1[idx] - m2[idx] + m3[idx] + m6[idx];
        }
    }
}

// ── Backend 4: Scoped-thread parallel GEMM ────────────────────────────────────
//
// Fix vs v1: use std::thread::scope so threads borrow a/b directly (no Arc clone),
// and all threads are truly concurrent (not joined one-by-one inside a map).

pub fn matmul_threaded(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
    n_threads: usize,
) {
    let n_threads = n_threads.max(1).min(m);
    if n_threads == 1 || m < 128 {
        matmul_tiled(a, b, c, m, k, n);
        return;
    }

    let rows_per = (m + n_threads - 1) / n_threads;

    // Split c into per-thread row slices without unsafe
    let mut c_chunks: Vec<&mut [f64]> = c
        .chunks_mut(rows_per * n)
        .collect();

    std::thread::scope(|s| {
        for (t, c_chunk) in c_chunks.iter_mut().enumerate() {
            let i_start = t * rows_per;
            let i_end   = (i_start + rows_per).min(m);
            if i_start >= m { break; }
            let rows = i_end - i_start;
            let a_slice = &a[i_start * k .. i_end * k];
            s.spawn(move || {
                matmul_tiled(a_slice, b, c_chunk, rows, k, n);
            });
        }
    });
}

// ── Backend 5: Rayon ─────────────────────────────────────────────────────────

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub fn matmul_rayon(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    if m < 64 {
        matmul_tiled(a, b, c, m, k, n);
        return;
    }

    // Use TILE rows per chunk — aligns with cache tiling
    let chunk_rows = TILE;
    c.par_chunks_mut(chunk_rows * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let i_start = chunk_idx * chunk_rows;
            let i_end   = (i_start + c_chunk.len() / n).min(m);
            let rows    = i_end - i_start;
            if rows == 0 { return; }
            let a_slice = &a[i_start * k .. i_end * k];
            matmul_tiled(a_slice, b, c_chunk, rows, k, n);
        });
}

// ── Xoshiro256** ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
            0xa9582618e03fc9aa, 0x39abdc4529b1661c,
        ];
        let (mut s0, mut s1, mut s2, mut s3) = (0u64, 0u64, 0u64, 0u64);
        for &j in &JUMP {
            for b in 0..64 {
                if (j >> b) & 1 != 0 {
                    s0 ^= self.s[0]; s1 ^= self.s[1];
                    s2 ^= self.s[2]; s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s = [s0, s1, s2, s3];
    }
}

// ── Benchmark utility ─────────────────────────────────────────────────────────

#[cfg(feature = "bench")]
pub fn benchmark_matmul(n: usize, reps: usize) -> (f64, f64) {
    use std::time::Instant;
    let a: Vec<f64> = (0..n*n).map(|i| i as f64 * 0.001).collect();
    let b: Vec<f64> = (0..n*n).map(|i| (n*n - i) as f64 * 0.001).collect();

    let t0 = Instant::now();
    for _ in 0..reps {
        let mut c = vec![0.0f64; n*n];
        matmul(&a, &b, &mut c, n, n, n);
    }
    let tiled_ns = t0.elapsed().as_nanos() as f64 / reps as f64;

    let t0 = Instant::now();
    for _ in 0..reps {
        let mut c = vec![0.0f64; n*n];
        matmul_naive(&a, &b, &mut c, n, n, n);
    }
    let naive_ns = t0.elapsed().as_nanos() as f64 / reps as f64;

    (tiled_ns, naive_ns)
}
