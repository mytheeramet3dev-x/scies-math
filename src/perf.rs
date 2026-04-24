//! High-performance matrix multiplication — Phase 1 optimisations.
//!
//! Three backends, selected automatically by matrix size:
//!
//! | Size (max dim) | Backend |
//! |---|---|
//! | < 64 | Naive (fast for tiny matrices) |
//! | 64 – 511 | Cache-tiled (TILE × TILE blocks) |
//! | ≥ 512 | Strassen recursive + tiling at leaves |
//!
//! All backends are pure-Rust, zero-dependency.
//! An optional multithreaded backend (`matmul_threaded`) is available
//! for square matrices where n ≥ 256.

// ══════════════════════════════════════════════════════════════════════════════
// Tile size — must be a power of 2.
// L1 cache line = 64 bytes → 8 f64 per line → 64×64 block ≈ 32 KiB (fits L1).
// ══════════════════════════════════════════════════════════════════════════════
const TILE: usize = 64;

// ══════════════════════════════════════════════════════════════════════════════
// Public entry-point
// ══════════════════════════════════════════════════════════════════════════════

/// Multiply `a` (m×k) by `b` (k×n) into pre-allocated `c` (m×n).
///
/// Automatically selects naive / tiled / Strassen depending on size.
/// `c` must already be zeroed by the caller.
#[inline]
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let max_dim = m.max(k).max(n);
    if max_dim < 64 {
        matmul_naive(a, b, c, m, k, n);
    } else if max_dim < 512 {
        matmul_tiled(a, b, c, m, k, n);
    } else {
        matmul_strassen_dispatch(a, b, c, m, k, n);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Backend 1: Naive (i-k-j loop order — good for small matrices)
// ══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn matmul_naive(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;
            // Inner loop is a contiguous AXPY — very SIMD-friendly.
            for j in 0..n {
                c[c_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Backend 2: Cache-tiled (i-k-j with TILE × TILE blocks)
// ══════════════════════════════════════════════════════════════════════════════

fn matmul_tiled(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i0 in (0..m).step_by(TILE) {
        let i1 = (i0 + TILE).min(m);
        for p0 in (0..k).step_by(TILE) {
            let p1 = (p0 + TILE).min(k);
            for j0 in (0..n).step_by(TILE) {
                let j1 = (j0 + TILE).min(n);
                // Micro-kernel: i-p-j within this tile.
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

// ══════════════════════════════════════════════════════════════════════════════
// Backend 3: Strassen (O(n^2.807) recursive, tile-leaf at TILE)
// ══════════════════════════════════════════════════════════════════════════════

/// Dispatch: pads to next power-of-2 if non-square, then calls Strassen.
fn matmul_strassen_dispatch(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    // If non-square or sizes differ, fall back to tiled (Strassen requires square pow-2).
    if m == k && k == n && m.is_power_of_two() {
        let mut a2 = a.to_vec();
        let mut b2 = b.to_vec();
        let mut c2 = vec![0.0f64; m * m];
        strassen(&a2, &b2, &mut c2, m);
        for (ci, &c2i) in c.iter_mut().zip(&c2) { *ci += c2i; }
    } else {
        // Pad to nearest power-of-2 square, compute, then slice result.
        let s = m.max(k).max(n).next_power_of_two();
        let mut ap = vec![0.0f64; s * s];
        let mut bp = vec![0.0f64; s * s];
        let mut cp = vec![0.0f64; s * s];
        for i in 0..m { for p in 0..k { ap[i*s+p] = a[i*k+p]; } }
        for p in 0..k { for j in 0..n { bp[p*s+j] = b[p*n+j]; } }
        strassen(&ap, &bp, &mut cp, s);
        for i in 0..m { for j in 0..n { c[i*n+j] += cp[i*s+j]; } }
    }
}

/// Recursive Strassen — `n` must be a power of two.
fn strassen(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // Leaf: fall through to tiled GEMM.
    if n <= TILE {
        matmul_tiled(a, b, c, n, n, n);
        return;
    }

    let h = n / 2;

    // Extract sub-matrices (row-major, stride = n).
    let sub = |m: &[f64], r: usize, col: usize| -> Vec<f64> {
        let mut out = vec![0.0f64; h * h];
        for i in 0..h { for j in 0..h { out[i*h+j] = m[(r+i)*n + col+j]; } }
        out
    };
    let add_m = |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(a,b)| a+b).collect() };
    let sub_m = |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(a,b)| a-b).collect() };
    let mut mm   = |a: &[f64], b: &[f64]| -> Vec<f64> { let mut out = vec![0.0f64; h*h]; strassen(a, b, &mut out, h); out };

    let a11 = sub(a, 0, 0); let a12 = sub(a, 0, h);
    let a21 = sub(a, h, 0); let a22 = sub(a, h, h);
    let b11 = sub(b, 0, 0); let b12 = sub(b, 0, h);
    let b21 = sub(b, h, 0); let b22 = sub(b, h, h);

    // 7 recursive multiplications.
    let m1 = mm(&add_m(&a11, &a22), &add_m(&b11, &b22));
    let m2 = mm(&add_m(&a21, &a22), &b11);
    let m3 = mm(&a11, &sub_m(&b12, &b22));
    let m4 = mm(&a22, &sub_m(&b21, &b11));
    let m5 = mm(&add_m(&a11, &a12), &b22);
    let m6 = mm(&sub_m(&a21, &a11), &add_m(&b11, &b12));
    let m7 = mm(&sub_m(&a12, &a22), &add_m(&b21, &b22));

    // Assemble result back into c.
    for i in 0..h {
        for j in 0..h {
            let idx = i*h+j;
            c[ i       *n + j      ] += m1[idx] + m4[idx] - m5[idx] + m7[idx]; // C11
            c[ i       *n + (h+j)  ] += m3[idx] + m5[idx];                      // C12
            c[(h+i)    *n + j      ] += m2[idx] + m4[idx];                       // C21
            c[(h+i)    *n + (h+j)  ] += m1[idx] - m2[idx] + m3[idx] + m6[idx]; // C22
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Backend 4: Multithreaded tiled GEMM (no external deps)
// ══════════════════════════════════════════════════════════════════════════════

/// Parallel tiled matmul using `std::thread`.
///
/// Splits rows of `a` across `n_threads` threads.  Safe because each thread
/// writes to a disjoint row-slice of `c`.
///
/// Useful for square matrices where n ≥ 256.
/// Falls back to `matmul_tiled` for small inputs or n_threads = 1.
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

    // Chunk rows across threads.
    let rows_per = (m + n_threads - 1) / n_threads;
    let a_arc = std::sync::Arc::new(a.to_vec());
    let b_arc = std::sync::Arc::new(b.to_vec());

    // Compute each chunk on a separate thread.
    let chunks: Vec<Vec<f64>> = (0..n_threads).map(|t| {
        let row_start = t * rows_per;
        let row_end   = (row_start + rows_per).min(m);
        if row_start >= m { return vec![0.0f64; 0]; }
        let a_clone = std::sync::Arc::clone(&a_arc);
        let b_clone = std::sync::Arc::clone(&b_arc);
        let rows = row_end - row_start;
        std::thread::spawn(move || {
            let a_slice = &a_clone[row_start * k .. row_end * k];
            let b_ref   = &b_clone[..];
            let mut local = vec![0.0f64; rows * n];
            matmul_tiled(a_slice, b_ref, &mut local, rows, k, n);
            local
        }).join().unwrap_or_else(|_| vec![0.0f64; rows * n])
    }).collect();

    // Write chunks into c.
    let mut offset = 0;
    for (t, chunk) in chunks.iter().enumerate() {
        let row_start = t * rows_per;
        let row_end   = (row_start + rows_per).min(m);
        if row_start >= m || chunk.is_empty() { break; }
        let rows = row_end - row_start;
        c[offset .. offset + rows * n].copy_from_slice(chunk);
        offset += rows * n;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// High-quality PRNG — Xoshiro256** (replaces LCG in monte_carlo / rng)
// ══════════════════════════════════════════════════════════════════════════════

/// Xoshiro256** state — 4 × u64.
///
/// Pass-by-value in hot loops; clone it freely.
/// Period ≈ 2²⁵⁶; passes BigCrush.
#[derive(Clone, Debug)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed from a single u64 (splitmix64 expansion).
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

    /// Uniform f64 in [0, 1).
    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64();
        // Use top 53 bits for mantissa.
        (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64()).max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    /// Uniform usize in `0..n`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    /// Jump ahead by 2^128 steps (parallel RNG streams).
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

// ══════════════════════════════════════════════════════════════════════════════
// Benchmarking utilities (feature-gated)
// ══════════════════════════════════════════════════════════════════════════════

/// Run a simple wall-clock microbenchmark of `matmul` vs naive.
///
/// Returns `(tiled_ns_per_op, naive_ns_per_op)` averaged over `reps` runs.
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
