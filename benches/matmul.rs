//! Benchmark: Matrix multiply — scies-math vs naive O(n³)
//!
//! Run with:
//!   cargo bench --bench matmul
//!
//! HTML report → target/criterion/matmul/report/index.html

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scies_math::linear_algebra::DynamicMatrix;
use scies_math::perf::{matmul, matmul_threaded};

fn make_matrix(n: usize, seed: f64) -> DynamicMatrix {
    let data: Vec<f64> = (0..n * n).map(|i| (i as f64 * seed * 0.001).sin()).collect();
    DynamicMatrix::new(n, n, data).unwrap()
}

// ── scies-math tiled/Strassen (auto-dispatch) ─────────────────────────────

fn bench_scies_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/scies-math");
    for &n in &[32usize, 64, 128, 256, 512] {
        let a = make_matrix(n, 1.7);
        let b = make_matrix(n, 3.1);
        group.throughput(Throughput::Elements((n * n * n) as u64)); // FLOPs proxy
        group.bench_with_input(BenchmarkId::new("auto", n), &n, |bench, _| {
            bench.iter(|| {
                let _ = criterion::black_box(a.mul_matrix(&b).unwrap());
            });
        });
    }
    group.finish();
}

// ── Naive triple-loop baseline ────────────────────────────────────────────

fn naive_matmul(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n { s += a[i*n+k] * b[k*n+j]; }
            c[i*n+j] = s;
        }
    }
}

fn bench_naive_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/naive");
    for &n in &[32usize, 64, 128, 256] {
        let a: Vec<f64> = (0..n*n).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..n*n).map(|i| (n*n - i) as f64 * 0.001).collect();
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bench, _| {
            bench.iter(|| {
                let mut out = vec![0.0f64; n * n];
                naive_matmul(&a, &b, &mut out, n);
                criterion::black_box(out)
            });
        });
    }
    group.finish();
}

// ── Tiled backend explicit ────────────────────────────────────────────────

fn bench_tiled_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/tiled");
    for &n in &[64usize, 128, 256, 512] {
        let a: Vec<f64> = (0..n*n).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..n*n).map(|i| (n*n - i) as f64 * 0.001).collect();
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("tiled", n), &n, |bench, _| {
            bench.iter(|| {
                let mut out = vec![0.0f64; n * n];
                matmul(&a, &b, &mut out, n, n, n);
                criterion::black_box(out)
            });
        });
    }
    group.finish();
}

// ── Multithreaded (4 threads) ────────────────────────────────────────────

fn bench_threaded_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/threaded-4");
    for &n in &[256usize, 512] {
        let a: Vec<f64> = (0..n*n).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..n*n).map(|i| (n*n - i) as f64 * 0.001).collect();
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("threads=4", n), &n, |bench, _| {
            bench.iter(|| {
                let mut out = vec![0.0f64; n * n];
                matmul_threaded(&a, &b, &mut out, n, n, n, 4);
                criterion::black_box(out)
            });
        });
    }
    group.finish();
}

// ── f32 vs f64 comparison ────────────────────────────────────────────────

fn bench_f32_vs_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/f32-vs-f64");
    let n = 256;
    let a64: Vec<f64> = (0..n*n).map(|i| i as f64 * 0.001).collect();
    let b64: Vec<f64> = (0..n*n).map(|i| (n*n - i) as f64 * 0.001).collect();
    let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b64.iter().map(|&v| v as f32).collect();

    group.bench_function("f64-256", |bench| {
        bench.iter(|| {
            let mut out = vec![0.0f64; n * n];
            matmul(&a64, &b64, &mut out, n, n, n);
            criterion::black_box(out)
        });
    });

    group.bench_function("f32-256-via-Mat", |bench| {
        use scies_math::generic::Mat;
        let ma = Mat::<f32>::from_fn(n, n, |r, c| a32[r*n+c]);
        let mb = Mat::<f32>::from_fn(n, n, |r, c| b32[r*n+c]);
        bench.iter(|| {
            criterion::black_box(ma.matmul(&mb).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scies_matmul,
    bench_naive_matmul,
    bench_tiled_matmul,
    bench_threaded_matmul,
    bench_f32_vs_f64,
);
criterion_main!(benches);
