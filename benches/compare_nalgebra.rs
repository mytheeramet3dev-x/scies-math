use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use scies_math::generic::Mat;
use scies_math::lazy::lazy;
use scies_math::perf::matmul_threaded;
use nalgebra::DMatrix;

fn make_scies_matrix(n: usize, seed: f64) -> Mat<f64> {
    let data: Vec<f64> = (0..n * n).map(|i| (i as f64 * seed * 0.001).sin()).collect();
    Mat::new(n, n, data).unwrap()
}

fn make_nalgebra_matrix(n: usize, seed: f64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |r, c| ((r * n + c) as f64 * seed * 0.001).sin())
}

// 1. Large Matrix Multiplication
// Comparing scies-math (multithreaded + SIMD) vs nalgebra (single-threaded pure Rust default)
fn bench_matmul_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/matmul_large_512x512");
    let n = 512usize;
    
    let sa = make_scies_matrix(n, 1.7);
    let sb = make_scies_matrix(n, 3.1);
    
    let na = make_nalgebra_matrix(n, 1.7);
    let nb = make_nalgebra_matrix(n, 3.1);
    
    group.throughput(Throughput::Elements((n * n * n) as u64));
    
    group.bench_function("nalgebra_pure_rust", |bench| {
        bench.iter(|| {
            criterion::black_box(&na * &nb)
        });
    });
    
    group.bench_function("scies_math_threaded_8", |bench| {
        let a_data = &sa.data;
        let b_data = &sb.data;
        bench.iter(|| {
            let mut out = vec![0.0; n * n];
            matmul_threaded(a_data, b_data, &mut out, n, n, n, 8);
            criterion::black_box(out)
        });
    });
    
    group.finish();
}

// 2. Element-wise operations (Lazy Evaluation vs Eager Evaluation)
// Operation: (A + B) * C
fn bench_lazy_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/element_wise_lazy_2048x2048");
    let n = 2048usize; // Large matrix to show memory allocation overhead
    
    let sa = make_scies_matrix(n, 1.1);
    let sb = make_scies_matrix(n, 1.2);
    let sc = make_scies_matrix(n, 1.3);
    
    let na = make_nalgebra_matrix(n, 1.1);
    let nb = make_nalgebra_matrix(n, 1.2);
    let nc = make_nalgebra_matrix(n, 1.3);
    
    group.throughput(Throughput::Elements((n * n) as u64));
    
    group.bench_function("nalgebra_eager_allocation", |bench| {
        bench.iter(|| {
            let temp = &na + &nb; // Allocates intermediate RAM
            let result = temp.component_mul(&nc);
            criterion::black_box(result)
        });
    });
    
    group.bench_function("scies_math_lazy_zero_alloc", |bench| {
        bench.iter(|| {
            // Evaluates in a single pass directly into final matrix
            let result = ((lazy(&sa) + lazy(&sb)) * lazy(&sc)).eval().unwrap();
            criterion::black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    compare_benches,
    bench_matmul_compare,
    bench_lazy_compare,
);
criterion_main!(compare_benches);
