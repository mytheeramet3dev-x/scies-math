mod common;

use common::{rectangular_matrix_data, spd_matrix_data, square_matrix_data, vector_data};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use scies_math_th::eigensystem::jacobi_eigen;
use scies_math_th::linear_algebra::DynamicMatrix;

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/lu_decompose");
    for size in [32usize, 64, 128] {
        let data = square_matrix_data(size, 13);
        let na_mat = DMatrix::from_row_slice(size, size, &data);
        let sm_mat = DynamicMatrix::new(size, size, data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(na_mat.clone()).lu()),
        );
        group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&sm_mat).lu_decompose().unwrap()),
        );
    }
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/qr_decompose");
    for size in [32usize, 64, 128] {
        let data = square_matrix_data(size, 17);
        let na_mat = DMatrix::from_row_slice(size, size, &data);
        let sm_mat = DynamicMatrix::new(size, size, data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(na_mat.clone()).qr()),
        );
        group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&sm_mat).qr_decompose().unwrap()),
        );
    }
    group.finish();
}

fn bench_least_squares(c: &mut Criterion) {
    let rows = 96usize;
    let cols = 32usize;
    let matrix_data = rectangular_matrix_data(rows, cols, 53);
    let rhs_data = vector_data(rows, 17);

    let sm_mat = DynamicMatrix::new(rows, cols, matrix_data.clone()).unwrap();
    let na_mat = DMatrix::from_row_slice(rows, cols, &matrix_data);
    let na_rhs = DVector::from_row_slice(&rhs_data);

    let mut group = c.benchmark_group("compare/least_squares");
    group.sample_size(20);
    group.bench_function(
        BenchmarkId::new("scies-math", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                black_box(&sm_mat)
                    .least_squares(black_box(&rhs_data))
                    .unwrap()
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("nalgebra", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                black_box(na_mat.clone())
                    .qr()
                    .solve(black_box(&na_rhs))
                    .unwrap()
            })
        },
    );
    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/cholesky_decompose");
    group.sample_size(20);

    for size in [32usize, 64, 128] {
        let data = spd_matrix_data(size);
        let na_mat = DMatrix::from_row_slice(size, size, &data);
        let sm_mat = DynamicMatrix::new(size, size, data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(na_mat.clone()).cholesky().unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&sm_mat).cholesky_decompose().unwrap()),
        );
    }

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/inverse");
    group.sample_size(20);

    for size in [16usize, 32, 64] {
        let data = square_matrix_data(size, 37);
        let na_mat = DMatrix::from_row_slice(size, size, &data);
        let sm_mat = DynamicMatrix::new(size, size, data.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(na_mat.clone()).try_inverse().unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&sm_mat).inverse().unwrap()),
        );
    }

    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let rows = 24usize;
    let cols = 16usize;
    let data = rectangular_matrix_data(rows, cols, 71);
    let na_mat = DMatrix::from_row_slice(rows, cols, &data);
    let sm_mat = DynamicMatrix::new(rows, cols, data.clone()).unwrap();

    let mut group = c.benchmark_group("compare/svd");
    group.sample_size(10);
    group.bench_function(
        BenchmarkId::new("nalgebra", format!("{rows}x{cols}")),
        |b| b.iter(|| black_box(na_mat.clone()).svd(true, true)),
    );
    group.bench_function(
        BenchmarkId::new("scies-math", format!("{rows}x{cols}")),
        |b| b.iter(|| black_box(&sm_mat).svd_golub_reinsch(1e-10, 512).unwrap()),
    );
    group.finish();
}

fn bench_eigen(c: &mut Criterion) {
    let size = 64usize;
    let data = spd_matrix_data(size);
    let na_mat = DMatrix::from_row_slice(size, size, &data);
    let sm_mat = DynamicMatrix::new(size, size, data.clone()).unwrap();

    let mut group = c.benchmark_group("compare/symmetric_eigen");
    group.sample_size(10);
    group.bench_function(
        BenchmarkId::new("nalgebra", format!("{size}x{size}")),
        |b| b.iter(|| black_box(na_mat.clone()).symmetric_eigen()),
    );
    group.bench_function(
        BenchmarkId::new("scies-math", format!("{size}x{size}")),
        |b| b.iter(|| black_box(jacobi_eigen(&sm_mat, 1e-10, 1000)).unwrap()),
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_lu,
    bench_qr,
    bench_least_squares,
    bench_cholesky,
    bench_inverse,
    bench_svd,
    bench_eigen
);
criterion_main!(benches);
