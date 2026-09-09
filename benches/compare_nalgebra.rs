mod common;

use common::{square_matrix_data, vector_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use nalgebra as na;
use nalgebra::{DMatrix, DVector};
use scies_math_th::generic::SMatrix;
use scies_math_th::lazy::lazy;
use scies_math_th::linear_algebra::{DynamicMatrix, Matrix3, Vector3};
#[cfg(feature = "parallel")]
use scies_math_th::perf::matmul_rayon;
use scies_math_th::perf::matmul_threaded;
use scies_math_th::transform::{Isometry3, Quaternion, Rotation3};

fn bench_small_vector_and_matrix_ops(c: &mut Criterion) {
    let scies_a = Vector3::new(1.25, -2.75, 3.5);
    let scies_b = Vector3::new(-4.0, 0.5, 2.125);
    let na_a = na::Vector3::new(1.25, -2.75, 3.5);
    let na_b = na::Vector3::new(-4.0, 0.5, 2.125);

    let mut dot_group = c.benchmark_group("compare/vector3_dot");
    dot_group.bench_function(BenchmarkId::new("scies-math", "f64"), |b| {
        b.iter(|| black_box(scies_a).dot(black_box(scies_b)))
    });
    dot_group.bench_function(BenchmarkId::new("nalgebra", "f64"), |b| {
        b.iter(|| black_box(na_a).dot(&black_box(na_b)))
    });
    dot_group.finish();

    let mut cross_group = c.benchmark_group("compare/vector3_cross");
    cross_group.bench_function(BenchmarkId::new("scies-math", "f64"), |b| {
        b.iter(|| black_box(scies_a).cross(black_box(scies_b)))
    });
    cross_group.bench_function(BenchmarkId::new("nalgebra", "f64"), |b| {
        b.iter(|| black_box(na_a).cross(&black_box(na_b)))
    });
    cross_group.finish();

    let scies_matrix = Matrix3::new([[3.0, -1.5, 0.25], [2.0, 5.0, -0.5], [0.75, 1.25, 4.0]]);
    let scies_vector = Vector3::new(1.0, -0.75, 2.25);
    let na_matrix = na::Matrix3::new(3.0, -1.5, 0.25, 2.0, 5.0, -0.5, 0.75, 1.25, 4.0);
    let na_vector = na::Vector3::new(1.0, -0.75, 2.25);

    let mut mul_group = c.benchmark_group("compare/matrix3_mul_vector");
    mul_group.bench_function(BenchmarkId::new("scies-math", "3x3"), |b| {
        b.iter(|| black_box(scies_matrix).mul_vector(black_box(scies_vector)))
    });
    mul_group.bench_function(BenchmarkId::new("nalgebra", "3x3"), |b| {
        b.iter(|| black_box(na_matrix) * black_box(na_vector))
    });
    mul_group.finish();
}

fn bench_static_smatrix_ops(c: &mut Criterion) {
    let scies_a = SMatrix::<f64, 4, 4>::from_array([
        [4.0, 1.0, -2.0, 0.5],
        [0.75, 3.5, 1.25, -1.0],
        [2.0, -0.25, 5.0, 1.75],
        [1.5, 0.0, -1.25, 2.5],
    ]);
    let scies_b = SMatrix::<f64, 4, 4>::from_array([
        [1.0, 0.5, -0.25, 2.0],
        [2.25, -1.0, 0.75, 0.0],
        [0.5, 1.5, 3.0, -0.5],
        [-1.25, 2.0, 0.0, 1.0],
    ]);
    let scies_v = [1.0, -2.0, 0.5, 3.0];

    let na_a = na::SMatrix::<f64, 4, 4>::from_row_slice(&[
        4.0, 1.0, -2.0, 0.5, 0.75, 3.5, 1.25, -1.0, 2.0, -0.25, 5.0, 1.75, 1.5, 0.0, -1.25, 2.5,
    ]);
    let na_b = na::SMatrix::<f64, 4, 4>::from_row_slice(&[
        1.0, 0.5, -0.25, 2.0, 2.25, -1.0, 0.75, 0.0, 0.5, 1.5, 3.0, -0.5, -1.25, 2.0, 0.0, 1.0,
    ]);
    let na_v = na::SVector::<f64, 4>::from_row_slice(&[1.0, -2.0, 0.5, 3.0]);

    let mut matmul_group = c.benchmark_group("compare/smatrix4_matmul");
    matmul_group.bench_function(BenchmarkId::new("scies-math", "4x4"), |b| {
        b.iter(|| black_box(scies_a).matmul(&black_box(scies_b)))
    });
    matmul_group.bench_function(BenchmarkId::new("nalgebra", "4x4"), |b| {
        b.iter(|| black_box(na_a) * black_box(na_b))
    });
    matmul_group.finish();

    let mut matvec_group = c.benchmark_group("compare/smatrix4_matvec");
    matvec_group.bench_function(BenchmarkId::new("scies-math", "4x4"), |b| {
        b.iter(|| black_box(scies_a).matvec(&black_box(scies_v)))
    });
    matvec_group.bench_function(BenchmarkId::new("nalgebra", "4x4"), |b| {
        b.iter(|| black_box(na_a) * black_box(na_v))
    });
    matvec_group.finish();
}

fn bench_dynamic_linear_algebra(c: &mut Criterion) {
    let mut matmul_group = c.benchmark_group("compare/dynamic_matmul");
    for size in [16usize, 32, 64] {
        let data_a = square_matrix_data(size, 7);
        let data_b = square_matrix_data(size, 19);

        let scies_a = DynamicMatrix::new(size, size, data_a.clone()).unwrap();
        let scies_b = DynamicMatrix::new(size, size, data_b.clone()).unwrap();
        let na_a = DMatrix::from_row_slice(size, size, &data_a);
        let na_b = DMatrix::from_row_slice(size, size, &data_b);

        matmul_group.throughput(Throughput::Elements((size * size * size) as u64));
        matmul_group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&scies_a).mul_matrix(black_box(&scies_b)).unwrap()),
        );
        matmul_group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| b.iter(|| black_box(&na_a) * black_box(&na_b)),
        );
    }
    matmul_group.finish();

    let mut solve_group = c.benchmark_group("compare/dynamic_solve_linear_system");
    for size in [16usize, 32, 64] {
        let matrix_data = square_matrix_data(size, 29);
        let rhs_data = vector_data(size, 11);

        let scies_matrix = DynamicMatrix::new(size, size, matrix_data.clone()).unwrap();
        let na_matrix = DMatrix::from_row_slice(size, size, &matrix_data);
        let na_rhs = DVector::from_row_slice(&rhs_data);

        solve_group.bench_with_input(
            BenchmarkId::new("scies-math", format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(&scies_matrix)
                        .solve_linear_system(black_box(&rhs_data))
                        .unwrap()
                })
            },
        );
        solve_group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(na_matrix.clone())
                        .lu()
                        .solve(black_box(&na_rhs))
                        .unwrap()
                })
            },
        );
    }
    solve_group.finish();
}

fn bench_transform_ops(c: &mut Criterion) {
    let scies_q1 = Quaternion::from_axis_angle([1.0, 2.0, 3.0], 0.73)
        .unwrap()
        .normalize()
        .unwrap();
    let scies_q2 = Quaternion::from_axis_angle([-2.0, 1.0, 0.5], -1.14)
        .unwrap()
        .normalize()
        .unwrap();
    let na_q1 = na::UnitQuaternion::from_axis_angle(
        &na::Unit::new_normalize(na::Vector3::new(1.0, 2.0, 3.0)),
        0.73,
    );
    let na_q2 = na::UnitQuaternion::from_axis_angle(
        &na::Unit::new_normalize(na::Vector3::new(-2.0, 1.0, 0.5)),
        -1.14,
    );

    let mut quat_group = c.benchmark_group("compare/quaternion_compose");
    quat_group.bench_function(BenchmarkId::new("scies-math", "unit"), |b| {
        b.iter(|| black_box(scies_q1).mul(black_box(scies_q2)))
    });
    quat_group.bench_function(BenchmarkId::new("nalgebra", "unit"), |b| {
        b.iter(|| black_box(na_q1) * black_box(na_q2))
    });
    quat_group.finish();

    let scies_rot = Rotation3::from_axis_angle([0.0, 1.0, 0.0], 1.1).unwrap();
    let na_rot = na::UnitQuaternion::from_axis_angle(
        &na::Unit::new_normalize(na::Vector3::new(0.0, 1.0, 0.0)),
        1.1,
    );
    let vec_arr = [1.5, -0.75, 2.25];
    let vec_na = na::Vector3::new(1.5, -0.75, 2.25);

    let mut rotate_group = c.benchmark_group("compare/rotation_transform_vector");
    rotate_group.bench_function(BenchmarkId::new("scies-math", "vec3"), |b| {
        b.iter(|| black_box(scies_rot).rotate(black_box(vec_arr)))
    });
    rotate_group.bench_function(BenchmarkId::new("nalgebra", "vec3"), |b| {
        b.iter(|| black_box(na_rot).transform_vector(&black_box(vec_na)))
    });
    rotate_group.finish();

    let scies_iso = Isometry3::new(scies_rot, [4.0, -2.5, 1.25]);
    let na_iso = na::Isometry3::from_parts(na::Translation3::new(4.0, -2.5, 1.25), na_rot);
    let point_arr = [0.5, 1.0, -3.0];
    let point_na = na::Point3::new(0.5, 1.0, -3.0);

    let mut iso_group = c.benchmark_group("compare/isometry_transform_point");
    iso_group.bench_function(BenchmarkId::new("scies-math", "point3"), |b| {
        b.iter(|| black_box(scies_iso).transform_point(black_box(point_arr)))
    });
    iso_group.bench_function(BenchmarkId::new("nalgebra", "point3"), |b| {
        b.iter(|| black_box(na_iso).transform_point(&black_box(point_na)))
    });
    iso_group.finish();
}

fn make_scies_matrix(n: usize, seed: f64) -> scies_math_th::generic::Mat<f64> {
    let data: Vec<f64> = (0..n * n)
        .map(|i| (i as f64 * seed * 0.001).sin())
        .collect();
    scies_math_th::generic::Mat::new(n, n, data).unwrap()
}

fn make_nalgebra_matrix(n: usize, seed: f64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |r, c| ((r * n + c) as f64 * seed * 0.001).sin())
}

fn bench_backend_showcases(c: &mut Criterion) {
    let mut matmul_group = c.benchmark_group("compare/backend_matmul_512x512");
    let n = 512usize;

    let scies_a = make_scies_matrix(n, 1.7);
    let scies_b = make_scies_matrix(n, 3.1);
    let na_a = make_nalgebra_matrix(n, 1.7);
    let na_b = make_nalgebra_matrix(n, 3.1);

    matmul_group.throughput(Throughput::Elements((n * n * n) as u64));
    matmul_group.bench_function("nalgebra_pure_rust", |bench| {
        bench.iter(|| criterion::black_box(&na_a * &na_b));
    });
    matmul_group.bench_function("scies_math_backend", |bench| {
        let a_data = &scies_a.data;
        let b_data = &scies_b.data;
        bench.iter(|| {
            let mut out = vec![0.0; n * n];
            #[cfg(feature = "parallel")]
            matmul_rayon(a_data, b_data, &mut out, n, n, n);
            #[cfg(not(feature = "parallel"))]
            matmul_threaded(a_data, b_data, &mut out, n, n, n, 8);
            criterion::black_box(out)
        });
    });
    matmul_group.finish();

    let mut lazy_group = c.benchmark_group("compare/elementwise_lazy_2048x2048");
    let n = 2048usize;
    let scies_a = make_scies_matrix(n, 1.1);
    let scies_b = make_scies_matrix(n, 1.2);
    let scies_c = make_scies_matrix(n, 1.3);
    let na_a = make_nalgebra_matrix(n, 1.1);
    let na_b = make_nalgebra_matrix(n, 1.2);
    let na_c = make_nalgebra_matrix(n, 1.3);

    lazy_group.throughput(Throughput::Elements((n * n) as u64));
    lazy_group.bench_function("nalgebra_eager_allocation", |bench| {
        bench.iter(|| {
            let temp = &na_a + &na_b;
            let result = temp.component_mul(&na_c);
            criterion::black_box(result)
        });
    });
    lazy_group.bench_function("scies_math_lazy_eval", |bench| {
        bench.iter(|| {
            let result = (lazy(&scies_a) + lazy(&scies_b))
                .hadamard(lazy(&scies_c))
                .eval()
                .unwrap();
            criterion::black_box(result)
        });
    });
    lazy_group.finish();
}

criterion_group!(
    compare_benches,
    bench_small_vector_and_matrix_ops,
    bench_static_smatrix_ops,
    bench_dynamic_linear_algebra,
    bench_transform_ops,
    bench_backend_showcases
);
criterion_main!(compare_benches);
