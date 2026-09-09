//! Benchmark: Quaternion and transform operations
//!
//! Run with:  cargo bench --bench transform

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use scies_math_th::generic::SMatrix;
use scies_math_th::transform::{Isometry3, Quaternion, Rotation3};

const N: usize = 100_000;

fn bench_quaternion(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform/quaternion");
    group.throughput(Throughput::Elements(N as u64));

    let q1 = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 0.5).unwrap();
    let q2 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 1.2).unwrap();
    let v = [1.0f64, 2.0, 3.0];

    group.bench_function("mul x100k", |b| {
        b.iter(|| {
            let mut q = q1;
            for _ in 0..N {
                q = criterion::black_box(q.mul(q2));
            }
            q
        });
    });

    group.bench_function("rotate_vec x100k", |b| {
        b.iter(|| {
            let mut r = v;
            for _ in 0..N {
                r = criterion::black_box(q1.rotate_vec(r));
            }
            r
        });
    });

    group.bench_function("slerp x100k", |b| {
        b.iter(|| {
            let mut t = 0.0f64;
            for _ in 0..N {
                t += 0.000_01;
                criterion::black_box(q1.slerp(q2, t % 1.0));
            }
            t
        });
    });

    group.bench_function("to_rotation_matrix x10k", |b| {
        b.iter(|| {
            let mut q = q1;
            for _ in 0..(N / 10) {
                q = q.mul(q2).normalize().unwrap();
                criterion::black_box(q.to_rotation_matrix());
            }
        });
    });

    group.finish();
}

fn bench_isometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform/isometry");
    group.throughput(Throughput::Elements(N as u64));

    let rot = Rotation3::from_axis_angle([0.0, 1.0, 0.0], 0.5).unwrap();
    let iso = Isometry3::new(rot, [1.0, 2.0, 3.0]);
    let p = [4.0f64, 5.0, 6.0];

    group.bench_function("transform_point x100k", |b| {
        b.iter(|| {
            let mut pt = p;
            for _ in 0..N {
                pt = criterion::black_box(iso.transform_point(pt));
            }
            pt
        });
    });

    group.bench_function("compose x100k", |b| {
        b.iter(|| {
            let mut cur = iso;
            for _ in 0..N {
                cur = criterion::black_box(cur.compose(&iso));
            }
            cur
        });
    });

    group.bench_function("to_matrix4 x10k", |b| {
        b.iter(|| {
            for _ in 0..(N / 10) {
                criterion::black_box(iso.to_matrix4());
            }
        });
    });

    group.finish();
}

fn bench_smatrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform/smatrix");
    group.throughput(Throughput::Elements(N as u64));

    let a: SMatrix<f64, 4, 4> = SMatrix::from_fn(|r, c| (r * 4 + c) as f64 * 0.1 + 1.0);
    let b: SMatrix<f64, 4, 4> = SMatrix::from_fn(|r, c| (r + c) as f64 * 0.05);

    group.bench_function("SMatrix<f64,4,4>::matmul x100k", |b_bench| {
        b_bench.iter(|| {
            let mut m = a;
            for _ in 0..N {
                m = criterion::black_box(m.matmul(&b));
            }
            m
        });
    });

    group.bench_function("SMatrix<f64,4,4>::add x100k", |b_bench| {
        b_bench.iter(|| {
            let mut m = a;
            for _ in 0..N {
                m = criterion::black_box(m + b);
            }
            m
        });
    });

    group.finish();
}

criterion_group!(benches, bench_quaternion, bench_isometry, bench_smatrix);
criterion_main!(benches);
