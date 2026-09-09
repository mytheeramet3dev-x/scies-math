//! Benchmark: RNG throughput — Xoshiro256** vs PCG64 vs Wyrand vs MT19937
//!
//! Run with:  cargo bench --bench rng

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use scies_math_th::rng_ext::{Mt19937, Pcg64, RngCore, Wyrand, Xoshiro256ss};

const N: usize = 1_000_000;

fn bench_rng_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng/u64-throughput");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("Xoshiro256**", |b| {
        let mut rng = Xoshiro256ss::new(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(rng.next_u64());
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("PCG64", |b| {
        let mut rng = Pcg64::from_seed(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(rng.next_u64());
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("Wyrand", |b| {
        let mut rng = Wyrand::new(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(rng.next_u64());
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("MT19937", |b| {
        let mut rng = Mt19937::new(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(rng.next_u64());
            }
            criterion::black_box(sum)
        });
    });

    group.finish();
}

fn bench_rng_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng/normal-variate");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("Xoshiro256** normal", |b| {
        let mut rng = Xoshiro256ss::new(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += rng.next_normal();
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("Wyrand normal", |b| {
        let mut rng = Wyrand::new(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += rng.next_normal();
            }
            criterion::black_box(sum)
        });
    });

    group.finish();
}

fn bench_rng_fill_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng/fill-bytes");
    let bytes = 1024 * 1024; // 1 MiB
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function("Xoshiro256** fill 1MiB", |b| {
        let mut rng = Xoshiro256ss::new(42);
        let mut buf = vec![0u8; bytes];
        b.iter(|| {
            rng.fill_bytes(&mut buf);
            criterion::black_box(buf[0])
        });
    });

    group.bench_function("Wyrand fill 1MiB", |b| {
        let mut rng = Wyrand::new(42);
        let mut buf = vec![0u8; bytes];
        b.iter(|| {
            rng.fill_bytes(&mut buf);
            criterion::black_box(buf[0])
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rng_throughput,
    bench_rng_normal,
    bench_rng_fill_bytes
);
criterion_main!(benches);
