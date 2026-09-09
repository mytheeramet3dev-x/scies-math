//! Benchmark: Statistical sampling — distributions + alias table
//!
//! Run with:  cargo bench --bench sampling

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use scies_math_th::rng_ext::{AliasTable, DefaultSampler, FastSampler, Xoshiro256ss};

const N: usize = 100_000;

fn bench_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/distributions");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("uniform f64", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.f64();
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("normal", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.normal(0.0, 1.0);
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("gamma(2.0,1.0)", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.gamma(2.0, 1.0);
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("beta(2.0,5.0)", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.beta(2.0, 5.0);
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("poisson(lambda=5.0)", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..N {
                sum = sum.wrapping_add(s.poisson(5.0));
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("exponential(lambda=1.5)", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.exponential(1.5);
            }
            criterion::black_box(sum)
        });
    });

    group.finish();
}

fn bench_alias_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/alias-table");

    for &k in &[4usize, 16, 64, 256, 1024] {
        let weights: Vec<f64> = (1..=k).map(|i| i as f64).collect();
        let table = AliasTable::new(&weights).unwrap();

        group.throughput(Throughput::Elements(N as u64));
        group.bench_with_input(BenchmarkId::new("sample-O(1)", k), &k, |b, _| {
            let mut rng = Xoshiro256ss::new(42);
            b.iter(|| {
                let mut sum = 0usize;
                for _ in 0..N {
                    sum = sum.wrapping_add(table.sample(&mut rng));
                }
                criterion::black_box(sum)
            });
        });
    }

    group.finish();
}

fn bench_shuffle(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/shuffle");

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("fisher-yates", n), &n, |b, &n| {
            let mut s = DefaultSampler::seeded(42);
            let mut v: Vec<usize> = (0..n).collect();
            b.iter(|| {
                s.shuffle(&mut v);
                criterion::black_box(v[0])
            });
        });
    }
    group.finish();
}

fn bench_fast_vs_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/rng-api-f64");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("DefaultSampler(Xoshiro256**)", |b| {
        let mut s = DefaultSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.f64();
            }
            criterion::black_box(sum)
        });
    });

    group.bench_function("FastSampler(Wyrand)", |b| {
        let mut s = FastSampler::seeded(42);
        b.iter(|| {
            let mut sum = 0.0f64;
            for _ in 0..N {
                sum += s.f64();
            }
            criterion::black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distributions,
    bench_alias_table,
    bench_shuffle,
    bench_fast_vs_quality,
);
criterion_main!(benches);
