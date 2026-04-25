//! Benchmark: ODE solvers — RK4 vs RK45 vs DOP853 vs BDF2 (stiff)
//!
//! Run with:  cargo bench --bench ode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scies_math::ode::{rk4, rk45};
use scies_math::ode_ext::{backward_euler, dopri8, adams_bashforth4};

// Simple harmonic oscillator: y'' + y = 0
// State: [position, velocity]
fn harmonic(t: f64, y: &[f64]) -> Vec<f64> {
    let _ = t;
    vec![y[1], -y[0]]
}

// Van der Pol (stiff for large mu): y'' - mu*(1-y²)*y' + y = 0
fn van_der_pol_stiff(t: f64, y: &[f64]) -> Vec<f64> {
    let mu = 100.0; // stiff
    let _ = t;
    vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
}

fn bench_ode_explicit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/explicit");

    // RK4 fixed step
    group.bench_function("RK4/harmonic/n=1000", |b| {
        b.iter(|| {
            criterion::black_box(rk4(harmonic, 0.0, vec![1.0, 0.0], 0.001, 1000))
        });
    });

    // RK45 adaptive
    group.bench_function("RK45/harmonic/tol=1e-6", |b| {
        b.iter(|| {
            criterion::black_box(
                rk45(harmonic, 0.0, 10.0, vec![1.0, 0.0], 1e-6, 1e-9, 0.1)
            )
        });
    });

    // DOP853 8th order adaptive
    group.bench_function("DOP853/harmonic/tol=1e-8", |b| {
        b.iter(|| {
            criterion::black_box(
                dopri8(harmonic, 0.0, 10.0, vec![1.0, 0.0], 1e-8, 1e-10, 0.1)
            )
        });
    });

    // Adams-Bashforth 4-step
    group.bench_function("AB4/harmonic/n=1000", |b| {
        b.iter(|| {
            criterion::black_box(
                adams_bashforth4(harmonic, 0.0, vec![1.0, 0.0], 0.001, 1000)
            )
        });
    });

    group.finish();
}

fn bench_ode_stiff(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/stiff");

    group.bench_function("BackwardEuler/VanDerPol(mu=100)/n=500", |b| {
        b.iter(|| {
            criterion::black_box(
                backward_euler(van_der_pol_stiff, 0.0, vec![2.0, 0.0], 0.01, 500)
            )
        });
    });

    group.finish();
}

fn bench_ode_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/system-size");

    for dim in [2usize, 8, 32, 128] {
        // Linear system: dy/dt = -y
        let f = move |_t: f64, y: &[f64]| -> Vec<f64> {
            y.iter().map(|&yi| -yi).collect()
        };
        let y0 = vec![1.0f64; dim];

        group.bench_with_input(BenchmarkId::new("RK4/n=100", dim), &dim, |b, _| {
            b.iter(|| {
                criterion::black_box(rk4(f, 0.0, y0.clone(), 0.01, 100))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ode_explicit, bench_ode_stiff, bench_ode_sizes);
criterion_main!(benches);
