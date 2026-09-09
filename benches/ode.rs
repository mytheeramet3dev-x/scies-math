//! Benchmark: ODE solvers — RK4 vs RK45 vs DOP853 vs Backward Euler (stiff)
//!
//! Run with:  cargo bench --bench ode

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scies_math_th::ode::{rk4_system, rk45};
use scies_math_th::ode_ext::{adams_bashforth4, backward_euler, dopri8};

// Simple harmonic oscillator: y'' + y = 0  → state [pos, vel]
fn harmonic(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0]]
}

// Van der Pol (stiff for large mu)
fn van_der_pol_stiff(_t: f64, y: &[f64]) -> Vec<f64> {
    let mu = 100.0;
    vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
}

fn bench_ode_explicit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/explicit");

    // RK4 fixed step: rk4_system(f, y0, t0, t_end, dt)
    group.bench_function("RK4/harmonic/n=1000", |b| {
        b.iter(|| {
            criterion::black_box(rk4_system(harmonic, &[1.0, 0.0], 0.0, 1.0, 0.001).unwrap())
        });
    });

    // RK45 adaptive: rk45(f, y0, t0, t_end, rtol, atol, max_steps)
    group.bench_function("RK45/harmonic/tol=1e-6", |b| {
        b.iter(|| {
            criterion::black_box(
                rk45(harmonic, &[1.0, 0.0], 0.0, 10.0, 1e-6, 1e-9, 10_000).unwrap(),
            )
        });
    });

    // DOP853: dopri8(f, y0, t0, t_end, rtol, atol, h0)
    group.bench_function("DOP853/harmonic/tol=1e-8", |b| {
        b.iter(|| {
            criterion::black_box(
                dopri8(harmonic, &[1.0, 0.0], 0.0, 10.0, 1e-8, 1e-10, 0.1).unwrap(),
            )
        });
    });

    // Adams-Bashforth 4: adams_bashforth4(f, y0, t0, t_end, h)
    group.bench_function("AB4/harmonic/n=1000", |b| {
        b.iter(|| {
            criterion::black_box(adams_bashforth4(harmonic, &[1.0, 0.0], 0.0, 1.0, 0.001).unwrap())
        });
    });

    group.finish();
}

fn bench_ode_stiff(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/stiff");

    // backward_euler(f, y0, t0, t_end, h)
    group.bench_function("BackwardEuler/VanDerPol(mu=100)/n=500", |b| {
        b.iter(|| {
            criterion::black_box(
                backward_euler(van_der_pol_stiff, &[2.0, 0.0], 0.0, 5.0, 0.01).unwrap(),
            )
        });
    });

    group.finish();
}

fn bench_ode_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode/system-size");

    for dim in [2usize, 8, 32, 128] {
        let y0: Vec<f64> = vec![1.0f64; dim];
        let t_end = 1.0;
        let dt = 0.01;

        group.bench_with_input(BenchmarkId::new("RK4/n=100", dim), &dim, |b, _| {
            b.iter(|| {
                let f = |_t: f64, y: &[f64]| -> Vec<f64> { y.iter().map(|&yi| -yi).collect() };
                criterion::black_box(rk4_system(f, &y0, 0.0, t_end, dt).unwrap())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ode_explicit,
    bench_ode_stiff,
    bench_ode_sizes
);
criterion_main!(benches);
