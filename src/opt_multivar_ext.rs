//! Extended multivariate optimisation: trust-region, adaptive gradient methods,
//! constrained optimisation, and global population-based algorithms.
//!
//! | Algorithm | Type | Best for |
//! |---|---|---|
//! | `trust_region_cg` | 2nd-order unconstrained | General smooth f |
//! | `adam` | 1st-order adaptive | Large-scale / ML |
//! | `adagrad` | 1st-order adaptive | Sparse gradients |
//! | `rmsprop` | 1st-order adaptive | Non-stationary |
//! | `projected_gradient` | 1st-order box-constrained | Simple bound constraints |
//! | `augmented_lagrangian` | Constrained (eq+ineq) | General constrained |
//! | `particle_swarm` | Global / derivative-free | Multimodal global |
//! | `differential_evolution` | Global / derivative-free | Continuous global |

use crate::errors::{SciError, SciResult};

// ─── numerical gradient helper ────────────────────────────────────────────────
fn fd_grad<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let mut xp = x.to_vec();
            xp[i] += h;
            let mut xm = x.to_vec();
            xm[i] -= h;
            (f(&xp) - f(&xm)) / (2.0 * h)
        })
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
fn axpy(a: f64, x: &[f64], y: &mut Vec<f64>) {
    y.iter_mut().zip(x).for_each(|(yi, xi)| *yi += a * xi);
}

// ══════════════════════════════════════════════════════════════════════════════
// Trust-region with Steihaug-CG subproblem
// ══════════════════════════════════════════════════════════════════════════════

/// **Trust-region method** with Steihaug conjugate-gradient subproblem solver.
///
/// Handles indefinite Hessians safely (hits boundary on negative curvature).
/// Uses finite-difference gradient and Hessian-vector products.
pub fn trust_region_cg<F>(f: F, x0: &[f64], tolerance: f64, max_iter: usize) -> SciResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let h = 1e-5;
    let mut delta = 1.0_f64; // trust radius
    let eta = 0.1_f64; // acceptance threshold

    for _ in 0..max_iter {
        let g = fd_grad(&f, &x, h);
        if norm(&g) < tolerance {
            return Ok(x);
        }

        // Hessian-vector product B·v via finite diff: B·v ≈ (∇f(x+hv) - ∇f(x-hv))/(2h)
        let bv = |v: &[f64]| -> Vec<f64> {
            let mut xp = x.clone();
            axpy(h, v, &mut xp);
            let mut xm = x.clone();
            let hv_neg: Vec<f64> = v.iter().map(|vi| -vi).collect();
            axpy(h, &hv_neg, &mut xm);
            let gp = fd_grad(&f, &xp, h);
            let gm = fd_grad(&f, &xm, h);
            gp.iter()
                .zip(&gm)
                .map(|(gpi, gmi)| (gpi - gmi) / (2.0 * h))
                .collect()
        };

        // Steihaug-CG subproblem: minimise m(p) = g·p + ½p·B·p  s.t. ‖p‖ ≤ δ
        let p = steihaug_cg(&g, &bv, delta, n, 50);

        let fx = f(&x);
        let xnew: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + pi).collect();
        let fxnew = f(&xnew);

        // Predicted reduction from quadratic model
        let bp = bv(&p);
        let model_dec = -dot(&g, &p) - 0.5 * dot(&p, &bp);
        let actual_dec = fx - fxnew;
        let rho = if model_dec.abs() < f64::EPSILON {
            1.0
        } else {
            actual_dec / model_dec
        };

        if rho > eta {
            x = xnew;
        }
        // Update trust radius
        delta = if rho < 0.25 {
            delta * 0.25
        } else if rho > 0.75 && norm(&p) > 0.9 * delta {
            (2.0 * delta).min(100.0)
        } else {
            delta
        };
    }
    Err(SciError::NonConvergent("trust_region_cg"))
}

fn steihaug_cg<B: Fn(&[f64]) -> Vec<f64>>(
    g: &[f64],
    bv: &B,
    delta: f64,
    n: usize,
    max_cg: usize,
) -> Vec<f64> {
    let mut p = vec![0.0f64; n];
    let mut r: Vec<f64> = g.iter().map(|gi| -gi).collect();
    let mut d = r.clone();
    let mut rr = dot(&r, &r);
    if rr.sqrt() < 1e-12 {
        return p;
    }
    for _ in 0..max_cg {
        let bd = bv(&d);
        let dbd = dot(&d, &bd);
        if dbd <= 0.0 {
            // Negative curvature: go to boundary
            let a = dot(&d, &d);
            let b2 = 2.0 * dot(&p, &d);
            let c = dot(&p, &p) - delta * delta;
            let disc = (b2 * b2 - 4.0 * a * c).sqrt();
            let tau = (-b2 + disc) / (2.0 * a);
            return p.iter().zip(&d).map(|(pi, di)| pi + tau * di).collect();
        }
        let alpha = rr / dbd;
        let p_new: Vec<f64> = p.iter().zip(&d).map(|(pi, di)| pi + alpha * di).collect();
        if norm(&p_new) >= delta {
            let a = dot(&d, &d);
            let b2 = 2.0 * dot(&p, &d);
            let c = dot(&p, &p) - delta * delta;
            let disc = (b2 * b2 - 4.0 * a * c).sqrt();
            let tau = (-b2 + disc) / (2.0 * a);
            return p.iter().zip(&d).map(|(pi, di)| pi + tau * di).collect();
        }
        p = p_new;
        r = r
            .iter()
            .zip(&bd)
            .map(|(ri, bdi)| ri - alpha * bdi)
            .collect();
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() < 1e-10 {
            break;
        }
        let beta = rr_new / rr;
        d = r.iter().zip(&d).map(|(ri, di)| ri + beta * di).collect();
        rr = rr_new;
    }
    p
}

// ══════════════════════════════════════════════════════════════════════════════
// Adaptive gradient methods (for ML-scale problems)
// ══════════════════════════════════════════════════════════════════════════════

/// **Adam** optimiser (Kingma & Ba 2014): adaptive moment estimation.
///
/// `grad_fn` computes the gradient ∇f(x) (may be stochastic).
pub fn adam<G>(
    grad_fn: G,
    x0: &[f64],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<f64>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0f64; n]; // 1st moment
    let mut v = vec![0.0f64; n]; // 2nd moment

    for t in 1..=max_iter {
        let g = grad_fn(&x);
        let g_norm = norm(&g);
        if g_norm < tolerance {
            return Ok(x);
        }
        let bt1 = beta1.powi(t as i32);
        let bt2 = beta2.powi(t as i32);
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - bt1);
            let v_hat = v[i] / (1.0 - bt2);
            x[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    Err(SciError::NonConvergent("adam"))
}

/// **AdaGrad** (Duchi et al.): accumulate squared gradients per parameter.
pub fn adagrad<G>(
    grad_fn: G,
    x0: &[f64],
    lr: f64,
    eps: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<f64>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g_sum_sq = vec![0.0f64; n];
    for _ in 0..max_iter {
        let g = grad_fn(&x);
        if norm(&g) < tolerance {
            return Ok(x);
        }
        for i in 0..n {
            g_sum_sq[i] += g[i] * g[i];
            x[i] -= lr * g[i] / (g_sum_sq[i].sqrt() + eps);
        }
    }
    Err(SciError::NonConvergent("adagrad"))
}

/// **RMSProp**: exponential moving average of squared gradients.
pub fn rmsprop<G>(
    grad_fn: G,
    x0: &[f64],
    lr: f64,
    rho: f64,
    eps: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<f64>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut eg2 = vec![0.0f64; n];
    for _ in 0..max_iter {
        let g = grad_fn(&x);
        if norm(&g) < tolerance {
            return Ok(x);
        }
        for i in 0..n {
            eg2[i] = rho * eg2[i] + (1.0 - rho) * g[i] * g[i];
            x[i] -= lr * g[i] / (eg2[i].sqrt() + eps);
        }
    }
    Err(SciError::NonConvergent("rmsprop"))
}

// ══════════════════════════════════════════════════════════════════════════════
// Box-constrained optimisation
// ══════════════════════════════════════════════════════════════════════════════

/// **Projected gradient descent** with box constraints `lb[i] <= x[i] <= ub[i]`.
///
/// Uses Armijo line search with projection onto the feasible box.
pub fn projected_gradient<F>(
    f: F,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if lb.len() != n || ub.len() != n {
        return Err(SciError::InvalidParameter("lb/ub length mismatch"));
    }
    let project = |x: &[f64]| -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| xi.clamp(lb[i], ub[i]))
            .collect()
    };
    let mut x = project(x0);
    let h = 1e-5;
    let mut lr = 1.0_f64;

    for _ in 0..max_iter {
        let g = fd_grad(&f, &x, h);
        // Projected gradient norm as convergence measure
        let pg: Vec<f64> = x
            .iter()
            .zip(&g)
            .enumerate()
            .map(|(i, (&xi, &gi))| {
                let xp = (xi - gi).clamp(lb[i], ub[i]);
                xp - xi
            })
            .collect();
        if norm(&pg) < tolerance {
            return Ok(x);
        }

        // Armijo backtracking in projected direction
        let fx = f(&x);
        let mut step = lr;
        loop {
            let xnew = project(
                &x.iter()
                    .zip(&g)
                    .map(|(&xi, &gi)| xi - step * gi)
                    .collect::<Vec<_>>(),
            );
            if f(&xnew) <= fx - 0.01 * dot(&g, &pg) {
                x = xnew;
                lr = step.min(2.0 * lr);
                break;
            }
            step *= 0.5;
            if step < 1e-16 {
                x = xnew;
                break;
            }
        }
    }
    Err(SciError::NonConvergent("projected_gradient"))
}

// ══════════════════════════════════════════════════════════════════════════════
// Augmented Lagrangian (equality + inequality constraints)
// ══════════════════════════════════════════════════════════════════════════════

/// **Augmented Lagrangian method** (Rockafellar 1973) for constrained problems:
///
/// $\min f(x) \quad \text{s.t.} \quad c_{\text{eq}, i}(x) = 0, \quad c_{\text{ineq}, j}(x) \le 0$
///
/// Uses L-BFGS (via `lbfgs` from opt_multivar) on the augmented Lagrangian subproblems.
pub fn augmented_lagrangian<F, CE, CI>(
    f: F,
    ceq: &[CE],
    cineq: &[CI],
    x0: &[f64],
    tolerance: f64,
    max_iter: usize,
) -> SciResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64 + Copy,
    CE: Fn(&[f64]) -> f64 + Copy,
    CI: Fn(&[f64]) -> f64 + Copy,
{
    let n_eq = ceq.len();
    let n_ineq = cineq.len();
    let mut x = x0.to_vec();
    let mut lam_eq = vec![0.0f64; n_eq]; // equality multipliers
    let mut lam_ineq = vec![0.0f64; n_ineq]; // inequality multipliers
    let mut mu = 1.0_f64; // penalty parameter

    for outer in 0..max_iter {
        // Augmented Lagrangian subproblem — minimise AL(x) using gradient descent
        let al = |xv: &[f64]| -> f64 {
            let mut val = f(xv);
            for (i, ce) in ceq.iter().enumerate() {
                let h = ce(xv);
                val += lam_eq[i] * h + 0.5 * mu * h * h;
            }
            for (j, ci) in cineq.iter().enumerate() {
                let g = ci(xv);
                let s = (lam_ineq[j] / mu + g).max(0.0); // slack
                val += -lam_ineq[j] * g + 0.5 * mu * s * s;
            }
            val
        };

        // Inner gradient descent on AL
        let h = 1e-5;
        let lr = 1.0 / mu;
        for _ in 0..200 {
            let g = fd_grad(&al, &x, h);
            if norm(&g) < tolerance * 0.1 {
                break;
            }
            axpy(-lr.min(0.1), &g, &mut x);
        }

        // Update multipliers
        let eq_viol: Vec<f64> = ceq.iter().map(|ce| ce(&x)).collect();
        let ineq_viol: Vec<f64> = cineq.iter().map(|ci| ci(&x)).collect();
        for i in 0..n_eq {
            lam_eq[i] += mu * eq_viol[i];
        }
        for j in 0..n_ineq {
            lam_ineq[j] = (lam_ineq[j] + mu * ineq_viol[j]).max(0.0);
        }

        // Convergence check
        let eq_norm = norm(&eq_viol);
        let ineq_norm = ineq_viol.iter().map(|&v| v.max(0.0)).fold(0.0f64, f64::max);
        if eq_norm < tolerance && ineq_norm < tolerance {
            return Ok(x);
        }
        mu = (mu * 2.0).min(1e6); // increase penalty
        if outer % 10 == 9 {
            mu = (mu * 0.5).max(1.0);
        } // dampen oscillation
    }
    Err(SciError::NonConvergent("augmented_lagrangian"))
}

// ══════════════════════════════════════════════════════════════════════════════
// Particle Swarm Optimisation
// ══════════════════════════════════════════════════════════════════════════════

/// **Particle Swarm Optimisation** (Kennedy & Eberhart 1995).
///
/// Global derivative-free method for continuous multimodal problems.
///
/// # Parameters
/// - `lb`, `ub` — search bounds per dimension
/// - `n_particles` — swarm size (typical: 20–50)
/// - `w` — inertia weight (0.7 is common)
/// - `c1`, `c2` — cognitive / social acceleration (typical: 1.5 each)
pub fn particle_swarm<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    n_particles: usize,
    w: f64,
    c1: f64,
    c2: f64,
    tolerance: f64,
    max_iter: usize,
    seed: u64,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    let n = lb.len();
    if ub.len() != n {
        return Err(SciError::InvalidParameter("lb/ub length mismatch"));
    }
    if n_particles < 2 {
        return Err(SciError::InvalidParameter("need ≥ 2 particles"));
    }

    let mut rng = Lcg64(seed.wrapping_add(1));
    let rand = |r: &mut Lcg64| -> f64 { r.next_f64() };

    // Initialise positions and velocities
    let mut pos: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..n)
                .map(|i| lb[i] + rand(&mut rng) * (ub[i] - lb[i]))
                .collect()
        })
        .collect();
    let mut vel: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..n)
                .map(|i| (rand(&mut rng) - 0.5) * (ub[i] - lb[i]) * 0.1)
                .collect()
        })
        .collect();

    let mut pbest: Vec<Vec<f64>> = pos.clone();
    let mut pbest_f: Vec<f64> = pbest.iter().map(|p| f(p)).collect();
    let (gbest_idx, _) = pbest_f
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let mut gbest = pbest[gbest_idx].clone();
    let mut gbest_f = pbest_f[gbest_idx];

    for _ in 0..max_iter {
        for i in 0..n_particles {
            for d in 0..n {
                let r1 = rand(&mut rng);
                let r2 = rand(&mut rng);
                vel[i][d] = w * vel[i][d]
                    + c1 * r1 * (pbest[i][d] - pos[i][d])
                    + c2 * r2 * (gbest[d] - pos[i][d]);
            }
            pos[i] = pos[i]
                .iter()
                .zip(&vel[i])
                .enumerate()
                .map(|(d, (&xi, &vi))| (xi + vi).clamp(lb[d], ub[d]))
                .collect();
            let fi = f(&pos[i]);
            if fi < pbest_f[i] {
                pbest_f[i] = fi;
                pbest[i] = pos[i].clone();
                if fi < gbest_f {
                    gbest_f = fi;
                    gbest = pos[i].clone();
                }
            }
        }
        if gbest_f < tolerance {
            break;
        }
    }
    Ok((gbest, gbest_f))
}

// ══════════════════════════════════════════════════════════════════════════════
// Differential Evolution
// ══════════════════════════════════════════════════════════════════════════════

/// **Differential Evolution** (Storn & Price 1997).
///
/// Robust global optimizer; works well on non-differentiable objectives.
///
/// Uses DE/rand/1/bin strategy.
///
/// # Parameters
/// - `pop_size` — population size (≥ 4; typical: 10×n_dim)
/// - `f_scale` — differential weight F ∈ [0, 2] (typical: 0.8)
/// - `cr` — crossover probability ∈ [0, 1] (typical: 0.9)
pub fn differential_evolution<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    pop_size: usize,
    f_scale: f64,
    cr: f64,
    tolerance: f64,
    max_iter: usize,
    seed: u64,
) -> SciResult<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    let n = lb.len();
    if ub.len() != n {
        return Err(SciError::InvalidParameter("lb/ub length mismatch"));
    }
    if pop_size < 4 {
        return Err(SciError::InvalidParameter("pop_size >= 4"));
    }

    let mut rng = Lcg64(seed.wrapping_add(42));
    let rand = |r: &mut Lcg64| -> f64 { r.next_f64() };
    let rand_int = |r: &mut Lcg64, hi: usize| -> usize { (r.next_f64() * hi as f64) as usize % hi };

    let mut pop: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            (0..n)
                .map(|i| lb[i] + rand(&mut rng) * (ub[i] - lb[i]))
                .collect()
        })
        .collect();
    let mut fit: Vec<f64> = pop.iter().map(|x| f(x)).collect();

    for _ in 0..max_iter {
        for i in 0..pop_size {
            // Pick 3 distinct indices ≠ i
            let (mut a, mut b, mut c) = (i, i, i);
            while a == i {
                a = rand_int(&mut rng, pop_size);
            }
            while b == i || b == a {
                b = rand_int(&mut rng, pop_size);
            }
            while c == i || c == a || c == b {
                c = rand_int(&mut rng, pop_size);
            }

            let j_rand = rand_int(&mut rng, n);
            let trial: Vec<f64> = (0..n)
                .map(|j| {
                    if j == j_rand || rand(&mut rng) < cr {
                        (pop[a][j] + f_scale * (pop[b][j] - pop[c][j])).clamp(lb[j], ub[j])
                    } else {
                        pop[i][j]
                    }
                })
                .collect();

            let ft = f(&trial);
            if ft <= fit[i] {
                pop[i] = trial;
                fit[i] = ft;
            }
        }
        let best_f = fit.iter().cloned().fold(f64::INFINITY, f64::min);
        if best_f < tolerance {
            break;
        }
    }
    let (best_idx, _) = fit
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    Ok((pop[best_idx].clone(), fit[best_idx]))
}

// ─── Minimal seeded LCG RNG ──────────────────────────────────────────────────
struct Lcg64(u64);
impl Lcg64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }
}
