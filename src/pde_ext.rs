//! Extended PDE solvers: 2D Heat (ADI), 2D Wave, Burgers, Fisher-KPP, BVP shooting, Chebyshev.
use crate::errors::{SciError, SciResult};

// ─── 2D Heat equation — ADI (Alternating Direction Implicit, Peaceman-Rachford) ──

/// Solve ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) on [0,1]² with Dirichlet BC = 0.
///
/// `u0[j*nx + i]` is the initial condition at grid point (i, j).
/// Returns the solution after `n_steps` time steps.
pub fn heat_2d_adi(
    u0: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    alpha: f64,
    n_steps: usize,
) -> SciResult<Vec<f64>> {
    if u0.len() != nx * ny {
        return Err(SciError::InvalidParameter("u0 length != nx*ny"));
    }
    let rx = alpha * dt / (2.0 * dx * dx);
    let ry = alpha * dt / (2.0 * dy * dy);
    let mut u = u0.to_vec();
    let mut tmp = vec![0.0f64; nx * ny];

    for _ in 0..n_steps {
        // x-sweep: implicit in x, explicit in y
        for j in 1..(ny - 1) {
            let mut rhs: Vec<f64> = (0..nx)
                .map(|i| {
                    if i == 0 || i == nx - 1 {
                        return 0.0;
                    }
                    let uy = u[(j - 1) * nx + i] - 2.0 * u[j * nx + i] + u[(j + 1) * nx + i];
                    u[j * nx + i] + ry * uy
                })
                .collect();
            // Thomas algorithm for tridiagonal (-rx, 1+2rx, -rx)
            thomas_solve_inplace(&mut rhs, rx, 1.0 + 2.0 * rx, rx);
            for i in 0..nx {
                tmp[j * nx + i] = rhs[i];
            }
        }
        // y-sweep: implicit in y, explicit in x
        for i in 1..(nx - 1) {
            let mut rhs: Vec<f64> = (0..ny)
                .map(|j| {
                    if j == 0 || j == ny - 1 {
                        return 0.0;
                    }
                    let ux = tmp[j * nx + i - 1] - 2.0 * tmp[j * nx + i] + tmp[j * nx + i + 1];
                    tmp[j * nx + i] + rx * ux
                })
                .collect();
            thomas_solve_inplace(&mut rhs, ry, 1.0 + 2.0 * ry, ry);
            for j in 0..ny {
                u[j * nx + i] = rhs[j];
            }
        }
    }
    Ok(u)
}

// ─── 2D Wave equation — explicit FTCS ────────────────────────────────────────

/// Solve ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²) with Dirichlet BC = 0.
///
/// Pass `u_prev` (t−dt) and `u_curr` (t=0); returns solution at t = n_steps·dt.
pub fn wave_2d(
    u_prev: &[f64],
    u_curr: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    c: f64,
    n_steps: usize,
) -> SciResult<Vec<f64>> {
    if u_prev.len() != nx * ny || u_curr.len() != nx * ny {
        return Err(SciError::InvalidParameter("grid size mismatch"));
    }
    let rx = (c * dt / dx).powi(2);
    let ry = (c * dt / dy).powi(2);
    let mut up = u_prev.to_vec();
    let mut uc = u_curr.to_vec();
    let mut un = vec![0.0f64; nx * ny];

    for _ in 0..n_steps {
        for j in 1..(ny - 1) {
            for i in 1..(nx - 1) {
                let idx = j * nx + i;
                let lap_x = uc[idx - 1] - 2.0 * uc[idx] + uc[idx + 1];
                let lap_y = uc[(j - 1) * nx + i] - 2.0 * uc[idx] + uc[(j + 1) * nx + i];
                un[idx] = 2.0 * uc[idx] - up[idx] + rx * lap_x + ry * lap_y;
            }
        }
        core::mem::swap(&mut up, &mut uc);
        core::mem::swap(&mut uc, &mut un);
    }
    Ok(uc)
}

// ─── Burgers' equation (1D viscous) ──────────────────────────────────────────

/// Solve ∂u/∂t + u∂u/∂x = ν∂²u/∂x²  on periodic domain via upwind+central FD.
///
/// Returns solution at `t_end` sampled every `save_every` steps.
pub fn burgers_1d(u0: &[f64], dx: f64, dt: f64, nu: f64, t_end: f64) -> SciResult<Vec<f64>> {
    let n = u0.len();
    if n < 3 {
        return Err(SciError::InvalidParameter("need at least 3 points"));
    }
    let n_steps = (t_end / dt).ceil() as usize;
    let mut u = u0.to_vec();
    let mut un = vec![0.0f64; n];

    for _ in 0..n_steps {
        for i in 0..n {
            let im = (i + n - 1) % n;
            let ip = (i + 1) % n;
            // Upwind for convection, central for diffusion
            let conv = if u[i] >= 0.0 {
                u[i] * (u[i] - u[im]) / dx
            } else {
                u[i] * (u[ip] - u[i]) / dx
            };
            let diff = nu * (u[ip] - 2.0 * u[i] + u[im]) / (dx * dx);
            un[i] = u[i] + dt * (-conv + diff);
        }
        core::mem::swap(&mut u, &mut un);
    }
    Ok(u)
}

// ─── Fisher-KPP reaction-diffusion ───────────────────────────────────────────

/// Solve ∂u/∂t = D·∂²u/∂x² + r·u(1−u) on [0, L] with Neumann BC (zero flux).
///
/// Uses explicit Euler. Stability requires dt ≤ dx²/(2D).
pub fn fisher_kpp(
    u0: &[f64],
    dx: f64,
    dt: f64,
    diffusion: f64,
    growth: f64,
    t_end: f64,
) -> SciResult<Vec<f64>> {
    let n = u0.len();
    if n < 3 {
        return Err(SciError::InvalidParameter("need at least 3 points"));
    }
    let n_steps = (t_end / dt).ceil() as usize;
    let r = diffusion * dt / (dx * dx);
    let mut u = u0.to_vec();
    let mut un = vec![0.0f64; n];

    for _ in 0..n_steps {
        for i in 0..n {
            let im = if i == 0 { 0 } else { i - 1 }; // Neumann BC
            let ip = if i == n - 1 { n - 1 } else { i + 1 };
            let lap = u[ip] - 2.0 * u[i] + u[im];
            un[i] = u[i] + r * lap + dt * growth * u[i] * (1.0 - u[i]);
            un[i] = un[i].clamp(0.0, 1.0);
        }
        core::mem::swap(&mut u, &mut un);
    }
    Ok(u)
}

// ─── BVP via shooting method ─────────────────────────────────────────────────

/// Solve the BVP  u'' = f(x, u, u')  with  u(a)=ya, u(b)=yb
/// by single shooting + bisection on the initial slope.
///
/// Returns `(x_grid, u_values)` sampled at `n_points` equally-spaced nodes.
pub fn bvp_shooting<F>(
    f: F,
    a: f64,
    b: f64,
    ya: f64,
    yb: f64,
    n_points: usize,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<(Vec<f64>, Vec<f64>)>
where
    F: Fn(f64, f64, f64) -> f64,
{
    if n_points < 2 {
        return Err(SciError::InvalidParameter("n_points >= 2"));
    }
    let h = (b - a) / (n_points - 1) as f64;

    // Shoot with initial slope `s`, return u(b)
    let shoot = |s: f64| -> f64 {
        let mut x = a;
        let mut u = ya;
        let mut du = s;
        for _ in 0..(n_points - 1) {
            let k1u = du;
            let k1du = f(x, u, du);
            let k2u = du + 0.5 * h * k1du;
            let k2du = f(x + 0.5 * h, u + 0.5 * h * k1u, du + 0.5 * h * k1du);
            let k3u = du + 0.5 * h * k2du;
            let k3du = f(x + 0.5 * h, u + 0.5 * h * k2u, du + 0.5 * h * k2du);
            let k4u = du + h * k3du;
            let k4du = f(x + h, u + h * k3u, du + h * k3du);
            u += h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            du += h / 6.0 * (k1du + 2.0 * k2du + 2.0 * k3du + k4du);
            x += h;
        }
        u - yb
    };

    // Bisection to find the correct slope
    let mut s_lo = -1e3_f64;
    let mut s_hi = 1e3_f64;
    let mut f_lo = shoot(s_lo);
    for _ in 0..max_iter {
        let s_mid = (s_lo + s_hi) / 2.0;
        let f_mid = shoot(s_mid);
        if f_mid.abs() < tolerance {
            s_lo = s_mid;
            break;
        }
        if f_lo * f_mid < 0.0 {
            s_hi = s_mid;
        } else {
            s_lo = s_mid;
            f_lo = f_mid;
        }
    }

    // Re-integrate with found slope
    let s_star = (s_lo + s_hi) / 2.0;
    let mut xs = Vec::with_capacity(n_points);
    let mut us = Vec::with_capacity(n_points);
    let mut x = a;
    let mut u = ya;
    let mut du = s_star;
    xs.push(x);
    us.push(u);
    for _ in 0..(n_points - 1) {
        let k1u = du;
        let k1du = f(x, u, du);
        let k2u = du + 0.5 * h * k1du;
        let k2du = f(x + 0.5 * h, u + 0.5 * h * k1u, du + 0.5 * h * k1du);
        let k3u = du + 0.5 * h * k2du;
        let k3du = f(x + 0.5 * h, u + 0.5 * h * k2u, du + 0.5 * h * k2du);
        let k4u = du + h * k3du;
        let k4du = f(x + h, u + h * k3u, du + h * k3du);
        u += h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        du += h / 6.0 * (k1du + 2.0 * k2du + 2.0 * k3du + k4du);
        x += h;
        xs.push(x);
        us.push(u);
    }
    Ok((xs, us))
}

// ─── Chebyshev spectral collocation for linear BVPs ──────────────────────────

/// Solve the linear BVP  a₂(x)u'' + a₁(x)u' + a₀(x)u = f(x) on [−1, 1]
/// with Dirichlet BCs u(−1) = u_left, u(1) = u_right.
///
/// Uses an N-point Chebyshev collocation grid (Chebyshev-Gauss-Lobatto nodes).
/// Returns `(nodes, solution)` where `nodes` are the Chebyshev collocation points.
pub fn chebyshev_bvp<A2, A1, A0, F>(
    n: usize,
    a2: A2,
    a1: A1,
    a0: A0,
    rhs: F,
    u_left: f64,
    u_right: f64,
) -> SciResult<(Vec<f64>, Vec<f64>)>
where
    A2: Fn(f64) -> f64,
    A1: Fn(f64) -> f64,
    A0: Fn(f64) -> f64,
    F: Fn(f64) -> f64,
{
    if n < 4 {
        return Err(SciError::InvalidParameter("n >= 4 required"));
    }

    // Chebyshev-Gauss-Lobatto nodes on [-1,1]: x_j = cos(jπ/N), j=0..N
    let nn = n - 1; // nn = N (degree)
    let pi = core::f64::consts::PI;
    let x: Vec<f64> = (0..=nn)
        .map(|j| (j as f64 * pi / nn as f64).cos())
        .collect();

    // Chebyshev differentiation matrix D (Fornberg algorithm)
    let d1 = cheb_diff_matrix(&x, nn + 1);
    let d2 = mat_mul_sq(&d1, &d1, nn + 1);

    // Build linear system L·u = rhs at interior nodes
    // Interior indices: 1..nn-1 (indices 1 to nn-1)
    let m = nn - 1; // number of interior points
    let mut a_mat = vec![0.0f64; m * m];
    let mut rhs_vec = vec![0.0f64; m];

    for (ii, gi) in (1..nn).enumerate() {
        let xi = x[gi];
        rhs_vec[ii] = rhs(xi)
            - a2(xi) * (d2[gi * (nn + 1)] * u_right + d2[gi * (nn + 1) + nn] * u_left)
            - a1(xi) * (d1[gi * (nn + 1)] * u_right + d1[gi * (nn + 1) + nn] * u_left);
        for (jj, gj) in (1..nn).enumerate() {
            a_mat[ii * m + jj] = a2(xi) * d2[gi * (nn + 1) + gj]
                + a1(xi) * d1[gi * (nn + 1) + gj]
                + a0(xi) * if gi == gj { 1.0 } else { 0.0 };
        }
    }

    // Solve with Gaussian elimination
    let u_int = gauss_elim(&a_mat, &rhs_vec, m)?;

    // Assemble full solution
    let mut u = vec![0.0f64; nn + 1];
    u[0] = u_right; // x[0] = 1
    u[nn] = u_left; // x[nn] = -1
    for (ii, gi) in (1..nn).enumerate() {
        u[gi] = u_int[ii];
    }

    Ok((x, u))
}

// ─── Method of Lines helper ───────────────────────────────────────────────────

/// Discretise the 1D diffusion-reaction PDE
/// ∂u/∂t = D·∂²u/∂x² + reaction(u)
/// into a vector ODE system du/dt = f(t, u) via central finite differences.
///
/// Boundary conditions: Dirichlet at both ends (u[0] = bc_left, u[n-1] = bc_right).
///
/// Returns a closure suitable for [`crate::ode`] vector solvers.
pub fn mol_diffusion_reaction<R>(
    n: usize,
    dx: f64,
    diffusion: f64,
    reaction: R,
    bc_left: f64,
    bc_right: f64,
) -> impl Fn(f64, &[f64]) -> Vec<f64>
where
    R: Fn(f64, f64) -> f64 + 'static,
{
    let r = diffusion / (dx * dx);
    move |t: f64, u: &[f64]| -> Vec<f64> {
        let mut du = vec![0.0f64; n];
        for i in 0..n {
            let um = if i == 0 { bc_left } else { u[i - 1] };
            let up = if i == n - 1 { bc_right } else { u[i + 1] };
            du[i] = r * (um - 2.0 * u[i] + up) + reaction(t, u[i]);
        }
        du
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Thomas algorithm for tridiagonal system with constant coefficients.
/// Overwrites `rhs` with the solution in-place.
fn thomas_solve_inplace(rhs: &mut Vec<f64>, lo: f64, diag: f64, hi: f64) {
    let n = rhs.len();
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = rhs.clone();
    c_prime[0] = hi / diag;
    d_prime[0] = rhs[0] / diag;
    for i in 1..n {
        let m = diag - lo * c_prime[i - 1];
        c_prime[i] = hi / m;
        d_prime[i] = (rhs[i] - lo * d_prime[i - 1]) / m;
    }
    rhs[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = d_prime[i] - c_prime[i] * rhs[i + 1];
    }
}

/// Chebyshev differentiation matrix for nodes x (length n).
fn cheb_diff_matrix(x: &[f64], n: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n * n];
    let c = |k: usize| -> f64 { if k == 0 || k == n - 1 { 2.0 } else { 1.0 } };
    for i in 0..n {
        for j in 0..n {
            if i != j {
                d[i * n + j] =
                    c(i) / c(j) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 } / (x[i] - x[j]);
            }
        }
        // Diagonal: negative sum of row
        let row_sum: f64 = (0..n).filter(|&k| k != i).map(|k| d[i * n + k]).sum();
        d[i * n + i] = -row_sum;
    }
    d
}

/// Square matrix multiplication C = A·B, size n×n.
fn mat_mul_sq(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Gaussian elimination with partial pivoting for dense m×m system.
fn gauss_elim(a: &[f64], b: &[f64], m: usize) -> SciResult<Vec<f64>> {
    let mut mat: Vec<f64> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();
    for col in 0..m {
        // Partial pivot
        let pivot = (col..m)
            .max_by(|&r1, &r2| mat[r1 * m + col].abs().total_cmp(&mat[r2 * m + col].abs()))
            .unwrap();
        if mat[pivot * m + col].abs() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        mat.swap_within(col * m..(col + 1) * m, pivot * m..(pivot + 1) * m, m);
        rhs.swap(col, pivot);
        let scale = mat[col * m + col];
        for row in (col + 1)..m {
            let f = mat[row * m + col] / scale;
            for j in col..m {
                mat[row * m + j] -= f * mat[col * m + j];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0f64; m];
    for i in (0..m).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..m {
            x[i] -= mat[i * m + j] * x[j];
        }
        x[i] /= mat[i * m + i];
    }
    Ok(x)
}

trait SwapWithin {
    fn swap_within(&mut self, r1: core::ops::Range<usize>, r2: core::ops::Range<usize>, _n: usize);
}
impl SwapWithin for Vec<f64> {
    fn swap_within(&mut self, r1: core::ops::Range<usize>, r2: core::ops::Range<usize>, _n: usize) {
        let (a, b) = (r1.start, r2.start);
        let len = r1.len();
        for k in 0..len {
            self.swap(a + k, b + k);
        }
    }
}
