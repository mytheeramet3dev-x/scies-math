//! Reverse-mode automatic differentiation (backpropagation).
//!
//! Reverse AD builds a computation graph during the forward pass and then
//! propagates gradients backward.  The cost is O(forward) regardless of the
//! number of input variables — ideal when computing ∂L/∂θ for many parameters.
//!
//! # Types
//!
//! - [`Tape`] — records operations during the forward pass
//! - [`Var`] — a tracked scalar variable on the tape
//!
//! # Usage
//!
//! ```ignore
//! use scies_math_th::reverse_ad::{Tape, backward};
//!
//! let tape = Tape::new();
//! let x = tape.var(3.0);
//! let y = tape.var(2.0);
//! let z = x * x + x * y; // z = x² + xy = 9 + 6 = 15
//!
//! let grads = backward(&tape, z);
//! // dz/dx = 2x + y = 8
//! // dz/dy = x = 3
//! ```
//!
//! # Hessian
//!
//! Compute the Hessian matrix H[i,j] = ∂²f/∂xᵢ∂xⱼ by applying forward AD
//! over a reverse-AD gradient:
//!
//! ```ignore
//! use scies_math_th::autodiff::hessian;
//!
//! let h = hessian(|v| v[0]*v[0] + v[1]*v[1], &[1.0, 2.0]);
//! // h ≈ [[2, 0], [0, 2]]
//! ```
use crate::errors::{SciError, SciResult};
use core::cell::RefCell;

// ─────────────────────────────────────────────────────────────────────────────
// Tape
// ─────────────────────────────────────────────────────────────────────────────

struct TapeNode {
    /// (parent_index, local_gradient ∂this/∂parent) pairs.
    parents: Vec<(usize, f64)>,
}

/// Computation graph — created fresh for each forward pass.
pub struct Tape {
    nodes: RefCell<Vec<TapeNode>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
        }
    }

    /// Create a leaf variable (input).
    pub fn var(&self, value: f64) -> Var<'_> {
        let idx = self.push(vec![]);
        Var {
            tape: self,
            index: idx,
            value,
        }
    }

    /// Create a constant (no gradient flows back through it).
    pub fn constant(&self, value: f64) -> Var<'_> {
        // Same as var — caller simply won't query its gradient.
        self.var(value)
    }

    fn push(&self, parents: Vec<(usize, f64)>) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(TapeNode { parents });
        idx
    }

    /// Reverse accumulation from `output`.
    /// Returns `GradMap` — query with `map.of(&var)`.
    pub fn backward(&self, output: &Var<'_>) -> GradMap {
        let nodes = self.nodes.borrow();
        let n = nodes.len();
        let mut g = vec![0.0f64; n];
        g[output.index] = 1.0;
        for i in (0..n).rev() {
            let gi = g[i];
            if gi == 0.0 {
                continue;
            }
            for &(parent, w) in &nodes[i].parents {
                g[parent] += w * gi;
            }
        }
        GradMap { grads: g }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradMap
// ─────────────────────────────────────────────────────────────────────────────

/// Gradient values obtained from a backward pass.
pub struct GradMap {
    grads: Vec<f64>,
}

impl GradMap {
    /// Gradient of the output with respect to `var`.
    pub fn of(&self, var: &Var<'_>) -> f64 {
        self.grads.get(var.index).copied().unwrap_or(0.0)
    }

    /// Raw gradient slice (indexed by tape node index).
    pub fn raw(&self) -> &[f64] {
        &self.grads
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Var
// ─────────────────────────────────────────────────────────────────────────────

/// A variable on the computation tape.  `Copy` — pass by value freely.
#[derive(Copy, Clone)]
pub struct Var<'t> {
    tape: &'t Tape,
    /// Position on the tape (use to index into `GradMap`).
    pub index: usize,
    /// Primal (forward) value.
    pub value: f64,
}

impl<'t> Var<'t> {
    fn node(&self, parents: Vec<(usize, f64)>, value: f64) -> Self {
        let idx = self.tape.push(parents);
        Self {
            tape: self.tape,
            index: idx,
            value,
        }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    pub fn add(self, rhs: Var<'t>) -> Var<'t> {
        self.node(
            vec![(self.index, 1.0), (rhs.index, 1.0)],
            self.value + rhs.value,
        )
    }
    pub fn sub(self, rhs: Var<'t>) -> Var<'t> {
        self.node(
            vec![(self.index, 1.0), (rhs.index, -1.0)],
            self.value - rhs.value,
        )
    }
    pub fn mul(self, rhs: Var<'t>) -> Var<'t> {
        self.node(
            vec![(self.index, rhs.value), (rhs.index, self.value)],
            self.value * rhs.value,
        )
    }
    pub fn div(self, rhs: Var<'t>) -> SciResult<Var<'t>> {
        if rhs.value.abs() < f64::EPSILON {
            return Err(SciError::DivisionByZero);
        }
        let v = self.value / rhs.value;
        Ok(self.node(
            vec![
                (self.index, 1.0 / rhs.value),
                (rhs.index, -self.value / (rhs.value * rhs.value)),
            ],
            v,
        ))
    }
    pub fn neg(self) -> Var<'t> {
        self.node(vec![(self.index, -1.0)], -self.value)
    }
    /// Multiply by a scalar constant.
    pub fn scale(self, c: f64) -> Var<'t> {
        self.node(vec![(self.index, c)], self.value * c)
    }
    /// Add a scalar constant.
    pub fn add_const(self, c: f64) -> Var<'t> {
        self.node(vec![(self.index, 1.0)], self.value + c)
    }

    // ── Math functions ────────────────────────────────────────────────────────

    pub fn sin(self) -> Var<'t> {
        self.node(vec![(self.index, self.value.cos())], self.value.sin())
    }
    pub fn cos(self) -> Var<'t> {
        self.node(vec![(self.index, -self.value.sin())], self.value.cos())
    }
    pub fn tan(self) -> SciResult<Var<'t>> {
        let c = self.value.cos();
        if c.abs() < f64::EPSILON {
            return Err(SciError::DomainError("tan undefined at this point"));
        }
        Ok(self.node(vec![(self.index, 1.0 / (c * c))], self.value.tan()))
    }
    pub fn exp(self) -> Var<'t> {
        let e = self.value.exp();
        self.node(vec![(self.index, e)], e)
    }
    pub fn ln(self) -> SciResult<Var<'t>> {
        if self.value <= 0.0 {
            return Err(SciError::DomainError("ln requires positive input"));
        }
        Ok(self.node(vec![(self.index, 1.0 / self.value)], self.value.ln()))
    }
    pub fn sqrt(self) -> SciResult<Var<'t>> {
        if self.value < 0.0 {
            return Err(SciError::DomainError("sqrt requires non-negative input"));
        }
        let s = self.value.sqrt();
        let g = if s > f64::EPSILON { 0.5 / s } else { 0.0 };
        Ok(self.node(vec![(self.index, g)], s))
    }
    pub fn powi(self, n: i32) -> Var<'t> {
        self.node(
            vec![(self.index, n as f64 * self.value.powi(n - 1))],
            self.value.powi(n),
        )
    }
    pub fn powf(self, n: f64) -> Var<'t> {
        self.node(
            vec![(self.index, n * self.value.powf(n - 1.0))],
            self.value.powf(n),
        )
    }
    pub fn tanh(self) -> Var<'t> {
        let t = self.value.tanh();
        self.node(vec![(self.index, 1.0 - t * t)], t)
    }
    /// Logistic sigmoid σ(x) = 1/(1+e^−x).
    pub fn sigmoid(self) -> Var<'t> {
        let s = 1.0 / (1.0 + (-self.value).exp());
        self.node(vec![(self.index, s * (1.0 - s))], s)
    }
    /// ReLU — sub-gradient 0 at x = 0.
    pub fn relu(self) -> Var<'t> {
        let g = if self.value > 0.0 { 1.0 } else { 0.0 };
        self.node(vec![(self.index, g)], self.value.max(0.0))
    }
    pub fn abs(self) -> Var<'t> {
        let g = if self.value >= 0.0 { 1.0 } else { -1.0 };
        self.node(vec![(self.index, g)], self.value.abs())
    }
    pub fn sinh(self) -> Var<'t> {
        self.node(vec![(self.index, self.value.cosh())], self.value.sinh())
    }
    pub fn cosh(self) -> Var<'t> {
        self.node(vec![(self.index, self.value.sinh())], self.value.cosh())
    }
    pub fn atan(self) -> Var<'t> {
        self.node(
            vec![(self.index, 1.0 / (1.0 + self.value * self.value))],
            self.value.atan(),
        )
    }
    pub fn asin(self) -> SciResult<Var<'t>> {
        if self.value.abs() >= 1.0 {
            return Err(SciError::DomainError("asin domain: |x| < 1"));
        }
        let g = 1.0 / (1.0 - self.value * self.value).sqrt();
        Ok(self.node(vec![(self.index, g)], self.value.asin()))
    }
    pub fn acos(self) -> SciResult<Var<'t>> {
        if self.value.abs() >= 1.0 {
            return Err(SciError::DomainError("acos domain: |x| < 1"));
        }
        let g = -1.0 / (1.0 - self.value * self.value).sqrt();
        Ok(self.node(vec![(self.index, g)], self.value.acos()))
    }
}

// ── Operator overloading ──────────────────────────────────────────────────────

impl<'t> core::ops::Add for Var<'t> {
    type Output = Var<'t>;
    fn add(self, rhs: Var<'t>) -> Var<'t> {
        Var::add(self, rhs)
    }
}
impl<'t> core::ops::Sub for Var<'t> {
    type Output = Var<'t>;
    fn sub(self, rhs: Var<'t>) -> Var<'t> {
        Var::sub(self, rhs)
    }
}
impl<'t> core::ops::Mul for Var<'t> {
    type Output = Var<'t>;
    fn mul(self, rhs: Var<'t>) -> Var<'t> {
        Var::mul(self, rhs)
    }
}
impl<'t> core::ops::Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Var<'t> {
        Var::neg(self)
    }
}
impl<'t> core::ops::Add<f64> for Var<'t> {
    type Output = Var<'t>;
    fn add(self, c: f64) -> Var<'t> {
        self.add_const(c)
    }
}
impl<'t> core::ops::Mul<f64> for Var<'t> {
    type Output = Var<'t>;
    fn mul(self, c: f64) -> Var<'t> {
        self.scale(c)
    }
}
impl<'t> core::ops::Sub<f64> for Var<'t> {
    type Output = Var<'t>;
    fn sub(self, c: f64) -> Var<'t> {
        self.add_const(-c)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience entry-points
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar derivative `df/dx` at `x` — single input, single output.
///
/// Uses one forward + one backward pass.
pub fn grad<F>(f: F, x: f64) -> f64
where
    F: for<'t> Fn(Var<'t>) -> Var<'t>,
{
    let tape = Tape::new();
    let xv = tape.var(x);
    let y = f(xv);
    tape.backward(&y).of(&xv)
}

/// Gradient vector `∇f` at `inputs` — multiple inputs, scalar output.
///
/// One forward pass + one backward pass.
pub fn gradient<F>(f: F, inputs: &[f64]) -> Vec<f64>
where
    F: for<'t> Fn(&[Var<'t>]) -> Var<'t>,
{
    let tape = Tape::new();
    let vars: Vec<Var<'_>> = inputs.iter().map(|&x| tape.var(x)).collect();
    let y = f(&vars);
    let gmap = tape.backward(&y);
    vars.iter().map(|v| gmap.of(v)).collect()
}

/// Jacobian matrix `J[i][j] = ∂f_i/∂x_j` — one backward pass per output.
///
/// Efficient when `n_outputs` << `n_inputs`.
pub fn jacobian<F>(f: F, inputs: &[f64], n_outputs: usize) -> SciResult<Vec<Vec<f64>>>
where
    F: for<'t> Fn(&[Var<'t>]) -> Vec<Var<'t>>,
{
    if inputs.is_empty() || n_outputs == 0 {
        return Err(SciError::InvalidParameter(
            "inputs and n_outputs must be non-empty",
        ));
    }
    let mut jac = vec![vec![0.0f64; inputs.len()]; n_outputs];
    for i in 0..n_outputs {
        let tape = Tape::new();
        let vars: Vec<Var<'_>> = inputs.iter().map(|&x| tape.var(x)).collect();
        let ys = f(&vars);
        if i >= ys.len() {
            return Err(SciError::InvalidParameter(
                "f returned fewer outputs than n_outputs",
            ));
        }
        let gmap = tape.backward(&ys[i]);
        for (j, v) in vars.iter().enumerate() {
            jac[i][j] = gmap.of(v);
        }
    }
    Ok(jac)
}

/// Hessian matrix `H[i][j] = ∂²f/∂x_i∂x_j` via finite differences of reverse gradients.
pub fn hessian<F>(f: F, inputs: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64 + Copy,
{
    let n = inputs.len();
    let grad_at = |x: &[f64]| -> Vec<f64> {
        gradient(
            |vars: &[Var<'_>]| {
                let vals: Vec<f64> = vars.iter().map(|v| v.value).collect();
                let tape = vars[0].tape;
                tape.var(f(&vals))
            },
            x,
        )
    };
    let mut result = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = inputs.to_vec();
        let mut xm = inputs.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        for j in 0..n {
            result[i][j] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    result
}
