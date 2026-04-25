//! Neural Network Primitives
//!
//! Provides structures to build, train, and evaluate Feed-Forward Neural Networks.
//! This module uses high-performance matrix operations from `linear_algebra` 
//! and implements Backpropagation with the Adam optimizer.
//!
//! # Features
//! - **Layers**: Dense (Fully Connected)
//! - **Activations**: Linear, ReLU, Sigmoid, Tanh
//! - **Loss Functions**: Mean Squared Error (MSE)
//! - **Optimizers**: Adam, SGD

use crate::linear_algebra::DynamicMatrix;
use crate::errors::{SciError, SciResult};

// ══════════════════════════════════════════════════════════════════════════════
// Activations
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Self::Linear => x,
            Self::ReLU => if x > 0.0 { x } else { 0.0 },
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            },
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Layers
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: DynamicMatrix, // (input_dim, output_dim)
    pub biases: Vec<f64>,       // length = output_dim
    pub activation: Activation,
    
    // Cached for backward pass
    inputs: Option<DynamicMatrix>,
    z: Option<DynamicMatrix>,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> SciResult<Self> {
        // He Initialization for random weights
        let mut rng_state = 42u64; // Simple deterministic LCG
        let mut rand_norm = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state >> 32) as f64 + 0.5) / 4294967296.0;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng_state >> 32) as f64 + 0.5) / 4294967296.0;
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        };

        let std_dev = (2.0 / input_dim as f64).sqrt();
        
        let mut weights_data = Vec::with_capacity(input_dim * output_dim);
        for _ in 0..(input_dim * output_dim) {
            weights_data.push(rand_norm() * std_dev);
        }
        
        let weights = DynamicMatrix::new(input_dim, output_dim, weights_data)?;
        let biases = vec![0.0; output_dim];

        Ok(Self {
            weights,
            biases,
            activation,
            inputs: None,
            z: None,
        })
    }
    
    pub fn forward(&mut self, inputs: &DynamicMatrix) -> SciResult<DynamicMatrix> {
        self.inputs = Some(inputs.clone());
        
        // Z = X * W + b
        let mut z = inputs.mul_matrix(&self.weights)?;
        for r in 0..z.rows() {
            for c in 0..z.cols() {
                let val = z.get(r, c)? + self.biases[c];
                z.set(r, c, val)?;
            }
        }
        self.z = Some(z.clone());
        
        // A = activation(Z)
        let mut a = z;
        for r in 0..a.rows() {
            for c in 0..a.cols() {
                let val = self.activation.forward(a.get(r, c)?);
                a.set(r, c, val)?;
            }
        }
        Ok(a)
    }
    
    pub fn backward(&mut self, d_a: &DynamicMatrix) -> SciResult<(DynamicMatrix, DynamicMatrix, Vec<f64>)> {
        let inputs = self.inputs.as_ref().unwrap();
        let z = self.z.as_ref().unwrap();
        
        if d_a.rows() != z.rows() || d_a.cols() != z.cols() {
            return Err(SciError::InvalidParameter("Derivative shape mismatch in backward pass"));
        }
        
        // dZ = dA * activation_derivative(Z)
        let mut d_z = d_a.clone();
        for r in 0..d_z.rows() {
            for c in 0..d_z.cols() {
                let d_val = d_z.get(r, c)? * self.activation.derivative(z.get(r, c)?);
                d_z.set(r, c, d_val)?;
            }
        }
        
        // dW = X^T * dZ
        let inputs_t = inputs.transpose();
        let d_w = inputs_t.mul_matrix(&d_z)?;
        
        // db = sum(dZ, axis=0)
        let mut d_b = vec![0.0; d_z.cols()];
        for r in 0..d_z.rows() {
            for c in 0..d_z.cols() {
                d_b[c] += d_z.get(r, c)?;
            }
        }
        
        // dX = dZ * W^T
        let weights_t = self.weights.transpose();
        let d_inputs = d_z.mul_matrix(&weights_t)?;
        
        Ok((d_inputs, d_w, d_b))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Models
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<DenseLayer>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    
    pub fn add(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }
    
    pub fn forward(&mut self, mut inputs: DynamicMatrix) -> SciResult<DynamicMatrix> {
        for layer in &mut self.layers {
            inputs = layer.forward(&inputs)?;
        }
        Ok(inputs)
    }
    
    pub fn backward(&mut self, mut d_a: DynamicMatrix) -> SciResult<Vec<(DynamicMatrix, Vec<f64>)>> {
        let mut gradients = Vec::new();
        for layer in self.layers.iter_mut().rev() {
            let (d_inputs, d_w, d_b) = layer.backward(&d_a)?;
            gradients.push((d_w, d_b));
            d_a = d_inputs;
        }
        gradients.reverse();
        Ok(gradients)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Optimizers
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct AdamOptimizer {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    t: usize,
    m_w: Vec<DynamicMatrix>,
    v_w: Vec<DynamicMatrix>,
    m_b: Vec<Vec<f64>>,
    v_b: Vec<Vec<f64>>,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64, layers: &[DenseLayer]) -> SciResult<Self> {
        let mut m_w = Vec::new();
        let mut v_w = Vec::new();
        let mut m_b = Vec::new();
        let mut v_b = Vec::new();
        
        for layer in layers {
            let r = layer.weights.rows();
            let c = layer.weights.cols();
            m_w.push(DynamicMatrix::zeros(r, c)?);
            v_w.push(DynamicMatrix::zeros(r, c)?);
            m_b.push(vec![0.0; c]);
            v_b.push(vec![0.0; c]);
        }
        
        Ok(Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w,
            v_w,
            m_b,
            v_b,
        })
    }
    
    pub fn step(&mut self, model: &mut Sequential, gradients: &[(DynamicMatrix, Vec<f64>)]) -> SciResult<()> {
        self.t += 1;
        
        for (i, layer) in model.layers.iter_mut().enumerate() {
            let (d_w, d_b) = &gradients[i];
            
            // Update weights
            for r in 0..layer.weights.rows() {
                for c in 0..layer.weights.cols() {
                    let grad = d_w.get(r, c)?;
                    
                    let mut m = self.m_w[i].get(r, c)?;
                    let mut v = self.v_w[i].get(r, c)?;
                    
                    m = self.beta1 * m + (1.0 - self.beta1) * grad;
                    v = self.beta2 * v + (1.0 - self.beta2) * grad * grad;
                    
                    self.m_w[i].set(r, c, m)?;
                    self.v_w[i].set(r, c, v)?;
                    
                    let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
                    let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));
                    
                    let w = layer.weights.get(r, c)?;
                    layer.weights.set(r, c, w - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon))?;
                }
            }
            
            // Update biases
            for c in 0..layer.biases.len() {
                let grad = d_b[c];
                
                let mut m = self.m_b[i][c];
                let mut v = self.v_b[i][c];
                
                m = self.beta1 * m + (1.0 - self.beta1) * grad;
                v = self.beta2 * v + (1.0 - self.beta2) * grad * grad;
                
                self.m_b[i][c] = m;
                self.v_b[i][c] = v;
                
                let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));
                
                layer.biases[c] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Losses
// ══════════════════════════════════════════════════════════════════════════════

pub struct MSELoss;

impl MSELoss {
    pub fn forward(preds: &DynamicMatrix, targets: &DynamicMatrix) -> SciResult<f64> {
        if preds.rows() != targets.rows() || preds.cols() != targets.cols() {
            return Err(SciError::InvalidParameter("Shape mismatch for MSE Loss"));
        }
        
        let mut loss = 0.0;
        for r in 0..preds.rows() {
            for c in 0..preds.cols() {
                let diff = preds.get(r, c)? - targets.get(r, c)?;
                loss += diff * diff;
            }
        }
        Ok(loss / (preds.rows() * preds.cols()) as f64)
    }
    
    pub fn backward(preds: &DynamicMatrix, targets: &DynamicMatrix) -> SciResult<DynamicMatrix> {
        let mut d_a = preds.clone();
        let factor = 2.0 / (preds.rows() * preds.cols()) as f64;
        for r in 0..d_a.rows() {
            for c in 0..d_a.cols() {
                let val = (preds.get(r, c)? - targets.get(r, c)?) * factor;
                d_a.set(r, c, val)?;
            }
        }
        Ok(d_a)
    }
}
