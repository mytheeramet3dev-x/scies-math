# Neural Network Primitives (`nn`)

The `nn` module provides foundational building blocks for multi-layer feed-forward neural networks (MLP), backpropagation, non-linear activation functions, and gradient-based training optimizers.

---

## 1. Components

### Activation Functions
- **Linear**: $f(x) = x, \; f'(x) = 1$
- **ReLU**: $f(x) = \max(0, x), \; f'(x) = \mathbb{I}(x > 0)$
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}, \; \sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \; \tanh'(x) = 1 - \tanh^2(x)$

### Dense Layer (`Dense`)
Computes affine transformation followed by activation:

$$Z = X W + b, \quad A = \sigma(Z)$$

Weights are initialized via He normal initialization for ReLU or Xavier/Glorot initialization for Tanh/Sigmoid.

### Optimizers
- **SGD**: Classic stochastic gradient descent with learning rate $\eta$.
- **Adam**: Adaptive moment estimation maintaining exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 2. Code Example

```rust
use scies_math_th::nn::{Activation, Dense, Sequential};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construct MLP: 2 inputs -> 4 hidden (ReLU) -> 1 output (Linear)
    let mut model = Sequential::new();
    model.add(Dense::new(2, 4, Activation::ReLU));
    model.add(Dense::new(4, 1, Activation::Linear));

    // Input feature vector [x0, x1]
    let input = vec![1.0, 2.0];
    let output = model.forward(&input)?;

    println!("Model output shape: {}, value: {:.4}", output.len(), output[0]);

    Ok(())
}
```
