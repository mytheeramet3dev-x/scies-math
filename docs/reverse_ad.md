# `reverse_ad` Module Documentation

Reverse-mode automatic differentiation (backpropagation).

Reverse AD builds a computation graph during the forward pass and then
propagates gradients backward.  The cost is O(forward) regardless of the
number of input variables — ideal when computing ∂L/∂θ for many parameters.

# Types

- [`Tape`] — records operations during the forward pass
- [`Var`] — a tracked scalar variable on the tape

# Usage

```rust
use scies_math::reverse_ad::{Tape, backward};

let tape = Tape::new();
let x = tape.var(3.0);
let y = tape.var(2.0);
let z = x * x + x * y; // z = x² + xy = 9 + 6 = 15

let grads = backward(&tape, z);
// dz/dx = 2x + y = 8
// dz/dy = x = 3
```

# Hessian

Compute the Hessian matrix H[i,j] = ∂²f/∂xᵢ∂xⱼ by applying forward AD
over a reverse-AD gradient:

```rust
use scies_math::autodiff::hessian;

let h = hessian(|v| v[0]*v[0] + v[1]*v[1], &[1.0, 2.0]);
// h ≈ [[2, 0], [0, 2]]
```