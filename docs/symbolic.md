# `symbolic` Module Documentation

Symbolic mathematics kernel: expression AST, canonical simplification, symbolic derivatives, and polynomial algebra.

## Overview

| Component | Description |
|---|---|
| [`Expr`] | AST for mathematical expressions (Integer, Rational, Symbol, Add, Mul, Pow, Sin, Cos, Exp, Ln, Sqrt) |
| [`simplify`] | Canonical algebraic simplification and constant folding |
| [`diff`] | Exact symbolic differentiation |
| [`Polynomial`] | Univariate polynomial algebra with Horner's evaluation and Euclidean division |

## Usage Examples

### Symbolic Differentiation

```rust
use scies_math_th::symbolic::{Expr, diff};

let x = Expr::sym("x");
let f = Expr::Add(vec![
    Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(3))), // x^3
    Expr::Mul(vec![Expr::int(2), x.clone()]),               // 2x
]);

let df = diff(&f, "x"); // 3x^2 + 2
println!("Derivative: {}", df);
```
