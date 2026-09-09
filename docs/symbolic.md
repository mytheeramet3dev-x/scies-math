# Symbolic Computation and Polynomial Algebra (`symbolic`)

The `symbolic` module provides a lightweight, zero-dependency symbolic expression tree engine, algebraic simplification, symbolic differentiation, and polynomial algebra.

---

## 1. The Symbolic Expression Tree (`Expr`)

`Expr` is an enum representing mathematical expression ASTs:

```rust
pub enum Expr {
    Integer(i64),
    Rational(Rational),
    Symbol(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Sqrt(Box<Expr>),
}
```

### Constructors & Helpers
- `Expr::int(n)`: Creates `Expr::Integer(n)`
- `Expr::rat(p, q)`: Creates `Expr::Rational(Rational::new(p, q))`
- `Expr::sym("x")`: Creates indeterminate variable `Expr::Symbol("x")`

---

## 2. Canonical Simplification (`simplify`)

`expr.simplify()` performs bottom-up recursive algebraic reductions:

1. **Additive Identities**:
   - $x + 0 \to x$
   - Constant folding: $\sum c_i \to C$
   - Flattening associative additions: $(a + b) + c \to a + b + c$
2. **Multiplicative Identities**:
   - $x \cdot 1 \to x$
   - $x \cdot 0 \to 0$
   - Constant folding: $\prod c_i \to C$
   - Flattening associative multiplications: $(a \cdot b) \cdot c \to a \cdot b \cdot c$
3. **Exponent Rules**:
   - $x^0 \to 1$
   - $x^1 \to x$
   - $1^x \to 1$
   - Constant evaluation: $c_1^{c_2} \to C$
4. **Special Function Identities**:
   - $\sin(0) \to 0$, $\cos(0) \to 1$, $\exp(0) \to 1$, $\ln(1) \to 0$, $\sqrt{0} \to 0$, $\sqrt{1} \to 1$

---

## 3. Symbolic Differentiation (`diff`)

`expr.diff(var)` applies the analytic calculus rules:
- **Linearity**: $\frac{d}{dx}(u + v) = u' + v'$
- **Product Rule**: $\frac{d}{dx}(u \cdot v) = u' v + u v'$
- **Power Rule / Chain Rule**: $\frac{d}{dx}(u^v) = u^v \left( v' \ln(u) + \frac{v u'}{u} \right)$
- **Trigonometric / Transcendental**:
  - $\frac{d}{dx}\sin(u) = \cos(u) \cdot u'$
  - $\frac{d}{dx}\cos(u) = -\sin(u) \cdot u'$
  - $\frac{d}{dx}\exp(u) = \exp(u) \cdot u'$
  - $\frac{d}{dx}\ln(u) = \frac{u'}{u}$

---

## 4. Polynomial Algebra (`Polynomial`)

Represents univariate real polynomials $P(x) = \sum_{k=0}^n a_k x^k$ stored as coefficient vectors $[a_0, a_1, \dots, a_n]$:

- **Horner's Method**: `poly.eval(x)` evaluates in $O(n)$ multiplications.
- **Euclidean Division with Remainder**:
  `poly_a.div_rem(&poly_b) -> SciResult<(Polynomial, Polynomial)>` computes unique $Q(x), R(x)$ such that:
  $$A(x) = B(x) \cdot Q(x) + R(x), \quad \deg(R) < \deg(B)$$
  Returns `Err(SciError::DivisionByZero)` if $B(x) = 0$.

---

## 5. Usage Example

```rust
use scies_math_th::symbolic::{Expr, Polynomial};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Expr::sym("x");

    // Expression: f(x) = x * x * x + sin(x)
    let f = Expr::Add(vec![
        Expr::Mul(vec![x.clone(), x.clone(), x.clone()]),
        Expr::Sin(Box::new(x.clone())),
    ]);

    let df = f.diff("x").simplify();
    println!("d/dx [{}] = {}", f.simplify(), df);

    // Polynomial Euclidean Division: (2x^2 + 5x + 3) / (x + 1)
    let a = Polynomial::new(vec![3.0, 5.0, 2.0])?;
    let b = Polynomial::new(vec![1.0, 1.0])?;

    let (q, r) = a.div_rem(&b)?;
    println!("Quotient:  {:?}", q.coefficients()); // [3.0, 2.0] => 2x + 3
    println!("Remainder: {:?}", r.coefficients()); // [0.0] => 0

    Ok(())
}
```
