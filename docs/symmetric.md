# `symmetric` Module Documentation

Packed symmetric matrix storage ($n(n+1)/2$ elements) and linear algebra operations.

## Overview

Storing an $n \times n$ symmetric matrix in packed triangular layout reduces memory consumption by 50%, which is critical when constructing large moment matrices for polynomial and SDP relaxations.

```rust
use scies_math_th::symmetric::SymmetricMatrix;

let sym = SymmetricMatrix::new(2, vec![
    2.0,       // (0, 0)
    1.0, 3.0,  // (1, 0), (1, 1)
]).unwrap();

let x = vec![1.0, 2.0];
let qf = sym.quadratic_form(&x).unwrap(); // x^T A x = 18.0
```
