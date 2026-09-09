# Generic Matrices and Fixed-Size Stack Arrays (`generic`)

The `generic` module provides generic scalar containers, dynamically sized matrices (`Mat<T>`), and zero-allocation const-generic fixed-size stack matrices (`SMatrix<T, R, C>`).

---

## 1. Type Comparison and Memory Model

| Type | Dimensions | Allocation | Scalar Constraint | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `Mat<T>` | Runtime $(M \times N)$ | Heap (`Vec<T>`) | `T: Scalar` | Variable-size datasets, large matrices |
| `SMatrix<T, R, C>`| Compile-time $(R \times C)$ | **Stack (`[T; R*C]`)** | `T: Scalar` | 3D graphics (3x3, 4x4), embedded systems, robotics |
| `Vec1<T>` | Runtime $(N \times 1)$ | Heap (`Vec<T>`) | `T: Scalar` | Column vectors |

### The `Scalar` Trait
Constrains types to numeric scalars supporting addition, subtraction, multiplication, division, zero, and one. Implemented for `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, and custom exact types.

---

## 2. Stack Matrices (`SMatrix<T, R, C>`)

`SMatrix` uses Rust's const generics to guarantee zero heap allocation overhead, maximum cache locality, and auto-vectorization:

```rust
use scies_math_th::generic::SMatrix;

fn main() {
    // 4x4 identity matrix allocated purely on the stack
    let eye: SMatrix<f64, 4, 4> = SMatrix::identity();
    assert_eq!(eye.get(0, 0), 1.0);
    assert_eq!(eye.get(0, 1), 0.0);

    // 2x2 matrix multiplication
    let a: SMatrix<f64, 2, 2> = SMatrix::from_array([
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    let b: SMatrix<f64, 2, 2> = SMatrix::from_array([
        [2.0, 0.0],
        [1.0, 2.0],
    ]);

    let c = a.mul(&b);
    println!("Matrix product: {:?}", c.as_slice());
}
```

---

## 3. Dynamic Matrices (`Mat<T>`)

```rust
use scies_math_th::generic::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Mat::<f64>::from_fn(3, 3, |r, c| if r == c { 2.0 } else { -1.0 });
    m.scale_mut(0.5);

    println!("Center element: {:.2}", m.get(1, 1)?); // 1.00
    Ok(())
}
```
