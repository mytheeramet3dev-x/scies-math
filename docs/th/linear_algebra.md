# เอกสารอ้างอิง: พีชคณิตเชิงเส้น (`linear_algebra`)

โมดูล `linear_algebra` ให้บริการโครงสร้างข้อมูลเมทริกซ์แบบหนาแน่น (`DynamicMatrix`), การคูณเมทริกซ์ประสิทธิภาพสูงด้วย Blocked GEMM, และการแยกตัวประกอบเมทริกซ์มาตรฐาน (LU, Householder QR, Cholesky, และ SVD)

---

## 1. เมทริกซ์ไดนามิก (`DynamicMatrix`)

จัดเก็บเมทริกซ์ขนาด $M \times N$ ในหน่วยความจำแบบแถวเป็นหลัก (Row-Major Contiguous Array) ด้วยชนิดข้อมูล `f64`

### การคูณเมทริกซ์แบบ Blocked GEMM
ใช้อัลกอริทึม Cache-Blocked Matrix Multiplication เพื่อเพิ่มอัตรา Cache Hit ใน L1/L2 data cache ของ CPU:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

เมื่อเปิดใช้งานฟีเจอร์ `parallel` การคูณเมทริกซ์ขนาดใหญ่จะกระจายงานข้ามเธรดอัตโนมัติผ่าน `rayon`

---

## 2. การแยกตัวประกอบเมทริกซ์ (Matrix Decompositions)

| การแยกตัวประกอบ | เมธอด | รูปแบบการแยกตัวประกอบ | เงื่อนไขของเมทริกซ์ | การใช้งานหลัก |
| :--- | :--- | :--- | :--- | :--- |
| **LU Decomposition** | `lu_decompose()` | $P A = L U$ | เมทริกซ์จัตุรัสไม่เอกฐาน | แก้ระบบสมการ $A x = b$, หา $\det(A)$ |
| **QR Decomposition** | `qr_decompose()` | $A = Q R$ | เมทริกซ์ $m \ge n$ ใด ๆ | Least Squares Regression, ฐานเชิงตั้งฉาก |
| **Cholesky** | `cholesky_decompose()` | $A = L L^T$ | สมมาตรและบวกแน่นอน ($A \succ 0$) | การแก้ระบบสมการ SPD ด้วยความเร็วเป็น 2 เท่าของ LU |
| **SVD** | `singular_value_decompose()` | $A = U \Sigma V^T$ | เมทริกซ์ขนาดใด ๆ | การประเมินค่าแรงค์, เมทริกซ์ผกผันเทียม, Condition Number |

---

## 3. ตัวอย่างการใช้งาน

```rust
use scies_math_th::linear_algebra::DynamicMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = DynamicMatrix::new(3, 3, vec![
        12.0, -51.0, 4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ])?;

    // 1. Householder QR Decomposition
    let qr = a.qr_decompose()?;
    let reconstructed = qr.q.mul_matrix(&qr.r)?;
    assert!((a.get(0, 0)? - reconstructed.get(0, 0)?).abs() < 1e-10);

    // 2. Singular Value Decomposition (SVD)
    let svd = a.singular_value_decompose(1e-10, 200)?;
    println!("Singular values: {:?}", svd.singular_values);
    println!("Condition Number (2-norm): {:.4}", svd.condition_number());

    Ok(())
}
```
