# เอกสารอ้างอิง: พีชคณิตเมทริกซ์เบาบางและตัวแก้ระบบสมการแบบวนซ้ำ (`sparse`)

โมดูล `sparse` ให้บริการโครงสร้างข้อมูลเมทริกซ์เบาบางแบบ Compressed Sparse Row (CSR), ตัวปรับสภาพล่วงหน้า (Preconditioners), ตัวแก้ระบบสมการเชิงเส้นแบบวนซ้ำ, และ Lanczos Eigensolver สำหรับเมทริกซ์สมมาตรขนาดใหญ่

---

## 1. โครงสร้าง Compressed Sparse Row (`SparseMatrixCsr`)

จัดเก็บเมทริกซ์ขนาด $m \times n$ ที่มีสมาชิกไม่เป็นศูนย์จำนวน $N_{\text{nz}}$ ตัว ด้วยเวกเตอร์สามชุด:
- `values: Vec<f64>`: ค่าเชิงตัวเลขที่ไม่เป็นศูนย์
- `col_indices: Vec<usize>`: ดัชนีคอลัมน์ของแต่ละค่า
- `row_pointers: Vec<usize>`: ตำแหน่งออฟเซ็ตเริ่มต้นของแต่ละแถวในเวกเตอร์ `values`

### การตรวจสอบ Invariants (`validate_invariants`)
- `row_pointers[0] == 0` และมีค่าเพิ่มขึ้นทางเดียวอย่างต่อเนื่อง (Monotonically non-decreasing)
- ดัชนีคอลัมน์เรียงลำดับจากน้อยไปมากในแต่ละแถวและไม่มีตัวซ้ำ
- ค่าตัวเลขทั้งหมดต้องเป็นจำนวนจำกัด (ปฏิเสธค่า NaN และ Inf)

---

## 2. ตัวแก้ระบบสมการเชิงเส้นแบบวนซ้ำ (Iterative Linear Solvers)

- `conjugate_gradient`: เหมาะสำหรับเมทริกซ์สมมาตรและบวกแน่นอน (Symmetric Positive Definite - SPD)
- `preconditioned_cg`: เร่งความเร็วการลู่เข้าของ CG ด้วยตัวปรับสภาพล่วงหน้า Jacobi หรือ Incomplete LU ($\text{ILU}(0)$)
- `gmres` / `preconditioned_gmres`: อัลกอริทึม Generalized Minimal Residual สำหรับเมทริกซ์ทั่วไปที่ไม่สมมาตร
- `bicgstab`: Bi-Conjugate Gradient Stabilized ให้การลู่เข้าที่ราบรื่น

---

## 3. Lanczos Symmetric Eigensolver (`lanczos_eigen`)

หาค่าไอเกนสูงสุด $k$ ค่าแรกและเวกเตอร์ไอเกนของเมทริกซ์สมมาตรขนาดใหญ่ด้วยการวนซ้ำแบบ Lanczos พร้อมการปรับตั้งฉากใหม่แบบเต็มรูปแบบ (Full Gram-Schmidt Reorthogonalization):

- **เวกเตอร์เริ่มต้นแบบกำหนดชัดเจน**: $q_1 = [1/\sqrt{n}, \dots, 1/\sqrt{n}]^T$ (Deterministic)
- **การติดตามค่าตกค้าง**: บันทึกค่า $\|A v_i - \lambda_i v_i\|_2$ สำหรับแต่ละคู่ Ritz pair
- **กฎเครื่องหมายมาตรฐาน**: บังคับให้สัมประสิทธิ์ตัวแรกที่มีนัยสำคัญของ eigenvector มีเครื่องหมายเป็นบวกเสมอ

---

## 4. ตัวอย่างการใช้งาน

```rust
use scies_math_th::sparse::{SparseMatrixCsr, lanczos_eigen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let triplets = vec![
        (0, 0, 2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3, 2.0), (3, 4, -1.0),
        (4, 3, -1.0), (4, 4, 2.0),
    ];

    let csr = SparseMatrixCsr::from_triplets(5, 5, &triplets)?;
    csr.validate_invariants()?;

    let eig = lanczos_eigen(&csr, 2, 20, 1e-8)?;
    assert!(eig.converged);

    println!("ค่าไอเกนสูงสุด 2 ค่าแรก: {:?}", eig.eigenvalues);
    println!("ค่าความตกค้าง ||Av - λv||: {:?}", eig.residuals);

    Ok(())
}
```
