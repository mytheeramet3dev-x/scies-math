# เอกสารอ้างอิง: Semidefinite Programming (`sdp`)

โมดูล `sdp` และ `sdp_primitives` ให้บริการโครงสร้างข้อมูลสำหรับปัญหา Semidefinite Programming (SDP), การวินิจฉัยสเปกตรัมของเมทริกซ์, การฉายภาพลงบนโคนเมทริกซ์กึ่งบวกแน่นอน (PSD Cone Projection), ตัวแก้ปัญหา ADMM ลำดับที่หนึ่ง, และฟังก์ชันตรวจสอบความถูกต้องของผลลัพธ์แบบอิสระ (Independent KKT Verifier)

---

## 1. นิยามปัญหาทางคณิตศาสตร์ (Mathematical Formulation)

ปัญหา SDP ใน `scies-math-th` ถูกกำหนดบนโคนของเมทริกซ์สมมาตรกึ่งบวกแน่นอนจริง $\mathcal{S}_+^n$:

### ปัญหาพรีมัล (Primal SDP)
$$\min_{X \in \mathcal{S}^n} \langle C, X \rangle \quad \text{subject to} \quad \langle A_i, X \rangle = b_i \quad (i = 1, \dots, m), \quad X \succeq 0$$

โดยที่ $\langle A, B \rangle = \text{Tr}(A^T B) = \sum_{j,k} A_{jk} B_{jk}$ คือผลคูณภายในแบบเทรซ (Frobenius Inner Product)

### ปัญหาดูอัล (Dual SDP)
$$\max_{y \in \mathbb{R}^m, S \in \mathcal{S}^n} b^T y \quad \text{subject to} \quad \sum_{i=1}^m y_i A_i + S = C, \quad S \succeq 0$$

โดยที่ $S \in \mathcal{S}_+^n$ คือเมทริกซ์ช่องว่างดูอัล (Dual Slack Matrix)

---

## 2. เงื่อนไขความเหมาะที่สุดของ Karush-Kuhn-Tucker (KKT Conditions)

ชุดผลเฉลย $(X^*, y^*, S^*)$ จะถือว่าเป็นผลเฉลยที่เหมาะสมที่สุดจริง (Verified Optimal Solution) ก็ต่อเมื่อสอดคล้องกับเงื่อนไขทั้ง 4 ข้อดังต่อไปนี้:

1. **ความเป็นไปได้ของพรีมัล (Primal Feasibility)**:
   $$\langle A_i, X \rangle = b_i \quad (\forall i \in \{1, \dots, m\}), \quad X \succeq 0$$
2. **ความเป็นไปได้ของดูอัล (Dual Feasibility)**:
   $$\sum_{i=1}^m y_i A_i + S = C, \quad S \succeq 0$$
3. **ความหลวมแบบสมบูรณ์ (Complementary Slackness)**:
   $$\langle X, S \rangle = \text{Tr}(X S) = 0$$
4. **ช่องว่างดูอัลลิตีเป็นศูนย์ (Zero Duality Gap)**:
   $$\langle C, X \rangle - b^T y = 0$$

### ตัวตรวจสอบอิสระภายนอก (`verify_sdp_candidate`)

ฟังก์ชัน `verify_sdp_candidate` ทำหน้าที่ตรวจสอบผลเฉลยที่คำนวณได้แบบ a-posteriori โดยไม่พึ่งพาสถานะภายในของตัวแก้สมการ และส่งกลับรายงาน [`SdpResidualReport`]:

```rust
pub fn verify_sdp_candidate(
    problem: &SdpProblem,
    x: &SymmetricMatrix,
    y: &[f64],
    s: &SymmetricMatrix,
    eigen_tolerance: f64,
) -> SciResult<SdpResidualReport>
```

ตัวชี้วัดความตกค้างที่ถูกตรวจสอบใน `SdpResidualReport`:
- `primal_equality_residual`: $r_p = \max_i |\langle A_i, X \rangle - b_i|$ (ความคลาดเคลื่อนของสมการพรีมัล)
- `primal_psd_violation`: $v_p = \max(0, -\lambda_{\min}(X))$ (การละเมิดความเป็นกึ่งบวกแน่นอนของพรีมัล)
- `dual_equality_residual`: $R_d = \|C - \sum y_i A_i - S\|_F$ (ความคลาดเคลื่อนของสมการดูอัล)
- `dual_psd_violation`: $v_d = \max(0, -\lambda_{\min}(S))$ (การละเมิดความเป็นกึ่งบวกแน่นอนของดูอัล)
- `relative_duality_gap`: $\frac{|\langle C, X \rangle - b^T y|}{1 + |\langle C, X \rangle| + |b^T y|}$ (ช่องว่างดูอัลลิตีสัมพัทธ์)
- `complementary_slackness_residual`: $\frac{|\text{Tr}(X S)|}{n}$

เมธอด `report.is_optimal(tol)` จะคืนค่าเป็น `true` ก็ต่อเมื่อค่าตกค้างทุกตัวมีค่าไม่เกินค่า `tol` ที่กำหนดไว้จริงเท่านั้น

---

## 3. สถาปัตยกรรมอัลกอริทึม ADMM (`SdpProblem::solve`)

`SdpProblem::solve` ใช้วิธี Alternating Direction Method of Multipliers (ADMM) ซึ่งเป็นอัลกอริทึมลำดับที่หนึ่งแบบ Operator-Splitting:

1. **ฟังก์ชัน Augmented Lagrangian**:
   $$\mathcal{L}_\rho(X, Z, \Lambda) = \langle C, X \rangle + \langle \Lambda, X - Z \rangle + \frac{\rho}{2} \|X - Z\|_F^2 \quad \text{subject to} \quad \mathcal{A}(X) = b, \; Z \in \mathcal{S}_+^n$$
2. **การอัปเดตตัวแปร $X$ (การฉายภาพลงบน Affine Subspace)**:
   แก้ระบบสมการเชิงเส้นโดยใช้เมทริกซ์ Gram แบบไม่ใส่ Regularization $G_{ij} = \langle A_i, A_j \rangle$ ผ่านการแยกตัวประกอบแบบ LU หากข้อจำกัดมีลักษณะพึ่งพิงเชิงเส้นกัน (Linearly Dependent) ระบบจะส่งกลับ `Err(SciError::SingularMatrix)` โดยไม่แอบใส่ค่าชดเชยเงียบ ๆ
3. **การอัปเดตตัวแปร $Z$ (การฉายภาพลงบน PSD Cone)**:
   $$Z^{k+1} = \Pi_{\mathcal{S}_+^n}\left(X^{k+1} + \frac{1}{\rho}\Lambda^k\right)$$
   คำนวณการแยกตัวประกอบสเปกตรัมผ่านอัลกอริทึม Jacobi และตัดค่าไอเกนที่เป็นลบให้กลายเป็นศูนย์
4. **การอัปเดตตัวคูณลากรองจ์ $\Lambda$**:
   $$\Lambda^{k+1} = \Lambda^k + \rho (X^{k+1} - Z^{k+1})$$

---

## 4. ตัวอย่างโค้ดการใช้งาน

```rust
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus, verify_sdp_candidate};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // กำหนดเมทริกซ์ต้นทุน C (ขนาด 2x2)
    // C = [1.0, 0.0; 0.0, 2.0]
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?;

    // ข้อจำกัดที่ 1: X_00 = 1.0
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?;
    // ข้อจำกัดที่ 2: X_11 = 1.0
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?;

    let problem = SdpProblem::new(c, vec![a1, a2], vec![1.0, 1.0])?;

    let config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 500,
        rho: 1.0,
        strict: true, // หาก strict = true จะบังคับรัน verifier ตรวจสอบความถูกต้อง
    };

    let solution = problem.solve(&config)?;
    assert_eq!(solution.status, SdpStatus::Optimal);

    println!("Primal Objective: {:.6}", solution.primal_objective); // ~ 3.000000
    println!("Dual Objective:   {:.6}", solution.dual_objective);

    // ทำการตรวจสอบผลเฉลยด้วยตนเองอีกชั้นหนึ่ง
    let report = verify_sdp_candidate(&problem, &solution.x, &solution.y, &solution.s, 1e-4)?;
    assert!(report.is_optimal(1e-4));
    println!("ผ่านการรับรองความถูกต้อง a-posteriori เรียบร้อย!");

    Ok(())
}
```

---

## 5. การวินิจฉัยสเปกตรัมของเมทริกซ์ (`sdp_primitives`)

โมดูล `sdp_primitives` ให้บริการเครื่องมือตรวจสอบคุณสมบัติทางสเปกตรัม:

- `diagnose_psd(&SymmetricMatrix, tol)`: ปฏิเสธค่า NaN/Inf และส่งกลับโครงสร้าง `PsdDiagnosis`:
  - `is_psd`: ค่า $\lambda_{\min} \ge -\text{tol}$ หรือไม่
  - `is_strictly_pd`: ค่า $\lambda_{\min} > \text{tol}$ (บวกแน่นอนแท้จริง) หรือไม่
  - `min_eigenvalue`, `max_eigenvalue`: ค่าไอเกนต่ำสุดและสูงสุด
  - `condition_number`: ค่าอัตราส่วน $\lambda_{\max} / \lambda_{\min}$
  - `numerical_rank`: จำนวนค่าไอเกนที่มีค่ามากกว่าค่าตัดเกณฑ์
- `project_to_psd_cone(&SymmetricMatrix)`: ฉายภาพเมทริกซ์สมมาตรใด ๆ ลงบนโคน $\mathcal{S}_+^n$ โดยมีระยะห่าง Frobenius น้อยที่สุด:
  $$\Pi_{\mathcal{S}_+^n}(A) = \sum_{i: \lambda_i > 0} \lambda_i v_i v_i^T$$
