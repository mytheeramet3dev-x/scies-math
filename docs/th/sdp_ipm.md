# เอกสารอ้างอิง: ตัวแก้สมการ SDP แบบ Interior-Point (`sdp_ipm`)

โมดูล `sdp_ipm` ให้บริการตัวแก้สมการ Semidefinite Programming ความแม่นยำสูงด้วยระเบียบวิธี Interior-Point แบบ Mehrotra Predictor-Corrector Primal-Dual (IPM)

---

## 1. ภาพรวมอัลกอริทึม (Algorithm Overview)

ระเบียบวิธี Interior-Point จะแก้ระบบสมการ KKT ที่ถูกรบกวน (Perturbed KKT System) ตามเส้นทางแกนกลาง (Central Path) ที่ถูกกำหนดด้วยพารามิเตอร์บาเรียร์ $\mu > 0$:

$$X S = \mu I$$

$$\langle A_i, X \rangle = b_i \quad (i = 1, \dots, m), \quad X \succ 0$$

$$\sum_{i=1}^m y_i A_i + S = C, \quad S \succ 0$$

เมื่อลดค่า $\mu \to 0$ ลำดับของผลเฉลย $(X(\mu), y(\mu), S(\mu))$ จะลู่เข้าสู่จุดที่เหมาะสมที่สุดทั้งในมุมพรีมัลและดูอัลอย่างแม่นยำ

### ขั้นตอนการทำงานแบบ Mehrotra Predictor-Corrector

ในแต่ละรอบของการวนซ้ำ (Iteration) อัลกอริทึมจะประกอบด้วย 2 ขั้นตอนหลัก:

1. **ขั้นตอนทำนายแบบแอฟไฟน์ (Affine-Scaling Predictor Step)**:
   แก้ระบบสมการเชิงเส้นของนิวตันโดยกำหนดให้ $\mu = 0$ เพื่อหาทิศทางก้าวเชิงแอฟไฟน์ $(\Delta X_{\text{aff}}, \Delta y_{\text{aff}}, \Delta S_{\text{aff}})$
2. **ขั้นตอนปรับศูนย์กลางและแก้ไขความโค้ง (Centering & Corrector Step)**:
   ประเมินขนาดก้าว $\alpha_{\text{primal}}^{\text{aff}}, \alpha_{\text{dual}}^{\text{aff}}$ เพื่อคำนวณพารามิเตอร์การปรับศูนย์กลาง:
   $$\sigma = \left(\frac{\mu_{\text{aff}}}{\mu}\right)^3 \in [0, 1]$$
   จากนั้นแก้ระบบสมการอีกครั้งโดยเพิ่มเวกเตอร์ทางขวามือด้วยเทอมปรับศูนย์กลาง $\sigma \mu I$ และเทอมแก้ไขความไม่เชิงเส้น $-\Delta X_{\text{aff}} \Delta S_{\text{aff}}$
3. **การคำนวณขนาดก้าวพร้อมระบบหน่วงความปลอดภัย (Step-Damping)**:
   $$\alpha_P = \min\left(1, \gamma \cdot \sup\{\alpha > 0 : X + \alpha \Delta X \succeq 0\}\right)$$
   $$\alpha_D = \min\left(1, \gamma \cdot \sup\{\alpha > 0 : S + \alpha \Delta S \succeq 0\}\right)$$
   โดยที่ $\gamma \in (0, 1)$ คือตัวคูณลดระยะเพื่อป้องกันไม่ให้จุดคำนวณแตะขอบของโคน (ค่าเริ่มต้นคือ $0.95$)

---

## 2. การตั้งค่าและตรวจสอบพารามิเตอร์ (`SdpIpmConfig`)

| พารามิเตอร์ | ชนิดข้อมูล | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `tolerance` | `f64` | `1e-7` | ค่าความคลาดเคลื่อนเป้าหมายสำหรับช่องว่างดูอัลลิตีสัมพัทธ์ (ต้อง $> 0$) |
| `max_iterations` | `usize` | `100` | จำนวนรอบการวนซ้ำสูงสุด (ต้อง $> 0$) |
| `step_damping` | `f64` | `0.95` | แฟกเตอร์รักษาระยะห่างจากขอบเขต $\gamma \in (0, 1)$ |

หากค่าพารามิเตอร์ใดอยู่นอกช่วงที่กำหนด ฟังก์ชัน `solve_sdp_ipm` จะส่งกลับ `Err(SciError::InvalidParameter)` ทันที

---

## 3. ระบบสมการเชิงเส้น Schur Complement

ในแต่ละรอบการคำนวณ ทิศทางของตัวแปรดูอัล $\Delta y$ จะถูกหาจากการแก้ระบบสมการ Schur Complement:

$$M \Delta y = r$$

โดยที่เมทริกซ์ $M$ ขนาด $m \times m$ มีนิยามดังนี้:

$$M_{ij} = \text{Tr}\left(A_i S^{-1} A_j X\right)$$

- หากเมทริกซ์ $M$ เป็นซิงกูลาร์ (Singular) หรือมีสภาพแย่ (Ill-Conditioned) ตัวแก้สมการจะส่งกลับ `Err(SciError::SingularMatrix)` โดยไม่ใช้ Fallback ซ่อนข้อผิดพลาด
- หากการหาเมทริกซ์ผกผันของ $S$ พบค่าไอเกนเป็นศูนย์ จะส่งกลับ `Err(SciError::DivisionByZero)`

---

## 4. ตัวอย่างการใช้งาน

```rust
use scies_math_th::sdp::SdpProblem;
use scies_math_th::sdp_ipm::{solve_sdp_ipm, SdpIpmConfig};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?;
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?;
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?;
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b)?;

    let config = SdpIpmConfig {
        tolerance: 1e-8,
        max_iterations: 60,
        step_damping: 0.95,
    };

    let solution = solve_sdp_ipm(&problem, &config)?;

    println!("Primal Objective: {:.8}", solution.primal_objective);
    println!("Dual Objective:   {:.8}", solution.dual_objective);
    println!("จำนวนรอบการคำนวณ: {}", solution.iterations);

    // ตรวจสอบความถูกต้องของรายงาน KKT
    assert!(solution.residuals.is_optimal(1e-7));
    println!("Duality Gap:      {:.2e}", solution.residuals.relative_duality_gap);

    Ok(())
}
```
