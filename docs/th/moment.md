# เอกสารอ้างอิง: ลำดับชั้น Moment-SOS และการผ่อนปรนพหุนาม (`moment`)

โมดูล `moment` จัดเตรียมการคำนวณตามลำดับชั้น Moment-Sum-of-Squares (Moment-SOS หรือลำดับชั้นของ Lasserre) สำหรับการแก้ปัญหาการหาค่าเหมาะที่สุดของพหุนาม (Polynomial Optimization) และการผ่อนปรนปัญหาความสอดคล้องของประพจน์ตรรกศาสตร์บูลีน (Boolean Satisfiability / 3-SAT) ให้อยู่ในรูป Semidefinite Programming (SDP)

---

## 1. รากฐานทางคณิตศาสตร์ (Mathematical Foundations)

### เมทริกซ์โมเมนต์แบบตัดทอน (Truncated Moment Matrix)

ให้ $x = (x_1, \dots, x_n)$ เป็นเวกเตอร์ของตัวแปร และ $\mathbf{v}_d(x)$ เป็นเวกเตอร์ของเอกนาม (Monomials) ทั้งหมดที่มีดีกรีไม่เกิน $d$ เมทริกซ์โมเมนต์ดีกรี $d$ ซึ่งเขียนแทนด้วย $M_d(y)$ คือเมทริกซ์สมมาตรขนาด $N \times N$ ที่มีนิยามดังนี้:

$$[M_d(y)]_{\alpha, \beta} = y_{\alpha + \beta}$$

โดยที่ $\alpha, \beta \in \mathbb{N}^n$ คือมัลติอินเด็กซ์ที่มีขนาด $|\alpha|, |\beta| \le d$ และ $y_\gamma = \mathbb{E}[x^\gamma]$ คือค่าซูโดโมเมนต์ (Pseudo-Moment) ของเอกนาม $x^\gamma$

### การลดทอนในไอดีลผลหารบูลีน (Boolean Quotient Reductions)

สำหรับปัญหาการหาค่าเหมาะที่สุดบนตัวแปรบูลีน เอกนามจะถูกทอนรูปภายใต้ไอดีลพีชคณิตของโดเมน:

1. **โดเมนศูนย์-หนึ่ง ($\{0, 1\}^n$)**:
   เกิดจากไอดีล $x_i^2 - x_i = 0$ ดังนั้นเลขยกกำลังใด ๆ $x_i^k$ ($k \ge 1$) จะลดรูปเหลือเพียง $x_i$:
   $$x_i^k \equiv x_i \pmod{x_i^2 - x_i}$$
2. **โดเมนบวก-ลบหนึ่ง ($\{-1, +1\}^n$)**:
   เกิดจากไอดีล $x_i^2 - 1 = 0$ เลขยกกำลังจะลดรูปตามภาวะคู่หรือคี่:
   $$x_i^k \equiv x_i^{k \bmod 2} \pmod{x_i^2 - 1}$$

### การเรียงลำดับเอกนามแบบ DegRevLex

เอกนามใน `MonomialBasis` จะถูกจัดเรียงตามลำดับ Graded Reverse Lexicographic (DegRevLex):
1. เรียงตามดีกรีรวม $|\alpha| = \sum \alpha_i$ จากน้อยไปมาก
2. หากดีกรีรวมเท่ากัน จะเปรียบเทียบเลขชี้กำลังจากตัวแปรตัวสุดท้าย $x_n$ ย้อนกลับมายัง $x_1$ โดยเอกนามที่มีเลขชี้กำลังน้อยกว่าจะมาก่อน

---

## 2. การเข้ารหัสประโยค CNF และ 3-SAT เป็นพหุนาม

ประโยคเชื่อมย่อย (Clause) ในรูป $C = (\ell_1 \lor \dots \lor \ell_k)$ จะถูกแปลงเป็นตัวบ่งชี้ความผิดพลาด (Error Indicator) $e_i$ ในโดเมน $\{0, 1\}$:

$$e_i = \begin{cases} 1 - x_v & \text{หาก } \ell_i = x_v \text{ (Positive literal)} \\ x_v & \text{หาก } \ell_i = \neg x_v \text{ (Negated literal)} \end{cases}$$

ประโยค $C$ จะถูกละเมิดก็ต่อเมื่อทุกตัวแปรในประโยคเป็นเท็จ ($e_1 = 1, \dots, e_k = 1$) ดังนั้นประโยคจะเป็นจริงก็ต่อเมื่อพหุนามแสดงการละเมิดมีค่าเป็นศูนย์:

$$P_C(x) = \prod_{i=1}^k e_i = 0$$

### การตรวจสอบขอบเขตดีกรีที่จำเป็น

ในการกำหนดข้อจำกัด $\mathbb{E}[P_C(x)] = 0$ บนเมทริกซ์โมเมนต์ $M_d(y)$ ดีกรีสูงสุดของ $P_C(x)$ จะต้องไม่เกินขอบเขตโมเมนต์ที่มีอยู่คือ $2d$ หาก $k > 2d$ ฟังก์ชันจะส่งกลับข้อผิดพลาดที่ระบุดีกรีที่ต้องใช้ชัดเจน:

```rust
Err(SciError::InvalidParameter("relaxation degree d is insufficient to represent clause polynomial of degree k"))
```

---

## 3. ตัวแปลงรูปแบบไฟล์ DIMACS CNF และ SDPA Sparse Format

### ตัวแปลงไฟล์ DIMACS CNF
โครงสร้าง `CnfFormula` มีฟังก์ชันตรวจสอบและแปลงไฟล์มาตรฐาน DIMACS CNF (`.cnf`):
- `CnfFormula::from_dimacs(&str)`: อ่านไฟล์พร้อมตรวจนับจำนวนตัวแปรและประโยค
- `formula.to_dimacs()`: ส่งออกสูตรกลับเป็นสตริง DIMACS
- `formula.solve_brute_force()`: ออราเคิลตรวจสอบความสอดคล้องแบบแม่นตรง $100\%$ ผ่านตารางค่าความจริงสำหรับสูตรขนาดเล็ก ($n \le 20$)

### ตัวแปลงรูปแบบไฟล์ SDPA Sparse Format (`.dat-s`)
- `export_sdpa_sparse(&SdpProblem, title)`: ส่งออกปัญหา SDP ในรูปแบบสากลที่สามารถนำไปรันกับ Solver ภายนอกได้ (เช่น SDPA, Mosek, SeDuMi)
- `import_sdpa_sparse(&str)`: นำเข้าไฟล์ SDPA พร้อมระบบตรวจสอบความถูกต้องอย่างเคร่งครัด

---

## 4. ตัวอย่างโค้ดการผ่อนปรน 3-SAT

```rust
use scies_math_th::moment::sat_encoding::{Clause, CnfFormula, Literal};
use scies_math_th::sdp::SdpSolverConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let formula = CnfFormula::new(
        3,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1), Literal::positive(2)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1), Literal::negative(2)]),
        ],
    );

    // สร้างการผ่อนปรนดีกรี 2 (รองรับโมเมนต์สูงสุดดีกรี 4)
    let (sdp, report) = formula.build_moment_sdp_relaxation_with_report(2)?;

    println!("จำนวนตัวแปร:             {}", report.num_variables);
    println!("ลำดับการผ่อนปรนดีกรี:      {}", report.relaxation_degree);
    println!("ขนาดของเมทริกซ์โมเมนต์:     {} x {}", report.basis_size, report.basis_size);
    println!("จำนวนโมเมนต์ทั้งหมด:       {}", report.num_moments);
    println!("จำนวนข้อจำกัดสมการ:       {}", report.num_generated_constraints);

    // แก้ปัญหาการผ่อนปรนด้วย ADMM
    let config = SdpSolverConfig::default();
    let solution = sdp.solve(&config)?;
    println!("สถานะผลลัพธ์ของ Solver: {:?}", solution.status);

    Ok(())
}
```
