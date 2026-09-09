# เอกสารอ้างอิง: เลขคณิตเศษส่วนแท้และช่วงความคลาดเคลื่อน (`exact`)

โมดูล `exact` ให้บริการการคำนวณเศษส่วนแท้ที่แม่นยำทางพีชคณิตปราศจากข้อผิดพลาดจากการปัดเศษ (`Rational`) และการคำนวณขอบเขตของช่วงความคลาดเคลื่อน (`Interval`)

---

## 1. เลขคณิตเศษส่วนแท้ 128 บิต (`Rational`)

ตัวเลขเศษส่วนจะถูกเก็บในรูปบัญญัติ (Canonical Form) ประกอบด้วยเศษ $p \in \mathbb{Z}$ และส่วน $q \in \mathbb{N}^+$:

$$r = \frac{p}{q}, \quad \gcd(|p|, q) = 1, \quad q > 0$$

โดยใช้ชนิดข้อมูลจำนวนเต็ม 128 บิตแบบมีเครื่องหมาย (`i128`) ทำให้รองรับจำนวนเต็มได้สูงสุดถึง $\pm (2^{127} - 1) \approx \pm 1.7 \times 10^{38}$

### กฎความปลอดภัยที่ระบบบังคับใช้
- **ปฏิเสธตัวส่วนเป็นศูนย์**: `Rational::new(p, 0)` จะส่งกลับ `Err(SciError::DivisionByZero)` เสมอ
- **ทอนเป็นเศษส่วนอย่างต่ำอัตโนมัติ**: คำนวณตัวหารร่วมมาก ($\gcd$) ผ่านขั้นตอนวิธียูคลิดทันทีที่สร้างออบเจกต์
- **Checked Arithmetic ป้องกันการล้นขอบเขต (Overflow)**:
  - `checked_add(&self, other: &Self) -> SciResult<Self>`
  - `checked_sub(&self, other: &Self) -> SciResult<Self>`
  - `checked_mul(&self, other: &Self) -> SciResult<Self>`
  - `checked_div(&self, other: &Self) -> SciResult<Self>`
  - `checked_powi(&self, exponent: i32) -> SciResult<Self>`
- หากผลการคำนวณเกินขอบเขต 128 บิต จะคืนค่า `Err(SciError::ExactArithmeticOverflow)` แทนการเกิด Undefined Behavior

### การประมาณค่าเศษส่วนต่อเนื่องด้วย Stern-Brocot

แปลงค่าทศนิยมของ floating-point ให้กลับมาเป็นเศษส่วนอย่างต่ำที่ดีที่สุดภายใต้ขอบเขตตัวส่วน $q \le \text{max\_denom}$:

```rust
use scies_math_th::exact::Rational;

let pi_approx = Rational::from_f64_approx(3.141592653589793, 1000)?;
assert_eq!(pi_approx, Rational::new(355, 113)?);
```

---

## 2. เลขคณิตช่วงความคลาดเคลื่อน (`Interval`)

แทนจำนวนจริงด้วยช่วงปิด $[a, b] = \{x \in \mathbb{R} : a \le x \le b\}$:

- **การบวก**: $[a, b] + [c, d] = [a + c, b + d]$
- **การลบ**: $[a, b] - [c, d] = [a - d, b - c]$
- **การคูณ**: $[a, b] \cdot [c, d] = [\min(ac, ad, bc, bd), \max(ac, ad, bc, bd)]$
- **การหาร**: $[a, b] / [c, d] = [a, b] \cdot [1/d, 1/c]$ (โดยที่ $0 \notin [c, d]$)

> **ข้อสังเกตเชิงตัวเลข**: ในเวอร์ชัน 0.3.0 โครงสร้าง `Interval` ทำงานบน IEEE 754 `f64` ด้วยโหมดการปัดเศษมาตรฐาน หากงานคำนวณต้องการการการันตีทางคณิตศาสตร์แบบสัมบูรณ์ (Sub-epsilon rigorous bounds) แนะนำให้ใช้โครงสร้างเศษส่วนแท้ `Rational`

---

## 3. ตัวอย่างการใช้งาน

```rust
use scies_math_th::exact::{Interval, Rational};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. การคำนวณเศษส่วนแท้
    let r1 = Rational::new(1, 3)?;
    let r2 = Rational::new(5, 6)?;

    let sum = r1.checked_add(&r2)?;
    assert_eq!(sum, Rational::new(7, 6)?);

    let prod = r1.checked_mul(&r2)?;
    assert_eq!(prod, Rational::new(5, 18)?);

    // 2. การคำนวณช่วงความคลาดเคลื่อน
    let i1 = Interval::new(1.0, 2.0)?;
    let i2 = Interval::new(3.0, 4.0)?;

    let i_sum = (i1 + i2)?;
    assert_eq!(i_sum.lower(), 4.0);
    assert_eq!(i_sum.upper(), 6.0);
    assert!(i_sum.contains(5.0));

    Ok(())
}
```
