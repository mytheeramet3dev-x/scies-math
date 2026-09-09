# เอกสารอ้างอิง: พีชคณิตสัญลักษณ์และแคลคูลัส (`symbolic`)

โมดูล `symbolic` ให้บริการระบบพีชคณิตเชิงสัญลักษณ์ (Symbolic Computation Engine) แบบ zero-dependency รองรับต้นไม้นิพจน์สัญลักษณ์, การจัดรูปอย่างง่ายตามกฎพีชคณิต, การหาอนุพันธ์เชิงสัญลักษณ์, และการหารพหุนามยูคลิด

---

## 1. ต้นไม้นิพจน์สัญลักษณ์ (`Expr`)

`Expr` คือ enum ที่แทนโหนดต่าง ๆ ในต้นไม้นิพจน์คณิตศาสตร์ (Abstract Syntax Tree):

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

---

## 2. การจัดรูปอย่างง่ายแบบบัญญัติ (`simplify`)

เมธอด `expr.simplify()` ทำการลดรูปนิพจน์จากล่างขึ้นบนตามกฎพีชคณิตมาตรฐาน:

1. **เอกลักษณ์การบวก**:
   - $x + 0 \to x$
   - การรวมค่าคงที่: $\sum c_i \to C$
   - การคลายวงเล็บการบวกที่ซ้อนกัน: $(a + b) + c \to a + b + c$
2. **เอกลักษณ์การคูณ**:
   - $x \cdot 1 \to x$
   - $x \cdot 0 \to 0$
   - การคูณรวมค่าคงที่: $\prod c_i \to C$
3. **กฎเลขยกกำลัง**:
   - $x^0 \to 1, \quad x^1 \to x, \quad 1^x \to 1$
4. **ฟังก์ชันอดิศัย**:
   - $\sin(0) \to 0, \quad \cos(0) \to 1, \quad \exp(0) \to 1, \quad \ln(1) \to 0$

---

## 3. การหาอนุพันธ์เชิงสัญลักษณ์ (`diff`)

เมธอด `expr.diff(var)` ใช้กฎการหาอนุพันธ์เชิงวิเคราะห์:
- **ความเป็นเชิงเส้น**: $\frac{d}{dx}(u + v) = u' + v'$
- **กฎผลคูณ**: $\frac{d}{dx}(u \cdot v) = u' v + u v'$
- **กฎลูกโซ่และเลขยกกำลัง**: $\frac{d}{dx}(u^v) = u^v \left( v' \ln(u) + \frac{v u'}{u} \right)$
- **ฟังก์ชันตรีโกณมิติ**: $\frac{d}{dx}\sin(u) = \cos(u) \cdot u', \quad \frac{d}{dx}\cos(u) = -\sin(u) \cdot u'$

---

## 4. พีชคณิตพหุนาม (`Polynomial`)

แทนพหุนามตัวแปรเดียว $P(x) = \sum_{k=0}^n a_k x^k$ ผ่านเวกเตอร์สัมประสิทธิ์ $[a_0, a_1, \dots, a_n]$:
- **วิธีการของฮอร์เนอร์ (Horner's Method)**: คำนวณค่า $P(x)$ ด้วย $O(n)$
- **การหารพหุนามยูคลิดพร้อมเศษ (Euclidean Division)**:
  `poly_a.div_rem(&poly_b) -> SciResult<(Polynomial, Polynomial)>` หาผลหาร $Q(x)$ และเศษเหลือ $R(x)$ ที่ไม่ซ้ำกัน:
  $$A(x) = B(x) \cdot Q(x) + R(x), \quad \deg(R) < \deg(B)$$

---

## 5. ตัวอย่างการใช้งาน

```rust
use scies_math_th::symbolic::{Expr, Polynomial};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Expr::sym("x");

    // สร้างนิพจน์: f(x) = x^3 + sin(x)
    let f = Expr::Add(vec![
        Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(3))),
        Expr::Sin(Box::new(x.clone())),
    ]);

    let df = f.diff("x").simplify();
    println!("d/dx [{}] = {}", f.simplify(), df);

    // การหารพหุนาม: (2x^2 + 5x + 3) / (x + 1)
    let a = Polynomial::new(vec![3.0, 5.0, 2.0])?;
    let b = Polynomial::new(vec![1.0, 1.0])?;

    let (q, r) = a.div_rem(&b)?;
    println!("ผลหาร Quotient:  {:?}", q.coefficients()); // [3.0, 2.0] => 2x + 3
    println!("เศษเหลือ Remainder: {:?}", r.coefficients()); // [0.0] => 0

    Ok(())
}
```
