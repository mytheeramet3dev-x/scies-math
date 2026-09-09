# เอกสารอ้างอิง: การหาอนุพันธ์อัตโนมัติ (`autodiff`, `reverse_ad`)

โมดูล `autodiff` และ `reverse_ad` ให้บริการการหาอนุพันธ์อัตโนมัติทั้งแบบไปข้างหน้า (Forward-Mode AD) โดยใช้ Dual Numbers และแบบย้อนกลับ (Reverse-Mode AD) โดยใช้ Wengert Tape

---

## 1. เปรียบเทียบระเบียบวิธี (Forward vs Reverse AD)

| มิติการเปรียบเทียบ | Forward-Mode AD (`autodiff`) | Reverse-Mode AD (`reverse_ad`) |
| :--- | :--- | :--- |
| **โครงสร้างข้อมูลหลัก** | [`Dual`] ($\langle u, u' \rangle$ โดยที่ $\varepsilon^2 = 0$) | [`Tape`] (กราฟการคำนวณ Wengert Tape) |
| **ความซับซ้อนในการหา $\nabla f$** | $O(n)$ รอบการส่งผ่านไปข้างหน้า | **$O(1)$ รอบการส่งผ่านย้อนกลับ** |
| **ความต้องการหน่วยความจำ** | ต่ำมาก ($O(1)$ บน Stack) | ปานกลาง ($O(\text{จำนวนการดำเนินการ})$ บน Heap) |
| **สถานการณ์ที่เหมาะสม** | จำนวนอินพุตน้อยกว่าเอาต์พุต ($n \le m$) หรือฟังก์ชันสเกลาร์ $f: \mathbb{R} \to \mathbb{R}$ | จำนวนอินพุตมากกว่าเอาต์พุตมาก ($n \gg m$) เช่น Machine Learning / Neural Networks |

---

## 2. ตัวอย่างการใช้งาน Forward-Mode AD

```rust
use scies_math_th::autodiff::{grad, jacobian, Dual};

fn main() {
    // 1. อนุพันธ์สเกลาร์: f(x) = x^3 + sin(x) ที่ x = 2.0
    let df = grad(|x| x.powi(3) + x.sin(), 2.0);
    println!("df/dx: {:.8}", df);

    // 2. จาโคเบียนหลายตัวแปร: f(x, y) = x^2 * y + exp(x + y)
    let f_multi = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        x * x * y + (x + y).exp()
    };

    let grad_vec = jacobian(f_multi, &[1.0, 2.0]);
    println!("เกรเดียนต์ ∇f: [df/dx = {:.6}, df/dy = {:.6}]", grad_vec[0], grad_vec[1]);
}
```

---

## 3. ตัวอย่างการใช้งาน Reverse-Mode AD

```rust
use scies_math_th::reverse_ad::Tape;

fn main() {
    let tape = Tape::new();

    let x = tape.var(3.0);
    let y = tape.var(2.0);

    // z = x^2 * y + sin(x)
    let z = (x * x * y) + x.sin();

    // รันการสะสมเกรเดียนต์ย้อนกลับ
    let grads = tape.backward(&z);

    println!("ค่า z:     {:.6}", z.val());
    println!("∂z/∂x:    {:.8}", grads.of(&x));
    println!("∂z/∂y:    {:.8}", grads.of(&y));
}
```
