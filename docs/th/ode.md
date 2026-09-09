# เอกสารอ้างอิง: ตัวแก้สมการเชิงอนุพันธ์สามัญ (`ode`, `ode_ext`)

โมดูล `ode` และ `ode_ext` ให้บริการตัวแก้ปัญหาค่าเริ่มต้น (Initial Value Problems - IVPs) สำหรับสมการเชิงอนุพันธ์สามัญ (ODEs) ทั้งแบบออร์เดอร์คงที่, แบบปรับขนาดก้าวอัตโนมัติ (Adaptive), สำหรับระบบ Stiff, และตัวหาปริพันธ์แบบเรขาคณิต (Symplectic Integrators)

---

## 1. ตารางเลือกใช้ตัวแก้สมการ

| ตัวแก้สมการ | ออร์เดอร์ | รูปแบบ | จุดเด่นและการใช้งาน |
| :--- | :---: | :--- | :--- |
| `rk4` / `rk4_system` | 4 | Fixed-step | Classical Runge-Kutta มาตรฐานสำหรับโจทย์ทั่วไป |
| `rk45` | 4(5) | Adaptive | Dormand-Prince ปรับขนาดก้าว $\Delta t$ อัตโนมัติด้วยการควบคุมความคลาดเคลื่อน |
| `dopri8` | 8(5,3) | Adaptive | DOP853 ความแม่นยำสูงมาก ($10^{-10} - 10^{-14}$) สำหรับวิถีโคจรดาราศาสตร์ |
| `bdf2` | 2 | Fully Implicit | Backward Differentiation Formula สำหรับระบบ Stiff ป้องกันการสั่นเทียม |
| `backward_euler` | 1 | Fully Implicit | ระบบ Stiff อย่างยิ่ง มีเสถียรภาพแบบ A-stable และ L-stable |
| `stormer_verlet` | 2 | Symplectic | Leapfrog สำหรับกลศาสตร์ฮามิลโทเนียนและการจำลองโมเลกุล อนุรักษ์พลังงานระยะยาว |
| `yoshida4` | 4 | Symplectic | การประกอบวิธีซิมเพลกติกออร์เดอร์ 4 สำหรับวิถีดวงดาว |

---

## 2. ตัวอย่างการใช้งาน Adaptive RK45

```rust
use scies_math_th::ode::rk45;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ฮาร์มอนิกออสซิลเลเตอร์: y0' = y1, y1' = -y0
    let harmonic = |_t: f64, y: &[f64]| vec![y[1], -y[0]];

    let sol = rk45(
        harmonic,
        0.0, std::f64::consts::PI, // t ∈ [0, π]
        vec![1.0, 0.0],            // y(0) = 1.0, y'(0) = 0.0
        1e-7, 1e-10,               // rtol, atol
        0.1,                       // dt_init
    )?;

    let final_y = sol.y.last().unwrap();
    println!("y(π)  = {:.6} (ค่าแม่นตรง: -1.0)", final_y[0]);
    println!("y'(π) = {:.6} (ค่าแม่นตรง:  0.0)", final_y[1]);

    Ok(())
}
```

---

## 3. ตัวอย่างการจำลองวงโคจรด้วย Symplectic Verlet

```rust
use scies_math_th::ode_ext::stormer_verlet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ศักย์โน้มถ่วง: V(q) = -1 / ||q||
    let grad_v = |q: &[f64]| {
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let r3 = r * r * r;
        vec![q[0] / r3, q[1] / r3]
    };

    let sol = stormer_verlet(
        grad_v,
        vec![1.0, 0.0], // ตำแหน่งเริ่มต้น
        vec![0.0, 1.0], // โมเมนตัมเริ่มต้น (ความเร็ววงโคจรวงกลม)
        0.001,          // ขนาดก้าวเวลา dt
        10000,          // 10,000 สเต็ป
    )?;

    println!("ตำแหน่งเมื่อครบรอบ: {:?}", &sol.y.last().unwrap()[0..2]);
    Ok(())
}
```
