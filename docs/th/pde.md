# เอกสารอ้างอิง: ตัวแก้สมการเชิงอนุพันธ์ย่อยและปัญหาค่าขอบเขต (`pde`, `pde_ext`)

โมดูล `pde` และ `pde_ext` ให้บริการตัวแก้สมการเชิงอนุพันธ์ย่อย (PDEs) แบบมิติต่าง ๆ และปัญหาค่าขอบเขตสองจุด (Two-Point BVPs)

---

## 1. สมการความร้อน 2 มิติผ่าน ADI (`heat_2d_adi`)

แก้สมการการแพร่ความร้อนแบบพาราโบลิก 2 มิติ:

$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

ใช้วิธี Peaceman-Rachford Alternating Direction Implicit (ADI) ซึ่งแบ่งสเต็ปเวลาเป็นสองครึ่งสเต็ป (Implicit ในแนวแกน $x$ และแกน $y$ สลับกัน) โดยแก้ระบบสมการ Tridiagonal ผ่าน Thomas algorithm ทำให้มี **เสถียรภาพแบบไร้เงื่อนไข (Unconditionally Stable)**

```rust
use scies_math_th::pde_ext::heat_2d_adi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx = 21;
    let ny = 21;
    let mut u0 = vec![0.0f64; nx * ny];
    u0[(ny / 2) * nx + (nx / 2)] = 100.0; // แหล่งความร้อนตรงกลาง

    let u_final = heat_2d_adi(&u0, nx, ny, 0.05, 0.05, 0.001, 1.0, 50)?;
    println!("อุณหภูมิตรงกลางหลังแพร่: {:.4}", u_final[(ny / 2) * nx + (nx / 2)]);
    Ok(())
}
```

---

## 2. สมการคลื่นไฮเพอร์โบลิก 2 มิติ (`wave_2d`)

แก้สมการคลื่น $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$ โดยต้องสอดคล้องกับเงื่อนไขเสถียรภาพของ Courant-Friedrichs-Lewy (CFL):

$$C = c \Delta t \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}} \le 1$$

---

## 3. สมการเบอร์เกอร์สและการไหลแบบไม่เชิงเส้น (`burgers_1d`)

แก้สมการ $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$ โดยใช้ **Upwind Differencing** สำหรับพจน์ Advection และ Central Difference สำหรับพจน์ Diffusion เพื่อจำลองคลื่นกระแทก (Shock Wave) โดยไม่เกิดการสั่นเชิงตัวเลข

---

## 4. ปัญหาค่าขอบเขตสองจุด (BVP)

- **วิธี Single Shooting (`bvp_shooting`)**: ยิงวิถีด้วย RK4 และปรับความชันเริ่มต้นด้วย Bisection Root Finding
- **Chebyshev Spectral Collocation (`chebyshev_bvp`)**: แก้ปัญหา $a_2(x) u'' + a_1(x) u' + a_0(x) u = f(x)$ บนจุด Chebyshev-Gauss-Lobatto ด้วยเมทริกซ์อนุพันธ์ของ Fornberg ให้ความแม่นยำระดับ Exponential
