# เอกสารอ้างอิง: การหาค่าเหมาะที่สุดหลายตัวแปร (`opt_multivar`, `opt_multivar_ext`)

โมดูล `opt_multivar` และ `opt_multivar_ext` ให้บริการอัลกอริทึมการหาค่าเหมาะที่สุดแบบไม่มีเงื่อนไขบังคับ (Unconstrained) และแบบมีเงื่อนไขบังคับ (Constrained) รวมถึงขั้นตอนวิธีเชิงวิวัฒนาการแบบโกลบอล

---

## 1. การเปรียบเทียบอัลกอริทึม

| อัลกอริทึม | ประเภท | เกรเดียนต์ | ข้อจำกัดที่รองรับ | เหมาะสำหรับ |
| :--- | :--- | :--- | :--- | :--- |
| `bfgs` | Quasi-Newton | แม่นตรง (Dual AD) | ไม่มี | ฟังก์ชันที่เรียบ $C^2$ มิติปานกลาง ($n \le 1000$) |
| `lbfgs` | Limited-Memory BFGS | แม่นตรง (Dual AD) | ไม่มี | ปัญหาขนาดใหญ่ ($n > 10^4$) ประหยัด RAM |
| `conjugate_gradient` | Nonlinear CG (Polak-Ribière)| แม่นตรง (Dual AD) | ไม่มี | ปัญหาขนาดใหญ่ที่ไม่ต้องการเก็บ Hessian |
| `nelder_mead` | Downhill Simplex | ไม่ใช้ (0-th order) | ไม่มี | ฟังก์ชันที่หาอนุพันธ์ไม่ได้หรือมีสัญญาณรบกวน |
| `trust_region_cg` | Trust-Region (Steihaug) | Finite Difference | ไม่มี | ฟังก์ชันที่มีจุดอานม้า (Saddle points) หรือ Hessian ไม่แน่นอน |
| `projected_gradient`| Projected Gradient | Finite Difference | ขอบเขตแบบกล่อง $[l_i, u_i]$ | ปัญหาที่มีข้อจำกัดขอบเขตตัวแปรชัดเจน |
| `augmented_lagrangian`| Sequential Penalty | Finite Difference | สมการและอสมการไม่เชิงเส้น | ปัญหาที่มีข้อจำกัดทั่วไป |
| `particle_swarm` | Swarm Heuristic (PSO) | ไม่ใช้ | ขอบเขตแบบกล่อง | ปัญหา Global optimization ที่มีหลายยอดเขา |

---

## 2. Quasi-Newton BFGS และ L-BFGS

### BFGS
ประมาณค่า Hessian ผกผัน $H_k \approx \nabla^2 f(x_k)^{-1}$ ผ่านการอัปเดตแบบ Rank-2 ร่วมกับ Armijo Backtracking Line Search:

$$f(x_k + \alpha_k p_k) \le f(x_k) + c_1 \alpha_k \nabla f(x_k)^T p_k$$

### L-BFGS
เก็บเฉพาะเวกเตอร์การกระจัดล่าสุด $m$ คู่ $\{s_k, y_k\}$ และคำนวณทิศทางการก้าวผ่านขั้นตอนวิธี Two-Loop Recursion ด้วยเวลา $O(m \cdot n)$ และหน่วยความจำ $O(m \cdot n)$

---

## 3. ตัวอย่างการใช้งาน

```rust
use scies_math_th::autodiff::Dual;
use scies_math_th::opt_multivar::bfgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ฟังก์ชัน Rosenbrock: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    let rosenbrock = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        let one = Dual::from(1.0);
        let hundred = Dual::from(100.0);
        (one - x) * (one - x) + hundred * (y - x * x) * (y - x * x)
    };

    let x0 = [-1.2, 1.0];
    let (x_star, f_star) = bfgs(rosenbrock, &x0, 1e-6, 500)?;

    println!("จุดต่ำสุด x*: [x = {:.6}, y = {:.6}]", x_star[0], x_star[1]);
    println!("ค่า f(x*):   {:.2e}", f_star);

    Ok(())
}
```
