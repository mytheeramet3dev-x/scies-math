# เอกสารอ้างอิงรายโมดูลภาษาไทย (`docs/th/`)

โฟลเดอร์นี้รวบรวมเอกสารทางเทคนิคและคู่มือการใช้งานเชิงลึกรายโมดูลของ `scies-math-th` (v0.3.0) ฉบับภาษาไทย พร้อมสมการคณิตศาสตร์ ทฤษฎีบทเบื้องหลัง และตัวอย่างโค้ดที่คอมไพล์ได้จริง

---

## สารบัญเอกสารรายโมดูล

| เอกสาร | โมดูลที่เกี่ยวข้อง | หัวข้อหลักและเนื้อหา |
| :--- | :--- | :--- |
| [sdp.md](sdp.md) | `sdp`, `sdp_primitives` | Semidefinite Programming, รูปแบบ Primal-Dual, KKT Residuals, ตัวแก้สมการ ADMM และการฉายภาพลงบนโคน PSD |
| [sdp_ipm.md](sdp_ipm.md) | `sdp_ipm` | Mehrotra Predictor-Corrector Primal-Dual Interior-Point Method (IPM), Schur Complement, และ Step-Damping |
| [moment.md](moment.md) | `moment` | ลำดับชั้น Moment-SOS (Lasserre), ฐาน DegRevLex Monomials, การลดทอนใน Boolean ideal, การผ่อนปรน 3-SAT, และ SDPA I/O |
| [exact.md](exact.md) | `exact` | เลขคณิตเศษส่วนแท้ 128 บิต (`Rational`), การป้องกัน Overflow, อัลกอริทึม Stern-Brocot, และช่วงความคลาดเคลื่อน (`Interval`) |
| [symbolic.md](symbolic.md) | `symbolic` | ต้นไม้นิพจน์สัญลักษณ์ (`Expr`), Canonical Simplifier, อนุพันธ์สัญลักษณ์, และการหารพหุนามยูคลิดพร้อมเศษ |
| [units.md](units.md) | `units` | การวิเคราะห์มิติทางฟิสิกส์ 7 มิติหลักตามมาตรฐาน SI, พีชคณิตมิติ, และปริมาณที่ปลอดภัยทางมิติ (`Quantity`) |
| [verification.md](verification.md) | `verification` | การบันทึกสายธารการอนุพัทธ์ (Derivation Trees), Provenance metadata, และสะพานส่งออกทฤษฎีบท Lean 4 / Mathlib4 |
| [linear_algebra.md](linear_algebra.md) | `linear_algebra` | เมทริกซ์แบบไดนามิก, Blocked GEMM, การแยกตัวประกอบ (LU, Householder QR, Cholesky, SVD) และ Condition Number |
| [sparse.md](sparse.md) | `sparse` | โครงสร้าง Compressed Sparse Row (CSR), การตรวจ Invariants, Lanczos Eigensolver และ Iterative Solvers (PCG, GMRES, BiCGSTAB) |
| [opt_multivar.md](opt_multivar.md) | `opt_multivar`, `opt_multivar_ext` | การหาค่าเหมาะที่สุดหลายตัวแปร (BFGS, L-BFGS, Armijo line search, Trust-Region, Projected Gradient, Augmented Lagrangian) |
| [autodiff.md](autodiff.md) | `autodiff`, `reverse_ad` | อนุพันธ์อัตโนมัติแบบไปข้างหน้าด้วย Dual Numbers และแบบย้อนกลับด้วย Wengert Tape (VJP, Reverse AD) |
| [ode.md](ode.md) | `ode`, `ode_ext` | ตัวแก้สมการเชิงอนุพันธ์สามัญ: Classical RK4, Adaptive Dormand-Prince RK45, Stiff (BDF2/Implicit), และ Symplectic Integrators (Leapfrog/Yoshida) |
| [pde.md](pde.md) | `pde`, `pde_ext` | ตัวแก้สมการเชิงอนุพันธ์ย่อย: 2D Heat ADI (Peaceman-Rachford), 2D Wave (CFL), Burgers' Shock Waves, Fisher-KPP, และ Shooting BVP |

---

## เอกสารหลักอื่น ๆ
- [ภาพรวมภาษาไทยฉบับหลัก](../../README.th.md)
- [English Documentation Root](../README.md)
- [ลำดับชั้นความน่าเชื่อถือทางคณิตศาสตร์](../numerical_reliability.md)
- [เมทริกซ์ความถูกต้องของแต่ละระบบย่อย](../api_reliability_matrix.md)
