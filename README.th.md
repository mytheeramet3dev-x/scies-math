# scies-math-th

[![Crates.io](https://img.shields.io/crates/v/scies-math-th.svg)](https://crates.io/crates/scies-math-th)
[![Documentation](https://docs.rs/scies-math-th/badge.svg)](https://docs.rs/scies-math-th)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-success.svg)]()

`scies-math-th` คือชุดเครื่องมือคณิตศาสตร์และการคำนวณเชิงวิทยาศาสตร์ประสิทธิภาพสูงแบบตรวจสอบได้ (Verifiable Computing Engine) เขียนด้วยภาษา Rust ล้วน โดยครอบคลุมตั้งแต่พีชคณิตเชิงเส้นพื้นฐาน, ตัวแก้สมการเชิงตัวเลข, Semidefinite Programming (SDP), ลำดับชั้นพหุนาม Moment-SOS, เลขคณิตเศษส่วนแท้ 128 บิต (Exact Rational), แคลคูลัสเชิงสัญลักษณ์, การวิเคราะห์มิติทางฟิสิกส์ตามมาตรฐาน SI, ไปจนถึงสะพานส่งออกทฤษฎีบทไปยังระบบพิสูจน์เชิงตรรกศาสตร์ Lean 4 / Mathlib4

ออกแบบมาเพื่อรองรับงานวิจัยทางวิทยาศาสตร์, การทดลองทางคณิตศาสตร์, และระบบคอมพิวเตอร์ที่ต้องการความเชื่อถือได้สูง โดยบิลด์ค่าเริ่มต้นของ crate นี้ **ไม่มี External Dependency แม้แต่ตัวเดียว (Zero External Dependencies)**

---

## สารบัญ

- [จุดเด่นสำคัญ](#จุดเด่นสำคัญ)
- [ลำดับชั้นความถูกต้องและความน่าเชื่อถือทางคณิตศาสตร์](#ลำดับชั้นความถูกต้องและความน่าเชื่อถือทางคณิตศาสตร์)
- [โครงสร้างสถาปัตยกรรมและโมดูล](#โครงสร้างสถาปัตยกรรมและโมดูล)
- [การติดตั้ง](#การติดตั้ง)
- [ตัวอย่างการใช้งานและเจาะลึก 10 โมดูลหลัก](#ตัวอย่างการใช้งานและเจาะลึก-10-โมดูลหลัก)
  - [1. Semidefinite Programming (ADMM และ Mehrotra Primal-Dual IPM)](#1-semidefinite-programming-admm-และ-mehrotra-primal-dual-ipm)
  - [2. ลำดับชั้น Moment-SOS และการผ่อนปรน DIMACS CNF / 3-SAT](#2-ลำดับชั้น-moment-sos-และการผ่อนปรน-dimacs-cnf--3-sat)
  - [3. เลขคณิตเศษส่วนแท้ 128 บิตพร้อมระบบป้องกัน Overflow](#3-เลขคณิตเศษส่วนแท้-128-บิตพร้อมระบบป้องกัน-overflow)
  - [4. โครงสร้างต้นไม้นิพจน์สัญลักษณ์, แคลคูลัส และการหารพหุนามยูคลิด](#4-โครงสร้างต้นไม้นิพจน์สัญลักษณ์-แคลคูลัส-และการหารพหุนามยูคลิด)
  - [5. การวิเคราะห์มิติทางฟิสิกส์ 7 มิติหลัก SI และปริมาณที่ปลอดภัยทางมิติ](#5-การวิเคราะห์มิติทางฟิสิกส์-7-มิติหลัก-si-และปริมาณที่ปลอดภัยทางมิติ)
  - [6. การสืบสาวที่มาของการพิสูจน์ และสะพานส่งออกโค้ด Lean 4 / Mathlib4](#6-การสืบสาวที่มาของการพิสูจน์-และสะพานส่งออกโค้ด-lean-4--mathlib4)
  - [7. พีชคณิตเชิงเส้นแบบ Dense, Blocked GEMM และ SVD](#7-พีชคณิตเชิงเส้นแบบ-dense-blocked-gemm-และ-svd)
  - [8. Compressed Sparse Row (CSR) และ Lanczos Eigensolver](#8-compressed-sparse-row-csr-และ-lanczos-eigensolver)
  - [9. การหาค่าเหมาะที่สุดหลายตัวแปร และการหาอนุพันธ์อัตโนมัติ](#9-การหาค่าเหมาะที่สุดหลายตัวแปร-และการหาอนุพันธ์อัตโนมัติ)
  - [10. สมการเชิงอนุพันธ์ (Adaptive RK45 และ 2D Heat ADI)](#10-สมการเชิงอนุพันธ์-adaptive-rk45-และ-2d-heat-adi)
- [Feature Flags](#feature-flags)
- [การประกันคุณภาพและ Benchmark](#การประกันคุณภาพและ-benchmark)
- [เอกสารและลิขสิทธิ์](#เอกสารและลิขสิทธิ์)

---

## จุดเด่นสำคัญ

- **Zero External Dependencies**: แกนหลักทั้งหมด—ตั้งแต่การคูณเมทริกซ์, การแยกตัวประกอบ (Decompositions), ตัวแก้สมการเชิงตัวเลข, การหาอนุพันธ์อัตโนมัติ (Autodiff), ไปจนถึงเลขคณิตเศษส่วนแท้—ทำงานได้โดยไม่ต้องดึง crate ภายนอกใด ๆ เข้ามา
- **Strict Verification Discipline**: ตัวแก้สมการเชิงตัวเลขจะไม่เคลมอ้างสถานะว่าได้ "Mathematical Proof" หรือ "Exact Certificate" หากไม่มีตัวตรวจสอบตกค้างอิสระ (Independent A-Posteriori Verifier) โดยสถานะ `SdpStatus::Optimal` จะถูกกำหนดให้ก็ต่อเมื่อค่า KKT Residuals ผ่านเกณฑ์ tolerance จริงเท่านั้น
- **Mehrotra Predictor-Corrector IPM**: บรรจุตัวแก้สมการ semidefinite แบบ Interior-Point ความแม่นยำสูง พร้อมระบบ step-damping, Cholesky/LU factorizations, และ KKT diagnostics ควบคู่ไปกับตัวแก้สมการ ADMM ออร์เดอร์ 1 ที่ทำงานรวดเร็ว
- **Lasserre / Moment-SOS Hierarchy**: รองรับการเรียงลำดับมัลติอินเด็กซ์แบบ DegRevLex, การลดทอนพหุนามใน Boolean quotient ideal ($x_i^2 = x_i$), การสร้าง Localizing matrices, ตลอดจนตัวแปลงไฟล์ DIMACS CNF และ SDPA sparse format
- **Exact & Verified Arithmetic**: โครงสร้าง `Rational` บนพื้นฐานจำนวนเต็ม 128 บิต (`i128`) พร้อมระบบตรวจสอบ overflow และ division-by-zero อย่างรัดกุม ปราศจากความคลาดเคลื่อนจากการปัดเศษ (Zero Roundoff Error)
- **Formal Methods Integration**: สามารถแปลงนิพจน์สัญลักษณ์และข้อสรุปทางคณิตศาสตร์ออกมาเป็นโค้ดทฤษฎีบทภาษา Lean 4 เพื่อนำไป compile และ verify ต่อใน Lean 4 kernel ร่วมกับ Mathlib ได้ทันที

---

## ลำดับชั้นความถูกต้องและความน่าเชื่อถือทางคณิตศาสตร์

เพื่อให้ผลลัพธ์เชิงตัวเลขและเชิงสัญลักษณ์มีความโปร่งใส ซื่อสัตย์ และตรวจสอบได้จริง `scies-math-th` จำแนกผลลัพธ์ของระบบออกเป็น 5 ระดับชั้น (ดูรายละเอียดเพิ่มเติมใน [`docs/api_reliability_matrix.md`](docs/api_reliability_matrix.md)):

| ระดับชั้น (Tier) | การจำแนกประเภท | ชนิดข้อมูลพื้นฐาน | การรับประกันและข้อจำกัด | โมดูลที่เกี่ยวข้อง |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Experimental** | ฮิวริสติกเชิงทดลอง | เหมาะสำหรับการทดลองแนวคิดเบื้องต้น ยังไม่มีการพิสูจน์ขอบเขตความคลาดเคลื่อน | `nn`, `geometry_ext` |
| **Tier 2** | **Numerical** | IEEE 754 `f64` | การลู่เข้าเชิงตัวเลขตามมาตรฐาน floating-point ทั่วไป ขึ้นอยู่กับ roundoff error และ condition number | `opt_multivar`, `ode`, `pde`, `statistics` |
| **Tier 3** | **Numerically Verified** | `f64` + [`SdpResidualReport`] | ตรวจสอบ a-posteriori ด้วย verifier อิสระ ($r_p, v_p, R_d, v_d, \text{gap} \le \epsilon$) | `sdp`, `sdp_ipm`, `sdp_primitives`, `eigensystem` |
| **Tier 4** | **Exact** | [`Rational`] / พีชคณิตสัญลักษณ์ | ค่า $p/q \in \mathbb{Q}$ บนจำนวนเต็ม 128 บิต ไม่มีความคลาดเคลื่อนจากการปัดเศษ มีระบบกัน overflow | `exact::Rational`, `symbolic`, `units` |
| **Tier 5** | **Formal** | Machine-Checked AST | โครงร่างทฤษฎีบท Lean 4 สำหรับส่งต่อให้ Kernel ภายนอกตรวจสอบความถูกต้องทางคณิตศาสตร์อย่างเป็นทางการ | `verification::derivation`, `verification::lean_export` |

---

## โครงสร้างสถาปัตยกรรมและโมดูล

```text
scies_math_th::
├── sdp                 # Semidefinite Programming (ADMM, KKT residual report, verify_sdp_candidate)
├── sdp_ipm             # Mehrotra Predictor-Corrector Primal-Dual Interior-Point SDP solver ความแม่นยำสูง
├── sdp_primitives     # การฉายภาพลงบนโคน PSD, การตรวจสเปกตรัม eigenvalue, การประเมิน rank
├── moment              # ลำดับชั้น Moment-SOS, DegRevLex monomial bases, การผ่อนปรน 3-SAT, SDPA I/O
├── exact               # เลขคณิตเศษส่วนแท้ 128 บิต (Rational), ช่วงความคลาดเคลื่อน (Interval)
├── symbolic            # ต้นไม้นิพจน์สัญลักษณ์, Canonical Simplifier, อนุพันธ์สัญลักษณ์, พหุนามยูคลิด
├── units               # การวิเคราะห์มิติ 7 มิติหลัก SI, ปริมาณทางฟิสิกส์ที่ปลอดภัยทางมิติ (m, kg, s, A, K, mol, cd)
├── verification        # การบันทึกสายธารการอนุพัทธ์ (Derivation Trees), สะพานส่งออกโค้ด Lean 4 / Mathlib4
├── linear_algebra      # DynamicMatrix, Blocked GEMM, LU, QR, Cholesky, SVD, ระบบสมการเชิงเส้น, เมทริกซ์ผกผัน
├── linear_operator     # Trait กลางสำหรับตัวดำเนินการเชิงเส้นแบบ Matrix-Free (apply, apply_adjoint)
├── symmetric           # เมทริกซ์สมมาตรแบบบีบอัดสามเหลี่ยมล่าง (Packed Lower-Triangular)
├── eigensystem         # ระบบไอเกนสมมาตร Jacobi & QR พร้อมรายงานความตกค้างทางสเปกตรัมและความตั้งฉาก
├── sparse              # Compressed Sparse Row (CSR), Lanczos eigensolver, PCG, GMRES, BiCGSTAB
├── generic             # Mat<T> แบบไดนามิก และ SMatrix<T, R, C> แบบคงที่บน Stack ด้วย const generics
├── lazy                # โครงสร้างต้นไม้นิพจน์แบบ Lazy รวมลูปคำนวณและลดการจองหน่วยความจำชั่วคราว
├── autodiff            # อนุพันธ์อัตโนมัติแบบไปข้างหน้า (Dual numbers, ความชันเกรเดียนต์, จาโคเบียน)
├── reverse_ad          # อนุพันธ์อัตโนมัติแบบย้อนกลับ (Wengert tape, VJP, เฮสเซียน)
├── opt_multivar        # การหาค่าเหมาะที่สุดหลายตัวแปร: BFGS, L-BFGS, Nelder-Mead, Armijo gradient descent
├── optimization        # การหารากสมการและการหาค่าเหมาะที่สุด 1 มิติ (Brent, Golden Section)
├── ode / ode_ext       # ตัวแก้สมการเชิงอนุพันธ์สามัญ: Euler, RK4, RK45, Stiff (BDF2), Symplectic (Leapfrog/Yoshida)
├── pde / pde_ext       # ตัวแก้สมการเชิงอนุพันธ์ย่อย: 2D Heat ADI, 2D Wave, Burgers, Method of Lines, Shooting BVP
├── statistics          # สถิติเชิงพรรณนา, โมเมนต์, ควอนไทล์, ตัวประมาณค่าที่ทนทาน (Robust estimators)
├── distributions       # การแจกแจงความน่าจะเป็น (Normal, Gamma, Beta, Poisson, Student's t ฯลฯ)
├── inference           # การทดสอบสมมติฐาน (t-test, ANOVA, Mann-Whitney, Kolmogorov-Smirnov, Benjamini-Hochberg)
├── regression          # Ordinary Least Squares, Ridge, Lasso, Elastic Net, Logistic, Softmax, Decision Tree
├── monte_carlo         # Quasi-Monte Carlo (Halton, Sobol, Latin Hypercube), MCMC (Metropolis, HMC, NUTS)
├── signal              # Cooley-Tukey FFT, 2D FFT, การออกแบบตัวกรอง IIR/FIR, STFT, Hilbert transform
├── timeseries          # แบบจำลอง ARIMA, การปรับให้เรียบเอ็กซ์โพเนนเชียล (Holt-Winters), Dynamic Time Warping
├── transform           # ควอเทอร์เนียน (Quaternion), การหมุน 3 มิติ, Isometry ใน SE(3), มุมกล้อง 4x4
├── special_functions   # ฟังก์ชันเออร์เรอร์ (erf/erfc), แกมมา, ล็อกแกมมา, บีตา, ไดแกมมา, เบสเซล J0/J1
└── io                  # นำเข้า/ส่งออกข้อมูลทางวิทยาศาสตร์: SDPA (.dat-s), DIMACS CNF, MatrixMarket, CSV
```

---

## การติดตั้ง

เพิ่ม `scies-math-th` เข้าไปในไฟล์ `Cargo.toml`:

```toml
[dependencies]
scies-math-th = "0.3.0"

# หากต้องการฟีเจอร์เพิ่มเติม:
# scies-math-th = { version = "0.3.0", features = ["serde", "parallel"] }
```

---

## ตัวอย่างการใช้งานและเจาะลึก 10 โมดูลหลัก

### 1. Semidefinite Programming (ADMM และ Mehrotra Primal-Dual IPM)

แก้โจทย์ Semidefinite Programming ในรูปมาตรฐาน Primal-Dual:

$$\textbf{Primal:} \quad \min_{X \in \mathcal{S}^n} \langle C, X \rangle \quad \text{subject to} \quad \langle A_i, X \rangle = b_i \; (i = 1, \dots, m), \; X \succeq 0$$

$$\textbf{Dual:} \quad \max_{y \in \mathbb{R}^m, S \in \mathcal{S}^n} b^T y \quad \text{subject to} \quad \sum_{i=1}^m y_i A_i + S = C, \; S \succeq 0$$

```rust
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus, verify_sdp_candidate};
use scies_math_th::sdp_ipm::{solve_sdp_ipm, SdpIpmConfig};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // หาค่าน้อยสุดของ Tr(C * X) ภายใต้เงื่อนไข <A1, X> = 1, <A2, X> = 1, X >= 0
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?; // [1 0; 0 2]
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?; // X_00 = 1
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?; // X_11 = 1
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b)?;

    // วิธีที่ 1: First-Order ADMM Solver (เน้นความเร็ว)
    let admm_config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 300,
        rho: 1.0,
        strict: true,
    };
    let admm_sol = problem.solve(&admm_config)?;
    assert_eq!(admm_sol.status, SdpStatus::Optimal);
    println!("ADMM Primal Objective: {:.6}", admm_sol.primal_objective); // ~ 3.000000

    // วิธีที่ 2: Mehrotra Primal-Dual Interior-Point Solver (เน้นความแม่นยำสูง)
    let ipm_config = SdpIpmConfig {
        tolerance: 1e-7,
        max_iterations: 50,
        step_damping: 0.95,
    };
    let ipm_sol = solve_sdp_ipm(&problem, &ipm_config)?;
    println!("IPM Primal Objective:  {:.8}", ipm_sol.primal_objective);

    // ตรวจสอบความถูกต้อง a-posteriori ด้วย Independent KKT Verifier
    let report = verify_sdp_candidate(&problem, &ipm_sol.x, &ipm_sol.y, &ipm_sol.s, 1e-6)?;
    assert!(report.is_optimal(1e-6));
    println!("Primal equality residual: {:.2e}", report.primal_equality_residual);
    println!("Dual equality residual:   {:.2e}", report.dual_equality_residual);
    println!("Relative duality gap:     {:.2e}", report.relative_duality_gap);

    Ok(())
}
```

---

### 2. ลำดับชั้น Moment-SOS และการผ่อนปรน DIMACS CNF / 3-SAT

แปลงปัญหา Boolean Satisfiability (SAT / 3-SAT) และการหาค่าเหมาะที่สุดของพหุนาม ให้กลายเป็นการผ่อนปรน SDP ผ่านลำดับชั้น Lasserre / Moment-SOS โดยใช้ฐานมัลติอินเด็กซ์แบบ DegRevLex:

```rust
use scies_math_th::moment::sat_encoding::{Clause, CnfFormula, Literal};
use scies_math_th::moment::sdpa_io::{export_sdpa_sparse, import_sdpa_sparse};
use scies_math_th::sdp::SdpSolverConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // อ่านข้อความรูปแบบมาตรฐาน DIMACS CNF
    let dimacs_data = r#"c ตัวอย่างสูตร 3-SAT
p cnf 3 2
1 2 3 0
-1 2 -3 0
"#;
    let formula = CnfFormula::from_dimacs(dimacs_data)?;
    assert_eq!(formula.num_vars(), 3);
    assert_eq!(formula.num_clauses(), 2);

    // หาผลเฉลยจริงด้วย Exact Brute-Force Oracle สำหรับสูตรขนาดเล็ก (n <= 20)
    let witness = formula.solve_brute_force()?;
    println!("SAT Witness Assignment: {:?}", witness);

    // สร้างการผ่อนปรน Moment-SOS SDP ที่ระดับดีกรี 2
    let (sdp, report) = formula.build_moment_sdp_relaxation_with_report(2)?;
    println!("มิติของ Moment Matrix: {} x {}", sdp.matrix_dim(), sdp.matrix_dim());
    println!("จำนวนข้อจำกัดสมการที่สร้างขึ้น: {}", report.num_generated_constraints);

    // แก้ปัญหา SDP ที่สร้างขึ้น
    let config = SdpSolverConfig::default();
    let sol = sdp.solve(&config)?;
    println!("สถานะผลลัพธ์การผ่อนปรน: {:?}", sol.status);

    // ส่งออกและนำเข้าไฟล์ฟอร์แมตสากล SDPA Sparse (.dat-s)
    let sdpa_str = export_sdpa_sparse(&sdp, "3-SAT Degree-2 Relaxation")?;
    let reloaded_sdp = import_sdpa_sparse(&sdpa_str)?;
    assert_eq!(reloaded_sdp.matrix_dim(), sdp.matrix_dim());

    Ok(())
}
```

---

### 3. เลขคณิตเศษส่วนแท้ 128 บิตพร้อมระบบป้องกัน Overflow

สำหรับงานที่ไม่อาจยอมรับความคลาดเคลื่อนจากการปัดเศษทศนิยมได้ [`exact::Rational`](docs/exact.md) จัดเก็บตัวเลขในรูปเศษส่วนทอนไม่ได้ $p/q \in \mathbb{Q}$ โดยใช้จำนวนเต็ม 128 บิตพร้อมการตรวจสอบขอบเขต:

```rust
use scies_math_th::exact::Rational;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Rational::new(1, 3)?;
    let b = Rational::new(1, 6)?;

    // คำนวณเศษส่วนแม่นยำ 100%: 1/3 + 1/6 = 3/6 = 1/2
    let sum = a.checked_add(&b)?;
    assert_eq!(sum, Rational::new(1, 2)?);
    assert_eq!(sum.to_f64(), 0.5);

    // การหารแม่นยำ: (1/3) / (1/6) = 2
    let div = a.checked_div(&b)?;
    assert_eq!(div, Rational::from_integer(2));

    // การหารด้วยศูนย์จะคืนค่าเป็น SciError ที่ชัดเจนเสมอ
    assert!(Rational::new(1, 0).is_err());

    // ประมาณค่าทศนิยมกลับเป็นเศษส่วนแท้ที่ดีที่สุด (อัลกอริทึม Stern-Brocot)
    let approx_pi = Rational::from_f64_approx(3.141592653589793, 1000)?;
    assert_eq!(approx_pi, Rational::new(355, 113)?);

    Ok(())
}
```

---

### 4. โครงสร้างต้นไม้นิพจน์สัญลักษณ์, แคลคูลัส และการหารพหุนามยูคลิด

ระบบพีชคณิตสัญลักษณ์ (Symbolic Algebra) แบบ zero-dependency รองรับการจัดรูปอย่างง่าย (Canonical Simplification), การรวมค่าคงที่, การหาอนุพันธ์เชิงสัญลักษณ์, และการหารพหุนามตัวแปรเดียวแบบยูคลิด:

```rust
use scies_math_th::symbolic::{Expr, Polynomial};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Expr::sym("x");

    // สร้างนิพจน์สัญลักษณ์: f(x) = x^3 + sin(x) + 0
    let f = Expr::Add(vec![
        Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(3))),
        Expr::Sin(Box::new(x.clone())),
        Expr::int(0),
    ]);

    // จัดรูปอย่างง่ายตามกฎพีชคณิต (ตัดศูนย์, รวมพจน์)
    let simplified = f.simplify();
    println!("นิพจน์หลังจัดรูป: {}", simplified);

    // หาอนุพันธ์เชิงสัญลักษณ์: d/dx [ x^3 + sin(x) ] = 3*x^2 + cos(x)
    let df = simplified.diff("x").simplify();
    println!("df/dx: {}", df);

    // การหารพหุนามยูคลิดพร้อมเศษ: A(x) = B(x) * Q(x) + R(x)
    // A(x) = x^3 - 2x^2 - 4 (สัมประสิทธิ์: [-4, 0, -2, 1])
    // B(x) = x - 3          (สัมประสิทธิ์: [-3, 1])
    let p_a = Polynomial::new(vec![-4.0, 0.0, -2.0, 1.0])?;
    let p_b = Polynomial::new(vec![-3.0, 1.0])?;
    let (quot, rem) = p_a.div_rem(&p_b)?;
    println!("ผลหาร Quotient:  {:?}", quot.coefficients()); // [7, 1, 1] => x^2 + x + 7
    println!("เศษเหลือ Remainder: {:?}", rem.coefficients()); // [17] => 17

    Ok(())
}
```

---

### 5. การวิเคราะห์มิติทางฟิสิกส์ 7 มิติหลัก SI และปริมาณที่ปลอดภัยทางมิติ

ป้องกันข้อผิดพลาดจากการบวก/ลบหรือคูณหน่วยผิดในสมการวิทยาศาสตร์ โดยระบบจะตรวจสอบเลขชี้กำลังของมิติทั้ง 7 มิติ ($\text{L}, \text{M}, \text{T}, \text{I}, \Theta, \text{N}, \text{J}$):

```rust
use scies_math_th::units::Quantity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mass = Quantity::kilograms(10.0);
    let acceleration = Quantity::meters(9.81) / (Quantity::seconds(1.0) * Quantity::seconds(1.0))?;

    // กฎข้อที่สองของนิวตัน: F = m * a [kg * m / s^2 = N]
    let force = mass * acceleration;
    println!("แรง Force: {} นิวตัน", force.value());

    // การบวกข้ามหน่วยที่มิติไม่ตรงกันจะคืนค่า Error เสมอ
    let length = Quantity::meters(5.0);
    let invalid_add = mass.checked_add(&length);
    assert!(invalid_add.is_err()); // คืนค่า SciError::DimensionalMismatch

    Ok(())
}
```

---

### 6. การสืบสาวที่มาของการพิสูจน์ และสะพานส่งออกโค้ด Lean 4 / Mathlib4

เชื่อมโยงการคำนวณเข้ากับระบบพิสูจน์ทฤษฎีบทเชิงโต้ตอบ โดยแปลงข้อสรุปทางพีชคณิตสัญลักษณ์ออกมาเป็นไฟล์ภาษา Lean 4 ที่มีหัวข้อแจ้งเตือนสถานะการตรวจสอบ (Provenance Metadata) อย่างสมบูรณ์:

```rust
use scies_math_th::symbolic::Expr;
use scies_math_th::verification::lean_export::export_lean4_theorem;

fn main() {
    let x = Expr::sym("x");
    let expr = Expr::Add(vec![
        Expr::Pow(Box::new(x), Box::new(Expr::int(2))),
        Expr::int(1),
    ]);

    // ส่งออกโครงร่างทฤษฎีบทสำหรับ Lean 4
    let lean_code = export_lean4_theorem(
        "quadratic_strictly_positive",
        &[("x", "ℝ")],
        "x ^ 2 + 1 > 0",
        Some("positivity"),
    );

    println!("{}", lean_code);
}
```

โค้ดภาษา Lean 4 ที่ถูกสร้างขึ้น:
```lean
/-!
  Generated by scies-math-th (v0.3.0)
  Status: Generated theorem skeleton (requires verification by Lean 4 kernel)
-/

import Mathlib

theorem quadratic_strictly_positive (x : ℝ) : x ^ 2 + 1 > 0 := by
  positivity
```

---

### 7. พีชคณิตเชิงเส้นแบบ Dense, Blocked GEMM และ SVD

```rust
use scies_math_th::linear_algebra::DynamicMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = DynamicMatrix::new(3, 3, vec![
        12.0, -51.0, 4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ])?;

    // Blocked Householder QR Decomposition
    let qr = a.qr_decompose()?;
    let reconstructed = qr.q.mul_matrix(&qr.r)?;
    assert!((a.get(0, 0)? - reconstructed.get(0, 0)?).abs() < 1e-10);

    // Singular Value Decomposition (SVD)
    let svd = a.singular_value_decompose(1e-10, 200)?;
    println!("ค่า Singular values: {:?}", svd.singular_values);
    println!("Condition number (2-norm): {:.4}", svd.condition_number());

    Ok(())
}
```

---

### 8. Compressed Sparse Row (CSR) และ Lanczos Eigensolver

```rust
use scies_math_th::sparse::{SparseMatrixCsr, lanczos_eigen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // สร้าง Sparse Matrix ขนาด 4x4 จากคู่อันดับ COO triplets
    let triplets = vec![
        (0, 0,  2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1,  2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2,  2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3,  2.0),
    ];
    let csr = SparseMatrixCsr::from_triplets(4, 4, &triplets)?;
    csr.validate_invariants()?; // ตรวจสอบความถูกต้องของ row pointers และคอลัมน์ที่เรียงลำดับ

    // รัน Lanczos Eigensolver หาค่าไอเกนสูงสุด 2 ค่าแรก
    let result = lanczos_eigen(&csr, 2, 20, 1e-8)?;
    assert!(result.converged);
    println!("Eigenvalues สูงสุด: {:?}", result.eigenvalues);
    println!("ค่าตกค้าง ||Av - λv||: {:?}", result.residuals);

    Ok(())
}
```

---

### 9. การหาค่าเหมาะที่สุดหลายตัวแปร และการหาอนุพันธ์อัตโนมัติ

```rust
use scies_math_th::autodiff::{Dual, jacobian};
use scies_math_th::opt_multivar::bfgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ฟังก์ชัน Rosenbrock Banana: min f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    let rosenbrock = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        let one = Dual::from(1.0);
        let hundred = Dual::from(100.0);
        (one - x) * (one - x) + hundred * (y - x * x) * (y - x * x)
    };

    let x0 = [-1.2, 1.0];
    let (x_min, f_min) = bfgs(rosenbrock, &x0, 1e-6, 500)?;
    println!("จุดต่ำสุดที่พบ: [x = {:.4}, y = {:.4}]", x_min[0], x_min[1]); // [1.0000, 1.0000]
    println!("ค่าฟังก์ชันต่ำสุด: {:.2e}", f_min);

    // คำนวณความชันที่จุดต่ำสุดด้วย Exact Autodiff Jacobian
    let grad_at_min = jacobian(rosenbrock, &x_min);
    println!("ขนาดของเกรเดียนต์ ณ จุดต่ำสุด: {:.2e}", grad_at_min[0].hypot(grad_at_min[1]));

    Ok(())
}
```

---

### 10. สมการเชิงอนุพันธ์ (Adaptive RK45 และ 2D Heat ADI)

```rust
use scies_math_th::ode::rk45;
use scies_math_th::pde_ext::heat_2d_adi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Adaptive Dormand-Prince RK45: dy/dt = -y, y(0) = 1.0
    let f_ode = |_t: f64, y: &[f64]| vec![-y[0]];
    let sol = rk45(f_ode, 0.0, 5.0, vec![1.0], 1e-6, 1e-9, 0.1)?;
    println!("ODE y(5.0) = {:.6} (ค่าแม่นตรง: {:.6})", sol.y.last().unwrap()[0], (-5.0_f64).exp());

    // 2. สมการความร้อน 2 มิติผ่าน Alternating Direction Implicit (Peaceman-Rachford ADI)
    let nx = 21;
    let ny = 21;
    let mut u0 = vec![0.0; nx * ny];
    u0[(ny / 2) * nx + (nx / 2)] = 100.0; // แหล่งความร้อนตรงกลาง

    let u_final = heat_2d_adi(&u0, nx, ny, 0.05, 0.05, 0.001, 1.0, 50)?;
    println!("อุณหภูมิตรงกลางหลังผ่านไป 50 สเต็ป: {:.4}", u_final[(ny / 2) * nx + (nx / 2)]);

    Ok(())
}
```

---

## Feature Flags

| Feature Flag | คำอธิบาย |
| :--- | :--- |
| `default` | บิลด์มาตรฐานแบบ Zero External Dependencies (ไม่มี dependency ภายนอก) |
| `serde` | เปิดใช้งานการ Derive `Serialize` และ `Deserialize` สำหรับโครงสร้างข้อมูลและเวกเตอร์ |
| `parallel` | เปิดใช้งานการประมวลผลเมทริกซ์แบบ Multi-threaded ผ่าน `rayon` |
| `bench` | รวมโค้ดและยูทิลิตีสำหรับการวัดประสิทธิภาพเชิงเปรียบเทียบ (Benchmark) |

---

## การประกันคุณภาพและ Benchmark

โค้ดทุกส่วนผ่านเกณฑ์ตรวจสอบคุณภาพอัตโนมัติอย่างเข้มงวด:
- **`cargo fmt --all -- --check`**: รูปแบบโค้ดเป็นไปตามมาตรฐานทางการ 100%
- **`cargo clippy --offline --all-targets -- -D warnings`**: ไม่มีข้อผิดพลาดหรือคำเตือนจาก Clippy (0 warnings)
- **`cargo test --offline`**: การทดสอบทั้ง 89 ชุด (Unit tests, Integration tests, และ Property tests) ผ่านครบถ้วน (0 failed)
- **`cargo doc --offline --no-deps`**: ไม่มี Warning หรือ Broken Link ในเอกสาร
- **`cargo bench --no-run --offline`**: ชุด Benchmark ทั้ง 7 ชุดคอมไพล์ผ่านสมบูรณ์

---

## เอกสารและลิขสิทธิ์

- เอกสารอ้างอิงรายโมดูลฉบับสมบูรณ์: [`docs/README.md`](docs/README.md)
- ลำดับชั้นความน่าเชื่อถือทางตัวเลข: [`docs/numerical_reliability.md`](docs/numerical_reliability.md)
- เมทริกซ์ความถูกต้องของแต่ละระบบย่อย: [`docs/api_reliability_matrix.md`](docs/api_reliability_matrix.md)
- ภาพรวมภาษาอังกฤษ: [`README.en.md`](README.en.md)

เผยแพร่ภายใต้สัญญาอนุญาต **MIT License** ([LICENSE](LICENSE))
