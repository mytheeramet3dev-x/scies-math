# scies-math-th

`scies-math-th` คือชุดเครื่องมือคณิตศาสตร์สำหรับ Rust ที่ออกแบบมาให้ใช้งานจริงได้กับงานวิทยาศาสตร์และวิศวกรรม โดยยังคงหลักการสำคัญคือค่าเริ่มต้นต้องเบา ไม่ดึง dependency ที่ไม่จำเป็นเข้ามา

โปรเจกต์นี้ถูกวางให้เป็นรากฐานระยะยาวสำหรับงานคำนวณเชิงวิทยาศาสตร์ ไม่ได้ตั้งใจแก้โจทย์เชิงตัวเลขเพียงชนิดเดียว แต่ตั้งใจเป็นชุด primitive กลางที่โมดูลอื่นในสายวิทยาศาสตร์ ชีววิทยาคำนวณ และงานวิศวกรรมสามารถพึ่งพาได้อย่างมั่นคง

## โปรเจกต์นี้มีไว้เพื่ออะไร

- primitive ทางคณิตศาสตร์สำหรับงาน scientific computing
- พีชคณิตเชิงเส้นและ workflow ของเมทริกซ์
- สถิติ ความน่าจะเป็น และ inference
- วิธีเชิงตัวเลข เช่น optimization, interpolation, ODE และ PDE
- การประมวลผลสัญญาณและอนุกรมเวลา
- เครื่องมือสนับสนุนงานวิทยาศาสตร์ เช่น random sampling, autodiff, geometry และ graph algorithms

## หลักการออกแบบ

- ค่าเริ่มต้นต้องเป็น `zero dependency`
- API สาธารณะควร generic ในจุดที่เหมาะสม
- ข้อมูลขนาดคงที่ควรใช้ const generics เมื่อเป็นไปได้
- งานข้อมูลขนาดใหญ่ควรใช้แนวคิดแบบ lazy หรือ streaming
- ส่วนที่เป็น hot path ต้องมี benchmark
- เอกสารต้องครบพอให้เรียนรู้จาก docs ได้จริง
- ฟังก์ชันคณิตศาสตร์ที่ใช้ร่วมกันควรใช้ซ้ำจาก `scies-math-th` เอง ไม่ควรเขียนซ้ำที่อื่น

## ภาพรวมสถาปัตยกรรม

โครงของ crate แบ่งเป็นชั้นหลัก ๆ ดังนี้

### 1. ชั้นชนิดข้อมูลแกนกลาง

โมดูล `generic`, `complex`, `tensor`, และ `lazy` เป็นพื้นฐานของ container เชิงตัวเลขและ expression building blocks ที่ใช้ทั้งโปรเจกต์

### 2. ชั้นพีชคณิตเชิงเส้น

โมดูล `linear_algebra`, `transform`, `geometry_ext`, และ `sparse` ครอบคลุมการจัดการเมทริกซ์, transform, decomposition และเครื่องมือเรขาคณิต

### 3. ชั้นสถิติ

โมดูล `statistics`, `probability`, `distributions`, `inference`, `regression`, และ `monte_carlo` ใช้รองรับ workflow ทางสถิติทั่วไป

### 4. ชั้นวิธีเชิงตัวเลข

โมดูล `optimization`, `interpolation`, `ode`, `pde`, และ `special_functions` ใช้สำหรับอัลกอริทึมเชิงตัวเลขและฟังก์ชันพิเศษ

### 5. ชั้น workflow ทางวิทยาศาสตร์

โมดูล `signal`, `timeseries`, `autodiff`, `rng_ext`, `graph`, `nn`, และ `io` รองรับงานวิทยาศาสตร์และการจัดการข้อมูลที่กว้างขึ้น

## Feature flags

### `serde`

เปิดการใช้งาน `Serialize` และ `Deserialize` สำหรับชนิดข้อมูลสาธารณะที่รองรับ

### `parallel`

เปิด implementation แบบขนานในส่วนที่ crate รองรับ

### `bench`

เปิดส่วนช่วยสำหรับ benchmark และเส้นทางที่ใช้กับการวัดประสิทธิภาพ

ค่าเริ่มต้นของ feature คือว่างทั้งหมด

## ตัวตนของแพ็กเกจ

- ชื่อบน crates.io: `scies-math-th`
- Rust import path: `scies_math_th`
- Repository: `https://github.com/mytheeramet3dev-x/scies-math-th`
- โฟลเดอร์เอกสารโมดูล: `docs/`

## โครงสร้างแพ็กเกจ

```text
scies_math_th::
├── generic         # Mat<T>, SMatrix<T, R, C>, scalar traits
├── lazy            # Lazy expression trees สำหรับการคำนวณเชิงเมทริกซ์
├── complex         # การคำนวณจำนวนเชิงซ้อน
├── tensor          # Tensor container และ decomposition helpers
├── linear_algebra  # เมทริกซ์, เวกเตอร์, decomposition, solvers
├── transform       # Quaternion, rotation และ rigid transform
├── geometry_ext    # เครื่องมือเรขาคณิตขั้นสูง
├── sparse          # sparse matrix และ sparse solver utilities
├── statistics      # descriptive statistics
├── probability     # เครื่องมือความน่าจะเป็น
├── distributions   # probability distributions
├── inference       # hypothesis testing และ inference helpers
├── regression      # linear และ regularized regression
├── monte_carlo     # Monte Carlo integration และ sampling
├── optimization    # scalar optimization
├── opt_multivar    # multivariate optimization
├── interpolation   # interpolation utilities
├── ode             # ordinary differential equation solvers
├── ode_ext         # extended ODE routines
├── pde             # partial differential equation solvers
├── pde_ext         # extended PDE routines
├── special_functions # Erf, Gamma, Beta, Bessel และฟังก์ชันใกล้เคียง
├── signal          # FFT และ signal processing utilities
├── timeseries      # ARIMA, smoothing, DTW
├── autodiff        # forward-mode automatic differentiation
├── reverse_ad      # reverse-mode automatic differentiation
├── rng             # core RNG utilities
├── rng_ext         # RNG engines และ sampler APIs
├── graph           # graph algorithms
├── nn              # neural-network primitives
└── io              # scientific I/O helpers
```

## ตัวอย่าง

### สร้างเมทริกซ์แบบ generic

```rust
use scies_math_th::generic::{Mat, SMatrix};

fn main() {
    let a = Mat::<f64>::from_fn(2, 2, |r, c| (r + c) as f64);
    let eye: SMatrix<f64, 2, 2> = SMatrix::identity();
    let _ = (a, eye);
}
```

### ใช้ lazy evaluation

```rust
use scies_math_th::generic::Mat;
use scies_math_th::lazy::lazy;

fn main() {
    let a = Mat::<f64>::from_fn(4, 4, |r, c| (r + c) as f64);
    let result = (lazy(&a) + lazy(&a)).scale(0.5).eval().unwrap();
    let _ = result;
}
```

## การติดตั้ง

```toml
[dependencies]
scies-math-th = "0.2.4"

# ถ้าต้องการ serde:
scies-math-th = { version = "0.2.4", features = ["serde"] }
```

## Benchmark

การ benchmark เป็นส่วนหนึ่งของแนวคิดของโปรเจกต์ ไม่ใช่สิ่งที่ค่อยเติมทีหลัง

ชุด benchmark หลักอยู่ใน `benches/` และตั้งใจใช้ติดตามงานประเภท:

- matrix multiplication
- RNG และ sampling
- ODE solvers
- transform และ decomposition workflows

## เอกสาร

- [สารบัญโมดูล](docs/README.md)
- [English overview](README.en.md)
- [หน้าหลัก](README.md)

หน้าเอกสารระดับโมดูลควรอ่านได้ด้วยตัวเอง และควรมีตัวอย่างทุกครั้งที่ API นั้นเป็น public และไม่ trivial

## หมายเหตุการเผยแพร่

เวอร์ชัน `0.2.4` ยังรักษา default build แบบ zero-dependency เอาไว้ แต่ขยาย public surface และคุณภาพของเอกสารให้แน่นขึ้น รุ่นถัดไปควรเติบโตเป็นกลุ่มโมดูล ไม่ใช่เพิ่มแบบกระจัดกระจาย

