# เอกสารอ้างอิง: การวิเคราะห์มิติทางฟิสิกส์และหน่วยวัด (`units`)

โมดูล `units` ให้บริการระบบวิเคราะห์มิติทางฟิสิกส์ที่เข้มงวดตามมาตรฐานมิติฐานทั้ง 7 ของระบบหน่วยสากล (SI Base Dimensions) ช่วยป้องกันข้อผิดพลาดจากการคำนวณสมการฟิสิกส์ข้ามหน่วย

---

## 1. มิติฐานทั้ง 7 ของระบบ SI (`Dimension`)

มิติทางฟิสิกส์ใด ๆ จะถูกแทนด้วยทูเพิลของเลขชี้กำลังตรรกยะบนปริมาณฐานทั้ง 7:

$$\text{dim}(Q) = \text{L}^a \cdot \text{M}^b \cdot \text{T}^c \cdot \text{I}^d \cdot \Theta^e \cdot \text{N}^f \cdot \text{J}^g$$

| สัญลักษณ์ | ปริมาณฐาน (Base Quantity) | หน่วย SI | ค่าคงที่ในโมดูล |
| :--- | :--- | :--- | :--- |
| $\text{L}$ | ความยาว (Length) | เมตร ($\text{m}$) | `Dimension::LENGTH` |
| $\text{M}$ | มวล (Mass) | กิโลกรัม ($\text{kg}$) | `Dimension::MASS` |
| $\text{T}$ | เวลา (Time) | วินาที ($\text{s}$) | `Dimension::TIME` |
| $\text{I}$ | กระแสไฟฟ้า (Electric Current) | แอมแปร์ ($\text{A}$) | `Dimension::CURRENT` |
| $\Theta$ | อุณหภูมิอุณหพลวัต (Temperature) | เคลวิน ($\text{K}$) | `Dimension::TEMPERATURE` |
| $\text{N}$ | ปริมาณสาร (Amount of Substance) | โมล ($\text{mol}$) | `Dimension::AMOUNT` |
| $\text{J}$ | ความเข้มของการส่องสว่าง (Luminosity)| แคนเดลา ($\text{cd}$) | `Dimension::LUMINOUS_INTENSITY` |

---

## 2. กฎพีชคณิตเชิงมิติ (Dimensional Algebra)

- **การคูณ**: นำเลขชี้กำลังของแต่ละมิติมาบวกกัน: $\text{dim}(A \cdot B) = \text{dim}(A) + \text{dim}(B)$
- **การหาร**: นำเลขชี้กำลังของแต่ละมิติมาลบกัน: $\text{dim}(A / B) = \text{dim}(A) - \text{dim}(B)$
- **การยกกำลัง**: นำเลขชี้กำลังมาคูณด้วยสเกลาร์: $\text{dim}(A^k) = k \cdot \text{dim}(A)$
- **การบวกและลบ**: กระทำได้เฉพาะปริมาณที่มีมิติตรงกันทุกประการเท่านั้น หากมิติไม่ตรงกันระบบจะส่งกลับ:
  ```rust
  Err(SciError::DimensionalMismatch { expected, found })
  ```

---

## 3. ตัวอย่างการใช้งาน

```rust
use scies_math_th::units::Quantity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mass = Quantity::kilograms(2.5);
    let time = Quantity::seconds(0.5);
    let distance = Quantity::meters(10.0);

    // ความเร็ว: v = d / t [m / s]
    let velocity = (distance / time)?;
    assert_eq!(velocity.value(), 20.0);

    // โมเมนตัม: p = m * v [kg * m / s]
    let momentum = mass * velocity;
    assert_eq!(momentum.value(), 50.0);

    // พลังงานจลน์: E = 0.5 * m * v^2 [kg * m^2 / s^2 = Joules]
    let kinetic_energy = (mass * velocity * velocity)?.scale(0.5);
    assert_eq!(kinetic_energy.value(), 500.0);

    // การบวกมวลกับระยะทางจะล้มเหลวอย่างปลอดภัยทันที
    assert!(mass.checked_add(&distance).is_err());

    Ok(())
}
```
