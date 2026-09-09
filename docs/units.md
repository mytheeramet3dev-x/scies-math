# Dimensional Analysis and Physical Units (`units`)

The `units` module provides rigorous dimensional analysis based on the 7 fundamental SI base dimensions, preventing unit mismatches in scientific equations.

---

## 1. The SI Base Dimensions (`Dimension`)

A physical dimension is represented as a rational exponent tuple across the 7 SI base quantities:

$$\text{dim}(Q) = \text{L}^a \cdot \text{M}^b \cdot \text{T}^c \cdot \text{I}^d \cdot \Theta^e \cdot \text{N}^f \cdot \text{J}^g$$

| Symbol | Base Quantity | SI Unit | Dimension Constant |
| :--- | :--- | :--- | :--- |
| $\text{L}$ | Length | meter ($\text{m}$) | `Dimension::LENGTH` |
| $\text{M}$ | Mass | kilogram ($\text{kg}$) | `Dimension::MASS` |
| $\text{T}$ | Time | second ($\text{s}$) | `Dimension::TIME` |
| $\text{I}$ | Electric Current | ampere ($\text{A}$) | `Dimension::CURRENT` |
| $\Theta$ | Thermodynamic Temperature | kelvin ($\text{K}$) | `Dimension::TEMPERATURE` |
| $\text{N}$ | Amount of Substance | mole ($\text{mol}$) | `Dimension::AMOUNT` |
| $\text{J}$ | Luminous Intensity | candela ($\text{cd}$) | `Dimension::LUMINOUS_INTENSITY` |

---

## 2. Dimensional Algebra

- **Multiplication**: Dimensions add exponents: $\text{dim}(A \cdot B) = \text{dim}(A) + \text{dim}(B)$
- **Division**: Dimensions subtract exponents: $\text{dim}(A / B) = \text{dim}(A) - \text{dim}(B)$
- **Powers**: Dimensions multiply exponents: $\text{dim}(A^k) = k \cdot \text{dim}(A)$
- **Addition / Subtraction**: Only permitted between quantities with identical dimensions. Attempting to add length to mass returns:
  ```rust
  Err(SciError::DimensionalMismatch { expected, found })
  ```

---

## 3. Physical Quantities (`Quantity`)

A `Quantity` pairs a scalar floating-point value with a `Dimension`:

```rust
use scies_math_th::units::Quantity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mass = Quantity::kilograms(2.5);
    let time = Quantity::seconds(0.5);
    let distance = Quantity::meters(10.0);

    // Velocity: v = d / t [m / s]
    let velocity = (distance / time)?;
    assert_eq!(velocity.value(), 20.0);

    // Momentum: p = m * v [kg * m / s]
    let momentum = mass * velocity;
    assert_eq!(momentum.value(), 50.0);

    // Kinetic Energy: E = 0.5 * m * v^2 [kg * m^2 / s^2 = Joules]
    let kinetic_energy = (mass * velocity * velocity)?.scale(0.5);
    assert_eq!(kinetic_energy.value(), 500.0);

    // Adding incompatible units triggers checked error
    assert!(mass.checked_add(&distance).is_err());

    Ok(())
}
```
