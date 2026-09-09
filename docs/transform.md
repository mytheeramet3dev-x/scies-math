# 3D Geometric Transforms and Quaternions (`transform`)

The `transform` module provides representations for 3D spatial rotations, rigid body isometries ($\text{SE}(3)$), spherical linear interpolation (SLERP), and 4x4 projective transformation matrices.

---

## 1. Unit Quaternions (`Quaternion`)

Represents a rotation in $\mathbb{R}^3$ via a hypercomplex number $q = w + x i + y j + z k$:

$$q = \cos\left(\frac{\theta}{2}\right) + \mathbf{u} \sin\left(\frac{\theta}{2}\right)$$

where $\mathbf{u} \in \mathbb{R}^3$ ($\|\mathbf{u}\| = 1$) is the rotation axis and $\theta$ is the rotation angle.

### Hamilton Product
$$q_1 \otimes q_2 = (w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2) + (w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2)$$

### Vector Rotation
A 3D vector $v \in \mathbb{R}^3$ is rotated without gimbal lock via:

$$v_{\text{rot}} = q \otimes v_{\text{pure}} \otimes q^{-1}$$

### Spherical Linear Interpolation (`slerp`)
Interpolates between orientations along the shortest great-circle arc with constant angular velocity:

$$\text{SLERP}(q_1, q_2; t) = \frac{\sin((1 - t) \Omega)}{\sin(\Omega)} q_1 + \frac{\sin(t \Omega)}{\sin(\Omega)} q_2, \quad \cos(\Omega) = q_1 \cdot q_2$$

---

## 2. Rigid Isometries in $\text{SE}(3)$ (`Isometry3`)

Combines a unit quaternion rotation $R \in \text{SO}(3)$ and translation vector $t \in \mathbb{R}^3$:

$$T(x) = R x + t$$

Group composition:

$$T_1 \circ T_2 = \langle R_1 R_2, R_1 t_2 + t_1 \rangle, \quad T^{-1} = \langle R^{-1}, -R^{-1} t \rangle$$

---

## 3. 4x4 Projective Transforms (`Transform3D`)

- `Transform3D::look_at(eye, target, up)`: View transform matrix.
- `Transform3D::perspective(fov_y, aspect, near, far)`: Frustum perspective projection.
- `Transform3D::orthographic(left, right, bottom, top, near, far)`: Orthographic projection.

---

## 4. Code Example

```rust
use scies_math_th::transform::{Isometry3, Quaternion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 90-degree rotation about the Z-axis
    let axis = [0.0, 0.0, 1.0];
    let angle = std::f64::consts::FRAC_PI_2;
    let q = Quaternion::from_axis_angle(&axis, angle)?;

    // Rotate unit X vector [1, 0, 0] -> Expect [0, 1, 0]
    let v = [1.0, 0.0, 0.0];
    let v_rot = q.rotate_vector(&v);
    println!("Rotated vector: [{:.4}, {:.4}, {:.4}]", v_rot[0], v_rot[1], v_rot[2]);

    // Rigid transformation: rotate 90 deg about Z, then translate by [0, 0, 5]
    let iso = Isometry3::new(q, [0.0, 0.0, 5.0]);
    let v_trans = iso.transform_point(&v);
    println!("Transformed point: [{:.4}, {:.4}, {:.4}]", v_trans[0], v_trans[1], v_trans[2]); // [0, 1, 5]

    Ok(())
}
```
