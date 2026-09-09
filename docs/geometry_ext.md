# Advanced Computational Geometry (`geometry_ext`)

The `geometry_ext` module provides 3D collision detection and spatial acceleration data structures for physics simulations, robotics, and computer graphics.

---

## 1. Gilbert-Johnson-Keerthi (GJK) Collision Detection

The GJK algorithm determines whether two arbitrary convex shapes $A$ and $B$ intersect by querying the **Minkowski Difference**:

$$A \ominus B = \{a - b \mid a \in A, \; b \in B\}$$

Two convex shapes intersect if and only if the origin $0 \in \mathbb{R}^3$ is contained within $A \ominus B$:

$$A \cap B \ne \emptyset \iff \mathbf{0} \in A \ominus B$$

### The `ConvexShape` Trait
Any shape that implements the support mapping function can be tested via GJK:

$$\text{support}_A(\mathbf{d}) = \arg\max_{a \in A} (a \cdot \mathbf{d})$$

The Minkowski support point is simply:

$$s_{A \ominus B}(\mathbf{d}) = \text{support}_A(\mathbf{d}) - \text{support}_B(-\mathbf{d})$$

Supported primitives: `Sphere`, `Aabb3`, and custom polyhedral convex hulls.

---

## 2. Bounding Volume Hierarchy (`BvhTree`)

Hierarchical binary tree of Axis-Aligned Bounding Boxes (AABB) providing $O(\log N)$ broad-phase collision queries and raycast intersections, compared to $O(N)$ brute-force checking.

---

## 3. Code Example

```rust
use scies_math_th::geometry::{Aabb3, Point3, Sphere};
use scies_math_th::geometry_ext::gjk_intersection;

fn main() {
    let sphere = Sphere::new(Point3::new(0.0, 0.0, 0.0), 1.0);

    // Overlapping box
    let box1 = Aabb3::new(Point3::new(0.5, 0.5, 0.5), Point3::new(2.0, 2.0, 2.0));
    assert!(gjk_intersection(&sphere, &box1));

    // Separated box
    let box2 = Aabb3::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));
    assert!(!gjk_intersection(&sphere, &box2));
}
```
