//! Advanced Computational Geometry
//!
//! This module provides advanced geometric algorithms for 3D collision detection
//! and spatial partitioning, essential for physics engines and simulations.
//!
//! # Features
//! - **GJK Algorithm**: Gilbert-Johnson-Keerthi collision detection for any convex shape.
//! - **BVH (Bounding Volume Hierarchy)**: Spatial acceleration structure for fast raycasting and broad-phase collision.

use crate::errors::{SciError, SciResult};
use crate::geometry::{Aabb3, Point3, Ray3, Sphere};
use crate::linear_algebra::Vector3;

/// Trait for convex shapes that can compute a support point in a given direction.
/// This is the core requirement for the GJK algorithm.
pub trait ConvexShape {
    /// Returns the furthest point on the shape in the given direction.
    fn support(&self, direction: Vector3) -> Point3;
}

impl ConvexShape for Sphere {
    fn support(&self, direction: Vector3) -> Point3 {
        let dir_norm = direction.magnitude();
        if dir_norm <= f64::EPSILON {
            return self.center;
        }
        let normalized = Vector3::new(
            direction.x / dir_norm,
            direction.y / dir_norm,
            direction.z / dir_norm,
        );
        Point3::new(
            self.center.x + normalized.x * self.radius,
            self.center.y + normalized.y * self.radius,
            self.center.z + normalized.z * self.radius,
        )
    }
}

impl ConvexShape for Aabb3 {
    fn support(&self, direction: Vector3) -> Point3 {
        Point3::new(
            if direction.x > 0.0 { self.max.x } else { self.min.x },
            if direction.y > 0.0 { self.max.y } else { self.min.y },
            if direction.z > 0.0 { self.max.z } else { self.min.z },
        )
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GJK Algorithm
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
struct Simplex {
    points: [Vector3; 4],
    size: usize,
}

impl Simplex {
    fn new() -> Self {
        Self {
            points: [Vector3::new(0.0, 0.0, 0.0); 4],
            size: 0,
        }
    }

    fn push_front(&mut self, point: Vector3) {
        self.points = [point, self.points[0], self.points[1], self.points[2]];
        self.size = (self.size + 1).min(4);
    }
}

fn subtract_points(a: Point3, b: Point3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// Checks if two convex shapes intersect using the GJK algorithm.
pub fn gjk_intersect<A: ConvexShape, B: ConvexShape>(shape_a: &A, shape_b: &B) -> bool {
    // Initial direction
    let mut d = Vector3::new(1.0, 0.0, 0.0);
    
    // Get first support point for Minkowski difference
    let p_a = shape_a.support(d);
    let p_b = shape_b.support(d * -1.0);
    let mut a = subtract_points(p_a, p_b);
    
    let mut simplex = Simplex::new();
    simplex.push_front(a);
    
    d = Vector3::new(-a.x, -a.y, -a.z);
    
    for _ in 0..64 { // Max iterations to prevent infinite loops
        let p_a = shape_a.support(d);
        let p_b = shape_b.support(d * -1.0);
        a = subtract_points(p_a, p_b);
        
        if a.dot(d) < 0.0 {
            // The support point didn't pass the origin along direction d
            return false;
        }
        
        simplex.push_front(a);
        
        if handle_simplex(&mut simplex, &mut d) {
            return true;
        }
    }
    
    false
}

fn handle_simplex(simplex: &mut Simplex, d: &mut Vector3) -> bool {
    match simplex.size {
        2 => line_case(simplex, d),
        3 => triangle_case(simplex, d),
        4 => tetrahedron_case(simplex, d),
        _ => unreachable!(),
    }
}

fn line_case(simplex: &mut Simplex, d: &mut Vector3) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    
    let ab = Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z);
    let ao = Vector3::new(-a.x, -a.y, -a.z);
    
    if ab.dot(ao) > 0.0 {
        *d = ab.cross(ao).cross(ab);
        if d.magnitude() <= f64::EPSILON {
            *d = Vector3::new(1.0, 0.0, 0.0); // Fallback
        }
    } else {
        simplex.size = 1;
        *d = ao;
    }
    false
}

fn triangle_case(simplex: &mut Simplex, d: &mut Vector3) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];
    
    let ab = Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z);
    let ac = Vector3::new(c.x - a.x, c.y - a.y, c.z - a.z);
    let ao = Vector3::new(-a.x, -a.y, -a.z);
    
    let abc = ab.cross(ac);
    
    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            simplex.points[1] = c;
            simplex.size = 2;
            *d = ac.cross(ao).cross(ac);
        } else {
            simplex.size = 2;
            return line_case(simplex, d);
        }
    } else {
        if ab.cross(abc).dot(ao) > 0.0 {
            simplex.size = 2;
            return line_case(simplex, d);
        } else {
            if abc.dot(ao) > 0.0 {
                *d = abc;
            } else {
                simplex.points = [a, c, b, simplex.points[3]];
                *d = abc * -1.0;
            }
        }
    }
    false
}

fn tetrahedron_case(simplex: &mut Simplex, d: &mut Vector3) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];
    let d_pt = simplex.points[3];
    
    let ab = Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z);
    let ac = Vector3::new(c.x - a.x, c.y - a.y, c.z - a.z);
    let ad = Vector3::new(d_pt.x - a.x, d_pt.y - a.y, d_pt.z - a.z);
    let ao = Vector3::new(-a.x, -a.y, -a.z);
    
    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);
    
    if abc.dot(ao) > 0.0 {
        simplex.size = 3;
        simplex.points = [a, b, c, d_pt];
        return triangle_case(simplex, d);
    }
    
    if acd.dot(ao) > 0.0 {
        simplex.size = 3;
        simplex.points = [a, c, d_pt, b];
        return triangle_case(simplex, d);
    }
    
    if adb.dot(ao) > 0.0 {
        simplex.size = 3;
        simplex.points = [a, d_pt, b, c];
        return triangle_case(simplex, d);
    }
    
    true // Origin is inside the tetrahedron
}

// ══════════════════════════════════════════════════════════════════════════════
// Bounding Volume Hierarchy (BVH)
// ══════════════════════════════════════════════════════════════════════════════

/// A node in the Bounding Volume Hierarchy.
#[derive(Debug, Clone)]
pub struct BvhNode<T> {
    pub aabb: Aabb3,
    pub payload: Option<T>,
    pub left: Option<Box<BvhNode<T>>>,
    pub right: Option<Box<BvhNode<T>>>,
}

impl<T: Clone> BvhNode<T> {
    /// Builds a BVH tree from a list of items and their corresponding AABBs.
    pub fn build(mut items: Vec<(Aabb3, T)>) -> SciResult<Option<Box<Self>>> {
        if items.is_empty() {
            return Ok(None);
        }
        
        if items.len() == 1 {
            let (aabb, payload) = items.pop().unwrap();
            return Ok(Some(Box::new(BvhNode {
                aabb,
                payload: Some(payload),
                left: None,
                right: None,
            })));
        }
        
        // Compute bounding box for all elements
        let mut min_pt = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_pt = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        
        for (aabb, _) in &items {
            min_pt.x = min_pt.x.min(aabb.min.x);
            min_pt.y = min_pt.y.min(aabb.min.y);
            min_pt.z = min_pt.z.min(aabb.min.z);
            max_pt.x = max_pt.x.max(aabb.max.x);
            max_pt.y = max_pt.y.max(aabb.max.y);
            max_pt.z = max_pt.z.max(aabb.max.z);
        }
        
        let node_aabb = Aabb3::new(min_pt, max_pt)?;
        
        // Find longest axis
        let dx = max_pt.x - min_pt.x;
        let dy = max_pt.y - min_pt.y;
        let dz = max_pt.z - min_pt.z;
        
        let axis = if dx > dy && dx > dz {
            0
        } else if dy > dz {
            1
        } else {
            2
        };
        
        // Sort items by the center of their AABB along the longest axis
        items.sort_by(|(a, _), (b, _)| {
            let ca = match axis {
                0 => a.min.x + a.max.x,
                1 => a.min.y + a.max.y,
                _ => a.min.z + a.max.z,
            };
            let cb = match axis {
                0 => b.min.x + b.max.x,
                1 => b.min.y + b.max.y,
                _ => b.min.z + b.max.z,
            };
            ca.partial_cmp(&cb).unwrap_or(core::cmp::Ordering::Equal)
        });
        
        let mid = items.len() / 2;
        let right_items = items.split_off(mid);
        let left_items = items;
        
        Ok(Some(Box::new(BvhNode {
            aabb: node_aabb,
            payload: None,
            left: Self::build(left_items)?,
            right: Self::build(right_items)?,
        })))
    }
    
    /// Queries the BVH for all items whose AABB intersects with the given AABB.
    pub fn query_aabb<'a>(&'a self, query: Aabb3, results: &mut Vec<&'a T>) {
        if !self.aabb.intersects(query) {
            return;
        }
        
        if let Some(ref payload) = self.payload {
            results.push(payload);
        }
        
        if let Some(ref left) = self.left {
            left.query_aabb(query, results);
        }
        
        if let Some(ref right) = self.right {
            right.query_aabb(query, results);
        }
    }
}
