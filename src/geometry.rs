use crate::errors::{SciError, SciResult};
use crate::linear_algebra::{Vector2, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn to_vector(self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    pub fn to_vector(self) -> Vector3 {
        Vector3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegment2 {
    pub start: Point2,
    pub end: Point2,
}

impl LineSegment2 {
    pub const fn new(start: Point2, end: Point2) -> Self {
        Self { start, end }
    }

    pub fn length(self) -> f64 {
        self.start.distance(self.end)
    }

    pub fn midpoint(self) -> Point2 {
        Point2::new(
            0.5 * (self.start.x + self.end.x),
            0.5 * (self.start.y + self.end.y),
        )
    }

    pub fn distance_to_point(self, point: Point2) -> f64 {
        let ab = Vector2::new(self.end.x - self.start.x, self.end.y - self.start.y);
        let ap = Vector2::new(point.x - self.start.x, point.y - self.start.y);
        let denom = ab.dot(ab);
        if denom <= f64::EPSILON {
            return self.start.distance(point);
        }
        let t = (ap.dot(ab) / denom).clamp(0.0, 1.0);
        let projected = Point2::new(self.start.x + ab.x * t, self.start.y + ab.y * t);
        projected.distance(point)
    }

    pub fn intersects(self, other: Self) -> bool {
        fn orientation(a: Point2, b: Point2, c: Point2) -> f64 {
            (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        }

        fn on_segment(a: Point2, b: Point2, c: Point2) -> bool {
            b.x >= a.x.min(c.x) - 1e-12
                && b.x <= a.x.max(c.x) + 1e-12
                && b.y >= a.y.min(c.y) - 1e-12
                && b.y <= a.y.max(c.y) + 1e-12
        }

        let o1 = orientation(self.start, self.end, other.start);
        let o2 = orientation(self.start, self.end, other.end);
        let o3 = orientation(other.start, other.end, self.start);
        let o4 = orientation(other.start, other.end, self.end);

        if o1.abs() <= 1e-12 && on_segment(self.start, other.start, self.end) {
            return true;
        }
        if o2.abs() <= 1e-12 && on_segment(self.start, other.end, self.end) {
            return true;
        }
        if o3.abs() <= 1e-12 && on_segment(other.start, self.start, other.end) {
            return true;
        }
        if o4.abs() <= 1e-12 && on_segment(other.start, self.end, other.end) {
            return true;
        }

        (o1 > 0.0) != (o2 > 0.0) && (o3 > 0.0) != (o4 > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Point2,
    pub radius: f64,
}

impl Circle {
    pub fn new(center: Point2, radius: f64) -> SciResult<Self> {
        if radius < 0.0 {
            return Err(SciError::InvalidParameter("radius must be non-negative"));
        }
        Ok(Self { center, radius })
    }

    pub fn contains(self, point: Point2) -> bool {
        self.center.distance(point) <= self.radius + 1e-12
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray3 {
    pub origin: Point3,
    pub direction: Vector3,
}

impl Ray3 {
    pub fn new(origin: Point3, direction: Vector3) -> SciResult<Self> {
        if direction.magnitude() <= f64::EPSILON {
            return Err(SciError::InvalidParameter("direction must be non-zero"));
        }
        Ok(Self { origin, direction })
    }

    pub fn point_at(self, t: f64) -> Point3 {
        Point3::new(
            self.origin.x + self.direction.x * t,
            self.origin.y + self.direction.y * t,
            self.origin.z + self.direction.z * t,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub point: Point3,
    pub normal: Vector3,
}

impl Plane {
    pub fn new(point: Point3, normal: Vector3) -> SciResult<Self> {
        if normal.magnitude() <= f64::EPSILON {
            return Err(SciError::InvalidParameter("normal must be non-zero"));
        }
        Ok(Self { point, normal })
    }

    pub fn signed_distance(self, p: Point3) -> f64 {
        let delta = Vector3::new(p.x - self.point.x, p.y - self.point.y, p.z - self.point.z);
        self.normal.dot(delta) / self.normal.magnitude()
    }

    pub fn intersect_ray(self, ray: Ray3) -> Option<Point3> {
        let numerator = self.normal.dot(Vector3::new(
            self.point.x - ray.origin.x,
            self.point.y - ray.origin.y,
            self.point.z - ray.origin.z,
        ));
        let denominator = self.normal.dot(ray.direction);
        if denominator.abs() <= 1e-12 {
            return None;
        }
        let t = numerator / denominator;
        if t < 0.0 {
            return None;
        }
        Some(ray.point_at(t))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64) -> SciResult<Self> {
        if radius < 0.0 {
            return Err(SciError::InvalidParameter("radius must be non-negative"));
        }
        Ok(Self { center, radius })
    }

    pub fn contains(self, point: Point3) -> bool {
        self.center.distance(point) <= self.radius + 1e-12
    }

    pub fn intersects(self, other: Self) -> bool {
        self.center.distance(other.center) <= self.radius + other.radius
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb3 {
    pub fn new(min: Point3, max: Point3) -> SciResult<Self> {
        if min.x > max.x || min.y > max.y || min.z > max.z {
            return Err(SciError::InvalidParameter(
                "min must be <= max on every axis",
            ));
        }
        Ok(Self { min, max })
    }

    pub fn contains(self, point: Point3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub base_center: Point3,
    pub radius: f64,
    pub height: f64,
}

impl Cylinder {
    pub fn new(base_center: Point3, radius: f64, height: f64) -> SciResult<Self> {
        if radius < 0.0 || height < 0.0 {
            return Err(SciError::InvalidParameter(
                "radius and height must be non-negative",
            ));
        }
        Ok(Self {
            base_center,
            radius,
            height,
        })
    }

    pub fn contains(self, point: Point3) -> bool {
        let dx = point.x - self.base_center.x;
        let dz = point.z - self.base_center.z;
        let radial = (dx * dx + dz * dz).sqrt();
        radial <= self.radius + 1e-12
            && point.y >= self.base_center.y
            && point.y <= self.base_center.y + self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    pub apex: Point3,
    pub height: f64,
    pub base_radius: f64,
}

impl Cone {
    pub fn new(apex: Point3, height: f64, base_radius: f64) -> SciResult<Self> {
        if height < 0.0 || base_radius < 0.0 {
            return Err(SciError::InvalidParameter(
                "height and base_radius must be non-negative",
            ));
        }
        Ok(Self {
            apex,
            height,
            base_radius,
        })
    }

    pub fn contains(self, point: Point3) -> bool {
        let dy = point.y - self.apex.y;
        if dy < 0.0 || dy > self.height {
            return false;
        }
        let allowed_radius = if self.height <= f64::EPSILON {
            0.0
        } else {
            self.base_radius * (dy / self.height)
        };
        let dx = point.x - self.apex.x;
        let dz = point.z - self.apex.z;
        (dx * dx + dz * dz).sqrt() <= allowed_radius + 1e-12
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierCurve3 {
    pub p0: Point3,
    pub p1: Point3,
    pub p2: Point3,
    pub p3: Point3,
}

impl BezierCurve3 {
    pub const fn new(p0: Point3, p1: Point3, p2: Point3, p3: Point3) -> Self {
        Self { p0, p1, p2, p3 }
    }

    pub fn sample(self, t: f64) -> Point3 {
        let u = t.clamp(0.0, 1.0);
        let omt = 1.0 - u;
        let b0 = omt.powi(3);
        let b1 = 3.0 * omt.powi(2) * u;
        let b2 = 3.0 * omt * u.powi(2);
        let b3 = u.powi(3);
        Point3::new(
            b0 * self.p0.x + b1 * self.p1.x + b2 * self.p2.x + b3 * self.p3.x,
            b0 * self.p0.y + b1 * self.p1.y + b2 * self.p2.y + b3 * self.p3.y,
            b0 * self.p0.z + b1 * self.p1.z + b2 * self.p2.z + b3 * self.p3.z,
        )
    }
}

pub fn extrude_profile(profile: &[Point2], depth: f64) -> SciResult<Vec<Point3>> {
    if profile.len() < 2 {
        return Err(SciError::InvalidParameter(
            "profile must contain at least 2 points",
        ));
    }
    let mut vertices = Vec::with_capacity(profile.len() * 2);
    for &point in profile {
        vertices.push(Point3::new(point.x, point.y, 0.0));
    }
    for &point in profile {
        vertices.push(Point3::new(point.x, point.y, depth));
    }
    Ok(vertices)
}

pub fn revolve_profile(profile: &[Point2], segments: usize) -> SciResult<Vec<Point3>> {
    if profile.len() < 2 {
        return Err(SciError::InvalidParameter(
            "profile must contain at least 2 points",
        ));
    }
    if segments < 3 {
        return Err(SciError::InvalidParameter("segments must be at least 3"));
    }

    let mut vertices = Vec::with_capacity(profile.len() * segments);
    for step in 0..segments {
        let theta = 2.0 * core::f64::consts::PI * step as f64 / segments as f64;
        let (sin_t, cos_t) = theta.sin_cos();
        for &point in profile {
            vertices.push(Point3::new(point.x * cos_t, point.y, point.x * sin_t));
        }
    }
    Ok(vertices)
}
