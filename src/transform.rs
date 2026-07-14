//! Phase 3 — Geometric transforms: Quaternion, Rotation, Isometry, Projective.

use crate::errors::{SciError, SciResult};
use crate::generic::SMatrix;



// ══════════════════════════════════════════════════════════════════════════════
// Quaternion
// ══════════════════════════════════════════════════════════════════════════════

/// Unit quaternion q = w + xi + yj + zk.  Represents 3D rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion { pub w: f64, pub x: f64, pub y: f64, pub z: f64 }

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self { Self { w, x, y, z } }
    pub fn identity() -> Self { Self::new(1.0, 0.0, 0.0, 0.0) }
    pub fn pure(x: f64, y: f64, z: f64) -> Self { Self::new(0.0, x, y, z) }

    pub fn norm(self) -> f64 { (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }

    pub fn normalize(self) -> SciResult<Self> {
        let n = self.norm();
        if n < f64::EPSILON { return Err(SciError::DivisionByZero); }
        Ok(Self::new(self.w/n, self.x/n, self.y/n, self.z/n))
    }

    pub fn conjugate(self) -> Self { Self::new(self.w, -self.x, -self.y, -self.z) }
    pub fn inverse(self) -> SciResult<Self> {
        let n2 = self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z;
        if n2 < f64::EPSILON { return Err(SciError::DivisionByZero); }
        Ok(Self::new(self.w/n2, -self.x/n2, -self.y/n2, -self.z/n2))
    }

    /// Hamilton product.
    pub fn mul(self, r: Self) -> Self {
        Self::new(
            self.w*r.w - self.x*r.x - self.y*r.y - self.z*r.z,
            self.w*r.x + self.x*r.w + self.y*r.z - self.z*r.y,
            self.w*r.y - self.x*r.z + self.y*r.w + self.z*r.x,
            self.w*r.z + self.x*r.y - self.y*r.x + self.z*r.w,
        )
    }

    /// Rotate vector v = (x,y,z) using q v q*.
    pub fn rotate_vec(self, v: [f64; 3]) -> [f64; 3] {
        let p = Self::pure(v[0], v[1], v[2]);
        let r = self.mul(p).mul(self.conjugate());
        [r.x, r.y, r.z]
    }

    /// Build from axis-angle (axis need not be normalized).
    pub fn from_axis_angle(axis: [f64; 3], angle_rad: f64) -> SciResult<Self> {
        let n = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt();
        if n < f64::EPSILON { return Err(SciError::InvalidParameter("zero axis")); }
        let (s, c) = ((angle_rad/2.0).sin(), (angle_rad/2.0).cos());
        Ok(Self::new(c, axis[0]/n*s, axis[1]/n*s, axis[2]/n*s))
    }

    /// Build from Euler angles (roll=X, pitch=Y, yaw=Z) in radians.
    pub fn from_euler_xyz(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr,cr) = ((roll/2.0).sin(), (roll/2.0).cos());
        let (sp,cp) = ((pitch/2.0).sin(), (pitch/2.0).cos());
        let (sy,cy) = ((yaw/2.0).sin(), (yaw/2.0).cos());
        Self::new(
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        )
    }

    /// Convert to 3×3 rotation matrix.
    pub fn to_rotation_matrix(self) -> SMatrix<f64, 3, 3> {
        let (w,x,y,z) = (self.w, self.x, self.y, self.z);
        SMatrix::from_array([
            [1.0-2.0*(y*y+z*z),     2.0*(x*y-w*z),     2.0*(x*z+w*y)],
            [    2.0*(x*y+w*z), 1.0-2.0*(x*x+z*z),     2.0*(y*z-w*x)],
            [    2.0*(x*z-w*y),     2.0*(y*z+w*x), 1.0-2.0*(x*x+y*y)],
        ])
    }

    /// Build from 3×3 rotation matrix (Shepperd method).
    pub fn from_rotation_matrix(m: &SMatrix<f64, 3, 3>) -> SciResult<Self> {
        let trace = m.data[0][0] + m.data[1][1] + m.data[2][2];
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Ok(Self::new(0.25/s,
                (m.data[2][1]-m.data[1][2])*s,
                (m.data[0][2]-m.data[2][0])*s,
                (m.data[1][0]-m.data[0][1])*s,
            ))
        } else if m.data[0][0] > m.data[1][1] && m.data[0][0] > m.data[2][2] {
            let s = 2.0*(1.0+m.data[0][0]-m.data[1][1]-m.data[2][2]).sqrt();
            Ok(Self::new((m.data[2][1]-m.data[1][2])/s, 0.25*s,
                (m.data[0][1]+m.data[1][0])/s, (m.data[0][2]+m.data[2][0])/s))
        } else if m.data[1][1] > m.data[2][2] {
            let s = 2.0*(1.0+m.data[1][1]-m.data[0][0]-m.data[2][2]).sqrt();
            Ok(Self::new((m.data[0][2]-m.data[2][0])/s, (m.data[0][1]+m.data[1][0])/s,
                0.25*s, (m.data[1][2]+m.data[2][1])/s))
        } else {
            let s = 2.0*(1.0+m.data[2][2]-m.data[0][0]-m.data[1][1]).sqrt();
            Ok(Self::new((m.data[1][0]-m.data[0][1])/s, (m.data[0][2]+m.data[2][0])/s,
                (m.data[1][2]+m.data[2][1])/s, 0.25*s))
        }
    }

    /// Spherical linear interpolation (slerp) between two unit quaternions.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z;
        let other = if dot < 0.0 {
            dot = -dot;
            Self::new(-other.w, -other.x, -other.y, -other.z)
        } else { other };
        if dot > 0.9995 {
            // Linear interpolation for nearly identical quaternions
            let r = Self::new(self.w+t*(other.w-self.w), self.x+t*(other.x-self.x),
                self.y+t*(other.y-self.y), self.z+t*(other.z-self.z));
            return r.normalize().unwrap_or(self);
        }
        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let (sin0, sin_t) = (theta_0.sin(), theta.sin());
        let s1 = (theta_0 - theta).sin() / sin0;
        let s2 = sin_t / sin0;
        Self::new(s1*self.w + s2*other.w, s1*self.x + s2*other.x,
            s1*self.y + s2*other.y, s1*self.z + s2*other.z)
    }

    /// Extract angle (radians).
    pub fn angle(self) -> f64 { 2.0 * self.w.clamp(-1.0, 1.0).acos() }

    /// Extract axis (unit vector). Returns None for identity quaternion.
    pub fn axis(self) -> Option<[f64; 3]> {
        let s2 = (self.x*self.x + self.y*self.y + self.z*self.z).sqrt();
        if s2 < f64::EPSILON { return None; }
        Some([self.x/s2, self.y/s2, self.z/s2])
    }

    /// To Euler angles (roll, pitch, yaw) in radians.
    pub fn to_euler_xyz(self) -> [f64; 3] {
        let (w,x,y,z) = (self.w, self.x, self.y, self.z);
        let roll  = (2.0*(w*x+y*z)).atan2(1.0 - 2.0*(x*x+y*y));
        let pitch = (2.0*(w*y - z*x)).clamp(-1.0, 1.0).asin();
        let yaw   = (2.0*(w*z+x*y)).atan2(1.0 - 2.0*(y*y+z*z));
        [roll, pitch, yaw]
    }
}

impl core::ops::Mul for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Quaternion::mul(self, rhs) }
}

impl core::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:.4} + {:.4}i + {:.4}j + {:.4}k", self.w, self.x, self.y, self.z)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Rotation3 — thin newtype over Quaternion (always unit)
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation3(Quaternion);

impl Rotation3 {
    pub fn identity() -> Self { Self(Quaternion::identity()) }

    pub fn from_axis_angle(axis: [f64; 3], angle_rad: f64) -> SciResult<Self> {
        Ok(Self(Quaternion::from_axis_angle(axis, angle_rad)?.normalize()?))
    }

    pub fn from_euler_xyz(roll: f64, pitch: f64, yaw: f64) -> SciResult<Self> {
        Ok(Self(Quaternion::from_euler_xyz(roll, pitch, yaw).normalize()?))
    }

    pub fn from_matrix(m: &SMatrix<f64, 3, 3>) -> SciResult<Self> {
        Ok(Self(Quaternion::from_rotation_matrix(m)?.normalize()?))
    }

    pub fn to_matrix(&self) -> SMatrix<f64, 3, 3> { self.0.to_rotation_matrix() }
    pub fn quaternion(&self) -> Quaternion { self.0 }
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] { self.0.rotate_vec(v) }
    pub fn inverse(&self) -> Self { Self(self.0.conjugate()) }

    pub fn compose(&self, other: &Self) -> Self { Self(self.0.mul(other.0)) }
    pub fn slerp(&self, other: &Self, t: f64) -> SciResult<Self> {
        Ok(Self(self.0.slerp(other.0, t).normalize()?))
    }
    pub fn angle(&self) -> f64 { self.0.angle() }
    pub fn axis(&self) -> Option<[f64; 3]> { self.0.axis() }
    pub fn to_euler_xyz(&self) -> [f64; 3] { self.0.to_euler_xyz() }
}

// ══════════════════════════════════════════════════════════════════════════════
// Isometry3 — rigid body transform: Rotation + Translation
// ══════════════════════════════════════════════════════════════════════════════

/// Rigid body transform: p' = R·p + t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Isometry3 {
    pub rotation: Rotation3,
    pub translation: [f64; 3],
}

impl Isometry3 {
    pub fn identity() -> Self {
        Self { rotation: Rotation3::identity(), translation: [0.0; 3] }
    }

    pub fn new(rotation: Rotation3, translation: [f64; 3]) -> Self {
        Self { rotation, translation }
    }

    pub fn from_parts(rot: SMatrix<f64, 3, 3>, t: [f64; 3]) -> SciResult<Self> {
        Ok(Self { rotation: Rotation3::from_matrix(&rot)?, translation: t })
    }

    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let rp = self.rotation.rotate(p);
        [rp[0]+self.translation[0], rp[1]+self.translation[1], rp[2]+self.translation[2]]
    }

    pub fn transform_vec(&self, v: [f64; 3]) -> [f64; 3] { self.rotation.rotate(v) }

    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        let t = self.translation;
        let rt = r_inv.rotate([t[0], t[1], t[2]]);
        Self { rotation: r_inv, translation: [-rt[0], -rt[1], -rt[2]] }
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation.compose(&other.rotation),
            translation: self.transform_point(other.translation),
        }
    }

    /// Convert to 4×4 homogeneous matrix.
    pub fn to_matrix4(&self) -> SMatrix<f64, 4, 4> {
        let r = self.rotation.to_matrix();
        let t = self.translation;
        SMatrix::from_array([
            [r.data[0][0], r.data[0][1], r.data[0][2], t[0]],
            [r.data[1][0], r.data[1][1], r.data[1][2], t[1]],
            [r.data[2][0], r.data[2][1], r.data[2][2], t[2]],
            [0.0,          0.0,          0.0,           1.0 ],
        ])
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Similarity3 — Isometry + uniform scale
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity3 { pub isometry: Isometry3, pub scale: f64 }

impl Similarity3 {
    pub fn identity() -> Self { Self { isometry: Isometry3::identity(), scale: 1.0 } }
    pub fn new(iso: Isometry3, scale: f64) -> SciResult<Self> {
        if scale <= 0.0 { return Err(SciError::InvalidParameter("scale must be > 0")); }
        Ok(Self { isometry: iso, scale })
    }
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let rp = self.isometry.transform_point(p);
        [rp[0]*self.scale, rp[1]*self.scale, rp[2]*self.scale]
    }
    pub fn inverse(&self) -> SciResult<Self> {
        let s_inv = 1.0 / self.scale;
        let iso_inv = self.isometry.inverse();
        // Scale the inverted translation
        let t = iso_inv.translation;
        let iso = Isometry3::new(iso_inv.rotation, [t[0]*s_inv, t[1]*s_inv, t[2]*s_inv]);
        Ok(Self { isometry: iso, scale: s_inv })
    }
    pub fn to_matrix4(&self) -> SMatrix<f64, 4, 4> {
        let r = self.isometry.rotation.to_matrix();
        let t = self.isometry.translation;
        let s = self.scale;
        SMatrix::from_array([
            [r.data[0][0]*s, r.data[0][1]*s, r.data[0][2]*s, t[0]],
            [r.data[1][0]*s, r.data[1][1]*s, r.data[1][2]*s, t[1]],
            [r.data[2][0]*s, r.data[2][1]*s, r.data[2][2]*s, t[2]],
            [0.0,            0.0,            0.0,             1.0 ],
        ])
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Projective / Camera transforms (4×4 homogeneous matrices)
// ══════════════════════════════════════════════════════════════════════════════

/// Perspective projection matrix (column-major, right-handed, NDC [-1,1]).
pub fn perspective(fov_y_rad: f64, aspect: f64, near: f64, far: f64) -> SciResult<SMatrix<f64,4,4>> {
    if near <= 0.0 || far <= near { return Err(SciError::InvalidParameter("0 < near < far")); }
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    Ok(SMatrix::from_array([
        [f/aspect,  0.0,  0.0,                        0.0],
        [0.0,         f,  0.0,                        0.0],
        [0.0,       0.0, -(far+near)/(far-near),      -1.0],
        [0.0,       0.0, -(2.0*far*near)/(far-near),   0.0],
    ]))
}

/// Orthographic projection matrix.
pub fn ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> SMatrix<f64,4,4> {
    SMatrix::from_array([
        [2.0/(right-left), 0.0,              0.0,            -(right+left)/(right-left)],
        [0.0,              2.0/(top-bottom), 0.0,            -(top+bottom)/(top-bottom)],
        [0.0,              0.0,             -2.0/(far-near), -(far+near)/(far-near)    ],
        [0.0,              0.0,              0.0,             1.0                      ],
    ])
}

/// Look-at view matrix (right-handed).
pub fn look_at(eye: [f64;3], center: [f64;3], up: [f64;3]) -> SciResult<SMatrix<f64,4,4>> {
    let f = normalize3([center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]])?;
    let s = normalize3(cross3(f, up))?;
    let u = cross3(s, f);
    Ok(SMatrix::from_array([
        [ s[0],  s[1],  s[2], -dot3(s,eye)],
        [ u[0],  u[1],  u[2], -dot3(u,eye)],
        [-f[0], -f[1], -f[2],  dot3(f,eye)],
        [ 0.0,   0.0,   0.0,   1.0        ],
    ]))
}

/// Translation matrix.
pub fn translation_matrix(t: [f64;3]) -> SMatrix<f64,4,4> {
    let mut m = SMatrix::<f64,4,4>::identity();
    m.data[0][3] = t[0]; m.data[1][3] = t[1]; m.data[2][3] = t[2];
    m
}

/// Uniform scale matrix.
pub fn scale_matrix(s: f64) -> SMatrix<f64,4,4> {
    SMatrix::from_array([
        [s,   0.0, 0.0, 0.0],
        [0.0, s,   0.0, 0.0],
        [0.0, 0.0, s,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

// ══════════════════════════════════════════════════════════════════════════════
// Dual Quaternion (rigid body transforms without gimbal lock)
// ══════════════════════════════════════════════════════════════════════════════

/// Dual quaternion  q̂ = q_r + ε·q_d  where ε² = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuaternion { pub real: Quaternion, pub dual: Quaternion }

impl DualQuaternion {
    pub fn identity() -> Self {
        Self { real: Quaternion::identity(), dual: Quaternion::new(0.0, 0.0, 0.0, 0.0) }
    }

    /// Build from rotation and translation.
    pub fn from_rotation_translation(r: Quaternion, t: [f64; 3]) -> SciResult<Self> {
        let r = r.normalize()?;
        let t_q = Quaternion::new(0.0, t[0]*0.5, t[1]*0.5, t[2]*0.5);
        Ok(Self { real: r, dual: t_q.mul(r) })
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            real: self.real.mul(other.real),
            dual: self.real.mul(other.dual).add(self.dual.mul(other.real)),
        }
    }

    pub fn conjugate(&self) -> Self {
        Self { real: self.real.conjugate(), dual: self.dual.conjugate() }
    }

    /// Transform a point using dual quaternion.
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let p_dq = DualQuaternion {
            real: Quaternion::identity(),
            dual: Quaternion::new(0.0, p[0], p[1], p[2]),
        };
        let result = self.mul(&p_dq).mul(&self.conjugate());
        [result.dual.x, result.dual.y, result.dual.z]
    }

    /// Extract translation vector.
    pub fn translation(&self) -> [f64; 3] {
        let t = self.dual.mul(self.real.conjugate());
        [2.0*t.x, 2.0*t.y, 2.0*t.z]
    }

    pub fn to_isometry(&self) -> SciResult<Isometry3> {
        let r = Rotation3::from_matrix(&self.real.to_rotation_matrix())?;
        Ok(Isometry3::new(r, self.translation()))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Private helpers
// ══════════════════════════════════════════════════════════════════════════════

fn dot3(a: [f64;3], b: [f64;3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

fn cross3(a: [f64;3], b: [f64;3]) -> [f64;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn normalize3(v: [f64;3]) -> SciResult<[f64;3]> {
    let n = dot3(v, v).sqrt();
    if n < f64::EPSILON { return Err(SciError::InvalidParameter("zero vector")); }
    Ok([v[0]/n, v[1]/n, v[2]/n])
}

// Quaternion add (used internally only)
impl Quaternion {
    fn add(self, other: Self) -> Self {
        Self::new(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
    }
}
