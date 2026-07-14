//! Phase 5C — `serde` feature: Serialize / Deserialize for core types.
//!
//! Enable via `Cargo.toml`:
//! ```toml
//! [dependencies]
//! scies-math-th = { version = "0.2.3", features = ["serde"] }
//! ```

#[cfg(feature = "serde")]
pub use self::inner::*;

#[cfg(feature = "serde")]
mod inner {
    use crate::generic::{Mat, Scalar, Vec1};
    use crate::transform::{Isometry3, Quaternion, Rotation3};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    // ── Mat<T> ──────────────────────────────────────────────────────────────

    impl<T: Scalar + Serialize> Serialize for Mat<T> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeStruct;
            let mut st = s.serialize_struct("Mat", 3)?;
            st.serialize_field("rows", &self.rows)?;
            st.serialize_field("cols", &self.cols)?;
            st.serialize_field("data", &self.data)?;
            st.end()
        }
    }

    impl<'de, T> Deserialize<'de> for Mat<T>
    where
        T: Scalar + Deserialize<'de>,
    {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            use serde::de::{MapAccess, Visitor};
            use core::fmt;

            struct MatVisitor<T>(core::marker::PhantomData<T>);

            impl<'de, T: Scalar + Deserialize<'de>> Visitor<'de> for MatVisitor<T> {
                type Value = Mat<T>;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a Mat struct with rows, cols, data")
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let mut rows: Option<usize> = None;
                    let mut cols: Option<usize> = None;
                    let mut data: Option<Vec<T>> = None;
                    while let Some(key) = map.next_key::<&str>()? {
                        match key {
                            "rows" => rows = Some(map.next_value()?),
                            "cols" => cols = Some(map.next_value()?),
                            "data" => data = Some(map.next_value()?),
                            _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
                        }
                    }
                    let rows = rows.ok_or_else(|| serde::de::Error::missing_field("rows"))?;
                    let cols = cols.ok_or_else(|| serde::de::Error::missing_field("cols"))?;
                    let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;
                    Mat::new(rows, cols, data).map_err(serde::de::Error::custom)
                }
            }

            d.deserialize_struct("Mat", &["rows", "cols", "data"], MatVisitor(core::marker::PhantomData))
        }
    }

    // ── Vec1<T> ─────────────────────────────────────────────────────────────

    impl<T: Scalar + Serialize> Serialize for Vec1<T> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            self.data.serialize(s)
        }
    }

    impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for Vec1<T> {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let data = Vec::<T>::deserialize(d)?;
            Ok(Vec1::new(data))
        }
    }

    // ── Quaternion ───────────────────────────────────────────────────────────

    impl Serialize for Quaternion {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            [self.w, self.x, self.y, self.z].serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for Quaternion {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let [w, x, y, z] = <[f64; 4]>::deserialize(d)?;
            Ok(Quaternion::new(w, x, y, z))
        }
    }

    // ── Rotation3 ───────────────────────────────────────────────────────────

    impl Serialize for Rotation3 {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            self.quaternion().serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for Rotation3 {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let q = Quaternion::deserialize(d)?;
            Rotation3::from_matrix(&q.to_rotation_matrix())
                .map_err(serde::de::Error::custom)
        }
    }

    // ── Isometry3 ───────────────────────────────────────────────────────────

    impl Serialize for Isometry3 {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeStruct;
            let mut st = s.serialize_struct("Isometry3", 2)?;
            st.serialize_field("rotation", &self.rotation)?;
            st.serialize_field("translation", &self.translation)?;
            st.end()
        }
    }

    impl<'de> Deserialize<'de> for Isometry3 {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            struct Helper { rotation: Rotation3, translation: [f64; 3] }
            let h = Helper::deserialize(d)?;
            Ok(Isometry3::new(h.rotation, h.translation))
        }
    }
}
