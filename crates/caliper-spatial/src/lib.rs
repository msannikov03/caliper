//! SE(3) / SO(3) spatial math for Caliper.
use nalgebra::{Isometry3, Vector6};

/// A rigid-body transform (rotation + translation) in SE(3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3(pub Isometry3<f64>);

impl Se3 {
    pub fn identity() -> Self {
        Se3(Isometry3::identity())
    }
    /// Translation component as `[x, y, z]`.
    pub fn translation(&self) -> [f64; 3] {
        let t = self.0.translation.vector;
        [t.x, t.y, t.z]
    }
    /// Log map to a 6-vector twist. (Phase-0 stub: returns zero.)
    pub fn log(&self) -> Vector6<f64> {
        Vector6::zeros()
    }
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}
