//! Forward kinematics, Jacobians, and singularity analysis.
use nalgebra::{DMatrix, DVector};

/// A geometric Jacobian (6 x ndof).
pub struct Jacobian(pub DMatrix<f64>);

impl Jacobian {
    /// Singular values of the Jacobian (descending).
    pub fn singular_values(&self) -> DVector<f64> {
        self.0.clone().svd(false, false).singular_values
    }
    /// Yoshikawa manipulability = product of singular values.
    pub fn manipulability(&self) -> f64 {
        self.singular_values().iter().product()
    }
}
