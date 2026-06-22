//! Robot model: load from URDF, expose the kinematic tree.
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to read URDF: {0}")]
    Urdf(String),
}

/// A robot model. (Phase-0: name + movable joints; full tree lands in Phase 1.)
#[derive(Clone, Debug, Default)]
pub struct Robot {
    pub name: String,
    pub joint_names: Vec<String>,
}

impl Robot {
    /// Parse a URDF file into a model.
    pub fn from_urdf(path: &Path) -> Result<Self, Error> {
        let urdf = urdf_rs::read_file(path).map_err(|e| Error::Urdf(e.to_string()))?;
        let joint_names = urdf
            .joints
            .iter()
            .filter(|j| !matches!(j.joint_type, urdf_rs::JointType::Fixed))
            .map(|j| j.name.clone())
            .collect();
        Ok(Robot {
            name: urdf.name,
            joint_names,
        })
    }
    /// Number of movable degrees of freedom.
    pub fn ndof(&self) -> usize {
        self.joint_names.len()
    }
}
