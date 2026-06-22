//! Hardware/sim abstraction: the `RobotBackend` trait + a `SimBackend`.

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("backend error: {0}")]
    Backend(String),
}

/// A robot backend — simulated or real — driven by identical commands.
pub trait RobotBackend {
    fn dof(&self) -> usize;
    fn joint_positions(&self) -> Vec<f64>;
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error>;
}

/// A trivial kinematic simulation backend (holds the last commanded state).
#[derive(Clone, Debug)]
pub struct SimBackend {
    q: Vec<f64>,
}

impl SimBackend {
    pub fn new(dof: usize) -> Self {
        Self { q: vec![0.0; dof] }
    }
}

impl RobotBackend for SimBackend {
    fn dof(&self) -> usize {
        self.q.len()
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.q.clone()
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        if q.len() != self.q.len() {
            return Err(Error::Backend(format!(
                "expected {} joints, got {}",
                self.q.len(),
                q.len()
            )));
        }
        self.q.copy_from_slice(q);
        Ok(())
    }
}
