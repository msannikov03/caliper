//! The original trivial kinematic backend: a position teleport that holds the
//! last commanded pose. Upgraded to the Phase-5 trait — it reports zero velocity,
//! honors enable/e-stop, and rejects velocity/torque modes — but its physics is
//! still "the arm is wherever you last told it to be."

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite, check_len};

/// A trivial kinematic simulation backend (holds the last commanded state).
/// Position-only; enabled by default so it "just works" as a teleport target.
#[derive(Clone, Debug)]
pub struct SimBackend {
    q: Vec<f64>,
    enabled: bool,
    estopped: bool,
    t: f64,
}

impl SimBackend {
    pub fn new(dof: usize) -> Self {
        Self {
            q: vec![0.0; dof],
            enabled: true,
            estopped: false,
            t: 0.0,
        }
    }
    /// Seed the held pose (validated). Convenience for tests / initial conditions.
    pub fn with_positions(mut self, q: &[f64]) -> Result<Self, Error> {
        check_len(q.len(), self.q.len())?;
        check_finite("q", q)?;
        self.q.copy_from_slice(q);
        Ok(self)
    }
}

impl RobotBackend for SimBackend {
    fn dof(&self) -> usize {
        self.q.len()
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.q.clone()
    }
    fn enable(&mut self) -> Result<(), Error> {
        if self.estopped {
            return Err(Error::EStopActive);
        }
        self.enabled = true;
        Ok(())
    }
    fn disable(&mut self) -> Result<(), Error> {
        self.enabled = false;
        Ok(())
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn estop(&mut self) -> Result<(), Error> {
        self.estopped = true;
        self.enabled = false;
        Ok(())
    }
    fn clear_estop(&mut self) -> Result<(), Error> {
        self.estopped = false;
        Ok(())
    }
    fn is_estopped(&self) -> bool {
        self.estopped
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Position
    }
    fn set_mode(&mut self, mode: ControlMode) -> Result<(), Error> {
        match mode {
            ControlMode::Position => Ok(()),
            m => Err(Error::UnsupportedMode(m)),
        }
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        check_len(q.len(), self.q.len())?;
        check_finite("q", q)?;
        if self.estopped {
            return Err(Error::EStopActive);
        }
        if !self.enabled {
            return Err(Error::NotEnabled);
        }
        self.q.copy_from_slice(q);
        Ok(())
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        let n = self.q.len();
        Ok(JointState {
            tick: 0,
            t: self.t,
            q: self.q.clone(),
            qd: Some(vec![0.0; n]),
            tau: None,
        })
    }
    fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.t += dt;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_and_holds() {
        let mut b = SimBackend::new(3);
        b.command_joint_positions(&[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(b.joint_positions(), vec![0.1, 0.2, 0.3]);
        let s = b.read_state().unwrap();
        assert_eq!(s.q, vec![0.1, 0.2, 0.3]);
        assert_eq!(s.qd, Some(vec![0.0; 3]));
    }

    #[test]
    fn rejects_bad_input() {
        let mut b = SimBackend::new(2);
        assert!(matches!(
            b.command_joint_positions(&[0.0]),
            Err(Error::DofMismatch { .. })
        ));
        assert!(matches!(
            b.command_joint_positions(&[0.0, f64::NAN]),
            Err(Error::NonFinite { .. })
        ));
    }

    #[test]
    fn unsupported_modes() {
        let mut b = SimBackend::new(2);
        assert!(matches!(
            b.command_joint_velocities(&[0.0, 0.0]),
            Err(Error::UnsupportedMode(ControlMode::Velocity))
        ));
        assert!(matches!(
            b.command_joint_torques(&[0.0, 0.0]),
            Err(Error::UnsupportedMode(ControlMode::Torque))
        ));
        assert!(matches!(
            b.set_mode(ControlMode::Torque),
            Err(Error::UnsupportedMode(ControlMode::Torque))
        ));
    }

    #[test]
    fn estop_latches_and_blocks() {
        let mut b = SimBackend::new(1);
        b.estop().unwrap();
        assert!(b.is_estopped() && !b.is_enabled());
        assert!(matches!(
            b.command_joint_positions(&[0.5]),
            Err(Error::EStopActive)
        ));
        assert!(matches!(b.enable(), Err(Error::EStopActive)));
        b.clear_estop().unwrap();
        b.enable().unwrap();
        b.command_joint_positions(&[0.5]).unwrap();
        assert_eq!(b.joint_positions(), vec![0.5]);
    }

    #[test]
    fn disabled_rejects() {
        let mut b = SimBackend::new(1);
        b.disable().unwrap();
        assert!(matches!(
            b.command_joint_positions(&[0.1]),
            Err(Error::NotEnabled)
        ));
    }

    #[test]
    fn object_safe_and_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn RobotBackend>>();
        let _b: Box<dyn RobotBackend> = Box::new(SimBackend::new(2));
    }
}
