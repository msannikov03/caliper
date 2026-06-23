//! A dynamics-backed backend: the load-bearing sim. It OWNS a Phase-4
//! [`Simulator`](caliper_dynamics::Simulator), so commanded torques drive real
//! rigid-body physics (gravity, Coriolis, inertia coupling). The control loop
//! drives it in [`Torque`](ControlMode::Torque) mode — `command_joint_torques`
//! sets the torque, then `step(dt)` integrates, as TWO decoupled calls so state
//! can be read between set and integrate. [`Position`](ControlMode::Position)
//! mode is a non-physical teleport (handy for seeding / kinematic parity).

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite, check_len};
use caliper_dynamics::Simulator;
use caliper_model::Model;
use nalgebra::Vector3;
use std::sync::Arc;

/// A torque-driven physical simulation backend (fixed-base, no contact).
pub struct PhysicsSimBackend {
    sim: Simulator,
    mode: ControlMode,
    enabled: bool,
    estopped: bool,
    dof: usize,
}

impl PhysicsSimBackend {
    /// Build over a model that carries `<inertial>` data (errors otherwise).
    pub fn new(model: Arc<Model>) -> Result<Self, Error> {
        let dof = model.ndof;
        let sim = Simulator::new(model)?;
        Ok(Self {
            sim,
            mode: ControlMode::Torque,
            enabled: false,
            estopped: false,
            dof,
        })
    }

    /// Override gravity (default Earth Z-up).
    pub fn set_gravity(&mut self, g: Vector3<f64>) {
        self.sim.set_gravity(g);
    }
    /// Override per-joint viscous damping.
    pub fn set_damping(&mut self, b: &[f64]) -> Result<(), Error> {
        self.sim.set_damping(b)?;
        Ok(())
    }
    /// Set the integrator's max internal substep (smaller = finer/stiffer-stable).
    pub fn set_sim_hmax(&mut self, h: f64) {
        if h.is_finite() && h > 0.0 {
            self.sim.h_max = h;
        }
    }
    /// Seed `(q, qd)` without advancing the clock (used for initial conditions).
    pub fn set_state(&mut self, q: &[f64], qd: &[f64]) -> Result<(), Error> {
        check_len(q.len(), self.dof)?;
        check_len(qd.len(), self.dof)?;
        check_finite("q", q)?;
        check_finite("qd", qd)?;
        self.sim.set_state(q, qd)?;
        Ok(())
    }
    /// Read-only access to the underlying simulator (energy, gravity, etc.).
    pub fn sim(&self) -> &Simulator {
        &self.sim
    }
}

impl RobotBackend for PhysicsSimBackend {
    fn dof(&self) -> usize {
        self.dof
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.sim.q().to_vec()
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
        let _ = self.sim.set_torque(&vec![0.0; self.dof]); // zero command on latch
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
        self.mode
    }
    fn set_mode(&mut self, mode: ControlMode) -> Result<(), Error> {
        match mode {
            ControlMode::Torque | ControlMode::Position => {
                self.mode = mode;
                Ok(())
            }
            m => Err(Error::UnsupportedMode(m)),
        }
    }
    fn command_joint_torques(&mut self, tau: &[f64]) -> Result<(), Error> {
        check_len(tau.len(), self.dof)?;
        check_finite("tau", tau)?;
        if self.estopped {
            return Err(Error::EStopActive);
        }
        if !self.enabled {
            return Err(Error::NotEnabled);
        }
        if self.mode != ControlMode::Torque {
            return Err(Error::UnsupportedMode(ControlMode::Torque));
        }
        self.sim.set_torque(tau)?; // does NOT step — the loop steps separately
        Ok(())
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        check_len(q.len(), self.dof)?;
        check_finite("q", q)?;
        if self.estopped {
            return Err(Error::EStopActive);
        }
        if !self.enabled {
            return Err(Error::NotEnabled);
        }
        if self.mode != ControlMode::Position {
            return Err(Error::UnsupportedMode(ControlMode::Position));
        }
        // Non-physical teleport (ignores dynamics); zero velocity at the new pose.
        self.sim.set_state(q, &vec![0.0; self.dof])?;
        Ok(())
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        Ok(JointState {
            tick: 0,
            t: self.sim.time(),
            q: self.sim.q().to_vec(),
            qd: Some(self.sim.qd().to_vec()),
            tau: None,
        })
    }
    fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.sim.step(dt)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_dynamics::{GRAVITY_EARTH, rnea};
    use std::path::Path;

    fn model(name: &str) -> Arc<Model> {
        Arc::new(
            Model::from_urdf(Path::new(&format!(
                "{}/../../oracle/fixtures/robots/{}",
                env!("CARGO_MANIFEST_DIR"),
                name
            )))
            .unwrap(),
        )
    }

    #[test]
    fn torque_passthrough_matches_bare_simulator() {
        // Driving the backend in Torque mode == driving the Simulator directly.
        let m = model("dyn_pendulum2.urdf");
        let mut b = PhysicsSimBackend::new(m.clone()).unwrap();
        b.set_state(&[0.3, -0.2], &[0.0, 0.0]).unwrap();
        b.enable().unwrap();
        let mut ref_sim = Simulator::new(m).unwrap();
        ref_sim.set_state(&[0.3, -0.2], &[0.0, 0.0]).unwrap();
        let tau = [0.7, -0.4];
        for _ in 0..200 {
            b.command_joint_torques(&tau).unwrap();
            b.step(1e-3).unwrap();
            ref_sim.set_torque(&tau).unwrap();
            ref_sim.step(1e-3).unwrap();
        }
        let s = b.read_state().unwrap();
        for i in 0..2 {
            assert!((s.q[i] - ref_sim.q()[i]).abs() < 1e-12, "q[{i}] drift");
            assert!((s.qd.as_ref().unwrap()[i] - ref_sim.qd()[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn gravity_hold_torque_equals_rnea() {
        // Commanding the static gravity torque holds the arm (qd stays ~0): the
        // gravity feed-forward IS rnea(q, 0, 0, g). Catches a sign flip.
        let m = model("dyn_pendulum2.urdf");
        let q0 = [0.4, 0.25];
        let mut b = PhysicsSimBackend::new(m.clone()).unwrap();
        b.set_damping(&[0.0, 0.0]).unwrap();
        b.set_state(&q0, &[0.0, 0.0]).unwrap();
        b.enable().unwrap();
        for _ in 0..300 {
            let q = b.read_state().unwrap().q;
            let g = rnea(&m, &q, &[0.0; 2], &[0.0; 2], &GRAVITY_EARTH).unwrap();
            b.command_joint_torques(g.as_slice()).unwrap();
            b.step(1e-3).unwrap();
        }
        let s = b.read_state().unwrap();
        for (i, (&qi, &q0i)) in s.q.iter().zip(q0.iter()).enumerate() {
            assert!((qi - q0i).abs() < 1e-2, "drifted from hold at {i}");
        }
    }

    #[test]
    fn torque_blocked_until_enabled_and_mode() {
        let m = model("dyn_pendulum2.urdf");
        let mut b = PhysicsSimBackend::new(m).unwrap();
        assert!(matches!(
            b.command_joint_torques(&[0.0, 0.0]),
            Err(Error::NotEnabled)
        ));
        b.enable().unwrap();
        b.command_joint_torques(&[0.0, 0.0]).unwrap();
        // Position teleport requires Position mode.
        assert!(matches!(
            b.command_joint_positions(&[0.1, 0.1]),
            Err(Error::UnsupportedMode(ControlMode::Position))
        ));
        b.set_mode(ControlMode::Position).unwrap();
        b.command_joint_positions(&[0.1, 0.1]).unwrap();
        assert_eq!(b.joint_positions(), vec![0.1, 0.1]);
    }

    #[test]
    fn estop_zeros_and_blocks() {
        let m = model("dyn_pendulum2.urdf");
        let mut b = PhysicsSimBackend::new(m).unwrap();
        b.enable().unwrap();
        b.estop().unwrap();
        assert!(b.is_estopped() && !b.is_enabled());
        assert!(matches!(
            b.command_joint_torques(&[1.0, 1.0]),
            Err(Error::EStopActive)
        ));
    }
}
