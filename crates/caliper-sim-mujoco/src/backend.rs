//! [`MujocoBackend`] — MuJoCo contact simulation behind the caliper-hal
//! [`RobotBackend`] seam, so the EXISTING `ControlLoop` / `SafetyMonitor` /
//! teleop / recording stack drives a contact sim unchanged.
//!
//! Actuation is fixed at CONSTRUCTION, because a MuJoCo `<position>` servo
//! keeps acting no matter what you write elsewhere — the two styles cannot
//! share one model honestly:
//!
//! - [`MujocoBackend::new`] (or `with_options` + `Actuation::TorqueDirect`):
//!   `Torque` mode writes `qfrc_applied` directly (no actuators at all — the
//!   recon-verified path); `Position` mode is a NON-PHYSICAL teleport, exactly
//!   mirroring `caliper_hal::PhysicsSimBackend` semantics.
//! - `with_options` + `Actuation::PositionServo`: `Position` mode writes the
//!   servo targets (`ctrl`), MuJoCo computes the torque — physical tracking.
//!   `Torque` mode is UNSUPPORTED on this variant and errors loudly.

use crate::MujocoError;
use crate::mjcf::{Actuation, MjcfOptions};
use crate::sim::MujocoSim;
use caliper_hal::{ControlMode, Error, JointState, RobotBackend};
use caliper_model::Model;

/// A torque- or position-servo-driven MuJoCo contact-sim backend (fixed base,
/// gravity + contacts). Tick-driven and clock-free like every caliper backend:
/// nothing advances until `step(dt)`.
pub struct MujocoBackend {
    sim: MujocoSim,
    mode: ControlMode,
    enabled: bool,
    estopped: bool,
    dof: usize,
    position_actuated: bool,
}

impl MujocoBackend {
    /// Torque-direct backend with default [`MjcfOptions`] (Earth gravity,
    /// 1 ms timestep, no ground plane).
    pub fn new(model: &Model) -> Result<Self, MujocoError> {
        Self::with_options(model, &MjcfOptions::default())
    }

    /// Backend with explicit MJCF options; `opt.actuation` picks the variant
    /// (see module docs).
    pub fn with_options(model: &Model, opt: &MjcfOptions) -> Result<Self, MujocoError> {
        let position_actuated = matches!(opt.actuation, Actuation::PositionServo { .. });
        let sim = MujocoSim::from_caliper_model_with(model, opt)?;
        Ok(Self {
            dof: sim.ndof(),
            mode: if position_actuated {
                ControlMode::Position
            } else {
                ControlMode::Torque
            },
            enabled: false,
            estopped: false,
            position_actuated,
            sim,
        })
    }

    /// Seed `(q, qd)` without advancing the clock (initial conditions).
    pub fn set_state(&mut self, q: &[f64], qd: &[f64]) -> Result<(), Error> {
        self.sim.set_state(q, qd)?;
        Ok(())
    }

    /// Contact-sim readout (the reason this backend exists).
    pub fn sim(&self) -> &MujocoSim {
        &self.sim
    }
    pub fn sim_mut(&mut self) -> &mut MujocoSim {
        &mut self.sim
    }
}

fn check_len(got: usize, expected: usize) -> Result<(), Error> {
    if got != expected {
        return Err(Error::DofMismatch { expected, got });
    }
    Ok(())
}

fn check_finite(what: &'static str, xs: &[f64]) -> Result<(), Error> {
    if !xs.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite { what });
    }
    Ok(())
}

impl RobotBackend for MujocoBackend {
    fn dof(&self) -> usize {
        self.dof
    }
    fn joint_names(&self) -> Vec<String> {
        self.sim.joint_names().to_vec()
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.sim.qpos()
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

    /// Latching. Torque-direct: zero `qfrc_applied` (passive dynamics remain —
    /// the arm falls under gravity, as a de-energized robot would). Servo:
    /// freeze `ctrl` at the CURRENT position (a zeroed servo target would
    /// actively drive to q=0, the opposite of a stop).
    fn estop(&mut self) -> Result<(), Error> {
        self.estopped = true;
        self.enabled = false;
        let zeros = vec![0.0; self.dof];
        self.sim.set_joint_torques(&zeros)?;
        if self.sim.nu() > 0 {
            let hold = self.sim.qpos();
            self.sim.set_ctrl(&hold)?;
        }
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
            ControlMode::Position => {
                self.mode = mode;
                Ok(())
            }
            ControlMode::Torque if !self.position_actuated => {
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
        self.sim.set_joint_torques(tau)?; // does NOT step — the loop steps
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
        if self.position_actuated {
            // Physical: hand MuJoCo's <position> servos the targets.
            self.sim.set_ctrl(q)?;
        } else {
            // Non-physical teleport (mirrors PhysicsSimBackend): zero velocity
            // at the new pose.
            self.sim.set_state(q, &vec![0.0; self.dof])?;
        }
        Ok(())
    }

    fn read_state(&mut self) -> Result<JointState, Error> {
        Ok(JointState {
            tick: 0,
            t: self.sim.time(),
            q: self.sim.qpos(),
            qd: Some(self.sim.qvel()),
            tau: None,
        })
    }

    fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.sim.step(dt)?;
        Ok(())
    }
}
