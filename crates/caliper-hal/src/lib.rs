//! Hardware/sim abstraction layer + a deterministic control stack.
//!
//! Phase 5 turns the old 3-method `RobotBackend` stub into a real backend
//! contract — control modes, lifecycle/safety, atomic state readback, and a
//! tick-driven `step(dt)` — plus everything that drives it: a pure
//! [`SafetyMonitor`], [`Setpoint`] sources (incl. teleop), a deterministic
//! [`ControlLoop`], LeRobotDataset record/replay (feature `dataset`), and
//! feature-gated hardware skeletons (`can`, `dynamixel`).
//!
//! EVERYTHING here is **tick-driven and clock-free**: nothing advances until
//! the loop calls `step(dt)`, and `t == tick * dt`. No `Instant::now`, no
//! wall-clock — so a rollout is bit-for-bit reproducible and unit-testable
//! without any real robot. Collision lives in the separate `caliper-collision`
//! crate (rapier pulls the banned `parry`); it plugs in here via [`SafetyCheck`].
pub mod backends;
mod control;
mod physics;
mod safety;
mod setpoint;
mod sim;
mod streaming;
mod teleop;

#[cfg(feature = "dataset")]
mod recorder;

pub use control::{ControlLoop, Frame, FrameSink, Gains, VecSink};
pub use physics::PhysicsSimBackend;
pub use safety::{
    SafetyBackend, SafetyCheck, SafetyConfig, SafetyError, SafetyMonitor, Verdict, WatchdogAction,
};
pub use setpoint::{HoldSetpoint, Setpoint, Target, TeleopSetpoint, TrajectorySetpoint};
pub use sim::SimBackend;
pub use teleop::{JogHandle, JogSource, JointMap, LeaderFollowerSource, ScriptedSource};

#[cfg(feature = "dataset")]
pub use recorder::{DatasetReader, DatasetSpec, Episode, Recorder, replay_frame};

#[cfg(feature = "can")]
pub use backends::can_mks::{CanMksBackend, MksFrame};
#[cfg(feature = "dynamixel")]
pub use backends::dynamixel::DynamixelBackend;
#[cfg(feature = "remote")]
pub use backends::remote::{RemoteBackend, RemoteCmd};

use caliper_dynamics::DynError;

/// How a joint command is interpreted. Hardware servos must be switched between
/// these explicitly; the trait makes the active mode first-class instead of
/// implicit-via-the-position-setter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ControlMode {
    #[default]
    Position,
    Velocity,
    Torque,
}

/// An atomic, timestamped state readback. `q` is always present; `qd`/`tau` are
/// `None` when a backend cannot measure them. `t` is the backend-local clock;
/// `tick` is filled by a driver (`0` when read standalone) — the [`ControlLoop`]
/// is the authority on tick/time for recorded [`Frame`]s.
#[derive(Clone, Debug)]
pub struct JointState {
    pub tick: u64,
    pub t: f64,
    pub q: Vec<f64>,
    pub qd: Option<Vec<f64>>,
    pub tau: Option<Vec<f64>>,
}

impl JointState {
    /// Measured velocity or zeros (the control loop's PD treats unmeasured
    /// velocity as zero rather than refusing to run).
    pub fn qd_or_zero(&self) -> Vec<f64> {
        self.qd.clone().unwrap_or_else(|| vec![0.0; self.q.len()])
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("expected {expected} joints, got {got}")]
    DofMismatch { expected: usize, got: usize },
    #[error("non-finite value in {what}")]
    NonFinite { what: &'static str },
    #[error("backend not enabled")]
    NotEnabled,
    #[error("emergency stop active")]
    EStopActive,
    #[error("control mode {0:?} not supported by this backend")]
    UnsupportedMode(ControlMode),
    #[error("joint {joint} value {value} outside limit [{lo}, {hi}]")]
    JointLimit {
        joint: usize,
        value: f64,
        lo: f64,
        hi: f64,
    },
    #[error("not connected")]
    NotConnected,
    /// Hardware skeletons return THIS on any bus op — never a fake success.
    #[error("hardware required: {0}")]
    HardwareRequired(&'static str),
    #[error("collision: {0}")]
    Collision(String),
    #[cfg(any(feature = "can", feature = "dynamixel"))]
    #[error("io: {0}")]
    Io(String),
    #[cfg(any(feature = "can", feature = "dynamixel"))]
    #[error("timeout")]
    Timeout,
    #[error(transparent)]
    Dyn(#[from] DynError),
    /// Escape hatch for backend-specific messages.
    #[error("backend error: {0}")]
    Backend(String),
}

/// A robot backend — simulated or real — driven by identical commands.
///
/// Object-safe and `Send`, so a [`ControlLoop`] can own a `Box<dyn RobotBackend>`
/// on its own thread. Every method beyond the original three has a default impl,
/// so a minimal backend only writes `dof`, `joint_positions`, `read_state`, and
/// the command setter(s) it supports. Commands MUST validate (enabled, not
/// e-stopped, length, finiteness, mode-active) before acting.
pub trait RobotBackend: Send {
    fn dof(&self) -> usize;

    fn joint_names(&self) -> Vec<String> {
        (0..self.dof()).map(|i| format!("joint_{i}")).collect()
    }

    // ---- lifecycle / safety (no-ops are valid for a trivial sim) ----
    fn enable(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn disable(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn is_enabled(&self) -> bool {
        true
    }
    /// Latching: zeros commands, drops out of enabled, blocks commands until cleared.
    ///
    /// No safe default — a silent no-op would let a loop keep driving a backend it
    /// believes is de-energized. Every real backend MUST override this; the default
    /// fails loudly so an un-implemented e-stop can never be mistaken for a working one.
    fn estop(&mut self) -> Result<(), Error> {
        Err(Error::Backend(
            "estop() not implemented by this backend".into(),
        ))
    }
    fn clear_estop(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn is_estopped(&self) -> bool {
        false
    }

    // ---- mode ----
    fn mode(&self) -> ControlMode {
        ControlMode::Position
    }
    fn set_mode(&mut self, mode: ControlMode) -> Result<(), Error> {
        match mode {
            ControlMode::Position => Ok(()),
            m => Err(Error::UnsupportedMode(m)),
        }
    }

    // ---- commands (validate first) ----
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error>;
    fn command_joint_velocities(&mut self, qd: &[f64]) -> Result<(), Error> {
        let _ = qd;
        Err(Error::UnsupportedMode(ControlMode::Velocity))
    }
    fn command_joint_torques(&mut self, tau: &[f64]) -> Result<(), Error> {
        let _ = tau;
        Err(Error::UnsupportedMode(ControlMode::Torque))
    }

    // ---- state readback ----
    /// Atomic snapshot — the one method a real backend must make self-consistent.
    fn read_state(&mut self) -> Result<JointState, Error>;

    /// Last-known positions (kept from the original trait; cheap `&self` access).
    fn joint_positions(&self) -> Vec<f64>;

    /// Advance the backend's own dynamics by `dt`. Sim backends integrate; real
    /// backends are a no-op (hardware advances itself). Tick-driven.
    fn step(&mut self, dt: f64) -> Result<(), Error> {
        let _ = dt;
        Ok(())
    }
}

// ===== shared validation helpers =====

#[inline]
pub(crate) fn check_len(got: usize, expected: usize) -> Result<(), Error> {
    if got != expected {
        return Err(Error::DofMismatch { expected, got });
    }
    Ok(())
}

#[inline]
pub(crate) fn check_finite(what: &'static str, xs: &[f64]) -> Result<(), Error> {
    if !xs.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite { what });
    }
    Ok(())
}
