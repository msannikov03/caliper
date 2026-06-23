//! Per-tick target sources for the [`ControlLoop`](crate::ControlLoop). ONE seam
//! — `Setpoint::target(tick, t)` returns the desired `(q, qd, qdd)` this tick, or
//! `None` when the source has nothing to issue (the loop then runs the watchdog).
//! Teleop is just more `Setpoint` impls (see [`crate::teleop`]); there is no
//! second loop.

use caliper_motion::Trajectory;
use std::sync::{Arc, Mutex};

/// A desired joint-space target. `qd`/`qdd` are feed-forward terms (zero for a
/// pure position hold).
#[derive(Clone, Debug)]
pub struct Target {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
}

impl Target {
    /// A stationary target (zero velocity/acceleration feed-forward).
    pub fn hold(q: Vec<f64>) -> Self {
        let n = q.len();
        Target {
            q,
            qd: vec![0.0; n],
            qdd: vec![0.0; n],
        }
    }
}

/// A per-tick target source. `Send` so a loop owning it stays `Send`.
pub trait Setpoint: Send {
    /// Desired target at tick `tick` / time `t`, or `None` to issue nothing.
    fn target(&mut self, tick: u64, t: f64) -> Option<Target>;
    fn dof(&self) -> usize;
}

/// Hold a fixed pose forever.
pub struct HoldSetpoint {
    q: Vec<f64>,
}
impl HoldSetpoint {
    pub fn new(q: Vec<f64>) -> Self {
        Self { q }
    }
}
impl Setpoint for HoldSetpoint {
    fn target(&mut self, _tick: u64, _t: f64) -> Option<Target> {
        Some(Target::hold(self.q.clone()))
    }
    fn dof(&self) -> usize {
        self.q.len()
    }
}

/// Track a planned [`Trajectory`] by sampling it at the loop time. After the
/// trajectory ends it holds the final pose (so a tracking loop settles).
pub struct TrajectorySetpoint {
    traj: Trajectory,
    hold_after: bool,
}
impl TrajectorySetpoint {
    pub fn new(traj: Trajectory) -> Self {
        Self {
            traj,
            hold_after: true,
        }
    }
    /// Stop issuing targets (return `None`) once the trajectory ends.
    pub fn no_hold(mut self) -> Self {
        self.hold_after = false;
        self
    }
    pub fn duration(&self) -> f64 {
        self.traj.duration()
    }
}
impl Setpoint for TrajectorySetpoint {
    fn target(&mut self, _tick: u64, t: f64) -> Option<Target> {
        if t > self.traj.duration() && !self.hold_after {
            return None;
        }
        let s = self.traj.sample(t.min(self.traj.duration()));
        Some(Target {
            q: s.q,
            qd: s.qd,
            qdd: s.qdd,
        })
    }
    fn dof(&self) -> usize {
        self.traj.ndof()
    }
}

/// A target driven from outside the loop (e.g. a Studio "set target" command or a
/// live UI). Cloneable handle over a shared cell.
#[derive(Clone)]
pub struct TeleopSetpoint {
    dof: usize,
    shared: Arc<Mutex<Target>>,
}
impl TeleopSetpoint {
    pub fn new(q0: Vec<f64>) -> Self {
        let dof = q0.len();
        Self {
            dof,
            shared: Arc::new(Mutex::new(Target::hold(q0))),
        }
    }
    /// Push a new position target (zero feed-forward).
    pub fn set(&self, q: Vec<f64>) {
        if let Ok(mut g) = self.shared.lock() {
            *g = Target::hold(q);
        }
    }
    /// Push a full target (with feed-forward).
    pub fn set_target(&self, t: Target) {
        if let Ok(mut g) = self.shared.lock() {
            *g = t;
        }
    }
    pub fn handle(&self) -> Arc<Mutex<Target>> {
        self.shared.clone()
    }
}
impl Setpoint for TeleopSetpoint {
    fn target(&mut self, _tick: u64, _t: f64) -> Option<Target> {
        self.shared.lock().ok().map(|g| g.clone())
    }
    fn dof(&self) -> usize {
        self.dof
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_is_constant() {
        let mut s = HoldSetpoint::new(vec![0.1, 0.2]);
        assert_eq!(s.target(0, 0.0).unwrap().q, vec![0.1, 0.2]);
        assert_eq!(s.target(99, 9.9).unwrap().q, vec![0.1, 0.2]);
    }

    #[test]
    fn teleop_reflects_latest_set() {
        let s = TeleopSetpoint::new(vec![0.0, 0.0]);
        let mut c = s.clone();
        assert_eq!(c.target(0, 0.0).unwrap().q, vec![0.0, 0.0]);
        s.set(vec![0.5, -0.5]);
        assert_eq!(c.target(1, 0.01).unwrap().q, vec![0.5, -0.5]);
    }
}
