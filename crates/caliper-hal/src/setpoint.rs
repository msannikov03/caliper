//! Per-tick target sources for the [`ControlLoop`](crate::ControlLoop). ONE seam
//! — `Setpoint::target(tick, t)` returns the desired `(q, qd, qdd)` this tick, or
//! `None` when the source has nothing to issue (the loop then runs the watchdog).
//! Teleop is just more `Setpoint` impls (see [`crate::teleop`]); there is no
//! second loop.

use caliper_motion::Trajectory;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ticks a teleop target may persist without a refresh before it is treated as
/// stale. Once stale, [`TeleopSetpoint::target`] returns `None`, so the loop runs
/// the command watchdog (Hold/EStop) on a dead or frozen teleop link.
const DEFAULT_TELEOP_TICK_BUDGET: u64 = 100;

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
///
/// Freshness: every `set`/`set_target` bumps a shared sequence counter. If the
/// target is not refreshed within `budget` ticks, [`target`](Setpoint::target)
/// returns `None` instead of the stale value — the loop then runs its command
/// watchdog, so a dead or frozen teleop source can be Held or e-stopped rather
/// than commanding the last pose forever.
#[derive(Clone)]
pub struct TeleopSetpoint {
    dof: usize,
    shared: Arc<Mutex<Target>>,
    /// Shared freshness version, bumped on every `set`/`set_target`.
    seq: Arc<AtomicU64>,
    /// Max ticks a target may persist without a refresh before going stale.
    budget: u64,
    /// Per-instance: last seen `seq` and the tick at which it last changed.
    last_seq: u64,
    last_fresh_tick: u64,
}
impl TeleopSetpoint {
    pub fn new(q0: Vec<f64>) -> Self {
        let dof = q0.len();
        Self {
            dof,
            shared: Arc::new(Mutex::new(Target::hold(q0))),
            seq: Arc::new(AtomicU64::new(0)),
            budget: DEFAULT_TELEOP_TICK_BUDGET,
            last_seq: 0,
            last_fresh_tick: 0,
        }
    }
    /// Override the staleness budget (ticks a target may persist un-refreshed).
    pub fn with_tick_budget(mut self, ticks: u64) -> Self {
        self.budget = ticks;
        self
    }
    /// Push a new position target (zero feed-forward).
    pub fn set(&self, q: Vec<f64>) {
        if let Ok(mut g) = self.shared.lock() {
            *g = Target::hold(q);
            self.seq.fetch_add(1, Ordering::Release);
        }
    }
    /// Push a full target (with feed-forward).
    pub fn set_target(&self, t: Target) {
        if let Ok(mut g) = self.shared.lock() {
            *g = t;
            self.seq.fetch_add(1, Ordering::Release);
        }
    }
    pub fn handle(&self) -> Arc<Mutex<Target>> {
        self.shared.clone()
    }
}
impl Setpoint for TeleopSetpoint {
    fn target(&mut self, tick: u64, _t: f64) -> Option<Target> {
        let cur = self.seq.load(Ordering::Acquire);
        if cur != self.last_seq {
            self.last_seq = cur;
            self.last_fresh_tick = tick;
        }
        // Stale: not refreshed within the budget → hand off to the watchdog.
        if tick.saturating_sub(self.last_fresh_tick) > self.budget {
            return None;
        }
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

    #[test]
    fn teleop_goes_stale_without_refresh() {
        let mut s = TeleopSetpoint::new(vec![0.0]).with_tick_budget(3);
        assert!(s.target(0, 0.0).is_some());
        assert!(s.target(3, 0.0).is_some()); // diff == budget: still fresh
        assert!(s.target(4, 0.0).is_none()); // diff > budget: stale → watchdog
        // a refresh revives it, then it can go stale again
        s.set(vec![0.1]);
        assert!(s.target(5, 0.0).is_some());
        assert!(s.target(9, 0.0).is_none());
    }
}
