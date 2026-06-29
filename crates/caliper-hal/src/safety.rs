//! The single safety enforcement core. [`SafetyMonitor`] is PURE and clock-free:
//! `gate(desired, dt)` clamps a target to joint limits, rate-limits the per-tick
//! step to `vmax·dt`, and refuses motion while e-stopped; `tick_idle` advances a
//! command watchdog. The [`ControlLoop`](crate::ControlLoop) calls it inline;
//! [`SafetyBackend`] is the same logic as a decorator for loop-bypassing callers,
//! and also runs registered [`SafetyCheck`]s (the collision hook, kept rapier-free
//! because the check is an object-safe trait implemented in `caliper-collision`).

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite};
use caliper_dynamics::rnea;
use caliper_model::Model;
use nalgebra::Vector3;

/// What the watchdog does when commands stop arriving.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WatchdogAction {
    /// Re-issue the last safe command (default — fail-safe hold).
    #[default]
    Hold,
    /// Latch an e-stop.
    EStop,
}

/// Per-joint safety envelope. `from_model` pulls position/velocity/effort limits
/// straight off the compiled [`Model`].
#[derive(Clone, Debug)]
pub struct SafetyConfig {
    pub pos: Vec<Option<(f64, f64)>>,
    pub vmax: Vec<Option<f64>>,
    pub effort: Vec<Option<f64>>,
    /// Watchdog timeout in seconds (`None`/≤0 disables). Trips after this much
    /// idle (no command) elapses.
    pub watchdog_timeout: Option<f64>,
    pub watchdog_action: WatchdogAction,
}

impl SafetyConfig {
    pub fn from_model(model: &Model) -> Self {
        Self {
            pos: model.limits.clone(),
            vmax: model.vel_limit.clone(),
            effort: model.effort_limit.clone(),
            watchdog_timeout: None,
            watchdog_action: WatchdogAction::Hold,
        }
    }
    pub fn dof(&self) -> usize {
        self.pos.len()
    }
    pub fn with_watchdog(mut self, timeout: f64, action: WatchdogAction) -> Self {
        self.watchdog_timeout = (timeout > 0.0).then_some(timeout);
        self.watchdog_action = action;
        self
    }
}

/// Outcome of a single `gate`/`tick_idle`. Booleans are per-call; the monitor
/// keeps a monotonic `warn_count` across calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    pub clamped_position: bool,
    pub limited_velocity: bool,
    pub estopped: bool,
    pub watchdog_tripped: bool,
}

impl Verdict {
    pub fn ok(&self) -> bool {
        !(self.clamped_position || self.limited_velocity || self.estopped || self.watchdog_tripped)
    }
}

/// A safety fault that maps to a hard backend error.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum SafetyError {
    #[error("emergency stop active")]
    EStop,
    #[error("effort fault at joint {joint}: |{tau:.3}| > cap {cap:.3}")]
    EffortFault { joint: usize, tau: f64, cap: f64 },
    #[error("safety check rejected: {0}")]
    Check(String),
}

impl From<SafetyError> for Error {
    fn from(e: SafetyError) -> Self {
        match e {
            SafetyError::EStop => Error::EStopActive,
            SafetyError::Check(r) => Error::Collision(r),
            other => Error::Backend(other.to_string()),
        }
    }
}

/// The pure safety core. Holds the last *sanitized* command as the rate-limit
/// anchor and the e-stop latch. No wall-clock, no I/O.
#[derive(Clone, Debug)]
pub struct SafetyMonitor {
    cfg: SafetyConfig,
    q_prev: Vec<f64>,
    estopped: bool,
    idle: f64,
    pub warn_count: u64,
}

impl SafetyMonitor {
    /// Anchor the rate limiter at `q0` (typically the initial measured pose). The
    /// anchor is built defensively at exactly `cfg.dof()` length: missing entries
    /// default to 0, non-finite values are sanitized, and every component is clamped
    /// into its position limit so a bad/short `q0` can never poison or panic `gate`.
    pub fn new(cfg: SafetyConfig, q0: &[f64]) -> Self {
        let n = cfg.dof();
        let mut q_prev = vec![0.0; n];
        for (i, slot) in q_prev.iter_mut().enumerate() {
            let mut qi = q0.get(i).copied().unwrap_or(0.0);
            if !qi.is_finite() {
                qi = 0.0;
            }
            if let Some(Some((lo, hi))) = cfg.pos.get(i).copied() {
                qi = qi.clamp(lo, hi);
            }
            *slot = qi;
        }
        Self {
            q_prev,
            cfg,
            estopped: false,
            idle: 0.0,
            warn_count: 0,
        }
    }
    pub fn dof(&self) -> usize {
        self.cfg.dof()
    }
    pub fn config(&self) -> &SafetyConfig {
        &self.cfg
    }
    pub fn last(&self) -> &[f64] {
        &self.q_prev
    }
    pub fn is_estopped(&self) -> bool {
        self.estopped
    }
    pub fn estop(&mut self) {
        self.estopped = true;
    }
    pub fn clear_estop(&mut self) {
        self.estopped = false;
        self.idle = 0.0;
    }

    /// Sanitize a desired target: clamp to position limits, then rate-limit the
    /// step to `vmax·dt` from the last sanitized command. Returns the safe target
    /// and a [`Verdict`]. While e-stopped, returns the held pose with no motion.
    pub fn gate(&mut self, desired: &[f64], dt: f64) -> (Vec<f64>, Verdict) {
        let n = self.cfg.dof();
        let mut v = Verdict::default();
        if self.estopped {
            v.estopped = true;
            self.warn_count += 1;
            return (self.q_prev.clone(), v);
        }
        let mut out = vec![0.0; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut target = desired.get(i).copied().unwrap_or(self.q_prev[i]);
            if !target.is_finite() {
                target = self.q_prev[i];
            }
            // 1) position clamp
            if let Some(Some((lo, hi))) = self.cfg.pos.get(i).copied() {
                let c = target.clamp(lo, hi);
                if c != target {
                    v.clamped_position = true;
                }
                target = c;
            }
            // 2) velocity (per-tick step) clamp
            if let Some(Some(vmax)) = self.cfg.vmax.get(i).copied() {
                let max_step = vmax.abs() * dt.max(0.0);
                let delta = target - self.q_prev[i];
                if delta.abs() > max_step {
                    target = self.q_prev[i] + delta.signum() * max_step;
                    v.limited_velocity = true;
                }
            }
            *slot = target;
        }
        if v.clamped_position || v.limited_velocity {
            self.warn_count += 1;
        }
        self.q_prev.copy_from_slice(&out);
        self.idle = 0.0;
        (out, v)
    }

    /// Advance the watchdog when a tick issues NO command. Returns a [`Verdict`]
    /// with `watchdog_tripped`/`estopped` set if the timeout elapsed. On a `Hold`
    /// the safe target to re-issue is [`last`](Self::last).
    pub fn tick_idle(&mut self, dt: f64) -> Verdict {
        let mut v = Verdict::default();
        if self.estopped {
            v.estopped = true;
            return v;
        }
        self.idle += dt.max(0.0);
        if let Some(timeout) = self.cfg.watchdog_timeout
            && self.idle > timeout
        {
            v.watchdog_tripped = true;
            self.warn_count += 1;
            if self.cfg.watchdog_action == WatchdogAction::EStop {
                self.estopped = true;
                v.estopped = true;
            }
        }
        v
    }

    /// Static effort check: predict the holding torque `rnea(q, 0, 0, g)` and
    /// latch an e-stop if any joint exceeds its effort cap. This is a STATIC
    /// (gravity-only) prediction — it does NOT catch dynamic/contact overload.
    pub fn check_effort(
        &mut self,
        model: &Model,
        q: &[f64],
        gravity: &Vector3<f64>,
    ) -> Result<(), SafetyError> {
        let zeros = vec![0.0; model.ndof];
        let tau = rnea(model, q, &zeros, &zeros, gravity)
            .map_err(|e| SafetyError::Check(e.to_string()))?;
        for i in 0..model.ndof {
            if let Some(Some(cap)) = self.cfg.effort.get(i).copied()
                && tau[i].abs() > cap
            {
                self.estopped = true;
                return Err(SafetyError::EffortFault {
                    joint: i,
                    tau: tau[i],
                    cap,
                });
            }
        }
        Ok(())
    }
}

/// An object-safe configuration check (e.g. collision) over a target pose.
/// Implemented by `caliper_collision::CollisionModel`, keeping caliper-hal
/// free of rapier/parry.
pub trait SafetyCheck: Send {
    /// `Ok(())` if `q` is acceptable, `Err(reason)` to reject the command.
    fn check(&self, q: &[f64]) -> Result<(), String>;
}

/// A [`RobotBackend`] decorator that gates every position command through a
/// [`SafetyMonitor`] and any registered [`SafetyCheck`]s before delegating.
pub struct SafetyBackend<B: RobotBackend> {
    inner: B,
    monitor: SafetyMonitor,
    checks: Vec<Box<dyn SafetyCheck>>,
    dt: f64,
}

impl<B: RobotBackend> SafetyBackend<B> {
    pub fn new(inner: B, monitor: SafetyMonitor, dt: f64) -> Self {
        Self {
            inner,
            monitor,
            checks: Vec::new(),
            dt,
        }
    }
    pub fn add_check(&mut self, check: Box<dyn SafetyCheck>) {
        self.checks.push(check);
    }
    pub fn monitor(&self) -> &SafetyMonitor {
        &self.monitor
    }
    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: RobotBackend> RobotBackend for SafetyBackend<B> {
    fn dof(&self) -> usize {
        self.inner.dof()
    }
    fn joint_names(&self) -> Vec<String> {
        self.inner.joint_names()
    }
    fn enable(&mut self) -> Result<(), Error> {
        self.inner.enable()
    }
    fn disable(&mut self) -> Result<(), Error> {
        self.inner.disable()
    }
    fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }
    fn estop(&mut self) -> Result<(), Error> {
        self.monitor.estop();
        self.inner.estop()
    }
    fn clear_estop(&mut self) -> Result<(), Error> {
        self.monitor.clear_estop();
        self.inner.clear_estop()
    }
    fn is_estopped(&self) -> bool {
        self.monitor.is_estopped() || self.inner.is_estopped()
    }
    fn mode(&self) -> ControlMode {
        self.inner.mode()
    }
    fn set_mode(&mut self, mode: ControlMode) -> Result<(), Error> {
        self.inner.set_mode(mode)
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        let (safe, verdict) = self.monitor.gate(q, self.dt);
        if verdict.estopped {
            return Err(Error::EStopActive);
        }
        for c in &self.checks {
            c.check(&safe).map_err(Error::Collision)?;
        }
        self.inner.command_joint_positions(&safe)
    }
    fn command_joint_velocities(&mut self, qd: &[f64]) -> Result<(), Error> {
        // Mirror the position path: never let a non-position command bypass the
        // monitor. Reject while e-stopped, reject non-finite, then clamp each
        // joint to its velocity cap (where the model defines one) before delegating.
        if self.monitor.is_estopped() {
            return Err(Error::EStopActive);
        }
        check_finite("joint velocities", qd)?;
        let cfg = self.monitor.config();
        let safe: Vec<f64> = qd
            .iter()
            .enumerate()
            .map(|(i, &v)| match cfg.vmax.get(i).copied().flatten() {
                Some(c) => v.clamp(-c.abs(), c.abs()),
                None => v,
            })
            .collect();
        self.inner.command_joint_velocities(&safe)
    }
    fn command_joint_torques(&mut self, tau: &[f64]) -> Result<(), Error> {
        if self.monitor.is_estopped() {
            return Err(Error::EStopActive);
        }
        check_finite("joint torques", tau)?;
        self.inner.command_joint_torques(tau)
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        self.inner.read_state()
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.inner.joint_positions()
    }
    fn step(&mut self, dt: f64) -> Result<(), Error> {
        self.inner.step(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper_dynamics::GRAVITY_EARTH;
    use std::path::Path;

    fn cfg2() -> SafetyConfig {
        SafetyConfig {
            pos: vec![Some((-1.0, 1.0)), Some((-1.0, 1.0))],
            vmax: vec![Some(2.0), Some(2.0)],
            effort: vec![None, None],
            watchdog_timeout: None,
            watchdog_action: WatchdogAction::Hold,
        }
    }

    #[test]
    fn clamps_position() {
        let mut m = SafetyMonitor::new(cfg2(), &[0.0, 0.0]);
        let (q, v) = m.gate(&[5.0, -5.0], 1.0); // dt=1 → vmax step = 2.0
        // clamp to ±1 then rate-limit to ±2 from 0 → ±1 wins
        assert_eq!(q, vec![1.0, -1.0]);
        assert!(v.clamped_position);
        assert_eq!(m.warn_count, 1);
    }

    #[test]
    fn rate_limits_velocity() {
        let mut m = SafetyMonitor::new(cfg2(), &[0.0, 0.0]);
        let (q, v) = m.gate(&[0.9, 0.9], 0.1); // max step = 2.0*0.1 = 0.2
        assert!((q[0] - 0.2).abs() < 1e-12 && (q[1] - 0.2).abs() < 1e-12);
        assert!(v.limited_velocity && !v.clamped_position);
    }

    #[test]
    fn estop_is_sticky_and_holds() {
        let mut m = SafetyMonitor::new(cfg2(), &[0.3, -0.3]);
        m.estop();
        let (q, v) = m.gate(&[0.5, 0.5], 0.1);
        assert_eq!(q, vec![0.3, -0.3]); // held, no motion
        assert!(v.estopped);
        m.clear_estop();
        let (q2, _) = m.gate(&[0.35, -0.3], 0.1);
        assert!((q2[0] - 0.35).abs() < 1e-12);
    }

    #[test]
    fn watchdog_trips_and_estops() {
        let cfg = cfg2().with_watchdog(0.05, WatchdogAction::EStop);
        let mut m = SafetyMonitor::new(cfg, &[0.0, 0.0]);
        assert!(!m.tick_idle(0.03).watchdog_tripped); // 0.03 < 0.05
        let v = m.tick_idle(0.03); // total 0.06 > 0.05
        assert!(v.watchdog_tripped && v.estopped);
        assert!(m.is_estopped());
    }

    /// A minimal backend that accepts velocity/torque and records the last command,
    /// so we can prove the [`SafetyBackend`] gate (estop + finiteness + vmax clamp)
    /// runs BEFORE delegation on the non-position paths.
    #[derive(Default)]
    struct RecBackend {
        dof: usize,
        last_qd: Vec<f64>,
        last_tau: Vec<f64>,
        estopped: bool,
    }
    impl RobotBackend for RecBackend {
        fn dof(&self) -> usize {
            self.dof
        }
        fn estop(&mut self) -> Result<(), Error> {
            self.estopped = true;
            Ok(())
        }
        fn clear_estop(&mut self) -> Result<(), Error> {
            self.estopped = false;
            Ok(())
        }
        fn is_estopped(&self) -> bool {
            self.estopped
        }
        fn command_joint_positions(&mut self, _q: &[f64]) -> Result<(), Error> {
            Ok(())
        }
        fn command_joint_velocities(&mut self, qd: &[f64]) -> Result<(), Error> {
            self.last_qd = qd.to_vec();
            Ok(())
        }
        fn command_joint_torques(&mut self, tau: &[f64]) -> Result<(), Error> {
            self.last_tau = tau.to_vec();
            Ok(())
        }
        fn read_state(&mut self) -> Result<JointState, Error> {
            Ok(JointState {
                tick: 0,
                t: 0.0,
                q: vec![0.0; self.dof],
                qd: None,
                tau: None,
            })
        }
        fn joint_positions(&self) -> Vec<f64> {
            vec![0.0; self.dof]
        }
    }

    #[test]
    fn safety_backend_gates_velocity_and_torque() {
        let mut b = SafetyBackend::new(
            RecBackend {
                dof: 2,
                ..Default::default()
            },
            SafetyMonitor::new(cfg2(), &[0.0, 0.0]),
            0.1,
        );
        // velocity clamped to vmax (cfg2 vmax = 2.0)
        b.command_joint_velocities(&[5.0, -5.0]).unwrap();
        assert_eq!(b.inner().last_qd, vec![2.0, -2.0]);
        // non-finite rejected before delegation
        assert!(matches!(
            b.command_joint_velocities(&[f64::NAN, 0.0]),
            Err(Error::NonFinite { .. })
        ));
        assert!(matches!(
            b.command_joint_torques(&[0.0, f64::INFINITY]),
            Err(Error::NonFinite { .. })
        ));
        // after e-stop, both non-position paths are refused
        b.estop().unwrap();
        assert!(matches!(
            b.command_joint_velocities(&[0.1, 0.1]),
            Err(Error::EStopActive)
        ));
        assert!(matches!(
            b.command_joint_torques(&[0.1, 0.1]),
            Err(Error::EStopActive)
        ));
    }

    #[test]
    fn monitor_new_is_defensive() {
        // short, non-finite, out-of-limit q0 → anchored at dof length, sanitized & clamped
        let m = SafetyMonitor::new(cfg2(), &[5.0, f64::NAN]);
        assert_eq!(m.last(), &[1.0, 0.0]); // 5.0 clamped to +1.0; NaN→0.0 (within ±1)
        // gate must not panic with mismatched-length cfg vectors
        let mut cfg = cfg2();
        cfg.vmax = vec![Some(2.0)]; // shorter than pos (len 2)
        let mut m2 = SafetyMonitor::new(cfg, &[0.0, 0.0]);
        let (_q, _v) = m2.gate(&[0.1, 0.1], 0.1);
    }

    #[test]
    fn effort_fault_estops() {
        let model = Model::from_urdf(Path::new(&format!(
            "{}/../../oracle/fixtures/robots/dyn_pendulum2.urdf",
            env!("CARGO_MANIFEST_DIR")
        )))
        .unwrap();
        // A tiny effort cap is exceeded by the gravity hold torque on a tilted arm.
        let mut cfg = SafetyConfig::from_model(&model);
        cfg.effort = vec![Some(1e-4); model.ndof];
        let mut m = SafetyMonitor::new(cfg, &vec![0.0; model.ndof]);
        let err = m
            .check_effort(&model, &[0.6, 0.4], &GRAVITY_EARTH)
            .unwrap_err();
        assert!(matches!(err, SafetyError::EffortFault { .. }));
        assert!(m.is_estopped());
    }
}
