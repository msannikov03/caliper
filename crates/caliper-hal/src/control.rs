//! The deterministic control loop. Tick-driven, clock-free (`t == tick·dt`), and
//! monomorphized over the backend `B`. Each tick: read state → resolve a target
//! (gated through the [`SafetyMonitor`]) → compute a command for the backend's
//! control mode → write it → (in Torque mode) integrate → emit a [`Frame`].
//!
//! The command law per mode:
//! - **Position**: passthrough `q*` (the backend teleports / a servo tracks).
//! - **Velocity**: `qd* + kp·(q* − q)`.
//! - **Torque**: COMPUTED-TORQUE / inverse-dynamics control —
//!   `τ = M(q)·(q̈* + kp·(q*−q) + kd·(q̇*−q̇)) + (C(q,q̇)·q̇ + g(q))`,
//!   where `M = crba(q)` and `C·q̇ + g = rnea(q,q̇,0,g)`. This cancels the full
//!   nonlinear dynamics, so the closed-loop error obeys `ë + kd·ė + kp·e = 0` with
//!   UNIT effective inertia on every joint. One gain pair is therefore stable on
//!   any robot — a fixed PD instead diverges on low-inertia wrists (the explicit
//!   damping term `h·kd/I` exceeds the integrator's stability limit). Dropping or
//!   sign-flipping the `g(q)` term makes the regulation test fail.

use crate::setpoint::Setpoint;
use crate::{ControlMode, Error, RobotBackend, SafetyConfig, SafetyMonitor};
use caliper_dynamics::{GRAVITY_EARTH, crba, rnea};
use caliper_model::Model;
use nalgebra::{DVector, Vector3};
use std::sync::Arc;

/// PD gains for the velocity/torque command laws.
#[derive(Clone, Copy, Debug)]
pub struct Gains {
    pub kp: f64,
    pub kd: f64,
}
impl Default for Gains {
    // Computed-torque gains define the linear error dynamics `ë + kd·ė + kp·e = 0`:
    // kp = ωn², kd = 2ζωn. Here ωn ≈ 10 rad/s, ζ = 1 (critically damped) → ~0.4 s
    // settling, robust on any robot. Tune per task via `with_gains`.
    fn default() -> Self {
        Gains {
            kp: 100.0,
            kd: 20.0,
        }
    }
}

/// One recorded control step. `measured` is the state read BEFORE the command
/// (LeRobot `observation.state`); `command` is the commanded target q (`action`).
#[derive(Clone, Debug)]
pub struct Frame {
    pub tick: u64,
    pub t: f64,
    pub measured: Vec<f64>,
    pub measured_qd: Vec<f64>,
    pub command: Vec<f64>,
    pub warn: bool,
}

/// Sink for recorded frames (the recorder, or a `VecSink` for tests/replay bake).
pub trait FrameSink {
    fn push(&mut self, f: &Frame);
}

/// A trivial in-memory sink.
#[derive(Default)]
pub struct VecSink {
    pub frames: Vec<Frame>,
}
impl FrameSink for VecSink {
    fn push(&mut self, f: &Frame) {
        self.frames.push(f.clone());
    }
}

/// The deterministic control loop driving a single backend.
pub struct ControlLoop<B: RobotBackend> {
    backend: B,
    model: Arc<Model>,
    gravity: Vector3<f64>,
    gains: Gains,
    monitor: SafetyMonitor,
    dt: f64,
    tick: u64,
}

impl<B: RobotBackend> ControlLoop<B> {
    /// Build a loop, enabling the backend and anchoring the safety monitor at the
    /// current measured pose. `dt` is the fixed control period (seconds).
    pub fn new(mut backend: B, model: Arc<Model>, dt: f64) -> Result<Self, Error> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Backend(format!("control dt must be > 0, got {dt}")));
        }
        backend.enable()?;
        let q0 = backend.read_state()?.q;
        let monitor = SafetyMonitor::new(SafetyConfig::from_model(&model), &q0);
        Ok(Self {
            backend,
            model,
            gravity: GRAVITY_EARTH,
            gains: Gains::default(),
            monitor,
            dt,
            tick: 0,
        })
    }

    pub fn with_gains(mut self, gains: Gains) -> Self {
        self.gains = gains;
        self
    }
    pub fn with_gravity(mut self, g: Vector3<f64>) -> Self {
        self.gravity = g;
        self
    }
    pub fn with_monitor(mut self, m: SafetyMonitor) -> Self {
        self.monitor = m;
        self
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }
    pub fn time(&self) -> f64 {
        self.tick as f64 * self.dt
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
    pub fn monitor(&self) -> &SafetyMonitor {
        &self.monitor
    }
    pub fn gravity(&self) -> Vector3<f64> {
        self.gravity
    }

    /// Latch an e-stop in both the monitor and the backend.
    pub fn estop(&mut self) -> Result<(), Error> {
        self.monitor.estop();
        self.backend.estop()
    }

    /// Advance one tick. Returns the recorded [`Frame`] and pushes it to `sink`.
    pub fn step(
        &mut self,
        sp: &mut dyn Setpoint,
        sink: Option<&mut dyn FrameSink>,
    ) -> Result<Frame, Error> {
        let n = self.model.ndof;
        let t = self.tick as f64 * self.dt;
        let measured = self.backend.read_state()?;
        let q_now = measured.q.clone();
        let qd_now = measured.qd_or_zero();

        // Resolve the target through the safety gate, or hold via the watchdog.
        let (target_q, target_qd, target_qdd, warn) = match sp.target(self.tick, t) {
            Some(tg) => {
                let (safe, v) = self.monitor.gate(&tg.q, self.dt);
                let qd = if tg.qd.len() == n {
                    tg.qd
                } else {
                    vec![0.0; n]
                };
                let qdd = if tg.qdd.len() == n {
                    tg.qdd
                } else {
                    vec![0.0; n]
                };
                (safe, qd, qdd, !v.ok())
            }
            None => {
                let v = self.monitor.tick_idle(self.dt);
                (
                    self.monitor.last().to_vec(),
                    vec![0.0; n],
                    vec![0.0; n],
                    !v.ok(),
                )
            }
        };

        let mode = self.backend.mode();
        let cmd =
            self.compute_command(mode, &target_q, &target_qd, &target_qdd, &q_now, &qd_now)?;
        match mode {
            ControlMode::Torque => {
                self.backend.command_joint_torques(&cmd)?;
                self.backend.step(self.dt)?; // integrate AFTER setting torque
            }
            ControlMode::Position => self.backend.command_joint_positions(&cmd)?,
            ControlMode::Velocity => self.backend.command_joint_velocities(&cmd)?,
        }

        let frame = Frame {
            tick: self.tick,
            t,
            measured: q_now,
            measured_qd: qd_now,
            command: target_q,
            warn,
        };
        if let Some(s) = sink {
            s.push(&frame);
        }
        self.tick += 1;
        Ok(frame)
    }

    /// Run `n` ticks (no recording).
    pub fn run_to(&mut self, sp: &mut dyn Setpoint, n: usize) -> Result<(), Error> {
        for _ in 0..n {
            self.step(sp, None)?;
        }
        Ok(())
    }

    /// Run `n` ticks into a fresh [`VecSink`] and return the recorded frames.
    pub fn run_record(&mut self, sp: &mut dyn Setpoint, n: usize) -> Result<Vec<Frame>, Error> {
        let mut sink = VecSink::default();
        for _ in 0..n {
            self.step(sp, Some(&mut sink))?;
        }
        Ok(sink.frames)
    }

    fn compute_command(
        &self,
        mode: ControlMode,
        target_q: &[f64],
        target_qd: &[f64],
        target_qdd: &[f64],
        q: &[f64],
        qd: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let n = self.model.ndof;
        let kp = self.gains.kp;
        let kd = self.gains.kd;
        let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
        // Output magnitude caps from the SAME model limits the monitor enforces, so
        // the command itself — not just the position reference — is bounded. A large
        // tracking error (far setpoint, perturbed state) otherwise yields an
        // unbounded torque/velocity on the real-robot path.
        let cfg = self.monitor.config();
        let cap = |x: f64, lim: Option<f64>| match lim {
            Some(c) => x.clamp(-c.abs(), c.abs()),
            None => x,
        };
        match mode {
            ControlMode::Position => Ok(target_q.to_vec()),
            ControlMode::Velocity => Ok((0..n)
                .map(|i| cap(at(target_qd, i) + kp * (target_q[i] - q[i]), cfg.vmax[i]))
                .collect()),
            ControlMode::Torque => {
                // computed torque: τ = M·a_des + (C·q̇ + g),  a_des = q̈* + kp·e + kd·ė
                let a_des = DVector::from_iterator(
                    n,
                    (0..n).map(|i| {
                        at(target_qdd, i)
                            + kp * (target_q[i] - q[i])
                            + kd * (at(target_qd, i) - qd[i])
                    }),
                );
                let mmat = crba(&self.model, q)?;
                let bias = rnea(&self.model, q, qd, &vec![0.0; n], &self.gravity)?; // C·q̇ + g
                let tau = &mmat * &a_des + &bias;
                // saturate each joint torque to its effort limit (clamp; safety bound)
                Ok((0..n).map(|i| cap(tau[i], cfg.effort[i])).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PhysicsSimBackend;
    use crate::setpoint::{HoldSetpoint, TrajectorySetpoint};
    use crate::sim::SimBackend;
    use caliper_motion::{MotionLimits, move_j};
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
    fn converges_to_setpoint_under_gravity() {
        // PRIMARY oracle: Torque-mode PD + gravity-FF on a physical 2-DOF pendulum
        // regulates to the setpoint with velocity → 0.
        let m = model("dyn_pendulum2.urdf");
        let mut backend = PhysicsSimBackend::new(m.clone()).unwrap();
        backend.set_state(&[0.6, -0.4], &[0.0, 0.0]).unwrap();
        let mut loopy = ControlLoop::new(backend, m, 1e-3).unwrap();
        let target = vec![0.2, 0.1];
        let mut sp = HoldSetpoint::new(target.clone());
        loopy.run_to(&mut sp, 6000).unwrap();
        let s = loopy.backend().joint_positions();
        let qd = loopy.backend_mut().read_state().unwrap().qd_or_zero();
        let qe = ((s[0] - target[0]).powi(2) + (s[1] - target[1]).powi(2)).sqrt();
        let ve = (qd[0] * qd[0] + qd[1] * qd[1]).sqrt();
        assert!(qe < 1e-3, "position error {qe:e}");
        assert!(ve < 1e-3, "velocity error {ve:e}");
    }

    #[test]
    fn gravity_ff_sign_matters() {
        // With gravity-FF FLIPPED, the loop must NOT hold (guards the Phase-4 sign
        // bug class). We emulate the flip by setting gravity to +g and checking the
        // arm diverges from the setpoint it would otherwise hold.
        let m = model("dyn_pendulum2.urdf");
        let mut backend = PhysicsSimBackend::new(m.clone()).unwrap();
        backend.set_state(&[0.2, 0.1], &[0.0, 0.0]).unwrap();
        // backend gravity stays correct; loop FF gravity is sign-flipped → double gravity
        let mut loopy = ControlLoop::new(backend, m, 1e-3)
            .unwrap()
            .with_gravity(-GRAVITY_EARTH);
        let mut sp = HoldSetpoint::new(vec![0.2, 0.1]);
        loopy.run_to(&mut sp, 2000).unwrap();
        let s = loopy.backend().joint_positions();
        let err = ((s[0] - 0.2).powi(2) + (s[1] - 0.1).powi(2)).sqrt();
        assert!(err > 1e-2, "wrong-sign FF should NOT hold, err={err:e}");
    }

    #[test]
    fn tracks_trajectory_on_kinematic_backend() {
        // Position-mode passthrough: the loop drives a SimBackend exactly along a
        // jerk-limited MOVE_J (rate limiter never bites because move_j respects vmax).
        let m = model("showcase6.urdf");
        let q0 = vec![0.0; 6];
        let q1 = vec![0.4, -0.3, 0.5, 0.2, -0.4, 0.1];
        let lim = MotionLimits {
            vmax: vec![1.0; 6],
            amax: vec![2.0; 6],
            jmax: vec![8.0; 6],
        };
        let traj = move_j(&m, &q0, &q1, &lim).unwrap();
        let dur = traj.duration();
        let backend = SimBackend::new(6).with_positions(&q0).unwrap();
        let mut loopy = ControlLoop::new(backend, m, 1e-3).unwrap();
        let mut sp = TrajectorySetpoint::new(traj.clone());
        let dt = loopy.dt();
        let n = (dur / dt).ceil() as usize + 100;
        let mut worst = 0.0f64;
        for _ in 0..n {
            let f = loopy.step(&mut sp, None).unwrap();
            let want = traj.q_at(f.t);
            let got = loopy.backend().joint_positions();
            for i in 0..6 {
                worst = worst.max((got[i] - want[i]).abs());
            }
        }
        assert!(worst < 5e-3, "tracking error {worst:e}");
        // settled at the goal
        let end = loopy.backend().joint_positions();
        for i in 0..6 {
            assert!((end[i] - q1[i]).abs() < 1e-6, "joint {i} not settled");
        }
    }

    #[test]
    fn deterministic_bitwise() {
        // Two identical runs produce bit-for-bit identical frames (no wall-clock).
        let run = || {
            let m = model("dyn_pendulum2.urdf");
            let mut b = PhysicsSimBackend::new(m.clone()).unwrap();
            b.set_state(&[0.5, -0.2], &[0.0, 0.0]).unwrap();
            let mut loopy = ControlLoop::new(b, m, 1e-3).unwrap();
            let mut sp = HoldSetpoint::new(vec![0.1, 0.0]);
            loopy.run_record(&mut sp, 500).unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa.tick, fb.tick);
            assert_eq!(fa.t.to_bits(), fb.t.to_bits());
            for i in 0..fa.measured.len() {
                assert_eq!(fa.measured[i].to_bits(), fb.measured[i].to_bits());
                assert_eq!(fa.command[i].to_bits(), fb.command[i].to_bits());
            }
        }
    }

    #[test]
    fn torque_saturates_to_effort_cap() {
        // Effort capped FAR below the gravity hold torque → the command is starved,
        // so the arm cannot hold its setpoint and sags. Proves the per-joint torque
        // is actually bounded by the effort limit (not just the position reference).
        let m = model("dyn_pendulum2.urdf");
        let mut cfg = SafetyConfig::from_model(&m);
        cfg.effort = vec![Some(0.05), Some(0.05)]; // 0.05 N·m ≪ gravity torque
        let hold = vec![0.4, 0.3];
        let mut backend = PhysicsSimBackend::new(m.clone()).unwrap();
        backend.set_state(&hold, &[0.0, 0.0]).unwrap();
        let monitor = SafetyMonitor::new(cfg, &hold);
        let mut loopy = ControlLoop::new(backend, m, 1e-3)
            .unwrap()
            .with_monitor(monitor);
        let mut sp = HoldSetpoint::new(hold.clone());
        loopy.run_to(&mut sp, 1500).unwrap();
        let q = loopy.backend().joint_positions();
        let moved = (q[0] - hold[0]).abs().max((q[1] - hold[1]).abs());
        assert!(
            moved > 0.05,
            "effort-starved arm should sag, moved={moved:e}"
        );
        // sanity: with the real (20 N·m) caps it holds instead
        let m2 = model("dyn_pendulum2.urdf");
        let mut b2 = PhysicsSimBackend::new(m2.clone()).unwrap();
        b2.set_state(&hold, &[0.0, 0.0]).unwrap();
        let mut l2 = ControlLoop::new(b2, m2, 1e-3).unwrap();
        l2.run_to(&mut HoldSetpoint::new(hold.clone()), 1500)
            .unwrap();
        let q2 = l2.backend().joint_positions();
        assert!(
            (q2[0] - hold[0]).abs().max((q2[1] - hold[1]).abs()) < 1e-2,
            "uncapped arm should hold"
        );
    }
}
