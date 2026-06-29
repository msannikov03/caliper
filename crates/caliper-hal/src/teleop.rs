//! Teleoperation as [`Setpoint`] sources — no separate loop. A scripted pose
//! stream, a leader-follower mapping (the classic LeRobot teleop, fully testable
//! with two sim backends), and an integrated velocity jog for live UI control.

use crate::RobotBackend;
use crate::setpoint::{Setpoint, Target};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ticks a jog velocity may persist without a refresh before [`JogSource`] stops
/// integrating it and yields `None` (so the loop watchdog holds a dead UI).
const DEFAULT_JOG_TICK_BUDGET: u64 = 100;

/// Affine leader→follower joint map:
/// `follower_q[i] = scale[i] * leader_q[perm[i]] + offset[i]`.
#[derive(Clone, Debug)]
pub struct JointMap {
    pub perm: Vec<usize>,
    pub scale: Vec<f64>,
    pub offset: Vec<f64>,
}
impl JointMap {
    pub fn identity(n: usize) -> Self {
        Self {
            perm: (0..n).collect(),
            scale: vec![1.0; n],
            offset: vec![0.0; n],
        }
    }
    pub fn apply(&self, leader_q: &[f64]) -> Vec<f64> {
        (0..self.perm.len())
            .map(|i| self.scale[i] * leader_q[self.perm[i]] + self.offset[i])
            .collect()
    }
    pub fn dof(&self) -> usize {
        self.perm.len()
    }
}

/// Leader-follower teleop: each tick, read the leader backend's measured pose and
/// map it to a follower target. Pure-sim testable (two backends, no hardware).
pub struct LeaderFollowerSource {
    leader: Box<dyn RobotBackend>,
    map: JointMap,
}
impl LeaderFollowerSource {
    pub fn new(leader: Box<dyn RobotBackend>, map: JointMap) -> Self {
        Self { leader, map }
    }
    /// Mutable access to the leader (e.g. to drive it from a scripted stream).
    pub fn leader_mut(&mut self) -> &mut dyn RobotBackend {
        self.leader.as_mut()
    }
}
impl Setpoint for LeaderFollowerSource {
    fn target(&mut self, _tick: u64, _t: f64) -> Option<Target> {
        let q = self.leader.read_state().ok()?.q;
        Some(Target::hold(self.map.apply(&q)))
    }
    fn dof(&self) -> usize {
        self.map.dof()
    }
}

/// A scripted sequence of poses, each held for `hold` ticks, optionally wrapping.
pub struct ScriptedSource {
    poses: Vec<Vec<f64>>,
    hold: u64,
    wrap: bool,
}
impl ScriptedSource {
    pub fn new(poses: Vec<Vec<f64>>, hold: u64, wrap: bool) -> Self {
        Self {
            poses,
            hold: hold.max(1),
            wrap,
        }
    }
}
impl Setpoint for ScriptedSource {
    fn target(&mut self, tick: u64, _t: f64) -> Option<Target> {
        if self.poses.is_empty() {
            return None;
        }
        let idx = (tick / self.hold) as usize;
        let idx = if self.wrap {
            idx % self.poses.len()
        } else if idx >= self.poses.len() {
            self.poses.len() - 1
        } else {
            idx
        };
        Some(Target::hold(self.poses[idx].clone()))
    }
    fn dof(&self) -> usize {
        self.poses.first().map(|p| p.len()).unwrap_or(0)
    }
}

/// A shared joint-space velocity command (rad/s). Set from a UI; integrated by a
/// [`JogSource`].
#[derive(Clone)]
pub struct JogHandle {
    v: Arc<Mutex<Vec<f64>>>,
    /// Freshness version, bumped on every `set`.
    seq: Arc<AtomicU64>,
}
impl JogHandle {
    pub fn new(n: usize) -> Self {
        Self {
            v: Arc::new(Mutex::new(vec![0.0; n])),
            seq: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn set(&self, v: Vec<f64>) {
        if let Ok(mut g) = self.v.lock() {
            *g = v;
            self.seq.fetch_add(1, Ordering::Release);
        }
    }
    fn get(&self) -> Vec<f64> {
        self.v.lock().map(|g| g.clone()).unwrap_or_default()
    }
    fn version(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }
}

/// Integrates a [`JogHandle`] velocity into a moving position target:
/// `q ← q + v·dt`. Safety clamping is the monitor's job, not the jogger's.
pub struct JogSource {
    q: Vec<f64>,
    handle: JogHandle,
    dt: f64,
    /// Max ticks a jog velocity may persist without a refresh before going stale.
    budget: u64,
    last_seq: u64,
    last_fresh_tick: u64,
}
impl JogSource {
    pub fn new(q0: Vec<f64>, dt: f64) -> (Self, JogHandle) {
        let h = JogHandle::new(q0.len());
        (
            Self {
                q: q0,
                handle: h.clone(),
                dt,
                budget: DEFAULT_JOG_TICK_BUDGET,
                last_seq: 0,
                last_fresh_tick: 0,
            },
            h,
        )
    }
    /// Override the staleness budget (ticks a jog velocity may persist un-refreshed).
    pub fn with_tick_budget(mut self, ticks: u64) -> Self {
        self.budget = ticks;
        self
    }
}
impl Setpoint for JogSource {
    fn target(&mut self, tick: u64, _t: f64) -> Option<Target> {
        let cur = self.handle.version();
        if cur != self.last_seq {
            self.last_seq = cur;
            self.last_fresh_tick = tick;
        }
        // Stale jog command (dead/frozen UI) → stop integrating, let the watchdog hold.
        if tick.saturating_sub(self.last_fresh_tick) > self.budget {
            return None;
        }
        let v = self.handle.get();
        for i in 0..self.q.len() {
            // Skip non-finite components: a NaN/inf would permanently poison `q`.
            let vi = v.get(i).copied().unwrap_or(0.0);
            if vi.is_finite() {
                self.q[i] += vi * self.dt;
            }
        }
        Some(Target::hold(self.q.clone()))
    }
    fn dof(&self) -> usize {
        self.q.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimBackend;

    #[test]
    fn joint_map_affine() {
        let m = JointMap {
            perm: vec![1, 0],
            scale: vec![2.0, -1.0],
            offset: vec![0.1, 0.0],
        };
        assert_eq!(m.apply(&[0.3, 0.5]), vec![2.0 * 0.5 + 0.1, -0.3]);
    }

    #[test]
    fn leader_follower_mirrors_leader() {
        let mut leader = SimBackend::new(2);
        leader.command_joint_positions(&[0.4, -0.2]).unwrap();
        let mut src = LeaderFollowerSource::new(Box::new(leader), JointMap::identity(2));
        let t = src.target(0, 0.0).unwrap();
        assert_eq!(t.q, vec![0.4, -0.2]);
    }

    #[test]
    fn scripted_steps_and_wraps() {
        let mut s = ScriptedSource::new(vec![vec![0.0], vec![1.0]], 2, true);
        assert_eq!(s.target(0, 0.0).unwrap().q, vec![0.0]);
        assert_eq!(s.target(1, 0.0).unwrap().q, vec![0.0]);
        assert_eq!(s.target(2, 0.0).unwrap().q, vec![1.0]);
        assert_eq!(s.target(4, 0.0).unwrap().q, vec![0.0]); // wrapped
    }

    #[test]
    fn jog_integrates_velocity() {
        let (mut src, handle) = JogSource::new(vec![0.0, 0.0], 0.1);
        handle.set(vec![1.0, -2.0]);
        let t1 = src.target(0, 0.0).unwrap();
        assert!((t1.q[0] - 0.1).abs() < 1e-12 && (t1.q[1] + 0.2).abs() < 1e-12);
        let t2 = src.target(1, 0.1).unwrap();
        assert!((t2.q[0] - 0.2).abs() < 1e-12 && (t2.q[1] + 0.4).abs() < 1e-12);
    }

    #[test]
    fn jog_skips_nonfinite_velocity() {
        let (mut src, handle) = JogSource::new(vec![0.0, 0.0], 0.1);
        handle.set(vec![f64::NAN, 2.0]);
        let t = src.target(0, 0.0).unwrap();
        assert_eq!(t.q[0], 0.0); // NaN component skipped, not integrated
        assert!((t.q[1] - 0.2).abs() < 1e-12);
        // q stays finite on subsequent ticks (never poisoned)
        let t2 = src.target(1, 0.1).unwrap();
        assert!(t2.q[0].is_finite() && t2.q[1].is_finite());
    }

    #[test]
    fn jog_goes_stale_without_refresh() {
        let (src, handle) = JogSource::new(vec![0.0], 0.1);
        let mut src = src.with_tick_budget(2);
        handle.set(vec![1.0]);
        assert!(src.target(0, 0.0).is_some());
        assert!(src.target(2, 0.0).is_some()); // diff == budget: still fresh
        assert!(src.target(3, 0.0).is_none()); // diff > budget: stale → watchdog
    }
}
