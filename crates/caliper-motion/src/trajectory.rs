use crate::limits::MotionLimits;
use crate::scurve::ScurveProfile;
use caliper_kinematics::fk_frame;
use caliper_model::Model;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum TrajKind {
    MoveJ,
    MoveL,
    MoveC,
    /// A planned (RRT) path, retimed onto a uniform knot grid (samples like the
    /// Cartesian kinds — via the knot interpolator, not closed-form profiles).
    Plan,
}

#[derive(Clone, Debug)]
pub struct TrajState {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
}

/// Time-synchronized multi-joint trajectory. For MOVE_J this is closed-form
/// (per-dof ScurveProfiles). For MOVE_L/MOVE_C it carries pre-solved knot rows
/// (q/qd/qdd on a uniform dt grid) and samples by C²-preserving interpolation.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub kind: TrajKind,
    pub ndof: usize,
    pub duration: f64,
    pub limits: MotionLimits,
    /// `true` when the whole path was realized; `false` for a best-effort prefix
    /// (Cartesian move truncated at an unreachable / discontinuous sample).
    pub completed: bool,
    /// Path fraction s∈[0,1] actually reached (1.0 for MOVE_J and full Cartesian).
    pub reached: f64,
    // MOVE_J representation:
    q0: Vec<f64>,
    profiles: Vec<ScurveProfile>,
    // Cartesian representation (empty for MOVE_J):
    dt: f64,
    knots_q: Vec<Vec<f64>>,
    knots_qd: Vec<Vec<f64>>,
    knots_qdd: Vec<Vec<f64>>,
}

impl Trajectory {
    pub(crate) fn from_profiles(
        q0: Vec<f64>,
        profiles: Vec<ScurveProfile>,
        duration: f64,
        limits: MotionLimits,
    ) -> Self {
        let ndof = q0.len();
        Trajectory {
            kind: TrajKind::MoveJ,
            ndof,
            duration,
            limits,
            completed: true,
            reached: 1.0,
            q0,
            profiles,
            dt: 0.0,
            knots_q: vec![],
            knots_qd: vec![],
            knots_qdd: vec![],
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_knots(
        kind: TrajKind,
        dt: f64,
        q: Vec<Vec<f64>>,
        qd: Vec<Vec<f64>>,
        qdd: Vec<Vec<f64>>,
        limits: MotionLimits,
        completed: bool,
        reached: f64,
    ) -> Self {
        let ndof = q.first().map(|r| r.len()).unwrap_or(0);
        let duration = if q.is_empty() {
            0.0
        } else {
            (q.len() - 1) as f64 * dt
        };
        Trajectory {
            kind,
            ndof,
            duration,
            limits,
            completed,
            reached,
            q0: vec![],
            profiles: vec![],
            dt,
            knots_q: q,
            knots_qd: qd,
            knots_qdd: qdd,
        }
    }

    pub fn ndof(&self) -> usize {
        self.ndof
    }
    pub fn duration(&self) -> f64 {
        self.duration
    }
    pub fn limits(&self) -> &MotionLimits {
        &self.limits
    }

    pub fn sample(&self, t: f64) -> TrajState {
        match self.kind {
            TrajKind::MoveJ => {
                let mut q = self.q0.clone();
                let mut qd = vec![0.0; self.ndof];
                let mut qdd = vec![0.0; self.ndof];
                for i in 0..self.ndof {
                    let (dp, v, a) = self.profiles[i].sample(t);
                    q[i] = self.q0[i] + dp;
                    qd[i] = v;
                    qdd[i] = a;
                }
                TrajState { q, qd, qdd }
            }
            _ => self.sample_knots(t),
        }
    }

    fn sample_knots(&self, t: f64) -> TrajState {
        if self.knots_q.is_empty() {
            return TrajState {
                q: vec![],
                qd: vec![],
                qdd: vec![],
            };
        }
        let tt = t.clamp(0.0, self.duration);
        let fk = (tt / self.dt).floor() as usize;
        let n = self.knots_q.len();
        if fk + 1 >= n {
            // Surface the STORED terminal knot state: zeros for a fully-realized
            // move (rest boundary), the true one-sided qd/qdd at the cut for a
            // truncated best-effort prefix — never fabricate rest.
            let last = n - 1;
            return TrajState {
                q: self.knots_q[last].clone(),
                qd: self.knots_qd[last].clone(),
                qdd: self.knots_qdd[last].clone(),
            };
        }
        let frac = (tt - fk as f64 * self.dt) / self.dt;
        // cubic Hermite on q using stored qd at the two knots (C¹; C² adequate here
        // because s(t) is C² and IK is locally smooth — single Cartesian segment).
        let mut q = vec![0.0; self.ndof];
        let mut qd = vec![0.0; self.ndof];
        let mut qdd = vec![0.0; self.ndof];
        let h = self.dt;
        let (h00, h10, h01, h11) = hermite(frac);
        for i in 0..self.ndof {
            let p0 = self.knots_q[fk][i];
            let p1 = self.knots_q[fk + 1][i];
            let m0 = self.knots_qd[fk][i] * h;
            let m1 = self.knots_qd[fk + 1][i] * h;
            q[i] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
            qd[i] = (1.0 - frac) * self.knots_qd[fk][i] + frac * self.knots_qd[fk + 1][i];
            qdd[i] = (1.0 - frac) * self.knots_qdd[fk][i] + frac * self.knots_qdd[fk + 1][i];
        }
        TrajState { q, qd, qdd }
    }

    pub fn q_at(&self, t: f64) -> Vec<f64> {
        self.sample(t).q
    }

    pub fn sample_grid(&self, n: usize) -> Vec<(f64, TrajState)> {
        let n = n.max(2);
        (0..n)
            .map(|k| {
                let t = self.duration * k as f64 / (n - 1) as f64;
                (t, self.sample(t))
            })
            .collect()
    }

    /// World XYZ of `frame` along the trajectory at `n` samples (engine FK).
    pub fn tip_path(&self, model: &Model, frame: usize, n: usize) -> Vec<[f64; 3]> {
        self.sample_grid(n)
            .into_iter()
            .map(|(_, s)| {
                let t = fk_frame(model, &s.q, frame).translation();
                [t[0], t[1], t[2]]
            })
            .collect()
    }
}

#[inline]
fn hermite(t: f64) -> (f64, f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    )
}
