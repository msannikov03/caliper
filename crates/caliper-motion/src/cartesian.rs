//! Cartesian MOVE_L / MOVE_C: lerp(position)+slerp(orientation) along a 1-DOF
//! jerk-limited path scalar s(t), warm-seeded per-sample IK, FD derivatives.
use crate::MotionError;
use crate::limits::MotionLimits;
use crate::scurve::{ScurveProfile, plan_scurve, plan_scurve_to_duration};
use crate::trajectory::{TrajKind, Trajectory};
use caliper_ik::{IkOpts, ik};
use caliper_kinematics::fk_frame;
use caliper_model::Model;
use caliper_spatial::Se3;
use nalgebra::{UnitQuaternion, Vector3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveLMode {
    Decoupled,
    Screw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnFailure {
    /// Reject the whole move if any sample is unreachable.
    Abort,
    /// Best-effort: return the realized prefix (Trajectory.completed = false).
    Truncate,
}

#[derive(Clone, Debug)]
pub struct CartesianMoveOpts {
    pub limits: MotionLimits, // joint vel/acc/jerk for ρ scaling
    pub v_lin: f64,
    pub a_lin: f64,
    pub j_lin: f64, // Cartesian linear caps (m/s, m/s², m/s³)
    pub v_ang: f64,
    pub a_ang: f64,
    pub j_ang: f64, // angular caps (rad/...)
    pub dt: f64,    // control grid (s)
    pub ik: IkOpts, // restarts forced to 1 internally
    pub continuity_thresh: f64,
    pub on_failure: OnFailure,
    pub mode: MoveLMode,
}

impl CartesianMoveOpts {
    /// Reject non-finite / non-positive caps + dt before they reach the time-grid
    /// (`n = (total/dt).ceil() as usize` would overflow/hang on dt<=0), mirroring the
    /// joint-space path's `MotionLimits` / `retime_waypoints` guards.
    pub fn validate(&self) -> Result<(), MotionError> {
        let pos = [
            ("v_lin", self.v_lin),
            ("a_lin", self.a_lin),
            ("j_lin", self.j_lin),
            ("v_ang", self.v_ang),
            ("a_ang", self.a_ang),
            ("j_ang", self.j_ang),
            ("dt", self.dt),
        ];
        for (name, v) in pos {
            if !(v.is_finite() && v > 0.0) {
                return Err(MotionError::BadParam(name));
            }
        }
        Ok(())
    }

    pub fn defaults(limits: MotionLimits) -> Self {
        Self {
            limits,
            v_lin: 0.25,
            a_lin: 1.0,
            j_lin: 5.0,
            v_ang: 1.0,
            a_ang: 5.0,
            j_ang: 25.0,
            dt: 0.01,
            // Tracking IK: a sub-micron tolerance (1e-7) is plenty for a straight
            // path and converges reliably warm-seeded; the engine default 1e-9 is
            // so strict it rejects an essentially-perfect solution (residual ~2e-9).
            ik: IkOpts {
                restarts: 1,
                tol_pos: 1e-6,
                tol_rot: 1e-6,
                max_iters: 200,
                ..Default::default()
            },
            continuity_thresh: 0.5,
            // best-effort prefix by default: the arm plays up to the wall and the
            // Trajectory flags where it stopped (completed=false, reached=s).
            on_failure: OnFailure::Truncate,
            mode: MoveLMode::Decoupled,
        }
    }
}

/// Pure geometry: pose along MOVE_L at s∈[0,1].
pub fn move_l_pose(t0: &Se3, t1: &Se3, s: f64, mode: MoveLMode) -> Se3 {
    match mode {
        MoveLMode::Decoupled => {
            let p = t0.translation_vec() + s * (t1.translation_vec() - t0.translation_vec());
            let r = slerp_short(&t0.0.rotation, &t1.0.rotation, s);
            Se3::from_parts(p, r)
        }
        MoveLMode::Screw => {
            let rel = Se3(t0.0.inverse() * t1.0).log();
            let step = caliper_spatial::Twist(rel.0 * s);
            Se3(t0.0 * Se3::exp(&step).0)
        }
    }
}

/// Antipodal-safe slerp (nalgebra slerp panics at θ≈π; fall back to nlerp).
fn slerp_short(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>, s: f64) -> UnitQuaternion<f64> {
    a.try_slerp(b, s, 1e-6).unwrap_or_else(|| {
        let bb = if a.coords.dot(&b.coords) < 0.0 {
            -b.coords
        } else {
            b.coords
        };
        let c = a.coords * (1.0 - s) + bb * s;
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::from(c))
    })
}

/// True iff a pose's translation and rotation quaternion are all finite.
fn goal_is_finite(t: &Se3) -> bool {
    t.translation_vec().iter().all(|x| x.is_finite())
        && t.0.rotation.coords.iter().all(|x| x.is_finite())
}

pub fn move_l(
    model: &Model,
    frame: usize,
    q_start: &[f64],
    goal: &Se3,
    opts: &CartesianMoveOpts,
) -> Result<Trajectory, MotionError> {
    if q_start.len() != model.ndof {
        return Err(MotionError::DimMismatch);
    }
    opts.validate()?;
    if !goal_is_finite(goal) {
        return Err(MotionError::BadParam("goal pose is non-finite"));
    }
    let t0 = fk_frame(model, q_start, frame); // actual start pose (no re-IK)
    let l = (goal.translation_vec() - t0.translation_vec()).norm();
    let phi = (t0.0.inverse() * goal.0).rotation.angle();
    if l < 1e-9 && phi < 1e-9 {
        return Err(MotionError::ZeroLengthSegment);
    }
    let pose_of = |s: f64| move_l_pose(&t0, goal, s, opts.mode);
    sample_cartesian(
        model,
        frame,
        q_start,
        &pose_of,
        l,
        phi,
        TrajKind::MoveL,
        opts,
    )
}

pub fn move_c(
    model: &Model,
    frame: usize,
    q_start: &[f64],
    via: &Vector3<f64>,
    end: &Se3,
    opts: &CartesianMoveOpts,
) -> Result<Trajectory, MotionError> {
    if q_start.len() != model.ndof {
        return Err(MotionError::DimMismatch);
    }
    opts.validate()?;
    if !goal_is_finite(end) || !via.iter().all(|x| x.is_finite()) {
        return Err(MotionError::BadParam("via/end pose is non-finite"));
    }
    let t0 = fk_frame(model, q_start, frame);
    let p0 = t0.translation_vec();
    let p1 = end.translation_vec();
    let arc = fit_arc(&p0, via, &p1)?;
    let r0 = t0.0.rotation;
    let r1 = end.0.rotation;
    let pose_of = |s: f64| {
        let ang = s * arc.phi;
        let p = arc.c + arc.r * (ang.cos() * arc.u + ang.sin() * arc.v);
        Se3::from_parts(p, slerp_short(&r0, &r1, s))
    };
    let l = arc.r * arc.phi; // translational arc length
    let phi_rot = (t0.0.inverse() * end.0).rotation.angle();
    sample_cartesian(
        model,
        frame,
        q_start,
        &pose_of,
        l,
        phi_rot,
        TrajKind::MoveC,
        opts,
    )
}

struct ArcGeom {
    c: Vector3<f64>,
    r: f64,
    u: Vector3<f64>,
    v: Vector3<f64>,
    phi: f64,
}

fn fit_arc(
    p0: &Vector3<f64>,
    pv: &Vector3<f64>,
    p1: &Vector3<f64>,
) -> Result<ArcGeom, MotionError> {
    let a = p0 - pv;
    let b = p1 - pv;
    let axb = a.cross(&b);
    let denom = 2.0 * axb.norm_squared();
    if denom < 1e-18 || axb.norm() < 1e-9 * a.norm() * b.norm() {
        return Err(MotionError::CollinearArc);
    }
    let c = *pv + (a.norm_squared() * b.cross(&axb) + b.norm_squared() * axb.cross(&a)) / denom;
    let r = (p0 - c).norm();
    let n = axb.normalize();
    let u = (p0 - c) / r;
    let v = n.cross(&u);
    let ang = |p: &Vector3<f64>| {
        let d = p - c;
        d.dot(&v).atan2(d.dot(&u))
    };
    let two_pi = std::f64::consts::TAU;
    let wrap = |x: f64| {
        let mut y = x % two_pi;
        if y < 0.0 {
            y += two_pi;
        }
        y
    };
    let phi_v = wrap(ang(pv));
    let mut phi_1 = wrap(ang(p1));
    if phi_1 < phi_v {
        phi_1 += two_pi;
    }
    Ok(ArcGeom {
        c,
        r,
        u,
        v,
        phi: phi_1,
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_cartesian(
    model: &Model,
    frame: usize,
    q_start: &[f64],
    pose_of: &dyn Fn(f64) -> Se3,
    len_lin: f64,
    len_ang: f64,
    kind: TrajKind,
    opts: &CartesianMoveOpts,
) -> Result<Trajectory, MotionError> {
    // s(t) caps: min over modalities (skip a modality whose length≈0).
    let cap = |lin: f64, ang: f64| {
        let mut m = f64::INFINITY;
        if len_lin > 1e-9 {
            m = m.min(lin / len_lin);
        }
        if len_ang > 1e-9 {
            m = m.min(ang / len_ang);
        }
        m
    };
    let vs = cap(opts.v_lin, opts.v_ang);
    let as_ = cap(opts.a_lin, opts.a_ang);
    let js = cap(opts.j_lin, opts.j_ang);
    let mut prof = plan_scurve(1.0, vs, as_, js);

    let ikopts = IkOpts {
        restarts: 1,
        ..opts.ik.clone()
    };

    // Solve the whole grid for a given timing profile. Returns (rows, last reached
    // s, stop reason). A stop reason means the path was truncated at that sample.
    let solve = |prof: &ScurveProfile| -> (Vec<Vec<f64>>, f64, Option<MotionError>) {
        let total = prof.total().max(opts.dt);
        let n = (total / opts.dt).ceil() as usize + 1;
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut seed = q_start.to_vec();
        let mut last_s = 0.0;
        for k in 0..n {
            let t = (k as f64 * opts.dt).min(total);
            let (s, _, _) = prof.sample(t); // s ∈ [0,1]
            let pose = pose_of(s);
            let res = ik(model, frame, &pose, &seed, &ikopts);
            if !res.success || !res.q.iter().all(|x| x.is_finite()) {
                return (
                    rows,
                    last_s,
                    Some(MotionError::Unreachable {
                        s,
                        residual: res.residual,
                    }),
                );
            }
            let jump: f64 = res
                .q
                .iter()
                .zip(&seed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if k > 0 && jump > opts.continuity_thresh {
                return (rows, last_s, Some(MotionError::Discontinuity { s, jump }));
            }
            seed = res.q.clone();
            rows.push(res.q);
            last_s = s;
        }
        (rows, 1.0, None)
    };

    let finish = |rows: Vec<Vec<f64>>,
                  reached: f64,
                  stop: Option<MotionError>|
     -> Result<Trajectory, MotionError> {
        if let Some(e) = &stop {
            // Abort, or too short to interpolate → hard error.
            if opts.on_failure == OnFailure::Abort || rows.len() < 2 {
                return Err(e.clone());
            }
        }
        let (qd, qdd) = fd_derivs(&rows, opts.dt);
        Ok(Trajectory::from_knots(
            kind,
            opts.dt,
            rows,
            qd,
            qdd,
            opts.limits.clone(),
            stop.is_none(),
            reached,
        ))
    };

    // first pass
    let (rows, reached, stop) = solve(&prof);

    // joint-limit ρ scaling (rare; near-singular) — only on a fully-solved path.
    if stop.is_none() {
        let (qd, _) = fd_derivs(&rows, opts.dt);
        let mut worst = 0.0f64;
        for row in &qd {
            for (i, &v) in row.iter().enumerate() {
                worst = worst.max(v.abs() / opts.limits.vmax[i]);
            }
        }
        if worst > 1.0 {
            let new_total = prof.total().max(opts.dt) * worst;
            prof = plan_scurve_to_duration(1.0, new_total, vs, as_, js);
            let (rows2, reached2, stop2) = solve(&prof);
            return finish(rows2, reached2, stop2);
        }
    }
    finish(rows, reached, stop)
}

fn fd_derivs(rows: &[Vec<f64>], dt: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = rows.len();
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut qd = vec![vec![0.0; d]; n];
    let mut qdd = vec![vec![0.0; d]; n];
    for k in 0..n {
        for i in 0..d {
            qd[k][i] = if k == 0 || k + 1 == n {
                0.0 // rest boundary
            } else {
                (rows[k + 1][i] - rows[k - 1][i]) / (2.0 * dt)
            };
        }
    }
    for k in 0..n {
        for i in 0..d {
            qdd[k][i] = if k == 0 || k + 1 == n {
                0.0
            } else {
                (qd[k + 1][i] - qd[k - 1][i]) / (2.0 * dt)
            };
        }
    }
    (qd, qdd)
}
