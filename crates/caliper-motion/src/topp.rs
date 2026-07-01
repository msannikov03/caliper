//! Time-optimal (acceleration-limited) parameterization of a joint-space
//! WAYPOINT path, with **corner stops** at every interior waypoint.
//!
//! Why corner stops. A piecewise-linear path `q(s)` has a *discontinuous*
//! tangent `q'(s)` at every interior waypoint, so `q''(s)` is an (unbounded)
//! Dirac there. Joint acceleration along the path is
//! `q̈ᵢ = q'ᵢ·s̈ + q''ᵢ·ṡ²`; the second term explodes at a corner unless the path
//! velocity `ṡ` is zero there. (A prior non-stopping attempt blew amax by ~1.8×
//! at corners for exactly this reason.) We therefore drive `ṡ → 0` at each
//! interior waypoint: the `q''·ṡ²` spike vanishes and `q̈ = q'ᵢ·s̈` stays bounded.
//!
//! Per segment the path tangent `q'(s) = Δq` is *constant*, so along one segment
//!   |q̇ᵢ| = |Δqᵢ|·ṡ  ≤ vmaxᵢ   ⟺  ṡ ≤ ṡ_max = minᵢ vmaxᵢ/|Δqᵢ|
//!   |q̈ᵢ| = |Δqᵢ|·s̈  ≤ amaxᵢ   ⟺  s̈ ≤ s̈_max = minᵢ amaxᵢ/|Δqᵢ|
//! and a rest-to-rest **bang-bang** (trapezoid/triangle) profile in `s` over
//! `[0,1]` is time-optimal subject to those two scalar bounds. Segments are
//! concatenated (rest between them) and resampled onto a uniform `dt` grid via
//! the crate-private `Trajectory::from_knots`.
//!
//! Scope. This is the honest, provably-bounded subset: corner *stops*. Blended
//! corners (non-zero `ṡ` through a smoothed corner — true TOPP-RA over a curved
//! path) are future work and would require first smoothing the geometric path so
//! `q''(s)` is finite.

use crate::MotionError;
use crate::limits::MotionLimits;
use crate::trajectory::{TrajKind, Trajectory};

const EPS_DQ: f64 = 1e-9;

/// Rest-to-rest bang-bang profile of a path scalar `s` over `[0, 1]` under a
/// velocity cap `vcap` (= ṡ_max) and acceleration cap `acap` (= s̈_max). Either
/// a symmetric trapezoid (cruise at `vcap`) or, for a short move, a triangle.
#[derive(Clone, Copy, Debug)]
struct BangBang {
    acap: f64,
    vpeak: f64,
    t1: f64, // end of accel phase
    t2: f64, // end of cruise phase
    t3: f64, // total
    s1: f64, // s at t1
    s2: f64, // s at t2
}

impl BangBang {
    /// Plan over total path length `len` (here always 1.0). `vcap`,`acap` > 0.
    fn plan(len: f64, vcap: f64, acap: f64) -> Self {
        // Peak velocity of a pure triangle (accel then decel, no cruise) that
        // covers `len`: v² = acap·len  (each ramp covers len/2 = v²/2acap).
        let v_tri = (acap * len).sqrt();
        if v_tri <= vcap {
            // triangle: t1 == t2
            let t1 = v_tri / acap;
            let s1 = 0.5 * acap * t1 * t1;
            BangBang {
                acap,
                vpeak: v_tri,
                t1,
                t2: t1,
                t3: 2.0 * t1,
                s1,
                s2: s1,
            }
        } else {
            // trapezoid: ramp to vcap, cruise, ramp down
            let t_acc = vcap / acap;
            let s_acc = 0.5 * acap * t_acc * t_acc; // = vcap²/2acap
            let s_cruise = len - 2.0 * s_acc;
            let t_cruise = s_cruise / vcap;
            BangBang {
                acap,
                vpeak: vcap,
                t1: t_acc,
                t2: t_acc + t_cruise,
                t3: 2.0 * t_acc + t_cruise,
                s1: s_acc,
                s2: s_acc + s_cruise,
            }
        }
    }

    fn total(&self) -> f64 {
        self.t3
    }

    /// (s, ṡ, s̈) at local time `t`. Clamped to `[0, t3]`; `s` clamped to `[0,1]`
    /// to absorb round-off so endpoints land machine-exact.
    fn sample(&self, t: f64) -> (f64, f64, f64) {
        let t = t.clamp(0.0, self.t3);
        let (s, sd, sdd) = if t <= self.t1 {
            (0.5 * self.acap * t * t, self.acap * t, self.acap)
        } else if t <= self.t2 {
            (self.s1 + self.vpeak * (t - self.t1), self.vpeak, 0.0)
        } else {
            let d = t - self.t2;
            (
                self.s2 + self.vpeak * d - 0.5 * self.acap * d * d,
                self.vpeak - self.acap * d,
                -self.acap,
            )
        };
        (s.clamp(0.0, 1.0), sd, sdd)
    }
}

struct Seg {
    q0: Vec<f64>,
    dq: Vec<f64>,
    prof: BangBang,
    t_start: f64,
    dur: f64,
}

/// Time-optimal (acceleration-limited, corner-stopping) retiming of `waypoints`
/// (≥1 rows, each length = ndof) onto a uniform `dt` grid.
///
/// Velocity and acceleration are bounded by `limits.vmax`/`limits.amax` on every
/// joint everywhere along the path (jerk is *not* limited — bang-bang has step
/// accelerations). Each interior waypoint is a rest (`q̇ = 0`) corner stop.
pub fn retime_time_optimal(
    waypoints: &[Vec<f64>],
    limits: &MotionLimits,
    dt: f64,
) -> Result<Trajectory, MotionError> {
    let n = limits.ndof();
    if waypoints.is_empty() || !(dt.is_finite() && dt > 0.0) {
        return Err(MotionError::DimMismatch);
    }
    for w in waypoints {
        if w.len() != n || !w.iter().all(|x| x.is_finite()) {
            return Err(MotionError::DimMismatch);
        }
    }
    for i in 0..n {
        if !(limits.vmax[i] > 0.0 && limits.amax[i] > 0.0) {
            return Err(MotionError::BadLimit(i));
        }
    }

    // One rest-to-rest bang-bang per non-degenerate segment.
    let mut segs: Vec<Seg> = Vec::new();
    let mut total = 0.0;
    for pair in waypoints.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let dq: Vec<f64> = (0..n).map(|i| b[i] - a[i]).collect();
        let (mut vcap, mut acap) = (f64::INFINITY, f64::INFINITY);
        for (i, dqi) in dq.iter().enumerate() {
            let d = dqi.abs();
            if d > EPS_DQ {
                vcap = vcap.min(limits.vmax[i] / d);
                acap = acap.min(limits.amax[i] / d);
            }
        }
        if !vcap.is_finite() {
            continue; // zero-Δq segment (duplicate waypoint) → skip
        }
        let prof = BangBang::plan(1.0, vcap, acap);
        let dur = prof.total();
        segs.push(Seg {
            q0: a.clone(),
            dq,
            prof,
            t_start: total,
            dur,
        });
        total += dur;
    }

    // Degenerate: all waypoints identical → single rest knot.
    if segs.is_empty() || total <= EPS_DQ {
        return Ok(Trajectory::from_knots(
            TrajKind::Plan,
            dt,
            vec![waypoints[0].clone()],
            vec![vec![0.0; n]],
            vec![vec![0.0; n]],
            limits.clone(),
            true,
            1.0,
        ));
    }

    // Resample on a uniform global grid (last sample lands exactly on `total`).
    let nsteps = (total / dt).ceil() as usize;
    let mut qs = Vec::with_capacity(nsteps + 1);
    let mut qds = Vec::with_capacity(nsteps + 1);
    let mut qdds = Vec::with_capacity(nsteps + 1);
    for k in 0..=nsteps {
        let t = (k as f64 * dt).min(total);
        let (q, v, a) = sample_at(&segs, t);
        qs.push(q);
        qds.push(v);
        qdds.push(a);
    }

    Ok(Trajectory::from_knots(
        TrajKind::Plan,
        dt,
        qs,
        qds,
        qdds,
        limits.clone(),
        true,
        1.0,
    ))
}

fn sample_at(segs: &[Seg], t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // First segment whose window contains t (last segment owns the final instant).
    let seg = segs
        .iter()
        .find(|s| t < s.t_start + s.dur + 1e-12)
        .unwrap_or_else(|| segs.last().unwrap());
    let local = (t - seg.t_start).clamp(0.0, seg.dur);
    let (s, sd, sdd) = seg.prof.sample(local);
    let q = seg
        .q0
        .iter()
        .zip(&seg.dq)
        .map(|(q0, dq)| q0 + dq * s)
        .collect();
    let qd = seg.dq.iter().map(|dq| dq * sd).collect();
    let qdd = seg.dq.iter().map(|dq| dq * sdd).collect();
    (q, qd, qdd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retime_waypoints;

    fn lim(n: usize) -> MotionLimits {
        MotionLimits {
            vmax: vec![1.0, 1.5],
            amax: vec![3.0, 4.0],
            jmax: vec![20.0; n],
        }
    }

    // Densely sample the trajectory and return the worst per-joint
    // velocity/acceleration ratio against the limits.
    fn worst_ratios(traj: &Trajectory, l: &MotionLimits, dt: f64) -> (f64, f64) {
        let dur = traj.duration();
        let n = (dur / dt).ceil() as usize;
        let (mut wv, mut wa) = (0.0_f64, 0.0_f64);
        for k in 0..=n {
            let t = (k as f64 * dt).min(dur);
            let s = traj.sample(t);
            for i in 0..s.qd.len() {
                wv = wv.max(s.qd[i].abs() / l.vmax[i]);
                wa = wa.max(s.qdd[i].abs() / l.amax[i]);
            }
        }
        (wv, wa)
    }

    // (a) DECISIVE: velocity AND acceleration stay within limits everywhere on a
    // dense grid — the multi-corner path the prior attempt failed (1.79×). Tight.
    #[test]
    fn bounds_velocity_and_accel_at_corners() {
        // A zig-zag whose corners have very different incoming/outgoing tangents,
        // so a non-stopping retime would spike acceleration at the corner.
        let wps = vec![
            vec![0.0, 0.0],
            vec![0.6, 0.05],
            vec![0.62, 0.9],
            vec![1.3, 0.1],
            vec![0.2, 0.4],
        ];
        let l = lim(2);
        let dt = 1e-3;
        let traj = retime_time_optimal(&wps, &l, dt).unwrap();
        let (wv, wa) = worst_ratios(&traj, &l, dt);
        assert!(wv <= 1.02, "vmax exceeded: ratio {wv}");
        assert!(wa <= 1.02, "amax exceeded: ratio {wa}"); // prior attempt: 1.79
    }

    // (b) Every distinct waypoint is reached, and with ~zero velocity (corner stop).
    #[test]
    fn hits_every_waypoint_at_rest() {
        let wps = vec![
            vec![0.0, 0.0],
            vec![0.5, -0.3],
            vec![0.5, -0.3], // duplicate → skipped, no spurious extra stop
            vec![1.0, 0.2],
            vec![0.3, 0.7],
        ];
        let l = lim(2);
        let dt = 1e-3;
        let traj = retime_time_optimal(&wps, &l, dt).unwrap();
        let dur = traj.duration();
        // densely sample; for each distinct waypoint find the closest pass.
        let grid: Vec<_> = {
            let n = (dur / dt).ceil() as usize;
            (0..=n)
                .map(|k| {
                    let t = (k as f64 * dt).min(dur);
                    (t, traj.sample(t))
                })
                .collect()
        };
        let distinct = [
            vec![0.0, 0.0],
            vec![0.5, -0.3],
            vec![1.0, 0.2],
            vec![0.3, 0.7],
        ];
        for wp in &distinct {
            let mut best = f64::INFINITY;
            let mut qd_there = f64::INFINITY;
            for (_, s) in &grid {
                let d = (0..2).map(|i| (s.q[i] - wp[i]).powi(2)).sum::<f64>().sqrt();
                if d < best {
                    best = d;
                    qd_there = s.qd.iter().map(|v| v.abs()).fold(0.0, f64::max);
                }
            }
            assert!(best < 2e-3, "waypoint {wp:?} not reached (min dist {best})");
            assert!(
                qd_there < 5e-2,
                "waypoint {wp:?} not a rest (|qd| {qd_there})"
            );
        }
    }

    // (c) Rest at both ends.
    #[test]
    fn rest_at_both_ends() {
        let wps = vec![vec![0.0, 0.0], vec![0.7, -0.4], vec![0.2, 0.5]];
        let l = lim(2);
        let traj = retime_time_optimal(&wps, &l, 1e-3).unwrap();
        let dur = traj.duration();
        for s in [traj.sample(0.0), traj.sample(dur)] {
            for v in &s.qd {
                assert!(v.abs() < 1e-9, "not at rest: {v}");
            }
        }
    }

    // (d) For a SINGLE-segment move (no corners), acceleration-limited bang-bang is
    // no slower than the conservative jerk-limited retime_waypoints. (Multi-segment
    // is intentionally NOT asserted faster — corner stops can cost time; honest.)
    #[test]
    fn single_segment_not_slower_than_jerk_limited() {
        let wps = vec![vec![0.0, 0.0], vec![1.0, -0.6]];
        let l = lim(2);
        let dt = 1e-4;
        let topp = retime_time_optimal(&wps, &l, dt).unwrap();
        let jerk = retime_waypoints(&wps, &l, dt).unwrap();
        assert!(
            topp.duration() <= jerk.duration() + 1e-9,
            "bang-bang {} should beat jerk-limited {}",
            topp.duration(),
            jerk.duration()
        );
        // and still within limits
        let (wv, wa) = worst_ratios(&topp, &l, dt);
        assert!(wv <= 1.02 && wa <= 1.02, "limits: v {wv} a {wa}");
    }

    // Profile algebra: bang-bang lands at s=1, ṡ=0 for both triangle and trapezoid.
    #[test]
    fn bangbang_reaches_unit_at_rest() {
        // trapezoid (cruise): low accel, high enough vcap reached
        let tp = BangBang::plan(1.0, 0.5, 2.0);
        let (s, sd, _) = tp.sample(tp.total());
        assert!((s - 1.0).abs() < 1e-12 && sd.abs() < 1e-12);
        assert!(tp.t2 > tp.t1, "expected a cruise phase");
        // triangle (no cruise): high vcap never reached
        let tr = BangBang::plan(1.0, 100.0, 2.0);
        let (s, sd, _) = tr.sample(tr.total());
        assert!((s - 1.0).abs() < 1e-12 && sd.abs() < 1e-12);
        assert!((tr.t2 - tr.t1).abs() < 1e-15, "expected no cruise phase");
        // peak velocity caps honored
        assert!(tp.vpeak <= 0.5 + 1e-12 && tr.vpeak <= 100.0);
    }

    #[test]
    fn rejects_bad_input() {
        let l = lim(2);
        assert!(retime_time_optimal(&[], &l, 1e-3).is_err());
        assert!(retime_time_optimal(&[vec![0.0]], &l, 1e-3).is_err()); // wrong ndof
        assert!(retime_time_optimal(&[vec![0.0, 0.0], vec![1.0, 1.0]], &l, 0.0).is_err()); // dt
        assert!(retime_time_optimal(&[vec![0.0, f64::NAN], vec![1.0, 1.0]], &l, 1e-3).is_err());
    }

    #[test]
    fn all_identical_is_rest() {
        let wps = vec![vec![0.3, 0.3], vec![0.3, 0.3]];
        let traj = retime_time_optimal(&wps, &lim(2), 1e-3).unwrap();
        assert_eq!(traj.q_at(0.0), vec![0.3, 0.3]);
    }
}
