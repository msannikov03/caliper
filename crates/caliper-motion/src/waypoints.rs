//! Time-parameterize a collision-free joint-space WAYPOINT path (e.g. from the
//! planner) into a [`Trajectory`] that respects per-joint velocity/accel/jerk
//! limits and plays through the standard transport.
//!
//! Each consecutive pair is driven by a rest-to-rest **path-scalar** S-curve
//! `s: 0→1` whose limits are the tightest each joint allows
//! (`ṡ_max = minᵢ vmaxᵢ/|Δqᵢ|`, likewise accel/jerk), so `qd = Δq·ṡ`, `qdd = Δq·s̈`
//! stay within limits on every joint. v1 rests at interior waypoints (stop-and-go
//! corners; blended/TOPP retiming is deferred). Derivatives are analytic, so this
//! reuses the crate-private `Trajectory::from_knots` without touching its FD path.

use crate::MotionError;
use crate::limits::MotionLimits;
use crate::scurve::{ScurveProfile, plan_scurve};
use crate::trajectory::{TrajKind, Trajectory};

struct Seg {
    q0: Vec<f64>,
    dq: Vec<f64>,
    prof: ScurveProfile,
    t_start: f64,
    dur: f64,
}

const EPS_DQ: f64 = 1e-9;

/// Retime `waypoints` (≥1 rows, each length = ndof) onto a uniform `dt` grid.
pub fn retime_waypoints(
    waypoints: &[Vec<f64>],
    limits: &MotionLimits,
    dt: f64,
) -> Result<Trajectory, MotionError> {
    let n = limits.ndof();
    if waypoints.is_empty() || !(dt.is_finite() && dt > 0.0) {
        return Err(MotionError::DimMismatch);
    }
    for w in waypoints {
        if w.len() != n {
            return Err(MotionError::DimMismatch);
        }
        if !w.iter().all(|x| x.is_finite()) {
            return Err(MotionError::DimMismatch);
        }
    }
    for i in 0..n {
        if !(limits.vmax[i] > 0.0 && limits.amax[i] > 0.0 && limits.jmax[i] > 0.0) {
            return Err(MotionError::BadLimit(i));
        }
    }

    // build a path-scalar S-curve per (non-degenerate) segment
    let mut segs: Vec<Seg> = Vec::new();
    let mut total = 0.0;
    for pair in waypoints.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let dq: Vec<f64> = (0..n).map(|i| b[i] - a[i]).collect();
        let (mut smax, mut samax, mut sjmax) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for (i, dqi) in dq.iter().enumerate() {
            let d = dqi.abs();
            if d > EPS_DQ {
                smax = smax.min(limits.vmax[i] / d);
                samax = samax.min(limits.amax[i] / d);
                sjmax = sjmax.min(limits.jmax[i] / d);
            }
        }
        if !smax.is_finite() {
            continue; // zero-Δq segment (duplicate waypoint) → skip
        }
        let prof = plan_scurve(1.0, smax, samax, sjmax); // path scalar 0→1
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

    // degenerate: all waypoints identical → a single rest knot
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

    // sample on a uniform global grid (last sample lands exactly on `total`)
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
    // first segment whose window contains t (last segment owns the final instant)
    let seg = segs
        .iter()
        .find(|s| t < s.t_start + s.dur + 1e-12)
        .unwrap_or_else(|| segs.last().unwrap());
    let local = (t - seg.t_start).clamp(0.0, seg.dur);
    let (s, sd, sdd) = seg.prof.sample(local); // path scalar + derivatives
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

    fn lim(n: usize) -> MotionLimits {
        MotionLimits {
            vmax: vec![1.0; n],
            amax: vec![3.0; n],
            jmax: vec![20.0; n],
        }
    }

    #[test]
    fn respects_limits_and_endpoints() {
        let wps = vec![
            vec![0.0, 0.0],
            vec![0.5, -0.3],
            vec![0.5, -0.3], // duplicate (degenerate segment, skipped)
            vec![1.0, 0.2],
        ];
        let dt = 1e-3;
        let traj = retime_waypoints(&wps, &lim(2), dt).unwrap();
        let dur = traj.duration();
        assert!(dur > 0.0);
        // endpoints exact
        let s0 = traj.sample(0.0);
        let s1 = traj.sample(dur);
        for i in 0..2 {
            assert!((s0.q[i] - wps[0][i]).abs() < 1e-6, "start joint {i}");
            assert!(
                (s1.q[i] - wps.last().unwrap()[i]).abs() < 1e-3,
                "end joint {i}"
            );
        }
        // limits respected on a fine grid (small tolerance for interpolation)
        let l = lim(2);
        let n = (dur / 1e-3) as usize;
        for k in 0..=n {
            let t = k as f64 * 1e-3;
            let s = traj.sample(t);
            for i in 0..2 {
                assert!(
                    s.qd[i].abs() <= l.vmax[i] * 1.02,
                    "vmax j{i} @ {t}: {}",
                    s.qd[i]
                );
                assert!(
                    s.qdd[i].abs() <= l.amax[i] * 1.05,
                    "amax j{i} @ {t}: {}",
                    s.qdd[i]
                );
            }
        }
    }

    #[test]
    fn all_identical_is_rest() {
        let wps = vec![vec![0.3, 0.3], vec![0.3, 0.3]];
        let traj = retime_waypoints(&wps, &lim(2), 1e-3).unwrap();
        assert_eq!(traj.q_at(0.0), vec![0.3, 0.3]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(retime_waypoints(&[], &lim(2), 1e-3).is_err());
        assert!(retime_waypoints(&[vec![0.0]], &lim(2), 1e-3).is_err()); // wrong ndof
        assert!(retime_waypoints(&[vec![0.0, 0.0], vec![1.0, 1.0]], &lim(2), 0.0).is_err()); // dt
    }
}
