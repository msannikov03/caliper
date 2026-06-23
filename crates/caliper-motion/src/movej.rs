use crate::MotionError;
use crate::limits::MotionLimits;
use crate::scurve::{plan_scurve, plan_scurve_to_duration};
use crate::trajectory::Trajectory;
use caliper_model::Model;

const EPS_T: f64 = 1e-9;

/// Rest-to-rest, jerk-limited, time-synchronized joint-space move.
pub fn move_j(
    model: &Model,
    q0: &[f64],
    q1: &[f64],
    limits: &MotionLimits,
) -> Result<Trajectory, MotionError> {
    let n = model.ndof;
    if q0.len() != n || q1.len() != n || limits.ndof() != n {
        return Err(MotionError::DimMismatch);
    }
    // endpoints must respect position limits (continuous joints unbounded)
    for (i, lim) in model.limits.iter().enumerate() {
        if let Some((lo, hi)) = lim
            && (q0[i] < *lo - 1e-9
                || q0[i] > *hi + 1e-9
                || q1[i] < *lo - 1e-9
                || q1[i] > *hi + 1e-9)
        {
            return Err(MotionError::OutOfLimits(i));
        }
    }
    let mut profiles = Vec::with_capacity(n);
    let mut tmax = 0.0f64;
    for i in 0..n {
        let p = plan_scurve(q1[i] - q0[i], limits.vmax[i], limits.amax[i], limits.jmax[i]);
        tmax = tmax.max(p.total());
        profiles.push(p);
    }
    if tmax < EPS_T {
        return Ok(Trajectory::from_profiles(
            q0.to_vec(),
            profiles,
            0.0,
            limits.clone(),
        ));
    }
    for i in 0..n {
        if profiles[i].total() < tmax - EPS_T {
            profiles[i] = plan_scurve_to_duration(
                q1[i] - q0[i],
                tmax,
                limits.vmax[i],
                limits.amax[i],
                limits.jmax[i],
            );
        }
    }
    Ok(Trajectory::from_profiles(
        q0.to_vec(),
        profiles,
        tmax,
        limits.clone(),
    ))
}
