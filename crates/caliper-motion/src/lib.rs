//! Jerk-limited offline trajectory generation (Phase 3, simulation-only).
//!
//! v1 = rest-to-rest, time-synchronized, jerk-limited 7-segment S-curves
//! (MOVE_J), plus Cartesian MOVE_L / MOVE_C built on a 1-DOF path-scalar S-curve
//! with per-sample IK. Everything is closed-form per segment: `sample(t)` is O(1)
//! and machine-exact at endpoints.
mod cartesian;
mod limits;
mod movej;
mod poses;
mod scurve;
mod trajectory;
mod waypoints;

pub use cartesian::{CartesianMoveOpts, MoveLMode, OnFailure, move_c, move_l, move_l_pose};
pub use limits::{JointOverride, MotionLimits, MotionLimitsConfig};
pub use movej::move_j;
pub use poses::{NamedPose, PoseLibrary};
pub use scurve::{ScurveProfile, plan_scurve, plan_scurve_to_duration};
pub use trajectory::{TrajKind, TrajState, Trajectory};
pub use waypoints::retime_waypoints;

#[derive(thiserror::Error, Debug, Clone)]
pub enum MotionError {
    #[error("dimension mismatch (q/limits vs ndof)")]
    DimMismatch,
    #[error("configuration outside joint position limits at dof {0}")]
    OutOfLimits(usize),
    #[error("non-positive motion limit (v/a/jerk must be > 0) at dof {0}")]
    BadLimit(usize),
    #[error("time-sync bisection failed to converge")]
    SyncFailed,
    #[error("IK unreachable at path fraction s={s:.4} (residual {residual:.2e})")]
    Unreachable { s: f64, residual: f64 },
    #[error("joint-space discontinuity (branch flip) at s={s:.4}, jump {jump:.3}")]
    Discontinuity { s: f64, jump: f64 },
    #[error("three points are collinear; no arc defined")]
    CollinearArc,
    #[error("zero-length Cartesian segment with no rotation")]
    ZeroLengthSegment,
}

#[cfg(test)]
mod tests;
