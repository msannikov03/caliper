//! Operational-space (resolved-rate) and null-space redundancy resolution.
//!
//! For a redundant arm (`ndof > 6`) the task Jacobian `J` (6 × n) has a
//! non-trivial null space `ker(J)` of self-motion modes: joint velocities that
//! move the arm WITHOUT moving the end-effector. This module resolves a desired
//! end-effector spatial velocity into joint velocities and, optionally, injects
//! a secondary objective through the null-space projector so it never disturbs
//! the tracked task.
//!
//! Two generalized inverses appear, deliberately:
//!  * the **damped** (Levenberg–Marquardt / DLS) inverse
//!    `J⁺_λ = Jᵀ(JJᵀ + λ²I)⁻¹` drives the PRIMARY task — its manipulability-gated
//!    `λ` (same idea as the DLS IK loop) keeps `qd` bounded through singularities;
//!  * the undamped **Moore–Penrose** inverse `J⁺` builds the null-space projector
//!    `N = I − J⁺J`. Only a reflexive `{1}`-inverse gives `J·N = 0` exactly, so
//!    the projector MUST use the MP inverse — a damped inverse would leak the
//!    secondary motion into the tip. `qd = J⁺_λ·v + N·z`.
//!
//! All spatial velocities are expressed with the world-aligned (LOCAL_WORLD_ALIGNED)
//! tip Jacobian, rows `[v; ω]`, matching [`JacFrame::World`]. Pure `nalgebra`,
//! deterministic, no extra deps.
use caliper_kinematics::{JacFrame, jacobian};
use caliper_model::Model;
use nalgebra::{Cholesky, DMatrix, DVector};

/// Damping configuration for redundancy resolution. Mirrors the IK DLS knobs:
/// damping stays at a tiny Tikhonov `lambda_floor_sq` while manipulability
/// `w = ∏σᵢ` is healthy, and ramps up to `lambda0_sq` as `w → 0`.
#[derive(Clone, Debug)]
pub struct RedundancyOpts {
    /// Maximum squared damping `λ²`, reached as manipulability collapses.
    pub lambda0_sq: f64,
    /// Always-on Tikhonov floor so `JJᵀ + λ²I` stays SPD and `qd` stays finite.
    pub lambda_floor_sq: f64,
    /// Manipulability threshold below which damping ramps in.
    pub w_thresh: f64,
}

impl Default for RedundancyOpts {
    fn default() -> Self {
        // Same defaults as `IkOpts` so behaviour matches the DLS IK loop.
        Self {
            lambda0_sq: 1e-3,
            lambda_floor_sq: 1e-10,
            w_thresh: 1e-3,
        }
    }
}

/// Manipulability-gated squared damping `λ²` for Jacobian `j` (6 × n).
/// Identical shape to the IK loop: floor while `w ≥ w_thresh`, quadratic ramp to
/// `lambda0_sq` as `w → 0`.
fn damping_sq(j: &DMatrix<f64>, opts: &RedundancyOpts) -> f64 {
    let sv = j.clone().svd(false, false).singular_values;
    let w: f64 = sv.iter().product();
    let base = if w >= opts.w_thresh {
        0.0
    } else {
        opts.lambda0_sq * (1.0 - w / opts.w_thresh).powi(2)
    };
    base.max(opts.lambda_floor_sq)
}

/// Damped least-squares (task-space) pseudo-inverse `J⁺_λ = Jᵀ(JJᵀ + λ²I)⁻¹`,
/// n × 6. `JJᵀ` is 6 × 6 and SPD once the `λ²` floor is added, so a Cholesky
/// solve suffices; a defensive SVD fallback covers any numerical failure.
fn damped_pinv(j: &DMatrix<f64>, lambda_sq: f64) -> DMatrix<f64> {
    let m = j.nrows(); // 6
    let mut h = j * j.transpose(); // m × m, SPD after the floor
    for i in 0..m {
        h[(i, i)] += lambda_sq;
    }
    match Cholesky::new(h) {
        Some(c) => j.transpose() * c.inverse(),
        None => mp_pinv(j),
    }
}

/// Undamped Moore–Penrose pseudo-inverse via SVD (n × 6). Used to build the exact
/// null-space projector. Falls back to a zero inverse only if nalgebra fails to
/// factor (it should not, given finite inputs).
fn mp_pinv(j: &DMatrix<f64>) -> DMatrix<f64> {
    j.clone()
        .svd(true, true)
        .pseudo_inverse(1e-12)
        .unwrap_or_else(|_| DMatrix::zeros(j.ncols(), j.nrows()))
}

/// Resolved-rate (operational-space) control: joint velocities that realize the
/// desired end-effector spatial velocity `v_desired` (a 6-vector `[v; ω]`, in the
/// world-aligned tip frame) at configuration `q`, via the manipulability-gated
/// damped pseudo-inverse. Returns `qd` of length `model.ndof`.
///
/// Invalid input (wrong lengths or non-finite values) yields a zero vector.
pub fn resolved_rate(
    model: &Model,
    frame: usize,
    q: &[f64],
    v_desired: &[f64],
    opts: &RedundancyOpts,
) -> Vec<f64> {
    let n = model.ndof;
    if !valid(q, n) || v_desired.len() != 6 || !v_desired.iter().all(|x| x.is_finite()) {
        return vec![0.0; n];
    }
    let (_, j) = jacobian(model, q, frame, JacFrame::World);
    let lambda_sq = damping_sq(&j, opts);
    let qd = damped_pinv(&j, lambda_sq) * DVector::from_column_slice(v_desired);
    qd.iter().copied().collect()
}

/// Resolved-rate with a null-space secondary objective:
/// `qd = J⁺_λ·v_desired + (I − J⁺J)·z_secondary`.
///
/// `z_secondary` (length `ndof`) is a desired joint velocity for a secondary goal
/// (e.g. [`joint_limit_avoidance_gradient`] or a manipulability-maximizing
/// direction). Its component inside `ker(J)` is applied; the rest is projected
/// out, so the secondary motion leaves the end-effector velocity unchanged. The
/// projector uses the exact Moore–Penrose inverse, guaranteeing `J·(I−J⁺J) = 0`.
///
/// Invalid input yields a zero vector.
pub fn nullspace_step(
    model: &Model,
    frame: usize,
    q: &[f64],
    v_desired: &[f64],
    z_secondary: &[f64],
    opts: &RedundancyOpts,
) -> Vec<f64> {
    let n = model.ndof;
    if !valid(q, n)
        || v_desired.len() != 6
        || !v_desired.iter().all(|x| x.is_finite())
        || z_secondary.len() != n
        || !z_secondary.iter().all(|x| x.is_finite())
    {
        return vec![0.0; n];
    }
    let (_, j) = jacobian(model, q, frame, JacFrame::World);
    let lambda_sq = damping_sq(&j, opts);
    let primary = damped_pinv(&j, lambda_sq) * DVector::from_column_slice(v_desired);

    // Null-space projection: N·z = z − J⁺(J·z), formed without an n×n projector.
    let z = DVector::from_column_slice(z_secondary);
    let projected = &z - mp_pinv(&j) * (&j * &z);

    (primary + projected).iter().copied().collect()
}

/// Steepest-descent direction of a joint-limit-avoidance potential — a ready-made
/// secondary objective for [`nullspace_step`]. For each bounded joint it points
/// away from the nearer limit toward the mid-range; unbounded joints contribute 0.
///
/// The potential is `H(q) = ½ Σ ((qᵢ − midᵢ)/rangeᵢ)²`; this returns `−∇H`, i.e.
/// `−(qᵢ − midᵢ)/rangeᵢ²` per bounded joint. Scale it by a gain at the call site.
/// Length `model.ndof`.
pub fn joint_limit_avoidance_gradient(model: &Model, q: &[f64]) -> Vec<f64> {
    let n = model.ndof;
    if !valid(q, n) {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| match model.limits.get(i).and_then(|l| *l) {
            Some((lo, hi)) if hi > lo => {
                let mid = 0.5 * (lo + hi);
                let range = hi - lo;
                -(q[i] - mid) / (range * range)
            }
            _ => 0.0,
        })
        .collect()
}

#[inline]
fn valid(q: &[f64], n: usize) -> bool {
    q.len() == n && q.iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;
    use caliper_model::Model;
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }

    /// Generic, well-conditioned (full task-rank) config of the 7-DOF arm.
    const Q7: [f64; 7] = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1, 0.35];

    /// Numerically apply the SAME world Jacobian the resolver used: achieved tip
    /// twist = J · qd. This is the decisive cross-check against the validated
    /// `caliper_kinematics::jacobian`.
    fn achieved_twist(m: &Model, q: &[f64], frame: usize, qd: &[f64]) -> DVector<f64> {
        let (_, j) = jacobian(m, q, frame, JacFrame::World);
        &j * DVector::from_column_slice(qd)
    }

    /// (a) resolved_rate realizes the commanded tip velocity: J·qd ≈ v_desired for
    /// arbitrary reachable v on the redundant arm (rank(J)=6 ⇒ range(J)=ℝ⁶).
    #[test]
    fn resolved_rate_achieves_commanded_twist() {
        let m = load("redundant7.urdf");
        let f = m.tip_frame();
        let opts = RedundancyOpts::default();
        let mut rng = SplitMix64(0xABCDEF);
        for _ in 0..40 {
            let v: Vec<f64> = (0..6).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
            let qd = resolved_rate(&m, f, &Q7, &v, &opts);
            assert_eq!(qd.len(), 7);
            let got = achieved_twist(&m, &Q7, f, &qd);
            let want = DVector::from_column_slice(&v);
            // Small residual is the damped-inverse bias λ²/(σ²+λ²); a WRONG inverse
            // would miss by O(‖v‖)~1, so this cleanly separates correct from broken.
            assert!(
                (&got - &want).norm() < 1e-5,
                "achieved twist error {:.2e}",
                (&got - &want).norm()
            );
        }
    }

    /// (b) a PURE null-space command (v=0, z≠0) leaves the tip velocity ≈0 yet moves
    /// the joints — the defining property of a correct null-space projector.
    #[test]
    fn pure_nullspace_motion_leaves_tip_still_but_moves_joints() {
        let m = load("redundant7.urdf");
        let f = m.tip_frame();
        let opts = RedundancyOpts::default();
        let zero_v = [0.0; 6];
        let mut rng = SplitMix64(0x13579);
        let mut moved_any = false;
        for _ in 0..40 {
            let z: Vec<f64> = (0..7).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
            let qd = nullspace_step(&m, f, &Q7, &zero_v, &z, &opts);
            let tip = achieved_twist(&m, &Q7, f, &qd);
            // exact MP projector ⇒ tip velocity vanishes to SVD precision
            assert!(tip.norm() < 1e-9, "tip disturbed by {:.2e}", tip.norm());
            let qd_norm = DVector::from_column_slice(&qd).norm();
            if qd_norm > 1e-6 {
                moved_any = true;
                // the applied motion is genuinely the projection of z onto ker(J),
                // so it is bounded by ‖z‖ (never amplified).
                assert!(qd_norm <= DVector::from_column_slice(&z).norm() + 1e-9);
            }
        }
        assert!(moved_any, "null-space motion never moved the joints");
    }

    /// (c) on a NON-redundant 6-DOF arm the null space is empty, so (I−J⁺J)z ≈ 0:
    /// a pure secondary command produces (essentially) no joint motion.
    #[test]
    fn nonredundant_nullspace_is_empty() {
        let m = load("showcase6.urdf");
        assert_eq!(m.ndof, 6);
        let f = m.tip_frame();
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1]; // generic, non-singular
        let opts = RedundancyOpts::default();
        let zero_v = [0.0; 6];
        let mut rng = SplitMix64(0x2468);
        for _ in 0..30 {
            let z: Vec<f64> = (0..6).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
            let qd = nullspace_step(&m, f, &q, &zero_v, &z, &opts);
            let qd_norm = DVector::from_column_slice(&qd).norm();
            assert!(qd_norm < 1e-6, "6-DOF projector left motion {qd_norm:.2e}");
        }
    }

    /// (d) composition: with BOTH a task command and a secondary objective, the tip
    /// still tracks v_desired (the null part contributes exactly 0 to the tip),
    /// proving primary and secondary terms compose without cross-talk.
    #[test]
    fn secondary_objective_does_not_disturb_task() {
        let m = load("redundant7.urdf");
        let f = m.tip_frame();
        let opts = RedundancyOpts::default();
        let mut rng = SplitMix64(0xFEED);
        for _ in 0..40 {
            let v: Vec<f64> = (0..6).map(|_| (rng.next_f64() - 0.5) * 1.5).collect();
            // secondary = joint-limit-avoidance descent, scaled up
            let g = joint_limit_avoidance_gradient(&m, &Q7);
            let z: Vec<f64> = g.iter().map(|x| 5.0 * x).collect();
            let qd = nullspace_step(&m, f, &Q7, &v, &z, &opts);
            let got = achieved_twist(&m, &Q7, f, &qd);
            let want = DVector::from_column_slice(&v);
            // The null-space term adds EXACTLY 0 to the tip; only the primary's
            // damped-inverse bias remains. A cross-talking projector would leak z.
            assert!(
                (&got - &want).norm() < 1e-5,
                "task disturbed by secondary: {:.2e}",
                (&got - &want).norm()
            );
        }
    }

    /// joint_limit_avoidance_gradient points toward mid-range (descent of the
    /// limit potential) and is zero exactly at the center.
    #[test]
    fn limit_gradient_points_to_midrange() {
        let m = load("redundant7.urdf"); // all joints bounded [-2.9, 2.9], mid = 0
        // near the upper limit ⇒ gradient pushes negative (back toward 0)
        let q_hi = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
        let g = joint_limit_avoidance_gradient(&m, &q_hi);
        assert!(
            g.iter().all(|&x| x < 0.0),
            "should push away from upper limit"
        );
        // exactly centered ⇒ zero gradient
        let g0 = joint_limit_avoidance_gradient(&m, &[0.0; 7]);
        assert!(g0.iter().all(|&x| x.abs() < 1e-15));
    }

    /// Invalid inputs return a zero vector of the right length (never panic).
    #[test]
    fn invalid_inputs_return_zero() {
        let m = load("redundant7.urdf");
        let f = m.tip_frame();
        let opts = RedundancyOpts::default();
        // wrong q length
        assert_eq!(
            resolved_rate(&m, f, &[0.0; 3], &[0.0; 6], &opts),
            vec![0.0; 7]
        );
        // wrong v length
        assert_eq!(resolved_rate(&m, f, &Q7, &[0.0; 5], &opts), vec![0.0; 7]);
        // non-finite q
        let mut bad = Q7;
        bad[0] = f64::NAN;
        assert_eq!(resolved_rate(&m, f, &bad, &[0.0; 6], &opts), vec![0.0; 7]);
        // wrong z length in nullspace_step
        assert_eq!(
            nullspace_step(&m, f, &Q7, &[0.0; 6], &[0.0; 3], &opts),
            vec![0.0; 7]
        );
    }
}
