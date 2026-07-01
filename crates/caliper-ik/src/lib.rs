//! Inverse kinematics: damped least squares / Levenberg–Marquardt with
//! manipulability-gated damping, step clamping, joint limits, and multi-restart.
//!
//! Uses the local (body-frame) error `log6(T_cur⁻¹·T_target)` paired with the
//! body Jacobian — the standard, frame-consistent CLIK formulation.
use caliper_kinematics::{JacFrame, fk_frame, jacobian};
use caliper_model::Model;
use caliper_spatial::Se3;
use nalgebra::{Cholesky, DVector};

mod analytic;
pub use analytic::analytic_ik_6r;

mod redundancy;
pub use redundancy::{
    RedundancyOpts, joint_limit_avoidance_gradient, nullspace_step, resolved_rate,
};

const UNBOUNDED: f64 = 1.0e6;

#[derive(Clone, Debug)]
pub struct IkOpts {
    pub tol_pos: f64,
    pub tol_rot: f64,
    pub max_iters: usize,
    pub restarts: usize,
    pub lambda0_sq: f64,
    /// Always-on Tikhonov floor on the damping so `JᵀJ` stays SPD (essential for
    /// redundant arms where `JᵀJ` is rank-deficient and Cholesky would fail).
    pub lambda_floor_sq: f64,
    pub w_thresh: f64,
    pub step_clamp: f64,
    pub seed_rng: u64,
}

impl Default for IkOpts {
    fn default() -> Self {
        Self {
            tol_pos: 1e-9,
            tol_rot: 1e-9,
            max_iters: 100,
            restarts: 8,
            lambda0_sq: 1e-3,
            lambda_floor_sq: 1e-10,
            w_thresh: 1e-3,
            step_clamp: 0.3,
            seed_rng: 0xC0FFEE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IkResult {
    pub success: bool,
    pub q: Vec<f64>,
    pub residual: f64,
    pub iters: usize,
    pub restarts_used: usize,
}

/// Solve IK for `frame` to reach `target`, starting from `seed`.
pub fn ik(model: &Model, frame: usize, target: &Se3, seed: &[f64], opts: &IkOpts) -> IkResult {
    if seed.len() != model.ndof {
        return IkResult {
            success: false,
            q: vec![0.0; model.ndof],
            residual: f64::INFINITY,
            iters: 0,
            restarts_used: 0,
        };
    }
    let (lo, hi) = limits(model);
    let mut rng = SplitMix64(opts.seed_rng);
    let mut best: Option<IkResult> = None;
    for r in 0..opts.restarts.max(1) {
        let mut q0 = if r == 0 {
            seed.to_vec()
        } else {
            random_in_limits(&lo, &hi, &mut rng)
        };
        clamp(&mut q0, &lo, &hi);
        let res = solve_one(model, frame, target, q0, &lo, &hi, opts, r);
        if res.success {
            return res;
        }
        if best.as_ref().is_none_or(|b| res.residual < b.residual) {
            best = Some(res);
        }
    }
    best.expect("restart loop ran at least once")
}

#[allow(clippy::too_many_arguments)]
fn solve_one(
    model: &Model,
    frame: usize,
    target: &Se3,
    mut q: Vec<f64>,
    lo: &[f64],
    hi: &[f64],
    opts: &IkOpts,
    restart: usize,
) -> IkResult {
    let n = model.ndof;
    for it in 0..opts.max_iters {
        // A non-finite q (e.g. a divergent step near a singularity on an unbounded
        // joint) would feed NaN into nalgebra's SVD, which never terminates. Bail.
        if !q.iter().all(|x| x.is_finite()) {
            return IkResult {
                success: false,
                q,
                residual: f64::INFINITY,
                iters: it,
                restarts_used: restart,
            };
        }
        let t_cur = fk_frame(model, &q, frame);
        // local-frame error twist e = log6(T_cur⁻¹ · T_target), [v; ω]
        let twist = Se3(t_cur.0.inverse() * target.0).log().0;
        let e = DVector::from_iterator(6, twist.iter().copied());
        let lin = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        let ang = (e[3] * e[3] + e[4] * e[4] + e[5] * e[5]).sqrt();
        if lin < opts.tol_pos && ang < opts.tol_rot {
            return IkResult {
                success: true,
                q,
                residual: e.norm(),
                iters: it,
                restarts_used: restart,
            };
        }

        let (_, j) = jacobian(model, &q, frame, JacFrame::Body); // 6 × n, body frame

        // manipulability-gated damping
        let sv = j.clone().svd(false, false).singular_values;
        let w: f64 = sv.iter().product();
        let lambda_sq = if w >= opts.w_thresh {
            0.0
        } else {
            opts.lambda0_sq * (1.0 - w / opts.w_thresh).powi(2)
        }
        .max(opts.lambda_floor_sq); // floor keeps H SPD for redundant (wide-J) arms

        // LM normal equations: (JᵀJ + λ²(I + diag(JᵀJ))) dq = Jᵀe
        // INTENTIONAL: this is the joint-space Levenberg–Marquardt formulation —
        // damping is applied to the n×n `JᵀJ` (with a Tikhonov SPD floor), NOT the
        // task-space `JJᵀ`. Both are valid; this one is Pinocchio-cross-validated.
        // Do not "fix" it to a task-space damped pseudo-inverse.
        let jt = j.transpose();
        let mut h = &jt * &j;
        let g = &jt * &e;
        let diag = h.diagonal();
        for i in 0..n {
            h[(i, i)] += lambda_sq * (1.0 + diag[i]);
        }
        let mut dq = match Cholesky::new(h) {
            Some(c) => c.solve(&g),
            None => j
                .clone()
                .svd(true, true)
                .solve(&e, 1e-12)
                .unwrap_or_else(|_| DVector::zeros(n)),
        };

        // trust-region step clamp
        let mx = dq.amax();
        if mx > opts.step_clamp {
            dq *= opts.step_clamp / mx;
        }
        for i in 0..n {
            q[i] = (q[i] + dq[i]).clamp(lo[i], hi[i]);
        }
    }
    let e = Se3(fk_frame(model, &q, frame).0.inverse() * target.0)
        .log()
        .0;
    IkResult {
        success: false,
        q,
        residual: e.norm(),
        iters: opts.max_iters,
        restarts_used: restart,
    }
}

fn limits(model: &Model) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![-UNBOUNDED; model.ndof];
    let mut hi = vec![UNBOUNDED; model.ndof];
    for (i, l) in model.limits.iter().enumerate() {
        if let Some((a, b)) = l {
            lo[i] = *a;
            hi[i] = *b;
        }
    }
    (lo, hi)
}

fn clamp(q: &mut [f64], lo: &[f64], hi: &[f64]) {
    for i in 0..q.len() {
        q[i] = q[i].clamp(lo[i], hi[i]);
    }
}

fn random_in_limits(lo: &[f64], hi: &[f64], rng: &mut SplitMix64) -> Vec<f64> {
    (0..lo.len())
        .map(|i| {
            let (a, b) = if lo[i] <= -UNBOUNDED * 0.5 {
                (-std::f64::consts::PI, std::f64::consts::PI)
            } else {
                (lo[i], hi[i])
            };
            a + rng.next_f64() * (b - a)
        })
        .collect()
}

/// Deterministic SplitMix64 — keeps IK dependency-free + reproducible.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }
    fn toy() -> Model {
        load("toy.urdf")
    }

    /// FK(IK(FK(q))) == FK(q): the headline round-trip, over a deterministic sweep.
    #[test]
    fn ik_round_trip() {
        let m = toy();
        let frame = m.tip_frame();
        let opts = IkOpts::default();
        let mut rng = SplitMix64(0x12345);
        let mut worst = 0.0_f64;
        for _ in 0..300 {
            let q_true: Vec<f64> = (0..m.ndof).map(|_| (rng.next_f64() - 0.5) * 2.5).collect();
            let target = fk_frame(&m, &q_true, frame);
            // start from a perturbed seed
            let seed: Vec<f64> = q_true
                .iter()
                .map(|&x| x + (rng.next_f64() - 0.5) * 0.6)
                .collect();
            let res = ik(&m, frame, &target, &seed, &opts);
            assert!(res.success, "IK failed (residual {:.2e})", res.residual);
            let achieved = fk_frame(&m, &res.q, frame);
            let err = Se3(achieved.0.inverse() * target.0).log().0.norm();
            worst = worst.max(err);
            assert!(err < 1e-8, "FK(IK) pose error {err:.3e}");
        }
        assert!(worst < 1e-8);
    }

    /// Multi-restart recovers from a deliberately bad seed.
    #[test]
    fn ik_multi_restart_recovers() {
        let m = toy();
        let frame = m.tip_frame();
        let target = fk_frame(&m, &[0.8, -1.1], frame);
        let res = ik(&m, frame, &target, &[3.0, 3.0], &IkOpts::default());
        assert!(res.success);
        let err = Se3(fk_frame(&m, &res.q, frame).0.inverse() * target.0)
            .log()
            .0
            .norm();
        assert!(err < 1e-8);
    }

    #[test]
    fn ik_redundant_7dof() {
        let m = load("redundant7.urdf");
        assert_eq!(m.ndof, 7);
        let frame = m.tip_frame();
        let opts = IkOpts::default();
        let mut rng = SplitMix64(0x777);
        for _ in 0..50 {
            let q_true: Vec<f64> = (0..7).map(|_| (rng.next_f64() - 0.5) * 2.0).collect();
            let target = fk_frame(&m, &q_true, frame);
            let seed: Vec<f64> = q_true
                .iter()
                .map(|&x| x + (rng.next_f64() - 0.5) * 0.4)
                .collect();
            let res = ik(&m, frame, &target, &seed, &opts);
            assert!(
                res.success,
                "redundant IK failed: residual {:.2e}",
                res.residual
            );
            let err = Se3(fk_frame(&m, &res.q, frame).0.inverse() * target.0)
                .log()
                .0
                .norm();
            assert!(err < 1e-8, "pose error {err:.2e}");
        }
    }

    #[test]
    fn ik_prismatic_joint() {
        let m = load("prismatic.urdf");
        let frame = m.tip_frame();
        let target = fk_frame(&m, &[0.5, 0.3], frame);
        let res = ik(&m, frame, &target, &[0.0, 0.1], &IkOpts::default());
        assert!(res.success);
        let err = Se3(fk_frame(&m, &res.q, frame).0.inverse() * target.0)
            .log()
            .0
            .norm();
        assert!(err < 1e-8);
    }

    #[test]
    fn ik_rejects_wrong_seed_length() {
        let m = toy();
        let frame = m.tip_frame();
        let target = fk_frame(&m, &[0.0, 0.0], frame);
        let res = ik(&m, frame, &target, &[0.0, 0.0, 0.0], &IkOpts::default());
        assert!(!res.success && !res.residual.is_finite());
    }

    /// A non-finite seed must fail fast, never enter the non-terminating SVD.
    /// (If the solve_one guard regresses, this test hangs and the suite times out.)
    #[test]
    fn ik_bails_on_nonfinite_seed_without_hanging() {
        let m = toy();
        let f = m.tip_frame();
        let target = fk_frame(&m, &[0.3, 0.2], f);
        let res = ik(
            &m,
            f,
            &target,
            &[f64::NAN, 0.0],
            &IkOpts {
                restarts: 1,
                ..Default::default()
            },
        );
        assert!(!res.success);
    }
}
