//! Inverse kinematics: damped least squares / Levenberg–Marquardt with
//! manipulability-gated damping, step clamping, joint limits, and multi-restart.
//!
//! Uses the local (body-frame) error `log6(T_cur⁻¹·T_target)` paired with the
//! body Jacobian — the standard, frame-consistent CLIK formulation.
use caliper_kinematics::{JacFrame, fk_frame, jacobian};
use caliper_model::Model;
use caliper_spatial::Se3;
use nalgebra::{Cholesky, DMatrix, DVector};

const UNBOUNDED: f64 = 1.0e6;

#[derive(Clone, Debug)]
pub struct IkOpts {
    pub tol_pos: f64,
    pub tol_rot: f64,
    pub max_iters: usize,
    pub restarts: usize,
    pub lambda0_sq: f64,
    pub w_thresh: f64,
    pub step_clamp: f64,
    pub seed_rng: u64,
}

impl Default for IkOpts {
    fn default() -> Self {
        Self {
            tol_pos: 1e-10,
            tol_rot: 1e-10,
            max_iters: 100,
            restarts: 8,
            lambda0_sq: 1e-3,
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
    best.unwrap()
}

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
        };

        // LM normal equations: (JᵀJ + λ²(I + diag(JᵀJ))) dq = Jᵀe
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

    fn toy() -> Model {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/fixtures/robots/toy.urdf"
        );
        Model::from_urdf(Path::new(p)).unwrap()
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
}
