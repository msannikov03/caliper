//! Kinematic calibration for Caliper.
//!
//! The verifiable core of the calibration tail: **joint-offset (zero) calibration**.
//! A real robot's encoders read joint angles in a frame whose zero is offset from the
//! kinematic model's zero by an unknown constant vector `δ` (mechanical assembly,
//! homing, encoder mounting). Given a set of observations
//! `{(commanded qₖ, measured tip pose Tₖ)}`, this crate estimates `δ` such that
//! `FK(qₖ + δ) ≈ Tₖ` for every observation, by damped Gauss–Newton least squares.
//!
//! # Method
//! For each observation the residual is the **body-frame** error twist
//!
//! ```text
//! rₖ = log6( FK(qₖ + δ)⁻¹ · Tₖ )      ∈ se(3),  stored [v; ω]
//! ```
//!
//! At the true offset `δ*`, `FK(qₖ + δ*) = Tₖ`, the error pose is the identity, and
//! every `rₖ = 0`. Differentiating the residual w.r.t. `δ` gives, to first order at the
//! evaluation point, `d rₖ / d δ = −J_b(qₖ + δ)`, where `J_b` is the **LOCAL (body)
//! geometric manipulator Jacobian** of `frame` — exactly [`caliper_kinematics::jacobian`]
//! with [`JacFrame::Body`]. (The residual lives in the tip's body frame, so the Jacobian
//! must too; the sign folds into the normal equations.) Stacking over observations, each
//! Gauss–Newton step solves the damped normal equations
//!
//! ```text
//! ( Σₖ Jₖᵀ Jₖ + λ I ) Δδ = Σₖ Jₖᵀ rₖ ,      δ ← δ + Δδ .
//! ```
//!
//! `δ*` is a fixed point for **any** `λ ≥ 0`: there `rₖ = 0`, so the gradient
//! `Σ Jₖᵀ rₖ` vanishes and `Δδ = 0`. The Levenberg damping `λ` therefore only stabilises
//! the path — on noise-free data the iteration converges to `δ*` exactly, with no bias.
use caliper_kinematics::{JacFrame, fk_frame, jacobian};
use caliper_model::Model;
use caliper_spatial::Se3;
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors from [`calibrate_joint_offsets`] — all input-validation failures.
#[derive(Debug, Error, PartialEq)]
pub enum CalibError {
    /// No observations were supplied; there is nothing to fit.
    #[error("no observations provided")]
    NoObservations,
    /// `frame` is not a valid frame index for `model`.
    #[error("frame index {frame} out of range (model has {n} frames)")]
    BadFrame { frame: usize, n: usize },
    /// An observation's configuration length does not match `model.ndof`.
    #[error("observation {idx}: configuration has {got} joints, expected ndof = {want}")]
    DimMismatch { idx: usize, got: usize, want: usize },
    /// An observation's commanded configuration holds a non-finite value.
    #[error("observation {idx}: non-finite value in commanded configuration")]
    NonFiniteConfig { idx: usize },
    /// An observation's measured pose holds a non-finite value.
    #[error("observation {idx}: non-finite value in measured pose")]
    NonFinitePose { idx: usize },
    /// An option (`lambda`, `tol_step`, `tol_residual`) is non-finite or `lambda < 0`.
    #[error("invalid options: lambda/tolerances must be finite and lambda >= 0")]
    BadOptions,
}

/// Tunables for the damped Gauss–Newton solve.
#[derive(Clone, Copy, Debug)]
pub struct CalibOptions {
    /// Maximum Gauss–Newton iterations.
    pub max_iters: usize,
    /// Levenberg damping `λ ≥ 0` added to the normal-equation diagonal. Small — it only
    /// regularises ill-conditioned / unobservable directions; it does not bias the
    /// noise-free fixed point (gradient vanishes there).
    pub lambda: f64,
    /// Converge once a step's `‖Δδ‖` falls below this.
    pub tol_step: f64,
    /// Converge once the RMS residual falls below this.
    pub tol_residual: f64,
}

impl Default for CalibOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            lambda: 1e-9,
            tol_step: 1e-12,
            tol_residual: 1e-12,
        }
    }
}

/// Result of a joint-offset calibration.
#[derive(Clone, Debug)]
pub struct CalibResult {
    /// Estimated per-joint zero offsets `δ` (length `model.ndof`). Joints that do not
    /// influence `frame` (off the root→frame path) are unobservable and stay `0`.
    pub offsets: Vec<f64>,
    /// RMS body-twist residual at `offsets`: `sqrt( Σₖ ‖rₖ‖² / N )` over the `N`
    /// observations. `→ 0` on a perfect fit.
    pub rms_residual: f64,
    /// Gauss–Newton iterations actually run.
    pub iters: usize,
    /// Whether a convergence criterion (residual or step tolerance) was met.
    pub converged: bool,
}

/// Estimate joint-zero offsets `δ` so that `FK(qₖ + δ) ≈ Tₖ` for every observation,
/// by damped Gauss–Newton (see the crate docs). Starts from `δ = 0`.
///
/// `observations` is a slice of `(commanded configuration, measured tip pose)` pairs;
/// each configuration must have length `model.ndof`. `frame` selects the measured frame
/// (e.g. `model.tip_frame()`).
pub fn calibrate_joint_offsets(
    model: &Model,
    frame: usize,
    observations: &[(Vec<f64>, Se3)],
    opts: CalibOptions,
) -> Result<CalibResult, CalibError> {
    // ---- input validation --------------------------------------------------
    if observations.is_empty() {
        return Err(CalibError::NoObservations);
    }
    if frame >= model.frames.len() {
        return Err(CalibError::BadFrame {
            frame,
            n: model.frames.len(),
        });
    }
    if !opts.lambda.is_finite()
        || opts.lambda < 0.0
        || !opts.tol_step.is_finite()
        || !opts.tol_residual.is_finite()
    {
        return Err(CalibError::BadOptions);
    }
    let n = model.ndof;
    for (idx, (q, t)) in observations.iter().enumerate() {
        if q.len() != n {
            return Err(CalibError::DimMismatch {
                idx,
                got: q.len(),
                want: n,
            });
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(CalibError::NonFiniteConfig { idx });
        }
        if t.0.to_homogeneous().iter().any(|x| !x.is_finite()) {
            return Err(CalibError::NonFinitePose { idx });
        }
    }

    let mut delta = vec![0.0_f64; n];

    // A fixed-only (0-DOF) robot has nothing to calibrate: report the fit at δ = 0.
    if n == 0 {
        let rms = compute_rms(model, frame, observations, &delta);
        return Ok(CalibResult {
            offsets: delta,
            rms_residual: rms,
            iters: 0,
            converged: true,
        });
    }

    // ---- damped Gauss–Newton ----------------------------------------------
    let mut iters = 0;
    let mut converged = false;
    for it in 0..opts.max_iters {
        iters = it + 1;
        let (mut a, g, sumsq) = accumulate(model, frame, observations, &delta);
        let rms = (sumsq / observations.len() as f64).sqrt();
        if rms < opts.tol_residual {
            converged = true;
            break;
        }
        for i in 0..n {
            a[(i, i)] += opts.lambda; // (JᵀJ + λI)
        }
        let step = solve_spd(&a, &g);
        for i in 0..n {
            delta[i] += step[i];
        }
        if step.norm() < opts.tol_step {
            converged = true;
            break;
        }
    }

    // Recompute the RMS at the FINAL offsets so the reported residual always matches
    // the returned `offsets` (the per-iteration value is for the pre-step estimate).
    let rms_residual = compute_rms(model, frame, observations, &delta);
    Ok(CalibResult {
        offsets: delta,
        rms_residual,
        iters,
        converged,
    })
}

/// Accumulate the stacked normal equations at the current `delta`:
/// `H = Σ Jₖᵀ Jₖ`, `g = Σ Jₖᵀ rₖ`, and `Σ ‖rₖ‖²`. `Jₖ` is the LOCAL (body) Jacobian and
/// `rₖ = log6( FK(qₖ+δ)⁻¹ Tₖ )`, both fused from one [`jacobian`] call per observation.
fn accumulate(
    model: &Model,
    frame: usize,
    observations: &[(Vec<f64>, Se3)],
    delta: &[f64],
) -> (DMatrix<f64>, DVector<f64>, f64) {
    let n = model.ndof;
    let mut h = DMatrix::<f64>::zeros(n, n);
    let mut g = DVector::<f64>::zeros(n);
    let mut sumsq = 0.0;
    let mut phi = vec![0.0_f64; n];
    for (q, t) in observations {
        for i in 0..n {
            phi[i] = q[i] + delta[i];
        }
        // `jacobian` returns (FK pose of `frame`, body Jacobian) — FK comes free.
        let (ee, j) = jacobian(model, &phi, frame, JacFrame::Body);
        let err = ee.inverse().compose(t);
        let r = DVector::from_column_slice(err.log().0.as_slice()); // [v; ω]
        sumsq += r.dot(&r);
        h += j.transpose() * &j;
        g += j.transpose() * &r;
    }
    (h, g, sumsq)
}

/// RMS body-twist residual `sqrt( Σ ‖rₖ‖² / N )` at `delta` — uses the independent
/// [`fk_frame`] FK path (no Jacobian), so it cross-checks the fused FK in [`accumulate`].
fn compute_rms(
    model: &Model,
    frame: usize,
    observations: &[(Vec<f64>, Se3)],
    delta: &[f64],
) -> f64 {
    let n = model.ndof;
    let mut sumsq = 0.0;
    let mut phi = vec![0.0_f64; n];
    for (q, t) in observations {
        for i in 0..n {
            phi[i] = q[i] + delta[i];
        }
        let ee = fk_frame(model, &phi, frame);
        let r = ee.inverse().compose(t).log().0;
        sumsq += r.dot(&r);
    }
    (sumsq / observations.len() as f64).sqrt()
}

/// Solve `A x = b` for symmetric-PSD `A` (here `JᵀJ + λI`). Cholesky first (fast, exact
/// for SPD), then LU, then an SVD pseudo-inverse for a rank-deficient `A`.
fn solve_spd(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    if let Some(chol) = a.clone().cholesky() {
        return chol.solve(b);
    }
    if let Some(x) = a.clone().lu().solve(b) {
        return x;
    }
    match a.clone().pseudo_inverse(1e-12) {
        Ok(pinv) => pinv * b,
        Err(_) => DVector::zeros(b.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*; // brings fk_frame, jacobian, Se3, Model, … into scope
    use caliper_spatial::Twist;
    use nalgebra::Vector6;
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }

    /// Deterministic splitmix64 PRNG — repo style, no rand dependency.
    struct SplitMix(u64);
    impl SplitMix {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        }
    }

    /// Synthesize observations `Tₖ = FK(qₖ + δ*)` with the VALIDATED FK, for random `qₖ`.
    fn synth(
        model: &Model,
        frame: usize,
        delta_star: &[f64],
        nobs: usize,
        seed: u64,
    ) -> Vec<(Vec<f64>, Se3)> {
        let mut rng = SplitMix::new(seed);
        (0..nobs)
            .map(|_| {
                let q: Vec<f64> = (0..model.ndof).map(|_| rng.range(-1.0, 1.0)).collect();
                let phi: Vec<f64> = q.iter().zip(delta_star).map(|(a, b)| a + b).collect();
                let t = fk_frame(model, &phi, frame);
                (q, t)
            })
            .collect()
    }

    fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Decisive cross-validation: a known offset is recovered to < 1e-6 with rms → 0
    /// on a non-redundant 6-DOF arm.
    #[test]
    fn recovers_known_offset_6dof() {
        let m = load("showcase6.urdf");
        let f = m.tip_frame();
        let delta_star = [0.05, -0.10, 0.07, -0.03, 0.12, -0.08];
        let obs = synth(&m, f, &delta_star, 16, 0xC0FFEE);
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.converged, "should converge");
        assert!(
            max_abs_err(&res.offsets, &delta_star) < 1e-6,
            "offset error {:e} (offsets {:?})",
            max_abs_err(&res.offsets, &delta_star),
            res.offsets
        );
        assert!(res.rms_residual < 1e-8, "rms {:e}", res.rms_residual);
    }

    /// Same recovery on a 2-DOF arm — covers the small-DOF path.
    #[test]
    fn recovers_known_offset_2dof() {
        let m = load("toy.urdf");
        let f = m.tip_frame();
        let delta_star = [0.13, -0.21];
        let obs = synth(&m, f, &delta_star, 8, 0x1234_5678);
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.converged);
        assert!(max_abs_err(&res.offsets, &delta_star) < 1e-6);
        assert!(res.rms_residual < 1e-8);
    }

    /// Mixed revolute + prismatic: the prismatic offset (a linear zero) must also be
    /// recovered — exercises the `[z; 0]` Jacobian column.
    #[test]
    fn recovers_offset_with_prismatic_joint() {
        let m = load("prismatic.urdf");
        let f = m.tip_frame();
        let delta_star = [0.09, -0.06];
        let obs = synth(&m, f, &delta_star, 10, 0xABCD);
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.converged);
        assert!(
            max_abs_err(&res.offsets, &delta_star) < 1e-6,
            "offsets {:?}",
            res.offsets
        );
    }

    /// No-op: observations already match the model (δ* = 0) ⇒ recovered δ ≈ 0, fast.
    #[test]
    fn no_op_when_already_calibrated() {
        let m = load("showcase6.urdf");
        let f = m.tip_frame();
        let zero = [0.0; 6];
        let obs = synth(&m, f, &zero, 12, 0xFEED);
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.converged);
        assert!(
            res.offsets.iter().all(|x| x.abs() < 1e-9),
            "offsets {:?}",
            res.offsets
        );
        assert!(res.rms_residual < 1e-10);
        // δ = 0 is already the fixed point: the very first residual check converges.
        assert_eq!(res.iters, 1);
    }

    /// Noise robustness: perturb each measured pose by a small body twist (~1e-4) and
    /// assert the recovered offset error stays bounded and small (not exact).
    #[test]
    fn noise_gives_bounded_offset_error() {
        let m = load("showcase6.urdf");
        let f = m.tip_frame();
        let delta_star = [0.04, -0.08, 0.06, -0.02, 0.09, -0.05];
        let mut obs = synth(&m, f, &delta_star, 24, 0x5EED);
        let mut rng = SplitMix::new(0xBADF00D);
        let eps = 1e-4;
        for (_, t) in obs.iter_mut() {
            let xi = Vector6::from_iterator((0..6).map(|_| rng.range(-eps, eps)));
            *t = t.compose(&Se3::exp(&Twist(xi))); // small body-frame measurement noise
        }
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        let err = max_abs_err(&res.offsets, &delta_star);
        // Bounded by the noise level (with margin for conditioning); strictly nonzero.
        assert!(
            err < 1e-2,
            "offset error {err:e} should be bounded by noise"
        );
        assert!(err > 0.0, "noisy data should not recover exactly");
        assert!(res.rms_residual < 1e-2, "rms {:e}", res.rms_residual);
    }

    /// Convergence really happens: a poor initial residual (no calibration applied)
    /// shrinks by orders of magnitude after solving.
    #[test]
    fn residual_collapses_from_uncalibrated_state() {
        let m = load("showcase6.urdf");
        let f = m.tip_frame();
        let delta_star = [0.10, -0.15, 0.12, -0.08, 0.14, -0.11];
        let obs = synth(&m, f, &delta_star, 16, 0x9999);
        let rms_uncalibrated = compute_rms(&m, f, &obs, &[0.0; 6]);
        assert!(
            rms_uncalibrated > 1e-2,
            "the test setup must be miscalibrated"
        );
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.rms_residual < rms_uncalibrated * 1e-6);
    }

    // -------- input validation ---------------------------------------------

    #[test]
    fn rejects_empty_observations() {
        let m = load("toy.urdf");
        let err =
            calibrate_joint_offsets(&m, m.tip_frame(), &[], CalibOptions::default()).unwrap_err();
        assert_eq!(err, CalibError::NoObservations);
    }

    #[test]
    fn rejects_bad_frame() {
        let m = load("toy.urdf");
        let obs = synth(&m, m.tip_frame(), &[0.0, 0.0], 2, 1);
        let bad = m.frames.len();
        let err = calibrate_joint_offsets(&m, bad, &obs, CalibOptions::default()).unwrap_err();
        assert_eq!(
            err,
            CalibError::BadFrame {
                frame: bad,
                n: m.frames.len()
            }
        );
    }

    #[test]
    fn rejects_dim_mismatch() {
        let m = load("toy.urdf");
        let obs = vec![(
            vec![0.1, 0.2, 0.3],
            fk_frame(&m, &[0.0, 0.0], m.tip_frame()),
        )];
        let err =
            calibrate_joint_offsets(&m, m.tip_frame(), &obs, CalibOptions::default()).unwrap_err();
        assert_eq!(
            err,
            CalibError::DimMismatch {
                idx: 0,
                got: 3,
                want: 2
            }
        );
    }

    #[test]
    fn rejects_non_finite_config() {
        let m = load("toy.urdf");
        let obs = vec![(
            vec![f64::NAN, 0.0],
            fk_frame(&m, &[0.0, 0.0], m.tip_frame()),
        )];
        let err =
            calibrate_joint_offsets(&m, m.tip_frame(), &obs, CalibOptions::default()).unwrap_err();
        assert_eq!(err, CalibError::NonFiniteConfig { idx: 0 });
    }

    #[test]
    fn rejects_non_finite_pose() {
        let m = load("toy.urdf");
        let bad_pose = Se3::from_parts(
            nalgebra::Vector3::new(f64::INFINITY, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let obs = vec![(vec![0.0, 0.0], bad_pose)];
        let err =
            calibrate_joint_offsets(&m, m.tip_frame(), &obs, CalibOptions::default()).unwrap_err();
        assert_eq!(err, CalibError::NonFinitePose { idx: 0 });
    }

    #[test]
    fn rejects_bad_options() {
        let m = load("toy.urdf");
        let obs = synth(&m, m.tip_frame(), &[0.0, 0.0], 2, 7);
        let opts = CalibOptions {
            lambda: -1.0,
            ..CalibOptions::default()
        };
        let err = calibrate_joint_offsets(&m, m.tip_frame(), &obs, opts).unwrap_err();
        assert_eq!(err, CalibError::BadOptions);
    }

    /// A fixed-only (0-DOF) model is a trivial, well-defined no-op.
    #[test]
    fn zero_dof_is_trivial_noop() {
        let m = load("fixed_only.urdf");
        assert_eq!(m.ndof, 0);
        let f = m.tip_frame();
        let obs = vec![(vec![], fk_frame(&m, &[], f))];
        let res = calibrate_joint_offsets(&m, f, &obs, CalibOptions::default()).unwrap();
        assert!(res.converged);
        assert!(res.offsets.is_empty());
        assert!(res.rms_residual < 1e-12);
    }
}
