//! Closed-form (analytic) inverse kinematics for a canonical spherical-wrist 6R
//! arm — the `showcase6` family: a vertical base joint, two parallel "shoulder /
//! elbow" joints, and a spherical wrist whose last three axes intersect at a point.
//!
//! This is the textbook *kinematic decoupling*: joints 1–3 place the wrist centre
//! (which is invariant to the wrist joints), then joints 4–6 realise the residual
//! orientation as a ZYZ Euler triple. All discrete branches (shoulder front/back,
//! elbow up/down, wrist flip) are enumerated and every returned branch is
//! independently re-checked against [`fk_frame`] before being kept, so a returned
//! solution is guaranteed to reproduce the target to floating-point precision.
//!
//! Structure detection is deliberately STRICT: [`analytic_ik_6r`] returns `None`
//! whenever the model is not a recognised spherical-wrist 6R in the exact canonical
//! alignment the closed form assumes (so callers fall back to the numeric solver).
//! Reading real geometry from the [`Model`] (offsets, link lengths, the wrist
//! centre) keeps it correct for any robot in this family, not just `showcase6`.
use caliper_kinematics::fk_frame;
use caliper_model::{JointKind, Model};
use caliper_spatial::Se3;
use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use std::f64::consts::PI;

/// Detection / verification tolerances. `STRUCT_TOL` gates the strict structural
/// match; `VERIFY_TOL` is the loose gross-error filter on each branch's FK residual
/// (a correct branch lands ~1e-12, far below this — the filter only drops a branch
/// produced by a degenerate/gimbal split, never a genuine solution).
const STRUCT_TOL: f64 = 1e-9;
const VERIFY_TOL: f64 = 1e-7;

/// Closed-form IK for a canonical spherical-wrist 6R arm.
///
/// Returns `Some(branches)` with every joint-limit-respecting solution that
/// reproduces `target` for `frame` (each `Vec<f64>` is a full `ndof==6` config).
/// When `seed` is given, the branch nearest the seed (per-joint wrapped distance)
/// is placed first. Returns `Some(vec![])` when the structure is recognised but the
/// pose is unreachable, and `None` when the model is NOT a recognised spherical-wrist
/// 6R — the signal for the caller to fall back to numeric [`crate::ik`].
pub fn analytic_ik_6r(
    model: &Model,
    frame: usize,
    target: &Se3,
    seed: Option<&[f64]>,
) -> Option<Vec<Vec<f64>>> {
    let p = detect(model, frame)?;

    // ----- wrist centre in world, from the target pose -----
    // tool = X6 · F  ⇒  X6 = target · F⁻¹ ; wrist centre = X6 · c6 (a point).
    let x6 = target.compose(&p.f_offset.inverse());
    let wc_world = (x6.0 * Point3::from(p.c6)).coords;
    let w = wc_world - p.o1; // wrist centre relative to joint-1 origin
    let (wx, wy, wz) = (w.x, w.y, w.z);

    let r_target = target.rotation();
    let r_f_t = p.f_offset.rotation().transpose();

    let mut out: Vec<Vec<f64>> = Vec::new();

    // ----- q1: shoulder front / back -----
    let r = (wx * wx + wy * wy).sqrt();
    if r < STRUCT_TOL {
        // wrist centre on the base axis: q1 indeterminate, skip (singular).
        return Some(out);
    }
    let ratio = p.d_y / r;
    if ratio.abs() > 1.0 + 1e-9 {
        return Some(out); // shoulder offset unreachable
    }
    let asin_s = ratio.clamp(-1.0, 1.0).asin();
    let phi = wy.atan2(wx);
    for &q1 in &[phi - asin_s, phi - (PI - asin_s)] {
        let (c1, s1) = (q1.cos(), q1.sin());
        // planar target relative to the joint-2 pivot, in the F1 (post-q1) plane.
        let x_f1 = wx * c1 + wy * s1;
        let dx = x_f1 - p.o2.x;
        let dz = wz - p.o2.z;
        let dd = dx * dx + dz * dz;
        let cos_q3 = (dd - p.l_a * p.l_a - p.l_b * p.l_b) / (2.0 * p.l_a * p.l_b);
        if cos_q3.abs() > 1.0 + 1e-9 {
            continue; // elbow cannot reach this distance
        }
        let q3_mag = cos_q3.clamp(-1.0, 1.0).acos();
        // ----- q3: elbow up / down -----
        for &q3 in &[q3_mag, -q3_mag] {
            let q2 = dx.atan2(dz) - (p.l_b * q3.sin()).atan2(p.l_a + p.l_b * q3.cos());
            // residual orientation the wrist must produce: M = R03ᵀ · R_target · R_Fᵀ
            let r03 = rot_z(q1) * rot_y(q2 + q3);
            let m = r03.transpose() * r_target * r_f_t;
            let seed_q4 = seed.map(|s| s[3]);
            // ----- q4,q5,q6: wrist flip -----
            for wrist in zyz_branches(&m, seed_q4) {
                let raw = [q1, q2, q3, wrist[0], wrist[1], wrist[2]];
                if let Some(q) = fit_limits(model, &raw) {
                    // independent FK re-check: only keep branches that truly land.
                    if fk_residual(model, frame, target, &q) < VERIFY_TOL {
                        push_unique(&mut out, q);
                    }
                }
            }
        }
    }

    if let Some(s) = seed {
        // bring the seed-nearest branch to the front (stable for the rest).
        out.sort_by(|a, b| {
            seed_dist(a, s)
                .partial_cmp(&seed_dist(b, s))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Some(out)
}

/// Constant geometry of a recognised canonical spherical-wrist 6R, read from the
/// `Model` so the closed form is exact for any robot in the family.
struct WristParams {
    o1: Vector3<f64>, // joint-1 origin (base → j1 translation)
    o2: Vector3<f64>, // joint-2 origin in the F1 (post-q1) frame
    l_a: f64,         // upper-arm length (j2 → j3, along local z)
    l_b: f64,         // forearm length (j3 → wrist centre, along local z)
    d_y: f64,         // constant lateral (shoulder) offset in the F1 frame
    c6: Vector3<f64>, // wrist centre expressed in the joint-6 frame
    f_offset: Se3,    // joint-6 frame → tool (`frame`) fixed transform
}

/// Strictly recognise the canonical spherical-wrist 6R alignment. Every condition
/// must hold (to `STRUCT_TOL`) or we return `None` so the algebra below is valid.
fn detect(model: &Model, frame: usize) -> Option<WristParams> {
    if model.ndof != 6 || model.kind.iter().any(|&k| k != JointKind::Revolute) {
        return None;
    }
    // serial chain j1..j6
    let want_parent = [None, Some(0), Some(1), Some(2), Some(3), Some(4)];
    if model.parent[..] != want_parent[..] {
        return None;
    }
    if frame >= model.frames.len() || model.frames[frame].anchor != Some(5) {
        return None; // tool must ride on the last (wrist) joint
    }
    // all inter-joint fixed rotations identity (canonical alignment)
    for pj in &model.parent_to_joint {
        if (pj.rotation() - Matrix3::identity()).norm() > STRUCT_TOL {
            return None;
        }
    }
    // exact canonical axis pattern: z, y, y | z, y, z
    let want_axis = [
        Vector3::z(),
        Vector3::y(),
        Vector3::y(),
        Vector3::z(),
        Vector3::y(),
        Vector3::z(),
    ];
    for (a, e) in model.axis.iter().zip(want_axis.iter()) {
        if (a - e).norm() > STRUCT_TOL {
            return None;
        }
    }

    let o1 = model.parent_to_joint[0].translation_vec();
    let o2 = model.parent_to_joint[1].translation_vec();
    let pj3 = model.parent_to_joint[2].translation_vec();
    if pj3.x.abs() > STRUCT_TOL {
        return None; // upper-arm must lie along the local z (clean planar 2R)
    }
    let l_a = pj3.z;

    // ----- wrist centre, as the common intersection of axes 4,5,6, in j3's frame -----
    let t3 = model.parent_to_joint[3];
    let t4 = t3.compose(&model.parent_to_joint[4]);
    let t5 = t4.compose(&model.parent_to_joint[5]);
    let lines = [
        (t3.translation_vec(), t3.rotation() * model.axis[3]),
        (t4.translation_vec(), t4.rotation() * model.axis[4]),
        (t5.translation_vec(), t5.rotation() * model.axis[5]),
    ];
    let wc2 = intersect_lines(&lines)?; // wrist centre in joint-3 frame
    if wc2.x.abs() > STRUCT_TOL {
        return None; // forearm must lie along the local z
    }
    let l_b = wc2.z;
    if l_a.abs() < STRUCT_TOL || l_b.abs() < STRUCT_TOL {
        return None; // degenerate link lengths
    }

    // constant lateral offset folded across the y-invariant Ry chain
    let d_y = o2.y + pj3.y + wc2.y;
    // wrist centre in the joint-6 frame (invariant to the wrist angles)
    let c6 = (t5.0.inverse() * Point3::from(wc2)).coords;

    Some(WristParams {
        o1,
        o2,
        l_a,
        l_b,
        d_y,
        c6,
        f_offset: model.frames[frame].offset,
    })
}

/// Least-squares common point of a set of lines `(point, unit-direction)`; returns
/// `None` if the lines are near-parallel (no well-defined intersection) or the
/// residual distance exceeds `VERIFY_TOL` (axes do not actually intersect).
fn intersect_lines(lines: &[(Vector3<f64>, Vector3<f64>)]) -> Option<Vector3<f64>> {
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for (p, d) in lines {
        let dn = d.normalize();
        let proj = Matrix3::identity() - dn * dn.transpose(); // ⊥-projector onto the line
        a += proj;
        b += proj * p;
    }
    let x = a.try_inverse()? * b;
    let mut resid_sq = 0.0;
    for (p, d) in lines {
        let dn = d.normalize();
        let proj = Matrix3::identity() - dn * dn.transpose();
        resid_sq += (proj * (x - p)).norm_squared();
    }
    (resid_sq.sqrt() < VERIFY_TOL).then_some(x)
}

/// The two ZYZ Euler solutions of `M = Rz(a)·Ry(b)·Rz(c)` (wrist flip). At a wrist
/// gimbal (b≈0 or b≈π) only one branch survives, parametrised by `seed_q4`.
fn zyz_branches(m: &Matrix3<f64>, seed_q4: Option<f64>) -> Vec<[f64; 3]> {
    let (m02, m12, m22) = (m[(0, 2)], m[(1, 2)], m[(2, 2)]);
    let (m20, m21) = (m[(2, 0)], m[(2, 1)]);
    let sb = (m02 * m02 + m12 * m12).sqrt(); // = |sin b|
    if sb < 1e-8 {
        let q4 = seed_q4.unwrap_or(0.0);
        if m22 >= 0.0 {
            // b≈0: only (q4+q6) is determined.
            let total = m[(1, 0)].atan2(m[(0, 0)]);
            vec![[q4, 0.0, total - q4]]
        } else {
            // b≈π: only (q4−q6) is determined.
            let diff = (-m[(0, 1)]).atan2(-m[(0, 0)]);
            vec![[q4, PI, q4 - diff]]
        }
    } else {
        let b = sb.atan2(m22); // ∈ (0,π)
        vec![
            [m12.atan2(m02), b, m21.atan2(-m20)],
            [(-m12).atan2(-m02), -b, (-m21).atan2(m20)],
        ]
    }
}

/// Map each raw joint angle into its limits (wrapping by 2π where that helps);
/// `None` if any joint cannot be made to fit. Continuous joints (no limit) are
/// wrapped to the principal branch.
fn fit_limits(model: &Model, raw: &[f64; 6]) -> Option<Vec<f64>> {
    let mut q = Vec::with_capacity(6);
    for (i, &val) in raw.iter().enumerate() {
        match model.limits[i] {
            None => q.push(wrap_pi(val)),
            Some((lo, hi)) => {
                let mut hit = None;
                for k in -2..=2 {
                    let v = val + (k as f64) * 2.0 * PI;
                    if v >= lo - 1e-9 && v <= hi + 1e-9 {
                        hit = Some(v.clamp(lo, hi));
                        break;
                    }
                }
                q.push(hit?);
            }
        }
    }
    Some(q)
}

fn fk_residual(model: &Model, frame: usize, target: &Se3, q: &[f64]) -> f64 {
    let achieved = fk_frame(model, q, frame);
    Se3(achieved.0.inverse() * target.0).log().0.norm()
}

/// Append `q` unless an effectively-identical config is already present.
fn push_unique(out: &mut Vec<Vec<f64>>, q: Vec<f64>) {
    let dup = out
        .iter()
        .any(|e| e.iter().zip(&q).all(|(a, b)| wrap_pi(a - b).abs() < 1e-6));
    if !dup {
        out.push(q);
    }
}

/// Sum of squared per-joint wrapped angular distances to the seed.
fn seed_dist(q: &[f64], seed: &[f64]) -> f64 {
    q.iter()
        .zip(seed)
        .map(|(a, b)| {
            let d = wrap_pi(a - b);
            d * d
        })
        .sum()
}

/// Wrap an angle into (−π, π].
fn wrap_pi(a: f64) -> f64 {
    let mut x = (a + PI).rem_euclid(2.0 * PI) - PI;
    if x <= -PI {
        x += 2.0 * PI;
    }
    x
}

fn rot_z(a: f64) -> Matrix3<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), a)
        .to_rotation_matrix()
        .into_inner()
}

fn rot_y(a: f64) -> Matrix3<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), a)
        .to_rotation_matrix()
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IkOpts, ik};
    use std::path::Path;

    fn load(name: &str) -> Model {
        let p = format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        Model::from_urdf(Path::new(&p)).unwrap()
    }

    /// Deterministic SplitMix64 → [0,1), matching the repo idiom (no rand crate).
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + self.f() * (hi - lo)
        }
        /// A random config within a safe band of showcase6's limits, away from the
        /// wrist gimbal (q5 bounded off zero) so a clean 8-branch tree exists.
        fn config(&mut self) -> [f64; 6] {
            [
                self.range(-2.0, 2.0),
                self.range(-1.5, 1.5),
                self.range(-2.0, 2.0),
                self.range(-2.0, 2.0),
                self.range(0.25, 1.6),
                self.range(-2.5, 2.5),
            ]
        }
    }

    fn wrapped_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| wrap_pi(x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// CROSS-VALIDATION: over many random reachable poses, every returned branch
    /// must reproduce the target via FK to < 1e-9, the TRUE config must appear among
    /// the branches, every branch obeys joint limits, and the branch count is sane.
    #[test]
    fn analytic_branches_reproduce_target_and_recover_truth() {
        let m = load("showcase6.urdf");
        let frame = m.tip_frame();
        let mut rng = Rng(0xCA11_BE12);
        let mut worst = 0.0_f64;
        let mut max_branches = 0usize;
        for _ in 0..400 {
            let q_true = rng.config();
            let target = fk_frame(&m, &q_true, frame);
            let seed: Vec<f64> = q_true.iter().map(|&x| x + (rng.f() - 0.5) * 0.3).collect();
            let branches = analytic_ik_6r(&m, frame, &target, Some(&seed))
                .expect("showcase6 must be recognised as spherical-wrist 6R");
            assert!(
                !branches.is_empty(),
                "a reachable pose must yield >=1 branch"
            );
            assert!(
                branches.len() <= 8,
                "spherical-wrist 6R has at most 8 branches, got {}",
                branches.len()
            );
            max_branches = max_branches.max(branches.len());
            // every branch lands on the target AND respects limits
            for q in &branches {
                assert_eq!(q.len(), 6);
                let err = fk_residual(&m, frame, &target, q);
                worst = worst.max(err);
                assert!(err < 1e-9, "branch FK residual {err:.3e}");
                for (qi, lim) in q.iter().zip(&m.limits) {
                    if let Some((lo, hi)) = lim {
                        assert!(*qi >= lo - 1e-9 && *qi <= hi + 1e-9, "branch out of limits");
                    }
                }
            }
            // the ground-truth config is reproduced by some branch
            let recovered = branches
                .iter()
                .map(|q| wrapped_dist(q, &q_true))
                .fold(f64::INFINITY, f64::min);
            assert!(
                recovered < 1e-6,
                "true config not recovered (min dist {recovered:.2e})"
            );
        }
        assert!(worst < 1e-9, "worst branch residual {worst:.3e}");
        assert!(max_branches >= 2, "expected multiple branches somewhere");
    }

    /// The seed-nearest branch (returned first) must land on the target AND agree
    /// with the engine's NUMERIC IK solution's tip pose — two independent solvers.
    #[test]
    fn seed_nearest_matches_numeric_ik() {
        let m = load("showcase6.urdf");
        let frame = m.tip_frame();
        let opts = IkOpts::default();
        let mut rng = Rng(0x5EED_1234);
        let mut compared = 0;
        for _ in 0..120 {
            let q_true = rng.config();
            let target = fk_frame(&m, &q_true, frame);
            let seed: Vec<f64> = q_true.iter().map(|&x| x + (rng.f() - 0.5) * 0.25).collect();

            let branches = analytic_ik_6r(&m, frame, &target, Some(&seed)).unwrap();
            assert!(!branches.is_empty());
            let nearest = &branches[0];
            // nearest reproduces the target (the analytic solver is exact + always solves)
            assert!(fk_residual(&m, frame, &target, nearest) < 1e-9);
            // it is genuinely the closest branch to the seed
            for q in &branches {
                assert!(seed_dist(nearest, &seed) <= seed_dist(q, &seed) + 1e-12);
            }

            // Cross-check against the numeric solver. The iterative solver can fail to
            // converge from a far seed near a singularity (the analytic one does not —
            // that robustness is a feature). So WHEN numeric succeeds, require the two
            // solvers' tips to coincide; otherwise skip (analytic already validated above).
            let num = ik(&m, frame, &target, &seed, &opts);
            if num.success {
                let num_tip = fk_frame(&m, &num.q, frame);
                let ana_tip = fk_frame(&m, nearest, frame);
                let agree = Se3(num_tip.0.inverse() * ana_tip.0).log().0.norm();
                assert!(agree < 1e-7, "analytic vs numeric tip mismatch {agree:.3e}");
                compared += 1;
            }
        }
        // the cross-check must actually have run on a healthy majority of poses
        assert!(
            compared >= 90,
            "numeric IK converged on only {compared}/120 poses"
        );
    }

    /// Non-spherical-wrist / wrong-DOF fixtures must return `None` so callers fall
    /// back to the numeric solver.
    #[test]
    fn returns_none_on_non_spherical_wrist() {
        let r7 = load("redundant7.urdf");
        assert!(analytic_ik_6r(&r7, r7.tip_frame(), &Se3::identity(), None).is_none());
        let toy = load("toy.urdf");
        assert!(analytic_ik_6r(&toy, toy.tip_frame(), &Se3::identity(), None).is_none());
        let pris = load("prismatic.urdf");
        assert!(analytic_ik_6r(&pris, pris.tip_frame(), &Se3::identity(), None).is_none());
    }

    /// Without a seed the solver still returns all valid, distinct, on-target
    /// branches (order unspecified).
    #[test]
    fn no_seed_returns_distinct_on_target_branches() {
        let m = load("showcase6.urdf");
        let frame = m.tip_frame();
        let mut rng = Rng(0x00A1_1B00);
        for _ in 0..60 {
            let q_true = rng.config();
            let target = fk_frame(&m, &q_true, frame);
            let branches = analytic_ik_6r(&m, frame, &target, None).unwrap();
            assert!(!branches.is_empty());
            for (i, a) in branches.iter().enumerate() {
                assert!(fk_residual(&m, frame, &target, a) < 1e-9);
                for b in branches.iter().skip(i + 1) {
                    assert!(wrapped_dist(a, b) > 1e-6, "duplicate branches returned");
                }
            }
        }
    }

    /// The wrist centre is invariant to the wrist joints: detection's `c6` placed in
    /// the world from FK must be independent of q4,q5,q6 — a structural sanity check.
    #[test]
    fn wrist_centre_is_invariant_to_wrist_joints() {
        let m = load("showcase6.urdf");
        let frame = m.tip_frame();
        let p = detect(&m, frame).expect("recognised");
        let base = {
            let q = [0.3, -0.4, 0.6, 0.0, 0.7, 0.0];
            let x6 = fk_frame(&m, &q, frame).compose(&p.f_offset.inverse());
            (x6.0 * Point3::from(p.c6)).coords
        };
        for wrist in [[1.2, 0.7, -0.9], [-2.0, 1.3, 2.5], [0.5, 0.2, 1.1]] {
            let q = [0.3, -0.4, 0.6, wrist[0], wrist[1], wrist[2]];
            let x6 = fk_frame(&m, &q, frame).compose(&p.f_offset.inverse());
            let wc = (x6.0 * Point3::from(p.c6)).coords;
            assert!(
                (wc - base).norm() < 1e-12,
                "wrist centre moved with wrist joints"
            );
        }
    }
}
