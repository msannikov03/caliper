//! Forward kinematics, Jacobians, and singularity analysis.
use caliper_model::{JointKind, Model};
use caliper_spatial::{Se3, exp_prismatic, exp_revolute};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3, Vector6};

/// Local exp of movable joint `i` at configuration `q`.
#[inline]
fn joint_local(model: &Model, i: usize, q: &[f64]) -> Se3 {
    match model.kind[i] {
        JointKind::Revolute => exp_revolute(&model.axis[i], q[i]),
        JointKind::Prismatic => exp_prismatic(&model.axis[i], q[i]),
    }
}

/// Forward kinematics: world pose of every movable joint frame, in topological
/// order. `q.len() == out.len() == model.ndof`. No allocation.
pub fn fk_joints(model: &Model, q: &[f64], out: &mut [Se3]) {
    debug_assert_eq!(q.len(), model.ndof);
    debug_assert_eq!(out.len(), model.ndof);
    for i in 0..model.ndof {
        let parent_world = match model.parent[i] {
            Some(p) => out[p],
            None => Se3::identity(),
        };
        out[i] = parent_world
            .compose(&model.parent_to_joint[i])
            .compose(&joint_local(model, i, q));
    }
}

/// World pose of a link frame, given precomputed movable-joint world poses.
pub fn frame_pose(model: &Model, joint_world: &[Se3], frame: usize) -> Se3 {
    let lf = &model.frames[frame];
    let base = match lf.anchor {
        Some(j) => joint_world[j],
        None => Se3::identity(),
    };
    base.compose(&lf.offset)
}

/// World pose of a single frame, computed independently by walking the
/// root→frame ancestor chain (the cross-check path against [`fk_joints`]).
pub fn fk_frame(model: &Model, q: &[f64], frame: usize) -> Se3 {
    let lf = &model.frames[frame];
    let base = match lf.anchor {
        Some(j) => fk_movable(model, q, j),
        None => Se3::identity(),
    };
    base.compose(&lf.offset)
}

fn fk_movable(model: &Model, q: &[f64], j: usize) -> Se3 {
    let parent_world = match model.parent[j] {
        Some(p) => fk_movable(model, q, p),
        None => Se3::identity(),
    };
    parent_world
        .compose(&model.parent_to_joint[j])
        .compose(&joint_local(model, j, q))
}

/// World pose of the model's tip frame.
pub fn fk_tip(model: &Model, q: &[f64]) -> Se3 {
    fk_frame(model, q, model.tip_frame())
}

/// Reference frame for a geometric Jacobian.
#[derive(Clone, Copy, Debug)]
pub enum JacFrame {
    /// World-aligned at the frame origin — Pinocchio `LOCAL_WORLD_ALIGNED`.
    World,
    /// Frame-local — Pinocchio `LOCAL`.
    Body,
}

/// Geometric Jacobian (6 × ndof, rows `[v; ω]`) of `frame`, fused with FK.
/// Joints not on the root→`frame` path get zero columns. Returns `(ee_pose, J)`.
pub fn jacobian(model: &Model, q: &[f64], frame: usize, jframe: JacFrame) -> (Se3, DMatrix<f64>) {
    let mut jw = vec![Se3::identity(); model.ndof];
    fk_joints(model, q, &mut jw);
    let ee = frame_pose(model, &jw, frame);
    let p_e = ee.translation_vec();

    // ancestor mask: which movable joints lie on the path to `frame`
    let mut is_anc = vec![false; model.ndof];
    let mut cur = model.frames[frame].anchor;
    while let Some(j) = cur {
        is_anc[j] = true;
        cur = model.parent[j];
    }

    let mut jac = DMatrix::<f64>::zeros(6, model.ndof);
    for i in 0..model.ndof {
        if !is_anc[i] {
            continue;
        }
        let z = jw[i].rotation() * model.axis[i]; // world joint axis
        let p_i = jw[i].translation_vec();
        let (lin, ang) = match model.kind[i] {
            JointKind::Revolute => (z.cross(&(p_e - p_i)), z), // [ z×(p_e−p_i) ; z ]
            JointKind::Prismatic => (z, Vector3::zeros()),     // [ z ; 0 ]
        };
        jac[(0, i)] = lin.x;
        jac[(1, i)] = lin.y;
        jac[(2, i)] = lin.z;
        jac[(3, i)] = ang.x;
        jac[(4, i)] = ang.y;
        jac[(5, i)] = ang.z;
    }

    match jframe {
        JacFrame::World => (ee, jac),
        JacFrame::Body => {
            let rt = ee.rotation().transpose();
            (ee, rotate_twist_rows(&jac, &rt))
        }
    }
}

/// Left-multiply each column's linear and angular 3-blocks by `rm`.
fn rotate_twist_rows(j: &DMatrix<f64>, rm: &Matrix3<f64>) -> DMatrix<f64> {
    let mut out = j.clone();
    for c in 0..j.ncols() {
        let lin = rm * Vector3::new(j[(0, c)], j[(1, c)], j[(2, c)]);
        let ang = rm * Vector3::new(j[(3, c)], j[(4, c)], j[(5, c)]);
        out[(0, c)] = lin.x;
        out[(1, c)] = lin.y;
        out[(2, c)] = lin.z;
        out[(3, c)] = ang.x;
        out[(4, c)] = ang.y;
        out[(5, c)] = ang.z;
    }
    out
}

/// How the smallest singular direction is lost at a singularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SingularityKind {
    None,
    Wrist,
    Elbow,
    Shoulder,
    Boundary,
}

/// Tolerances for singularity analysis + the governor (metric / rad robots).
#[derive(Clone, Copy, Debug)]
pub struct SingularityParams {
    /// Relative nullspace tolerance: `σ < eps_null·σ_max` ⇒ a null direction.
    pub eps_null: f64,
    /// `σ_min` below which the governor engages.
    pub eps_activate: f64,
    /// Maximum DLS damping `λ`.
    pub lambda_max: f64,
}
impl Default for SingularityParams {
    fn default() -> Self {
        Self {
            eps_null: 1e-6,
            eps_activate: 1e-2,
            lambda_max: 1e-1,
        }
    }
}

/// A typed, structured singularity report — Caliper's signature feature.
#[derive(Clone, Debug)]
pub struct SingularityReport {
    pub manipulability: f64,
    pub condition_number: f64,
    pub sigma_min: f64,
    pub kind: SingularityKind,
    pub offending_joints: Vec<usize>,
    /// `ndof × m` (m = number of near-zero singular values).
    pub nullspace_basis: DMatrix<f64>,
    /// `ndof`, the unit right-singular vector of `σ_min`.
    pub escape_direction: DVector<f64>,
    /// The three smallest singular values, ascending.
    pub sigma: [f64; 3],
}

/// A geometric Jacobian wrapper carrying the SVD-based singularity analysis.
pub struct Jacobian(pub DMatrix<f64>);

impl Jacobian {
    /// Singular values (descending). Empty for a 0-DOF Jacobian.
    pub fn singular_values(&self) -> DVector<f64> {
        if self.0.ncols() == 0 || self.0.nrows() == 0 {
            return DVector::zeros(0);
        }
        self.0.clone().svd(false, false).singular_values
    }
    /// Yoshikawa manipulability = product of singular values (0 for a 0-DOF Jacobian).
    pub fn manipulability(&self) -> f64 {
        if self.0.ncols() == 0 {
            return 0.0;
        }
        self.singular_values().iter().product()
    }

    /// The single SVD → the full [`SingularityReport`]. Returns a sentinel for a
    /// legal 0-DOF (fixed-only) robot instead of running an empty SVD.
    pub fn analyze(&self, p: &SingularityParams) -> SingularityReport {
        let j = &self.0;
        let n = j.ncols();
        if n == 0 || j.nrows() == 0 {
            return SingularityReport {
                manipulability: 0.0,
                condition_number: f64::INFINITY,
                sigma_min: 0.0,
                kind: SingularityKind::None,
                offending_joints: vec![],
                nullspace_basis: DMatrix::zeros(n, 0),
                escape_direction: DVector::zeros(n),
                sigma: [0.0; 3],
            };
        }
        let svd = j.clone().svd(true, true);
        let s = &svd.singular_values;
        let k = s.len();
        let u = svd.u.as_ref().expect("u");
        let vt = svd.v_t.as_ref().expect("v_t");

        let sigma_max = s[0];
        let sigma_min = s[k - 1];
        let manipulability: f64 = s.iter().product();
        let condition_number = if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };

        let mut sigma = [0.0; 3];
        for (i, slot) in sigma.iter_mut().enumerate() {
            *slot = if k > i { s[k - 1 - i] } else { 0.0 };
        }

        let tol = p.eps_null * sigma_max;
        let null_idx: Vec<usize> = (0..k).filter(|&i| s[i] < tol).collect();
        let mut nullspace_basis = DMatrix::<f64>::zeros(n, null_idx.len());
        for (c, &i) in null_idx.iter().enumerate() {
            let col = DVector::from_iterator(n, vt.row(i).iter().copied());
            nullspace_basis.set_column(c, &col);
        }

        let escape_direction = DVector::from_iterator(n, vt.row(k - 1).iter().copied());
        let u_min = Vector6::from_iterator(u.column(k - 1).iter().copied());
        let kind = classify(&u_min, sigma_min, p.eps_activate);

        let emax = escape_direction
            .iter()
            .fold(0.0_f64, |a, &x| a.max(x.abs()));
        let offending_joints = (0..n)
            .filter(|&i| emax > 0.0 && escape_direction[i].abs() > 0.5 * emax)
            .collect();

        SingularityReport {
            manipulability,
            condition_number,
            sigma_min,
            kind,
            offending_joints,
            nullspace_basis,
            escape_direction,
            sigma,
        }
    }
}

/// Classify the lost direction from the smallest left-singular vector
/// `u_min = [v(0..3); ω(3..6)]`. (Per-topology geometric tests can refine this.)
fn classify(u_min: &Vector6<f64>, sigma_min: f64, eps: f64) -> SingularityKind {
    if sigma_min >= eps {
        return SingularityKind::None;
    }
    let lin = Vector3::new(u_min[0], u_min[1], u_min[2]).norm();
    let ang = Vector3::new(u_min[3], u_min[4], u_min[5]).norm();
    if ang > 1.5 * lin {
        SingularityKind::Wrist // orientation DOF collapsed
    } else if lin > 1.5 * ang {
        SingularityKind::Boundary // translation DOF collapsed at a reach edge
    } else {
        SingularityKind::Elbow // mixed translation + orientation
    }
}

/// Wraps a solver's output to stay safe near singularities.
pub struct SingularityGovernor {
    pub params: SingularityParams,
}

impl SingularityGovernor {
    pub fn new(params: SingularityParams) -> Self {
        Self { params }
    }
    /// Smooth, C¹ damping `λ²` that ramps in as `σ_min` drops below `eps_activate`.
    pub fn damping_sq(&self, sigma_min: f64) -> f64 {
        let e = self.params.eps_activate;
        if sigma_min >= e {
            0.0
        } else {
            let r = sigma_min / e;
            self.params.lambda_max.powi(2) * (1.0 - r * r)
        }
    }
    /// Scale a commanded 6-twist near a singularity: attenuate components along
    /// ill-conditioned directions when *approaching*, let them escape when
    /// *leaving*. `u`,`s` come from the same SVD used for the report.
    pub fn scale_twist(
        &self,
        v_cmd: &Vector6<f64>,
        u: &DMatrix<f64>,
        s: &DVector<f64>,
        prev_sigma_min: f64,
    ) -> Vector6<f64> {
        let k = s.len();
        let lambda2 = self.damping_sq(s[k - 1]);
        let approaching = s[k - 1] < prev_sigma_min;
        let mut out = Vector6::zeros();
        for i in 0..k {
            let ui = Vector6::from_iterator(u.column(i).iter().copied());
            let c = ui.dot(v_cmd);
            let sg = s[i];
            let gain = if approaching {
                sg * sg / (sg * sg + lambda2)
            } else {
                1.0
            };
            out += (c * gain) * ui;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};
    use std::f64::consts::PI;
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

    #[test]
    fn fk_home_pose() {
        let m = toy();
        let p = fk_tip(&m, &[0.0, 0.0]).translation();
        assert!((p[0] - 0.2).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
        assert!((p[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fk_revolute_rotates_tip() {
        let m = toy();
        let p = fk_tip(&m, &[PI / 2.0, 0.0]).translation();
        assert!(p[0].abs() < 1e-12);
        assert!((p[1] - 0.2).abs() < 1e-12);
        assert!((p[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fk_frame_matches_fk_joints() {
        let m = toy();
        let q = [0.3, -0.4];
        let mut jw = vec![Se3::identity(); m.ndof];
        fk_joints(&m, &q, &mut jw);
        for f in 0..m.frames.len() {
            let a = fk_frame(&m, &q, f);
            let b = frame_pose(&m, &jw, f);
            assert!((a.0.to_homogeneous() - b.0.to_homogeneous()).norm() < 1e-12);
        }
    }

    fn rot_log(m: &Matrix3<f64>) -> Vector3<f64> {
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*m)).scaled_axis()
    }

    fn fd_jacobian(m: &Model, q: &[f64], frame: usize, h: f64) -> DMatrix<f64> {
        let mut jfd = DMatrix::<f64>::zeros(6, m.ndof);
        for i in 0..m.ndof {
            let mut qp = q.to_vec();
            let mut qm = q.to_vec();
            qp[i] += h;
            qm[i] -= h;
            let tp = fk_frame(m, &qp, frame);
            let tm = fk_frame(m, &qm, frame);
            let v = (tp.translation_vec() - tm.translation_vec()) / (2.0 * h);
            let w = rot_log(&(tp.rotation() * tm.rotation().transpose())) / (2.0 * h);
            jfd[(0, i)] = v.x;
            jfd[(1, i)] = v.y;
            jfd[(2, i)] = v.z;
            jfd[(3, i)] = w.x;
            jfd[(4, i)] = w.y;
            jfd[(5, i)] = w.z;
        }
        jfd
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let m = toy();
        let frame = m.tip_frame();
        for q in [[0.3, -0.4], [1.0, 0.5], [-0.7, 1.2]] {
            let (_, jac) = jacobian(&m, &q, frame, JacFrame::World);
            let jfd = fd_jacobian(&m, &q, frame, 1e-6);
            assert!(
                (&jac - &jfd).norm() < 1e-6,
                "analytic Jacobian vs finite-difference"
            );
        }
    }

    #[test]
    fn jacobian_body_world_consistent() {
        // J_body = blkdiag(Rᵀ,Rᵀ)·J_world  ⇒  rotating J_body by R recovers J_world
        let m = toy();
        let q = [0.6, -0.9];
        let f = m.tip_frame();
        let (ee, jw) = jacobian(&m, &q, f, JacFrame::World);
        let (_, jb) = jacobian(&m, &q, f, JacFrame::Body);
        let back = rotate_twist_rows(&jb, &ee.rotation());
        assert!((&back - &jw).norm() < 1e-12);
    }

    fn singular_synthetic() -> Jacobian {
        // 6×3 with column 2 == column 1 ⇒ rank 2 ⇒ a 1-D nullspace.
        let mut m = DMatrix::<f64>::zeros(6, 3);
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 1.0;
        m[(2, 2)] = 1.0;
        let c1 = m.column(1).into_owned();
        m.set_column(2, &c1);
        Jacobian(m)
    }

    #[test]
    fn analyze_svd_identities() {
        let m = toy();
        let (_, j) = jacobian(&m, &[0.4, 0.7], m.tip_frame(), JacFrame::World);
        let jac = Jacobian(j.clone());
        let rep = jac.analyze(&SingularityParams::default());
        assert!((rep.escape_direction.norm() - 1.0).abs() < 1e-9);
        // ‖J · v_min‖ == σ_min  (SVD identity)
        assert!(((&j * &rep.escape_direction).norm() - rep.sigma_min).abs() < 1e-9);
        assert!((rep.manipulability - jac.manipulability()).abs() < 1e-12);
        assert_eq!(rep.kind, SingularityKind::None); // generic config is well-conditioned
    }

    #[test]
    fn analyze_detects_rank_deficiency() {
        let jac = singular_synthetic();
        let rep = jac.analyze(&SingularityParams::default());
        assert!(rep.sigma_min < 1e-9);
        assert!(rep.condition_number > 1e10);
        assert_eq!(rep.nullspace_basis.ncols(), 1);
        assert_ne!(rep.kind, SingularityKind::None);
        let null = rep.nullspace_basis.column(0).into_owned();
        assert!((&jac.0 * &null).norm() < 1e-9); // J · nullspace ≈ 0
    }

    #[test]
    fn governor_damping_is_continuous_and_monotone() {
        let g = SingularityGovernor::new(SingularityParams::default());
        let e = g.params.eps_activate;
        assert_eq!(g.damping_sq(e + 1.0), 0.0); // off above threshold
        assert!(g.damping_sq(e - 1e-12) < 1e-6); // continuous at the boundary
        assert!(g.damping_sq(0.0) > 0.0); // engaged at the singularity
        assert!(g.damping_sq(e * 0.5) > g.damping_sq(e * 0.9)); // more damping as σ→0
    }

    #[test]
    fn prismatic_fk_and_jacobian() {
        let m = load("prismatic.urdf");
        let frame = m.tip_frame();
        let home = fk_tip(&m, &[0.0, 0.0]).translation();
        let moved = fk_tip(&m, &[0.0, 0.25]).translation();
        assert!((moved[0] - (home[0] + 0.25)).abs() < 1e-12);
        assert!((moved[1] - home[1]).abs() < 1e-12 && (moved[2] - home[2]).abs() < 1e-12);
        // analytic Jacobian (incl. the prismatic [z;0] column) vs finite-difference
        let (_, jac) = jacobian(&m, &[0.3, 0.2], frame, JacFrame::World);
        let jfd = fd_jacobian(&m, &[0.3, 0.2], frame, 1e-6);
        assert!((&jac - &jfd).norm() < 1e-6);
    }

    #[test]
    fn branched_ancestor_masking() {
        let m = load("branched.urdf");
        let q = vec![0.3; m.ndof];
        let fa = m.frame_id("tipA").unwrap();
        let fb = m.frame_id("tipB").unwrap();
        let (_, ja) = jacobian(&m, &q, fa, JacFrame::World);
        let (_, jb) = jacobian(&m, &q, fb, JacFrame::World);
        let col = |name: &str| m.joint_names.iter().position(|n| n == name).unwrap();
        // tipA does not depend on j3; tipB does not depend on j2
        assert!(ja.column(col("j3")).norm() < 1e-15);
        assert!(jb.column(col("j2")).norm() < 1e-15);
        // but each DOES depend on its own branch joint
        assert!(ja.column(col("j2")).norm() > 1e-6);
        assert!(jb.column(col("j3")).norm() > 1e-6);
    }

    #[test]
    fn zero_dof_analyze_does_not_panic() {
        let m = load("fixed_only.urdf");
        assert_eq!(m.ndof, 0);
        let (_, j) = jacobian(&m, &[], m.tip_frame(), JacFrame::World);
        let rep = Jacobian(j).analyze(&SingularityParams::default());
        assert_eq!(rep.kind, SingularityKind::None);
        assert!((fk_tip(&m, &[]).translation()[2] - 0.3).abs() < 1e-12);
    }
}
