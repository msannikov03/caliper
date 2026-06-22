//! Forward kinematics, Jacobians, and singularity analysis.
use caliper_model::{JointKind, Model};
use caliper_spatial::{Se3, exp_prismatic, exp_revolute};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

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

/// A geometric Jacobian wrapper carrying the SVD-based analysis used by the
/// singularity stack (built in the next step).
pub struct Jacobian(pub DMatrix<f64>);

impl Jacobian {
    /// Singular values of the Jacobian (descending).
    pub fn singular_values(&self) -> DVector<f64> {
        self.0.clone().svd(false, false).singular_values
    }
    /// Yoshikawa manipulability = product of singular values.
    pub fn manipulability(&self) -> f64 {
        self.singular_values().iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};
    use std::f64::consts::PI;
    use std::path::Path;

    fn toy() -> Model {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/fixtures/robots/toy.urdf"
        );
        Model::from_urdf(Path::new(p)).unwrap()
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
}
