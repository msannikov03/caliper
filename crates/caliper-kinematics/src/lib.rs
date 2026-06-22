//! Forward kinematics, Jacobians, and singularity analysis.
use caliper_model::{JointKind, Model};
use caliper_spatial::{Se3, exp_prismatic, exp_revolute};
use nalgebra::{DMatrix, DVector};

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

/// A geometric Jacobian (6 × ndof). (Analytic construction lands in the next step;
/// this carries the SVD-based analysis helpers used by the singularity stack.)
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
        // q=0: base→l1 (+0.1 z) then l1→l2 (+0.2 x) ⇒ tip at (0.2, 0, 0.1)
        let m = toy();
        let p = fk_tip(&m, &[0.0, 0.0]).translation();
        assert!((p[0] - 0.2).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
        assert!((p[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fk_revolute_rotates_tip() {
        // j1 is z-revolute at (0,0,0.1); rotating +90° sends the +0.2x tip offset to +0.2y
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
}
