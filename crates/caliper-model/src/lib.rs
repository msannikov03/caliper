//! Robot model: parse URDF, compile to a frozen struct-of-arrays kinematic model.
//!
//! The editable URDF tree is compiled once into a [`Model`]: movable joints in
//! topological order (parent index < own index), fixed joints folded away, and
//! every link exposed as a queryable/renderable frame. Algorithms (FK, Jacobian,
//! IK) are free functions over `(&Model, &[f64])`.
use caliper_spatial::Se3;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::path::Path;

/// Movable joint type (Phase 1). `Fixed` joints are folded out at compile time;
/// `Continuous` is treated as `Revolute` without limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointKind {
    Revolute,
    Prismatic,
}

#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("urdf parse: {0}")]
    Parse(String),
    #[error("joint references unknown link `{0}`")]
    DanglingLink(String),
    #[error("unsupported joint type `{0}` (Phase 1: revolute/continuous/prismatic/fixed)")]
    UnsupportedJoint(String),
    #[error("link `{0}` has multiple parent joints (not a tree)")]
    MultiParent(String),
    #[error("no root link found")]
    NoRoot,
    #[error("multiple root links (forest not supported)")]
    MultiRoot,
    #[error("joint `{0}` has a zero-length axis")]
    ZeroAxis(String),
}

/// A renderable / queryable link frame, hung off its nearest movable joint.
#[derive(Clone, Debug)]
pub struct LinkFrame {
    pub name: String,
    /// Movable joint this frame rides on (`None` = root / world).
    pub anchor: Option<usize>,
    /// Anchor-joint frame → this link frame (the folded fixed chain).
    pub offset: Se3,
}

/// Frozen, struct-of-arrays kinematic model. Movable joints are in topological
/// order (`parent[i] < i`).
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub ndof: usize,
    pub joint_names: Vec<String>,
    pub kind: Vec<JointKind>,
    pub parent: Vec<Option<usize>>,
    /// Parent-movable frame → this joint frame (fixed, baked at compile time).
    pub parent_to_joint: Vec<Se3>,
    /// Normalized, joint-local axis.
    pub axis: Vec<Vector3<f64>>,
    pub limits: Vec<Option<(f64, f64)>>,
    pub frames: Vec<LinkFrame>,
    pub frame_index: HashMap<String, usize>,
}

impl Model {
    pub fn from_urdf(path: &Path) -> Result<Self, CompileError> {
        compile(&RobotTree::from_urdf(path)?)
    }
    pub fn frame_id(&self, name: &str) -> Option<usize> {
        self.frame_index.get(name).copied()
    }
    /// The last-registered link frame — a reasonable default tool/tip frame.
    pub fn tip_frame(&self) -> usize {
        self.frames.len() - 1
    }
    pub fn frame_name(&self, f: usize) -> &str {
        &self.frames[f].name
    }
    /// Clamp a configuration to joint limits, in place.
    pub fn clamp(&self, q: &mut [f64]) {
        for (i, lim) in self.limits.iter().enumerate() {
            if let Some((lo, hi)) = lim {
                q[i] = q[i].clamp(*lo, *hi);
            }
        }
    }
}

/// Back-compatible facade used by the faces (CLI / Python / Studio).
#[derive(Clone, Debug)]
pub struct Robot {
    pub name: String,
    pub joint_names: Vec<String>,
    pub model: Model,
}

impl Robot {
    pub fn from_urdf(path: &Path) -> Result<Self, CompileError> {
        let model = Model::from_urdf(path)?;
        Ok(Robot {
            name: model.name.clone(),
            joint_names: model.joint_names.clone(),
            model,
        })
    }
    pub fn ndof(&self) -> usize {
        self.model.ndof
    }
}

// ===== editable tree (mirror of URDF, index-addressed) =====

#[derive(Clone, Copy, Debug)]
enum RawKind {
    Revolute,
    Prismatic,
    Fixed,
}

struct EditJoint {
    name: String,
    kind: RawKind,
    parent: usize,
    child: usize,
    origin: Se3,
    axis: Vector3<f64>,
    limits: Option<(f64, f64)>,
}

#[derive(Default)]
struct RobotTree {
    name: String,
    links: Vec<String>,
    joints: Vec<EditJoint>,
    link_index: HashMap<String, usize>,
}

impl RobotTree {
    fn from_urdf(path: &Path) -> Result<Self, CompileError> {
        let u = urdf_rs::read_file(path).map_err(|e| CompileError::Parse(e.to_string()))?;
        let mut t = RobotTree {
            name: u.name.clone(),
            ..Default::default()
        };
        for l in &u.links {
            t.link_index.insert(l.name.clone(), t.links.len());
            t.links.push(l.name.clone());
        }
        for j in &u.joints {
            let parent = *t
                .link_index
                .get(&j.parent.link)
                .ok_or_else(|| CompileError::DanglingLink(j.parent.link.clone()))?;
            let child = *t
                .link_index
                .get(&j.child.link)
                .ok_or_else(|| CompileError::DanglingLink(j.child.link.clone()))?;
            let (kind, limits) = match &j.joint_type {
                urdf_rs::JointType::Revolute => {
                    (RawKind::Revolute, Some((j.limit.lower, j.limit.upper)))
                }
                urdf_rs::JointType::Continuous => (RawKind::Revolute, None),
                urdf_rs::JointType::Prismatic => {
                    (RawKind::Prismatic, Some((j.limit.lower, j.limit.upper)))
                }
                urdf_rs::JointType::Fixed => (RawKind::Fixed, None),
                other => return Err(CompileError::UnsupportedJoint(format!("{other:?}"))),
            };
            let a = j.axis.xyz.0;
            t.joints.push(EditJoint {
                name: j.name.clone(),
                kind,
                parent,
                child,
                origin: pose_to_se3(&j.origin),
                axis: Vector3::new(a[0], a[1], a[2]),
                limits,
            });
        }
        Ok(t)
    }
}

/// URDF `<origin>` (fixed-axis rpy) → SE(3). `from_euler_angles(r,p,y)` = Rz(y)Ry(p)Rx(r) = URDF rpy.
fn pose_to_se3(p: &urdf_rs::Pose) -> Se3 {
    let [x, y, z] = p.xyz.0;
    let [r, pi, ya] = p.rpy.0;
    Se3(Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_euler_angles(r, pi, ya),
    ))
}

fn normalize_axis(a: &Vector3<f64>) -> Option<Vector3<f64>> {
    let n = a.norm();
    (n > 1e-12).then(|| a / n)
}

fn register_frame(m: &mut Model, name: &str, anchor: Option<usize>, offset: Se3) {
    m.frame_index.insert(name.to_string(), m.frames.len());
    m.frames.push(LinkFrame {
        name: name.to_string(),
        anchor,
        offset,
    });
}

/// Compile the editable tree into the frozen [`Model`]: single-root check,
/// topological DFS (parent before child), fold fixed joints, assign dof indices.
fn compile(t: &RobotTree) -> Result<Model, CompileError> {
    let nlinks = t.links.len();
    let mut incoming: Vec<Option<usize>> = vec![None; nlinks];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nlinks];
    for (ji, j) in t.joints.iter().enumerate() {
        if incoming[j.child].is_some() {
            return Err(CompileError::MultiParent(t.links[j.child].clone()));
        }
        incoming[j.child] = Some(ji);
        children[j.parent].push(ji);
    }
    let roots: Vec<usize> = (0..nlinks).filter(|&l| incoming[l].is_none()).collect();
    let root = match roots.as_slice() {
        [r] => *r,
        [] => return Err(CompileError::NoRoot),
        _ => return Err(CompileError::MultiRoot),
    };

    let mut m = Model {
        name: t.name.clone(),
        ndof: 0,
        joint_names: vec![],
        kind: vec![],
        parent: vec![],
        parent_to_joint: vec![],
        axis: vec![],
        limits: vec![],
        frames: vec![],
        frame_index: HashMap::new(),
    };
    // per link: (nearest movable-joint ancestor, accumulated fixed offset from it)
    let mut anchor: Vec<(Option<usize>, Se3)> = vec![(None, Se3::identity()); nlinks];
    register_frame(&mut m, &t.links[root], None, Se3::identity());

    let mut stack = vec![root];
    while let Some(link) = stack.pop() {
        let (anc, fold) = anchor[link];
        for &ji in &children[link] {
            let j = &t.joints[ji];
            let placement = fold.compose(&j.origin); // anchor frame → joint frame
            match j.kind {
                RawKind::Fixed => {
                    anchor[j.child] = (anc, placement);
                    register_frame(&mut m, &t.links[j.child], anc, placement);
                }
                RawKind::Revolute | RawKind::Prismatic => {
                    let mi = m.ndof;
                    let axis = normalize_axis(&j.axis)
                        .ok_or_else(|| CompileError::ZeroAxis(j.name.clone()))?;
                    let kind = match j.kind {
                        RawKind::Prismatic => JointKind::Prismatic,
                        _ => JointKind::Revolute,
                    };
                    m.joint_names.push(j.name.clone());
                    m.kind.push(kind);
                    m.parent.push(anc);
                    m.parent_to_joint.push(placement);
                    m.axis.push(axis);
                    m.limits.push(j.limits);
                    anchor[j.child] = (Some(mi), Se3::identity());
                    register_frame(&mut m, &t.links[j.child], Some(mi), Se3::identity());
                    m.ndof += 1;
                }
            }
            stack.push(j.child);
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Model {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/fixtures/robots/toy.urdf"
        );
        Model::from_urdf(Path::new(p)).unwrap()
    }

    #[test]
    fn compiles_toy() {
        let m = toy();
        assert_eq!(m.ndof, 2);
        assert_eq!(m.joint_names, vec!["j1", "j2"]);
        assert_eq!(m.parent, vec![None, Some(0)]);
        assert_eq!(m.kind, vec![JointKind::Revolute, JointKind::Revolute]);
        assert!((m.axis[0] - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        assert!((m.axis[1] - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
        assert_eq!(m.parent_to_joint[0].translation(), [0.0, 0.0, 0.1]);
        // every link is a frame; the tip is l2
        assert_eq!(m.frame_name(m.tip_frame()), "l2");
        assert!(m.frame_id("base").is_some());
    }

    #[test]
    fn clamp_respects_limits() {
        let m = toy();
        let mut q = vec![10.0, -10.0];
        m.clamp(&mut q);
        assert!(q[0] <= 3.1401 && q[1] >= -3.1401);
    }
}
