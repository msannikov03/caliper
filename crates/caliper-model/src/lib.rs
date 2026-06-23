//! Robot model: parse URDF, compile to a frozen struct-of-arrays kinematic model.
//!
//! The editable URDF tree is compiled once into a [`Model`]: movable joints in
//! topological order (parent index < own index), fixed joints folded away, and
//! every link exposed as a queryable/renderable frame. Algorithms (FK, Jacobian,
//! IK) are free functions over `(&Model, &[f64])`.
use caliper_spatial::{Se3, SpatialInertia};
use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};
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
    #[error("link `{0}` is unreachable from the root (cycle or disconnected subtree)")]
    Disconnected(String),
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
    /// URDF velocity limit per dof (rad/s revolute | m/s prismatic). `None` when
    /// the joint is continuous or the URDF omitted a positive velocity.
    pub vel_limit: Vec<Option<f64>>,
    /// URDF effort limit per dof. Parsed for later dynamics; UNUSED in Phase 3.
    pub effort_limit: Vec<Option<f64>>,
    pub frames: Vec<LinkFrame>,
    pub frame_index: HashMap<String, usize>,
    /// Per-movable-link spatial inertia (len == ndof), expressed in that joint's
    /// OWN frame at q=0, with all fixed-welded descendant links folded in. Zero
    /// for any movable link whose `<inertial>` (or a folded descendant's) was absent.
    pub inertia: Vec<SpatialInertia>,
    /// True iff every movable link AND every fixed link folded onto a movable
    /// parent carried a real `<inertial>` (mass>0). Dynamics entry points gate on this.
    pub has_inertia: bool,
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
    vel: Option<f64>,
    effort: Option<f64>,
}

#[derive(Default)]
struct RobotTree {
    name: String,
    links: Vec<String>,
    /// Parallel to `links`: parsed `<inertial>` (None when absent).
    link_inertia: Vec<Option<SpatialInertia>>,
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
            t.link_inertia.push(parse_inertial(&l.inertial));
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
                    (RawKind::Revolute, valid_limit(j.limit.lower, j.limit.upper))
                }
                urdf_rs::JointType::Continuous => (RawKind::Revolute, None),
                urdf_rs::JointType::Prismatic => (
                    RawKind::Prismatic,
                    valid_limit(j.limit.lower, j.limit.upper),
                ),
                urdf_rs::JointType::Fixed => (RawKind::Fixed, None),
                other => return Err(CompileError::UnsupportedJoint(format!("{other:?}"))),
            };
            // Velocity/effort: 0.0 ⇒ absent (mirrors valid_limit's lower==upper guard).
            // A present <limit> missing velocity= hard-errors in urdf-rs (no per-field
            // default) → surfaces as CompileError::Parse, which is correct.
            let vel = (j.limit.velocity > 0.0).then_some(j.limit.velocity);
            let effort = (j.limit.effort > 0.0).then_some(j.limit.effort);
            let a = j.axis.xyz.0;
            t.joints.push(EditJoint {
                name: j.name.clone(),
                kind,
                parent,
                child,
                origin: pose_to_se3(&j.origin),
                axis: Vector3::new(a[0], a[1], a[2]),
                limits,
                vel,
                effort,
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

/// Parse a urdf-rs `<inertial>` into a link-frame `SpatialInertia`, or `None` when
/// the block was absent. urdf-rs defaults a MISSING `<inertial>` to mass=0 + zero
/// tensor (`Link.inertial` is non-Option, `#[serde(default)]`) — so `mass<=0` means
/// "absent", which is exactly what gates `has_inertia`.
fn parse_inertial(inr: &urdf_rs::Inertial) -> Option<SpatialInertia> {
    let m = inr.mass.value;
    if !(m.is_finite() && m > 0.0) {
        return None; // missing, NaN, or non-physical mass → treat as absent
    }
    let [cx, cy, cz] = inr.origin.xyz.0;
    let [rr, rp, ry] = inr.origin.rpy.0;
    let com = Vector3::new(cx, cy, cz);
    let i = &inr.inertia;
    // tensor about the COM, in the COM-rpy-rotated axes
    let i_com = Matrix3::new(
        i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz,
    );
    // rotate into LINK axes: Ic = Rc · Icom · Rcᵀ
    let rc = UnitQuaternion::from_euler_angles(rr, rp, ry).to_rotation_matrix();
    let i_link = rc.into_inner() * i_com * rc.into_inner().transpose();
    Some(SpatialInertia::from_mass_com_inertia(m, com, i_link))
}

/// A URDF limit is meaningful only when `lower < upper`; otherwise treat the
/// joint as unbounded (a missing limit serializes as `0,0`).
fn valid_limit(lo: f64, hi: f64) -> Option<(f64, f64)> {
    (lo < hi).then_some((lo, hi))
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
        vel_limit: vec![],
        effort_limit: vec![],
        frames: vec![],
        frame_index: HashMap::new(),
        inertia: vec![],
        has_inertia: false,
    };
    // per link: (nearest movable-joint ancestor, accumulated fixed offset from it)
    let mut anchor: Vec<(Option<usize>, Se3)> = vec![(None, Se3::identity()); nlinks];
    // per-MOVABLE-joint composite spatial inertia in that joint's frame, summed over
    // the movable link itself + every fixed-welded descendant (folded). Grows with ndof.
    let mut inertia_accum: Vec<SpatialInertia> = Vec::new();
    let mut full_inertia = true; // false if any contributing link lacked <inertial>
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
                    // fold this fixed link's inertia onto its movable anchor (if any);
                    // anc == None means a fixed chain off the world → dropped (fixed-base).
                    if let Some(ai) = anc {
                        match t.link_inertia[j.child] {
                            Some(g) => {
                                inertia_accum[ai] = inertia_accum[ai].add(&g.transform(&placement))
                            }
                            None => full_inertia = false,
                        }
                    }
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
                    m.vel_limit.push(j.vel);
                    m.effort_limit.push(j.effort);
                    anchor[j.child] = (Some(mi), Se3::identity());
                    register_frame(&mut m, &t.links[j.child], Some(mi), Se3::identity());
                    // seed this movable joint's composite with its own child link
                    // (placement is identity in the joint's own frame).
                    match t.link_inertia[j.child] {
                        Some(g) => inertia_accum.push(g),
                        None => {
                            inertia_accum.push(SpatialInertia::zero());
                            full_inertia = false;
                        }
                    }
                    m.ndof += 1;
                }
            }
            stack.push(j.child);
        }
    }
    // Every link must have been reached: otherwise a cycle or a disconnected
    // subtree silently truncated the model.
    if m.frames.len() != nlinks {
        let orphan = (0..nlinks)
            .find(|&l| !m.frame_index.contains_key(&t.links[l]))
            .map(|l| t.links[l].clone())
            .unwrap_or_default();
        return Err(CompileError::Disconnected(orphan));
    }
    m.inertia = inertia_accum;
    m.has_inertia = full_inertia && m.ndof > 0;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!(
            "{}/../../oracle/fixtures/robots/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        )
    }
    fn load(name: &str) -> Model {
        Model::from_urdf(Path::new(&fixture(name))).unwrap()
    }
    fn toy() -> Model {
        load("toy.urdf")
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
    fn parses_velocity_limits() {
        assert_eq!(toy().vel_limit, vec![Some(3.0), Some(3.0)]);
        assert_eq!(load("redundant7.urdf").vel_limit, vec![Some(1.0); 7]);
        let p = load("prismatic.urdf");
        assert_eq!(p.vel_limit, vec![Some(3.0), Some(1.0)]);
        assert_eq!(p.effort_limit, vec![Some(10.0), Some(10.0)]);
    }

    #[test]
    fn inertia_does_not_change_kinematics() {
        // adding <inertial> to showcase6 leaves every kinematic array intact.
        let m = load("showcase6.urdf");
        assert_eq!(m.ndof, 6);
        assert_eq!(m.joint_names, vec!["j1", "j2", "j3", "j4", "j5", "j6"]);
        assert_eq!(
            m.parent,
            vec![None, Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
        assert!(
            m.has_inertia,
            "showcase6 must carry inertial data after Phase 4"
        );
        assert_eq!(m.inertia.len(), 6);
        for g in &m.inertia {
            assert!(g.mass() > 0.0);
        }
    }

    #[test]
    fn bare_fixture_reports_no_inertia() {
        let m = load("toy.urdf");
        assert!(!m.has_inertia);
        assert_eq!(m.inertia.len(), m.ndof);
        for g in &m.inertia {
            assert_eq!(g.mass(), 0.0);
        }
    }

    #[test]
    fn fold_conserves_mass() {
        // dyn_welded folds l2 (m=0.5) onto j1's composite with l1 (m=1.0) → 1.5.
        let m = load("dyn_welded.urdf");
        assert_eq!(m.ndof, 1);
        assert!(m.has_inertia);
        assert!((m.inertia[0].mass() - 1.5).abs() < 1e-12, "folded mass");
    }

    #[test]
    fn clamp_respects_limits() {
        let m = toy();
        let mut q = vec![10.0, -10.0];
        m.clamp(&mut q);
        assert!(q[0] <= 3.1401 && q[1] >= -3.1401);
    }

    #[test]
    fn prismatic_compiles() {
        let m = load("prismatic.urdf");
        assert_eq!(m.ndof, 2);
        assert_eq!(m.kind, vec![JointKind::Revolute, JointKind::Prismatic]);
        assert_eq!(m.limits[1], Some((0.0, 0.5)));
    }

    #[test]
    fn branched_compiles_with_folded_fixed() {
        let m = load("branched.urdf");
        assert_eq!(m.ndof, 3);
        for n in ["j1", "j2", "j3"] {
            assert!(m.joint_names.iter().any(|x| x == n), "missing {n}");
        }
        // the fixed joint f1 becomes a queryable frame, not a dof
        assert!(m.frame_id("fixmid").is_some());
        assert!(m.frame_id("tipA").is_some() && m.frame_id("tipB").is_some());
    }

    #[test]
    fn fixed_only_is_zero_dof() {
        let m = load("fixed_only.urdf");
        assert_eq!(m.ndof, 0);
        assert_eq!(m.frame_name(m.tip_frame()), "tip");
    }

    #[test]
    fn redundant7_compiles() {
        let m = load("redundant7.urdf");
        assert_eq!(m.ndof, 7);
        // topological invariant: parent index strictly less than own index
        for (i, p) in m.parent.iter().enumerate() {
            if let Some(p) = p {
                assert!(*p < i);
            }
        }
    }

    #[test]
    fn rejects_disconnected() {
        let err = Model::from_urdf(Path::new(&fixture("disconnected.urdf"))).unwrap_err();
        assert!(matches!(err, CompileError::Disconnected(_)), "got {err:?}");
    }
}
