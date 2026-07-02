//! Robot model: parse URDF, compile to a frozen struct-of-arrays kinematic model.
//!
//! The editable URDF tree is compiled once into a [`Model`]: movable joints in
//! topological order (parent index < own index), fixed joints folded away, and
//! every link exposed as a queryable/renderable frame. Algorithms (FK, Jacobian,
//! IK) are free functions over `(&Model, &[f64])`.
use caliper_spatial::{Se3, SpatialInertia};
use nalgebra::{Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub mod hull;
pub mod stl;

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
    #[error("joint `{joint}` mimics unknown movable joint `{src_joint}`")]
    MimicUnknownSource { joint: String, src_joint: String },
    #[error(
        "joint `{joint}` mimics `{src_joint}`, which is itself a mimic (mimic chains not supported)"
    )]
    MimicOfMimic { joint: String, src_joint: String },
    #[error("mimic on joint `{0}` is only supported for revolute/continuous/prismatic joints")]
    MimicUnsupportedJoint(String),
}

/// A URDF `<mimic>` constraint on a movable joint: `q_this = multiplier * q[source] + offset`.
/// The mimicking joint remains a FULL-SPACE dof (see [`Model::mimic`]); this struct only
/// records the constraint so reduced-space helpers/wrappers can enforce it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MimicInfo {
    /// Full-space movable-joint index of the DRIVING joint (never itself a mimic).
    pub source: usize,
    /// URDF `multiplier` (default `1.0`).
    pub multiplier: f64,
    /// URDF `offset` (default `0.0`).
    pub offset: f64,
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

/// A collision shape, in the shape's own local frame. Box/sphere/cylinder/capsule
/// are exact primitives; a `<mesh>` collision is loaded (STL), scaled, and reduced
/// to a [`CollisionShape::ConvexHull`] of its vertices (see `parse_collisions`).
/// Not `Copy` because `ConvexHull` owns a `Vec`; clone explicitly.
#[derive(Clone, Debug, PartialEq)]
pub enum CollisionShape {
    /// Axis-aligned box; `half` are the half-extents (URDF `size`/2).
    Box {
        half: Vector3<f64>,
    },
    Sphere {
        radius: f64,
    },
    /// Z-aligned cylinder (URDF convention).
    Cylinder {
        radius: f64,
        length: f64,
    },
    /// Z-aligned capsule (URDF convention): a cylinder of `length` (the core
    /// segment, from `-length/2` to `+length/2` along local Z) capped by a
    /// hemisphere of `radius` at each end. The total tip-to-tip extent is
    /// `length + 2*radius`. Checked as a swept sphere (segment ⊕ sphere).
    Capsule {
        radius: f64,
        length: f64,
    },
    /// Convex hull of a `<mesh>`'s (scaled) vertices, in the link-local frame.
    /// `points` are the hull vertices; a collider checks against their convex
    /// hull (GJK), which is exact for the support-function test.
    ConvexHull {
        points: Vec<Point3<f64>>,
    },
}

/// A collision primitive attached to a link [`LinkFrame`]. Its world pose is
/// `fk_frame(model, q, frame) · origin` — the frame already carries any folded
/// fixed-chain offset, so collision needs no separate fold.
#[derive(Clone, Debug)]
pub struct CollisionGeom {
    pub frame: usize,
    pub origin: Se3,
    pub shape: CollisionShape,
}

/// A RENDER-ONLY visual shape, in the shape's own local frame (mirrors
/// [`CollisionShape`] for primitives). Unlike a collision `<mesh>` (which is
/// loaded and hulled), a visual `<mesh>` is NOT loaded here — only its filename
/// is resolved to an absolute on-disk path so a renderer (Studio) can load it
/// itself. `path: None` means unresolvable: the visual is still KEPT so the
/// renderer can fall back to a procedural placeholder.
#[derive(Clone, Debug, PartialEq)]
pub enum VisualShape {
    /// `half` are the half-extents (URDF `size`/2), like [`CollisionShape::Box`].
    Box {
        half: Vector3<f64>,
    },
    Sphere {
        radius: f64,
    },
    /// Z-aligned (URDF convention).
    Cylinder {
        radius: f64,
        length: f64,
    },
    /// Z-aligned (URDF convention); `length` is the core segment, tip-to-tip
    /// extent is `length + 2*radius` (see [`CollisionShape::Capsule`]).
    Capsule {
        radius: f64,
        length: f64,
    },
    Mesh {
        /// RESOLVED absolute path (existed at parse time); `None` if the raw
        /// filename could not be resolved (see [`resolve_mesh_path`]).
        path: Option<PathBuf>,
        /// The raw URDF `filename` attribute, verbatim, for diagnostics/fallback.
        raw: String,
        /// Per-axis mesh scale (URDF `scale`, default `[1, 1, 1]`).
        scale: [f64; 3],
    },
}

/// A render-only visual attached to a link [`LinkFrame`]. Its world pose is
/// `fk_frame(model, q, frame) · origin`, exactly like [`CollisionGeom`] — the
/// frame already carries any folded fixed-chain offset.
#[derive(Clone, Debug)]
pub struct VisualGeom {
    pub frame: usize,
    pub origin: Se3,
    pub shape: VisualShape,
    /// RGBA in `[0, 1]`: the visual's inline `<material><color>` if present,
    /// else the color of the named top-level `<material>` it references, else `None`.
    pub color: Option<[f32; 4]>,
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
    /// Parsed `<collision>` geometry — box/sphere/cylinder/capsule primitives plus
    /// mesh colliders loaded as a [`CollisionShape::ConvexHull`] — each attached to
    /// a link frame. ⚠ A `<mesh>` that could not be loaded
    /// (missing/`package://`/non-STL file, unparsable, degenerate) is still
    /// SKIPPED: that link carries NO collider for the dropped part and is NOT
    /// collision-checked there. Callers should surface the uncovered count
    /// (`caliper_collision::CollisionModel::uncovered_frames`). Empty when absent.
    pub collision: Vec<CollisionGeom>,
    /// Frame indices of link frames that had a `<collision>` whose geometry was
    /// DROPPED (an unloadable mesh). Such a frame may carry other
    /// colliders too, so it is only PARTIALLY covered — a query can still report
    /// "clear" for its dropped part. Callers must treat these as not-fully-covered
    /// (see `caliper_collision::CollisionModel::uncovered_frames`).
    pub dropped_collider_frames: Vec<usize>,
    /// Parsed `<visual>` geometry — RENDER-ONLY, never consulted by kinematics,
    /// collision, or dynamics. Primitives carry exact dims; a `<mesh>` carries a
    /// resolved absolute path (or `None` when unresolvable — the visual is KEPT
    /// so a renderer can fall back to a procedural shape). Visual parsing can
    /// never fail `compile()`. Empty when the URDF has no `<visual>` elements.
    pub visuals: Vec<VisualGeom>,
    /// Per-movable-joint URDF `<mimic>` constraint (len == ndof; `None` = independent).
    ///
    /// ⚠ Mimic joints stay in the full-space arrays: `ndof`, `q`, and EVERY existing
    /// API remain FULL-SPACE, exactly as without mimics. A `Some(MimicInfo)` entry only
    /// RECORDS the constraint `q[i] = m * q[source] + b`; nothing here enforces it.
    /// Callers that want the constraint enforced work in the REDUCED space via
    /// [`Model::expand_mimic`] / [`Model::reduce_config`] and the `*_reduced` wrappers
    /// in `caliper-kinematics`. Mimic chains (a mimic whose source is a mimic) are
    /// rejected at compile time, so one expansion pass is always sufficient.
    pub mimic: Vec<Option<MimicInfo>>,
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

    // ===== mimic-joint helpers (reduced <-> full configuration mapping) =====
    //
    // The full space is UNCHANGED by mimics: every API on `Model` and every
    // algorithm crate keeps taking `q.len() == ndof`. The reduced space contains
    // only the independent joints, in full-space order; these helpers map between
    // the two so reduced-space wrappers can be built on the validated full-space math.

    /// True iff any movable joint carries a `<mimic>` constraint.
    pub fn has_mimic(&self) -> bool {
        self.mimic.iter().any(|m| m.is_some())
    }

    /// Number of INDEPENDENT dofs (`ndof` minus the mimic joints).
    pub fn ndof_independent(&self) -> usize {
        self.mimic.iter().filter(|m| m.is_none()).count()
    }

    /// Full-space indices of the independent joints, in full-space order. The k-th
    /// entry is the full-space dof that reduced coordinate k drives.
    pub fn independent_dofs(&self) -> Vec<usize> {
        (0..self.ndof)
            .filter(|&i| self.mimic[i].is_none())
            .collect()
    }

    /// Expand a reduced configuration (`q_red.len() == ndof_independent()`, ordered
    /// as [`Model::independent_dofs`]) to the full space: independents are placed
    /// by order, then every mimic is filled as `m * q_full[source] + b`. Sources are
    /// never mimics themselves (chains are rejected at compile), so one pass suffices.
    pub fn expand_mimic(&self, q_red: &[f64]) -> Vec<f64> {
        debug_assert_eq!(q_red.len(), self.ndof_independent());
        let mut q = vec![0.0; self.ndof];
        let mut k = 0;
        for (qi, mi) in q.iter_mut().zip(&self.mimic) {
            if mi.is_none() {
                *qi = q_red[k];
                k += 1;
            }
        }
        for i in 0..self.ndof {
            if let Some(mi) = &self.mimic[i] {
                q[i] = mi.multiplier * q[mi.source] + mi.offset;
            }
        }
        q
    }

    /// Project a full configuration onto the independent dofs (drops the mimic
    /// entries; it does NOT check that they satisfied their constraints).
    pub fn reduce_config(&self, q_full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(q_full.len(), self.ndof);
        (0..self.ndof)
            .filter(|&i| self.mimic[i].is_none())
            .map(|i| q_full[i])
            .collect()
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
    /// Raw `<mimic>`: (source joint NAME, multiplier, offset); defaults m=1, b=0.
    mimic: Option<(String, f64, f64)>,
}

#[derive(Default)]
struct RobotTree {
    name: String,
    links: Vec<String>,
    /// Parallel to `links`: parsed `<inertial>` (None when absent).
    link_inertia: Vec<Option<SpatialInertia>>,
    /// Parallel to `links`: parsed `<collision>` primitives (link-frame local).
    link_collision: Vec<Vec<(Se3, CollisionShape)>>,
    /// Parallel to `links`: true if the link had a `<collision>` whose geometry
    /// was DROPPED (an unloadable mesh) — that link is only partially covered.
    link_dropped: Vec<bool>,
    /// Parallel to `links`: parsed render-only `<visual>` shapes
    /// `(link-frame-local origin, shape, rgba)`.
    link_visual: Vec<Vec<ParsedVisual>>,
    joints: Vec<EditJoint>,
    link_index: HashMap<String, usize>,
}

impl RobotTree {
    fn from_urdf(path: &Path) -> Result<Self, CompileError> {
        let u = urdf_rs::read_file(path).map_err(|e| CompileError::Parse(e.to_string()))?;
        // Mesh `<collision filename=...>` is resolved relative to the URDF's directory.
        let base_dir = path.parent();
        let mut t = RobotTree {
            name: u.name.clone(),
            ..Default::default()
        };
        for l in &u.links {
            t.link_index.insert(l.name.clone(), t.links.len());
            t.links.push(l.name.clone());
            t.link_inertia.push(parse_inertial(&l.inertial));
            let (geoms, dropped) = parse_collisions(l, base_dir);
            t.link_collision.push(geoms);
            t.link_dropped.push(dropped);
            t.link_visual.push(parse_visuals(l, &u.materials, base_dir));
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
            // <mimic>: only meaningful on a 1-dof movable joint (urdf-rs defaults:
            // multiplier=1, offset=0 when the attributes are omitted).
            let mimic = match &j.mimic {
                Some(mm) => {
                    if matches!(kind, RawKind::Fixed) {
                        return Err(CompileError::MimicUnsupportedJoint(j.name.clone()));
                    }
                    Some((
                        mm.joint.clone(),
                        mm.multiplier.unwrap_or(1.0),
                        mm.offset.unwrap_or(0.0),
                    ))
                }
                None => None,
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
                vel,
                effort,
                mimic,
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

/// Parse a link's `<collision>` primitives into `(link-local origin, shape)`.
/// Box/sphere/cylinder/capsule are exact; a `<mesh>` is loaded (STL), scaled, and
/// reduced to a [`CollisionShape::ConvexHull`]. A mesh that cannot be loaded
/// (missing file, unsupported/`package://` path, unparsable STL, degenerate hull)
/// is SKIPPED (no collider; `dropped=true`) — the pre-existing, safe-by-omission
/// behavior (see the `Model::collision` note).
fn parse_collisions(
    link: &urdf_rs::Link,
    base_dir: Option<&Path>,
) -> (Vec<(Se3, CollisionShape)>, bool) {
    let mut out = Vec::new();
    let mut dropped = false;
    for c in &link.collision {
        let origin = pose_to_se3(&c.origin);
        let shape = match &c.geometry {
            urdf_rs::Geometry::Box { size } => {
                let [x, y, z] = size.0;
                CollisionShape::Box {
                    half: Vector3::new(x / 2.0, y / 2.0, z / 2.0),
                }
            }
            urdf_rs::Geometry::Sphere { radius } => CollisionShape::Sphere { radius: *radius },
            urdf_rs::Geometry::Cylinder { radius, length } => CollisionShape::Cylinder {
                radius: *radius,
                length: *length,
            },
            urdf_rs::Geometry::Capsule { radius, length } => CollisionShape::Capsule {
                radius: *radius,
                length: *length,
            },
            urdf_rs::Geometry::Mesh { filename, scale } => {
                match load_mesh_hull(filename, scale.as_ref(), base_dir) {
                    Some(points) => CollisionShape::ConvexHull { points },
                    None => {
                        dropped = true; // unloadable mesh → keep DROPPED (safe)
                        continue;
                    }
                }
            }
        };
        out.push((origin, shape));
    }
    (out, dropped)
}

/// Parse a link's `<visual>` list into `(link-local origin, shape, rgba)` —
/// RENDER-ONLY, infallible (an unresolvable mesh is KEPT with `path: None`,
/// see [`VisualShape::Mesh`]). Color: the visual's inline material color wins;
/// else a named material is resolved against the robot's top-level `materials`;
/// else `None`.
/// A parsed `<visual>` before frame attachment: `(link-local origin, shape, rgba)`.
type ParsedVisual = (Se3, VisualShape, Option<[f32; 4]>);

fn parse_visuals(
    link: &urdf_rs::Link,
    materials: &[urdf_rs::Material],
    base_dir: Option<&Path>,
) -> Vec<ParsedVisual> {
    link.visual
        .iter()
        .map(|v| {
            let origin = pose_to_se3(&v.origin);
            let shape = match &v.geometry {
                urdf_rs::Geometry::Box { size } => {
                    let [x, y, z] = size.0;
                    VisualShape::Box {
                        half: Vector3::new(x / 2.0, y / 2.0, z / 2.0),
                    }
                }
                urdf_rs::Geometry::Sphere { radius } => VisualShape::Sphere { radius: *radius },
                urdf_rs::Geometry::Cylinder { radius, length } => VisualShape::Cylinder {
                    radius: *radius,
                    length: *length,
                },
                urdf_rs::Geometry::Capsule { radius, length } => VisualShape::Capsule {
                    radius: *radius,
                    length: *length,
                },
                urdf_rs::Geometry::Mesh { filename, scale } => VisualShape::Mesh {
                    path: resolve_mesh_path(filename, base_dir),
                    raw: filename.clone(),
                    scale: scale.as_ref().map(|s| s.0).unwrap_or([1.0; 3]),
                },
            };
            (
                origin,
                shape,
                resolve_visual_color(v.material.as_ref(), materials),
            )
        })
        .collect()
}

/// The rgba for a visual's material: inline `<color>` wins; else the referenced
/// top-level named `<material>`'s color; else `None`.
fn resolve_visual_color(
    mat: Option<&urdf_rs::Material>,
    named: &[urdf_rs::Material],
) -> Option<[f32; 4]> {
    let m = mat?;
    let color = m.color.as_ref().or_else(|| {
        named
            .iter()
            .find(|n| n.name == m.name)
            .and_then(|n| n.color.as_ref())
    })?;
    let [r, g, b, a] = color.rgba.0;
    Some([r as f32, g as f32, b as f32, a as f32])
}

/// Resolve a URDF `<mesh filename=...>` to an absolute on-disk path, or `None`.
/// Reusable for both visual and (future) collision meshes. Search order:
/// - `file://` prefix is stripped first;
/// - a plain relative path: `urdf_dir/<raw>` if it exists;
/// - an absolute path: itself, if it exists;
/// - `package://<pkg>/<rest>`: `urdf_dir/<rest>`; then for `urdf_dir` and each
///   of its ancestors `A` up to 6 levels: `A/<pkg>/<rest>` and `A/<rest>`; then
///   for each root `R` in `CALIPER_PACKAGE_PATH` (colon-separated):
///   `R/<pkg>/<rest>` and `R/<rest>`. First existing file wins.
pub(crate) fn resolve_mesh_path(raw: &str, urdf_dir: Option<&Path>) -> Option<PathBuf> {
    let name = raw.strip_prefix("file://").unwrap_or(raw);
    if let Some(pkg_rest) = name.strip_prefix("package://") {
        let (pkg, rest) = pkg_rest.split_once('/')?;
        if pkg.is_empty() || rest.is_empty() {
            return None;
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(dir) = urdf_dir {
            candidates.push(dir.join(rest));
            // ancestors() yields `dir` itself first, then up to 6 parent levels.
            for a in dir.ancestors().take(7) {
                candidates.push(a.join(pkg).join(rest));
                candidates.push(a.join(rest));
            }
        }
        if let Ok(roots) = std::env::var("CALIPER_PACKAGE_PATH") {
            for r in roots.split(':').filter(|s| !s.is_empty()) {
                candidates.push(Path::new(r).join(pkg).join(rest));
                candidates.push(Path::new(r).join(rest));
            }
        }
        return candidates.into_iter().find(|c| c.is_file()).map(absolutize);
    }
    let p = Path::new(name);
    let cand = if p.is_absolute() {
        p.to_path_buf()
    } else {
        urdf_dir?.join(p)
    };
    cand.is_file().then(|| absolutize(cand))
}

/// Best-effort absolute form of an existing path (canonicalize, falling back to
/// the path itself — it was just checked to exist, so this normally succeeds).
fn absolutize(p: PathBuf) -> PathBuf {
    std::fs::canonicalize(&p).unwrap_or(p)
}

/// Resolve a `<mesh>` filename, load its STL, apply the URDF per-axis `scale`,
/// and reduce to a convex hull of vertices. `None` on any failure (the caller
/// then keeps the mesh dropped). Only plain/relative/`file://` paths are
/// resolved; `package://` is unsupported here (no package map) → `None`.
fn load_mesh_hull(
    filename: &str,
    scale: Option<&urdf_rs::Vec3>,
    base_dir: Option<&Path>,
) -> Option<Vec<Point3<f64>>> {
    let name = filename.strip_prefix("file://").unwrap_or(filename);
    if name.starts_with("package://") {
        return None; // not resolvable without a package map
    }
    let path = {
        let p = Path::new(name);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir?.join(p)
        }
    };
    // Only STL is supported by the pure-Rust loader.
    let is_stl = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("stl"))
        .unwrap_or(false);
    if !is_stl {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    let mut verts = stl::parse_stl(&bytes)?;
    if let Some(s) = scale {
        let [sx, sy, sz] = s.0;
        for v in &mut verts {
            v.coords.component_mul_assign(&Vector3::new(sx, sy, sz));
        }
    }
    let hull = hull::convex_hull(&verts);
    // need at least a triangle to be a meaningful collider
    (hull.len() >= 3).then_some(hull)
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
        collision: vec![],
        dropped_collider_frames: vec![],
        visuals: vec![],
        mimic: vec![],
    };
    // per-MOVABLE-joint raw mimic spec (source NAME, m, b) gathered during the DFS;
    // resolved to full-space indices once all movable joints are numbered.
    let mut mimic_raw: Vec<Option<(String, f64, f64)>> = Vec::new();
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
                    mimic_raw.push(j.mimic.clone());
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
    // Resolve mimic sources to full-space movable-joint indices. Chains (a mimic
    // whose source is itself a mimic, including self-mimic) are rejected so that a
    // single expand_mimic pass is always correct.
    let jname_index: HashMap<&str, usize> = m
        .joint_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    m.mimic = mimic_raw
        .iter()
        .enumerate()
        .map(|(i, raw)| match raw {
            None => Ok(None),
            Some((src, mult, off)) => {
                let source = *jname_index.get(src.as_str()).ok_or_else(|| {
                    CompileError::MimicUnknownSource {
                        joint: m.joint_names[i].clone(),
                        src_joint: src.clone(),
                    }
                })?;
                if mimic_raw[source].is_some() {
                    return Err(CompileError::MimicOfMimic {
                        joint: m.joint_names[i].clone(),
                        src_joint: src.clone(),
                    });
                }
                Ok(Some(MimicInfo {
                    source,
                    multiplier: *mult,
                    offset: *off,
                }))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Attach parsed collisions to their link frames (every link is a frame, so the
    // frame's offset already encodes any folded fixed chain — no separate fold).
    for (li, geoms) in t.link_collision.iter().enumerate() {
        if let Some(&frame) = m.frame_index.get(&t.links[li]) {
            for (origin, shape) in geoms {
                m.collision.push(CollisionGeom {
                    frame,
                    origin: *origin,
                    shape: shape.clone(),
                });
            }
            if t.link_dropped.get(li).copied().unwrap_or(false) {
                m.dropped_collider_frames.push(frame);
            }
        }
    }
    // Attach render-only visuals to their link frames, exactly like collision.
    for (li, vis) in t.link_visual.iter().enumerate() {
        if let Some(&frame) = m.frame_index.get(&t.links[li]) {
            for (origin, shape, color) in vis {
                m.visuals.push(VisualGeom {
                    frame,
                    origin: *origin,
                    shape: shape.clone(),
                    color: *color,
                });
            }
        }
    }
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
    fn parses_box_collisions() {
        let m = load("collide_arm.urdf");
        assert_eq!(m.collision.len(), 3, "three box colliders");
        for g in &m.collision {
            match &g.shape {
                CollisionShape::Box { half } => {
                    assert!((half.x - 0.06).abs() < 1e-12); // size 0.12 → half 0.06
                    assert!((half.z - 0.15).abs() < 1e-12); // size 0.3  → half 0.15
                }
                other => panic!("expected box, got {other:?}"),
            }
            // origin is the collision <origin> (z=0.15), in the link frame
            assert!((g.origin.translation()[2] - 0.15).abs() < 1e-12);
        }
    }

    #[test]
    fn parses_shapes_and_skips_mesh() {
        let m = load("collide_shapes.urdf");
        // sphere + cylinder parsed; mesh skipped → 2 colliders
        assert_eq!(m.collision.len(), 2);
        let mut saw_sphere = false;
        let mut saw_cyl = false;
        for g in &m.collision {
            match &g.shape {
                CollisionShape::Sphere { radius } => {
                    saw_sphere = true;
                    assert!((radius - 0.1).abs() < 1e-12);
                }
                CollisionShape::Cylinder { radius, length } => {
                    saw_cyl = true;
                    assert!((radius - 0.05).abs() < 1e-12 && (length - 0.2).abs() < 1e-12);
                    // cylinder rides the fixed-welded l2 frame (folded z=0.5 offset)
                    let f = &m.frames[g.frame];
                    assert!(
                        (f.offset.translation()[2] - 0.5).abs() < 1e-12,
                        "folded offset"
                    );
                }
                other => panic!("only sphere + cylinder here, got {other:?}"),
            }
        }
        assert!(saw_sphere && saw_cyl);
    }

    #[test]
    fn no_collision_geometry_is_empty() {
        assert!(load("toy.urdf").collision.is_empty());
    }

    #[test]
    fn parses_capsule_collision_not_dropped() {
        // <capsule radius=0.1 length=0.4> on l1 must now parse into a
        // CollisionShape::Capsule (NOT be dropped), closing the silent-drop gap.
        let m = load("collide_capsule.urdf");
        assert_eq!(m.collision.len(), 1, "the capsule became one collider");
        assert!(
            m.dropped_collider_frames.is_empty(),
            "a parsed capsule must NOT be reported as dropped"
        );
        match &m.collision[0].shape {
            CollisionShape::Capsule { radius, length } => {
                assert!((radius - 0.1).abs() < 1e-12);
                assert!((length - 0.4).abs() < 1e-12);
            }
            other => panic!("expected Capsule, got {other:?}"),
        }
        assert_eq!(m.collision[0].frame, m.frame_id("l1").unwrap());
    }

    #[test]
    fn loads_mesh_as_convex_hull() {
        // unit_cube.stl → ConvexHull of its 8 corners; the link is now COVERED
        // (NOT dropped), closing the silent-drop gap for mesh colliders.
        let m = load("collide_mesh.urdf");
        assert_eq!(m.collision.len(), 1, "the mesh became one collider");
        assert!(
            m.dropped_collider_frames.is_empty(),
            "a loaded mesh must NOT be reported as dropped"
        );
        match &m.collision[0].shape {
            CollisionShape::ConvexHull { points } => {
                assert_eq!(points.len(), 8, "unit cube hull = 8 corners");
                for p in points {
                    // every corner sits at +/-0.5 on all three axes
                    assert!(p.coords.iter().all(|c| (c.abs() - 0.5).abs() < 1e-9));
                }
            }
            other => panic!("expected ConvexHull, got {other:?}"),
        }
        // the mesh collider rides the l1 frame
        assert_eq!(m.collision[0].frame, m.frame_id("l1").unwrap());
    }

    #[test]
    fn mesh_scale_is_applied_before_hull() {
        let m = load("collide_mesh_scaled.urdf");
        match &m.collision[0].shape {
            CollisionShape::ConvexHull { points } => {
                assert_eq!(points.len(), 8);
                // scale 2 → corners at +/-1.0 on all three axes
                for p in points {
                    assert!(p.coords.iter().all(|c| (c.abs() - 1.0).abs() < 1e-9));
                }
            }
            other => panic!("expected ConvexHull, got {other:?}"),
        }
    }

    #[test]
    fn missing_mesh_file_stays_dropped() {
        // collide_shapes.urdf references a NON-existent hand.stl → must remain
        // DROPPED (safe-by-omission), preserving the pre-mesh-loader behavior.
        let m = load("collide_shapes.urdf");
        assert_eq!(
            m.collision.len(),
            2,
            "sphere + cylinder only; mesh unloadable"
        );
        assert!(
            m.collision
                .iter()
                .all(|g| !matches!(g.shape, CollisionShape::ConvexHull { .. })),
            "no convex hull from a missing file"
        );
        assert_eq!(
            m.dropped_collider_frames.len(),
            1,
            "the unloadable mesh frame is still tracked as dropped"
        );
    }

    #[test]
    fn rejects_disconnected() {
        let err = Model::from_urdf(Path::new(&fixture("disconnected.urdf"))).unwrap_err();
        assert!(matches!(err, CompileError::Disconnected(_)), "got {err:?}");
    }

    // ===== render-only <visual> parsing =====

    fn visuals_on<'a>(m: &'a Model, link: &str) -> Vec<&'a VisualGeom> {
        let f = m.frame_id(link).unwrap();
        m.visuals.iter().filter(|v| v.frame == f).collect()
    }

    #[test]
    fn parses_visual_primitives_with_colors() {
        let m = load("visual_arm.urdf");
        assert_eq!(m.ndof, 2, "visuals must not change kinematics");
        assert_eq!(m.visuals.len(), 6, "box + cylinder + sphere + 3 meshes");

        // base: box, inline material rgba, origin z=0.05
        let base = visuals_on(&m, "base");
        assert_eq!(base.len(), 1);
        match &base[0].shape {
            VisualShape::Box { half } => {
                assert!((half.x - 0.1).abs() < 1e-12); // size 0.2 → half 0.1
                assert!((half.z - 0.05).abs() < 1e-12); // size 0.1 → half 0.05
            }
            other => panic!("expected box, got {other:?}"),
        }
        assert!((base[0].origin.translation()[2] - 0.05).abs() < 1e-12);
        assert_eq!(base[0].color, Some([0.9, 0.1, 0.1, 1.0]), "inline rgba");

        // l1: cylinder with NAMED top-level material, plus a bare sphere
        let l1 = visuals_on(&m, "l1");
        assert_eq!(l1.len(), 2);
        match &l1[0].shape {
            VisualShape::Cylinder { radius, length } => {
                assert!((radius - 0.04).abs() < 1e-12 && (length - 0.3).abs() < 1e-12);
            }
            other => panic!("expected cylinder, got {other:?}"),
        }
        assert_eq!(
            l1[0].color,
            Some([0.2, 0.4, 0.8, 1.0]),
            "named material `steel` resolved against Robot.materials"
        );
        match &l1[1].shape {
            VisualShape::Sphere { radius } => assert!((radius - 0.05).abs() < 1e-12),
            other => panic!("expected sphere, got {other:?}"),
        }
        assert!((l1[1].origin.translation()[2] - 0.3).abs() < 1e-12);
        assert_eq!(l1[1].color, None, "no material → no color");
    }

    #[test]
    fn visual_mesh_paths_resolve() {
        let m = load("visual_arm.urdf");
        let l2 = visuals_on(&m, "l2");
        assert_eq!(l2.len(), 3, "all three meshes KEPT, resolvable or not");

        // relative path → absolute existing file, with scale
        match &l2[0].shape {
            VisualShape::Mesh { path, raw, scale } => {
                let p = path.as_ref().expect("relative mesh must resolve");
                assert!(p.is_absolute() && p.is_file(), "resolved: {p:?}");
                assert!(p.ends_with("visual_hand.stl"));
                assert_eq!(raw, "visual_hand.stl");
                assert_eq!(*scale, [2.0, 2.0, 2.0]);
            }
            other => panic!("expected mesh, got {other:?}"),
        }
        // package://demo_pkg/... → resolved via the urdf-dir ancestor search
        match &l2[1].shape {
            VisualShape::Mesh { path, raw, scale } => {
                let p = path.as_ref().expect("package mesh must resolve");
                assert!(p.is_absolute() && p.is_file(), "resolved: {p:?}");
                assert!(p.ends_with("demo_pkg/meshes/part.stl"), "got {p:?}");
                assert_eq!(raw, "package://demo_pkg/meshes/part.stl");
                assert_eq!(*scale, [1.0; 3], "default scale");
            }
            other => panic!("expected mesh, got {other:?}"),
        }
        // unresolvable → KEPT with path None; compile already succeeded above
        match &l2[2].shape {
            VisualShape::Mesh { path, raw, .. } => {
                assert!(path.is_none(), "ghost mesh must not resolve");
                assert_eq!(raw, "ghost_missing.stl");
            }
            other => panic!("expected mesh, got {other:?}"),
        }
    }

    #[test]
    fn package_path_env_resolution() {
        // With NO urdf dir, package:// can only resolve through CALIPER_PACKAGE_PATH.
        let raw = "package://demo_pkg/meshes/part.stl";
        // SAFETY: see below — this test is the only writer of this variable.
        unsafe { std::env::remove_var("CALIPER_PACKAGE_PATH") }; // hermetic start
        assert_eq!(resolve_mesh_path(raw, None), None, "no dir, no env → None");
        let fixtures = format!("{}/../../oracle/fixtures", env!("CARGO_MANIFEST_DIR"));
        // SAFETY: single-threaded mutation scoped to this test; the var is only
        // a FALLBACK for other tests (their meshes resolve earlier or are
        // non-package paths), so a concurrent read cannot change their result.
        unsafe { std::env::set_var("CALIPER_PACKAGE_PATH", &fixtures) };
        let p = resolve_mesh_path(raw, None);
        unsafe { std::env::remove_var("CALIPER_PACKAGE_PATH") };
        let p = p.expect("env root must resolve R/<pkg>/<rest>");
        assert!(p.is_absolute() && p.is_file(), "resolved: {p:?}");
        assert!(p.ends_with("demo_pkg/meshes/part.stl"));
    }

    // ===== mimic joints =====

    /// Compile a URDF from a literal string via a scratch temp file (bad-input
    /// cases don't warrant committed fixtures).
    fn compile_str(tag: &str, urdf: &str) -> Result<Model, CompileError> {
        let dir = std::env::temp_dir().join("caliper_mimic_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("{tag}_{}.urdf", std::process::id()));
        std::fs::write(&p, urdf).unwrap();
        let r = Model::from_urdf(&p);
        let _ = std::fs::remove_file(&p);
        r
    }

    #[test]
    fn parses_mimic_constraints() {
        let m = load("gripper_mimic.urdf");
        // full space is untouched: 4 movable joints, all arrays len 4
        assert_eq!(m.ndof, 4);
        assert_eq!(m.joint_names, vec!["arm", "wrist", "finger1", "finger2"]);
        assert_eq!(m.mimic.len(), 4);
        assert!(m.has_mimic());
        assert_eq!(m.mimic[0], None);
        assert_eq!(
            m.mimic[1],
            Some(MimicInfo {
                source: 0,
                multiplier: 0.5,
                offset: 0.1
            }),
            "wrist mimics arm"
        );
        assert_eq!(m.mimic[2], None);
        assert_eq!(
            m.mimic[3],
            Some(MimicInfo {
                source: 2,
                multiplier: -1.0,
                offset: 0.0
            }),
            "finger2 mimics finger1; omitted offset defaults to 0"
        );
        assert_eq!(m.ndof_independent(), 2);
        assert_eq!(m.independent_dofs(), vec![0, 2]);
    }

    #[test]
    fn mimic_defaults_multiplier_one_offset_zero() {
        let m = compile_str(
            "defaults",
            r#"<robot name="d">
                 <link name="a"/><link name="b"/><link name="c"/>
                 <joint name="j1" type="revolute">
                   <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
                   <limit lower="-1" upper="1" effort="1" velocity="1"/>
                 </joint>
                 <joint name="j2" type="revolute">
                   <parent link="b"/><child link="c"/><axis xyz="0 0 1"/>
                   <limit lower="-1" upper="1" effort="1" velocity="1"/>
                   <mimic joint="j1"/>
                 </joint>
               </robot>"#,
        )
        .unwrap();
        assert_eq!(
            m.mimic[1],
            Some(MimicInfo {
                source: 0,
                multiplier: 1.0,
                offset: 0.0
            })
        );
    }

    #[test]
    fn expand_reduce_round_trip() {
        let m = load("gripper_mimic.urdf");
        let q_red = [0.7, 0.03];
        let q_full = m.expand_mimic(&q_red);
        assert_eq!(q_full.len(), 4);
        assert_eq!(q_full[0], 0.7);
        assert!(
            (q_full[1] - (0.5 * 0.7 + 0.1)).abs() < 1e-15,
            "wrist = 0.5*arm + 0.1"
        );
        assert_eq!(q_full[2], 0.03);
        assert!((q_full[3] - (-0.03)).abs() < 1e-15, "finger2 = -finger1");
        assert_eq!(m.reduce_config(&q_full), q_red.to_vec());
    }

    #[test]
    fn no_mimic_helpers_degrade_to_identity() {
        let m = load("showcase6.urdf");
        assert!(!m.has_mimic());
        assert!(m.mimic.iter().all(|x| x.is_none()));
        assert_eq!(m.mimic.len(), m.ndof);
        assert_eq!(m.ndof_independent(), m.ndof);
        assert_eq!(m.independent_dofs(), (0..m.ndof).collect::<Vec<_>>());
        let q = [0.3, -0.4, 0.6, 0.2, -0.5, 0.1];
        assert_eq!(m.expand_mimic(&q), q.to_vec());
        assert_eq!(m.reduce_config(&q), q.to_vec());
    }

    #[test]
    fn rejects_mimic_of_mimic_and_unknown_source() {
        let base = |mimic2: &str| {
            format!(
                r#"<robot name="bad">
                     <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
                     <joint name="j1" type="revolute">
                       <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
                       <limit lower="-1" upper="1" effort="1" velocity="1"/>
                     </joint>
                     <joint name="j2" type="revolute">
                       <parent link="b"/><child link="c"/><axis xyz="0 0 1"/>
                       <limit lower="-1" upper="1" effort="1" velocity="1"/>
                       <mimic joint="j1"/>
                     </joint>
                     <joint name="j3" type="revolute">
                       <parent link="c"/><child link="d"/><axis xyz="0 0 1"/>
                       <limit lower="-1" upper="1" effort="1" velocity="1"/>
                       <mimic joint="{mimic2}"/>
                     </joint>
                   </robot>"#
            )
        };
        let err = compile_str("chain", &base("j2")).unwrap_err();
        assert!(
            matches!(&err, CompileError::MimicOfMimic { joint, src_joint }
                if joint == "j3" && src_joint == "j2"),
            "got {err:?}"
        );
        let err = compile_str("unknown", &base("ghost")).unwrap_err();
        assert!(
            matches!(&err, CompileError::MimicUnknownSource { joint, src_joint }
                if joint == "j3" && src_joint == "ghost"),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_mimic_on_fixed_joint() {
        let err = compile_str(
            "fixedmimic",
            r#"<robot name="bad">
                 <link name="a"/><link name="b"/><link name="c"/>
                 <joint name="j1" type="revolute">
                   <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
                   <limit lower="-1" upper="1" effort="1" velocity="1"/>
                 </joint>
                 <joint name="jf" type="fixed">
                   <parent link="b"/><child link="c"/>
                   <mimic joint="j1"/>
                 </joint>
               </robot>"#,
        )
        .unwrap_err();
        assert!(
            matches!(&err, CompileError::MimicUnsupportedJoint(j) if j == "jf"),
            "got {err:?}"
        );
    }

    #[test]
    fn existing_fixtures_visuals_and_kinematics_unchanged() {
        // fixtures without <visual> stay empty; kinematics untouched
        for name in ["toy.urdf", "collide_arm.urdf", "collide_shapes.urdf"] {
            assert!(load(name).visuals.is_empty(), "{name} has no <visual>");
        }
        let toy = load("toy.urdf");
        assert_eq!(toy.ndof, 2);
        assert_eq!(toy.joint_names, vec!["j1", "j2"]);
        // showcase6 carries 7 render-only visuals; ndof/joints unchanged
        let m = load("showcase6.urdf");
        assert_eq!(m.ndof, 6);
        assert_eq!(m.visuals.len(), 7);
        assert!(m.visuals.iter().all(|v| v.color.is_none()));
        assert!(m.visuals.iter().all(|v| v.frame < m.frames.len()));
    }
}
