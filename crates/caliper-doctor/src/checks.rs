//! The doctor's checks (codes A001–A014) over the lenient URDF DOM.
//!
//! Every check emits [`Finding`]s with a stable code, a plain-English message
//! that names the offending field and value AND states the consequence, and —
//! where a mechanical fix exists — `auto_fixable = true` pointing at the
//! matching repair. Heuristic checks say so with a literal `[heuristic]`
//! marker in the message. See the crate docs for the full check table.

use crate::resolve::{fmt_tried, resolve_mesh};
use crate::xml::Element;
use crate::{Finding, codes};
use caliper_spatial::Se3;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

/// Hull input cap — mirrors `caliper_model::hull::MAX_HULL_INPUT` (private
/// there): collision-mesh clouds above this are subsampled before hulling.
pub const HULL_VERT_CAP: usize = 1024;

// ===== lenient URDF view =====

/// Links and joints of a `<robot>` element, by name, plus the root link
/// (the one that is never any joint's `<child>`).
pub struct View<'a> {
    pub links: Vec<(String, &'a Element)>,
    pub joints: Vec<(String, &'a Element)>,
    pub root_link: Option<String>,
}

impl<'a> View<'a> {
    pub fn new(robot: &'a Element) -> Self {
        let mut links = Vec::new();
        let mut joints = Vec::new();
        for e in robot.elements() {
            match (e.name.as_str(), e.attr("name")) {
                ("link", Some(n)) => links.push((n.to_string(), e)),
                ("joint", Some(n)) => joints.push((n.to_string(), e)),
                _ => {}
            }
        }
        let children: HashSet<&str> = joints
            .iter()
            .filter_map(|(_, j)| j.child("child").and_then(|c| c.attr("link")))
            .collect();
        let root_link = links
            .iter()
            .map(|(n, _)| n)
            .find(|n| !children.contains(n.as_str()))
            .cloned();
        View {
            links,
            joints,
            root_link,
        }
    }
}

pub fn parse_f(el: &Element, key: &str) -> Option<f64> {
    el.attr(key)?.trim().parse::<f64>().ok()
}

pub fn parse_vec3(s: &str) -> Option<[f64; 3]> {
    let mut it = s.split_whitespace().map(str::parse::<f64>);
    let out = [it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?];
    it.next().is_none().then_some(out)
}

/// `<origin xyz rpy>` → SE(3), URDF fixed-axis rpy convention (missing element
/// or attributes default to zero, like urdf-rs).
pub fn parse_origin(origin: Option<&Element>) -> Se3 {
    let xyz = origin
        .and_then(|o| o.attr("xyz"))
        .and_then(parse_vec3)
        .unwrap_or([0.0; 3]);
    let rpy = origin
        .and_then(|o| o.attr("rpy"))
        .and_then(parse_vec3)
        .unwrap_or([0.0; 3]);
    Se3::from_parts(
        Vector3::new(xyz[0], xyz[1], xyz[2]),
        UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2]),
    )
}

/// A geometry shape as written in the URDF (dimensions must be positive and
/// finite, otherwise the shape does not parse).
pub enum Shape {
    Box { half: Vector3<f64> },
    Sphere { radius: f64 },
    Cylinder { radius: f64, length: f64 },
    Capsule { radius: f64, length: f64 },
    Mesh { raw: String, scale: [f64; 3] },
}

/// Parse the first shape under a `<geometry>` element.
pub fn parse_shape(geometry: &Element) -> Option<Shape> {
    let e = geometry.elements().next()?;
    let pos = |v: f64| (v.is_finite() && v > 0.0).then_some(v);
    match e.name.as_str() {
        "box" => {
            let s = parse_vec3(e.attr("size")?)?;
            let half = Vector3::new(pos(s[0])? / 2.0, pos(s[1])? / 2.0, pos(s[2])? / 2.0);
            Some(Shape::Box { half })
        }
        "sphere" => Some(Shape::Sphere {
            radius: pos(parse_f(e, "radius")?)?,
        }),
        "cylinder" => Some(Shape::Cylinder {
            radius: pos(parse_f(e, "radius")?)?,
            length: pos(parse_f(e, "length")?)?,
        }),
        "capsule" => Some(Shape::Capsule {
            radius: pos(parse_f(e, "radius")?)?,
            length: pos(parse_f(e, "length")?)?,
        }),
        "mesh" => Some(Shape::Mesh {
            raw: e.attr("filename")?.to_string(),
            scale: e.attr("scale").and_then(parse_vec3).unwrap_or([1.0; 3]),
        }),
        _ => None,
    }
}

/// A shape plus its `<origin>` within the link frame.
pub struct PlacedShape {
    pub origin: Se3,
    pub shape: Shape,
}

/// Parse every `<collision>` or `<visual>` (`tag`) of a link into placed
/// shapes, skipping unparseable ones.
pub fn placed_shapes(link: &Element, tag: &str) -> Vec<PlacedShape> {
    link.children_named(tag)
        .filter_map(|c| {
            let shape = parse_shape(c.child("geometry")?)?;
            Some(PlacedShape {
                origin: parse_origin(c.child("origin")),
                shape,
            })
        })
        .collect()
}

fn has_repair_geometry(link: &Element) -> bool {
    !placed_shapes(link, "collision").is_empty() || !placed_shapes(link, "visual").is_empty()
}

/// How a link's `<inertial>` reads, mirroring `caliper_model::parse_inertial`:
/// a missing block and a non-positive/non-finite mass are BOTH "absent".
pub enum InertialStatus {
    Missing,
    /// `<inertial>` present but its mass is non-positive or non-finite.
    ZeroMass(f64),
    Present {
        tensor: Matrix3<f64>,
    },
}

pub fn inertial_status(link: &Element) -> InertialStatus {
    let Some(inr) = link.child("inertial") else {
        return InertialStatus::Missing;
    };
    let mass = inr
        .child("mass")
        .and_then(|m| parse_f(m, "value"))
        .unwrap_or(0.0);
    if !(mass.is_finite() && mass > 0.0) {
        return InertialStatus::ZeroMass(mass);
    }
    let g = |k: &str| {
        inr.child("inertia")
            .and_then(|e| parse_f(e, k))
            .unwrap_or(0.0)
    };
    let (ixx, iyy, izz) = (g("ixx"), g("iyy"), g("izz"));
    let (ixy, ixz, iyz) = (g("ixy"), g("ixz"), g("iyz"));
    InertialStatus::Present {
        tensor: Matrix3::new(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz),
    }
}

// ===== the checks =====

/// Run every DOM-level check (A013 lives in `load`, next to xacro handling).
pub fn run(robot: &Element, dir: Option<&Path>) -> Vec<Finding> {
    let v = View::new(robot);
    let mut out = Vec::new();
    check_inertials(&v, &mut out);
    check_meshes(&v, dir, &mut out);
    check_visual_coverage(&v, &mut out);
    check_joints(&v, &mut out);
    check_mimics(&v, &mut out);
    out
}

/// A001 (missing/zero inertial), A002 (implausible tensor), A010 (zero-mass root).
fn check_inertials(v: &View, out: &mut Vec<Finding>) {
    let statuses: Vec<(&String, &Element, InertialStatus)> = v
        .links
        .iter()
        .map(|(n, l)| (n, *l, inertial_status(l)))
        .collect();
    let n_with = statuses
        .iter()
        .filter(|(_, _, s)| matches!(s, InertialStatus::Present { .. }))
        .count();
    for (name, link, status) in &statuses {
        let is_root = Some(*name) == v.root_link.as_ref();
        match status {
            InertialStatus::Present { tensor } => check_tensor(name, tensor, out),
            InertialStatus::Missing | InertialStatus::ZeroMass(_) => {
                let what = match status {
                    InertialStatus::ZeroMass(m) => {
                        format!("an <inertial> with mass {m} (treated as absent)")
                    }
                    _ => "no <inertial>".to_string(),
                };
                if is_root {
                    if n_with > 0 {
                        out.push(
                            Finding::info(
                                codes::CAD_ZERO_MASS_ROOT,
                                format!(
                                    "[heuristic] root link `{name}` has {what} while {n_with} \
                                     other link(s) carry mass — the signature of \
                                     onshape-to-robot and similar CAD exporters. Harmless for \
                                     a fixed base (caliper folds the root into the world), but \
                                     a floating-base export sees a massless base"
                                ),
                            )
                            .hint(
                                "if the base is a real body (mobile robots, free-floating \
                                 sims), give it an <inertial> in CAD",
                            ),
                        );
                    }
                } else {
                    let mut f = Finding::error(
                        codes::MISSING_INERTIAL,
                        format!(
                            "link `{name}` has {what}: caliper compiles it as massless, which \
                             disables dynamics for the WHOLE model (has_inertia = false) and \
                             makes MJCF export fail outright"
                        ),
                    )
                    .hint(
                        "repair with compute_inertials derives mass/COM/inertia from the \
                         link's collision (or visual) geometry at a given density \
                         (default 1000 kg/m³)",
                    );
                    if has_repair_geometry(link) {
                        f = f.auto();
                    }
                    out.push(f);
                }
            }
        }
    }
}

/// A002: the tensor of a massive link must be finite, positive semi-definite,
/// and satisfy the rigid-body triangle inequality on its PRINCIPAL moments
/// (eigenvalues — the raw diagonal test is only valid in principal axes).
fn check_tensor(link: &str, tensor: &Matrix3<f64>, out: &mut Vec<Finding>) {
    let hint = "re-export with the full tensor — common CAD converters drop the off-diagonal \
                terms or mix up units, either of which produces a non-physical tensor; repair \
                never overwrites explicit inertia numbers";
    if tensor.iter().any(|x| !x.is_finite()) {
        out.push(
            Finding::error(
                codes::IMPLAUSIBLE_INERTIA,
                format!(
                    "link `{link}` <inertia> contains a non-finite entry: nothing downstream \
                     (dynamics, MuJoCo) can consume it"
                ),
            )
            .hint(hint),
        );
        return;
    }
    let mut lam: [f64; 3] = nalgebra::linalg::SymmetricEigen::new(*tensor)
        .eigenvalues
        .as_slice()
        .try_into()
        .expect("3 eigenvalues");
    lam.sort_by(f64::total_cmp);
    let tol = 1e-9 * lam[2].abs().max(f64::MIN_POSITIVE);
    if lam[2] <= tol {
        out.push(
            Finding::error(
                codes::IMPLAUSIBLE_INERTIA,
                format!(
                    "link `{link}` has a positive mass but a zero inertia tensor: MuJoCo and \
                     most simulators reject a massive body with no rotational inertia"
                ),
            )
            .hint(hint),
        );
    } else if lam[0] < -tol {
        out.push(
            Finding::error(
                codes::IMPLAUSIBLE_INERTIA,
                format!(
                    "link `{link}` <inertia> has a negative principal moment ({:.4e}): no \
                     rigid body can produce it — simulation on this model is meaningless",
                    lam[0]
                ),
            )
            .hint(hint),
        );
    } else if lam[0] + lam[1] < lam[2] - tol {
        out.push(
            Finding::error(
                codes::IMPLAUSIBLE_INERTIA,
                format!(
                    "link `{link}` <inertia> violates the rigid-body triangle inequality: \
                     principal moments ({:.4e}, {:.4e}, {:.4e}) need {:.4e} + {:.4e} ≥ {:.4e}. \
                     No real mass distribution does this; MuJoCo refuses such bodies",
                    lam[0], lam[1], lam[2], lam[0], lam[1], lam[2]
                ),
            )
            .hint(hint),
        );
    }
}

/// Where a mesh reference appeared.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MeshRole {
    Collision,
    Visual,
}

pub struct MeshRef {
    pub link: String,
    pub role: MeshRole,
    pub raw: String,
}

/// Every `<mesh filename=..>` under a `<collision>` or `<visual>`, per link.
pub fn mesh_refs(v: &View) -> Vec<MeshRef> {
    let mut out = Vec::new();
    for (name, link) in &v.links {
        for (tag, role) in [
            ("collision", MeshRole::Collision),
            ("visual", MeshRole::Visual),
        ] {
            for c in link.children_named(tag) {
                if let Some(raw) = c
                    .child("geometry")
                    .and_then(|geometry| geometry.child("mesh"))
                    .and_then(|m| m.attr("filename"))
                {
                    out.push(MeshRef {
                        link: name.clone(),
                        role,
                        raw: raw.to_string(),
                    });
                }
            }
        }
    }
    out
}

pub fn mesh_basename(raw: &str) -> &str {
    raw.rsplit('/').next().unwrap_or(raw)
}

/// A003 (unresolvable/unloadable), A004 (duplicate basenames), A006 (huge hull input).
fn check_meshes(v: &View, dir: Option<&Path>, out: &mut Vec<Finding>) {
    let refs = mesh_refs(v);
    let resolved: Vec<(&MeshRef, Option<PathBuf>)> = refs
        .iter()
        .map(|r| {
            let res = resolve_mesh(&r.raw, dir);
            match res.resolved {
                Some(p) => (r, Some(p)),
                None => {
                    out.push(unresolvable_finding(r, &fmt_tried(&res.tried)));
                    (r, None)
                }
            }
        })
        .collect();

    // loadability + size of resolved COLLISION meshes (visual meshes are only
    // path-forwarded to the renderer; any format is acceptable there)
    for (r, path) in &resolved {
        let (Some(path), MeshRole::Collision) = (path, r.role) else {
            continue;
        };
        let is_stl = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("stl"));
        if !is_stl {
            let f = Finding::error(
                codes::MESH_UNRESOLVABLE,
                format!(
                    "collision mesh `{}` on link `{}` resolves to `{}`, but caliper's \
                     collision loader is STL-only: the collider is silently DROPPED and that \
                     part of the link is never collision-checked",
                    r.raw,
                    r.link,
                    path.display()
                ),
            );
            out.push(f.hint("convert the collision mesh to STL (visuals may stay in any format)"));
            continue;
        }
        let cloud = std::fs::read(path)
            .ok()
            .and_then(|b| caliper_model::stl::parse_stl(&b));
        match cloud {
            None => {
                let f = Finding::error(
                    codes::MESH_UNRESOLVABLE,
                    format!(
                        "collision mesh `{}` on link `{}` resolves to `{}` but is not a \
                         parseable STL: the collider is silently DROPPED and that part of \
                         the link is never collision-checked",
                        r.raw,
                        r.link,
                        path.display()
                    ),
                );
                out.push(f.hint("re-export the STL; both binary and ASCII are accepted"));
            }
            Some(cloud) if cloud.len() > HULL_VERT_CAP => {
                let f = Finding::info(
                    codes::COLLISION_MESH_HUGE,
                    format!(
                        "collision mesh `{}` on link `{}` has {} vertices \
                         (> {HULL_VERT_CAP}): caliper subsamples it before convex hulling — \
                         conservative and axis-exact, but an approximation, and loading is \
                         slower",
                        r.raw,
                        r.link,
                        cloud.len()
                    ),
                );
                out.push(f.hint(
                    "collision meshes only need the coarse envelope: decimate or replace \
                     with primitives",
                ));
            }
            Some(_) => {}
        }
    }

    // duplicate basenames: same basename referring to DIFFERENT files (distinct
    // canonical path, or distinct raw when unresolvable). Two spellings of one
    // file are fine.
    let mut groups: BTreeMap<&str, BTreeMap<String, BTreeSet<&str>>> = BTreeMap::new();
    for (r, path) in &resolved {
        let identity = match path {
            Some(p) => format!("file:{}", p.display()),
            None => format!("unresolved:{}", r.raw),
        };
        groups
            .entry(mesh_basename(&r.raw))
            .or_default()
            .entry(identity)
            .or_default()
            .insert(r.raw.as_str());
    }
    for (base, identities) in groups {
        if identities.len() < 2 {
            continue;
        }
        let raws: Vec<&str> = identities.values().flatten().copied().collect();
        out.push(
            Finding::warn(
                codes::DUPLICATE_MESH_BASENAME,
                format!(
                    "{} different mesh files are all named `{base}` ({}): MuJoCo and most \
                     asset pipelines identify meshes by basename, so exporting or copying \
                     them into one directory silently collides",
                    identities.len(),
                    raws.join(", ")
                ),
            )
            .hint(
                "repair with dedupe_mesh_basenames rewrites the later references to \
                 m2__<name>, m3__<name>, … and returns the matching file-copy plan",
            )
            .auto(),
        );
    }
}

fn unresolvable_finding(r: &MeshRef, tried: &str) -> Finding {
    match r.role {
        MeshRole::Collision => Finding::error(
            codes::MESH_UNRESOLVABLE,
            format!(
                "collision mesh `{}` on link `{}` cannot be resolved (tried {tried}): the \
                 collider is silently DROPPED, so that part of the link is never \
                 collision-checked",
                r.raw, r.link
            ),
        )
        .hint(
            "fix the filename, move the mesh next to the URDF, or point \
             CALIPER_PACKAGE_PATH at the package root",
        ),
        MeshRole::Visual => Finding::warn(
            codes::MESH_UNRESOLVABLE,
            format!(
                "visual mesh `{}` on link `{}` cannot be resolved (tried {tried}): the \
                 renderer falls back to a procedural placeholder",
                r.raw, r.link
            ),
        )
        .hint(
            "fix the filename, move the mesh next to the URDF, or point \
             CALIPER_PACKAGE_PATH at the package root",
        ),
    }
}

/// A005: a link that renders but cannot collide.
fn check_visual_coverage(v: &View, out: &mut Vec<Finding>) {
    for (name, link) in &v.links {
        let visuals = link.children_named("visual").count();
        if visuals > 0 && link.children_named("collision").count() == 0 {
            out.push(
                Finding::warn(
                    codes::VISUAL_WITHOUT_COLLISION,
                    format!(
                        "link `{name}` has {visuals} <visual> element(s) but NO <collision>: \
                         it renders normally yet is invisible to collision checking — a \
                         'clear' result can still sweep this link through an obstacle"
                    ),
                )
                .hint("add a <collision> (a primitive approximating the visual is enough)"),
            );
        }
    }
}

/// A007 (revolute without usable limits), A008 (zero axis), A009 (non-unit
/// axis), A014 (`<limit>` missing effort/velocity — a hard urdf-rs parse error).
fn check_joints(v: &View, out: &mut Vec<Finding>) {
    for (name, joint) in &v.joints {
        let jtype = joint.attr("type").unwrap_or("");
        if matches!(jtype, "revolute" | "continuous" | "prismatic")
            && let Some(axis) = joint.child("axis")
        {
            let raw = axis.attr("xyz").unwrap_or("");
            match parse_vec3(raw) {
                None => out.push(
                    Finding::error(
                        codes::ZERO_AXIS,
                        format!(
                            "joint `{name}` <axis xyz=\"{raw}\"> is not three numbers: the \
                             file will not load"
                        ),
                    )
                    .hint("write the axis as three floats, e.g. xyz=\"0 0 1\""),
                ),
                Some(a) => {
                    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                    if n < 1e-12 {
                        out.push(
                            Finding::error(
                                codes::ZERO_AXIS,
                                format!(
                                    "joint `{name}` <axis xyz=\"{raw}\"> has zero length: \
                                     caliper refuses to compile the model \
                                     (CompileError::ZeroAxis) — the joint's motion direction \
                                     is undefined"
                                ),
                            )
                            .hint("set the intended unit axis; the doctor cannot guess it"),
                        );
                    } else if (n - 1.0).abs() > 1e-6 {
                        out.push(
                            Finding::warn(
                                codes::AXIS_NOT_NORMALIZED,
                                format!(
                                    "joint `{name}` <axis xyz=\"{raw}\"> has length {n:.6}, \
                                     not 1: caliper normalizes it internally, but tools that \
                                     don't will scale velocities/torques by {n:.6}"
                                ),
                            )
                            .hint("repair with normalize_axes rewrites it as a unit vector")
                            .auto(),
                        );
                    }
                }
            }
        }
        let limit = joint.child("limit");
        if jtype == "revolute" {
            let usable = limit.is_some_and(|l| {
                let lo = parse_f(l, "lower").unwrap_or(0.0);
                let hi = parse_f(l, "upper").unwrap_or(0.0);
                lo < hi
            });
            if !usable {
                let detail = match limit {
                    None => "has no <limit> element".to_string(),
                    Some(l) => format!(
                        "has a degenerate range (lower={}, upper={})",
                        l.attr("lower").unwrap_or("0"),
                        l.attr("upper").unwrap_or("0")
                    ),
                };
                out.push(
                    Finding::warn(
                        codes::REVOLUTE_NO_LIMITS,
                        format!(
                            "revolute joint `{name}` {detail}: caliper treats it as \
                             unbounded — IK and planners sample the full circle and MJCF \
                             export emits no range, so nothing stops the joint at hardware \
                             stops"
                        ),
                    )
                    .hint(
                        "repair with inject_limits writes a conservative ±π range (tighten \
                         to the real hardware stops afterwards)",
                    )
                    .auto(),
                );
            }
        }
        // urdf-rs quirk (verified against its serde derives): inside a present
        // <limit>, lower/upper/effort default to 0 but velocity= has NO default
        // — omitting it rejects the whole file.
        if let Some(l) = limit
            && l.attr("velocity").is_none()
        {
            out.push(
                Finding::error(
                    codes::LIMIT_MISSING_ATTRS,
                    format!(
                        "joint `{name}` <limit> is missing velocity=: urdf-rs (the parser \
                         caliper uses) rejects the WHOLE file over this (a missing effort= \
                         merely parses as 0 = no effort limit)"
                    ),
                )
                .hint("repair with inject_limits fills a conservative velocity=1")
                .auto(),
            );
        }
    }
}

/// A011 (mimic of an unknown joint), A012 (mimic chains / self-mimic).
fn check_mimics(v: &View, out: &mut Vec<Finding>) {
    let names: HashSet<&str> = v.joints.iter().map(|(n, _)| n.as_str()).collect();
    let mimicking: HashSet<&str> = v
        .joints
        .iter()
        .filter(|(_, j)| j.child("mimic").is_some())
        .map(|(n, _)| n.as_str())
        .collect();
    for (name, joint) in &v.joints {
        let Some(m) = joint.child("mimic") else {
            continue;
        };
        let Some(src) = m.attr("joint") else {
            out.push(Finding::error(
                codes::MIMIC_UNKNOWN_SOURCE,
                format!(
                    "joint `{name}` has a <mimic> without a joint= attribute: there is \
                     nothing to follow and the file will not load"
                ),
            ));
            continue;
        };
        if !names.contains(src) {
            out.push(
                Finding::error(
                    codes::MIMIC_UNKNOWN_SOURCE,
                    format!(
                        "joint `{name}` mimics `{src}`, which does not exist: caliper \
                         refuses to compile (CompileError::MimicUnknownSource)"
                    ),
                )
                .hint("point the mimic at an existing non-mimic joint"),
            );
        } else if src == name || mimicking.contains(src) {
            out.push(
                Finding::error(
                    codes::MIMIC_CHAIN,
                    format!(
                        "joint `{name}` mimics `{src}`, which is itself a mimic: chains are \
                         unsupported everywhere (caliper: CompileError::MimicOfMimic)"
                    ),
                )
                .hint("point every mimic directly at the one driving joint"),
            );
        }
    }
}

/// A013 helper: xacro constructs left in a document — `xacro:` elements plus
/// `${..}` / `$(..)` substitution tokens in attributes or text.
pub fn xacro_leftovers(robot: &Element) -> Vec<String> {
    let mut out = Vec::new();
    scan_leftovers(robot, &mut out);
    out
}

fn scan_leftovers(e: &Element, out: &mut Vec<String>) {
    if e.name.starts_with("xacro:") {
        out.push(format!("element <{}>", e.name));
    }
    for (k, val) in &e.attrs {
        if val.contains("${") || val.contains("$(") {
            out.push(format!("attribute {k}=\"{val}\" on <{}>", e.name));
        }
    }
    for c in &e.children {
        match c {
            crate::xml::Node::Element(child) => scan_leftovers(child, out),
            crate::xml::Node::Text(t) if t.contains("${") || t.contains("$(") => {
                out.push(format!("text `{t}` in <{}>", e.name));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DoctorReport, Severity, codes, diagnose};
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../oracle/fixtures/robots/{name}"))
    }
    fn diag(name: &str) -> DoctorReport {
        diagnose(&fixture(name)).unwrap()
    }
    fn with_code<'a>(r: &'a DoctorReport, code: &str) -> Vec<&'a crate::Finding> {
        r.findings.iter().filter(|f| f.code == code).collect()
    }

    /// The one all-healthy fixture is the shared NEGATIVE case for every check:
    /// zero findings of any kind — and it must also genuinely compile.
    #[test]
    fn clean_fixture_has_zero_findings_and_compiles() {
        let r = diag("doctor_clean.urdf");
        assert!(
            r.findings.is_empty(),
            "expected clean, got:\n{}",
            r.render_text()
        );
        let m = caliper_model::Model::from_urdf(&fixture("doctor_clean.urdf")).unwrap();
        assert!(m.has_inertia, "the clean fixture carries full inertials");
    }

    #[test]
    fn missing_inertial_is_error_and_fixable_when_geometry_exists() {
        let r = diag("doctor_repairable.urdf");
        let f = with_code(&r, codes::MISSING_INERTIAL);
        assert_eq!(f.len(), 2, "l1 (box) and l2 (mesh):\n{}", r.render_text());
        for finding in &f {
            assert_eq!(finding.severity, Severity::Error);
            assert!(finding.auto_fixable, "geometry exists → fixable");
            assert!(finding.message.contains("no <inertial>"));
            assert!(finding.message.contains("MJCF"), "consequence named");
        }
        assert!(f.iter().any(|x| x.message.contains("`l1`")));
        assert!(f.iter().any(|x| x.message.contains("`l2`")));
    }

    #[test]
    fn negative_mass_counts_as_missing_and_names_the_value() {
        let r = diag("doctor_bad_inertia.urdf");
        let f = with_code(&r, codes::MISSING_INERTIAL);
        assert_eq!(f.len(), 1);
        assert!(f[0].message.contains("`l2`"));
        assert!(f[0].message.contains("mass -2"), "{}", f[0].message);
        assert!(!f[0].auto_fixable, "no geometry on l2 → not auto-fixable");
    }

    #[test]
    fn implausible_tensors_are_flagged_via_principal_moments() {
        let r = diag("doctor_bad_inertia.urdf");
        let f = with_code(&r, codes::IMPLAUSIBLE_INERTIA);
        assert_eq!(f.len(), 3, "{}", r.render_text());
        // l1: diagonal 1,1,3 → triangle inequality
        assert!(
            f.iter()
                .any(|x| x.message.contains("`l1`") && x.message.contains("triangle"))
        );
        // l3: positive mass, all-zero tensor
        assert!(
            f.iter()
                .any(|x| x.message.contains("`l3`") && x.message.contains("zero inertia"))
        );
        // l4: off-diagonal 2 with unit diagonal → NEGATIVE eigenvalue; the raw
        // diagonal (1,1,1) looks fine, so this proves the eigen path
        assert!(
            f.iter()
                .any(|x| x.message.contains("`l4`") && x.message.contains("negative"))
        );
        assert!(f.iter().all(|x| x.severity == Severity::Error));
    }

    #[test]
    fn unresolvable_collision_mesh_is_error_listing_search_paths() {
        let r = diag("doctor_mesh_missing.urdf");
        let f = with_code(&r, codes::MESH_UNRESOLVABLE);
        assert_eq!(f.len(), 2, "{}", r.render_text());
        let coll = f
            .iter()
            .find(|x| x.message.contains("nope.stl"))
            .expect("collision miss");
        assert_eq!(coll.severity, Severity::Error);
        assert!(coll.message.contains("tried"), "{}", coll.message);
        assert!(
            coll.message.contains("robots"),
            "names the searched dir: {}",
            coll.message
        );
        assert!(coll.message.contains("DROPPED"), "consequence named");
        let vis = f
            .iter()
            .find(|x| x.message.contains("ghost_pkg"))
            .expect("visual miss");
        assert_eq!(vis.severity, Severity::Warn);
        assert!(vis.message.contains("placeholder"));
    }

    #[test]
    fn duplicate_basenames_across_files_are_flagged() {
        let r = diag("doctor_dup_mesh.urdf");
        let f = with_code(&r, codes::DUPLICATE_MESH_BASENAME);
        assert_eq!(f.len(), 1, "{}", r.render_text());
        assert_eq!(f[0].severity, Severity::Warn);
        assert!(f[0].auto_fixable);
        assert!(f[0].message.contains("a/part.stl") && f[0].message.contains("b/part.stl"));
        // NEGATIVE half lives in doctor_clean.urdf: unit_cube.stl referenced as
        // both `unit_cube.stl` and `./unit_cube.stl` (same file) → no finding.
    }

    #[test]
    fn visual_without_collision_is_the_only_finding_on_its_fixture() {
        let r = diag("doctor_visual_only.urdf");
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        assert_eq!(r.findings[0].code, codes::VISUAL_WITHOUT_COLLISION);
        assert_eq!(r.findings[0].severity, Severity::Warn);
        assert!(r.findings[0].message.contains("`l1`"));
        assert!(r.findings[0].message.contains("collision checking"));
    }

    #[test]
    fn oversized_collision_mesh_gets_the_hull_cap_info() {
        // generated: 400 triangles = 1200 vertices > the 1024 hull cap
        let dir = crate::repair::tests::temp_dir("hullcap");
        let tris = 400u32;
        let mut stl = vec![0u8; 80];
        stl.extend_from_slice(&tris.to_le_bytes());
        for i in 0..tris {
            stl.extend_from_slice(&[0u8; 12]); // normal
            for v in 0..3 {
                let base = (i * 3 + v) as f32;
                for c in [base, base + 0.5, base + 0.25] {
                    stl.extend_from_slice(&c.to_le_bytes());
                }
            }
            stl.extend_from_slice(&[0u8; 2]);
        }
        std::fs::write(dir.join("big.stl"), stl).unwrap();
        std::fs::write(
            dir.join("big.urdf"),
            r#"<robot name="big"><link name="base">
                 <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
                 <collision><geometry><mesh filename="big.stl"/></geometry></collision>
               </link></robot>"#,
        )
        .unwrap();
        let r = diagnose(&dir.join("big.urdf")).unwrap();
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        assert_eq!(r.findings[0].code, crate::codes::COLLISION_MESH_HUGE);
        assert_eq!(r.findings[0].severity, Severity::Info);
        assert!(r.findings[0].message.contains("1200"));
    }

    #[test]
    fn revolute_limit_problems_are_flagged() {
        let r = diag("doctor_repairable.urdf");
        let no_lim = with_code(&r, codes::REVOLUTE_NO_LIMITS);
        assert_eq!(no_lim.len(), 2, "{}", r.render_text());
        assert!(
            no_lim
                .iter()
                .any(|f| f.message.contains("`j1`") && f.message.contains("no <limit>"))
        );
        assert!(
            no_lim
                .iter()
                .any(|f| f.message.contains("`j3`") && f.message.contains("degenerate"))
        );
        assert!(no_lim.iter().all(|f| f.auto_fixable));
        let missing = with_code(&r, codes::LIMIT_MISSING_ATTRS);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].severity, Severity::Error);
        assert!(missing[0].message.contains("`j2`"));
        assert!(missing[0].message.contains("effort") && missing[0].message.contains("velocity"));
    }

    #[test]
    fn zero_axis_is_the_only_finding_on_its_fixture() {
        let r = diag("doctor_zero_axis.urdf");
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        assert_eq!(r.findings[0].code, codes::ZERO_AXIS);
        assert_eq!(r.findings[0].severity, Severity::Error);
        assert!(r.findings[0].message.contains("ZeroAxis"));
    }

    #[test]
    fn non_normalized_axis_is_warned_with_the_value() {
        let r = diag("doctor_repairable.urdf");
        let f = with_code(&r, codes::AXIS_NOT_NORMALIZED);
        assert_eq!(f.len(), 1);
        assert!(f[0].message.contains("0 0 2"));
        assert!(f[0].auto_fixable);
    }

    #[test]
    fn onshape_zero_mass_root_is_an_info_heuristic() {
        let r = diag("doctor_onshape.urdf");
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        let f = &r.findings[0];
        assert_eq!(f.code, codes::CAD_ZERO_MASS_ROOT);
        assert_eq!(f.severity, Severity::Info);
        assert!(f.message.starts_with("[heuristic]"));
        assert!(f.message.contains("onshape-to-robot"));
    }

    #[test]
    fn mimic_unknown_source_and_chains_are_errors() {
        let r = diag("doctor_mimic.urdf");
        let unknown = with_code(&r, codes::MIMIC_UNKNOWN_SOURCE);
        assert_eq!(unknown.len(), 1, "{}", r.render_text());
        assert!(unknown[0].message.contains("`ghost`"));
        let chains = with_code(&r, codes::MIMIC_CHAIN);
        assert_eq!(chains.len(), 2, "j4 (chain) + j5 (self)");
        assert!(chains.iter().any(|f| f.message.contains("`j4`")));
        assert!(chains.iter().any(|f| f.message.contains("`j5`")));
        // j3's VALID mimic of j1 must NOT be flagged
        assert!(!r.findings.iter().any(|f| f.message.contains("`j3` mimics")));
    }

    #[test]
    fn xacro_leftovers_in_plain_urdf_are_an_error() {
        let r = diag("doctor_xacro_leftover.urdf");
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        let f = &r.findings[0];
        assert_eq!(f.code, codes::XACRO_LEFTOVERS);
        assert_eq!(f.severity, Severity::Error);
        assert!(f.message.contains("${w}") || f.message.contains("xacro:property"));
    }

    #[test]
    fn urdf_with_xmlns_xacro_warns_but_diagnoses_the_expansion() {
        let dir = crate::repair::tests::temp_dir("xmlns");
        std::fs::write(
            dir.join("x.urdf"),
            r#"<robot name="x" xmlns:xacro="http://www.ros.org/wiki/xacro">
                 <xacro:property name="m" value="1.0"/>
                 <link name="base">
                   <inertial><mass value="${m}"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
                 </link>
               </robot>"#,
        )
        .unwrap();
        let r = diagnose(&dir.join("x.urdf")).unwrap();
        assert_eq!(r.findings.len(), 1, "{}", r.render_text());
        assert_eq!(r.findings[0].code, codes::XACRO_LEFTOVERS);
        assert_eq!(
            r.findings[0].severity,
            Severity::Warn,
            "caliper CAN expand it — a portability warning, not an error"
        );
        // and the checks ran on the EXPANDED model: mass ${m} → 1.0, no A001
        assert!(!r.findings.iter().any(|f| f.code == codes::MISSING_INERTIAL));
    }

    #[test]
    fn xacro_extension_files_are_expanded_without_a013() {
        // the existing xacro fixture: expansion must succeed and A013 not fire
        let r = diagnose(&fixture("xacro_arm.xacro")).unwrap();
        assert!(!r.findings.iter().any(|f| f.code == codes::XACRO_LEFTOVERS));
    }
}
