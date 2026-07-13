//! Opt-in repairs. Each repair references the finding code it fixes, and the
//! result is always a REPAIRED COPY (`RepairOutcome::repaired_urdf`) — the
//! input file is NEVER touched. What could not be fixed lands in
//! `RepairOutcome::skipped` with the reason, never silently.

use crate::checks::{
    InertialStatus, PlacedShape, Shape, View, inertial_status, mesh_basename, mesh_refs, parse_f,
    parse_vec3, placed_shapes,
};
use crate::massprops::{
    MassProps, box_props, capsule_props, cylinder_props, extract_mass_com_inertia, mesh_props,
    sphere_props,
};
use crate::resolve::resolve_mesh;
use crate::xml::{Element, Node};
use crate::{DoctorError, codes, load};
use caliper_spatial::SpatialInertia;
use nalgebra::{Matrix3, Vector3};
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

/// Which repairs to apply. Everything is OFF by default: repairs rewrite
/// physics-relevant fields, so each one is an explicit opt-in.
#[derive(Clone, Debug)]
pub struct RepairOpts {
    /// Fill MISSING/zero `<inertial>`s (finding A001) from the link's collision
    /// (else visual) geometry, assuming uniform `density`. Never overwrites an
    /// explicit inertial with positive mass, and never touches the root link
    /// (see A010).
    pub compute_inertials: bool,
    /// Uniform material density (kg/m³) for `compute_inertials`. Default 1000
    /// (water — a sane mid-range for printed/machined robot parts).
    pub density: f64,
    /// Rewrite non-unit joint axes as unit vectors (A009).
    pub normalize_axes: bool,
    /// Rename later duplicate mesh basenames `m2__<name>`, … and return the
    /// matching file-copy plan (A004). The DOCUMENT is rewritten here; copying
    /// the mesh files per [`RepairOutcome::mesh_copies`] is the caller's move
    /// (the engine performs no file writes at all).
    pub dedupe_mesh_basenames: bool,
    /// Inject conservative limits: a ±π range on limit-less/degenerate
    /// revolute joints (A007) and the urdf-rs-mandatory `velocity=1` on a
    /// `<limit>` that omitted it (A014).
    pub inject_limits: bool,
}

impl Default for RepairOpts {
    fn default() -> Self {
        RepairOpts {
            compute_inertials: false,
            density: 1000.0,
            normalize_axes: false,
            dedupe_mesh_basenames: false,
            inject_limits: false,
        }
    }
}

impl RepairOpts {
    /// Every repair enabled, default density.
    pub fn all() -> Self {
        RepairOpts {
            compute_inertials: true,
            normalize_axes: true,
            dedupe_mesh_basenames: true,
            inject_limits: true,
            ..Self::default()
        }
    }
}

/// One repair that was applied (or skipped, with the reason in `detail`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct RepairAction {
    /// The finding code this repair addresses (e.g. "A001").
    pub code: String,
    /// The link/joint/mesh it touched.
    pub target: String,
    pub detail: String,
}

/// A file copy the CALLER must perform for a basename dedupe to resolve:
/// the rewritten URDF references `to`'s basename.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MeshCopy {
    pub from: PathBuf,
    pub to: PathBuf,
}

/// The repaired copy plus a full account of what happened.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RepairOutcome {
    /// The repaired URDF document. Write it wherever you like — the input
    /// file was not modified.
    pub repaired_urdf: String,
    pub applied: Vec<RepairAction>,
    /// Requested repairs that could NOT be performed, with reasons.
    pub skipped: Vec<RepairAction>,
    /// File copies the caller must make for renamed mesh references to resolve.
    pub mesh_copies: Vec<MeshCopy>,
}

/// Apply the opted-in repairs to a repaired COPY of the description at `path`
/// (`.xacro` input is expanded first, so the copy is always plain URDF).
pub fn repair(path: &Path, opts: &RepairOpts) -> Result<RepairOutcome, DoctorError> {
    let loaded = load(path)?;
    let mut robot = loaded.robot;
    let dir = loaded.dir;
    let mut applied = Vec::new();
    let mut skipped = Vec::new();
    let mut mesh_copies = Vec::new();
    if opts.normalize_axes {
        repair_axes(&mut robot, &mut applied);
    }
    if opts.inject_limits {
        repair_limits(&mut robot, &mut applied);
    }
    // inertials BEFORE dedupe: they read meshes through the ORIGINAL references
    // (the dedupe copy plan is not executed yet when we integrate).
    if opts.compute_inertials {
        repair_inertials(
            &mut robot,
            dir.as_deref(),
            opts.density,
            &mut applied,
            &mut skipped,
        );
    }
    if opts.dedupe_mesh_basenames {
        repair_dupes(
            &mut robot,
            dir.as_deref(),
            &mut applied,
            &mut skipped,
            &mut mesh_copies,
        );
    }
    Ok(RepairOutcome {
        repaired_urdf: crate::xml::write_document(&robot),
        applied,
        skipped,
        mesh_copies,
    })
}

/// A009: rewrite non-unit axes.
fn repair_axes(robot: &mut Element, applied: &mut Vec<RepairAction>) {
    for joint in robot.children_named_mut("joint") {
        let name = joint.attr("name").unwrap_or("?").to_string();
        let jtype = joint.attr("type").unwrap_or("").to_string();
        if !matches!(jtype.as_str(), "revolute" | "continuous" | "prismatic") {
            continue;
        }
        let Some(axis) = joint.child_mut("axis") else {
            continue;
        };
        let Some(a) = axis.attr("xyz").and_then(parse_vec3) else {
            continue;
        };
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if n < 1e-12 || (n - 1.0).abs() <= 1e-6 {
            continue; // zero axes (A008) are not guessable; unit axes are fine
        }
        let old = axis.attr("xyz").unwrap_or("").to_string();
        axis.set_attr("xyz", &format!("{} {} {}", a[0] / n, a[1] / n, a[2] / n));
        applied.push(RepairAction {
            code: codes::AXIS_NOT_NORMALIZED.to_string(),
            target: name,
            detail: format!("normalized axis `{old}` (length was {n:.6})"),
        });
    }
}

/// A007 + A014: conservative limit injection.
fn repair_limits(robot: &mut Element, applied: &mut Vec<RepairAction>) {
    for joint in robot.children_named_mut("joint") {
        let name = joint.attr("name").unwrap_or("?").to_string();
        let revolute = joint.attr("type") == Some("revolute");
        if joint.child("limit").is_none() {
            if revolute {
                let mut l = Element::new("limit");
                l.set_attr("lower", &(-PI).to_string());
                l.set_attr("upper", &PI.to_string());
                l.set_attr("effort", "1");
                l.set_attr("velocity", "1");
                joint.children.push(Node::Element(l));
                applied.push(RepairAction {
                    code: codes::REVOLUTE_NO_LIMITS.to_string(),
                    target: name,
                    detail: "injected conservative <limit lower=-π upper=π effort=1 \
                             velocity=1>; tighten to the real hardware stops"
                        .to_string(),
                });
            }
            continue;
        }
        let Some(l) = joint.child_mut("limit") else {
            continue; // unreachable: checked non-None above
        };
        // only velocity= is mandatory for urdf-rs; a missing effort= parses as 0
        if l.attr("velocity").is_none() {
            l.set_attr("velocity", "1");
            applied.push(RepairAction {
                code: codes::LIMIT_MISSING_ATTRS.to_string(),
                target: name.clone(),
                detail: "filled missing velocity=1 (urdf-rs rejects a <limit> without it)"
                    .to_string(),
            });
        }
        if revolute {
            let lo = parse_f(l, "lower").unwrap_or(0.0);
            let hi = parse_f(l, "upper").unwrap_or(0.0);
            let usable = lo < hi;
            if !usable {
                l.set_attr("lower", &(-PI).to_string());
                l.set_attr("upper", &PI.to_string());
                applied.push(RepairAction {
                    code: codes::REVOLUTE_NO_LIMITS.to_string(),
                    target: name,
                    detail: format!(
                        "replaced degenerate range (lower={lo}, upper={hi}) with a \
                         conservative ±π"
                    ),
                });
            }
        }
    }
}

/// Unit-density mass properties for one shape (meshes are resolved, loaded,
/// scaled, and integrated). `Err` carries the plain-English reason a link's
/// inertial could NOT be derived from this shape.
fn shape_props(shape: &Shape, dir: Option<&Path>) -> Result<MassProps, String> {
    match shape {
        Shape::Box { half } => Ok(box_props(*half)),
        Shape::Sphere { radius } => Ok(sphere_props(*radius)),
        Shape::Cylinder { radius, length } => Ok(cylinder_props(*radius, *length)),
        Shape::Capsule { radius, length } => Ok(capsule_props(*radius, *length)),
        Shape::Mesh { raw, scale } => {
            let Some(path) = resolve_mesh(raw, dir).resolved else {
                return Err(format!("mesh `{raw}` cannot be resolved"));
            };
            let is_stl = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("stl"));
            if !is_stl {
                return Err(format!("mesh `{raw}` is not an STL — cannot integrate it"));
            }
            let bytes =
                std::fs::read(path).map_err(|e| format!("mesh `{raw}`: read failed ({e})"))?;
            let mut cloud = caliper_model::stl::parse_stl(&bytes)
                .ok_or_else(|| format!("mesh `{raw}` is not a parseable STL"))?;
            for v in &mut cloud {
                v.coords
                    .component_mul_assign(&Vector3::new(scale[0], scale[1], scale[2]));
            }
            mesh_props(&cloud).ok_or_else(|| {
                format!(
                    "mesh `{raw}` is open, degenerate, or inconsistently wound — its signed \
                     volume integrates to ~0, so no honest inertial can be derived"
                )
            })
        }
    }
}

/// A001: fill missing/zero inertials by divergence-theorem/analytic integration
/// of the link's geometry. Skips the root link (see A010) and any link where
/// even one shape cannot be integrated (a partial inertial would be a lie).
fn repair_inertials(
    robot: &mut Element,
    dir: Option<&Path>,
    density: f64,
    applied: &mut Vec<RepairAction>,
    skipped: &mut Vec<RepairAction>,
) {
    struct Plan {
        link: String,
        inertial: Element,
        detail: String,
    }
    let mut plans: Vec<Plan> = Vec::new();
    {
        let v = View::new(robot);
        for (name, link) in &v.links {
            if Some(name) == v.root_link.as_ref() {
                continue;
            }
            if matches!(inertial_status(link), InertialStatus::Present { .. }) {
                continue;
            }
            let mut shapes: Vec<PlacedShape> = placed_shapes(link, "collision");
            let mut src = "collision";
            if shapes.is_empty() {
                shapes = placed_shapes(link, "visual");
                src = "visual";
            }
            if shapes.is_empty() {
                skipped.push(RepairAction {
                    code: codes::MISSING_INERTIAL.to_string(),
                    target: name.clone(),
                    detail: "no usable <collision> or <visual> geometry to derive an \
                             inertial from"
                        .to_string(),
                });
                continue;
            }
            let mut total = SpatialInertia::zero();
            let mut failure: Option<String> = None;
            for ps in &shapes {
                match shape_props(&ps.shape, dir) {
                    // shape-frame props → link frame via the shape's <origin>
                    Ok(p) => total = total.add(&p.spatial(density).transform(&ps.origin)),
                    Err(reason) => {
                        failure = Some(reason);
                        break;
                    }
                }
            }
            if let Some(reason) = failure {
                skipped.push(RepairAction {
                    code: codes::MISSING_INERTIAL.to_string(),
                    target: name.clone(),
                    detail: reason,
                });
                continue;
            }
            let (mass, com, i_com) = extract_mass_com_inertia(&total);
            if !(mass.is_finite() && mass > 1e-12) {
                skipped.push(RepairAction {
                    code: codes::MISSING_INERTIAL.to_string(),
                    target: name.clone(),
                    detail: format!("integrated mass is ~0 ({mass:.3e} kg) — refusing to write it"),
                });
                continue;
            }
            plans.push(Plan {
                link: name.clone(),
                inertial: inertial_element(mass, &com, &i_com),
                detail: format!(
                    "computed mass {mass:.6} kg from {} {src} shape(s) at density \
                     {density} kg/m³ (divergence-theorem integrals for meshes)",
                    shapes.len()
                ),
            });
        }
    }
    for plan in plans {
        let Some(link) = robot
            .children_named_mut("link")
            .find(|l| l.attr("name") == Some(plan.link.as_str()))
        else {
            continue;
        };
        // drop any zero-mass husk, then lead the link with the fresh inertial
        link.children
            .retain(|n| !matches!(n, Node::Element(e) if e.name == "inertial"));
        link.children.insert(0, Node::Element(plan.inertial));
        applied.push(RepairAction {
            code: codes::MISSING_INERTIAL.to_string(),
            target: plan.link,
            detail: plan.detail,
        });
    }
}

/// Build `<inertial>` with the tensor ABOUT THE COM in link axes (origin rpy
/// stays zero, so the numbers mean exactly what `caliper_model::parse_inertial`
/// reads back).
fn inertial_element(mass: f64, com: &Vector3<f64>, i_com: &Matrix3<f64>) -> Element {
    let mut origin = Element::new("origin");
    origin.set_attr("xyz", &format!("{} {} {}", com.x, com.y, com.z));
    origin.set_attr("rpy", "0 0 0");
    let mut m = Element::new("mass");
    m.set_attr("value", &mass.to_string());
    let mut i = Element::new("inertia");
    for (key, v) in [
        ("ixx", i_com[(0, 0)]),
        ("ixy", i_com[(0, 1)]),
        ("ixz", i_com[(0, 2)]),
        ("iyy", i_com[(1, 1)]),
        ("iyz", i_com[(1, 2)]),
        ("izz", i_com[(2, 2)]),
    ] {
        i.set_attr(key, &v.to_string());
    }
    let mut inertial = Element::new("inertial");
    inertial.children = vec![Node::Element(origin), Node::Element(m), Node::Element(i)];
    inertial
}

/// A004: rename later duplicate basenames and plan the file copies.
fn repair_dupes(
    robot: &mut Element,
    dir: Option<&Path>,
    applied: &mut Vec<RepairAction>,
    skipped: &mut Vec<RepairAction>,
    mesh_copies: &mut Vec<MeshCopy>,
) {
    // phase 1 (immutable): group refs by basename, identities in first-seen order
    let mut rename: BTreeMap<String, String> = BTreeMap::new();
    {
        let v = View::new(robot);
        // basename → [(identity, resolved, raw spellings)] in first-seen order
        type Group = Vec<(String, Option<PathBuf>, Vec<String>)>;
        let mut groups: BTreeMap<String, Group> = BTreeMap::new();
        for r in mesh_refs(&v) {
            let resolved = resolve_mesh(&r.raw, dir).resolved;
            let identity = match &resolved {
                Some(p) => format!("file:{}", p.display()),
                None => format!("unresolved:{}", r.raw),
            };
            let group = groups.entry(mesh_basename(&r.raw).to_string()).or_default();
            match group.iter_mut().find(|(id, _, _)| *id == identity) {
                Some((_, _, raws)) => {
                    if !raws.contains(&r.raw) {
                        raws.push(r.raw.clone());
                    }
                }
                None => group.push((identity, resolved, vec![r.raw.clone()])),
            }
        }
        for (base, identities) in &groups {
            if identities.len() < 2 {
                continue;
            }
            // the first-seen file keeps its name; every later one gets m<k>__
            for (k, (_, resolved, raws)) in identities.iter().enumerate().skip(1) {
                let new_base = format!("m{}__{base}", k + 1);
                let Some(from) = resolved else {
                    skipped.push(RepairAction {
                        code: codes::DUPLICATE_MESH_BASENAME.to_string(),
                        target: raws.join(", "),
                        detail: "duplicate basename, but the file is unresolvable — no copy \
                                 can be planned (fix the path first)"
                            .to_string(),
                    });
                    continue;
                };
                mesh_copies.push(MeshCopy {
                    from: from.clone(),
                    to: from.with_file_name(&new_base),
                });
                for raw in raws {
                    let new_raw = match raw.rsplit_once('/') {
                        Some((dir_part, _)) => format!("{dir_part}/{new_base}"),
                        None => new_base.clone(),
                    };
                    rename.insert(raw.clone(), new_raw);
                }
            }
        }
    }
    if rename.is_empty() {
        return;
    }
    // phase 2 (mutable): rewrite every renamed reference, wherever it appears
    for link in robot.children_named_mut("link") {
        for tag in ["collision", "visual"] {
            for c in link.children_named_mut(tag) {
                let Some(mesh) = c.child_mut("geometry").and_then(|g| g.child_mut("mesh")) else {
                    continue;
                };
                let Some(raw) = mesh.attr("filename").map(str::to_string) else {
                    continue;
                };
                if let Some(new_raw) = rename.get(&raw) {
                    mesh.set_attr("filename", new_raw);
                    applied.push(RepairAction {
                        code: codes::DUPLICATE_MESH_BASENAME.to_string(),
                        target: raw.clone(),
                        detail: format!(
                            "renamed to `{new_raw}`; copy the file per mesh_copies to make \
                             it resolve"
                        ),
                    });
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::diagnose;
    use crate::massprops::testmesh::{ascii_stl, cube_tris};
    use caliper_model::Model;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fresh scratch directory per call (unique via pid + counter).
    pub(crate) fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "caliper-doctor-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../oracle/fixtures/robots/{name}"))
    }

    /// Copy doctor_repairable.urdf into a temp dir with a WELL-WOUND unit cube
    /// STL beside it (the repo's unit_cube.stl is a hull fixture with
    /// inconsistent winding — see massprops).
    fn stage_repairable(tag: &str) -> (PathBuf, PathBuf) {
        let dir = temp_dir(tag);
        let urdf = dir.join("doctor_repairable.urdf");
        std::fs::copy(fixture("doctor_repairable.urdf"), &urdf).unwrap();
        let cube = ascii_stl(&cube_tris(0.5, Vector3::zeros()));
        std::fs::write(dir.join("unit_cube.stl"), cube).unwrap();
        (dir, urdf)
    }

    fn link_inertial(root: &Element, link: &str) -> Option<(f64, [f64; 3], [f64; 6])> {
        let l = root
            .children_named("link")
            .find(|l| l.attr("name") == Some(link))?;
        let inr = l.child("inertial")?;
        let mass = parse_f(inr.child("mass")?, "value")?;
        let xyz = parse_vec3(inr.child("origin")?.attr("xyz")?)?;
        let i = inr.child("inertia")?;
        let t = [
            parse_f(i, "ixx")?,
            parse_f(i, "iyy")?,
            parse_f(i, "izz")?,
            parse_f(i, "ixy")?,
            parse_f(i, "ixz")?,
            parse_f(i, "iyz")?,
        ];
        Some((mass, xyz, t))
    }

    /// THE round-trip guarantee: repair everything, then the copy must compile
    /// via caliper-model WITH inertia and re-diagnose completely clean.
    #[test]
    fn repair_round_trip_compiles_and_rediagnoses_clean() {
        let (dir, urdf) = stage_repairable("roundtrip");
        let before = std::fs::read(&urdf).unwrap();

        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        assert_eq!(
            std::fs::read(&urdf).unwrap(),
            before,
            "the INPUT file must never be mutated"
        );
        let codes_applied: Vec<&str> = out.applied.iter().map(|a| a.code.as_str()).collect();
        for (code, times) in [
            (codes::MISSING_INERTIAL, 2),   // l1 box + l2 mesh
            (codes::REVOLUTE_NO_LIMITS, 2), // j1 absent + j3 degenerate
            (codes::AXIS_NOT_NORMALIZED, 1),
            (codes::LIMIT_MISSING_ATTRS, 1),
        ] {
            assert_eq!(
                codes_applied.iter().filter(|&&c| c == code).count(),
                times,
                "applied {code}: {:?}",
                out.applied
            );
        }
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);

        let repaired = dir.join("repaired.urdf");
        std::fs::write(&repaired, &out.repaired_urdf).unwrap();
        let m = Model::from_urdf(&repaired).expect("repaired copy must compile");
        assert_eq!(m.ndof, 3);
        assert!(m.has_inertia, "computed inertials must gate dynamics ON");
        let r = diagnose(&repaired).unwrap();
        assert!(
            r.findings.is_empty(),
            "repaired copy must re-diagnose clean:\n{}",
            r.render_text()
        );
    }

    /// The computed box inertial must match the analytic value, INCLUDING the
    /// shape's <origin> offset: l1's 1 m³ cube sits at z = 0.5, so the COM
    /// moves there while the COM tensor stays m/6.
    #[test]
    fn computed_box_inertial_matches_analytic() {
        let (_dir, urdf) = stage_repairable("boxval");
        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        let root = crate::xml::parse_document(&out.repaired_urdf).unwrap();
        let (mass, xyz, t) = link_inertial(&root, "l1").expect("l1 got an inertial");
        assert!((mass - 1000.0).abs() < 1e-6, "1 m³ at 1000 kg/m³");
        assert!(xyz[0].abs() < 1e-9 && xyz[1].abs() < 1e-9);
        assert!((xyz[2] - 0.5).abs() < 1e-9, "COM follows the origin offset");
        for d in &t[..3] {
            assert!((d - 1000.0 / 6.0).abs() < 1e-6, "I = m·a²/6, got {d}");
        }
        for od in &t[3..] {
            assert!(od.abs() < 1e-9, "no off-diagonals for an axis-aligned box");
        }
        // negative half: l3's AUTHORED inertial must survive repair untouched
        let (m3, _, t3) = link_inertial(&root, "l3").expect("l3 keeps its inertial");
        assert!((m3 - 0.4).abs() < 1e-12, "explicit mass never overwritten");
        assert!((t3[0] - 0.004).abs() < 1e-12);
    }

    /// Same numbers via the MESH path — the divergence-theorem integrals feed
    /// the exact same analytic values end-to-end through STL load + repair.
    #[test]
    fn computed_mesh_inertial_matches_analytic() {
        let (_dir, urdf) = stage_repairable("meshval");
        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        let root = crate::xml::parse_document(&out.repaired_urdf).unwrap();
        let (mass, xyz, t) = link_inertial(&root, "l2").expect("l2 got an inertial");
        assert!((mass - 1000.0).abs() < 1e-3, "unit cube mesh, got {mass}");
        assert!(xyz.iter().all(|c| c.abs() < 1e-9), "centered cube");
        for d in &t[..3] {
            assert!((d - 1000.0 / 6.0).abs() < 1e-3, "I = m·a²/6, got {d}");
        }
    }

    /// The repo's inconsistently wound unit_cube.stl must be REFUSED: a
    /// skipped action with the reason, no invented inertial, and the copy
    /// still compiles (just without dynamics).
    #[test]
    fn open_or_miswound_mesh_is_refused_not_guessed() {
        let dir = temp_dir("miswound");
        let urdf = dir.join("doctor_repairable.urdf");
        std::fs::copy(fixture("doctor_repairable.urdf"), &urdf).unwrap();
        std::fs::copy(fixture("unit_cube.stl"), dir.join("unit_cube.stl")).unwrap();
        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        let skip = out
            .skipped
            .iter()
            .find(|s| s.target == "l2")
            .expect("l2's mesh must be refused");
        assert_eq!(skip.code, codes::MISSING_INERTIAL);
        assert!(skip.detail.contains("wound"), "{}", skip.detail);
        let root = crate::xml::parse_document(&out.repaired_urdf).unwrap();
        assert!(link_inertial(&root, "l2").is_none(), "nothing invented");
        let repaired = dir.join("repaired.urdf");
        std::fs::write(&repaired, &out.repaired_urdf).unwrap();
        let m = Model::from_urdf(&repaired).unwrap();
        assert!(!m.has_inertia, "l2 stayed massless — honestly");
    }

    #[test]
    fn injected_limits_and_normalized_axis_have_the_promised_values() {
        let (_dir, urdf) = stage_repairable("values");
        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        let root = crate::xml::parse_document(&out.repaired_urdf).unwrap();
        let joint = |n: &str| {
            root.children_named("joint")
                .find(|j| j.attr("name") == Some(n))
                .unwrap()
        };
        // j1: axis "0 0 2" → unit; injected fresh conservative limit
        let a = parse_vec3(joint("j1").child("axis").unwrap().attr("xyz").unwrap()).unwrap();
        assert!((a[2] - 1.0).abs() < 1e-12 && a[0] == 0.0 && a[1] == 0.0);
        let l1 = joint("j1").child("limit").expect("limit injected");
        assert!((parse_f(l1, "lower").unwrap() + PI).abs() < 1e-12);
        assert!((parse_f(l1, "upper").unwrap() - PI).abs() < 1e-12);
        assert_eq!(l1.attr("effort"), Some("1"));
        assert_eq!(l1.attr("velocity"), Some("1"));
        // j2: kept its authored range, gained the mandatory velocity= — and
        // effort= stays ABSENT (it parses as 0, so nothing needs inventing)
        let l2 = joint("j2").child("limit").unwrap();
        assert_eq!(l2.attr("lower"), Some("-1"));
        assert_eq!(l2.attr("velocity"), Some("1"));
        assert_eq!(l2.attr("effort"), None);
        // j3: degenerate 0..0 widened to ±π, authored effort kept
        let l3 = joint("j3").child("limit").unwrap();
        assert!((parse_f(l3, "lower").unwrap() + PI).abs() < 1e-12);
        assert_eq!(l3.attr("effort"), Some("5"));
    }

    #[test]
    fn dedupe_renames_later_files_and_plans_copies() {
        let dir = temp_dir("dedupe");
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b")).unwrap();
        let cube = ascii_stl(&cube_tris(0.05, Vector3::zeros()));
        std::fs::write(dir.join("a/part.stl"), &cube).unwrap();
        std::fs::write(dir.join("b/part.stl"), &cube).unwrap();
        let urdf = dir.join("r.urdf");
        std::fs::write(
            &urdf,
            r#"<robot name="r">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="l1"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><geometry><mesh filename="a/part.stl"/></geometry></collision></link>
  <link name="l2"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><geometry><mesh filename="b/part.stl"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"#,
        )
        .unwrap();
        let r = diagnose(&urdf).unwrap();
        assert_eq!(r.warnings, 1, "the dup finding:\n{}", r.render_text());

        let out = repair(
            &urdf,
            &RepairOpts {
                dedupe_mesh_basenames: true,
                ..RepairOpts::default()
            },
        )
        .unwrap();
        assert_eq!(out.mesh_copies.len(), 1, "{:?}", out.mesh_copies);
        assert!(
            out.applied
                .iter()
                .any(|a| a.code == codes::DUPLICATE_MESH_BASENAME)
        );
        // the caller's move: perform the planned copy, then everything resolves
        for c in &out.mesh_copies {
            std::fs::copy(&c.from, &c.to).unwrap();
        }
        let repaired = dir.join("repaired.urdf");
        std::fs::write(&repaired, &out.repaired_urdf).unwrap();
        let again = diagnose(&repaired).unwrap();
        assert!(
            again.findings.is_empty(),
            "no dup, no unresolvable:\n{}",
            again.render_text()
        );
        let m = Model::from_urdf(&repaired).unwrap();
        assert_eq!(m.collision.len(), 2, "both hulls load");
        assert!(m.dropped_collider_frames.is_empty());
    }

    #[test]
    fn default_opts_change_nothing_and_report_nothing() {
        let (_dir, urdf) = stage_repairable("noop");
        let out = repair(&urdf, &RepairOpts::default()).unwrap();
        assert!(out.applied.is_empty());
        assert!(out.skipped.is_empty());
        assert!(out.mesh_copies.is_empty());
        // still a structurally valid document
        crate::xml::parse_document(&out.repaired_urdf).unwrap();
    }

    #[test]
    fn link_without_geometry_is_skipped_with_reason() {
        let dir = temp_dir("nogeo");
        let urdf = dir.join("r.urdf");
        std::fs::write(
            &urdf,
            r#"<robot name="r">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="l1"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"#,
        )
        .unwrap();
        let out = repair(&urdf, &RepairOpts::all()).unwrap();
        assert_eq!(out.skipped.len(), 1);
        assert_eq!(out.skipped[0].target, "l1");
        assert!(out.skipped[0].detail.contains("no usable"));
        assert!(out.applied.is_empty());
        let repaired = dir.join("repaired.urdf");
        std::fs::write(&repaired, &out.repaired_urdf).unwrap();
        Model::from_urdf(&repaired).expect("still compiles, just without dynamics");
    }
}
