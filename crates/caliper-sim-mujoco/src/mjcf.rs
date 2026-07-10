//! Minimal MJCF generation from a caliper [`Model`].
//!
//! Deliberately SMALL: kinematic tree + 1-dof joints + inertials + primitive
//! collision geoms + (optional) ground plane + (optional) position actuators.
//! No meshes (hull colliders are counted, not exported), no sensors, no solver
//! tuning beyond the timestep — MuJoCo defaults apply. Pure string generation:
//! this module compiles and is tested WITHOUT MuJoCo present.
//!
//! Conventions locked in the header we emit:
//! - `<compiler angle="radian" autolimits="true"/>` — caliper is radians-only.
//! - `<option timestep gravity/>` — the ONLY solver knobs we set, so a given
//!   options struct always produces the same document (determinism lives in
//!   the string, not in post-load mutation).
//! - Bodies are emitted in caliper's topological joint order, so MuJoCo's
//!   `qpos`/`qvel` order matches caliper `q`/`qd` index-for-index (the sim
//!   layer still re-resolves by joint NAME and never assumes this).
//! - Free props ([`MjcfOptions::props`]) are emitted AFTER the robot bodies,
//!   so the robot's `qpos`/`qvel` prefix ordering is unchanged (each
//!   `<freejoint>` appends 7 qpos / 6 qvel entries at the end).

use crate::MujocoError;
use caliper_model::{CollisionShape, JointKind, Model};
use caliper_spatial::{Se3, SpatialInertia};
use nalgebra::{Matrix3, Vector3};
use std::fmt::Write as _;

/// How the generated model is driven. Mutually exclusive by construction: a
/// `<position>` servo applies `kp(ctrl − q) − kv·qd` on EVERY step, so it
/// cannot coexist with direct torque injection without fighting it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Actuation {
    /// No `<actuator>` block at all. Torques are injected straight into
    /// `qfrc_applied` (generalized force, one entry per dof) by the sim layer.
    TorqueDirect,
    /// One MJCF `<position>` actuator per joint (`kp`, `kv` gains). `ctrl`
    /// holds the target joint positions; MuJoCo computes the servo torque.
    PositionServo { kp: f64, kv: f64 },
}

/// Primitive shape of a free-floating prop body. Same conventions as the
/// robot colliders: box carries HALF-extents, cylinder `h` is the FULL length
/// (Z-aligned) — the generator emits MJCF's (radius, half-length) itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropShape {
    Box { half: [f64; 3] },
    Sphere { r: f64 },
    Cylinder { r: f64, h: f64 },
}

/// A free-floating rigid body dropped into the world: one `<freejoint>`, an
/// explicit COM `<inertial>` derived analytically from `mass` + the uniform
/// primitive, and one collision geom. The MJCF body is named
/// `prop_{sanitize(name)}` (see [`MjcfDocument::prop_bodies`]).
#[derive(Clone, Debug)]
pub struct PropSpec {
    /// User-facing name; must be non-empty and unique after sanitizing.
    pub name: String,
    pub shape: PropShape,
    /// Initial world position of the body frame (= the primitive's center).
    pub pos: [f64; 3],
    /// Initial world orientation, MJCF order `[w, x, y, z]`; `None` = identity.
    pub quat: Option<[f64; 4]>,
    /// Mass (kg), finite and > 0. Inertia is computed from it, never guessed.
    pub mass: f64,
    /// Geom `rgba` in [0,1]; `None` leaves the MuJoCo default.
    pub rgba: Option<[f32; 4]>,
}

/// Options for [`mjcf_from_model`]. `Default` = torque-driven, Earth gravity,
/// 1 ms timestep, no damping, no ground plane, no props — the closest match to
/// `caliper_hal::PhysicsSimBackend` defaults.
#[derive(Clone, Debug)]
pub struct MjcfOptions {
    /// MuJoCo integration timestep `h` (s). The sim layer steps in integer
    /// multiples of this.
    pub timestep: f64,
    /// World gravity (m/s²). Caliper URDF frames are Z-up, same as MuJoCo.
    pub gravity: [f64; 3],
    /// Uniform viscous joint damping (N·m·s/rad). Caliper's `Model` does not
    /// parse URDF `<dynamics damping>`, so this is a knob, not a translation.
    pub joint_damping: f64,
    /// `Some(z)` adds an infinite ground plane at world height `z` (a plane
    /// only collides from above).
    pub ground_plane: Option<f64>,
    pub actuation: Actuation,
    /// Raw MJCF elements injected VERBATIM inside `<worldbody>` (after the
    /// ground plane / world geoms, before the robot bodies) — the hook for
    /// `<camera>` / `<geom>` / `<light>` elements the generator does not
    /// model. The string is trusted XML: it is not escaped or validated
    /// beyond a well-formedness smoke check by MuJoCo at load time, and it
    /// adds no dof, so `qpos`/`qvel` ordering guarantees are untouched.
    /// Determinism: the string is emitted byte-for-byte, so a fixed options
    /// struct still produces a fixed document.
    pub extra_worldbody_xml: Option<String>,
    /// Free-floating primitive bodies (each: `<freejoint>` + explicit COM
    /// inertial + one geom), emitted in `<worldbody>` AFTER the robot bodies
    /// so the robot's `qpos`/`qvel` prefix is unchanged. Empty by default.
    pub props: Vec<PropSpec>,
}

impl Default for MjcfOptions {
    fn default() -> Self {
        Self {
            timestep: 1e-3,
            gravity: [0.0, 0.0, -9.81],
            joint_damping: 0.0,
            ground_plane: None,
            actuation: Actuation::TorqueDirect,
            extra_worldbody_xml: None,
            props: Vec::new(),
        }
    }
}

/// A generated MJCF document plus what the generator had to leave out.
#[derive(Clone, Debug)]
pub struct MjcfDocument {
    pub xml: String,
    /// Number of `CollisionShape::ConvexHull` colliders that were SKIPPED
    /// (mesh assets are deferred). Non-zero means the MuJoCo model has LESS
    /// collision coverage than `caliper-collision` on the same model —
    /// surface this to users, never swallow it.
    pub skipped_hull_colliders: usize,
    /// `(prop name, MJCF body name)` for every [`MjcfOptions::props`] entry,
    /// in input order — the name map the sim layer resolves body ids from.
    pub prop_bodies: Vec<(String, String)>,
}

/// Generate a minimal MJCF document from a caliper model. Errors when the
/// model lacks inertials (`has_inertia == false`) or an option is non-finite.
pub fn mjcf_from_model(m: &Model, opt: &MjcfOptions) -> Result<MjcfDocument, MujocoError> {
    if !m.has_inertia {
        return Err(MujocoError::NoInertia);
    }
    if !(opt.timestep.is_finite() && opt.timestep > 0.0) {
        return Err(MujocoError::Mjcf(format!(
            "timestep must be finite and > 0, got {}",
            opt.timestep
        )));
    }
    if !opt.gravity.iter().all(|g| g.is_finite()) {
        return Err(MujocoError::NonFinite { what: "gravity" });
    }
    if !(opt.joint_damping.is_finite() && opt.joint_damping >= 0.0) {
        return Err(MujocoError::Mjcf(format!(
            "joint_damping must be finite and >= 0, got {}",
            opt.joint_damping
        )));
    }

    // Props: validate everything and lock the name map BEFORE emitting a byte.
    let mut prop_bodies: Vec<(String, String)> = Vec::with_capacity(opt.props.len());
    {
        let mut seen = std::collections::HashSet::new();
        for p in &opt.props {
            validate_prop(p)?;
            let body = format!("prop_{}", sanitize(&p.name));
            if !seen.insert(body.clone()) {
                return Err(MujocoError::Mjcf(format!(
                    "duplicate prop name `{}` (after sanitizing)",
                    p.name
                )));
            }
            prop_bodies.push((p.name.clone(), body));
        }
    }

    // Collision geoms, grouped by the movable joint their frame rides on
    // (None = welded to the world/base). Geom pose in the body frame is
    // frame.offset ∘ collider.origin — the LinkFrame already carries the
    // folded fixed-chain offset.
    let mut world_geoms: Vec<String> = Vec::new();
    let mut body_geoms: Vec<Vec<String>> = vec![Vec::new(); m.ndof];
    let mut skipped_hull_colliders = 0usize;
    for (gi, g) in m.collision.iter().enumerate() {
        let frame = &m.frames[g.frame];
        let pose = frame.offset.compose(&g.origin);
        let name = format!("col{gi}_{}", sanitize(&frame.name));
        match geom_xml(&name, &pose, &g.shape) {
            Some(x) => match frame.anchor {
                Some(j) => body_geoms[j].push(x),
                None => world_geoms.push(x),
            },
            None => skipped_hull_colliders += 1,
        }
    }

    // Children lists in topological order (parent index < own index).
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); m.ndof];
    let mut roots: Vec<usize> = Vec::new();
    for (i, p) in m.parent.iter().enumerate() {
        match p {
            Some(pi) => kids[*pi].push(i),
            None => roots.push(i),
        }
    }

    let mut xml = String::new();
    let _ = writeln!(xml, "<mujoco model=\"{}\">", sanitize(&m.name));
    xml.push_str("  <compiler angle=\"radian\" autolimits=\"true\"/>\n");
    let _ = writeln!(
        xml,
        "  <option timestep=\"{}\" gravity=\"{} {} {}\"/>",
        f(opt.timestep),
        f(opt.gravity[0]),
        f(opt.gravity[1]),
        f(opt.gravity[2])
    );
    xml.push_str("  <worldbody>\n");
    if let Some(z) = opt.ground_plane {
        if !z.is_finite() {
            return Err(MujocoError::NonFinite {
                what: "ground_plane",
            });
        }
        let _ = writeln!(
            xml,
            "    <geom name=\"caliper_ground\" type=\"plane\" pos=\"0 0 {}\" size=\"5 5 0.1\"/>",
            f(z)
        );
    }
    for g in &world_geoms {
        let _ = writeln!(xml, "    {g}");
    }
    if let Some(extra) = &opt.extra_worldbody_xml {
        for line in extra.lines() {
            let _ = writeln!(xml, "    {}", line.trim_end());
        }
    }
    for &r in &roots {
        emit_body(&mut xml, m, opt, r, &kids, &body_geoms, 2)?;
    }
    // Free props AFTER the robot bodies (see the header: keeps the robot's
    // qpos/qvel prefix ordering; each freejoint appends 7 qpos / 6 qvel).
    for (p, (_, body)) in opt.props.iter().zip(&prop_bodies) {
        let mut battr = format!("pos=\"{} {} {}\"", f(p.pos[0]), f(p.pos[1]), f(p.pos[2]));
        if let Some(q) = p.quat {
            let _ = write!(
                battr,
                " quat=\"{} {} {} {}\"",
                f(q[0]),
                f(q[1]),
                f(q[2]),
                f(q[3])
            );
        }
        let _ = writeln!(xml, "    <body name=\"{body}\" {battr}>");
        let _ = writeln!(xml, "      <freejoint name=\"{body}_free\"/>");
        let di = prop_diaginertia(&p.shape, p.mass);
        let _ = writeln!(
            xml,
            "      <inertial pos=\"0 0 0\" mass=\"{}\" diaginertia=\"{} {} {}\"/>",
            f(p.mass),
            f(di[0]),
            f(di[1]),
            f(di[2])
        );
        let rgba = p
            .rgba
            .map(|c| format!(" rgba=\"{:?} {:?} {:?} {:?}\"", c[0], c[1], c[2], c[3]))
            .unwrap_or_default();
        let _ = writeln!(
            xml,
            "      <geom name=\"{body}_geom\" {}{rgba}/>",
            prop_geom_attrs(&p.shape)
        );
        let _ = writeln!(xml, "    </body>");
    }
    xml.push_str("  </worldbody>\n");

    if let Actuation::PositionServo { kp, kv } = opt.actuation {
        if !(kp.is_finite() && kp > 0.0 && kv.is_finite() && kv >= 0.0) {
            return Err(MujocoError::Mjcf(format!(
                "position servo gains must be finite with kp > 0, got kp={kp} kv={kv}"
            )));
        }
        xml.push_str("  <actuator>\n");
        for jn in &m.joint_names {
            let jn = sanitize(jn);
            let _ = writeln!(
                xml,
                "    <position name=\"pos_{jn}\" joint=\"{jn}\" kp=\"{}\" kv=\"{}\"/>",
                f(kp),
                f(kv)
            );
        }
        xml.push_str("  </actuator>\n");
    }
    xml.push_str("</mujoco>\n");

    Ok(MjcfDocument {
        xml,
        skipped_hull_colliders,
        prop_bodies,
    })
}

/// Reject a bad prop BEFORE it reaches the MuJoCo compiler (clearer errors,
/// and the generator stays deterministic-or-Err, never partially emitted).
fn validate_prop(p: &PropSpec) -> Result<(), MujocoError> {
    if p.name.trim().is_empty() {
        return Err(MujocoError::Mjcf("prop name must be non-empty".into()));
    }
    if !(p.mass.is_finite() && p.mass > 0.0) {
        return Err(MujocoError::Mjcf(format!(
            "prop `{}`: mass must be finite and > 0, got {}",
            p.name, p.mass
        )));
    }
    if !p.pos.iter().all(|x| x.is_finite()) {
        return Err(MujocoError::NonFinite { what: "prop pos" });
    }
    if let Some(q) = p.quat {
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !q.iter().all(|x| x.is_finite()) || norm < 1e-9 {
            return Err(MujocoError::Mjcf(format!(
                "prop `{}`: quat must be finite with non-zero norm",
                p.name
            )));
        }
    }
    if let Some(c) = p.rgba
        && !c.iter().all(|x| x.is_finite())
    {
        return Err(MujocoError::NonFinite { what: "prop rgba" });
    }
    let dims_ok = match p.shape {
        PropShape::Box { half } => half.iter().all(|x| x.is_finite() && *x > 0.0),
        PropShape::Sphere { r } => r.is_finite() && r > 0.0,
        PropShape::Cylinder { r, h } => r.is_finite() && r > 0.0 && h.is_finite() && h > 0.0,
    };
    if !dims_ok {
        return Err(MujocoError::Mjcf(format!(
            "prop `{}`: shape dimensions must be finite and > 0",
            p.name
        )));
    }
    Ok(())
}

/// Principal moments about the COM of a uniform-density primitive (kg·m²).
/// Box takes HALF-extents `[a,b,c]`: `Ixx = m/3·(b²+c²)` (= m/12 of the full
/// sides); sphere `2/5·m·r²`; cylinder (full length `h`, Z axis)
/// `Ixx = Iyy = m·(3r²+h²)/12`, `Izz = m·r²/2`.
fn prop_diaginertia(shape: &PropShape, mass: f64) -> [f64; 3] {
    match *shape {
        PropShape::Box { half: [a, b, c] } => [
            mass / 3.0 * (b * b + c * c),
            mass / 3.0 * (a * a + c * c),
            mass / 3.0 * (a * a + b * b),
        ],
        PropShape::Sphere { r } => {
            let i = 0.4 * mass * r * r;
            [i, i, i]
        }
        PropShape::Cylinder { r, h } => {
            let ixy = mass * (3.0 * r * r + h * h) / 12.0;
            [ixy, ixy, 0.5 * mass * r * r]
        }
    }
}

/// `type`/`size` attributes for a prop geom (MJCF cylinder size = radius +
/// HALF-length, same translation as the robot colliders).
fn prop_geom_attrs(shape: &PropShape) -> String {
    match *shape {
        PropShape::Box { half: [a, b, c] } => {
            format!("type=\"box\" size=\"{} {} {}\"", f(a), f(b), f(c))
        }
        PropShape::Sphere { r } => format!("type=\"sphere\" size=\"{}\"", f(r)),
        PropShape::Cylinder { r, h } => {
            format!("type=\"cylinder\" size=\"{} {}\"", f(r), f(h / 2.0))
        }
    }
}

fn emit_body(
    xml: &mut String,
    m: &Model,
    opt: &MjcfOptions,
    i: usize,
    kids: &[Vec<usize>],
    body_geoms: &[Vec<String>],
    depth: usize,
) -> Result<(), MujocoError> {
    let pad = "  ".repeat(depth);
    let jn = sanitize(&m.joint_names[i]);
    let _ = writeln!(
        xml,
        "{pad}<body name=\"b_{jn}\" {}>",
        pose_attrs(&m.parent_to_joint[i])
    );

    // Joint: at the body origin, axis in the body(=child link) frame — the
    // exact URDF convention caliper compiled from.
    let kind = match m.kind[i] {
        JointKind::Revolute => "hinge",
        JointKind::Prismatic => "slide",
    };
    let a = m.axis[i];
    let mut jattr = format!(
        "name=\"{jn}\" type=\"{kind}\" axis=\"{} {} {}\"",
        f(a.x),
        f(a.y),
        f(a.z)
    );
    if let Some((lo, hi)) = m.limits[i] {
        let _ = write!(jattr, " range=\"{} {}\"", f(lo), f(hi));
    }
    if opt.joint_damping > 0.0 {
        let _ = write!(jattr, " damping=\"{}\"", f(opt.joint_damping));
    }
    let _ = writeln!(xml, "{pad}  <joint {jattr}/>");

    // Inertial: caliper stores the spatial inertia about the joint origin;
    // MJCF wants mass + COM + the inertia ABOUT the COM in body axes.
    let (mass, com, i_com) = mass_com_inertia(&m.inertia[i]);
    if !(mass.is_finite() && mass > 0.0) {
        return Err(MujocoError::Mjcf(format!(
            "joint `{}` carries non-positive mass {mass}",
            m.joint_names[i]
        )));
    }
    let _ = writeln!(
        xml,
        "{pad}  <inertial pos=\"{} {} {}\" mass=\"{}\" fullinertia=\"{} {} {} {} {} {}\"/>",
        f(com.x),
        f(com.y),
        f(com.z),
        f(mass),
        f(i_com[(0, 0)]),
        f(i_com[(1, 1)]),
        f(i_com[(2, 2)]),
        f(i_com[(0, 1)]),
        f(i_com[(0, 2)]),
        f(i_com[(1, 2)])
    );

    for g in &body_geoms[i] {
        let _ = writeln!(xml, "{pad}  {g}");
    }
    for &c in &kids[i] {
        emit_body(xml, m, opt, c, kids, body_geoms, depth + 1)?;
    }
    let _ = writeln!(xml, "{pad}</body>");
    Ok(())
}

/// Primitive collider → MJCF `<geom>`; `None` for `ConvexHull` (meshes are
/// deferred — the caller COUNTS these, it must not drop them silently).
fn geom_xml(name: &str, pose: &Se3, shape: &CollisionShape) -> Option<String> {
    let body = match shape {
        CollisionShape::Box { half } => format!(
            "type=\"box\" size=\"{} {} {}\"",
            f(half.x),
            f(half.y),
            f(half.z)
        ),
        CollisionShape::Sphere { radius } => format!("type=\"sphere\" size=\"{}\"", f(*radius)),
        // MJCF cylinder/capsule sizes are (radius, HALF-length of the core
        // segment) — same Z-aligned convention as URDF/caliper.
        CollisionShape::Cylinder { radius, length } => format!(
            "type=\"cylinder\" size=\"{} {}\"",
            f(*radius),
            f(length / 2.0)
        ),
        CollisionShape::Capsule { radius, length } => format!(
            "type=\"capsule\" size=\"{} {}\"",
            f(*radius),
            f(length / 2.0)
        ),
        CollisionShape::ConvexHull { .. } => return None,
    };
    Some(format!(
        "<geom name=\"{name}\" {body} {}/>",
        pose_attrs(pose)
    ))
}

/// `pos="x y z" quat="w x y z"` (MJCF quaternion order is w-first).
fn pose_attrs(se: &Se3) -> String {
    let t = se.translation_vec();
    let q = se.0.rotation; // UnitQuaternion; derefs to w/i/j/k
    format!(
        "pos=\"{} {} {}\" quat=\"{} {} {} {}\"",
        f(t.x),
        f(t.y),
        f(t.z),
        f(q.w),
        f(q.i),
        f(q.j),
        f(q.k)
    )
}

/// Extract `(mass, com, inertia-about-com)` from a caliper [`SpatialInertia`]
/// (which is expressed about the LINK ORIGIN: `[[m·I, −m·[c]×], [m·[c]×, I_o]]`
/// with `I_o = I_com − m·[c]×[c]×`). Inverse of
/// `SpatialInertia::from_mass_com_inertia`.
pub fn mass_com_inertia(si: &SpatialInertia) -> (f64, Vector3<f64>, Matrix3<f64>) {
    let g = &si.0;
    let mass = g[(0, 0)];
    if !(mass.is_finite() && mass > 0.0) {
        return (mass, Vector3::zeros(), Matrix3::zeros());
    }
    let mcx = g.fixed_view::<3, 3>(3, 0); // = m·[c]×
    let com = Vector3::new(mcx[(2, 1)], mcx[(0, 2)], mcx[(1, 0)]) / mass;
    let cx = skew(&com);
    let i_o = g.fixed_view::<3, 3>(3, 3).into_owned();
    let i_com = i_o + mass * (cx * cx);
    (mass, com, i_com)
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Shortest-roundtrip float formatting (Rust's `{:?}` for f64).
fn f(x: f64) -> String {
    format!("{x:?}")
}

/// XML attribute values + MuJoCo names: escape the five XML specials and strip
/// whitespace (MuJoCo names must not contain spaces).
fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_whitespace() => out.push('_'),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::Path;
    use std::sync::Arc;

    fn model(name: &str) -> Arc<Model> {
        Arc::new(
            Model::from_urdf(Path::new(&format!(
                "{}/../../oracle/fixtures/robots/{}",
                env!("CARGO_MANIFEST_DIR"),
                name
            )))
            .unwrap(),
        )
    }

    #[test]
    fn pendulum2_document_shape() {
        let m = model("dyn_pendulum2.urdf");
        let doc = mjcf_from_model(&m, &MjcfOptions::default()).unwrap();
        let x = &doc.xml;
        assert!(x.contains("<mujoco model=\"dyn_pendulum2\">"));
        assert!(x.contains("angle=\"radian\""));
        assert!(x.contains("timestep=\"0.001\""));
        assert!(x.contains("gravity=\"0.0 0.0 -9.81\""));
        assert_eq!(x.matches("type=\"hinge\"").count(), 2);
        assert!(x.contains("name=\"j1\"") && x.contains("name=\"j2\""));
        assert!(x.contains("range=\"-3.14 3.14\""));
        // No actuators, no plane by default.
        assert!(!x.contains("<actuator>") && !x.contains("type=\"plane\""));
        assert_eq!(doc.skipped_hull_colliders, 0);
        // j2's body sits 0.3 up its parent, exactly the URDF origin.
        assert!(x.contains("<body name=\"b_j2\" pos=\"0.0 0.0 0.3\""));
    }

    #[test]
    fn mass_com_inertia_roundtrip() {
        // l1 of dyn_pendulum2: m=1, com=(0,0,0.15), I_com=diag(.0085,.0085,.0005).
        let m = model("dyn_pendulum2.urdf");
        let (mass, com, i_com) = mass_com_inertia(&m.inertia[1]);
        assert_relative_eq!(mass, 1.0, epsilon = 1e-12);
        assert_relative_eq!(com.z, 0.15, epsilon = 1e-12);
        assert_relative_eq!(i_com[(0, 0)], 0.0085, epsilon = 1e-12);
        assert_relative_eq!(i_com[(1, 1)], 0.0085, epsilon = 1e-12);
        assert_relative_eq!(i_com[(2, 2)], 0.0005, epsilon = 1e-12);
        assert_relative_eq!(i_com[(0, 1)].abs(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn collide_arm_geoms_plane_and_servos() {
        let m = model("collide_arm.urdf");
        let opt = MjcfOptions {
            ground_plane: Some(-0.2),
            actuation: Actuation::PositionServo { kp: 40.0, kv: 4.0 },
            joint_damping: 0.5,
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        let x = &doc.xml;
        assert_eq!(x.matches("type=\"box\"").count(), 3);
        // URDF box size 0.12 0.12 0.3 → half-extents, formatted exactly like
        // the generator (shortest-roundtrip — 0.12/2.0 is NOT the same double
        // as the literal 0.06, so mirror the arithmetic instead of the digits).
        let half = format!(
            "size=\"{:?} {:?} {:?}\"",
            0.12f64 / 2.0,
            0.12f64 / 2.0,
            0.3f64 / 2.0
        );
        assert!(x.contains(&half), "missing `{half}` in:\n{x}");
        assert!(x.contains("type=\"plane\"") && x.contains("pos=\"0 0 -0.2\""));
        assert_eq!(x.matches("<position ").count(), 3);
        assert!(x.contains("kp=\"40.0\"") && x.contains("kv=\"4.0\""));
        assert_eq!(x.matches("damping=\"0.5\"").count(), 3);
        assert_eq!(doc.skipped_hull_colliders, 0);
    }

    #[test]
    fn showcase6_six_hinges() {
        let m = model("showcase6.urdf");
        let doc = mjcf_from_model(&m, &MjcfOptions::default()).unwrap();
        assert_eq!(doc.xml.matches("type=\"hinge\"").count(), 6);
        assert_eq!(doc.xml.matches("<inertial ").count(), 6);
    }

    #[test]
    fn extra_worldbody_xml_injected_verbatim() {
        let m = model("dyn_pendulum2.urdf");
        let cam =
            "<camera name=\"ots\" pos=\"1.1 -1.1 0.9\" xyaxes=\"0.707 0.707 0 -0.4 0.4 0.825\"/>";
        let opt = MjcfOptions {
            ground_plane: Some(0.0),
            extra_worldbody_xml: Some(cam.to_string()),
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        let x = &doc.xml;
        assert!(x.contains(cam), "camera line missing:\n{x}");
        // inside <worldbody>, after the ground plane, before the first body
        let wb = x.find("<worldbody>").unwrap();
        let plane = x.find("type=\"plane\"").unwrap();
        let cam_at = x.find("<camera").unwrap();
        let body = x.find("<body").unwrap();
        assert!(wb < plane && plane < cam_at && cam_at < body);
        // determinism: same options, same document
        let doc2 = mjcf_from_model(&m, &opt).unwrap();
        assert_eq!(doc.xml, doc2.xml);
    }

    #[test]
    fn props_emitted_as_free_bodies() {
        let m = model("dyn_pendulum2.urdf");
        let opt = MjcfOptions {
            ground_plane: Some(0.0),
            props: vec![
                PropSpec {
                    name: "crate 1".into(),
                    shape: PropShape::Box {
                        half: [0.1, 0.2, 0.3],
                    },
                    pos: [0.5, 0.0, 0.4],
                    quat: None,
                    mass: 0.3,
                    rgba: Some([0.8, 0.2, 0.2, 1.0]),
                },
                PropSpec {
                    name: "ball".into(),
                    shape: PropShape::Sphere { r: 0.05 },
                    pos: [0.0, 0.5, 0.2],
                    quat: Some([1.0, 0.0, 0.0, 0.0]),
                    mass: 0.1,
                    rgba: None,
                },
                PropSpec {
                    name: "can".into(),
                    shape: PropShape::Cylinder { r: 0.04, h: 0.12 },
                    pos: [0.0, -0.5, 0.2],
                    quat: None,
                    mass: 0.2,
                    rgba: None,
                },
            ],
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        let x = &doc.xml;
        assert_eq!(x.matches("<freejoint").count(), 3);
        assert!(x.contains("<body name=\"prop_crate_1\" pos=\"0.5 0.0 0.4\">"));
        // box: Ixx = m/3·(b²+c²), formatted exactly like the generator
        let ixx = 0.3f64 / 3.0 * (0.2f64 * 0.2 + 0.3 * 0.3);
        assert!(x.contains(&format!("diaginertia=\"{ixx:?}")), "{x}");
        // sphere: 2/5·m·r² on all three axes
        let i_s = 0.4 * 0.1f64 * 0.05 * 0.05;
        assert!(x.contains(&format!("diaginertia=\"{i_s:?} {i_s:?} {i_s:?}\"")));
        // cylinder geom size = (radius, HALF-length)
        assert!(x.contains(&format!(
            "type=\"cylinder\" size=\"0.04 {:?}\"",
            0.12f64 / 2.0
        )));
        assert!(x.contains("rgba=\"0.8 0.2 0.2 1.0\""));
        // explicit quat forwarded on the ball body
        assert!(
            x.contains("<body name=\"prop_ball\" pos=\"0.0 0.5 0.2\" quat=\"1.0 0.0 0.0 0.0\">")
        );
        // name map, in input order
        assert_eq!(
            doc.prop_bodies,
            vec![
                ("crate 1".to_string(), "prop_crate_1".to_string()),
                ("ball".to_string(), "prop_ball".to_string()),
                ("can".to_string(), "prop_can".to_string()),
            ]
        );
        // props sit AFTER the last robot joint, inside <worldbody>
        let wb_end = x.find("</worldbody>").unwrap();
        let prop_at = x.find("prop_crate_1").unwrap();
        let last_joint = x.rfind("type=\"hinge\"").unwrap();
        assert!(last_joint < prop_at && prop_at < wb_end);
        // determinism: same options, same document
        assert_eq!(mjcf_from_model(&m, &opt).unwrap().xml, doc.xml);
    }

    #[test]
    fn bad_props_rejected() {
        let m = model("dyn_pendulum2.urdf");
        let prop = |name: &str, shape: PropShape, mass: f64, quat: Option<[f64; 4]>| PropSpec {
            name: name.into(),
            shape,
            pos: [0.0, 0.0, 0.5],
            quat,
            mass,
            rgba: None,
        };
        let sphere = PropShape::Sphere { r: 0.05 };
        let cases: Vec<Vec<PropSpec>> = vec![
            vec![prop("p", sphere, 0.0, None)],      // zero mass
            vec![prop("p", sphere, f64::NAN, None)], // NaN mass
            vec![prop("p", PropShape::Sphere { r: -0.05 }, 0.1, None)], // bad radius
            vec![prop(
                "p",
                PropShape::Box {
                    half: [0.1, 0.0, 0.1],
                },
                0.1,
                None,
            )], // zero half-extent
            vec![prop(
                "p",
                PropShape::Cylinder { r: 0.04, h: 0.0 },
                0.1,
                None,
            )], // zero length
            vec![prop("p", sphere, 0.1, Some([0.0; 4]))], // zero-norm quat
            vec![prop("  ", sphere, 0.1, None)],     // blank name
            vec![
                prop("a b", sphere, 0.1, None),
                prop("a_b", sphere, 0.1, None),
            ], // dup after sanitize
        ];
        for props in cases {
            let opt = MjcfOptions {
                props,
                ..Default::default()
            };
            assert!(
                mjcf_from_model(&m, &opt).is_err(),
                "accepted bad props: {:?}",
                opt.props
            );
        }
        let mut p = prop("p", sphere, 0.1, None);
        p.pos = [f64::NAN, 0.0, 0.0];
        let opt = MjcfOptions {
            props: vec![p],
            ..Default::default()
        };
        assert!(mjcf_from_model(&m, &opt).is_err(), "accepted NaN pos");
    }

    #[test]
    fn no_inertia_is_loud() {
        let m = model("collide_shapes.urdf"); // colliders but no <inertial>
        assert!(matches!(
            mjcf_from_model(&m, &MjcfOptions::default()),
            Err(MujocoError::NoInertia)
        ));
    }

    #[test]
    fn bad_options_rejected() {
        let m = model("dyn_pendulum2.urdf");
        for opt in [
            MjcfOptions {
                timestep: 0.0,
                ..Default::default()
            },
            MjcfOptions {
                timestep: f64::NAN,
                ..Default::default()
            },
            MjcfOptions {
                joint_damping: -1.0,
                ..Default::default()
            },
            MjcfOptions {
                gravity: [0.0, f64::INFINITY, 0.0],
                ..Default::default()
            },
            MjcfOptions {
                actuation: Actuation::PositionServo { kp: 0.0, kv: 0.0 },
                ..Default::default()
            },
        ] {
            assert!(mjcf_from_model(&m, &opt).is_err(), "accepted {opt:?}");
        }
    }
}
