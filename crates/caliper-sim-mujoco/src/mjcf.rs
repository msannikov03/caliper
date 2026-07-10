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

/// Options for [`mjcf_from_model`]. `Default` = torque-driven, Earth gravity,
/// 1 ms timestep, no damping, no ground plane — the closest match to
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
    })
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
