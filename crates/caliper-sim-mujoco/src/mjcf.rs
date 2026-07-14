//! Minimal MJCF generation from a caliper [`Model`].
//!
//! Deliberately SMALL: kinematic tree + 1-dof joints + inertials + collision
//! geoms + (optional) ground plane + (optional) position actuators. No
//! sensors, no solver tuning beyond `<option timestep gravity>` — MuJoCo
//! defaults apply. Pure string generation: this module compiles and is tested
//! WITHOUT MuJoCo present.
//!
//! # What IS and ISN'T exported
//!
//! The document is COLLISION/DYNAMICS-focused — it is meant to drop into the
//! MuJoCo / MJX / Warp / Newton ecosystem for physics, not for rendering.
//!
//! | caliper model feature | MJCF output |
//! |---|---|
//! | kinematic tree (revolute/prismatic) | nested `<body>` + `<joint type="hinge"/"slide">`, topological order (`qpos` order = caliper `q` order) |
//! | inertials | `<inertial>` (mass + COM + `fullinertia`) — REQUIRED, `has_inertia` must hold |
//! | joint limits | `range` on each `<joint>` (radians / meters) |
//! | primitive colliders (box/sphere/cylinder/capsule) | one `<geom>` each, always |
//! | mesh colliders ([`CollisionShape::ConvexHull`]) | SKIPPED by default (counted in [`MjcfDocument::skipped_hull_colliders`]); with [`MjcfOptions::export_hull_meshes`] each hull becomes an inline-vertex `<asset><mesh vertex="…">` + `<geom type="mesh">` — MuJoCo convex-hulls vertex-only meshes natively |
//! | contact materials ([`ContactMaterial`]) | opt-in `solref`/`solimp`/`friction` attributes on every emitted geom ([`MjcfOptions::default_material`], [`PropSpec::material`]); `None` = MuJoCo defaults, no attributes |
//! | convex decomposition ([`ColliderDecomposer`]) | opt-in seam: a decomposer splits each exported hull into pieces, one `<mesh>`+`<geom>` per piece; [`NaiveDecomposer`] = identity (one piece, byte-identical output) |
//! | visuals ([`caliper_model::VisualShape`]) | NOT exported — render meshes stay in Studio; MuJoCo sees collision geometry only |
//! | joint `<dynamics damping>` | NOT translated (caliper does not parse it); the uniform [`MjcfOptions::joint_damping`] knob instead |
//! | actuators | none by default (torque via `qfrc_applied`); opt-in `<position>` servos ([`Actuation::PositionServo`]) |
//! | sensors / transmissions / solver tuning | NOT exported — MuJoCo defaults |
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
use nalgebra::{Matrix3, Point3, Vector3};
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

/// A contact material: the three MuJoCo per-geom solver knobs bundled behind
/// names, because raw `solref`/`solimp` are a dark art (stiff contacts jitter
/// or explode, soft ones penetrate — the exact failure modes
/// `caliper_sim_mujoco::lint` detects).
///
/// # MuJoCo semantics (Modeling → Solver parameters)
/// - `solref = (timeconst, dampratio)` — the contact behaves like a virtual
///   mass-spring-damper: `timeconst` (s) is how fast penetration is pushed
///   out (MuJoCo requires `timeconst >= 2 * timestep` for the discretization
///   to be stable — the linter's first suggested fix), `dampratio` `1.0` =
///   critically damped (no bounce from the SOLVER itself; restitution still
///   comes from the impedance profile). We use the positive
///   `(timeconst, dampratio)` form only — MuJoCo's negative
///   "direct stiffness/damping" form is not modeled here.
/// - `solimp = (dmin, dmax, width)` — constraint impedance (how "hard" the
///   contact is, in `(0, 1)`) ramps from `dmin` at zero penetration to `dmax`
///   at penetration `width` (m). High `dmin`/`dmax` ≈ rigid (sub-mm
///   penetration); low values + wide `width` = visible squish. MuJoCo's
///   optional 4th/5th parameters (midpoint, power) are left at their
///   defaults.
/// - `friction = (slide, torsion, roll)` — tangential Coulomb coefficient
///   (dimensionless), torsional (m) and rolling (m) coefficients.
///
/// When both geoms of a contact pair carry parameters, MuJoCo mixes
/// `solref`/`solimp` by `solmix` weight (default: plain average) and takes
/// the element-wise MAXIMUM of the two `friction` vectors — so giving BOTH
/// sides the same preset makes the pair behave exactly as that preset.
///
/// # Where the preset numbers come from
/// Baseline: MuJoCo geom defaults are `solref=(0.02, 1.0)`,
/// `solimp=(0.9, 0.95, 0.001)`, `friction=(1.0, 0.005, 0.0001)`. Each preset
/// moves only the knobs its physical intuition justifies (sliding-friction
/// coefficients are standard dry-contact handbook ranges); everything is
/// critically damped (`dampratio = 1.0`) because bounce should come from
/// material softness, not solver ringing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactMaterial {
    /// Idealized hard contact: `solref=(0.002, 1.0)` — the stiffest timeconst
    /// that still satisfies `timeconst >= 2h` at the default `h = 1 ms`;
    /// `solimp=(0.95, 0.99, 0.001)` — near-unit impedance, penetration
    /// resolved within 1 mm; `friction=(1.0, 0.005, 0.0001)` — MuJoCo's
    /// default friction (rigidity says nothing about surface chemistry).
    Rigid,
    /// Compliant and grippy: `solref=(0.01, 1.0)` — 10 ms recovery, visibly
    /// softer than Rigid but still snappy; `solimp=(0.9, 0.95, 0.001)` —
    /// MuJoCo's default (medium-hard) impedance; `friction=(1.2, 0.01,
    /// 0.0002)` — dry rubber slides at μ ≈ 1.0–1.5 and its contact patch
    /// deforms, so torsional/rolling resistance doubles vs the default.
    Rubber,
    /// Very soft and dissipative: `solref=(0.04, 1.0)` — 40 ms recovery, the
    /// contact yields for several frames; `solimp=(0.5, 0.8, 0.01)` — low
    /// impedance ramping over a full centimeter = visible squish under load;
    /// `friction=(0.8, 0.005, 0.0001)` — slightly below default slide.
    Foam,
    /// Hard and slick: `solref=(0.002, 1.0)` and `solimp=(0.95, 0.99, 0.001)`
    /// as Rigid (steel IS the canonical rigid body at robot force scales);
    /// `friction=(0.4, 0.005, 0.0001)` — dry steel-on-steel slides at
    /// μ ≈ 0.4.
    Steel,
    /// Moderately hard: `solref=(0.005, 1.0)` — between Rigid and Rubber;
    /// `solimp=(0.9, 0.95, 0.002)` — default impedance over a 2 mm ramp
    /// (grain crush); `friction=(0.45, 0.005, 0.0001)` — dry wood-on-wood
    /// slides at μ ≈ 0.35–0.5.
    Wood,
    /// Raw passthrough for the user who knows the dark art. Validated at
    /// generation time: positive finite `solref`, `solimp` `d` values in
    /// `(0, 1)` with `dmin <= dmax` and `width > 0`, non-negative finite
    /// friction.
    Custom {
        /// `(timeconst, dampratio)` — see the enum docs.
        solref: (f64, f64),
        /// `(dmin, dmax, width)` — see the enum docs.
        solimp: (f64, f64, f64),
        /// `(slide, torsion, roll)` — see the enum docs.
        friction: (f64, f64, f64),
    },
}

impl ContactMaterial {
    /// `(timeconst, dampratio)` this material emits.
    pub fn solref(&self) -> (f64, f64) {
        match *self {
            Self::Rigid | Self::Steel => (0.002, 1.0),
            Self::Rubber => (0.01, 1.0),
            Self::Foam => (0.04, 1.0),
            Self::Wood => (0.005, 1.0),
            Self::Custom { solref, .. } => solref,
        }
    }

    /// `(dmin, dmax, width)` this material emits.
    pub fn solimp(&self) -> (f64, f64, f64) {
        match *self {
            Self::Rigid | Self::Steel => (0.95, 0.99, 0.001),
            Self::Rubber => (0.9, 0.95, 0.001),
            Self::Foam => (0.5, 0.8, 0.01),
            Self::Wood => (0.9, 0.95, 0.002),
            Self::Custom { solimp, .. } => solimp,
        }
    }

    /// `(slide, torsion, roll)` this material emits.
    pub fn friction(&self) -> (f64, f64, f64) {
        match *self {
            Self::Rigid => (1.0, 0.005, 0.0001),
            Self::Foam => (0.8, 0.005, 0.0001),
            Self::Rubber => (1.2, 0.01, 0.0002),
            Self::Steel => (0.4, 0.005, 0.0001),
            Self::Wood => (0.45, 0.005, 0.0001),
            Self::Custom { friction, .. } => friction,
        }
    }

    /// Reject a bad `Custom` BEFORE the MuJoCo compiler sees it (presets are
    /// correct by construction). `what` names the geom/prop for the error.
    fn validate(&self, what: &str) -> Result<(), MujocoError> {
        let (tc, dr) = self.solref();
        let (dmin, dmax, width) = self.solimp();
        let (fs, ft, fr) = self.friction();
        let bad = |msg: String| Err(MujocoError::Mjcf(format!("material on {what}: {msg}")));
        if !(tc.is_finite() && tc > 0.0 && dr.is_finite() && dr > 0.0) {
            return bad(format!(
                "solref (timeconst, dampratio) must be finite and > 0, got ({tc}, {dr})"
            ));
        }
        let imp_ok = |d: f64| d.is_finite() && d > 0.0 && d < 1.0;
        if !(imp_ok(dmin) && imp_ok(dmax) && dmin <= dmax && width.is_finite() && width > 0.0) {
            return bad(format!(
                "solimp needs 0 < dmin <= dmax < 1 and width > 0, got ({dmin}, {dmax}, {width})"
            ));
        }
        if !(fs.is_finite()
            && fs >= 0.0
            && ft.is_finite()
            && ft >= 0.0
            && fr.is_finite()
            && fr >= 0.0)
        {
            return bad(format!(
                "friction (slide, torsion, roll) must be finite and >= 0, got ({fs}, {ft}, {fr})"
            ));
        }
        Ok(())
    }

    /// ` solref="…" solimp="…" friction="…"` (leading space; appended to a
    /// `<geom>` attribute list).
    fn geom_attrs(&self) -> String {
        let (tc, dr) = self.solref();
        let (dmin, dmax, width) = self.solimp();
        let (fs, ft, fr) = self.friction();
        format!(
            " solref=\"{} {}\" solimp=\"{} {} {}\" friction=\"{} {} {}\"",
            f(tc),
            f(dr),
            f(dmin),
            f(dmax),
            f(width),
            f(fs),
            f(ft),
            f(fr)
        )
    }
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
    /// Contact material for THIS prop's geom; overrides
    /// [`MjcfOptions::default_material`]. `None` = inherit the default
    /// material (or MuJoCo defaults if there is none).
    pub material: Option<ContactMaterial>,
}

/// One convex piece of a decomposed collider, in the collider's own frame
/// (same frame as the input hull points).
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexPiece {
    /// Vertices of the piece; MuJoCo convex-hulls them, so >= 4 non-coplanar
    /// points are required (enforced at emission, exactly like plain hulls).
    pub points: Vec<Point3<f64>>,
}

/// The convex-decomposition SEAM: split one convex-hull collider into
/// several convex pieces so concave geometry (mugs, brackets, grippers) stops
/// colliding as its fat convex hull.
///
/// This crate deliberately ships NO real decomposition algorithm — the
/// intended future implementation is CoACD (Wei et al., SIGGRAPH 2022,
/// `github.com/SarahWeiii/CoACD`), plugged in from a heavier crate or an FFI
/// wrapper WITHOUT touching the generator: implement this trait, hand it to
/// [`MjcfOptions::hull_decomposer`], and every exported hull becomes one
/// `<mesh>` asset + `<geom>` per returned piece.
///
/// Contract:
/// - MUST be deterministic (pure function of the input points) — the
///   generator's "fixed options ⇒ fixed document" guarantee extends through
///   this seam. Seed any internal randomness from the input, never from time.
/// - Every returned piece needs >= 4 finite vertices, and the return must be
///   non-empty; violations fail generation loudly.
pub trait ColliderDecomposer: std::fmt::Debug + Send + Sync {
    /// Decompose one collider's hull points into convex pieces.
    fn decompose(&self, hull_points: &[Point3<f64>]) -> Vec<ConvexPiece>;
}

/// The identity decomposer: the single existing hull back, unchanged. With
/// this (or no decomposer at all) the generated document is byte-identical to
/// the pre-seam output — it exists so plumbing can be exercised end-to-end
/// before a real decomposer (CoACD) lands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NaiveDecomposer;

impl ColliderDecomposer for NaiveDecomposer {
    fn decompose(&self, hull_points: &[Point3<f64>]) -> Vec<ConvexPiece> {
        vec![ConvexPiece {
            points: hull_points.to_vec(),
        }]
    }
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
    /// Export [`CollisionShape::ConvexHull`] colliders as inline-vertex
    /// `<asset><mesh vertex="x y z …">` assets + `<geom type="mesh">` geoms.
    /// MuJoCo accepts vertex-only meshes embedded directly in the XML and
    /// convex-hulls them natively (qhull), which matches caliper's own GJK
    /// hull semantics — no mesh files are written. OFF by default (the
    /// pre-existing behavior: hulls are counted in
    /// [`MjcfDocument::skipped_hull_colliders`], never silently dropped);
    /// when ON, `skipped_hull_colliders` drops to 0. A hull with fewer than
    /// 4 vertices is rejected loudly (MuJoCo's hull needs a 3D simplex).
    pub export_hull_meshes: bool,
    /// Contact material stamped on EVERY emitted geom that has no more
    /// specific material: robot colliders, the ground plane, and props whose
    /// [`PropSpec::material`] is `None`. `None` (the default) emits no
    /// `solref`/`solimp`/`friction` attributes at all — plain MuJoCo
    /// defaults, byte-identical to the pre-material output.
    pub default_material: Option<ContactMaterial>,
    /// Convex-decomposition seam, consulted ONLY for hulls actually exported
    /// (i.e. when [`export_hull_meshes`](Self::export_hull_meshes) is on):
    /// each hull's points are decomposed and every piece becomes its own
    /// `<mesh>` asset + `<geom>`. `None` (the default) = one piece per hull,
    /// exactly like [`NaiveDecomposer`]. See [`ColliderDecomposer`] for the
    /// determinism contract and the CoACD pointer.
    pub hull_decomposer: Option<std::sync::Arc<dyn ColliderDecomposer>>,
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
            export_hull_meshes: false,
            default_material: None,
            hull_decomposer: None,
        }
    }
}

/// A generated MJCF document plus what the generator had to leave out.
#[derive(Clone, Debug)]
pub struct MjcfDocument {
    pub xml: String,
    /// Number of `CollisionShape::ConvexHull` colliders that were SKIPPED
    /// because [`MjcfOptions::export_hull_meshes`] was off (with it on, this
    /// is always 0). Non-zero means the MuJoCo model has LESS collision
    /// coverage than `caliper-collision` on the same model — surface this to
    /// users, never swallow it.
    pub skipped_hull_colliders: usize,
    /// `<body>` elements emitted: robot bodies (= ndof) + free-floating props.
    pub body_count: usize,
    /// Articulated robot `<joint>` elements (= ndof; prop `<freejoint>`s are
    /// NOT counted here).
    pub joint_count: usize,
    /// Collision `<geom>` elements emitted: robot colliders (incl. hull
    /// meshes when enabled) + prop geoms + the optional ground plane. Geoms
    /// inside [`MjcfOptions::extra_worldbody_xml`] are NOT counted (verbatim
    /// passthrough).
    pub geom_count: usize,
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

    if let Some(mat) = &opt.default_material {
        mat.validate("MjcfOptions::default_material")?;
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
    let mut hull_assets: Vec<String> = Vec::new();
    let mut skipped_hull_colliders = 0usize;
    let robot_mat = material_attrs(opt.default_material.as_ref());
    for (gi, g) in m.collision.iter().enumerate() {
        let frame = &m.frames[g.frame];
        let pose = frame.offset.compose(&g.origin);
        let name = format!("col{gi}_{}", sanitize(&frame.name));
        let mut emitted: Vec<String> = Vec::new();
        match geom_xml(&name, &pose, &g.shape, &robot_mat) {
            Some(x) => emitted.push(x),
            None if opt.export_hull_meshes => {
                let CollisionShape::ConvexHull { points } = &g.shape else {
                    unreachable!("geom_xml returns None only for ConvexHull");
                };
                // Decomposition seam: no decomposer = one piece (the hull
                // itself). A single piece keeps the pre-seam names, so the
                // identity path (None / NaiveDecomposer) is byte-identical.
                let pieces = match &opt.hull_decomposer {
                    Some(d) => d.decompose(points),
                    None => vec![ConvexPiece {
                        points: points.clone(),
                    }],
                };
                if pieces.is_empty() {
                    return Err(MujocoError::Mjcf(format!(
                        "decomposer returned no pieces for hull collider `{name}`"
                    )));
                }
                let multi = pieces.len() > 1;
                for (k, piece) in pieces.iter().enumerate() {
                    let gname = if multi {
                        format!("{name}_p{k}")
                    } else {
                        name.clone()
                    };
                    let mesh_name = format!("mesh_{gname}");
                    hull_assets.push(hull_mesh_asset(&mesh_name, &piece.points)?);
                    emitted.push(format!(
                        "<geom name=\"{gname}\" type=\"mesh\" mesh=\"{mesh_name}\"{robot_mat} {}/>",
                        pose_attrs(&pose)
                    ));
                }
            }
            None => {
                skipped_hull_colliders += 1;
            }
        }
        for x in emitted {
            match frame.anchor {
                Some(j) => body_geoms[j].push(x),
                None => world_geoms.push(x),
            }
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
    if !hull_assets.is_empty() {
        xml.push_str("  <asset>\n");
        for a in &hull_assets {
            let _ = writeln!(xml, "    {a}");
        }
        xml.push_str("  </asset>\n");
    }
    xml.push_str("  <worldbody>\n");
    if let Some(z) = opt.ground_plane {
        if !z.is_finite() {
            return Err(MujocoError::NonFinite {
                what: "ground_plane",
            });
        }
        let _ = writeln!(
            xml,
            "    <geom name=\"caliper_ground\" type=\"plane\" pos=\"0 0 {}\" size=\"5 5 0.1\"{robot_mat}/>",
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
        // Per-prop material wins over the document default.
        let mat = material_attrs(p.material.as_ref().or(opt.default_material.as_ref()));
        let _ = writeln!(
            xml,
            "      <geom name=\"{body}_geom\" {}{mat}{rgba}/>",
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

    let geom_count = usize::from(opt.ground_plane.is_some())
        + world_geoms.len()
        + body_geoms.iter().map(Vec::len).sum::<usize>()
        + opt.props.len();
    Ok(MjcfDocument {
        xml,
        skipped_hull_colliders,
        body_count: m.ndof + opt.props.len(),
        joint_count: m.ndof,
        geom_count,
        prop_bodies,
    })
}

/// One inline-vertex `<mesh>` asset from hull points. MuJoCo's `vertex`
/// attribute takes whitespace-separated coordinates (a multiple of 3) in the
/// mesh frame; with no `face` data MuJoCo builds the convex hull itself —
/// exactly caliper's hull semantics. Fails loudly on non-finite coordinates
/// or fewer than 4 vertices (a 3D hull needs a simplex; qhull would reject
/// it at load time with a far less helpful message).
fn hull_mesh_asset(name: &str, points: &[Point3<f64>]) -> Result<String, MujocoError> {
    if points.len() < 4 {
        return Err(MujocoError::Mjcf(format!(
            "hull mesh `{name}`: MuJoCo needs at least 4 vertices, got {}",
            points.len()
        )));
    }
    if !points
        .iter()
        .all(|p| p.coords.iter().all(|c| c.is_finite()))
    {
        return Err(MujocoError::NonFinite {
            what: "hull mesh vertex",
        });
    }
    let mut v = String::with_capacity(points.len() * 24);
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            v.push(' ');
        }
        let _ = write!(v, "{} {} {}", f(p.x), f(p.y), f(p.z));
    }
    Ok(format!("<mesh name=\"{name}\" vertex=\"{v}\"/>"))
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
    if let Some(mat) = &p.material {
        mat.validate(&format!("prop `{}`", p.name))?;
    }
    Ok(())
}

/// ` solref=… solimp=… friction=…` for a resolved material; empty for `None`
/// (MuJoCo defaults, no attributes — the pre-material output byte-for-byte).
fn material_attrs(mat: Option<&ContactMaterial>) -> String {
    mat.map(ContactMaterial::geom_attrs).unwrap_or_default()
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

/// Primitive collider → MJCF `<geom>`; `None` for `ConvexHull` — the caller
/// either exports it as a mesh asset ([`MjcfOptions::export_hull_meshes`]) or
/// COUNTS it as skipped, never drops it silently. `mat` is a pre-rendered
/// [`material_attrs`] string (empty or leading-space attributes).
fn geom_xml(name: &str, pose: &Se3, shape: &CollisionShape, mat: &str) -> Option<String> {
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
        "<geom name=\"{name}\" {body}{mat} {}/>",
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
                    material: None,
                },
                PropSpec {
                    name: "ball".into(),
                    shape: PropShape::Sphere { r: 0.05 },
                    pos: [0.0, 0.5, 0.2],
                    quat: Some([1.0, 0.0, 0.0, 0.0]),
                    mass: 0.1,
                    rgba: None,
                    material: None,
                },
                PropSpec {
                    name: "can".into(),
                    shape: PropShape::Cylinder { r: 0.04, h: 0.12 },
                    pos: [0.0, -0.5, 0.2],
                    quat: None,
                    mass: 0.2,
                    rgba: None,
                    material: None,
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
            material: None,
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

    /// A hull collider exports as an inline-vertex `<asset><mesh>` +
    /// `<geom type="mesh">` when `export_hull_meshes` is on (vertex data
    /// bit-exactly matching the model's hull points), and stays SKIPPED +
    /// counted when off. Pure string-level — no MuJoCo needed.
    #[test]
    fn hull_meshes_exported_when_enabled() {
        // Temp URDF referencing the shared unit-cube STL fixture (corners at
        // ±0.5) by ABSOLUTE path, plus an inertial so MJCF generation runs.
        let stl = format!(
            "{}/../../oracle/fixtures/robots/unit_cube.stl",
            env!("CARGO_MANIFEST_DIR")
        );
        let urdf = format!(
            r#"<?xml version="1.0"?>
<robot name="hullbot">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.001"/></inertial>
    <collision><origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry><mesh filename="{stl}"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
</robot>"#
        );
        let path =
            std::env::temp_dir().join(format!("caliper_mjcf_hull_{}.urdf", std::process::id()));
        std::fs::write(&path, urdf).unwrap();
        let m = Model::from_urdf(&path).unwrap();
        let points = m
            .collision
            .iter()
            .find_map(|g| match &g.shape {
                CollisionShape::ConvexHull { points } => Some(points.clone()),
                _ => None,
            })
            .expect("fixture must produce a ConvexHull collider");
        assert_eq!(points.len(), 8, "unit cube hull = 8 corners");

        // OFF (the default): skipped + counted, no mesh anywhere.
        let off = mjcf_from_model(&m, &MjcfOptions::default()).unwrap();
        assert_eq!(off.skipped_hull_colliders, 1);
        assert_eq!(off.geom_count, 0);
        assert!(!off.xml.contains("<asset>") && !off.xml.contains("type=\"mesh\""));

        // ON: <asset> before <worldbody>, geom references the mesh, skipped 0.
        let opt = MjcfOptions {
            export_hull_meshes: true,
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        assert_eq!(doc.skipped_hull_colliders, 0);
        assert_eq!(doc.body_count, 1);
        assert_eq!(doc.joint_count, 1);
        assert_eq!(doc.geom_count, 1);
        let x = &doc.xml;
        let asset_at = x.find("<asset>").expect("no <asset> block");
        assert!(asset_at < x.find("<worldbody>").unwrap());
        assert!(x.contains("<mesh name=\"mesh_col0_l1\" vertex=\""));
        assert!(
            x.contains("type=\"mesh\" mesh=\"mesh_col0_l1\""),
            "geom does not reference the mesh asset:\n{x}"
        );
        // Vertex data round-trips bit-exactly to the model's hull points
        // (shortest-roundtrip formatting is parse-exact).
        let vstart = x.find("vertex=\"").unwrap() + "vertex=\"".len();
        let vend = vstart + x[vstart..].find('"').unwrap();
        let nums: Vec<f64> = x[vstart..vend]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(nums.len(), 3 * points.len());
        for (i, p) in points.iter().enumerate() {
            for (k, c) in [p.x, p.y, p.z].iter().enumerate() {
                assert_eq!(
                    nums[3 * i + k].to_bits(),
                    c.to_bits(),
                    "vertex {i} coord {k} did not round-trip"
                );
            }
        }
        // Determinism: same options, same document.
        assert_eq!(mjcf_from_model(&m, &opt).unwrap().xml, doc.xml);
    }

    #[test]
    fn hull_mesh_asset_rejects_degenerate() {
        let tri = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(hull_mesh_asset("m", &tri).is_err(), "3 vertices accepted");
        let mut quad = tri.clone();
        quad.push(Point3::new(0.0, 0.0, f64::NAN));
        assert!(hull_mesh_asset("m", &quad).is_err(), "NaN vertex accepted");
    }

    #[test]
    fn counts_reported() {
        let m = model("collide_arm.urdf");
        let opt = MjcfOptions {
            ground_plane: Some(0.0),
            props: vec![PropSpec {
                name: "ball".into(),
                shape: PropShape::Sphere { r: 0.05 },
                pos: [0.5, 0.0, 0.3],
                quat: None,
                mass: 0.1,
                rgba: None,
                material: None,
            }],
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        assert_eq!(doc.joint_count, m.ndof);
        assert_eq!(doc.body_count, m.ndof + 1); // + 1 prop
        // 3 box colliders + ground plane + 1 prop geom
        assert_eq!(doc.geom_count, 3 + 1 + 1);
        assert_eq!(doc.geom_count, doc.xml.matches("<geom ").count());
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

    // ---- contact materials ----

    /// Each preset stamps its EXACT attribute set on every emitted geom —
    /// the numbers here are the documented derivations, spelled out so a
    /// preset can never drift silently.
    #[test]
    fn material_presets_emit_exact_attributes() {
        let m = model("collide_arm.urdf"); // 3 box colliders
        for (mat, attrs) in [
            (
                ContactMaterial::Rigid,
                " solref=\"0.002 1.0\" solimp=\"0.95 0.99 0.001\" friction=\"1.0 0.005 0.0001\"",
            ),
            (
                ContactMaterial::Rubber,
                " solref=\"0.01 1.0\" solimp=\"0.9 0.95 0.001\" friction=\"1.2 0.01 0.0002\"",
            ),
            (
                ContactMaterial::Foam,
                " solref=\"0.04 1.0\" solimp=\"0.5 0.8 0.01\" friction=\"0.8 0.005 0.0001\"",
            ),
            (
                ContactMaterial::Steel,
                " solref=\"0.002 1.0\" solimp=\"0.95 0.99 0.001\" friction=\"0.4 0.005 0.0001\"",
            ),
            (
                ContactMaterial::Wood,
                " solref=\"0.005 1.0\" solimp=\"0.9 0.95 0.002\" friction=\"0.45 0.005 0.0001\"",
            ),
        ] {
            let opt = MjcfOptions {
                ground_plane: Some(0.0),
                default_material: Some(mat),
                ..Default::default()
            };
            let doc = mjcf_from_model(&m, &opt).unwrap();
            let x = &doc.xml;
            // every geom (3 colliders + the ground plane) carries the full set
            assert_eq!(
                x.matches(attrs).count(),
                4,
                "{mat:?}: expected `{attrs}` on all 4 geoms in:\n{x}"
            );
            assert_eq!(x.matches("solref=").count(), 4, "{mat:?}");
            // determinism: same options, same document
            assert_eq!(mjcf_from_model(&m, &opt).unwrap().xml, doc.xml);
        }
    }

    #[test]
    fn custom_material_passes_through_verbatim() {
        let m = model("collide_arm.urdf");
        let opt = MjcfOptions {
            default_material: Some(ContactMaterial::Custom {
                solref: (0.025, 0.7),
                solimp: (0.6, 0.85, 0.003),
                friction: (0.33, 0.007, 0.00025),
            }),
            ..Default::default()
        };
        let x = mjcf_from_model(&m, &opt).unwrap().xml;
        assert_eq!(
            x.matches(
                " solref=\"0.025 0.7\" solimp=\"0.6 0.85 0.003\" friction=\"0.33 0.007 0.00025\""
            )
            .count(),
            3,
            "{x}"
        );
    }

    /// No material anywhere = MuJoCo defaults: not a single solver attribute
    /// in the document (byte-compatible with the pre-material generator).
    #[test]
    fn no_material_emits_no_attributes() {
        let m = model("collide_arm.urdf");
        let opt = MjcfOptions {
            ground_plane: Some(0.0),
            ..Default::default()
        };
        let x = mjcf_from_model(&m, &opt).unwrap().xml;
        for a in ["solref", "solimp", "friction"] {
            assert!(!x.contains(a), "unexpected `{a}` in:\n{x}");
        }
    }

    /// A prop's own material beats the document default; a prop WITHOUT one
    /// inherits the default.
    #[test]
    fn prop_material_overrides_default() {
        let m = model("dyn_pendulum2.urdf");
        let sphere = |name: &str, material: Option<ContactMaterial>| PropSpec {
            name: name.into(),
            shape: PropShape::Sphere { r: 0.05 },
            pos: [0.5, 0.0, 0.3],
            quat: None,
            mass: 0.1,
            rgba: None,
            material,
        };
        let opt = MjcfOptions {
            ground_plane: Some(0.0),
            default_material: Some(ContactMaterial::Foam),
            props: vec![
                sphere("soft", None),
                sphere("hard", Some(ContactMaterial::Steel)),
            ],
            ..Default::default()
        };
        let x = mjcf_from_model(&m, &opt).unwrap().xml;
        // `geom_named` matches the exact geom `name=`; props are emitted as
        // `<prop>_geom`, the ground plane as the bare `caliper_ground`.
        let geom_named = |name: &str| {
            x.lines()
                .find(|l| l.contains(&format!("name=\"{name}\"")) && l.contains("<geom"))
                .unwrap_or_else(|| panic!("no geom line for {name} in:\n{x}"))
                .to_string()
        };
        assert!(
            geom_named("prop_soft_geom").contains("solref=\"0.04 1.0\""),
            "{x}"
        );
        assert!(
            geom_named("prop_hard_geom").contains("solref=\"0.002 1.0\""),
            "{x}"
        );
        assert!(
            geom_named("prop_hard_geom").contains("friction=\"0.4 0.005 0.0001\""),
            "{x}"
        );
        // the ground plane inherits the default (Foam)
        assert!(
            geom_named("caliper_ground").contains("solimp=\"0.5 0.8 0.01\""),
            "{x}"
        );
    }

    #[test]
    fn bad_custom_material_rejected() {
        let m = model("dyn_pendulum2.urdf");
        let with =
            |solref: (f64, f64), solimp: (f64, f64, f64), friction: (f64, f64, f64)| MjcfOptions {
                default_material: Some(ContactMaterial::Custom {
                    solref,
                    solimp,
                    friction,
                }),
                ..Default::default()
            };
        let ok_ref = (0.02, 1.0);
        let ok_imp = (0.9, 0.95, 0.001);
        let ok_fric = (1.0, 0.005, 0.0001);
        for opt in [
            with((0.0, 1.0), ok_imp, ok_fric),           // zero timeconst
            with((f64::NAN, 1.0), ok_imp, ok_fric),      // NaN timeconst
            with((0.02, 0.0), ok_imp, ok_fric),          // zero dampratio
            with(ok_ref, (0.0, 0.95, 0.001), ok_fric),   // dmin out of (0,1)
            with(ok_ref, (0.9, 1.0, 0.001), ok_fric),    // dmax out of (0,1)
            with(ok_ref, (0.95, 0.9, 0.001), ok_fric),   // dmin > dmax
            with(ok_ref, (0.9, 0.95, 0.0), ok_fric),     // zero width
            with(ok_ref, ok_imp, (-0.1, 0.005, 0.0001)), // negative slide
            with(ok_ref, ok_imp, (1.0, f64::INFINITY, 0.0001)), // inf torsion
        ] {
            assert!(mjcf_from_model(&m, &opt).is_err(), "accepted {opt:?}");
        }
        // the same bad material on a PROP is rejected too
        let opt = MjcfOptions {
            props: vec![PropSpec {
                name: "p".into(),
                shape: PropShape::Sphere { r: 0.05 },
                pos: [0.0, 0.0, 0.5],
                quat: None,
                mass: 0.1,
                rgba: None,
                material: Some(ContactMaterial::Custom {
                    solref: (-0.01, 1.0),
                    solimp: ok_imp,
                    friction: ok_fric,
                }),
            }],
            ..Default::default()
        };
        assert!(
            mjcf_from_model(&m, &opt).is_err(),
            "accepted bad prop material"
        );
    }

    // ---- convex decomposition seam ----

    /// Temp URDF whose only collider is the shared unit-cube STL → a
    /// ConvexHull collider (8 corners at ±0.5). `tag` keeps per-test temp
    /// files distinct.
    fn hull_model(tag: &str) -> Model {
        let stl = format!(
            "{}/../../oracle/fixtures/robots/unit_cube.stl",
            env!("CARGO_MANIFEST_DIR")
        );
        let urdf = format!(
            r#"<?xml version="1.0"?>
<robot name="hullbot">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.001"/></inertial>
    <collision><origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry><mesh filename="{stl}"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
</robot>"#
        );
        let path = std::env::temp_dir().join(format!(
            "caliper_mjcf_decomp_{tag}_{}.urdf",
            std::process::id()
        ));
        std::fs::write(&path, urdf).unwrap();
        Model::from_urdf(&path).unwrap()
    }

    /// The identity decomposer changes NOTHING: document byte-identical to a
    /// plain hull export with no decomposer at all.
    #[test]
    fn naive_decomposer_is_identity() {
        let m = hull_model("naive");
        let plain = MjcfOptions {
            export_hull_meshes: true,
            ..Default::default()
        };
        let naive = MjcfOptions {
            export_hull_meshes: true,
            hull_decomposer: Some(Arc::new(NaiveDecomposer)),
            ..Default::default()
        };
        let a = mjcf_from_model(&m, &plain).unwrap();
        let b = mjcf_from_model(&m, &naive).unwrap();
        assert_eq!(a.xml, b.xml, "NaiveDecomposer must be a byte-level no-op");
        assert_eq!(b.skipped_hull_colliders, 0);
        assert_eq!(b.geom_count, 1);
    }

    /// A multi-piece decomposer emits one `<mesh>` asset + one `<geom>` per
    /// piece, with `_p{k}` suffixes; counts follow the pieces.
    #[test]
    fn multi_piece_decomposer_emits_per_piece() {
        /// Splits the cube hull into its bottom and top halves (4 corners
        /// each — passes the generator's >= 4-vertex gate; this is a
        /// STRING-level test, MuJoCo never loads it) — a deterministic pure
        /// function of the input, per the trait contract.
        #[derive(Debug)]
        struct SplitZ;
        impl ColliderDecomposer for SplitZ {
            fn decompose(&self, hull_points: &[Point3<f64>]) -> Vec<ConvexPiece> {
                let (lo, hi): (Vec<_>, Vec<_>) =
                    hull_points.iter().copied().partition(|p| p.z < 0.0);
                vec![ConvexPiece { points: lo }, ConvexPiece { points: hi }]
            }
        }
        let m = hull_model("split");
        let opt = MjcfOptions {
            export_hull_meshes: true,
            hull_decomposer: Some(Arc::new(SplitZ)),
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        let x = &doc.xml;
        assert_eq!(doc.skipped_hull_colliders, 0);
        assert_eq!(doc.geom_count, 2, "{x}");
        assert_eq!(x.matches("<mesh name=").count(), 2, "{x}");
        assert!(
            x.contains("<mesh name=\"mesh_col0_l1_p0\" vertex=\""),
            "{x}"
        );
        assert!(
            x.contains("<mesh name=\"mesh_col0_l1_p1\" vertex=\""),
            "{x}"
        );
        assert!(
            x.contains("<geom name=\"col0_l1_p0\" type=\"mesh\" mesh=\"mesh_col0_l1_p0\""),
            "{x}"
        );
        assert!(
            x.contains("<geom name=\"col0_l1_p1\" type=\"mesh\" mesh=\"mesh_col0_l1_p1\""),
            "{x}"
        );
        // both pieces share the collider's pose
        assert_eq!(x.matches("pos=\"0.0 0.0 0.2\"").count(), 2, "{x}");
        // determinism through the seam: same options, same document
        assert_eq!(mjcf_from_model(&m, &opt).unwrap().xml, doc.xml);
        // materials stamp decomposed pieces like any other geom
        let with_mat = MjcfOptions {
            default_material: Some(ContactMaterial::Rubber),
            ..opt.clone()
        };
        let xm = mjcf_from_model(&m, &with_mat).unwrap().xml;
        assert_eq!(xm.matches("solref=\"0.01 1.0\"").count(), 2, "{xm}");
    }

    /// Bad decomposer output fails generation loudly: empty piece lists and
    /// sub-simplex pieces are both rejected.
    #[test]
    fn bad_decomposer_output_rejected() {
        #[derive(Debug)]
        struct Empty;
        impl ColliderDecomposer for Empty {
            fn decompose(&self, _: &[Point3<f64>]) -> Vec<ConvexPiece> {
                Vec::new()
            }
        }
        #[derive(Debug)]
        struct Degenerate;
        impl ColliderDecomposer for Degenerate {
            fn decompose(&self, hull_points: &[Point3<f64>]) -> Vec<ConvexPiece> {
                vec![ConvexPiece {
                    points: hull_points[..3].to_vec(), // 3 < 4: no 3D simplex
                }]
            }
        }
        let m = hull_model("bad");
        for d in [
            Arc::new(Empty) as Arc<dyn ColliderDecomposer>,
            Arc::new(Degenerate),
        ] {
            let opt = MjcfOptions {
                export_hull_meshes: true,
                hull_decomposer: Some(d),
                ..Default::default()
            };
            assert!(mjcf_from_model(&m, &opt).is_err(), "accepted bad pieces");
        }
    }

    /// With `export_hull_meshes` OFF the decomposer is never consulted: the
    /// hull stays skipped-and-counted exactly as before.
    #[test]
    fn decomposer_ignored_when_hulls_not_exported() {
        #[derive(Debug)]
        struct Panics;
        impl ColliderDecomposer for Panics {
            fn decompose(&self, _: &[Point3<f64>]) -> Vec<ConvexPiece> {
                panic!("decomposer must not run when hulls are not exported");
            }
        }
        let m = hull_model("off");
        let opt = MjcfOptions {
            hull_decomposer: Some(Arc::new(Panics)),
            ..Default::default()
        };
        let doc = mjcf_from_model(&m, &opt).unwrap();
        assert_eq!(doc.skipped_hull_colliders, 1);
        assert!(!doc.xml.contains("<asset>"));
    }
}
