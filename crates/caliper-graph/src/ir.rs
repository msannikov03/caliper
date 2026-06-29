//! The serde-serializable graph intermediate representation (IR).
//!
//! This module IS the persisted `.caliper-graph.json` schema. JSON shape:
//!
//! ```json
//! {
//!   "nodes": [
//!     { "id": "start", "kind": { "type": "startConfig", "q": [0,0,0,0,0,0] } },
//!     { "id": "goal",  "kind": { "type": "goalPose", "m": [/* 16, col-major SE3 */] } },
//!     { "id": "ik",    "kind": { "type": "ik", "frame": null, "seed": null } },
//!     { "id": "mv",    "kind": { "type": "moveL", "frame": null } },
//!     { "id": "view",  "kind": { "type": "view" } }
//!   ],
//!   "edges": [
//!     { "from": "start", "fromPort": "config", "to": "mv", "toPort": "start" },
//!     { "from": "goal",  "fromPort": "pose",   "to": "ik", "toPort": "pose" },
//!     { "from": "ik",    "fromPort": "config", "to": "mv", "toPort": "start" },
//!     { "from": "mv",    "fromPort": "clip",   "to": "view", "toPort": "clip" }
//!   ],
//!   "metadata": { "name": "demo" }
//! }
//! ```
//!
//! Naming convention: all field names are **camelCase**; the `NodeKind` /
//! `PortValue` discriminants use a `"type"` tag (also camelCase values).
//!
//! Port addressing: every node exposes named, ordered input and output ports (see
//! [`NodeKind::in_ports`] / [`NodeKind::out_ports`]). An [`Edge`]'s `fromPort` /
//! `toPort` is a [`PortRef`] that is EITHER the port name (string) OR its
//! positional index (integer); both forms resolve to the same port. They default
//! to index `0` so single-port edges may omit them.

use serde::{Deserialize, Serialize};

/// The wire type carried by a port / an edge.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PortType {
    /// A joint configuration `q` (length = `model.ndof`).
    Config,
    /// A Cartesian pose (SE3 as a 16-element column-major homogeneous matrix).
    Pose,
    /// A baked, face-neutral trajectory ([`ClipData`]).
    Clip,
    /// A structured analysis result ([`ReportData`]).
    Report,
}

/// One node's behaviour + its inline parameters.
///
/// Variants group into SOURCES (no inputs), COMPUTE (dispatch to an existing
/// engine fn), and SINKS (no outputs). See [`in_ports`](Self::in_ports) /
/// [`out_ports`](Self::out_ports) for each variant's named ports.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum NodeKind {
    // ---- sources ----
    /// Emit a fixed start configuration. out: `config: Config`.
    StartConfig { q: Vec<f64> },
    /// Emit a fixed goal pose (`m` = 16 col-major SE3). out: `pose: Pose`.
    GoalPose { m: [f64; 16] },
    /// Emit a named configuration the CALLER already resolved to `q`
    /// (convenience source). out: `config: Config`.
    NamedConfig { q: Vec<f64>, name: String },

    // ---- compute (dispatch to an existing engine fn) ----
    /// Inverse kinematics → `caliper_ik::ik`. in: `pose: Pose` (req),
    /// `seed: Config` (opt; falls back to the `seed` param, else zeros).
    /// out: `config: Config`.
    Ik {
        #[serde(default)]
        frame: Option<String>,
        #[serde(default)]
        seed: Option<Vec<f64>>,
    },
    /// Jerk-limited joint move → `caliper_motion::move_j`. in: `start: Config`,
    /// `goal: Config`. out: `clip: Clip`.
    MoveJ {},
    /// Cartesian straight-line move → `caliper_motion::move_l`. in: `start:
    /// Config`, `goal: Pose`. out: `clip: Clip`.
    MoveL {
        #[serde(default)]
        frame: Option<String>,
    },
    /// Collision-aware RRT-Connect plan → `caliper_planning::Planner`, retimed via
    /// `caliper_motion::retime_waypoints`. in: `start: Config`, `goal:
    /// Config|Pose`. out: `clip: Clip`. Deterministic in `seed`.
    PlanRrt {
        seed: u64,
        #[serde(default)]
        ground: Option<f64>,
        #[serde(default)]
        boxes: Vec<([f64; 3], [f64; 3])>,
    },
    /// Computed-torque control rollout → `caliper_hal::ControlLoop` over a
    /// `PhysicsSimBackend`. in: `start: Config`, `goal: Config`. out: `clip:
    /// Clip`. Requires `model.has_inertia`.
    Control { kp: f64, kd: f64 },
    /// Passive/forced gravity drop → `caliper_dynamics::Simulator`. in: `start:
    /// Config`. out: `clip: Clip`. Requires `model.has_inertia`.
    GravityDrop {
        #[serde(default)]
        gravity: Option<[f64; 3]>,
        duration: f64,
        dt: f64,
    },
    /// Self/world collision query → `caliper_collision::CollisionModel`. in:
    /// `config: Config`. out: `report: Report`.
    CollisionCheck {
        #[serde(default)]
        ground: Option<f64>,
        #[serde(default)]
        boxes: Vec<([f64; 3], [f64; 3])>,
    },

    // ---- sinks ----
    /// Mark the upstream clip as the graph's terminal output. in: `clip: Clip`.
    View {},
    /// Extract a 1-D signal-vs-time series from a clip. in: `clip: Clip`.
    /// `signal` ∈ { `q<i>`, `qd<i>`, `tip_x`, `tip_y`, `tip_z`, `energy`, `t` }.
    Scope { signal: String },
}

/// A graph node: a unique `id` plus its [`NodeKind`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
}

/// A reference to a port — by NAME (string) or positional INDEX (integer).
/// Defaults to index `0`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PortRef {
    Index(usize),
    Name(String),
}

impl Default for PortRef {
    fn default() -> Self {
        PortRef::Index(0)
    }
}

impl PortRef {
    /// Resolve this reference against an ordered list of port names, returning the
    /// port index (or `None` if out of range / no matching name).
    pub fn resolve(&self, names: &[&str]) -> Option<usize> {
        match self {
            PortRef::Index(i) => (*i < names.len()).then_some(*i),
            PortRef::Name(n) => names.iter().position(|p| p == n),
        }
    }
}

/// A directed edge `from.fromPort → to.toPort`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub from: String,
    #[serde(default)]
    pub from_port: PortRef,
    pub to: String,
    #[serde(default)]
    pub to_port: PortRef,
}

/// Optional, free-form graph metadata (never affects execution).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GraphMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

/// The whole persisted document — what save/load (de)serializes.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GraphDoc {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(default)]
    pub metadata: GraphMeta,
}

// ===== runtime values flowing along edges =====

/// A baked, face-neutral trajectory: uniform-ish time samples with `q`/`qd` rows.
/// No FK frames here — faces bake their own render frames (this crate is
/// render-agnostic). `times.len() == qs.len() == qds.len()`; each row's length =
/// `model.ndof`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ClipData {
    pub times: Vec<f64>,
    pub qs: Vec<Vec<f64>>,
    pub qds: Vec<Vec<f64>>,
}

impl ClipData {
    pub fn len(&self) -> usize {
        self.times.len()
    }
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// A structured collision report (the [`NodeKind::CollisionCheck`] output).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReportData {
    pub collision: bool,
    /// Self-colliding `(frame_a, frame_b)` pairs (`a < b`).
    pub pairs: Vec<(usize, usize)>,
    /// Frames intersecting world geometry.
    pub world_hits: Vec<usize>,
    /// Union of all frames involved in any collision.
    pub colliding_frames: Vec<usize>,
    /// Frames NOT fully collision-covered (mesh/none) — collisions there are NOT
    /// detected; surface this rather than trusting a "clear" verdict blindly.
    pub uncovered_frames: usize,
}

/// A value cached per (node, out-port) and resolved as a node's input.
/// Externally tagged (`{ "config": [...] }`, `{ "clip": { ... } }`, ...).
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum PortValue {
    Config(Vec<f64>),
    /// 16-element column-major SE3 homogeneous matrix.
    Pose([f64; 16]),
    Clip(ClipData),
    Report(ReportData),
}

impl PortValue {
    pub fn port_type(&self) -> PortType {
        match self {
            PortValue::Config(_) => PortType::Config,
            PortValue::Pose(_) => PortType::Pose,
            PortValue::Clip(_) => PortType::Clip,
            PortValue::Report(_) => PortType::Report,
        }
    }
    pub fn as_config(&self) -> Option<&Vec<f64>> {
        match self {
            PortValue::Config(q) => Some(q),
            _ => None,
        }
    }
    pub fn as_pose(&self) -> Option<&[f64; 16]> {
        match self {
            PortValue::Pose(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_clip(&self) -> Option<&ClipData> {
        match self {
            PortValue::Clip(c) => Some(c),
            _ => None,
        }
    }
}

// ===== port specifications =====

/// One output port (always exactly one wire type).
#[derive(Clone, Copy, Debug)]
pub struct OutPort {
    pub name: &'static str,
    pub ty: PortType,
}

/// One input port (accepts one or more wire types; PlanRrt's goal is a union).
#[derive(Clone, Copy, Debug)]
pub struct InPort {
    pub name: &'static str,
    pub types: &'static [PortType],
    pub required: bool,
}

const CONFIG: &[PortType] = &[PortType::Config];
const POSE: &[PortType] = &[PortType::Pose];
const CLIP: &[PortType] = &[PortType::Clip];
const CONFIG_OR_POSE: &[PortType] = &[PortType::Config, PortType::Pose];

impl NodeKind {
    /// The stable discriminant string (matches the serde `"type"` tag).
    pub fn type_name(&self) -> &'static str {
        match self {
            NodeKind::StartConfig { .. } => "startConfig",
            NodeKind::GoalPose { .. } => "goalPose",
            NodeKind::NamedConfig { .. } => "namedConfig",
            NodeKind::Ik { .. } => "ik",
            NodeKind::MoveJ {} => "moveJ",
            NodeKind::MoveL { .. } => "moveL",
            NodeKind::PlanRrt { .. } => "planRrt",
            NodeKind::Control { .. } => "control",
            NodeKind::GravityDrop { .. } => "gravityDrop",
            NodeKind::CollisionCheck { .. } => "collisionCheck",
            NodeKind::View {} => "view",
            NodeKind::Scope { .. } => "scope",
        }
    }

    /// Ordered input ports.
    pub fn in_ports(&self) -> Vec<InPort> {
        let i = |name, types, required| InPort {
            name,
            types,
            required,
        };
        match self {
            NodeKind::StartConfig { .. }
            | NodeKind::GoalPose { .. }
            | NodeKind::NamedConfig { .. } => vec![],
            NodeKind::Ik { .. } => vec![i("pose", POSE, true), i("seed", CONFIG, false)],
            NodeKind::MoveJ {} => vec![i("start", CONFIG, true), i("goal", CONFIG, true)],
            NodeKind::MoveL { .. } => vec![i("start", CONFIG, true), i("goal", POSE, true)],
            NodeKind::PlanRrt { .. } => {
                vec![i("start", CONFIG, true), i("goal", CONFIG_OR_POSE, true)]
            }
            NodeKind::Control { .. } => vec![i("start", CONFIG, true), i("goal", CONFIG, true)],
            NodeKind::GravityDrop { .. } => vec![i("start", CONFIG, true)],
            NodeKind::CollisionCheck { .. } => vec![i("config", CONFIG, true)],
            NodeKind::View {} => vec![i("clip", CLIP, true)],
            NodeKind::Scope { .. } => vec![i("clip", CLIP, true)],
        }
    }

    /// Ordered output ports.
    pub fn out_ports(&self) -> Vec<OutPort> {
        let o = |name, ty| OutPort { name, ty };
        match self {
            NodeKind::StartConfig { .. } | NodeKind::NamedConfig { .. } => {
                vec![o("config", PortType::Config)]
            }
            NodeKind::GoalPose { .. } => vec![o("pose", PortType::Pose)],
            NodeKind::Ik { .. } => vec![o("config", PortType::Config)],
            NodeKind::MoveJ {}
            | NodeKind::MoveL { .. }
            | NodeKind::PlanRrt { .. }
            | NodeKind::Control { .. }
            | NodeKind::GravityDrop { .. } => vec![o("clip", PortType::Clip)],
            NodeKind::CollisionCheck { .. } => vec![o("report", PortType::Report)],
            NodeKind::View {} | NodeKind::Scope { .. } => vec![],
        }
    }

    /// Input-port names, in order (for [`PortRef::resolve`]).
    pub fn in_port_names(&self) -> Vec<&'static str> {
        self.in_ports().into_iter().map(|p| p.name).collect()
    }
    /// Output-port names, in order (for [`PortRef::resolve`]).
    pub fn out_port_names(&self) -> Vec<&'static str> {
        self.out_ports().into_iter().map(|p| p.name).collect()
    }
}

// ===== SE3 <-> 16-element column-major homogeneous matrix =====

use caliper_spatial::Se3;
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

/// `Se3` → 16-element **column-major** homogeneous matrix (the `Pose` wire form).
pub fn se3_to_pose(t: &Se3) -> [f64; 16] {
    let h: Matrix4<f64> = t.0.to_homogeneous();
    let mut m = [0.0_f64; 16];
    m.copy_from_slice(h.as_slice()); // nalgebra storage is column-major
    m
}

/// 16-element **column-major** homogeneous matrix → `Se3` (rotation re-projected
/// onto SO(3); the supplied basis may be slightly non-orthonormal).
pub fn pose_to_se3(m: &[f64; 16]) -> Se3 {
    let h = Matrix4::from_column_slice(m);
    let rot = h.fixed_view::<3, 3>(0, 0).into_owned();
    let trans = Vector3::new(h[(0, 3)], h[(1, 3)], h[(2, 3)]);
    Se3::from_parts(trans, UnitQuaternion::from_matrix(&rot))
}
