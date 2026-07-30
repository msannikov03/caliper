//! `*.caliper-task.json` — a manipulation TASK as a versioned file.
//!
//! A task artifact is the small, diffable thing that turns "drive the arm
//! around" into "do THIS, and here is what counts as done": which robot, the
//! scene it acts on (free props with contact materials, plus target ZONES that
//! exist only for the evaluator and the renderer), which joint is the gripper,
//! and the success criterion — the very same predicate schema
//! `caliper_learn.success` reads, so one file scores a Studio teleop take, a
//! sim rollout and an eval report identically.
//!
//! Always compiled, exactly like [`crate::mjcf`]: reading, validating and
//! judging a task needs no MuJoCo at all. Only RUNNING one does.
//!
//! ```text
//! {
//!   "version": 1,
//!   "name": "pick-cube",
//!   "robot": "so101.urdf",              // relative to the task file, or absolute
//!   "q0": [0, 0, 0, 0, 0, 0],           // optional start pose
//!   "scene": {
//!     "ground": 0.0,                    // optional, default 0.0
//!     "props": [ {"name": "cube", "kind": "box", "halfExtents": [.025,.025,.025],
//!                 "pos": [0.3,0,0.025], "mass": 0.1, "material": "wood"} ],
//!     "zones": [ {"name": "bin", "center": [0.4,0.2,0.02], "half": [.05,.05,.02]} ]
//!   },
//!   "gripper": {"joint": "6", "closed": "lo"},   // optional override
//!   "success": {"kind": "placed_in_zone", "prop": "cube", "zone": "bin",
//!               "settled_speed": 0.01},          // optional
//!   "horizonS": 20.0,                            // optional
//!   "fps": 50                                    // optional recording rate
//! }
//! ```
//!
//! # The rules, and why they are strict
//!
//! - `version` must be [`TASK_VERSION`]. A future file is refused, never
//!   guessed at.
//! - UNKNOWN KEYS RAISE — at the top level and inside every nested object.
//!   A `"settle_speed"` typo that silently dropped the settle requirement, or
//!   a `"halfExtent"` that silently produced a default-sized box, is exactly
//!   the class of bug this file format exists to make impossible. Same rule,
//!   same reason, as `caliper_learn.success.from_dict`.
//! - Props reuse the engine's own prop vocabulary
//!   ([`crate::mjcf::PropSpec`]): kinds `box` / `sphere` / `cylinder`,
//!   quaternions w-first, and a `material` that is either a preset NAME or the
//!   raw `{solref, solimp, friction}` knobs — the same two forms the python
//!   `material=` kwarg takes. Dimensions, mass and materials are validated by
//!   the generator's own [`crate::mjcf`] rulebook, not by a second copy of it.
//! - ZONES are evaluator-side only. MJCF has nothing to emit for them (they
//!   are not bodies and they never collide); `rgba` is a render hint.
//! - Inside `success`, a zone written as a STRING is resolved by name against
//!   `scene.zones` when the file loads, and the resolved form is what
//!   serializes back out — so a task file may name a zone once and a predicate
//!   handed to any other layer always carries real numbers.

pub mod success;

use crate::MujocoError;
use crate::mjcf::{ContactMaterial, PropShape, PropSpec, mujoco_name, validate_prop};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use success::{Predicate, SuccessTracker, Zone, ZoneRef};

/// The only schema version this crate reads or writes.
pub const TASK_VERSION: u32 = 1;

/// Prop mass used when the file omits one (kg) — the same default the Studio
/// prop editor applies.
pub const DEFAULT_PROP_MASS: f64 = 0.1;

/// Everything that can go wrong reading, validating or judging a task.
#[derive(thiserror::Error, Debug)]
pub enum TaskError {
    #[error("reading task file `{path}`: {detail}")]
    Io { path: String, detail: String },
    /// The JSON itself is wrong — malformed, or carrying a key/type the schema
    /// does not have. `0` names where it came from.
    #[error("not a valid caliper task file — {0}")]
    Parse(String),
    /// The file parsed but says something impossible.
    #[error("{0}")]
    Invalid(String),
    /// A predicate could not be answered from the state it was handed — a
    /// missing prop, or a settled check with no velocities. Never a `false`.
    #[error("{0}")]
    State(String),
}

impl From<MujocoError> for TaskError {
    fn from(e: MujocoError) -> Self {
        TaskError::Invalid(e.to_string())
    }
}

fn invalid(msg: impl Into<String>) -> TaskError {
    TaskError::Invalid(msg.into())
}

// ===== contact material at the file boundary =====

/// A contact material as a task file spells it: a preset NAME
/// (`"rigid" | "rubber" | "foam" | "steel" | "wood"`, case-insensitive) or the
/// raw MuJoCo knobs `{"solref": [timeconst, dampratio], "solimp": [dmin, dmax,
/// width], "friction": [slide, torsion, roll]}` — the exact two forms
/// `caliper.mjcf(..., material=...)` accepts in python.
///
/// SHAPE errors (unknown preset, missing/unknown/short keys) are rejected here;
/// VALUE errors (non-finite, out of range) are rejected by
/// [`ContactMaterial`]'s own validation, so both faces share one rulebook.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MaterialSpec {
    Preset(String),
    Custom(CustomMaterial),
}

/// The raw per-geom solver knobs. See [`ContactMaterial::Custom`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomMaterial {
    pub solref: [f64; 2],
    pub solimp: [f64; 3],
    pub friction: [f64; 3],
}

impl MaterialSpec {
    /// The engine material this spec names. Preset names are matched
    /// case-insensitively, exactly like the python face.
    pub fn to_contact_material(&self) -> Result<ContactMaterial, TaskError> {
        match self {
            MaterialSpec::Preset(name) => match name.to_ascii_lowercase().as_str() {
                "rigid" => Ok(ContactMaterial::Rigid),
                "rubber" => Ok(ContactMaterial::Rubber),
                "foam" => Ok(ContactMaterial::Foam),
                "steel" => Ok(ContactMaterial::Steel),
                "wood" => Ok(ContactMaterial::Wood),
                other => Err(invalid(format!(
                    "unknown material preset `{other}` — expected one of rigid, rubber, \
                     foam, steel, wood, or a custom object {{solref, solimp, friction}}"
                ))),
            },
            MaterialSpec::Custom(c) => Ok(ContactMaterial::Custom {
                solref: (c.solref[0], c.solref[1]),
                solimp: (c.solimp[0], c.solimp[1], c.solimp[2]),
                friction: (c.friction[0], c.friction[1], c.friction[2]),
            }),
        }
    }

    fn from_value(v: &Value) -> Result<Self, TaskError> {
        match v {
            Value::String(s) => Ok(MaterialSpec::Preset(s.clone())),
            Value::Object(obj) => {
                for k in obj.keys() {
                    if !matches!(k.as_str(), "solref" | "solimp" | "friction") {
                        return Err(invalid(format!(
                            "unknown material key `{k}` — a custom material takes exactly \
                             solref, solimp, friction"
                        )));
                    }
                }
                let get = |key: &str, len: usize| -> Result<Vec<f64>, TaskError> {
                    let raw = obj
                        .get(key)
                        .ok_or_else(|| invalid(format!("custom material is missing `{key}`")))?;
                    let arr = raw.as_array().ok_or_else(|| {
                        invalid(format!("material {key} must be a list of {len} numbers"))
                    })?;
                    if arr.len() != len {
                        return Err(invalid(format!(
                            "material {key} needs {len} values, got {}",
                            arr.len()
                        )));
                    }
                    arr.iter()
                        .map(|x| {
                            x.as_f64().ok_or_else(|| {
                                invalid(format!("material {key} values must be numbers"))
                            })
                        })
                        .collect()
                };
                let solref = get("solref", 2)?;
                let solimp = get("solimp", 3)?;
                let friction = get("friction", 3)?;
                Ok(MaterialSpec::Custom(CustomMaterial {
                    solref: [solref[0], solref[1]],
                    solimp: [solimp[0], solimp[1], solimp[2]],
                    friction: [friction[0], friction[1], friction[2]],
                }))
            }
            other => Err(invalid(format!(
                "material must be a preset name or a custom object \
                 {{solref, solimp, friction}}, got {other}"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for MaterialSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        MaterialSpec::from_value(&v).map_err(serde::de::Error::custom)
    }
}

// ===== the scene =====

/// One free-floating prop. Same field set (and same camelCase spelling) as the
/// Studio prop wire type, plus `material`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TaskProp {
    pub name: String,
    /// `"box"` | `"sphere"` | `"cylinder"`.
    pub kind: String,
    /// Box HALF-extents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub half_extents: Option<[f64; 3]>,
    /// Sphere / cylinder radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// Cylinder FULL length (Z-aligned).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<f64>,
    /// Initial world position of the primitive's center.
    pub pos: [f64; 3],
    /// Initial world orientation, w-first (MJCF order); absent = identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quat: Option<[f64; 4]>,
    /// Mass (kg); absent = [`DEFAULT_PROP_MASS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialSpec>,
}

impl TaskProp {
    /// The engine spec for this prop, validated by the generator's own rules.
    pub fn to_prop_spec(&self) -> Result<PropSpec, TaskError> {
        let shape = match self.kind.as_str() {
            "box" => PropShape::Box {
                half: self.half_extents.ok_or_else(|| {
                    invalid(format!("prop `{}`: box needs halfExtents", self.name))
                })?,
            },
            "sphere" => PropShape::Sphere {
                r: self
                    .radius
                    .ok_or_else(|| invalid(format!("prop `{}`: sphere needs radius", self.name)))?,
            },
            "cylinder" => PropShape::Cylinder {
                r: self.radius.ok_or_else(|| {
                    invalid(format!("prop `{}`: cylinder needs radius", self.name))
                })?,
                h: self.length.ok_or_else(|| {
                    invalid(format!("prop `{}`: cylinder needs length", self.name))
                })?,
            },
            k => {
                return Err(invalid(format!(
                    "prop `{}`: unknown kind `{k}` (box|sphere|cylinder)",
                    self.name
                )));
            }
        };
        let spec = PropSpec {
            name: self.name.clone(),
            shape,
            pos: self.pos,
            quat: self.quat,
            mass: self.mass.unwrap_or(DEFAULT_PROP_MASS),
            rgba: self.rgba,
            material: self
                .material
                .as_ref()
                .map(MaterialSpec::to_contact_material)
                .transpose()?,
        };
        validate_prop(&spec)?;
        Ok(spec)
    }
}

/// A named target region: an axis-aligned box the success predicate can point
/// at by name. Evaluator-side only — nothing is emitted into MJCF for it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ZoneSpec {
    pub name: String,
    pub center: [f64; 3],
    pub half: [f64; 3],
    /// Render hint only (a translucent box in the viewport).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[f32; 4]>,
}

impl ZoneSpec {
    /// The evaluator's zone (same inclusive-face semantics as python's).
    pub fn zone(&self) -> Result<Zone, TaskError> {
        Zone::new(self.center, self.half)
    }
}

/// The world the task acts on.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct SceneSpec {
    /// Ground-plane height; absent = 0.0 (see [`SceneSpec::ground_height`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub props: Vec<TaskProp>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub zones: Vec<ZoneSpec>,
}

impl SceneSpec {
    pub fn ground_height(&self) -> f64 {
        self.ground.unwrap_or(0.0)
    }

    /// The named zone, if this scene declares it.
    pub fn zone(&self, name: &str) -> Option<&ZoneSpec> {
        self.zones.iter().find(|z| z.name == name)
    }
}

/// A gripper-channel override, mirroring the live session's own knobs: which
/// joint is the gripper and which end of its range CLOSES it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct GripperSpec {
    /// Caliper joint name; absent = let the session auto-detect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint: Option<String>,
    /// `"lo"` (the usual convention) or `"hi"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed: Option<String>,
}

// ===== the task =====

/// One manipulation task, as loaded from a `*.caliper-task.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TaskSpec {
    /// Must be [`TASK_VERSION`].
    pub version: u32,
    pub name: String,
    /// The `robot` field verbatim — a path relative to the task file, or an
    /// absolute one. [`TaskSpec::robot_path`] is the resolved form.
    pub robot: String,
    /// Resolved absolute-or-relative-to-CWD robot path. Derived, never stored
    /// in the file (which is why the file round-trips exactly regardless of
    /// where it was loaded from).
    #[serde(skip)]
    resolved_robot: PathBuf,
    /// Start pose; absent = whatever the caller is already at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q0: Option<Vec<f64>>,
    pub scene: SceneSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gripper: Option<GripperSpec>,
    /// The success criterion; absent = the task defines no verdict (a free
    /// teleop scene).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<Predicate>,
    /// Episode time budget (s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon_s: Option<f64>,
    /// Recording rate for takes of this task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

impl TaskSpec {
    /// The robot path resolved against the task file's directory.
    pub fn robot_path(&self) -> &Path {
        &self.resolved_robot
    }

    /// Parse a task from JSON text, resolving relative paths and zone names
    /// against `base_dir` and the file's own scene. Every rule in the module
    /// docs is enforced here — this is the only way to obtain a `TaskSpec`
    /// whose zone references are resolved.
    pub fn from_json_str(text: &str, base_dir: &Path) -> Result<Self, TaskError> {
        Self::parse(text, base_dir, "<task text>")
    }

    /// The one parse path (`load_task` and [`from_json_str`](Self::from_json_str)
    /// differ only in what `origin` an error names).
    fn parse(text: &str, base_dir: &Path, origin: &str) -> Result<Self, TaskError> {
        let mut spec: TaskSpec =
            serde_json::from_str(text).map_err(|e| TaskError::Parse(format!("{origin}: {e}")))?;
        spec.finish(base_dir)?;
        Ok(spec)
    }

    /// Pretty JSON, ready to write. The resolved robot path and resolved zone
    /// references are the ONLY things that differ from the input text:
    /// a zone named in `success` comes back spelled out.
    pub fn to_json(&self) -> Result<String, TaskError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| invalid(format!("serializing task `{}`: {e}", self.name)))
    }

    /// Engine specs for every prop, in file order.
    pub fn prop_specs(&self) -> Result<Vec<PropSpec>, TaskError> {
        self.scene
            .props
            .iter()
            .map(TaskProp::to_prop_spec)
            .collect()
    }

    /// A tracker for this task's success criterion, if it has one. Call
    /// [`SuccessTracker::reset`] with the episode's first state before judging.
    pub fn success_tracker(&self) -> Option<SuccessTracker> {
        self.success.clone().map(SuccessTracker::new)
    }

    /// Resolve the robot path + zone names, then validate. Split out of
    /// [`from_json_str`](Self::from_json_str) so both entry points share one
    /// rulebook.
    fn finish(&mut self, base_dir: &Path) -> Result<(), TaskError> {
        self.validate_head()?;
        let robot = Path::new(&self.robot);
        self.resolved_robot = if robot.is_absolute() {
            robot.to_path_buf()
        } else {
            base_dir.join(robot)
        };
        if !self.resolved_robot.is_file() {
            return Err(invalid(format!(
                "task `{}`: robot `{}` does not resolve to a file (looked at {})",
                self.name,
                self.robot,
                self.resolved_robot.display()
            )));
        }
        self.resolve_zones()?;
        self.validate_scene()?;
        self.validate_success()?;
        Ok(())
    }

    /// Version / name / robot / q0 / horizon / fps / gripper.
    fn validate_head(&self) -> Result<(), TaskError> {
        if self.version != TASK_VERSION {
            return Err(invalid(format!(
                "task file version {} is not supported — this build reads version \
                 {TASK_VERSION}",
                self.version
            )));
        }
        if self.name.trim().is_empty() {
            return Err(invalid("task `name` must be a non-empty string"));
        }
        if self.robot.trim().is_empty() {
            return Err(invalid(format!(
                "task `{}`: `robot` must be a non-empty path",
                self.name
            )));
        }
        if let Some(q0) = &self.q0
            && (q0.is_empty() || !q0.iter().all(|x| x.is_finite()))
        {
            return Err(invalid(format!(
                "task `{}`: `q0` must be a non-empty list of finite numbers",
                self.name
            )));
        }
        if let Some(h) = self.horizon_s
            && !(h.is_finite() && h > 0.0)
        {
            return Err(invalid(format!(
                "task `{}`: `horizonS` must be finite and > 0, got {h}",
                self.name
            )));
        }
        if let Some(fps) = self.fps
            && fps == 0
        {
            return Err(invalid(format!("task `{}`: `fps` must be > 0", self.name)));
        }
        if let Some(g) = &self.gripper {
            if let Some(j) = &g.joint
                && j.trim().is_empty()
            {
                return Err(invalid(format!(
                    "task `{}`: gripper `joint` must be a non-empty joint name",
                    self.name
                )));
            }
            if let Some(c) = &g.closed
                && !matches!(c.as_str(), "lo" | "hi")
            {
                return Err(invalid(format!(
                    "task `{}`: gripper `closed` must be \"lo\" or \"hi\", got \"{c}\" \
                     — which limit CLOSES the gripper",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Props (through the generator's rules) and zones.
    fn validate_scene(&self) -> Result<(), TaskError> {
        let mut seen: Vec<String> = Vec::new();
        for p in &self.scene.props {
            // The generator's own rulebook: dimensions, mass, quat, rgba,
            // material values. One copy, not two.
            p.to_prop_spec()?;
            // ... plus the uniqueness rule MJCF body naming imposes.
            let body = mujoco_name(&p.name);
            if seen.contains(&body) {
                return Err(invalid(format!(
                    "task `{}`: duplicate prop name `{}` (after sanitizing)",
                    self.name, p.name
                )));
            }
            seen.push(body);
        }
        let mut zone_names: Vec<&str> = Vec::new();
        for z in &self.scene.zones {
            if z.name.trim().is_empty() {
                return Err(invalid(format!(
                    "task `{}`: every zone needs a non-empty name",
                    self.name
                )));
            }
            if zone_names.contains(&z.name.as_str()) {
                return Err(invalid(format!(
                    "task `{}`: duplicate zone name `{}`",
                    self.name, z.name
                )));
            }
            zone_names.push(&z.name);
            z.zone()
                .map_err(|e| invalid(format!("task `{}`: zone `{}`: {e}", self.name, z.name)))?;
            if let Some(c) = z.rgba
                && !c.iter().all(|x| x.is_finite())
            {
                return Err(invalid(format!(
                    "task `{}`: zone `{}` rgba must be finite",
                    self.name, z.name
                )));
            }
        }
        Ok(())
    }

    /// The success criterion's own rules, plus the cross-check the file makes
    /// possible: every prop it scores must actually be in the scene.
    fn validate_success(&self) -> Result<(), TaskError> {
        let Some(p) = &self.success else {
            return Ok(());
        };
        p.validate()?;
        let known: Vec<&str> = self.scene.props.iter().map(|p| p.name.as_str()).collect();
        for prop in p.prop_names() {
            if !known.contains(&prop) {
                return Err(invalid(format!(
                    "task `{}`: `success` scores prop `{prop}`, which the scene does not \
                     contain (props: {known:?})",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Replace every `success` zone written as a NAME with the zone that
    /// `scene.zones` declares under it.
    fn resolve_zones(&mut self) -> Result<(), TaskError> {
        let Some(pred) = self.success.as_mut() else {
            return Ok(());
        };
        let scene = &self.scene;
        let task = self.name.clone();
        pred.visit_zones_mut(&mut |slot: &mut ZoneRef| {
            let ZoneRef::Named(name) = slot else {
                return Ok(());
            };
            let Some(spec) = scene.zone(name) else {
                let known: Vec<&str> = scene.zones.iter().map(|z| z.name.as_str()).collect();
                return Err(invalid(format!(
                    "task `{task}`: `success` refers to zone `{name}`, which \
                     `scene.zones` does not declare (zones: {known:?})"
                )));
            };
            *slot = ZoneRef::Inline(spec.zone()?);
            Ok(())
        })
    }
}

/// Read and fully resolve a task file: robot path relative to the FILE, zone
/// names against the file's own scene, every rule in the module docs checked.
pub fn load_task(path: impl AsRef<Path>) -> Result<TaskSpec, TaskError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| TaskError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    // Relative paths and zone names resolve against the FILE's own directory.
    TaskSpec::parse(
        &text,
        path.parent().unwrap_or_else(|| Path::new(".")),
        &path.display().to_string(),
    )
}

/// Write a task file (pretty, newline-terminated). Round-trips through
/// [`load_task`] with the resolved-zone caveat in [`TaskSpec::to_json`].
pub fn save_task(path: impl AsRef<Path>, spec: &TaskSpec) -> Result<(), TaskError> {
    let path = path.as_ref();
    let mut text = spec.to_json()?;
    text.push('\n');
    std::fs::write(path, text).map_err(|e| TaskError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use success::{LiftedRef, SuccessState};

    fn tasks_dir() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/fixtures/tasks"
        ))
    }

    fn fixture(name: &str) -> PathBuf {
        tasks_dir().join(name)
    }

    fn value(text: &str) -> Value {
        serde_json::from_str(text).expect("json parses")
    }

    /// A minimal valid task, as text, with `patch` splices applied by the
    /// caller — the base every error case mutates.
    fn base_task() -> String {
        r#"{
          "version": 1,
          "name": "t",
          "robot": "../robots/gripper_arm.urdf",
          "scene": {
            "props": [
              {"name": "cube", "kind": "box", "halfExtents": [0.05, 0.05, 0.05],
               "pos": [0.0, 0.0, 0.05], "mass": 0.05}
            ],
            "zones": [
              {"name": "bin", "center": [0.4, 0.2, 0.02], "half": [0.05, 0.05, 0.02]}
            ]
          }
        }"#
        .to_string()
    }

    fn load_text(text: &str) -> Result<TaskSpec, TaskError> {
        TaskSpec::from_json_str(text, &tasks_dir())
    }

    // -- the fixtures --

    #[test]
    fn the_pick_cube_fixture_round_trips_exactly() {
        let path = fixture("pick_cube.caliper-task.json");
        let text = std::fs::read_to_string(&path).expect("fixture reads");
        let spec = load_task(&path).expect("fixture loads");

        assert_eq!(spec.version, TASK_VERSION);
        assert_eq!(spec.name, "pick-cube");
        assert_eq!(spec.robot, "../robots/gripper_arm.urdf");
        assert!(
            spec.robot_path().is_file(),
            "robot resolved to {}",
            spec.robot_path().display()
        );
        assert_eq!(spec.q0, Some(vec![0.0, 0.0, 0.02]));
        assert_eq!(spec.scene.ground_height(), 0.0);
        assert_eq!(spec.scene.props.len(), 1);
        assert_eq!(spec.scene.zones.len(), 1);
        assert_eq!(spec.fps, Some(50));
        assert_eq!(spec.horizon_s, Some(20.0));

        // Every byte of meaning survives a round-trip: the fixture's success
        // zone is spelled out, so nothing is resolved away.
        assert_eq!(
            value(&spec.to_json().unwrap()),
            value(&text),
            "the fixture does not round-trip"
        );
        // ... and re-parsing the serialized form gives the same task
        let again = TaskSpec::from_json_str(&spec.to_json().unwrap(), &tasks_dir()).unwrap();
        assert_eq!(again, spec);
    }

    #[test]
    fn the_pick_cube_fixture_carries_a_material_and_a_verdict() {
        let spec = load_task(fixture("pick_cube.caliper-task.json")).unwrap();
        let props = spec.prop_specs().unwrap();
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].name, "cube");
        assert_eq!(props[0].material, Some(ContactMaterial::Wood));
        assert!((props[0].mass - 0.05).abs() < 1e-12);

        let g = spec
            .gripper
            .as_ref()
            .expect("fixture overrides the gripper");
        assert_eq!(g.joint.as_deref(), Some("gripper"));
        assert_eq!(g.closed.as_deref(), Some("lo"));

        let pred = spec.success.as_ref().expect("fixture has a success");
        assert_eq!(
            pred.describe(),
            "cube's center is inside the 0.100 x 0.100 x 0.040 m box centered at \
             (0.400, 0.200, 0.020) and has come to rest (speed <= 0.010 m/s)"
        );
        // in the bin, at rest → success; in the bin but flying → not placed
        let mut t = spec.success_tracker().unwrap();
        let at_rest = SuccessState::from_positions([("cube", [0.4, 0.2, 0.02])])
            .with_velocities([("cube", [0.0, 0.0, 0.0])]);
        assert!(t.judge(&at_rest).unwrap());
        let flying = SuccessState::from_positions([("cube", [0.4, 0.2, 0.02])])
            .with_velocities([("cube", [0.0, 0.0, -1.5])]);
        assert!(!t.judge(&flying).unwrap());
    }

    #[test]
    fn the_lift_cube_fixture_resolves_its_zone_by_name() {
        let path = fixture("lift_cube.caliper-task.json");
        let spec = load_task(&path).unwrap();
        let bin = spec.scene.zone("bin").unwrap().zone().unwrap();

        // The file writes `"zone": "bin"`; what we hold is the resolved box.
        let Some(Predicate::AllOf(terms)) = spec.success.clone() else {
            panic!("expected an all_of, got {:?}", spec.success);
        };
        assert_eq!(
            terms[0],
            Predicate::Lifted {
                prop: "cube".into(),
                height: 0.05,
                reference: LiftedRef::Initial
            }
        );
        let Predicate::PlacedInZone { zone, .. } = &terms[1] else {
            panic!("expected a placed_in_zone, got {:?}", terms[1]);
        };
        assert_eq!(zone, &ZoneRef::Inline(bin));

        // Serialization carries the RESOLVED form (the documented exception to
        // an exact round-trip), and re-parsing it is a no-op.
        let json = spec.to_json().unwrap();
        assert!(
            json.contains("\"center\""),
            "the resolved zone must be spelled out:\n{json}"
        );
        assert_eq!(TaskSpec::from_json_str(&json, &tasks_dir()).unwrap(), spec);
    }

    // -- strictness --

    #[test]
    fn unknown_keys_raise_at_every_level() {
        let cases = [
            (r#""version": 1, "hoirzon": 3,"#, "hoirzon"),
            (r#""version": 1, "Name": "x","#, "Name"),
        ];
        for (splice, want) in cases {
            let text = base_task().replacen(r#""version": 1,"#, splice, 1);
            let err = load_text(&text).map(|_| ()).unwrap_err().to_string();
            assert!(err.contains(want), "for {splice}\n  got: {err}");
        }
        // nested: scene, prop, zone, gripper, material, success
        let nested = [
            (
                r#""scene": {"gravity": 9.8, "props": [], "zones": []}"#,
                "gravity",
            ),
            (
                r#""scene": {"props": [{"name":"c","kind":"box","halfExtent":[1,1,1],"pos":[0,0,1]}], "zones": []}"#,
                "halfExtent",
            ),
            (
                r#""scene": {"props": [], "zones": [{"name":"b","center":[0,0,0],"half":[1,1,1],"colour":[1,1,1,1]}]}"#,
                "colour",
            ),
            (
                r#""scene": {"props": [], "zones": []}, "gripper": {"joint": "g", "close": "lo"}"#,
                "close",
            ),
            (
                r#""scene": {"props": [{"name":"c","kind":"sphere","radius":0.05,"pos":[0,0,1],
                   "material":{"solref":[0.01,1.0],"solimp":[0.9,0.95,0.001],"frictoin":[1,0,0]}}], "zones": []}"#,
                "frictoin",
            ),
            (
                r#""scene": {"props": [{"name":"c","kind":"sphere","radius":0.05,"pos":[0,0,1]}], "zones": []},
                   "success": {"kind":"lifted","prop":"c","height":0.05,"reff":"initial"}"#,
                "reff",
            ),
        ];
        for (splice, want) in nested {
            let text = format!(
                r#"{{"version": 1, "name": "t", "robot": "../robots/gripper_arm.urdf", {splice}}}"#
            );
            let err = load_text(&text).map(|_| ()).unwrap_err().to_string();
            assert!(err.contains(want), "for {splice}\n  got: {err}");
        }
    }

    #[test]
    fn validation_refuses_impossible_tasks() {
        let cases: Vec<(String, &str)> = vec![
            (
                base_task().replacen(r#""version": 1"#, r#""version": 2"#, 1),
                "version 2 is not supported",
            ),
            (
                base_task().replacen(r#""name": "t""#, r#""name": "  ""#, 1),
                "`name` must be a non-empty",
            ),
            (
                base_task().replacen(
                    r#""robot": "../robots/gripper_arm.urdf""#,
                    r#""robot": "../robots/nope.urdf""#,
                    1,
                ),
                "does not resolve to a file",
            ),
            (
                base_task().replacen(r#""mass": 0.05"#, r#""mass": 0.0"#, 1),
                "mass must be finite and > 0",
            ),
            (
                base_task().replacen(r#""halfExtents": [0.05, 0.05, 0.05]"#, r#""radius": 0.05"#, 1),
                "box needs halfExtents",
            ),
            (
                base_task().replacen(r#""kind": "box""#, r#""kind": "cone""#, 1),
                "unknown kind `cone`",
            ),
            (
                base_task().replacen(r#""half": [0.05, 0.05, 0.02]"#, r#""half": [0.05, -0.05, 0.02]"#, 1),
                "non-negative",
            ),
            (
                base_task().replacen(
                    r#""zones": ["#,
                    r#""zones": [{"name": "bin", "center": [0,0,0], "half": [1,1,1]},"#,
                    1,
                ),
                "duplicate zone name `bin`",
            ),
            (
                base_task().replacen(
                    r#""props": ["#,
                    r#""props": [{"name": "cube", "kind": "sphere", "radius": 0.01, "pos": [1,1,1]},"#,
                    1,
                ),
                "duplicate prop name `cube`",
            ),
            (
                base_task().replacen(
                    r#""scene": {"#,
                    r#""success": {"kind":"lifted","prop":"ball","height":0.05}, "scene": {"#,
                    1,
                ),
                "which the scene does not contain",
            ),
            (
                base_task().replacen(
                    r#""scene": {"#,
                    r#""success": {"kind":"placed_in_zone","prop":"cube","zone":"crate"}, "scene": {"#,
                    1,
                ),
                "does not declare",
            ),
            (
                base_task().replacen(r#""version": 1,"#, r#""version": 1, "horizonS": 0.0,"#, 1),
                "`horizonS` must be finite and > 0",
            ),
            (
                base_task().replacen(r#""version": 1,"#, r#""version": 1, "fps": 0,"#, 1),
                "`fps` must be > 0",
            ),
            (
                base_task().replacen(r#""version": 1,"#, r#""version": 1, "q0": [0.0, null],"#, 1),
                "invalid",
            ),
            (
                base_task().replacen(
                    r#""version": 1,"#,
                    r#""version": 1, "gripper": {"joint": "gripper", "closed": "shut"},"#,
                    1,
                ),
                "must be \"lo\" or \"hi\"",
            ),
            (
                base_task().replacen(r#""version": 1,"#, r#""version": 1, "q0": [],"#, 1),
                "`q0` must be a non-empty",
            ),
            (
                base_task().replacen(r#""mass": 0.05"#, r#""mass": 0.05, "material": "granite""#, 1),
                "unknown material preset `granite`",
            ),
            (
                base_task().replacen(
                    r#""mass": 0.05"#,
                    r#""mass": 0.05, "material": {"solref": [0.01], "solimp": [0.9,0.95,0.001], "friction": [1,0,0]}"#,
                    1,
                ),
                "solref needs 2 values",
            ),
            (
                base_task().replacen(
                    r#""mass": 0.05"#,
                    r#""mass": 0.05, "material": {"solref": [0.01,1.0], "solimp": [0.9,0.95,0.0], "friction": [1,0,0]}"#,
                    1,
                ),
                "width > 0",
            ),
        ];
        for (text, want) in cases {
            let err = load_text(&text)
                .map(|_| ())
                .expect_err(&format!("must be refused:\n{text}"))
                .to_string();
            assert!(err.contains(want), "want `{want}`\n  got: {err}");
        }
    }

    #[test]
    fn a_scene_only_task_needs_no_success() {
        let text = r#"{
          "version": 1, "name": "free", "robot": "../robots/gripper_arm.urdf",
          "scene": {}
        }"#;
        let spec = load_text(text).unwrap();
        assert!(spec.success.is_none());
        assert!(spec.success_tracker().is_none());
        assert!(spec.scene.props.is_empty() && spec.scene.zones.is_empty());
        assert_eq!(spec.scene.ground_height(), 0.0);
        // an empty scene serializes back to an empty object
        assert_eq!(value(&spec.to_json().unwrap())["scene"], value("{}"));
    }

    #[test]
    fn a_custom_material_survives_the_round_trip() {
        let text = base_task().replacen(
            r#""mass": 0.05"#,
            r#""mass": 0.05, "material": {"solref": [0.01, 1.0],
               "solimp": [0.9, 0.95, 0.002], "friction": [1.2, 0.01, 0.0002]}"#,
            1,
        );
        let spec = load_text(&text).unwrap();
        assert_eq!(
            spec.prop_specs().unwrap()[0].material,
            Some(ContactMaterial::Custom {
                solref: (0.01, 1.0),
                solimp: (0.9, 0.95, 0.002),
                friction: (1.2, 0.01, 0.0002),
            })
        );
        assert_eq!(value(&spec.to_json().unwrap()), value(&text));
    }

    #[test]
    fn save_then_load_is_a_fixed_point() {
        let spec = load_task(fixture("pick_cube.caliper-task.json")).unwrap();
        let out = std::env::temp_dir().join(format!(
            "caliper_task_roundtrip_{}.caliper-task.json",
            std::process::id()
        ));
        // Robot paths are relative to the FILE, so the copy has to sit beside
        // the original for its robot to resolve — write an absolute path.
        let mut abs = spec.clone();
        abs.robot = spec.robot_path().display().to_string();
        save_task(&out, &abs).unwrap();
        let back = load_task(&out).unwrap();
        assert_eq!(back, abs);
        assert_eq!(back.robot_path(), spec.robot_path());
        let _ = std::fs::remove_file(&out);
    }

    // -- the Rust↔python parity table (mirrored by the learn wave) --

    /// One row of the parity table: a state, and the verdict each of the three
    /// predicates must return for it (`[A, B, C]`).
    struct ParityRow {
        label: &'static str,
        pos: [f64; 3],
        vel: [f64; 3],
        want: [bool; 3],
    }

    /// Three predicate JSONs, six hand-computed states, one expected verdict
    /// each. `caliper_learn.success` runs the SAME table; any divergence here
    /// is a divergence between the two faces of one contract.
    ///
    /// Two rows sit deliberately ON a boundary, because that is where two
    /// implementations drift: `nudged` clears 5 mm of a 50 mm bar, and `on face`
    /// sits exactly on the bin's +x face at exactly the settle speed (both
    /// INSIDE — faces and the speed limit are inclusive). Both faces do the same
    /// binary arithmetic, so both must agree.
    #[test]
    fn parity_table_with_the_python_evaluator() {
        const A: &str = r#"{"kind":"lifted","prop":"cube","height":0.05,"ref":"initial"}"#;
        const B: &str = r#"{"kind":"placed_in_zone","prop":"cube",
            "zone":{"center":[0.4,0.2,0.02],"half":[0.05,0.05,0.02]},
            "settled_speed":0.01}"#;
        const C: &str = r#"{"kind":"any_of","terms":[
            {"kind":"lifted","prop":"cube","height":0.05,"ref":"absolute"},
            {"kind":"placed_in_zone","prop":"cube",
             "zone":{"center":[0.4,0.2,0.02],"half":[0.05,0.05,0.02]},
             "settled_speed":null}]}"#;

        // The poses and velocities the rows are built from, named once.
        const SPAWN: [f64; 3] = [0.3, 0.0, 0.025]; // on the table, where S0 is
        const NUDGE: [f64; 3] = [0.3, 0.0, 0.03]; // 5 mm above SPAWN
        const HIGH: [f64; 3] = [0.3, 0.0, 0.2]; // carried well clear
        const IN_BIN: [f64; 3] = [0.4, 0.2, 0.02]; // the bin's center
        const ON_FACE: [f64; 3] = [0.45, 0.2, 0.02]; // exactly the bin's +x face
        const REST: [f64; 3] = [0.0, 0.0, 0.0];
        const RISING: [f64; 3] = [0.0, 0.0, 0.1];
        const FAST_UP: [f64; 3] = [0.0, 0.0, 0.5];
        const FALLING: [f64; 3] = [0.0, 0.0, -1.5];
        const AT_LIMIT: [f64; 3] = [0.01, 0.0, 0.0]; // exactly settled_speed

        // S0 = the reset baseline: the cube at rest where it spawned.
        let s0 = SuccessState::from_positions([("cube", SPAWN)]).with_velocities([("cube", REST)]);
        let row = |label, pos, vel, want| ParityRow {
            label,
            pos,
            vel,
            want,
        };
        let rows = [
            // no lift, not in the bin, and z below C's absolute 0.05 bar
            row("table", SPAWN, REST, [false, false, false]),
            row("nudged", NUDGE, RISING, [false, false, false]),
            // carried: A lifts, and C's absolute term holds
            row("carried", HIGH, FAST_UP, [true, false, true]),
            // in the bin, at rest, and BELOW the height it started at
            row("placed", IN_BIN, REST, [false, true, true]),
            // in the bin but still moving: `placed` fails, C's settle-free term holds
            row("flying", IN_BIN, FALLING, [false, false, true]),
            row("on face", ON_FACE, AT_LIMIT, [false, true, true]),
        ];

        for ParityRow {
            label,
            pos,
            vel,
            want: [a, b, c],
        } in rows
        {
            let state =
                SuccessState::from_positions([("cube", pos)]).with_velocities([("cube", vel)]);
            // A needs an episode baseline; B and C are pure functions of the state.
            let mut ta = SuccessTracker::new(Predicate::from_value(&value(A)).unwrap());
            ta.reset(Some(&s0)).unwrap();
            assert_eq!(ta.judge(&state).unwrap(), a, "A on `{label}`");
            let pb = Predicate::from_value(&value(B)).unwrap();
            assert_eq!(pb.eval(&state).unwrap(), b, "B on `{label}`");
            let pc = Predicate::from_value(&value(C)).unwrap();
            assert_eq!(pc.eval(&state).unwrap(), c, "C on `{label}`");
        }

        // The two error rows: B against a velocity-less state, and against a
        // state that does not carry the prop at all.
        let b = Predicate::from_value(&value(B)).unwrap();
        assert!(
            b.eval(&SuccessState::from_positions([("cube", [0.4, 0.2, 0.02])]))
                .is_err(),
            "a settled check with no velocities must ERROR, not pass"
        );
        assert!(
            b.eval(
                &SuccessState::from_positions([("ball", [0.4, 0.2, 0.02])])
                    .with_velocities([("ball", [0.0, 0.0, 0.0])])
            )
            .is_err(),
            "a missing prop must ERROR, not pass"
        );
    }
}
