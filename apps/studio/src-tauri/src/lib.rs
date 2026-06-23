//! Caliper Studio — Tauri backend. Wires the `caliper` engine to the UI as the
//! single source of truth for all kinematics (FK / IK / topology).
//!
//! The webview holds a parsed [`Model`] in shared state after `robot_info`, then
//! drives it with `get_frames` (live FK for every frame) and `solve_ik` (engine
//! IK on a gizmo-supplied target). Geometry is drawn procedurally from frame
//! world poses — the oracle fixtures carry no `<visual>` meshes.
//!
//! Matrix convention end-to-end: every 4×4 crossing the IPC boundary is
//! **column-major** `[f64; 16]`, i.e. THREE.Matrix4 element order
//! (`m[col*4 + row]`). `col_major_from_se3` / `se3_from_col_major` are exact
//! inverses (round-trip ≈ 4.5e-16, verified by `mat_roundtrip` below).
use caliper::ik::{ik, IkOpts};
use caliper::kinematics::{fk_joints, frame_pose};
use caliper::model::{JointKind, Model};
use caliper::spatial::Se3;
use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

/// Loaded robot, shared across commands. `None` until `robot_info` succeeds.
#[derive(Default)]
struct AppState {
    model: Mutex<Option<Model>>,
}

// ===== wire types (serde-facing; all kinematics live in the engine) =====

/// One renderable link frame. Carries everything the UI needs to draw a triad,
/// a rod back to its kinematic parent, and (when this frame is a movable joint's
/// own output frame) a correctly-oriented joint marker.
#[derive(Serialize)]
struct FrameInfo {
    name: String,
    /// Index into `frames` of the parent frame, or `-1` for the root frame.
    /// The UI draws a rod from this frame's origin to its parent's origin.
    parent: i64,
    /// Movable joint whose chain this frame rides on (`-1` = root / fixed-to-world).
    anchor: i64,
    /// If this frame is the *primary* (identity-offset) output frame of a movable
    /// joint, the index of that joint; else `-1`. Only these frames draw a joint
    /// marker. (Lets the UI map a slider to the frame whose marker should spin/slide.)
    #[serde(rename = "jointIndex")]
    joint_index: i64,
    /// `"revolute"` | `"prismatic"` for a primary joint frame; else `null`.
    #[serde(rename = "jointKind")]
    joint_kind: Option<String>,
    /// Joint-local rotation/translation axis for a primary joint frame; else `null`.
    /// Combined with this frame's world rotation it gives the world joint axis.
    axis: Option<[f64; 3]>,
}

/// Full structure the UI needs to build the scene + slider panel.
#[derive(Serialize)]
struct RobotInfo {
    name: String,
    ndof: usize,
    #[serde(rename = "jointNames")]
    joint_names: Vec<String>,
    /// Parallel to `jointNames`: `"revolute"` | `"prismatic"`.
    #[serde(rename = "jointKinds")]
    joint_kinds: Vec<String>,
    /// Parallel to `jointNames`: `[lo, hi]` or `null` when unbounded.
    limits: Vec<Option<[f64; 2]>>,
    /// Every link frame, in engine frame order (matches `get_frames` order).
    frames: Vec<FrameInfo>,
    /// Index into `frames` of the default tool/tip frame.
    tip: usize,
}

#[derive(Serialize)]
struct IkSolution {
    success: bool,
    q: Vec<f64>,
    residual: f64,
}

#[derive(Deserialize)]
struct IkRequest {
    /// Target pose as a column-major 4×4 (THREE.Matrix4 element order).
    target: [f64; 16],
    seed: Vec<f64>,
    /// Frame to solve for, by name. `None` → the model's default tip frame.
    frame: Option<String>,
}

// ===== conversions =====

/// THREE.Matrix4 is **column-major**: element `m[col*4 + row]`. Rebuild the
/// rotation `R` (top-left 3×3) and translation `t` (last column, rows 0..3),
/// renormalise `R` to the nearest proper rotation (the gizmo may hand us a
/// matrix with floating drift / non-unit scale), and pack into an `Se3`.
fn se3_from_col_major(m: &[f64; 16]) -> Se3 {
    // column c, row r  ->  m[c*4 + r]
    let r = Matrix3::new(
        m[0], m[4], m[8], // row 0: (0,0) (0,1) (0,2)
        m[1], m[5], m[9], // row 1
        m[2], m[6], m[10], // row 2
    );
    // last column, rows 0..3
    let t = Vector3::new(m[12], m[13], m[14]);
    // Project onto SO(3) so the quaternion is well-posed even if the incoming
    // basis has tiny non-orthogonality / scale. `UnitQuaternion::from_matrix` is
    // the iterative nearest-rotation, robust near θ ≈ π.
    let quat = UnitQuaternion::from_matrix(&r);
    Se3(Isometry3::from_parts(Translation3::from(t), quat))
}

/// `Se3` → column-major `[f64;16]` for THREE.Matrix4. We index the homogeneous
/// matrix logically by `(row, col)` and write `out[col*4 + row]`, so the mapping
/// is unambiguous and independent of nalgebra's internal storage order.
fn col_major_from_se3(t: &Se3) -> [f64; 16] {
    let h = t.0.to_homogeneous(); // 4×4, indexed (row, col)
    let mut out = [0.0_f64; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = h[(row, col)];
        }
    }
    out
}

fn is_identity_offset(t: &Se3) -> bool {
    let p = t.translation_vec().norm();
    let ang = t.0.rotation.angle();
    p < 1e-12 && ang < 1e-12
}

/// For each movable joint, the index of its *primary* frame: the unique frame
/// whose `anchor == Some(j)` with an identity offset (the joint's own output
/// link, registered right after the joint in `caliper-model::compile`).
fn primary_frames(model: &Model) -> Vec<usize> {
    let mut primary = vec![usize::MAX; model.ndof];
    for (fi, f) in model.frames.iter().enumerate() {
        if let Some(j) = f.anchor {
            if is_identity_offset(&f.offset) && primary[j] == usize::MAX {
                primary[j] = fi;
            }
        }
    }
    primary
}

/// Index of the parent frame to draw a rod toward, or `-1` for the root.
///
/// * root frame (`anchor == None`)               -> -1
/// * a joint's primary frame                     -> primary frame of `parent[j]` (or root)
/// * a fixed-folded frame (anchor == j, offset≠id) -> primary frame of `j`
fn frame_parents(model: &Model, primary: &[usize]) -> Vec<i64> {
    let root = model
        .frames
        .iter()
        .position(|f| f.anchor.is_none())
        .map(|i| i as i64)
        .unwrap_or(-1);

    let primary_or_root = |j: Option<usize>| -> i64 {
        match j {
            Some(pj) if primary[pj] != usize::MAX => primary[pj] as i64,
            _ => root,
        }
    };

    model
        .frames
        .iter()
        .enumerate()
        .map(|(fi, f)| match f.anchor {
            None => -1,
            Some(j) => {
                if primary[j] == fi {
                    primary_or_root(model.parent[j])
                } else {
                    let p = primary[j];
                    if p == usize::MAX {
                        root
                    } else {
                        p as i64
                    }
                }
            }
        })
        .collect()
}

/// Build the wire `RobotInfo` from a compiled model (pure; shared by the command
/// and the tests).
fn robot_info_from_model(model: &Model) -> RobotInfo {
    let primary = primary_frames(model);
    let parents = frame_parents(model, &primary);

    // joint index -> its primary frame index, inverted for quick frame lookup.
    let mut frame_joint = vec![-1_i64; model.frames.len()];
    for (j, &fi) in primary.iter().enumerate() {
        if fi != usize::MAX {
            frame_joint[fi] = j as i64;
        }
    }

    let frames: Vec<FrameInfo> = model
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let ji = frame_joint[i];
            let (joint_kind, axis) = if ji >= 0 {
                let j = ji as usize;
                let kind = match model.kind[j] {
                    JointKind::Revolute => "revolute".to_string(),
                    JointKind::Prismatic => "prismatic".to_string(),
                };
                let a = model.axis[j];
                (Some(kind), Some([a.x, a.y, a.z]))
            } else {
                (None, None)
            };
            FrameInfo {
                name: f.name.clone(),
                parent: parents[i],
                anchor: f.anchor.map(|a| a as i64).unwrap_or(-1),
                joint_index: ji,
                joint_kind,
                axis,
            }
        })
        .collect();

    let joint_kinds = model
        .kind
        .iter()
        .map(|k| match k {
            JointKind::Revolute => "revolute".to_string(),
            JointKind::Prismatic => "prismatic".to_string(),
        })
        .collect();

    let limits = model
        .limits
        .iter()
        .map(|l| l.map(|(lo, hi)| [lo, hi]))
        .collect();

    RobotInfo {
        name: model.name.clone(),
        ndof: model.ndof,
        joint_names: model.joint_names.clone(),
        joint_kinds,
        limits,
        frames,
        tip: model.tip_frame(),
    }
}

/// World pose of every frame at `q`, column-major (pure; shared with tests).
fn frames_at(model: &Model, q: &[f64]) -> Vec<[f64; 16]> {
    let mut joint_world = vec![Se3::identity(); model.ndof];
    fk_joints(model, q, &mut joint_world);
    (0..model.frames.len())
        .map(|f| col_major_from_se3(&frame_pose(model, &joint_world, f)))
        .collect()
}

fn ext_ok(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("urdf") || e.eq_ignore_ascii_case("xacro"))
}

// ===== commands =====

/// Engine version (proves the UI is talking to the real Rust core).
#[tauri::command]
fn engine_version() -> String {
    caliper::VERSION.to_string()
}

/// Absolute paths to the bundled oracle fixtures, as `[name, path]` pairs.
///
/// Resolved from `CARGO_MANIFEST_DIR` at compile time so the dropdown works in
/// both `tauri dev` and a release build without a file dialog. (`src-tauri` is
/// `<repo>/apps/studio/src-tauri`, so fixtures live three levels up.)
#[tauri::command]
fn fixtures() -> Vec<(String, String)> {
    let root = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../oracle/fixtures/robots"
    );
    ["showcase6", "toy", "prismatic", "branched", "redundant7"]
        .iter()
        .map(|n| (n.to_string(), format!("{root}/{n}.urdf")))
        .collect()
}

/// Minimal robot summary returned by the legacy `load_robot` command.
#[derive(Serialize)]
struct RobotSummary {
    name: String,
    dof: usize,
    joint_names: Vec<String>,
}

/// Load a robot from a URDF path and return its name/dof/joint_names. Kept for
/// back-compat with the original HUD; `robot_info` supersedes it. Stateless.
#[tauri::command]
fn load_robot(path: String) -> Result<RobotSummary, String> {
    let p = Path::new(&path);
    if !ext_ok(p) {
        return Err("only .urdf or .xacro files are supported".into());
    }
    let robot = caliper::model::Robot::from_urdf(p)
        .map_err(|_| "failed to load robot from the given URDF".to_string())?;
    Ok(RobotSummary {
        name: robot.name.clone(),
        dof: robot.ndof(),
        joint_names: robot.joint_names.clone(),
    })
}

/// Load a robot, cache it in shared state, and return its full structure:
/// topology (frames + parent rods + per-joint axis/kind + tip), joint kinds,
/// and limits. The authoritative loader — `get_frames` / `solve_ik` operate on
/// the model cached here, so the engine owns all kinematics.
#[tauri::command]
fn robot_info(path: String, state: tauri::State<'_, AppState>) -> Result<RobotInfo, String> {
    let p = Path::new(&path);
    if !ext_ok(p) {
        return Err("only .urdf or .xacro files are supported".into());
    }
    let model = Model::from_urdf(p).map_err(|_| "failed to load robot from the given URDF")?;
    let info = robot_info_from_model(&model);
    *state.model.lock().map_err(|_| "state lock poisoned")? = Some(model);
    Ok(info)
}

/// World pose of **every** frame at configuration `q`, as column-major `[f64;16]`
/// (THREE.Matrix4 order), in engine frame order (matches `robot_info.frames`).
#[tauri::command]
fn get_frames(q: Vec<f64>, state: tauri::State<'_, AppState>) -> Result<Vec<[f64; 16]>, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if q.len() != model.ndof {
        return Err(format!(
            "expected {} joint values, got {}",
            model.ndof,
            q.len()
        ));
    }
    let mut q = q;
    model.clamp(&mut q);
    Ok(frames_at(model, &q))
}

/// Solve IK for the (gizmo-supplied) target pose and return the configuration.
///
/// `target` is a column-major 4×4 (THREE.Matrix4). `frame` selects the goal
/// frame by name; `None` uses the model's default tip. `seed.len() == ndof`.
#[tauri::command]
fn solve_ik(req: IkRequest, state: tauri::State<'_, AppState>) -> Result<IkSolution, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;

    let frame = match req.frame {
        Some(name) => model
            .frame_id(&name)
            .ok_or_else(|| format!("unknown frame `{name}`"))?,
        None => model.tip_frame(),
    };
    if req.seed.len() != model.ndof {
        return Err(format!(
            "expected seed of length {}, got {}",
            model.ndof,
            req.seed.len()
        ));
    }

    let target = se3_from_col_major(&req.target);
    let res = ik(model, frame, &target, &req.seed, &IkOpts::default());
    Ok(IkSolution {
        success: res.success,
        q: res.q,
        residual: res.residual,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            engine_version,
            fixtures,
            load_robot,
            robot_info,
            get_frames,
            solve_ik
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliper::kinematics::fk_frame;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../oracle/fixtures/robots"
        ))
        .join(name)
    }
    fn load(name: &str) -> Model {
        Model::from_urdf(&fixture(name)).expect("fixture loads")
    }

    /// The #1 trap: column-major encode/decode must round-trip exactly.
    #[test]
    fn mat_roundtrip() {
        let t = Se3::from_parts(
            Vector3::new(0.3, -0.5, 0.7),
            UnitQuaternion::from_euler_angles(0.4, -0.9, 1.3),
        );
        let m = col_major_from_se3(&t);
        // translation lands in slots 12/13/14
        assert!((m[12] - 0.3).abs() < 1e-12);
        assert!((m[13] - (-0.5)).abs() < 1e-12);
        assert!((m[14] - 0.7).abs() < 1e-12);
        let back = se3_from_col_major(&m);
        let d = (t.0.to_homogeneous() - back.0.to_homogeneous()).norm();
        assert!(d < 1e-12, "round-trip error {d:e}");
    }

    /// Engine-is-source-of-truth: the command path equals the proven engine FK.
    #[test]
    fn frames_match_engine() {
        let m = load("toy.urdf");
        let q = [0.3, -0.4];
        let out = frames_at(&m, &q);
        for (f, got) in out.iter().enumerate() {
            let want = col_major_from_se3(&fk_frame(&m, &q, f));
            for k in 0..16 {
                assert!((got[k] - want[k]).abs() < 1e-12, "frame {f} elem {k}");
            }
        }
    }

    /// The default-tip world pose round-trips through the wire format.
    #[test]
    fn tip_world_pose() {
        let m = load("toy.urdf");
        let out = frames_at(&m, &[0.0, 0.0]);
        let tip = m.tip_frame();
        // toy tip at home = (0.2, 0, 0.1)
        assert!((out[tip][12] - 0.2).abs() < 1e-12);
        assert!(out[tip][13].abs() < 1e-12);
        assert!((out[tip][14] - 0.1).abs() < 1e-12);
    }

    /// IK through the command boundary (incl. [16]→Se3 reorthonormalisation).
    #[test]
    fn ik_command_roundtrip() {
        let m = load("toy.urdf");
        let tip = m.tip_frame();
        let q_true = [0.6, -0.9];
        let target = col_major_from_se3(&fk_frame(&m, &q_true, tip));
        let req = IkRequest {
            target,
            seed: vec![0.0; m.ndof],
            frame: None,
        };
        let goal = se3_from_col_major(&req.target);
        let res = ik(&m, tip, &goal, &req.seed, &IkOpts::default());
        assert!(res.success, "ik converged");
        let reached = fk_frame(&m, &res.q, tip);
        let d = (reached.0.to_homogeneous() - goal.0.to_homogeneous()).norm();
        assert!(d < 1e-8, "FK(IK) error {d:e}");
    }

    /// Topology: rods form a connected stick figure on the branched fixture.
    #[test]
    fn branched_topology() {
        let m = load("branched.urdf");
        let info = robot_info_from_model(&m);
        // root draws no rod; everyone else points at a valid lower-or-root frame.
        let root = info.frames.iter().position(|f| f.parent == -1).unwrap();
        for (i, f) in info.frames.iter().enumerate() {
            if i == root {
                continue;
            }
            assert!(f.parent >= 0, "frame {i} has no parent");
            assert!((f.parent as usize) < info.frames.len());
        }
        // exactly ndof frames carry a joint marker (one primary per movable joint)
        let markers = info.frames.iter().filter(|f| f.joint_index >= 0).count();
        assert_eq!(markers, m.ndof);
    }

    #[test]
    fn showcase6_loads() {
        let m = load("showcase6.urdf");
        assert_eq!(m.ndof, 6);
        let info = robot_info_from_model(&m);
        assert_eq!(info.frames.len(), 7);
        let markers = info.frames.iter().filter(|f| f.joint_index >= 0).count();
        assert_eq!(markers, 6);
    }
}
