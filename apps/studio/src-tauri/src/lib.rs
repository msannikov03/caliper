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
use caliper::dynamics::{Simulator, GRAVITY_EARTH};
use caliper::hal::{ControlLoop, Gains, HoldSetpoint, PhysicsSimBackend, RobotBackend};
use caliper::ik::{ik, IkOpts};
use caliper::kinematics::{
    fk_frame, fk_joints, frame_pose, jacobian, JacFrame, Jacobian, SingularityGovernor,
    SingularityKind, SingularityParams,
};
use caliper::model::{JointKind, Model};
use caliper::motion::{
    move_j, move_l, retime_waypoints, CartesianMoveOpts, MotionLimits, MotionLimitsConfig,
    PoseLibrary,
};
use caliper::planning::reach::{ReachChecker, ReachConfig, ReachStatus};
use caliper::planning::{Planner, PlannerConfig};
use caliper::spatial::Se3;
use caliper_collision::{CollisionModel, WorldScene};
use nalgebra::{
    Cholesky, DVector, Isometry3, Matrix3, SymmetricEigen, Translation3, UnitQuaternion, Vector3,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::Manager;

/// Loaded robot, shared across commands. `None` until `robot_info` succeeds.
#[derive(Default)]
struct AppState {
    model: Mutex<Option<Model>>,
    poses: Mutex<PoseLibrary>,
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
    /// Parallel to `jointNames`: URDF velocity limit or null.
    #[serde(rename = "velLimit")]
    vel_limit: Vec<Option<f64>>,
    /// True iff every link carried `<inertial>` (the Simulate button gates on this).
    #[serde(rename = "hasInertia")]
    has_inertia: bool,
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

/// Singularity + manipulability report for the HUD and the tip ellipsoid.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SingularityReportDto {
    manipulability: f64,
    /// None == ∞ (singular). Frontend renders "∞".
    condition_number: Option<f64>,
    sigma_min: f64,
    /// three smallest singular values, ascending.
    sigma: [f64; 3],
    /// "none" | "wrist" | "elbow" | "boundary".
    kind: String,
    /// joint indices dominating the escape direction.
    offending_joints: Vec<usize>,
    /// engine's σ_min activation threshold; HUD uses it for the distance bar.
    eps_activate: f64,
    /// URDF-world tip origin (ellipsoid center; frontend re-nests under DISPLAY_UP).
    tip_world: [f64; 3],
    /// 3 unit principal axes (each a column), URDF world.
    ellipsoid_axes: [[f64; 3]; 3],
    /// principal radii = sqrt(eig(Jv·Jvᵀ)) = linear singular values.
    ellipsoid_radii: [f64; 3],
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
        vel_limit: model.vel_limit.clone(),
        has_inertia: model.has_inertia,
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
    [
        "showcase6",
        "dyn_pendulum2",
        "toy",
        "prismatic",
        "branched",
        "redundant7",
    ]
    .iter()
    .map(|n| (n.to_string(), format!("{root}/{n}.urdf")))
    .collect()
}

/// Baked frame matrices + tip XYZ at `q` (shared by sim_drop with `frames_at`).
fn bake_frame_row(model: &Model, q: &[f64]) -> (Vec<[f64; 16]>, [f64; 3]) {
    let frames = frames_at(model, q);
    let tp = fk_frame(model, q, model.tip_frame()).translation();
    (frames, tp)
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
    if let Ok(mut p) = state.poses.lock() {
        p.clear();
    }
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
    // Decode + validate the target BEFORE taking the lock: a non-finite target
    // would feed the iterative SO(3) projection / IK loop garbage, and the lock
    // must not be held across that heavy work.
    if req.target.iter().any(|x| !x.is_finite()) {
        return Err("target contains a non-finite value (NaN/Inf)".into());
    }
    let target = se3_from_col_major(&req.target);

    // Clone the model into an Arc and release the state lock BEFORE the IK loop,
    // so other commands aren't frozen.
    let arc = {
        let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
        let model = guard.as_ref().ok_or("no robot loaded")?;
        Arc::new(model.clone())
    };
    let model: &Model = arc.as_ref();

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

    let res = ik(model, frame, &target, &req.seed, &IkOpts::default());
    Ok(IkSolution {
        success: res.success,
        q: res.q,
        residual: res.residual,
    })
}

fn kind_str(k: SingularityKind) -> String {
    match k {
        SingularityKind::None => "none",
        SingularityKind::Wrist => "wrist",
        SingularityKind::Elbow => "elbow",
        SingularityKind::Boundary => "boundary",
    }
    .to_string()
}

/// Full singularity + ellipsoid report at `q`. One SVD (report) + one 3×3
/// symmetric eig (ellipsoid), both off the SAME World (LOCAL_WORLD_ALIGNED)
/// Jacobian — σ are frame-invariant, ellipsoid axes are world-frame.
#[tauri::command]
fn analyze(q: Vec<f64>, state: tauri::State<'_, AppState>) -> Result<SingularityReportDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if q.len() != model.ndof {
        return Err(format!(
            "expected {} joint values, got {}",
            model.ndof,
            q.len()
        ));
    }
    // clamp does not sanitize NaN (clamp(NaN)=NaN, None-limit joints skip), and a
    // non-finite Jacobian hangs nalgebra's SVD — reject before any analysis work.
    if q.iter().any(|x| !x.is_finite()) {
        return Err("q contains a non-finite value (NaN/Inf)".into());
    }
    let mut q = q;
    model.clamp(&mut q);
    let tip = model.tip_frame();

    let (ee, jw) = jacobian(model, &q, tip, JacFrame::World);
    let report = Jacobian(jw.clone()).analyze(&SingularityParams::default());

    // Translational ellipsoid: eig(Jv·Jvᵀ), Jv = top 3 rows. STATIC Matrix3 core
    // (DMatrix→Matrix3 .into() is absent in nalgebra 0.35).
    let jv = jw.rows(0, 3).into_owned(); // 3 × ndof
    let a = Matrix3::from_fn(|r, c| jv.row(r).dot(&jv.row(c)));
    let eig = SymmetricEigen::new(a);
    let mut axes = [[0.0_f64; 3]; 3];
    let mut radii = [0.0_f64; 3];
    for k in 0..3 {
        radii[k] = eig.eigenvalues[k].max(0.0).sqrt();
        let col = eig.eigenvectors.column(k);
        axes[k] = [col[0], col[1], col[2]];
    }

    Ok(SingularityReportDto {
        manipulability: report.manipulability,
        condition_number: report
            .condition_number
            .is_finite()
            .then_some(report.condition_number),
        sigma_min: report.sigma_min,
        sigma: report.sigma,
        kind: kind_str(report.kind),
        offending_joints: report.offending_joints,
        eps_activate: SingularityParams::default().eps_activate,
        tip_world: ee.translation(),
        ellipsoid_axes: axes,
        ellipsoid_radii: radii,
    })
}

/// Governor-damped single-pass interactive IK for the gizmo drag. Restart-FREE
/// (restarts teleport the arm mid-drag) DLS loop using SingularityGovernor's C¹
/// damping ramp, loose tol, step clamp — stable when pulled past a singularity.
#[tauri::command]
fn solve_ik_governed(
    req: IkRequest,
    state: tauri::State<'_, AppState>,
) -> Result<IkSolution, String> {
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
    if req
        .seed
        .iter()
        .chain(req.target.iter())
        .any(|x| !x.is_finite())
    {
        return Err("seed/target contains a non-finite value (NaN/Inf)".into());
    }
    let target = se3_from_col_major(&req.target);
    let gov = SingularityGovernor::new(SingularityParams::default());
    let n = model.ndof;
    let mut q = req.seed.clone();
    let mut residual = f64::INFINITY;
    for _ in 0..12 {
        let t_cur = fk_frame(model, &q, frame);
        let e_tw = Se3(t_cur.0.inverse() * target.0).log().0; // [v; ω], body frame
        let e = DVector::from_iterator(6, e_tw.iter().copied());
        residual = e.norm();
        if residual < 1e-6 {
            break;
        }
        let (_, j) = jacobian(model, &q, frame, JacFrame::Body);
        let sigma_min = Jacobian(j.clone())
            .analyze(&SingularityParams::default())
            .sigma_min;
        let lambda2 = gov.damping_sq(sigma_min).max(1e-10); // C¹ ramp; floor keeps SPD
        let jt = j.transpose();
        let mut h = &jt * &j;
        for i in 0..n {
            h[(i, i)] += lambda2;
        }
        let g = &jt * &e;
        let mut dq = match Cholesky::new(h) {
            Some(c) => c.solve(&g),
            None => DVector::zeros(n),
        };
        let mx = dq.amax();
        if mx > 0.3 {
            dq *= 0.3 / mx;
        }
        for i in 0..n {
            let (lo, hi) = model.limits.get(i).and_then(|l| *l).unwrap_or((-1e6, 1e6));
            q[i] = (q[i] + dq[i]).clamp(lo, hi);
        }
    }
    Ok(IkSolution {
        success: residual < 1e-3,
        q,
        residual,
    })
}

// ===== Phase 3: motion planning + named poses =====

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrajectoryDto {
    kind: String, // "moveJ" | "moveL"
    duration: f64,
    ndof: usize,
    dt: f64,
    times: Vec<f64>,
    q: Vec<Vec<f64>>,            // N x ndof — drives playback
    qd: Vec<Vec<f64>>,           // N x ndof
    tip_path: Vec<[f64; 3]>,     // URDF-world tip XYZ (frontend re-nests under DISPLAY_UP)
    frames: Vec<Vec<[f64; 16]>>, // N x nframes col-major — baked so playback is render-only
    /// false = best-effort prefix (Cartesian truncated at the wall).
    ok: bool,
    /// path fraction realized (1.0 = full).
    reached: f64,
    max_jerk_ratio: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NamedPoseDto {
    name: String,
    q: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanMoveJReq {
    q_start: Vec<f64>,
    q_goal: Vec<f64>,
    dt: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanMoveLReq {
    q_start: Vec<f64>,
    target: [f64; 16],
    frame: Option<String>,
    dt: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanMoveToPoseReq {
    q_start: Vec<f64>,
    name: String,
    dt: Option<f64>,
}

/// Bake every frame's world matrix along the trajectory so the UI plays back
/// render-only (no per-frame engine round-trips). Uses the SAME FK as get_frames.
fn sample_to_dto(
    model: &Model,
    traj: &caliper::motion::Trajectory,
    dt: f64,
    tip: usize,
    kind: &str,
) -> TrajectoryDto {
    // clamp the sample period: dt<=0/NaN would make n overflow / allocate unbounded.
    let dt = if dt.is_finite() && dt > 1e-4 {
        dt.min(10.0)
    } else {
        0.02
    };
    let dur = traj.duration();
    let n = ((dur / dt).ceil() as usize).max(1) + 1;
    let mut times = vec![];
    let mut q = vec![];
    let mut qd = vec![];
    let mut tip_path = vec![];
    let mut frames = vec![];
    let mut max_jerk_ratio = 0.0f64;
    let lim = traj.limits();
    let mut prev_qdd: Option<Vec<f64>> = None;
    for k in 0..n {
        let t = (k as f64 * dt).min(dur);
        let s = traj.sample(t);
        times.push(t);
        frames.push(frames_at(model, &s.q));
        let tp = fk_frame(model, &s.q, tip).translation();
        tip_path.push([tp[0], tp[1], tp[2]]);
        if let Some(p) = &prev_qdd {
            for (i, (&cur, &prev)) in s.qdd.iter().zip(p.iter()).enumerate() {
                let jerk = (cur - prev) / dt;
                max_jerk_ratio = max_jerk_ratio.max(jerk.abs() / lim.jmax[i]);
            }
        }
        prev_qdd = Some(s.qdd.clone());
        q.push(s.q);
        qd.push(s.qd);
    }
    TrajectoryDto {
        kind: kind.into(),
        duration: dur,
        ndof: traj.ndof(),
        dt,
        times,
        q,
        qd,
        tip_path,
        frames,
        ok: traj.completed,
        reached: traj.reached,
        max_jerk_ratio,
    }
}

fn default_limits(model: &Model) -> Result<MotionLimits, String> {
    MotionLimits::from_model(model, &MotionLimitsConfig::default()).map_err(|e| e.to_string())
}

#[tauri::command]
fn plan_move_j(
    req: PlanMoveJReq,
    state: tauri::State<'_, AppState>,
) -> Result<TrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if req.q_start.len() != model.ndof || req.q_goal.len() != model.ndof {
        return Err(format!("expected {} joint values", model.ndof));
    }
    if req
        .q_start
        .iter()
        .chain(req.q_goal.iter())
        .any(|x| !x.is_finite())
    {
        return Err("q_start/q_goal contains a non-finite value".into());
    }
    let limits = default_limits(model)?;
    let traj = move_j(model, &req.q_start, &req.q_goal, &limits).map_err(|e| e.to_string())?;
    Ok(sample_to_dto(
        model,
        &traj,
        req.dt.unwrap_or(0.02),
        model.tip_frame(),
        "moveJ",
    ))
}

#[tauri::command]
fn plan_move_l(
    req: PlanMoveLReq,
    state: tauri::State<'_, AppState>,
) -> Result<TrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if req.q_start.len() != model.ndof {
        return Err(format!("expected {} joint values", model.ndof));
    }
    if req.q_start.iter().any(|x| !x.is_finite()) || req.target.iter().any(|x| !x.is_finite()) {
        return Err("q_start/target contains a non-finite value".into());
    }
    let frame = match req.frame {
        Some(name) => model
            .frame_id(&name)
            .ok_or_else(|| format!("unknown frame `{name}`"))?,
        None => model.tip_frame(),
    };
    let goal = se3_from_col_major(&req.target);
    let limits = default_limits(model)?;
    let opts = CartesianMoveOpts::defaults(limits);
    // best-effort: a truncated prefix returns Ok with ok:false; only hard errors Err.
    let traj = move_l(model, frame, &req.q_start, &goal, &opts).map_err(|e| e.to_string())?;
    Ok(sample_to_dto(
        model,
        &traj,
        req.dt.unwrap_or(0.02),
        model.tip_frame(),
        "moveL",
    ))
}

#[tauri::command]
fn save_pose(name: String, q: Vec<f64>, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if q.len() != model.ndof {
        return Err(format!("expected {} joint values", model.ndof));
    }
    if q.iter().any(|x| !x.is_finite()) {
        return Err("pose contains a non-finite value (NaN/Inf)".into());
    }
    state
        .poses
        .lock()
        .map_err(|_| "pose lock poisoned")?
        .upsert(name, q);
    Ok(())
}

#[tauri::command]
fn list_poses(state: tauri::State<'_, AppState>) -> Result<Vec<NamedPoseDto>, String> {
    Ok(state
        .poses
        .lock()
        .map_err(|_| "pose lock poisoned")?
        .list()
        .iter()
        .map(|p| NamedPoseDto {
            name: p.name.clone(),
            q: p.q.clone(),
        })
        .collect())
}

#[tauri::command]
fn delete_pose(name: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    state
        .poses
        .lock()
        .map_err(|_| "pose lock poisoned")?
        .remove(&name);
    Ok(())
}

#[tauri::command]
fn plan_move_to_pose(
    req: PlanMoveToPoseReq,
    state: tauri::State<'_, AppState>,
) -> Result<TrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    // clone q_goal in a tight scope so we never hold both locks across planning.
    let q_goal = {
        let p = state.poses.lock().map_err(|_| "pose lock poisoned")?;
        p.get(&req.name)
            .ok_or_else(|| format!("unknown pose `{}`", req.name))?
            .q
            .clone()
    };
    if req.q_start.len() != model.ndof || q_goal.len() != model.ndof {
        return Err(format!("expected {} joint values", model.ndof));
    }
    if req
        .q_start
        .iter()
        .chain(q_goal.iter())
        .any(|x| !x.is_finite())
    {
        return Err("q_start/pose contains a non-finite value".into());
    }
    let limits = default_limits(model)?;
    let traj = move_j(model, &req.q_start, &q_goal, &limits).map_err(|e| e.to_string())?;
    Ok(sample_to_dto(
        model,
        &traj,
        req.dt.unwrap_or(0.02),
        model.tip_frame(),
        "moveJ",
    ))
}

// ===== Phase 4: dynamics + gravity simulation =====

/// A baked gravity/torque rollout. Superset of TrajectoryDto so the frontend can
/// replay it through the SAME Phase-3 playback clock (baked frames, render-only).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SimTrajectoryDto {
    kind: String, // "sim"
    duration: f64,
    ndof: usize,
    dt: f64,
    times: Vec<f64>,
    q: Vec<Vec<f64>>,
    qd: Vec<Vec<f64>>,
    tip_path: Vec<[f64; 3]>,
    frames: Vec<Vec<[f64; 16]>>,
    energy: Vec<f64>,
    energy_drift: f64,
    settled: bool,
    gravity: [f64; 3],
    damping: f64,
    // playback-union fields so the frontend treats it as a TrajectoryDto:
    ok: bool,
    reached: f64,
    max_jerk_ratio: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SimDropReq {
    q_start: Vec<f64>,
    tau: Option<Vec<f64>>,
    gravity: Option<[f64; 3]>,
    damping: Option<f64>,
    duration: Option<f64>,
    dt: Option<f64>,      // render dt
    step_dt: Option<f64>, // integrator dt
    settle: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DynamicsDto {
    tau: Vec<f64>,
    gravity_torque: Vec<f64>,
    mass_matrix: Option<Vec<Vec<f64>>>,
}

/// Bake a gravity/torque rollout into a render-only trajectory.
#[tauri::command]
fn sim_drop(
    req: SimDropReq,
    state: tauri::State<'_, AppState>,
) -> Result<SimTrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if !model.has_inertia {
        return Err(
            "this robot has no inertial data — load one with <inertial> (showcase6 or dyn_pendulum2)"
                .into(),
        );
    }
    let n = model.ndof;
    if req.q_start.len() != n {
        return Err(format!("expected {n} joint values"));
    }
    if req.q_start.iter().any(|x| !x.is_finite()) {
        return Err("q_start contains a non-finite value".into());
    }
    if let Some(g) = req.gravity {
        if g.iter().any(|x| !x.is_finite()) {
            return Err("gravity contains a non-finite value".into());
        }
    }
    let render_dt = req.dt.unwrap_or(0.02).clamp(1e-3, 0.1);
    let step_dt = req.step_dt.unwrap_or(1e-3).clamp(1e-4, render_dt);
    let duration = req.duration.unwrap_or(4.0).clamp(0.1, 30.0);
    let damping = req.damping.unwrap_or(0.0).max(0.0);
    let settle = req.settle.unwrap_or(true);

    // Clone the model into an Arc and release the state lock BEFORE the heavy
    // (up to ~30 s) rollout, so other commands aren't frozen.
    let arc = Arc::new(model.clone());
    drop(guard);
    let model: &Model = arc.as_ref();
    let mut sim = Simulator::new(arc.clone()).map_err(|e| e.to_string())?;
    sim.h_max = step_dt;
    sim.set_gravity(
        req.gravity
            .map(|g| Vector3::new(g[0], g[1], g[2]))
            .unwrap_or(GRAVITY_EARTH),
    );
    sim.set_damping(&vec![damping; n])
        .map_err(|e| e.to_string())?;
    if let Some(tau) = &req.tau {
        if tau.len() != n {
            return Err(format!("tau needs {n} values"));
        }
        sim.set_torque(tau).map_err(|e| e.to_string())?;
    }
    sim.reset_to(&req.q_start, &vec![0.0; n])
        .map_err(|e| e.to_string())?;

    let e0 = sim.total_energy();
    let nsamp = ((duration / render_dt).ceil() as usize).max(1);
    let mut times = vec![];
    let mut q = vec![];
    let mut qd = vec![];
    let mut tip_path = vec![];
    let mut frames = vec![];
    let mut energy = vec![];
    let mut settled = false;
    let mut record = |sim: &Simulator, t: f64| {
        let (fr, tp) = bake_frame_row(model, sim.q());
        times.push(t);
        q.push(sim.q().to_vec());
        qd.push(sim.qd().to_vec());
        tip_path.push(tp);
        frames.push(fr);
        energy.push(sim.total_energy());
    };
    record(&sim, 0.0);
    for _ in 0..nsamp {
        sim.step(render_dt)
            .map_err(|e| format!("simulation diverged: {e}"))?;
        let qdmax = sim.qd().iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        record(&sim, sim.time());
        if settle && damping > 0.0 && qdmax < 1e-3 && sim.time() > 0.1 {
            settled = true;
            break;
        }
    }
    let drift = (energy.last().copied().unwrap_or(e0) - e0).abs() / e0.abs().max(1e-6);
    Ok(SimTrajectoryDto {
        kind: "sim".into(),
        duration: *times.last().unwrap_or(&0.0),
        ndof: n,
        dt: render_dt,
        times,
        q,
        qd,
        tip_path,
        frames,
        energy,
        energy_drift: drift,
        settled,
        gravity: sim.gravity.into(),
        damping,
        ok: true,
        reached: 1.0,
        max_jerk_ratio: 0.0,
    })
}

/// Inverse dynamics (+ optional mass matrix) at a configuration.
#[tauri::command]
fn dynamics_at(
    q: Vec<f64>,
    qd: Option<Vec<f64>>,
    qdd: Option<Vec<f64>>,
    gravity: Option<[f64; 3]>,
    mass_matrix: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<DynamicsDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if !model.has_inertia {
        return Err("robot has no inertial data".into());
    }
    let n = model.ndof;
    if q.len() != n {
        return Err(format!("expected {n} joint values"));
    }
    let qd = qd.unwrap_or_else(|| vec![0.0; n]);
    let qdd = qdd.unwrap_or_else(|| vec![0.0; n]);
    if q.iter()
        .chain(qd.iter())
        .chain(qdd.iter())
        .any(|x| !x.is_finite())
    {
        return Err("q/qd/qdd contains a non-finite value".into());
    }
    if let Some(a) = gravity {
        if a.iter().any(|x| !x.is_finite()) {
            return Err("gravity contains a non-finite value".into());
        }
    }
    let g = gravity
        .map(|a| Vector3::new(a[0], a[1], a[2]))
        .unwrap_or(GRAVITY_EARTH);
    let z = vec![0.0; n];
    let tau = caliper::dynamics::rnea(model, &q, &qd, &qdd, &g).map_err(|e| e.to_string())?;
    let gt = caliper::dynamics::rnea(model, &q, &z, &z, &g).map_err(|e| e.to_string())?;
    let mm = if mass_matrix.unwrap_or(false) {
        let m = caliper::dynamics::crba(model, &q).map_err(|e| e.to_string())?;
        Some(
            (0..m.nrows())
                .map(|r| (0..m.ncols()).map(|c| m[(r, c)]).collect())
                .collect(),
        )
    } else {
        None
    };
    Ok(DynamicsDto {
        tau: tau.as_slice().to_vec(),
        gravity_torque: gt.as_slice().to_vec(),
        mass_matrix: mm,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ControlRunReq {
    q_start: Vec<f64>,
    goal: Vec<f64>,
    kp: Option<f64>,
    kd: Option<f64>,
    gravity: Option<[f64; 3]>,
    duration: Option<f64>,
    dt: Option<f64>, // render dt; control runs internally at 1 kHz
}

/// Drive the arm to `goal` with the computed-torque control loop, baking the
/// closed-loop motion into a render-only trajectory the frontend plays through
/// the SAME Phase-3 transport (kind = "control").
#[tauri::command]
fn control_run(
    req: ControlRunReq,
    state: tauri::State<'_, AppState>,
) -> Result<SimTrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    if !model.has_inertia {
        return Err(
            "this robot has no inertial data — load one with <inertial> (showcase6 or dyn_pendulum2)"
                .into(),
        );
    }
    let n = model.ndof;
    if req.q_start.len() != n || req.goal.len() != n {
        return Err(format!("expected {n} joint values for q_start and goal"));
    }
    if req.q_start.iter().chain(&req.goal).any(|x| !x.is_finite()) {
        return Err("q_start/goal contains a non-finite value".into());
    }
    if let Some(g) = req.gravity {
        if g.iter().any(|x| !x.is_finite()) {
            return Err("gravity contains a non-finite value".into());
        }
    }
    let render_dt = req.dt.unwrap_or(0.02).clamp(2e-3, 0.1);
    let duration = req.duration.unwrap_or(3.0).clamp(0.1, 30.0);
    let kp = req.kp.unwrap_or(100.0);
    let kd = req.kd.unwrap_or(20.0);
    if !kp.is_finite() || !kd.is_finite() || kp < 0.0 || kd < 0.0 {
        return Err("kp/kd must be finite and non-negative".into());
    }
    let gravity = req
        .gravity
        .map(|g| Vector3::new(g[0], g[1], g[2]))
        .unwrap_or(GRAVITY_EARTH);

    // Clone the model into an Arc and release the state lock BEFORE the heavy
    // (up to ~30 s) closed-loop rollout, so other commands aren't frozen.
    let arc = Arc::new(model.clone());
    drop(guard);
    let model: &Model = arc.as_ref();
    let mut backend = PhysicsSimBackend::new(arc.clone()).map_err(|e| e.to_string())?;
    backend
        .set_state(&req.q_start, &vec![0.0; n])
        .map_err(|e| e.to_string())?;
    let ctrl_dt = 1e-3;
    let mut loopy = ControlLoop::new(backend, arc.clone(), ctrl_dt)
        .map_err(|e| e.to_string())?
        .with_gains(Gains { kp, kd })
        .with_gravity(gravity);
    let mut sp = HoldSetpoint::new(req.goal.clone());
    let steps_per_sample = ((render_dt / ctrl_dt).round() as usize).max(1);
    let nsamp = ((duration / render_dt).ceil() as usize).max(1);

    let (mut times, mut qs, mut qds) = (vec![], vec![], vec![]);
    let (mut tip_path, mut frames, mut energy) = (vec![], vec![], vec![]);
    // sampler that does NOT capture the output vecs (so we can read them outside)
    #[allow(clippy::type_complexity)]
    let sample = |loopy: &mut ControlLoop<PhysicsSimBackend>| -> Result<
        (f64, Vec<f64>, Vec<f64>, [f64; 3], Vec<[f64; 16]>, f64),
        String,
    > {
        let q = loopy.backend().joint_positions();
        let qd = loopy
            .backend_mut()
            .read_state()
            .map_err(|e| e.to_string())?
            .qd_or_zero();
        let (fr, tp) = bake_frame_row(model, &q);
        let e = loopy.backend().sim().total_energy();
        Ok((loopy.time(), q, qd, tp, fr, e))
    };
    let (t, q, qd, tp, fr, e) = sample(&mut loopy)?;
    times.push(t);
    qs.push(q);
    qds.push(qd);
    tip_path.push(tp);
    frames.push(fr);
    energy.push(e);
    let mut settled = false;
    for _ in 0..nsamp {
        for _ in 0..steps_per_sample {
            loopy.step(&mut sp, None).map_err(|e| e.to_string())?;
        }
        let (t, q, qd, tp, fr, e) = sample(&mut loopy)?;
        let qdmax = qd.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        let qerr = q
            .iter()
            .zip(&req.goal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        times.push(t);
        qs.push(q);
        qds.push(qd);
        tip_path.push(tp);
        frames.push(fr);
        energy.push(e);
        if qdmax < 1e-3 && qerr < 1e-3 && loopy.time() > 0.1 {
            settled = true;
            break;
        }
    }
    let e0 = energy.first().copied().unwrap_or(0.0);
    let drift = (energy.last().copied().unwrap_or(e0) - e0).abs() / e0.abs().max(1e-6);
    Ok(SimTrajectoryDto {
        kind: "control".into(),
        duration: *times.last().unwrap_or(&0.0),
        ndof: n,
        dt: render_dt,
        times,
        q: qs,
        qd: qds,
        tip_path,
        frames,
        energy,
        energy_drift: drift,
        settled,
        gravity: gravity.into(),
        damping: 0.0,
        ok: true,
        reached: 1.0,
        max_jerk_ratio: 0.0,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollisionReq {
    q: Vec<f64>,
    ground: Option<f64>,
    margin: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CollisionDto {
    collision: bool,
    /// Names of every link frame involved in a collision (for highlighting/labels).
    colliding_frames: Vec<String>,
    self_pairs: Vec<[String; 2]>,
    world_hits: Vec<String>,
    num_colliders: usize,
    uncovered_frames: usize,
}

/// Check self/world collisions at a configuration (passive overlay query).
#[tauri::command]
fn check_collision(
    req: CollisionReq,
    state: tauri::State<'_, AppState>,
) -> Result<CollisionDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    let n = model.ndof;
    if req.q.len() != n {
        return Err(format!("expected {n} joint values"));
    }
    if req.q.iter().any(|x| !x.is_finite()) {
        return Err("q contains a non-finite value".into());
    }
    let arc = Arc::new(model.clone());
    let mut scene = WorldScene::new();
    if let Some(z) = req.ground {
        if !z.is_finite() {
            return Err("ground must be finite".into());
        }
        scene = scene.with_ground(z);
    }
    let cm = CollisionModel::new(arc, scene, req.margin.unwrap_or(0.0).max(0.0));
    let rep = cm.query(&req.q).map_err(|e| e.to_string())?;
    let name = |f: usize| model.frame_name(f).to_string();
    Ok(CollisionDto {
        collision: rep.has_collision(),
        colliding_frames: rep.colliding_frames.iter().map(|&f| name(f)).collect(),
        self_pairs: rep
            .self_pairs
            .iter()
            .map(|&(a, b)| [name(a), name(b)])
            .collect(),
        world_hits: rep.world_hits.iter().map(|&f| name(f)).collect(),
        num_colliders: cm.num_colliders(),
        uncovered_frames: cm.uncovered_frames(),
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanRunReq {
    q_start: Vec<f64>,
    goal: Option<Vec<f64>>,
    target: Option<[f64; 12]>,
    ground: Option<f64>,
    boxes: Option<Vec<([f64; 3], [f64; 3])>>,
    seed: Option<u64>,
    dt: Option<f64>, // render dt
}

fn se3_from_12(t: &[f64; 12]) -> Result<Se3, String> {
    // reject non-finite BEFORE it reaches IK/SVD (a NaN target hangs golub-reinsch,
    // which in a command would deadlock the backend — the Phase-2 lesson).
    if !t.iter().all(|x| x.is_finite()) {
        return Err("target contains a non-finite value".into());
    }
    let rot = Matrix3::new(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
    Ok(Se3(Isometry3::from_parts(
        Translation3::new(t[9], t[10], t[11]),
        UnitQuaternion::from_matrix(&rot),
    )))
}

fn scene_from(ground: Option<f64>, boxes: Option<Vec<([f64; 3], [f64; 3])>>) -> WorldScene {
    let mut s = WorldScene::new();
    if let Some(z) = ground {
        s = s.with_ground(z);
    }
    for (c, h) in boxes.unwrap_or_default() {
        s = s.add_box(c, h);
    }
    s
}

/// Plan a collision-free path to a joint goal or Cartesian target, retime it, and
/// bake it into a render-only trajectory the frontend plays through the SAME
/// Phase-3 transport (kind = "plan").
#[tauri::command]
fn plan_run(
    req: PlanRunReq,
    state: tauri::State<'_, AppState>,
) -> Result<SimTrajectoryDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    let n = model.ndof;
    if req.q_start.len() != n || !req.q_start.iter().all(|x| x.is_finite()) {
        return Err(format!("q_start needs {n} finite values"));
    }
    if req.goal.is_some() == req.target.is_some() {
        return Err("pass exactly one of goal / target".into());
    }
    let render_dt = req.dt.unwrap_or(0.02).clamp(2e-3, 0.1);
    let limits = MotionLimits::from_model(model, &MotionLimitsConfig::default())
        .map_err(|e| e.to_string())?;
    let arc = Arc::new(model.clone());
    let cfg = PlannerConfig {
        seed: req.seed.unwrap_or(0xCA11),
        ..PlannerConfig::default()
    };
    let planner = Planner::new(arc, scene_from(req.ground, req.boxes), cfg);

    let traj = if let Some(goal) = req.goal {
        if goal.len() != n {
            return Err(format!("goal needs {n} values"));
        }
        planner
            .plan_trajectory(&req.q_start, &goal, &limits, render_dt)
            .map_err(|e| e.to_string())?
    } else {
        let pose = se3_from_12(&req.target.unwrap())?;
        let f = model.tip_frame();
        let path = planner
            .plan_to_pose(&req.q_start, &pose, f, &req.q_start)
            .map_err(|e| e.to_string())?;
        retime_waypoints(&path, &limits, render_dt).map_err(|e| e.to_string())?
    };

    // bake the trajectory into a render-only clip (same shape as sim/control)
    let dur = traj.duration();
    let nsamp = ((dur / render_dt).ceil() as usize).max(1);
    let (mut times, mut q, mut qd) = (vec![], vec![], vec![]);
    let (mut tip_path, mut frames) = (vec![], vec![]);
    for k in 0..=nsamp {
        let t = (k as f64 * render_dt).min(dur);
        let s = traj.sample(t);
        let (fr, tp) = bake_frame_row(model, &s.q);
        times.push(t);
        q.push(s.q);
        qd.push(s.qd);
        tip_path.push(tp);
        frames.push(fr);
    }
    let energy = vec![0.0; times.len()];
    Ok(SimTrajectoryDto {
        kind: "plan".into(),
        duration: *times.last().unwrap_or(&0.0),
        ndof: n,
        dt: render_dt,
        times,
        q,
        qd,
        tip_path,
        frames,
        energy,
        energy_drift: 0.0,
        settled: true,
        gravity: [0.0, 0.0, -9.81],
        damping: 0.0,
        ok: true,
        reached: 1.0,
        max_jerk_ratio: 0.0,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReachReq {
    target: [f64; 12],
    ground: Option<f64>,
    boxes: Option<Vec<([f64; 3], [f64; 3])>>,
    frame: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReachDto {
    status: String,
    residual: f64,
}

/// Collision-aware reachability of a Cartesian pose (Phase 6).
#[tauri::command]
fn reach_check(req: ReachReq, state: tauri::State<'_, AppState>) -> Result<ReachDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    let f = match &req.frame {
        None => model.tip_frame(),
        Some(name) => model
            .frame_id(name)
            .ok_or(format!("unknown frame `{name}`"))?,
    };
    let pose = se3_from_12(&req.target)?;
    let arc = Arc::new(model.clone());
    let rc = ReachChecker::new(
        arc,
        scene_from(req.ground, req.boxes),
        ReachConfig {
            frame: Some(f),
            ..ReachConfig::default()
        },
    );
    let v = rc.status(&pose);
    Ok(ReachDto {
        status: match v.status {
            ReachStatus::Reachable => "reachable",
            ReachStatus::Blocked => "blocked",
            ReachStatus::Unreachable => "unreachable",
        }
        .into(),
        residual: v.residual,
    })
}

// ===== Phase 8: dataflow graph (caliper-graph executor) =====

/// One `Scope` series extracted from the graph run (camelCase for the FE).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeDto {
    node_id: String,
    signal: String,
    t: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeErrDto {
    node_id: String,
    message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EdgeErrDto {
    edge_index: usize,
    message: String,
}

/// Validation diagnostics for the FE: an explicit `ok` flag plus the engine's
/// per-node / per-edge errors, topo order, and any cycle.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticsDto {
    ok: bool,
    node_errors: Vec<NodeErrDto>,
    edge_errors: Vec<EdgeErrDto>,
    topo_order: Vec<String>,
    cycle: Vec<String>,
}

/// One output port's `{name, type}` (type ∈ config|pose|clip|report) — drives the
/// FE's edge value badges / handle colours.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PortSummaryDto {
    name: String,
    #[serde(rename = "type")]
    ty: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeSummaryDto {
    id: String,
    out_ports: Vec<PortSummaryDto>,
}

/// Full result of `graph_run`: the (optional) terminal clip baked into the EXISTING
/// `TrajectoryDto` shape so the unchanged Transport/<Canvas> plays it, plus scope
/// series, validation diagnostics, and per-node out-port summaries.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GraphRunDto {
    trajectory: Option<TrajectoryDto>,
    scopes: Vec<ScopeDto>,
    diagnostics: DiagnosticsDto,
    node_summaries: Vec<NodeSummaryDto>,
}

fn port_type_str(t: caliper::graph::PortType) -> &'static str {
    match t {
        caliper::graph::PortType::Config => "config",
        caliper::graph::PortType::Pose => "pose",
        caliper::graph::PortType::Clip => "clip",
        caliper::graph::PortType::Report => "report",
    }
}

fn diag_to_dto(d: &caliper::graph::Diagnostics) -> DiagnosticsDto {
    DiagnosticsDto {
        ok: d.is_ok(),
        node_errors: d
            .node_errors
            .iter()
            .map(|e| NodeErrDto {
                node_id: e.node_id.clone(),
                message: e.message.clone(),
            })
            .collect(),
        edge_errors: d
            .edge_errors
            .iter()
            .map(|e| EdgeErrDto {
                edge_index: e.edge_index,
                message: e.message.clone(),
            })
            .collect(),
        topo_order: d.topo_order.clone(),
        cycle: d.cycle.clone(),
    }
}

/// Map a `GraphError` to a STRUCTURED string the FE can parse: the serde JSON of
/// the error (`{"kind":"validation","diagnostics":...}` or
/// `{"kind":"node","nodeId":...,"message":...}`), falling back to its Display.
fn graph_error_str(e: &caliper::graph::GraphError) -> String {
    serde_json::to_string(e).unwrap_or_else(|_| e.to_string())
}

/// Bake a face-neutral [`caliper::graph::ClipData`] into the EXISTING `TrajectoryDto`
/// shape: FK frames + tip XYZ per sample (same `bake_frame_row` the playback path
/// expects), so the FE plays the graph's terminal clip through the unchanged
/// Phase-3 transport (kind = "graph").
fn clip_to_trajectory(model: &Model, clip: &caliper::graph::ClipData) -> TrajectoryDto {
    let ndof = clip.qs.first().map(|r| r.len()).unwrap_or(model.ndof);
    let mut tip_path = Vec::with_capacity(clip.qs.len());
    let mut frames = Vec::with_capacity(clip.qs.len());
    for q in &clip.qs {
        let (fr, tp) = bake_frame_row(model, q);
        frames.push(fr);
        tip_path.push(tp);
    }
    let duration = clip.times.last().copied().unwrap_or(0.0);
    let dt = if clip.times.len() > 1 {
        clip.times[1] - clip.times[0]
    } else {
        caliper::graph::CLIP_DT
    };
    TrajectoryDto {
        kind: "graph".into(),
        duration,
        ndof,
        dt,
        times: clip.times.clone(),
        q: clip.qs.clone(),
        qd: clip.qds.clone(),
        tip_path,
        frames,
        ok: true,
        reached: 1.0,
        max_jerk_ratio: 0.0,
    }
}

/// Run a `.caliper-graph.json` document against the loaded robot. Clones the model
/// under the state lock then RELEASES it before the (potentially heavy) run, bakes
/// the terminal clip into a render-only trajectory, and returns scopes +
/// diagnostics + per-node out-port summaries.
#[tauri::command]
fn graph_run(graph_json: String, state: tauri::State<'_, AppState>) -> Result<GraphRunDto, String> {
    let doc: caliper::graph::GraphDoc =
        serde_json::from_str(&graph_json).map_err(|e| format!("invalid graph JSON: {e}"))?;

    // Clone the cached model and release the lock BEFORE running the graph.
    let model = {
        let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
        guard.as_ref().ok_or("no robot loaded")?.clone()
    };
    let robot = caliper::model::Robot {
        name: model.name.clone(),
        joint_names: model.joint_names.clone(),
        model,
    };

    // Per-node out-port summaries (independent of execution success) for edge badges.
    let node_summaries: Vec<NodeSummaryDto> = doc
        .nodes
        .iter()
        .map(|n| NodeSummaryDto {
            id: n.id.clone(),
            out_ports: n
                .kind
                .out_ports()
                .into_iter()
                .map(|p| PortSummaryDto {
                    name: p.name.to_string(),
                    ty: port_type_str(p.ty).to_string(),
                })
                .collect(),
        })
        .collect();

    let result = caliper::graph::run(&doc, &robot).map_err(|e| graph_error_str(&e))?;
    let trajectory = result
        .terminal_clip
        .as_ref()
        .map(|c| clip_to_trajectory(&robot.model, c));
    let scopes = result
        .scopes
        .iter()
        .map(|s| ScopeDto {
            node_id: s.node_id.clone(),
            signal: s.signal.clone(),
            t: s.t.clone(),
            y: s.y.clone(),
        })
        .collect();
    let diagnostics = diag_to_dto(&result.diagnostics);
    Ok(GraphRunDto {
        trajectory,
        scopes,
        diagnostics,
        node_summaries,
    })
}

/// Validate a `.caliper-graph.json` document against the loaded model (no run).
#[tauri::command]
fn graph_validate(
    graph_json: String,
    state: tauri::State<'_, AppState>,
) -> Result<DiagnosticsDto, String> {
    let doc: caliper::graph::GraphDoc =
        serde_json::from_str(&graph_json).map_err(|e| format!("invalid graph JSON: {e}"))?;
    let model = {
        let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
        guard.as_ref().ok_or("no robot loaded")?.clone()
    };
    Ok(diag_to_dto(&caliper::graph::validate(&doc, &model)))
}

// ----- graph persistence (.caliper-graph.json under the app data dir) -----

/// Sanitize a user-supplied graph name to a safe single-segment filename stem.
fn sanitize_name(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("graph name must not be empty".into());
    }
    let s: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        return Err("graph name must not be empty".into());
    }
    // If sanitizing replaced any character, append a short hash of the ORIGINAL so
    // distinct names (e.g. "a/b" vs "a.b") never collapse to the same file.
    if s == trimmed {
        Ok(s)
    } else {
        // FNV-1a 64-bit (no dep), truncated — deterministic per original name.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in trimmed.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Ok(format!("{s}-{:08x}", (h & 0xffff_ffff)))
    }
}

/// `<app_data_dir>/graphs`, created if absent.
fn graphs_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve app data dir: {e}"))?;
    let dir = base.join("graphs");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

#[tauri::command]
fn save_graph(app: tauri::AppHandle, name: String, graph_json: String) -> Result<(), String> {
    // Reject garbage before persisting so list/load always round-trips a GraphDoc.
    let _doc: caliper::graph::GraphDoc =
        serde_json::from_str(&graph_json).map_err(|e| format!("invalid graph JSON: {e}"))?;
    let dir = graphs_dir(&app)?;
    let safe = sanitize_name(&name)?;
    let path = dir.join(format!("{safe}.caliper-graph.json"));
    std::fs::write(&path, graph_json).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn load_graph(app: tauri::AppHandle, name: String) -> Result<String, String> {
    let dir = graphs_dir(&app)?;
    let safe = sanitize_name(&name)?;
    let path = dir.join(format!("{safe}.caliper-graph.json"));
    std::fs::read_to_string(&path).map_err(|e| format!("could not load graph `{name}`: {e}"))
}

#[tauri::command]
fn list_graphs(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let dir = graphs_dir(&app)?;
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if let Some(stem) = fname.strip_suffix(".caliper-graph.json") {
            out.push(stem.to_string());
        }
    }
    out.sort();
    Ok(out)
}

#[tauri::command]
fn delete_graph(app: tauri::AppHandle, name: String) -> Result<(), String> {
    let dir = graphs_dir(&app)?;
    let safe = sanitize_name(&name)?;
    let path = dir.join(format!("{safe}.caliper-graph.json"));
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    Ok(())
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
            solve_ik,
            analyze,
            solve_ik_governed,
            plan_move_j,
            plan_move_l,
            save_pose,
            list_poses,
            delete_pose,
            plan_move_to_pose,
            sim_drop,
            dynamics_at,
            control_run,
            check_collision,
            plan_run,
            reach_check,
            graph_run,
            graph_validate,
            save_graph,
            load_graph,
            list_graphs,
            delete_graph
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

    #[test]
    fn sim_drop_falls_under_gravity() {
        let m = load("showcase6.urdf");
        assert!(m.has_inertia);
        let mut sim = Simulator::new(std::sync::Arc::new(m.clone())).unwrap();
        sim.set_damping(&vec![0.05; m.ndof]).unwrap();
        // q=0 is an EXACT gravity equilibrium for showcase6 (every link COM sits on
        // the vertical Z-stack above its joint axis → zero gravity torque), so start
        // from a tilted pose with a real moment arm.
        let q0 = vec![0.0, 0.3, 0.3, 0.0, 0.2, 0.0];
        sim.reset_to(&q0, &vec![0.0; m.ndof]).unwrap();
        let e0 = sim.total_energy();
        for _ in 0..200 {
            sim.step(0.02).unwrap();
        }
        assert!(
            sim.q()
                .iter()
                .zip(&q0)
                .any(|(&x, &x0)| (x - x0).abs() > 0.05),
            "arm did not fall"
        );
        assert!(sim.total_energy() <= e0 + 1e-6, "damped energy increased");
    }

    #[test]
    fn control_run_drives_toward_goal() {
        // mirrors the control_run command path: computed-torque loop + bake.
        let m = load("showcase6.urdf");
        let arc = std::sync::Arc::new(m.clone());
        let mut backend = PhysicsSimBackend::new(arc.clone()).unwrap();
        let goal = vec![0.2, -0.1, 0.3, 0.0, 0.1, 0.0];
        backend
            .set_state(&vec![0.0; m.ndof], &vec![0.0; m.ndof])
            .unwrap();
        let mut loopy = ControlLoop::new(backend, arc, 1e-3).unwrap();
        let mut sp = HoldSetpoint::new(goal.clone());
        loopy.run_to(&mut sp, 8000).unwrap();
        let q = loopy.backend().joint_positions();
        let err = q
            .iter()
            .zip(&goal)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-2, "control did not reach goal: err={err:e}");
        // the bake the command uses yields one render matrix per drawn frame
        let (fr, _tp) = bake_frame_row(&m, &q);
        assert_eq!(fr.len(), frames_at(&m, &q).len());
    }

    #[test]
    fn check_collision_names_folded_pair() {
        let m = load("collide_arm.urdf");
        let cm = CollisionModel::new(std::sync::Arc::new(m.clone()), WorldScene::new(), 0.0);
        let rep = cm
            .query(&[0.0, std::f64::consts::PI, std::f64::consts::PI])
            .unwrap();
        assert!(rep.has_collision());
        let names: Vec<[String; 2]> = rep
            .self_pairs
            .iter()
            .map(|&(a, b)| [m.frame_name(a).to_string(), m.frame_name(b).to_string()])
            .collect();
        assert!(
            names
                .iter()
                .any(|p| p.contains(&"l1".to_string()) && p.contains(&"l3".to_string())),
            "expected an l1<->l3 self-collision pair, got {names:?}"
        );
    }

    #[test]
    fn plan_run_avoids_box() {
        // mirrors the plan_run command path: plan with a world box present, then
        // bake — the planned (and retimed) path must be collision-free.
        let m = load("collide_arm.urdf");
        let scene = WorldScene::new()
            .with_ground(-0.1)
            .add_box([0.6, 0.0, 0.3], [0.15, 0.15, 0.15]);
        let planner = Planner::new(
            std::sync::Arc::new(m.clone()),
            scene,
            PlannerConfig::default(),
        );
        let start = vec![0.0, 0.0, 0.0];
        let goal = vec![0.4, -0.4, 0.4];
        let path = planner.plan(&start, &goal).unwrap();
        assert!(
            planner.verify_path(&path),
            "planned path must be collision-free"
        );
        // the bake the command uses yields one render matrix per drawn frame
        let (fr, _tp) = bake_frame_row(&m, &path[0]);
        assert_eq!(fr.len(), frames_at(&m, &start).len());
    }

    // ===== Phase 8: graph backend =====

    fn robot_of(name: &str) -> caliper::model::Robot {
        let m = load(name);
        caliper::model::Robot {
            name: m.name.clone(),
            joint_names: m.joint_names.clone(),
            model: m,
        }
    }

    /// The terminal clip of a MoveJ graph bakes into the EXISTING TrajectoryDto
    /// shape: one FK frame-row + tip per clip sample (what playback expects).
    #[test]
    fn graph_run_bakes_clip_to_trajectory() {
        let robot = robot_of("toy.urdf");
        let json = r#"{
            "nodes":[
                {"id":"s","kind":{"type":"startConfig","q":[0.0,0.0]}},
                {"id":"g","kind":{"type":"startConfig","q":[0.4,-0.3]}},
                {"id":"mj","kind":{"type":"moveJ"}},
                {"id":"v","kind":{"type":"view"}}
            ],
            "edges":[
                {"from":"s","fromPort":"config","to":"mj","toPort":"start"},
                {"from":"g","fromPort":"config","to":"mj","toPort":"goal"},
                {"from":"mj","fromPort":"clip","to":"v","toPort":"clip"}
            ]
        }"#;
        let doc: caliper::graph::GraphDoc = serde_json::from_str(json).unwrap();
        let res = caliper::graph::run(&doc, &robot).unwrap();
        let clip = res.terminal_clip.as_ref().expect("terminal clip");
        assert!(clip.len() > 1);
        let traj = clip_to_trajectory(&robot.model, clip);
        assert_eq!(traj.times.len(), clip.len());
        assert_eq!(traj.frames.len(), clip.len());
        assert_eq!(traj.tip_path.len(), clip.len());
        assert_eq!(traj.q.len(), clip.qs.len());
        // one render matrix per drawn frame, matching the live FK path.
        assert_eq!(
            traj.frames[0].len(),
            frames_at(&robot.model, &clip.qs[0]).len()
        );
        assert!(traj.duration > 0.0);
        assert_eq!(traj.kind, "graph");
    }

    /// A failing run returns a STRUCTURED error string (serde JSON of GraphError).
    #[test]
    fn graph_error_is_structured_json() {
        // Control on a non-inertia robot (toy) fails validation inside run().
        let robot = robot_of("toy.urdf");
        assert!(!robot.model.has_inertia);
        let json = r#"{
            "nodes":[
                {"id":"s","kind":{"type":"startConfig","q":[0.0,0.0]}},
                {"id":"g","kind":{"type":"startConfig","q":[0.1,0.1]}},
                {"id":"c","kind":{"type":"control","kp":100.0,"kd":20.0}},
                {"id":"v","kind":{"type":"view"}}
            ],
            "edges":[
                {"from":"s","fromPort":"config","to":"c","toPort":"start"},
                {"from":"g","fromPort":"config","to":"c","toPort":"goal"},
                {"from":"c","fromPort":"clip","to":"v","toPort":"clip"}
            ]
        }"#;
        let doc: caliper::graph::GraphDoc = serde_json::from_str(json).unwrap();
        let err = caliper::graph::run(&doc, &robot).unwrap_err();
        let s = graph_error_str(&err);
        assert!(
            s.contains("\"kind\":\"validation\""),
            "structured kind: {s}"
        );
        assert!(s.contains("diagnostics"), "carries diagnostics: {s}");
    }

    /// `graph_validate`'s DTO reports a clean DAG as ok with a full topo order, and
    /// a cycle as not-ok.
    #[test]
    fn diag_dto_reports_ok_and_cycle() {
        let robot = robot_of("toy.urdf");
        let ok_json = r#"{
            "nodes":[
                {"id":"s","kind":{"type":"startConfig","q":[0.0,0.0]}},
                {"id":"g","kind":{"type":"startConfig","q":[0.1,0.1]}},
                {"id":"mj","kind":{"type":"moveJ"}},
                {"id":"v","kind":{"type":"view"}}
            ],
            "edges":[
                {"from":"s","fromPort":"config","to":"mj","toPort":"start"},
                {"from":"g","fromPort":"config","to":"mj","toPort":"goal"},
                {"from":"mj","fromPort":"clip","to":"v","toPort":"clip"}
            ]
        }"#;
        let doc: caliper::graph::GraphDoc = serde_json::from_str(ok_json).unwrap();
        let d = diag_to_dto(&caliper::graph::validate(&doc, &robot.model));
        assert!(d.ok);
        assert_eq!(d.topo_order.len(), 4);
        assert!(d.cycle.is_empty());

        // two IK nodes seeding each other form a cycle.
        let cyc_json = r#"{
            "nodes":[
                {"id":"a","kind":{"type":"ik"}},
                {"id":"b","kind":{"type":"ik"}}
            ],
            "edges":[
                {"from":"a","fromPort":"config","to":"b","toPort":"seed"},
                {"from":"b","fromPort":"config","to":"a","toPort":"seed"}
            ]
        }"#;
        let doc: caliper::graph::GraphDoc = serde_json::from_str(cyc_json).unwrap();
        let d = diag_to_dto(&caliper::graph::validate(&doc, &robot.model));
        assert!(!d.ok);
        assert!(!d.cycle.is_empty());
    }

    #[test]
    fn sanitize_name_strips_unsafe() {
        // already-safe names pass through unchanged
        assert_eq!(sanitize_name("ok-name_1").unwrap(), "ok-name_1");
        assert!(sanitize_name("   ").is_err());
        assert!(sanitize_name("").is_err());
        // unsafe chars are replaced AND a hash of the original is appended so
        // distinct originals never collide onto one file
        let a = sanitize_name("a/b").unwrap();
        let b = sanitize_name("a.b").unwrap();
        assert!(a.starts_with("a_b-") && b.starts_with("a_b-"));
        assert_ne!(a, b, "distinct unsafe names must map to distinct files");
    }
}
