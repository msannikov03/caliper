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
    move_j, move_l, CartesianMoveOpts, MotionLimits, MotionLimitsConfig, PoseLibrary,
};
use caliper::spatial::Se3;
use caliper_collision::{CollisionModel, WorldScene};
use nalgebra::{
    Cholesky, DVector, Isometry3, Matrix3, SymmetricEigen, Translation3, UnitQuaternion, Vector3,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

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

    let mut sim = Simulator::new(Arc::new(model.clone())).map_err(|e| e.to_string())?;
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
    let gravity = req
        .gravity
        .map(|g| Vector3::new(g[0], g[1], g[2]))
        .unwrap_or(GRAVITY_EARTH);

    let arc = Arc::new(model.clone());
    let mut backend = PhysicsSimBackend::new(arc.clone()).map_err(|e| e.to_string())?;
    backend
        .set_state(&req.q_start, &vec![0.0; n])
        .map_err(|e| e.to_string())?;
    let ctrl_dt = 1e-3;
    let mut loopy = ControlLoop::new(backend, arc, ctrl_dt)
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
            check_collision
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
}
