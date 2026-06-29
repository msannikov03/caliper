//! The deterministic executor. [`run`] validates, topo-sorts (Kahn, via
//! [`validate`]), then evaluates each node in order, dispatching every COMPUTE node
//! to an EXISTING engine fn. No new math lives here.

use crate::ir::{
    ClipData, GraphDoc, Node, NodeKind, PortValue, ReportData, pose_to_se3, se3_to_pose,
};
use crate::validate::{Diagnostics, Signal, parse_signal, validate};

use caliper_collision::{CollisionModel, WorldScene};
use caliper_dynamics::{GRAVITY_EARTH, Simulator, crba, potential_energy};
use caliper_hal::{ControlLoop, Gains, PhysicsSimBackend, TrajectorySetpoint};
use caliper_ik::{IkOpts, ik};
use caliper_kinematics::{fk_frame, fk_joints};
use caliper_model::{Model, Robot};
use caliper_motion::{
    CartesianMoveOpts, MotionLimits, MotionLimitsConfig, Trajectory, move_j, move_l,
    retime_waypoints,
};
use caliper_planning::{Planner, PlannerConfig};
use caliper_spatial::Se3;
use nalgebra::{DVector, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Sampling period (s) used to bake MoveJ/MoveL/PlanRrt trajectories into a
/// face-neutral [`ClipData`]. Matches `CartesianMoveOpts::defaults().dt`, so a
/// MoveL clip's samples land exactly on the trajectory's knots (parity).
pub const CLIP_DT: f64 = 0.01;

/// Control-loop tick period (s) for [`NodeKind::Control`] rollouts.
pub const CONTROL_DT: f64 = 1e-3;

/// Extra settle ticks appended after the reference trajectory in a Control rollout.
const CONTROL_SETTLE_TICKS: usize = 2000;

/// Cap on clip rows (decimation target) for Control / GravityDrop rollouts.
const MAX_CLIP_ROWS: usize = 2000;

/// A serde-friendly execution error carrying the failing node id + message.
#[derive(thiserror::Error, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphError {
    /// The graph did not pass [`validate`]; run never started.
    #[error("graph validation failed (see diagnostics)")]
    Validation { diagnostics: Diagnostics },
    /// A node's compute step failed; downstream nodes were not run.
    #[error("node `{node_id}` failed: {message}")]
    Node { node_id: String, message: String },
}

/// One extracted [`NodeKind::Scope`] series.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ScopeSeries {
    pub node_id: String,
    pub signal: String,
    pub t: Vec<f64>,
    pub y: Vec<f64>,
}

/// The result of a successful [`run`].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GraphResult {
    /// Per-node outputs, indexed by node id then output-port order.
    pub node_outputs: BTreeMap<String, Vec<PortValue>>,
    /// The graph's terminal clip: the `View` sink's upstream clip, else the last
    /// clip produced in topo order (`None` if the graph produced no clip).
    pub terminal_clip: Option<ClipData>,
    /// All `Scope` series, in topo order.
    pub scopes: Vec<ScopeSeries>,
    /// The validation diagnostics (always `is_ok()` on success).
    pub diagnostics: Diagnostics,
}

/// What a single node evaluation produced.
enum Outcome {
    /// Source / compute outputs (by out-port order).
    Outputs(Vec<PortValue>),
    /// A `View` sink captured this terminal clip.
    Terminal(ClipData),
    /// A `Scope` sink extracted this series.
    Scope(ScopeSeries),
}

/// Validate, then evaluate the graph deterministically. Bails with
/// [`GraphError::Validation`] if validation fails; otherwise dispatches each node
/// to its engine fn. The first node compute failure halts the run (its downstream
/// subtree, being later in topo order, never executes) and returns
/// [`GraphError::Node`].
pub fn run(doc: &GraphDoc, robot: &Robot) -> Result<GraphResult, GraphError> {
    let model_owned = robot.model.clone();
    let diagnostics = validate(doc, &model_owned);
    if !diagnostics.is_ok() {
        return Err(GraphError::Validation { diagnostics });
    }
    let model = Arc::new(model_owned);

    let node_by_id: HashMap<&str, &Node> = doc.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // (to_id, in_port_idx) -> (from_id, out_port_idx). Ports already resolve
    // (validation guaranteed it), so unwrap is safe here.
    let mut feeder: HashMap<(String, usize), (String, usize)> = HashMap::new();
    for edge in &doc.edges {
        let to = node_by_id[edge.to.as_str()];
        let from = node_by_id[edge.from.as_str()];
        let in_idx = edge.to_port.resolve(&to.kind.in_port_names()).unwrap();
        let out_idx = edge.from_port.resolve(&from.kind.out_port_names()).unwrap();
        feeder.insert((edge.to.clone(), in_idx), (edge.from.clone(), out_idx));
    }

    let mut cache: HashMap<String, Vec<PortValue>> = HashMap::new();
    let mut scopes: Vec<ScopeSeries> = Vec::new();
    let mut view_clip: Option<ClipData> = None;
    let mut last_clip: Option<ClipData> = None;

    for id in &diagnostics.topo_order {
        let node = node_by_id[id.as_str()];
        let outcome =
            eval_node(node, &model, &feeder, &cache).map_err(|message| GraphError::Node {
                node_id: id.clone(),
                message,
            })?;
        match outcome {
            Outcome::Outputs(outs) => {
                for v in &outs {
                    if let PortValue::Clip(c) = v {
                        last_clip = Some(c.clone());
                    }
                }
                cache.insert(id.clone(), outs);
            }
            Outcome::Terminal(c) => view_clip = Some(c),
            Outcome::Scope(s) => scopes.push(s),
        }
    }

    let node_outputs: BTreeMap<String, Vec<PortValue>> = cache.into_iter().collect();
    Ok(GraphResult {
        node_outputs,
        terminal_clip: view_clip.or(last_clip),
        scopes,
        diagnostics,
    })
}

/// Bake a [`Trajectory`] into a face-neutral [`ClipData`] by sampling on a uniform
/// `dt` grid (the last sample lands exactly on the duration).
pub fn bake_trajectory(traj: &Trajectory, dt: f64) -> ClipData {
    let dur = traj.duration();
    let nsteps = if dt > 0.0 && dur > 0.0 {
        (dur / dt).ceil() as usize
    } else {
        0
    };
    let mut c = ClipData::default();
    for k in 0..=nsteps {
        let t = (k as f64 * dt).min(dur);
        let s = traj.sample(t);
        c.times.push(t);
        c.qs.push(s.q);
        c.qds.push(s.qd);
    }
    c
}

// ===== node evaluation =====

fn eval_node(
    node: &Node,
    model: &Arc<Model>,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Result<Outcome, String> {
    let m: &Model = model;
    match &node.kind {
        NodeKind::StartConfig { q } | NodeKind::NamedConfig { q, .. } => {
            Ok(Outcome::Outputs(vec![PortValue::Config(q.clone())]))
        }
        NodeKind::GoalPose { m: pose } => Ok(Outcome::Outputs(vec![PortValue::Pose(*pose)])),

        NodeKind::Ik { frame, seed } => {
            let pose = req_pose(node, "pose", feeder, cache)?;
            let target = pose_to_se3(&pose);
            let f = resolve_frame(m, frame);
            let seed_q = match opt_config(node, "seed", feeder, cache) {
                Some(s) => s,
                None => seed.clone().unwrap_or_else(|| vec![0.0; m.ndof]),
            };
            if seed_q.len() != m.ndof {
                return Err(format!(
                    "ik seed length {} != ndof {}",
                    seed_q.len(),
                    m.ndof
                ));
            }
            let res = ik(m, f, &target, &seed_q, &IkOpts::default());
            if !res.success {
                return Err(format!(
                    "IK did not converge (residual {:.3e})",
                    res.residual
                ));
            }
            Ok(Outcome::Outputs(vec![PortValue::Config(res.q)]))
        }

        NodeKind::MoveJ {} => {
            let start = req_config(node, "start", feeder, cache)?;
            let goal = req_config(node, "goal", feeder, cache)?;
            let limits = limits_of(m)?;
            let traj = move_j(m, &start, &goal, &limits).map_err(|e| e.to_string())?;
            Ok(Outcome::Outputs(vec![PortValue::Clip(bake_trajectory(
                &traj, CLIP_DT,
            ))]))
        }

        NodeKind::MoveL { frame } => {
            let start = req_config(node, "start", feeder, cache)?;
            let goal = req_pose(node, "goal", feeder, cache)?;
            let f = resolve_frame(m, frame);
            let limits = limits_of(m)?;
            let opts = CartesianMoveOpts::defaults(limits);
            let goal_se3 = pose_to_se3(&goal);
            let traj = move_l(m, f, &start, &goal_se3, &opts).map_err(|e| e.to_string())?;
            Ok(Outcome::Outputs(vec![PortValue::Clip(bake_trajectory(
                &traj, CLIP_DT,
            ))]))
        }

        NodeKind::PlanRrt {
            seed,
            ground,
            boxes,
        } => {
            let start = req_config(node, "start", feeder, cache)?;
            let goal = req_input(node, "goal", feeder, cache)?;
            let scene = build_scene(*ground, boxes);
            let cfg = PlannerConfig {
                seed: *seed,
                ..PlannerConfig::default()
            };
            let planner = Planner::new(model.clone(), scene, cfg);
            let path = match goal {
                PortValue::Config(goal_q) => {
                    planner.plan(&start, &goal_q).map_err(|e| e.to_string())?
                }
                PortValue::Pose(pm) => {
                    let goal_se3 = pose_to_se3(&pm);
                    planner
                        .plan_to_pose(&start, &goal_se3, m.tip_frame(), &start)
                        .map_err(|e| e.to_string())?
                }
                other => {
                    return Err(format!(
                        "PlanRrt goal must be Config or Pose, got {:?}",
                        other.port_type()
                    ));
                }
            };
            let limits = limits_of(m)?;
            let traj = retime_waypoints(&path, &limits, CLIP_DT).map_err(|e| e.to_string())?;
            Ok(Outcome::Outputs(vec![PortValue::Clip(bake_trajectory(
                &traj, CLIP_DT,
            ))]))
        }

        NodeKind::Control { kp, kd } => {
            let start = req_config(node, "start", feeder, cache)?;
            let goal = req_config(node, "goal", feeder, cache)?;
            let clip = control_rollout(model, &start, &goal, *kp, *kd)?;
            Ok(Outcome::Outputs(vec![PortValue::Clip(clip)]))
        }

        NodeKind::GravityDrop {
            gravity,
            duration,
            dt,
        } => {
            let start = req_config(node, "start", feeder, cache)?;
            let g = gravity
                .as_ref()
                .map(|a| Vector3::new(a[0], a[1], a[2]))
                .unwrap_or(GRAVITY_EARTH);
            let clip = gravity_drop(model, &start, g, *duration, *dt)?;
            Ok(Outcome::Outputs(vec![PortValue::Clip(clip)]))
        }

        NodeKind::CollisionCheck { ground, boxes } => {
            let q = req_config(node, "config", feeder, cache)?;
            let scene = build_scene(*ground, boxes);
            let cm = CollisionModel::new(model.clone(), scene, 0.0);
            let rep = cm.query(&q).map_err(|e| e.to_string())?;
            let report = ReportData {
                collision: rep.has_collision(),
                pairs: rep.self_pairs,
                world_hits: rep.world_hits,
                colliding_frames: rep.colliding_frames,
                uncovered_frames: cm.uncovered_frames(),
            };
            Ok(Outcome::Outputs(vec![PortValue::Report(report)]))
        }

        NodeKind::View {} => {
            let clip = req_clip(node, "clip", feeder, cache)?;
            Ok(Outcome::Terminal(clip))
        }

        NodeKind::Scope { signal } => {
            let clip = req_clip(node, "clip", feeder, cache)?;
            let sig =
                parse_signal(signal).ok_or_else(|| format!("unknown scope signal `{signal}`"))?;
            let y = extract_series(m, &clip, sig)?;
            Ok(Outcome::Scope(ScopeSeries {
                node_id: node.id.clone(),
                signal: signal.clone(),
                t: clip.times.clone(),
                y,
            }))
        }
    }
}

// ===== engine-fn wrappers =====

fn limits_of(model: &Model) -> Result<MotionLimits, String> {
    MotionLimits::from_model(model, &MotionLimitsConfig::default()).map_err(|e| e.to_string())
}

fn control_rollout(
    model: &Arc<Model>,
    start: &[f64],
    goal: &[f64],
    kp: f64,
    kd: f64,
) -> Result<ClipData, String> {
    let m: &Model = model;
    let n = m.ndof;
    let limits = limits_of(m)?;
    let traj = move_j(m, start, goal, &limits).map_err(|e| e.to_string())?;
    let mut backend = PhysicsSimBackend::new(model.clone()).map_err(|e| e.to_string())?;
    backend
        .set_state(start, &vec![0.0; n])
        .map_err(|e| e.to_string())?;
    let mut loopy = ControlLoop::new(backend, model.clone(), CONTROL_DT)
        .map_err(|e| e.to_string())?
        .with_gains(Gains { kp, kd });
    let ticks = (traj.duration() / CONTROL_DT).ceil() as usize + CONTROL_SETTLE_TICKS;
    let mut sp = TrajectorySetpoint::new(traj);
    let frames = loopy
        .run_record(&mut sp, ticks)
        .map_err(|e| e.to_string())?;

    let stride = (frames.len() / MAX_CLIP_ROWS).max(1);
    let mut clip = ClipData::default();
    for (i, f) in frames.iter().enumerate() {
        if i % stride == 0 || i + 1 == frames.len() {
            clip.times.push(f.t);
            clip.qs.push(f.measured.clone());
            clip.qds.push(f.measured_qd.clone());
        }
    }
    Ok(clip)
}

fn gravity_drop(
    model: &Arc<Model>,
    start: &[f64],
    gravity: Vector3<f64>,
    duration: f64,
    dt: f64,
) -> Result<ClipData, String> {
    let n = model.ndof;
    let mut sim = Simulator::new(model.clone()).map_err(|e| e.to_string())?;
    sim.set_gravity(gravity);
    sim.reset_to(start, &vec![0.0; n])
        .map_err(|e| e.to_string())?;
    let nsteps = (duration / dt).ceil() as usize;
    let stride = (nsteps / MAX_CLIP_ROWS).max(1);
    let mut clip = ClipData::default();
    // initial sample
    clip.times.push(0.0);
    clip.qs.push(sim.q().to_vec());
    clip.qds.push(sim.qd().to_vec());
    for k in 1..=nsteps {
        sim.step(dt).map_err(|e| e.to_string())?;
        if k % stride == 0 || k == nsteps {
            clip.times.push(sim.time());
            clip.qs.push(sim.q().to_vec());
            clip.qds.push(sim.qd().to_vec());
        }
    }
    Ok(clip)
}

fn extract_series(model: &Model, clip: &ClipData, sig: Signal) -> Result<Vec<f64>, String> {
    match sig {
        Signal::Time => Ok(clip.times.clone()),
        Signal::Q(i) => clip
            .qs
            .iter()
            .map(|r| {
                r.get(i)
                    .copied()
                    .ok_or_else(|| "q index out of range".into())
            })
            .collect(),
        Signal::Qd(i) => clip
            .qds
            .iter()
            .map(|r| {
                r.get(i)
                    .copied()
                    .ok_or_else(|| "qd index out of range".into())
            })
            .collect(),
        Signal::TipX | Signal::TipY | Signal::TipZ => {
            let k = match sig {
                Signal::TipX => 0,
                Signal::TipY => 1,
                _ => 2,
            };
            let f = model.tip_frame();
            Ok(clip
                .qs
                .iter()
                .map(|r| fk_frame(model, r, f).translation()[k])
                .collect())
        }
        Signal::Energy => clip
            .qs
            .iter()
            .zip(&clip.qds)
            .map(|(q, qd)| energy_at(model, q, qd))
            .collect(),
    }
}

fn energy_at(model: &Model, q: &[f64], qd: &[f64]) -> Result<f64, String> {
    let mm = crba(model, q).map_err(|e| e.to_string())?;
    let qdv = DVector::from_row_slice(qd);
    let ke = 0.5 * (qdv.transpose() * &mm * &qdv)[(0, 0)];
    let mut jw = vec![Se3::identity(); model.ndof];
    fk_joints(model, q, &mut jw);
    let pe = potential_energy(model, &jw, &GRAVITY_EARTH);
    Ok(ke + pe)
}

// ===== input resolution helpers =====

fn resolve_frame(model: &Model, frame: &Option<String>) -> usize {
    match frame {
        None => model.tip_frame(),
        Some(name) => model.frame_id(name).unwrap_or_else(|| model.tip_frame()),
    }
}

fn build_scene(ground: Option<f64>, boxes: &[([f64; 3], [f64; 3])]) -> WorldScene {
    let mut s = WorldScene::new();
    if let Some(z) = ground {
        s = s.with_ground(z);
    }
    for (c, h) in boxes {
        s = s.add_box(*c, *h);
    }
    s
}

fn req_input(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Result<PortValue, String> {
    opt_input(node, port, feeder, cache)
        .ok_or_else(|| format!("missing required input on port `{port}`"))
}

fn opt_input(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Option<PortValue> {
    let names = node.kind.in_port_names();
    let idx = names.iter().position(|p| *p == port)?;
    let (fid, oidx) = feeder.get(&(node.id.clone(), idx))?;
    cache.get(fid).and_then(|v| v.get(*oidx)).cloned()
}

fn req_config(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Result<Vec<f64>, String> {
    match req_input(node, port, feeder, cache)? {
        PortValue::Config(q) => Ok(q),
        other => Err(format!(
            "port `{port}` expected Config, got {:?}",
            other.port_type()
        )),
    }
}

fn opt_config(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Option<Vec<f64>> {
    match opt_input(node, port, feeder, cache)? {
        PortValue::Config(q) => Some(q),
        _ => None,
    }
}

fn req_pose(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Result<[f64; 16], String> {
    match req_input(node, port, feeder, cache)? {
        PortValue::Pose(p) => Ok(p),
        other => Err(format!(
            "port `{port}` expected Pose, got {:?}",
            other.port_type()
        )),
    }
}

fn req_clip(
    node: &Node,
    port: &str,
    feeder: &HashMap<(String, usize), (String, usize)>,
    cache: &HashMap<String, Vec<PortValue>>,
) -> Result<ClipData, String> {
    match req_input(node, port, feeder, cache)? {
        PortValue::Clip(c) => Ok(c),
        other => Err(format!(
            "port `{port}` expected Clip, got {:?}",
            other.port_type()
        )),
    }
}

/// Convenience for callers/tests: a `Pose` value from an `Se3`.
pub fn pose_value(t: &Se3) -> PortValue {
    PortValue::Pose(se3_to_pose(t))
}
