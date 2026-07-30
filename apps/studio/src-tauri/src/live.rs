//! Live sim session (A1): a LIVE stepped simulation the human can watch and
//! drive, replacing bake-then-replay for interactive work. One background
//! thread owns the engine (MuJoCo contact sim or the builtin fixed-base
//! dynamics) and runs the SAME `ControlLoop` PD stack as the bake paths, but
//! holding a live-mutable [`TeleopSetpoint`] target instead of a fixed goal.
//! State streams to the webview as `live://state` events at a decimated rate;
//! every session end (stop / supersede / step error) emits `live://ended`.
//!
//! Pacing is wall-clock with an accumulator: each wakeup converts elapsed wall
//! time into whole physics steps of size `h`. Catch-up debt is capped at
//! [`CATCHUP_CAP_S`] — a stall drops sim time instead of spiraling. Pause stops
//! BOTH stepping and time accumulation (the arm is frozen mid-physics, never
//! de-energized — `disable`/`estop` would let it fall).
//!
//! Teleop recording (A3) rides the same thread: while the human drives the live
//! sim, `live_record_*` streams the control loop's own [`Frame`]s into a native
//! LeRobotDataset v3.0 at an exact tick decimation — every `(1/h)/fps` ticks,
//! one frame, so per-episode timestamps are exactly `frame_index / fps` with no
//! resampling. The [`DatasetWriter`] lives in the session thread; commands hand
//! it work through [`LiveShared`] and wait for a reply slot.
//!
//! # Grasping (B1/B2), and what is honest about it
//!
//! A session detects the robot's GRIPPER JOINT
//! ([`caliper::model::gripper`]) and exposes it as one extra control:
//! `live_gripper(closed)` moves that joint's slot in the same PD hold target
//! every other input writes — there is no second command path, and on the
//! builtin engine that is ALL it does.
//!
//! On the MuJoCo engine the gripper additionally drives a WELD heuristic, and
//! it is labeled as a heuristic everywhere it surfaces: caliper does not
//! simulate finger friction (that needs actuated fingers and contact-rich
//! tuning nobody's teleop rig actually runs). Instead, when the gripper is
//! commanded CLOSED and has either reached the closed target or stalled
//! against something while closing, and some prop is genuinely in contact with
//! a robot geom, that prop is WELDED to the gripper's link — the same
//! attach-on-grasp trick teleop data-collection rigs use. Opening releases the
//! weld and the prop falls naturally. The prop's identity streams out as
//! `held`; it is NOT written into recorded datasets in this phase (state and
//! action stay pure joints).
//!
//! Two limits worth stating plainly. The contact test accepts a prop touching
//! ANY robot geom, not specifically the jaw — a prop resting against the
//! forearm when the gripper closes will be taken. And a MuJoCo weld is a soft
//! constraint, so a carried prop sags a millimetre or two under a hard swing
//! rather than tracking rigidly.
//!
//! # Verdicts (Wave C)
//!
//! A session can also carry the SUCCESS PREDICATE of a task artifact
//! ([`LiveStartReq::success`], normally handed over verbatim by `task_open`):
//! the same JSON, judged by the same evaluator
//! ([`caliper_sim_mujoco::task::success`]), that scores a python rollout. It is
//! validated in `live_start` — before any thread exists — and then judged once
//! per emitted state from the sim's own prop poses and velocities, so
//! `live://state.success` always describes the poses in that same event.
//!
//! `live_reset` starts a fresh episode for the verdict too: a `lifted` bar is
//! re-anchored at the pose the reset installed, never at the previous
//! episode's. And `success` is `null` — not `false` — whenever nothing is being
//! scored (no predicate, or the builtin engine, which has no props for a
//! predicate to be about).

use crate::policy::{PolicyBus, PolicyHandle};
use crate::{bake_frame_row, logged, AppState, PropDto, PropTrackDto};
use caliper::hal::{ControlLoop, Frame, Gains, PhysicsSimBackend, TeleopSetpoint};
use caliper::model::Model;
use caliper_dataset::{DatasetSpec, DatasetWriter, FeatureSpec};
use caliper_sim_mujoco::task::success::{Predicate, SuccessState, SuccessTracker};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Maximum wall-clock debt (s) the pacer will convert into catch-up steps.
const CATCHUP_CAP_S: f64 = 0.25;

/// Recording rate used when a `live_record_start` omits `fps`.
const DEFAULT_RECORD_FPS: u32 = 50;

/// How long a recording command waits for the session thread to service its
/// request. Generous next to a step batch (bounded by [`CATCHUP_CAP_S`]) and
/// still short enough that a wedged thread surfaces as an error, not a hang.
const REC_REPLY_TIMEOUT: Duration = Duration::from_secs(2);

/// How close to the closed target — as a fraction of the open→closed span —
/// the gripper counts as CLOSED.
const GRASP_CLOSE_FRAC: f64 = 0.25;

/// Per-tick gripper motion below this fraction of the span counts as "not
/// advancing" for the blocked-gripper test.
const GRASP_STALL_EPS_FRAC: f64 = 1e-5;

/// Consecutive non-advancing ticks that mean a closing gripper is BLOCKED.
/// 50 ticks = 50 ms at the 1 kHz tick rate: long enough not to fire on PD
/// overshoot, short enough that a grab feels immediate.
const GRASP_STALL_TICKS: u32 = 50;

/// Monotonic session ids across the process lifetime.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

// ===== wire types =====

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveStartReq {
    q0: Vec<f64>,
    /// "mujoco" | "builtin"; default mujoco when compiled, else builtin.
    engine: Option<String>,
    /// Free props (mujoco only — a clear Err with builtin).
    #[serde(default)]
    props: Vec<PropDto>,
    /// Ground plane height (mujoco; default 0.0).
    ground: Option<f64>,
    kp: Option<f64>,
    kd: Option<f64>,
    /// Requested state-event rate; clamped to [10, 120] then snapped to an
    /// integer step decimation (the DTO echoes the ACTUAL rate).
    emit_hz: Option<f64>,
    /// Override the auto-detected gripper joint BY NAME (Err if the robot has
    /// no such joint, or it has no limits to open/close between). `None` =
    /// auto-detect; a robot with no gripper simply gets no channel.
    gripper_joint: Option<String>,
    /// Which end of the gripper joint's range CLOSES it: `"lo"` (the default,
    /// and the usual convention) or `"hi"`.
    gripper_closed: Option<String>,
    /// Success predicate to judge this session against, in the
    /// `caliper-task.json` / `caliper_learn.success` JSON schema (typically
    /// straight from `task_open`). Validated in `live_start` — an unknown key,
    /// an unresolved zone name or a prop the scene does not contain is an Err
    /// before any thread exists. `None` = no verdict is computed.
    #[serde(default)]
    success: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveStartedDto {
    session_id: u64,
    engine: String,
    /// Physics timestep (s).
    h: f64,
    /// Actual emission rate after decimation: `1 / (emit_every · h)`.
    emit_hz: f64,
    ndof: usize,
    /// Static prop shape/color info in build order; `frames` empty (live poses
    /// arrive per-event, not baked).
    props: Vec<PropTrackDto>,
    /// The gripper channel this session found, or `null` when the robot has no
    /// gripper joint — in which case `live_gripper` errors and `held` is
    /// always `null`.
    gripper: Option<GripperDto>,
}

/// The gripper channel of a live session: which joint it drives and the two
/// hold-target values `live_gripper` writes into that joint's slot.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GripperDto {
    /// Caliper joint name.
    joint: String,
    /// Index of that joint in `q` / `target` / `action`.
    index: usize,
    /// Hold target for OPEN — the joint limit at the open end, pulled 2% of
    /// the range inward so the PD target never slams the mechanical stop.
    open_target: f64,
    /// Hold target for CLOSED, inset the same way.
    closed_target: f64,
}

/// Live gripper state, streamed on every `live://state`.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GripperStateEvent {
    /// The last `live_gripper` intent (true = commanded closed). This is the
    /// COMMAND, not the measurement — a gripper closed on a prop reads
    /// `closed: true` with `q` still short of `closedTarget`.
    closed: bool,
    /// Measured position of the gripper joint.
    q: f64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LiveStateEvent {
    session_id: u64,
    tick: u64,
    t: f64,
    q: Vec<f64>,
    qd: Vec<f64>,
    /// Column-major world matrices, exactly what `get_frames` returns.
    frames: Vec<[f64; 16]>,
    tip: [f64; 3],
    ncon: u32,
    /// Per-prop `[x, y, z, qw, qx, qy, qz]`, build order matching
    /// `LiveStartedDto.props`; empty for builtin.
    props: Vec<[f64; 7]>,
    paused: bool,
    target: Vec<f64>,
    /// A take is in progress (A3) — the UI badges the viewport.
    recording: bool,
    /// Frames captured in the current take; 0 when not recording.
    rec_frames: u64,
    /// Gripper command + measurement (B1); `null` when the session has no
    /// gripper channel.
    gripper: Option<GripperStateEvent>,
    /// Name of the prop currently WELDED to the gripper (B2), or `null` when
    /// nothing is held. Always `null` on the builtin engine, which has no
    /// contacts to grasp with.
    held: Option<String>,
    /// The session's success verdict for THIS state, or `null` when there is
    /// nothing to judge — no predicate was supplied, or the engine is the
    /// builtin one, which has no props for a predicate to be about. `null` is
    /// deliberately distinct from `false`: it means "not evaluated", never
    /// "not yet succeeded".
    success: Option<bool>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LiveEndedEvent {
    session_id: u64,
    /// "stopped" | "superseded" | "error: <detail>".
    reason: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveStatusDto {
    session_id: u64,
    engine: String,
    paused: bool,
    t: f64,
    tick: u64,
    ndof: usize,
    /// A take is in progress; `live_record_status` has the detail.
    recording: bool,
}

// ===== recording wire types (A3) =====

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveRecordStartReq {
    /// Dataset directory to create (first call) / continue recording into.
    root: String,
    /// Task label for THIS episode (each take carries its own).
    task: String,
    /// Recording rate; must divide the tick rate exactly. Default 50.
    fps: Option<u32>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LiveRecordStartedDto {
    root: String,
    fps: u32,
    /// Physics ticks per recorded frame: `(1/h)/fps`.
    record_every: u64,
    /// 0-based index this episode gets when saved.
    episode_index: usize,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LiveRecordStoppedDto {
    saved: bool,
    /// Index of the saved episode; null for a discarded take.
    episode_index: Option<usize>,
    /// Frames that were in the take (saved or thrown away).
    frames: usize,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LiveRecordFinishedDto {
    root: String,
    episodes: usize,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LiveRecordStatusDto {
    root: String,
    fps: u32,
    recording: bool,
    /// Task label of the take in progress; null between takes.
    task: Option<String>,
    buffered_frames: usize,
    episodes_saved: usize,
}

// ===== emission seam (testable without a Tauri app) =====

/// Where session events go. The Tauri impl forwards to the webview; tests
/// collect into vectors, so the whole loop runs headless.
pub(crate) trait LiveEmitter: Send + 'static {
    fn state(&self, ev: &LiveStateEvent);
    fn ended(&self, ev: &LiveEndedEvent);
}

struct TauriEmitter(tauri::AppHandle);
impl LiveEmitter for TauriEmitter {
    fn state(&self, ev: &LiveStateEvent) {
        let _ = tauri::Emitter::emit(&self.0, "live://state", ev);
    }
    fn ended(&self, ev: &LiveEndedEvent) {
        let _ = tauri::Emitter::emit(&self.0, "live://ended", ev);
    }
}

// ===== command <-> thread shared state (no channels) =====

#[derive(Clone, Copy, Default)]
struct LiveStatusInner {
    paused: bool,
    t: f64,
    tick: u64,
}

/// One recording command handed to the session thread. The thread services at
/// most one per loop iteration (including while paused) and answers in
/// `rec_reply`.
enum RecRequest {
    /// Begin a take. `writer` is `Some` only on the call that OPENS the
    /// dataset — it is created in the command so root/spec errors surface
    /// synchronously — and `None` for every later take on the same dataset.
    Start {
        writer: Option<Box<DatasetWriter>>,
        root: String,
        fps: u32,
        record_every: u64,
        task: String,
    },
    /// End the take: `save` writes it as an episode, else it is thrown away.
    Stop { save: bool },
    /// Close the dataset (refused mid-take).
    Finish,
}

/// The thread's answer to one [`RecRequest`].
enum RecReply {
    Started {
        episode_index: usize,
    },
    Stopped {
        saved: bool,
        episode_index: Option<usize>,
        frames: usize,
    },
    Finished {
        root: String,
        episodes: usize,
    },
}

/// The thread-owned recorder, mirrored out for commands and state events. The
/// live counters (`recording`, `rec_frames`) are atomics because every state
/// event reads them.
#[derive(Clone)]
struct RecMirror {
    root: String,
    fps: u32,
    /// `Some` while a take is recording (that episode's task label).
    task: Option<String>,
    episodes_saved: usize,
}

pub(crate) struct LiveShared {
    stop: AtomicBool,
    /// Selects the ended reason when `stop` is raised by a superseding start.
    superseded: AtomicBool,
    paused: AtomicBool,
    /// Set by the thread on an error exit — commands treat the session as gone.
    dead: AtomicBool,
    target: Mutex<Vec<f64>>,
    reset_req: Mutex<Option<Vec<f64>>>,
    status: Mutex<LiveStatusInner>,
    /// Pending recording request / its reply, each stamped with the request id
    /// that `rec_seq` handed out. One in flight at a time — the commands
    /// serialize themselves on `rec_gate` before touching either — but a
    /// command that gives up waiting does NOT stop the thread from servicing
    /// its request, so the id is what keeps a late reply from being handed to
    /// the next command.
    rec_req: Mutex<Option<(u64, RecRequest)>>,
    rec_reply: Mutex<Option<(u64, Result<RecReply, String>)>>,
    /// Hands out the request ids; only ever incremented.
    rec_seq: AtomicU64,
    /// Held by a recording command for its whole request→reply cycle. NEVER
    /// taken by the session thread, so it cannot deadlock against it.
    rec_gate: Mutex<()>,
    /// What the thread's writer looks like right now; `None` = no dataset open.
    rec_mirror: Mutex<Option<RecMirror>>,
    recording: AtomicBool,
    rec_frames: AtomicU64,
    /// Grasp INTENT: the last `live_gripper` call. The matching hold-target
    /// move is written by the same command, so `target` stays the single
    /// source of truth for what the arm is commanded to do — this flag only
    /// says which way the human meant it.
    grasp_closed: AtomicBool,
    /// Prop the session thread currently has welded (B2); thread-owned,
    /// mirrored here for state events. Always `None` on builtin.
    held: Mutex<Option<String>>,
    /// Latest-wins observation slot for a policy driving this session (E1).
    /// Inert — one relaxed load per emitted state — until a policy attaches.
    pub(crate) policy_bus: PolicyBus,
}

impl LiveShared {
    fn new(q0: Vec<f64>) -> Self {
        Self {
            stop: AtomicBool::new(false),
            superseded: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            target: Mutex::new(q0),
            reset_req: Mutex::new(None),
            status: Mutex::new(LiveStatusInner::default()),
            rec_req: Mutex::new(None),
            rec_reply: Mutex::new(None),
            rec_seq: AtomicU64::new(0),
            rec_gate: Mutex::new(()),
            rec_mirror: Mutex::new(None),
            recording: AtomicBool::new(false),
            rec_frames: AtomicU64::new(0),
            grasp_closed: AtomicBool::new(false),
            held: Mutex::new(None),
            policy_bus: PolicyBus::default(),
        }
    }

    /// Move the PD hold target. THE single source of truth for what the arm is
    /// commanded to do — the sliders, the IK gizmo, the keyboard jog,
    /// `live_gripper` and a driving policy (E1) all land here, and the last
    /// writer before a step wins.
    pub(crate) fn write_target(&self, q: Vec<f64>) {
        if let Ok(mut t) = self.target.lock() {
            *t = q;
        }
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// The session is stopping or already dead — nothing riding on it (a policy
    /// bridge, say) should keep working.
    pub(crate) fn is_gone(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.dead.load(Ordering::Relaxed)
    }
}

/// The resolved gripper channel: one joint, two hold-target values. Built in
/// `live_start` (so a bad override fails before any thread exists) and shared
/// by the command side and the session thread.
#[derive(Clone, Debug)]
pub(crate) struct GripperChannel {
    index: usize,
    name: String,
    open_target: f64,
    closed_target: f64,
}

impl GripperChannel {
    /// Distance between the two targets — the span every grasp threshold is a
    /// fraction of. Always positive.
    fn span(&self) -> f64 {
        (self.open_target - self.closed_target).abs()
    }
}

/// A running live session, held in `AppState.live`. Dropping the slot without
/// `stop` would leak the thread — every taker must raise `stop` and join.
pub(crate) struct LiveSession {
    pub(crate) id: u64,
    engine: &'static str,
    pub(crate) ndof: usize,
    /// Physics timestep (s) — recording decimates against its tick rate.
    h: f64,
    /// Actual state-emission rate — a policy's obs cadence decimates from it.
    pub(crate) emit_hz: f64,
    /// Initial pose — the `live_reset(None)` restore point.
    q0: Vec<f64>,
    /// The gripper channel, when this robot has one (B1).
    gripper: Option<GripperChannel>,
    pub(crate) shared: Arc<LiveShared>,
    /// The policy driving this session (E1), if any. Held HERE so every way a
    /// session can end takes the python child with it — dropping the handle
    /// shuts the bridge down.
    pub(crate) policy: Mutex<Option<PolicyHandle>>,
    join: Option<JoinHandle<()>>,
}

impl LiveSession {
    /// The session thread exited with an error and the slot has not been reaped
    /// yet.
    pub(crate) fn is_dead(&self) -> bool {
        self.shared.dead.load(Ordering::Relaxed)
    }
}

// ===== pure pacing helpers =====

/// Physics steps between state emissions: `max(1, round(1/(hz·h)))`.
fn emit_every(emit_hz: f64, h: f64) -> u64 {
    ((1.0 / (emit_hz * h)).round() as u64).max(1)
}

/// Requested → effective emission rate: default 60, clamped to [10, 120].
fn clamp_emit_hz(requested: Option<f64>) -> f64 {
    requested.unwrap_or(60.0).clamp(10.0, 120.0)
}

/// Wall-time → whole-step converter with a catch-up cap. Pure (no clock): feed
/// it elapsed seconds, get the number of `h`-steps to run now. Debt beyond
/// `cap` is DROPPED — a long stall skips sim time instead of spiraling.
struct Pacer {
    h: f64,
    cap: f64,
    acc: f64,
}

impl Pacer {
    fn new(h: f64, cap: f64) -> Self {
        Self { h, cap, acc: 0.0 }
    }
    fn plan(&mut self, elapsed: f64) -> u64 {
        self.acc = (self.acc + elapsed.max(0.0)).min(self.cap);
        let n = (self.acc / self.h).floor() as u64;
        self.acc -= n as f64 * self.h;
        n
    }
    fn clear(&mut self) {
        self.acc = 0.0;
    }
}

// ===== grasp heuristic (B2) =====

/// Per-tick gripper history the blocked test needs. Reset whenever the intent
/// flips or a prop is taken, so a stall only ever describes the CURRENT close.
#[derive(Default)]
struct GraspTracker {
    /// Gripper position at the previous tick.
    last_q: Option<f64>,
    /// Consecutive ticks the gripper failed to advance toward closed.
    stalled: u32,
}

impl GraspTracker {
    fn clear(&mut self) {
        self.last_q = None;
        self.stalled = 0;
    }

    /// Feed one tick's measured gripper position; returns the running stall
    /// count. "Advancing" is motion toward the closed target by more than
    /// `GRASP_STALL_EPS_FRAC` of the span — a gripper resting on a prop moves
    /// by far less than that per millisecond.
    fn observe(&mut self, q: f64, g: &GripperChannel) -> u32 {
        let toward_closed = match self.last_q {
            Some(prev) => (prev - q) * (g.open_target - g.closed_target).signum(),
            None => f64::INFINITY, // first tick: nothing to compare, assume moving
        };
        self.last_q = Some(q);
        if toward_closed > GRASP_STALL_EPS_FRAC * g.span() {
            self.stalled = 0;
        } else {
            self.stalled = self.stalled.saturating_add(1);
        }
        self.stalled
    }
}

/// Is a gripper commanded closed actually closed ON something?
///
/// Two ways to qualify, because a gripper that has caught a prop NEVER reaches
/// its closed target — the prop is in the way:
/// 1. it got within [`GRASP_CLOSE_FRAC`] of the span of the closed target, or
/// 2. it is past the halfway point toward closed AND has stopped advancing for
///    [`GRASP_STALL_TICKS`] — i.e. it is pressing on something.
///
/// Pure: no sim, no clock. The CONTACT half of the grasp condition is checked
/// separately against the engine.
fn gripper_engaged(q: f64, g: &GripperChannel, stalled: u32) -> bool {
    let span = g.span();
    if span <= 0.0 {
        return false;
    }
    if (q - g.closed_target).abs() <= GRASP_CLOSE_FRAC * span {
        return true;
    }
    let past_half = (q - g.closed_target).abs() < (q - g.open_target).abs();
    past_half && stalled >= GRASP_STALL_TICKS
}

// ===== the engine =====

enum LiveEngine {
    Builtin(ControlLoop<PhysicsSimBackend>),
    #[cfg(feature = "mujoco")]
    Mujoco(ControlLoop<caliper_sim_mujoco::MujocoBackend>),
}

impl LiveEngine {
    /// One control step. Returns the loop's own [`Frame`] — `measured` is the
    /// state read before the command (`observation.state`), `command` the
    /// commanded target (`action`) — which is exactly what recording writes.
    fn step(&mut self, sp: &mut TeleopSetpoint) -> Result<Frame, String> {
        match self {
            LiveEngine::Builtin(l) => l.step(sp, None).map_err(|e| e.to_string()),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => l.step(sp, None).map_err(|e| e.to_string()),
        }
    }

    fn tick(&self) -> u64 {
        match self {
            LiveEngine::Builtin(l) => l.tick(),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => l.tick(),
        }
    }

    fn time(&self) -> f64 {
        match self {
            LiveEngine::Builtin(l) => l.time(),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => l.time(),
        }
    }

    /// `(q, qd, ncon, prop poses)` at the current state.
    fn snapshot(&self) -> (Vec<f64>, Vec<f64>, u32, Vec<[f64; 7]>) {
        match self {
            LiveEngine::Builtin(l) => {
                let sim = l.backend().sim();
                (sim.q().to_vec(), sim.qd().to_vec(), 0, Vec::new())
            }
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => {
                let sim = l.backend().sim();
                let props = sim
                    .prop_poses()
                    .into_iter()
                    .map(|(_, p, q)| [p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
                    .collect();
                (sim.qpos(), sim.qvel(), sim.ncon() as u32, props)
            }
        }
    }

    /// The evaluator's view of the scene: every prop's world center and world
    /// LINEAR velocity, by name. Empty on builtin (no props exist), which is
    /// why a builtin session never reports a verdict.
    fn success_state(&self) -> Result<SuccessState, String> {
        match self {
            LiveEngine::Builtin(_) => Ok(SuccessState::default()),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => {
                let sim = l.backend().sim();
                let pos = sim.prop_poses().into_iter().map(|(n, p, _)| (n, p));
                // Velocities come along unconditionally: a `settled_speed`
                // predicate that could not be answered would ERROR, and the
                // session has them for free.
                let vel = sim.prop_velocities().map_err(|e| e.to_string())?;
                Ok(SuccessState::from_positions(pos).with_velocities(vel))
            }
        }
    }

    /// Weld the first prop that is genuinely touching a robot geom, and return
    /// its name. `None` = nothing to grab (and always `None` on builtin, which
    /// has no contacts at all). The weld captures the CURRENT relative pose,
    /// so the prop does not snap when it is taken.
    fn attach_first_touching(&mut self) -> Result<Option<String>, String> {
        match self {
            LiveEngine::Builtin(_) => Ok(None),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => {
                let sim = l.backend_mut().sim_mut();
                let Some(prop) = sim.props_touching_robot().first().map(|s| s.to_string()) else {
                    return Ok(None);
                };
                sim.set_weld_active(&prop, true, true)
                    .map_err(|e| e.to_string())?;
                Ok(Some(prop))
            }
        }
    }

    /// Release every grasp weld. A released prop keeps its current state and
    /// falls naturally. Idempotent, and a no-op on builtin.
    fn release_all(&mut self) {
        match self {
            LiveEngine::Builtin(_) => {}
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => l.backend_mut().sim_mut().deactivate_all_welds(),
        }
    }

    /// Reset the sim to `q0` at rest and rebuild the loop AROUND THE SAME
    /// backend: fresh tick/time and a safety monitor re-anchored at `q0` (the
    /// old anchor would rate-limit-fight a distant jump). MuJoCo resets via
    /// `MujocoSim::reset` — the full `mj_resetData` (warmstart cleared), the
    /// crate's determinism anchor.
    fn reset(self, q0: &[f64], model: &Arc<Model>, gains: Gains, h: f64) -> Result<Self, String> {
        match self {
            LiveEngine::Builtin(l) => {
                let mut b = l.into_backend();
                let _ = caliper::hal::RobotBackend::clear_estop(&mut b);
                b.set_state(q0, &vec![0.0; q0.len()])
                    .map_err(|e| e.to_string())?;
                Ok(LiveEngine::Builtin(
                    ControlLoop::new(b, model.clone(), h)
                        .map_err(|e| e.to_string())?
                        .with_gains(gains),
                ))
            }
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => {
                let mut b = l.into_backend();
                let _ = caliper::hal::RobotBackend::clear_estop(&mut b);
                b.sim_mut().reset(q0).map_err(|e| e.to_string())?;
                Ok(LiveEngine::Mujoco(
                    ControlLoop::new(b, model.clone(), h)
                        .map_err(|e| e.to_string())?
                        .with_gains(gains),
                ))
            }
        }
    }
}

// ===== recording (thread-side) =====

/// The open dataset plus the take in progress. Lives in the session thread for
/// its whole life: the writer is never touched from a command, so appends stay
/// on the tick path with no lock contention and no torn episodes.
struct Recorder {
    writer: DatasetWriter,
    /// Root exactly as the command supplied it (what status/`Started` echo).
    root: String,
    fps: u32,
    /// Physics ticks per recorded frame — the exact decimation.
    record_every: u64,
    /// `Some(task)` while a take is recording.
    task: Option<String>,
    /// Joint count the writer's features were sized to.
    ndof: usize,
}

impl Recorder {
    /// Append one control frame as `observation.state` / `action`.
    fn append(&mut self, f: &Frame) -> Result<(), String> {
        if f.measured.len() != self.ndof || f.command.len() != self.ndof {
            return Err(format!(
                "control frame carries {} measured / {} commanded values but the dataset \
                 was opened for {} joints",
                f.measured.len(),
                f.command.len(),
                self.ndof
            ));
        }
        self.writer
            .add_frame(&[
                ("observation.state", f.measured.as_slice()),
                ("action", f.command.as_slice()),
            ])
            .map_err(|e| e.to_string())
    }

    /// Abandon the take in progress; the dataset stays open. Returns the frame
    /// count that was thrown away.
    fn discard_take(&mut self) -> usize {
        let n = self.writer.buffered_frames();
        self.writer.discard_buffered();
        self.task = None;
        n
    }
}

/// Publish the recorder's state for commands (`rec_mirror`) and state events
/// (the atomics). Called on every transition — never per appended frame.
fn publish_rec(shared: &LiveShared, rec: Option<&Recorder>) {
    let (recording, frames) = match rec {
        Some(r) => (r.task.is_some(), r.writer.buffered_frames() as u64),
        None => (false, 0),
    };
    shared.recording.store(recording, Ordering::Relaxed);
    shared.rec_frames.store(frames, Ordering::Relaxed);
    if let Ok(mut m) = shared.rec_mirror.lock() {
        *m = rec.map(|r| RecMirror {
            root: r.root.clone(),
            fps: r.fps,
            task: r.task.clone(),
            episodes_saved: r.writer.total_episodes(),
        });
    }
}

/// Offer one stepped frame to the recorder, appending it when the tick lands on
/// the decimation boundary. A rejected frame would leave a hole in the take, so
/// the take is dropped (the dataset stays open) and the UI sees recording go
/// false — loud in the log, never a panic on the sim thread.
fn record_tick(shared: &LiveShared, rec: &mut Option<Recorder>, frame: &Frame) {
    let Some(r) = rec.as_mut() else { return };
    if r.task.is_none() || !frame.tick.is_multiple_of(r.record_every) {
        return;
    }
    match r.append(frame) {
        Ok(()) => {
            shared
                .rec_frames
                .store(r.writer.buffered_frames() as u64, Ordering::Relaxed);
            return;
        }
        Err(e) => {
            let n = r.discard_take();
            log::error!(
                target: "studio::live",
                "recording aborted after {n} frames — the frame was rejected: {e}"
            );
        }
    }
    publish_rec(shared, rec.as_ref());
}

/// Service at most one pending recording request. Runs in the session thread,
/// including while paused, so a take can be stopped without resuming first.
fn service_rec(shared: &LiveShared, rec: &mut Option<Recorder>, ndof: usize) {
    let Some((id, req)) = shared.rec_req.lock().ok().and_then(|mut r| r.take()) else {
        return;
    };
    let reply = match req {
        RecRequest::Start {
            writer,
            root,
            fps,
            record_every,
            task,
        } => match writer {
            // Opening call: the command already validated the root and built
            // the writer. (A second one while a dataset is open is refused in
            // the command; guarded here too so a writer is never stranded.)
            Some(w) if rec.is_none() => {
                *rec = Some(Recorder {
                    writer: *w,
                    root,
                    fps,
                    record_every,
                    task: Some(task),
                    ndof,
                });
                Ok(RecReply::Started { episode_index: 0 })
            }
            Some(_) => Err("a dataset is already open — finish it first".into()),
            None => match rec.as_mut() {
                None => Err("no dataset open".into()),
                Some(r) if r.task.is_some() => {
                    Err("a take is already recording — stop it first".into())
                }
                Some(r) => {
                    r.task = Some(task);
                    Ok(RecReply::Started {
                        episode_index: r.writer.total_episodes(),
                    })
                }
            },
        },
        RecRequest::Stop { save } => match rec.as_mut() {
            None => Err("no dataset open".into()),
            // The take ends either way — an error below reports what happened
            // to its frames, it does not resume recording.
            Some(r) => match r.task.take() {
                None => Err("not recording".into()),
                Some(task) => {
                    let frames = r.writer.buffered_frames();
                    if !save {
                        r.writer.discard_buffered();
                        Ok(RecReply::Stopped {
                            saved: false,
                            episode_index: None,
                            frames,
                        })
                    } else if frames == 0 {
                        Err(
                            "the take captured no frames — nothing to save (was the sim \
                             paused the whole time?)"
                                .into(),
                        )
                    } else {
                        match r.writer.save_episode(&task) {
                            Ok(()) => Ok(RecReply::Stopped {
                                saved: true,
                                episode_index: Some(r.writer.total_episodes() - 1),
                                frames,
                            }),
                            Err(e) => {
                                r.writer.discard_buffered();
                                Err(format!("saving the episode failed: {e}"))
                            }
                        }
                    }
                }
            },
        },
        RecRequest::Finish => match rec.take() {
            None => Err("no dataset open".into()),
            Some(r) if r.task.is_some() => {
                *rec = Some(r);
                Err("stop the take first, then finish the dataset".into())
            }
            Some(r) => {
                let episodes = r.writer.total_episodes();
                match r.writer.finalize() {
                    Ok(root) => Ok(RecReply::Finished {
                        root: root.display().to_string(),
                        episodes,
                    }),
                    Err(e) => Err(format!("finalizing the dataset failed: {e}")),
                }
            }
        },
    };
    publish_rec(shared, rec.as_ref());
    // Stamped with the id of the request this answers: a command that already
    // timed out is no longer listening, and the next one must not mistake this
    // for its own answer.
    if let Ok(mut slot) = shared.rec_reply.lock() {
        *slot = Some((id, reply));
    }
}

/// Close an open dataset at session end: an interrupted take is never
/// half-saved, but whatever WAS saved must land on disk readable. Best effort —
/// a failure here is logged and must not stop the thread from exiting.
fn close_recorder(shared: &LiveShared, rec: &mut Option<Recorder>) {
    let Some(mut r) = rec.take() else { return };
    if r.task.is_some() {
        let n = r.discard_take();
        log::warn!(
            target: "studio::live",
            "live session ended mid-take — {n} unsaved frames discarded"
        );
    }
    if let Err(e) = r.writer.finalize() {
        log::warn!(target: "studio::live", "finalizing the recording dataset failed: {e}");
    }
    publish_rec(shared, None);
}

// ===== grasp (thread-side) =====

/// Publish what the session thread currently holds, for state events and
/// status. Called only on transitions.
fn publish_held(shared: &LiveShared, held: Option<&str>) {
    if let Ok(mut h) = shared.held.lock() {
        *h = held.map(str::to_string);
    }
}

/// One tick of the weld heuristic. Runs on EVERY tick (not per emit batch) so
/// a grab lands on the exact tick the conditions are met — a grasp that
/// depended on the event decimation would be a different grasp at a different
/// emit rate.
///
/// Costs nothing on the common path: it only looks at contacts when the human
/// is holding the gripper closed, nothing is held yet, and the gripper is
/// engaged. Once a prop is taken it is kept until the intent flips to open —
/// one prop at a time, first match wins.
fn service_grasp(
    shared: &LiveShared,
    engine: &mut LiveEngine,
    g: &GripperChannel,
    tracker: &mut GraspTracker,
    held: &mut Option<String>,
    frame: &Frame,
) -> Result<(), String> {
    let want_closed = shared.grasp_closed.load(Ordering::Relaxed);
    if !want_closed {
        tracker.clear();
        if held.take().is_some() {
            engine.release_all();
            publish_held(shared, None);
        }
        return Ok(());
    }
    if held.is_some() {
        return Ok(());
    }
    let Some(&q) = frame.measured.get(g.index) else {
        return Ok(());
    };
    let stalled = tracker.observe(q, g);
    if !gripper_engaged(q, g, stalled) {
        return Ok(());
    }
    if let Some(prop) = engine.attach_first_touching()? {
        log::info!(
            target: "studio::live",
            "gripper closed on `{prop}` — welded to the attach link (heuristic grasp)"
        );
        *held = Some(prop);
        publish_held(shared, held.as_deref());
    }
    Ok(())
}

/// Physics ticks per recorded frame. The tick rate must be an exact integer
/// multiple of `fps`: anything else would drift the real sample times away from
/// the `frame_index / fps` timestamps the dataset stores.
fn record_every(h: f64, fps: u32) -> Result<u64, String> {
    if fps == 0 {
        return Err("fps must be positive".into());
    }
    let rate = 1.0 / h;
    if !rate.is_finite() || (rate - rate.round()).abs() > 1e-6 || rate.round() < 1.0 {
        return Err(format!(
            "this engine's tick rate ({rate:.4} Hz) is not a whole number — recording needs \
             an integer tick rate"
        ));
    }
    let rate = rate.round() as u64;
    if !rate.is_multiple_of(u64::from(fps)) {
        let ok: Vec<String> = [10u64, 20, 25, 50, 100, 125, 200, 250, 500, 1000]
            .iter()
            .filter(|d| **d <= rate && rate.is_multiple_of(**d))
            .map(|d| d.to_string())
            .collect();
        return Err(format!(
            "fps {fps} must divide the {rate} Hz tick rate exactly (try {})",
            ok.join(", ")
        ));
    }
    Ok(rate / u64::from(fps))
}

// ===== the session loop =====

fn set_status(shared: &LiveShared, paused: bool, t: f64, tick: u64) {
    if let Ok(mut s) = shared.status.lock() {
        *s = LiveStatusInner { paused, t, tick };
    }
}

/// Start a fresh success episode: re-capture every `ref="initial"` baseline
/// from the CURRENT state. Called at session start and after every reset — a
/// lift is always measured from where this episode began, never from where the
/// last one did.
fn reset_success(success: &mut Option<SuccessTracker>, engine: &LiveEngine) -> Result<(), String> {
    let Some(t) = success.as_mut() else {
        return Ok(());
    };
    let st = engine.success_state()?;
    t.reset(Some(&st)).map_err(|e| e.to_string())
}

fn state_event(
    engine: &LiveEngine,
    model: &Model,
    shared: &LiveShared,
    session_id: u64,
    paused: bool,
    gripper: Option<&GripperChannel>,
    success: &mut Option<SuccessTracker>,
) -> Result<LiveStateEvent, String> {
    let (q, qd, ncon, props) = engine.snapshot();
    if !(q.iter().all(|x| x.is_finite()) && qd.iter().all(|x| x.is_finite())) {
        return Err("simulation state went non-finite".into());
    }
    // A driving policy (E1) reads THIS state — the one the UI is about to see —
    // out of a one-slot latest-wins bus. Publishing is a mutex write into
    // pre-grown vectors and a sequence bump; the sim thread never waits on the
    // policy process, and skips this entirely when nothing is attached.
    shared
        .policy_bus
        .publish(engine.tick(), engine.time(), &q, &qd);
    // Judged from the SAME state that is about to be emitted, so a `true` in an
    // event always describes the poses in that event. A predicate that cannot
    // be answered (a prop that vanished) ends the session loudly rather than
    // streaming `null` as if nothing were being scored.
    let verdict = match success.as_mut() {
        None => None,
        Some(t) => {
            let st = engine.success_state()?;
            Some(t.judge(&st).map_err(|e| format!("success: {e}"))?)
        }
    };
    let (frames, tip) = bake_frame_row(model, &q);
    let target = shared
        .target
        .lock()
        .map(|t| t.clone())
        .unwrap_or_else(|_| q.clone());
    let gripper = gripper.and_then(|g| {
        Some(GripperStateEvent {
            closed: shared.grasp_closed.load(Ordering::Relaxed),
            q: *q.get(g.index)?,
        })
    });
    let held = shared.held.lock().ok().and_then(|h| h.clone());
    Ok(LiveStateEvent {
        session_id,
        tick: engine.tick(),
        t: engine.time(),
        q,
        qd,
        frames,
        tip,
        ncon,
        props,
        paused,
        target,
        recording: shared.recording.load(Ordering::Relaxed),
        rec_frames: shared.rec_frames.load(Ordering::Relaxed),
        gripper,
        held,
        success: verdict,
    })
}

/// The session body, generic over the emitter so tests run it headlessly.
/// Owns the engine until the thread exits; every exit path emits `live://ended`
/// and closes an open recording exactly once.
#[allow(clippy::too_many_arguments)]
fn run_session<E: LiveEmitter>(
    engine: LiveEngine,
    model: Arc<Model>,
    gains: Gains,
    h: f64,
    emit_every: u64,
    session_id: u64,
    shared: Arc<LiveShared>,
    gripper: Option<GripperChannel>,
    success: Option<SuccessTracker>,
    emitter: E,
) {
    let mut rec: Option<Recorder> = None;
    let reason = session_loop(
        engine,
        &model,
        gains,
        h,
        emit_every,
        session_id,
        &shared,
        gripper.as_ref(),
        success,
        &emitter,
        &mut rec,
    );
    close_recorder(&shared, &mut rec);
    emitter.ended(&LiveEndedEvent { session_id, reason });
}

/// The loop proper: steps until the session must end and returns the
/// `live://ended` reason (setting `dead` itself on an error exit). Split out of
/// [`run_session`] so stop, supersede and every error path share ONE place that
/// closes the recording.
#[allow(clippy::too_many_arguments)]
fn session_loop<E: LiveEmitter>(
    mut engine: LiveEngine,
    model: &Arc<Model>,
    gains: Gains,
    h: f64,
    emit_every: u64,
    session_id: u64,
    shared: &Arc<LiveShared>,
    gripper: Option<&GripperChannel>,
    mut success: Option<SuccessTracker>,
    emitter: &E,
    rec: &mut Option<Recorder>,
) -> String {
    // Never-stale teleop budget: the live target is a UI hold value, not a
    // deadman link — pause/stop are the explicit controls here, and letting the
    // watchdog kick in mid-catch-up-batch would silently swap targets.
    let mut sp = {
        let q0 = shared
            .target
            .lock()
            .map(|t| t.clone())
            .unwrap_or_else(|_| vec![0.0; model.ndof]);
        TeleopSetpoint::new(q0).with_tick_budget(u64::MAX)
    };
    let ndof = model.ndof;
    let mut pacer = Pacer::new(h, CATCHUP_CAP_S);
    let mut last_wall = Instant::now();
    let mut last_paused = shared.paused.load(Ordering::Relaxed);
    let mut since_emit: u64 = 0;
    let mut grasp = GraspTracker::default();
    let mut held: Option<String> = None;

    macro_rules! die {
        ($detail:expr) => {{
            shared.dead.store(true, Ordering::Relaxed);
            return format!("error: {}", $detail);
        }};
    }
    macro_rules! emit_state_or_die {
        ($paused:expr) => {
            match state_event(
                &engine,
                model,
                shared,
                session_id,
                $paused,
                gripper,
                &mut success,
            ) {
                Ok(ev) => {
                    set_status(shared, $paused, ev.t, ev.tick);
                    emitter.state(&ev);
                }
                Err(e) => die!(e),
            }
        };
    }
    // Re-anchor the success episode at the current state (start / after reset).
    macro_rules! reset_success_or_die {
        () => {
            if let Err(e) = reset_success(&mut success, &engine) {
                die!(e);
            }
        };
    }

    reset_success_or_die!(); // the lift baseline is the pose we START from
    emit_state_or_die!(last_paused); // initial snapshot so the UI paints at once

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return if shared.superseded.load(Ordering::Relaxed) {
                "superseded".to_string()
            } else {
                "stopped".to_string()
            };
        }

        // Reset request (honored even while paused; paused flag survives).
        let req = shared.reset_req.lock().ok().and_then(|mut r| r.take());
        if let Some(qr) = req {
            // A take spanning a reset is not one demonstration: drop it, keep
            // the dataset open for the next one.
            if let Some(r) = rec.as_mut() {
                if r.task.is_some() {
                    let n = r.discard_take();
                    log::warn!(
                        target: "studio::live",
                        "live reset discarded the take in progress ({n} frames); \
                         the dataset stays open"
                    );
                }
            }
            publish_rec(shared, rec.as_ref());
            engine = match engine.reset(&qr, model, gains, h) {
                Ok(e) => e,
                Err(e) => die!(format!("reset failed: {e}")),
            };
            // A reset is a fresh epoch: nothing is held, and the gripper goes
            // back to OPEN intent so the flag agrees with the q0 hold target
            // the reset just installed.
            engine.release_all();
            held = None;
            grasp.clear();
            shared.grasp_closed.store(false, Ordering::Relaxed);
            publish_held(shared, None);
            if let Ok(mut t) = shared.target.lock() {
                *t = qr.clone();
            }
            sp = TeleopSetpoint::new(qr).with_tick_budget(u64::MAX);
            pacer.clear();
            last_wall = Instant::now();
            since_emit = 0;
            // A reset is a fresh EPISODE for the verdict too: the props are back
            // at their spawn poses, so that is where a lift is measured from.
            reset_success_or_die!();
            emit_state_or_die!(last_paused);
            continue;
        }

        // Recording requests are serviced while paused too, so a take can be
        // stopped or saved without resuming the sim.
        service_rec(shared, rec, ndof);

        let paused = shared.paused.load(Ordering::Relaxed);
        if paused != last_paused {
            last_paused = paused;
            // The paused span must not become catch-up debt on resume.
            pacer.clear();
            last_wall = Instant::now();
            emit_state_or_die!(paused);
        }
        if paused {
            std::thread::sleep(Duration::from_millis(5));
            last_wall = Instant::now();
            continue;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(last_wall).as_secs_f64();
        last_wall = now;
        let n = pacer.plan(elapsed);
        if n == 0 {
            std::thread::sleep(Duration::from_micros(300));
            continue;
        }
        if let Ok(t) = shared.target.lock() {
            sp.set(t.clone());
        }
        for _ in 0..n {
            let frame = match engine.step(&mut sp) {
                Ok(f) => f,
                Err(e) => die!(e),
            };
            if let Some(g) = gripper {
                if let Err(e) = service_grasp(shared, &mut engine, g, &mut grasp, &mut held, &frame)
                {
                    die!(format!("grasp failed: {e}"));
                }
            }
            record_tick(shared, rec, &frame);
            since_emit += 1;
            if since_emit >= emit_every {
                since_emit = 0;
                emit_state_or_die!(false);
            }
        }
        set_status(shared, false, engine.time(), engine.tick());
    }
}

// ===== engine construction (inside the command; errors before any thread) =====

/// Engine + its physics timestep. Validation errors return before any spawn.
///
/// `attach_link` (the gripper's own link) turns on the per-prop grasp welds in
/// the generated MJCF; `None` — no gripper channel, or the builtin engine —
/// emits no welds at all, so a session that cannot grasp carries no machinery
/// for it.
#[allow(clippy::too_many_arguments)]
fn build_engine(
    model: &Arc<Model>,
    engine_name: &str,
    q0: &[f64],
    props: &[PropDto],
    ground: f64,
    gains: Gains,
    attach_link: Option<String>,
) -> Result<(LiveEngine, f64), String> {
    let zeros = vec![0.0; q0.len()];
    match engine_name {
        "builtin" => {
            if !props.is_empty() {
                return Err("props need the mujoco engine — builtin has no contact".into());
            }
            let _ = attach_link; // no contacts, so nothing to weld to
            let h = 1e-3;
            let mut backend = PhysicsSimBackend::new(model.clone()).map_err(|e| e.to_string())?;
            backend.set_sim_hmax(h);
            backend.set_state(q0, &zeros).map_err(|e| e.to_string())?;
            let loopy = ControlLoop::new(backend, model.clone(), h)
                .map_err(|e| e.to_string())?
                .with_gains(gains);
            Ok((LiveEngine::Builtin(loopy), h))
        }
        "mujoco" => {
            #[cfg(feature = "mujoco")]
            {
                use caliper_sim_mujoco::mjcf::MjcfOptions;
                use caliper_sim_mujoco::MujocoBackend;
                let specs = props
                    .iter()
                    .map(crate::prop_spec)
                    .collect::<Result<Vec<_>, _>>()?;
                let opt = MjcfOptions {
                    ground_plane: Some(ground),
                    props: specs,
                    attach_link,
                    ..Default::default() // torque-direct, Earth gravity, 1 ms timestep
                };
                let h = opt.timestep;
                let mut backend =
                    MujocoBackend::with_options(model, &opt).map_err(|e| e.to_string())?;
                backend.set_state(q0, &zeros).map_err(|e| e.to_string())?;
                let loopy = ControlLoop::new(backend, model.clone(), h)
                    .map_err(|e| e.to_string())?
                    .with_gains(gains);
                Ok((LiveEngine::Mujoco(loopy), h))
            }
            #[cfg(not(feature = "mujoco"))]
            {
                let _ = (ground, attach_link);
                Err("contact sim not compiled — build studio with --features mujoco".into())
            }
        }
        other => Err(format!("unknown engine `{other}` (mujoco|builtin)")),
    }
}

/// Resolve the session's gripper channel: the explicit `gripperJoint`
/// override, else auto-detection ([`caliper::model::gripper`]).
///
/// An override that names a joint the robot does not have — or one with no
/// limits to open and close between — is an ERROR, because the human asked for
/// that specific channel. Auto-detection finding nothing is NOT an error: most
/// robots have no gripper, and they simply get no channel.
fn resolve_gripper(
    model: &Model,
    joint: Option<&str>,
    closed_end: Option<&str>,
) -> Result<Option<GripperChannel>, String> {
    let closed_at_lo = match closed_end {
        None | Some("lo") => true,
        Some("hi") => false,
        Some(other) => {
            return Err(format!(
                "unknown gripperClosed `{other}` — which limit CLOSES the gripper (lo|hi)"
            ));
        }
    };
    let index = match joint {
        Some(name) => match model.joint_names.iter().position(|j| j == name) {
            Some(i) => i,
            None => {
                return Err(format!(
                    "`{name}` is not a joint of this robot ({})",
                    model.joint_names.join(", ")
                ));
            }
        },
        None => match caliper::model::gripper::find_gripper_joint(model) {
            Some(i) => i,
            None => return Ok(None),
        },
    };
    let Some((open_target, closed_target)) =
        caliper::model::gripper::gripper_targets(model, index, closed_at_lo)
    else {
        return Err(format!(
            "joint `{}` has no usable limits — a gripper channel needs both, to know \
             where open and closed are",
            model.joint_names[index]
        ));
    };
    Ok(Some(GripperChannel {
        index,
        name: model.joint_names[index].clone(),
        open_target,
        closed_target,
    }))
}

/// Parse + validate the session's success predicate against the scene it will
/// judge, and decide whether this engine can judge it at all.
///
/// Everything a bad predicate can be is caught here: an unknown key or kind, a
/// zone still referenced by NAME (`task_open` resolves those against the task
/// file's `scene.zones`; a predicate arriving with one has no scene to resolve
/// against), and a prop the session's scene does not contain — which would
/// otherwise surface as a dead session on the first emitted state.
///
/// The BUILTIN engine has no props, so nothing a predicate could be about
/// exists: the predicate is still validated for shape, but no tracker is
/// installed and `live://state.success` stays `null` for the whole session. A
/// silent `false` there would read as "not succeeded yet" instead of "nobody is
/// scoring this".
fn resolve_success(
    spec: Option<&serde_json::Value>,
    engine_name: &str,
    props: &[PropDto],
) -> Result<Option<SuccessTracker>, String> {
    let Some(spec) = spec else { return Ok(None) };
    let pred = Predicate::from_value(spec).map_err(|e| format!("success: {e}"))?;
    if let Some(zone) = pred.unresolved_zone() {
        return Err(format!(
            "success refers to zone `{zone}` by name — a live session has no task file to \
             resolve it against; open the task (which resolves its own zones) or spell the \
             zone out as {{center, half}}"
        ));
    }
    if engine_name != "mujoco" {
        log::warn!(
            target: "studio::live",
            "a success predicate ({}) was supplied for the `{engine_name}` engine, which has \
             no props to judge — the session will report success = null",
            pred.name()
        );
        return Ok(None);
    }
    let known: Vec<&str> = props.iter().map(|p| p.name.as_str()).collect();
    for prop in pred.prop_names() {
        if !known.contains(&prop) {
            return Err(format!(
                "success scores prop `{prop}`, which this scene does not contain \
                 (props: {known:?})"
            ));
        }
    }
    Ok(Some(SuccessTracker::new(pred)))
}

// ===== command impls (take &AppState so tests drive them without tauri) =====

/// Raise `stop` on an existing session (reason per `superseded`) and join it.
fn stop_in_slot(slot: &mut Option<LiveSession>, superseded: bool) {
    if let Some(mut s) = slot.take() {
        // A policy dies with its session — dropping the handle stops the child
        // and joins its threads, so no python ever outlives the sim it drives.
        if let Ok(mut p) = s.policy.lock() {
            drop(p.take());
        }
        s.shared.superseded.store(superseded, Ordering::Relaxed);
        s.shared.stop.store(true, Ordering::Relaxed);
        if let Some(j) = s.join.take() {
            let _ = j.join();
        }
    }
}

pub(crate) fn live_start_on<E: LiveEmitter>(
    state: &AppState,
    req: LiveStartReq,
    emitter: E,
) -> Result<LiveStartedDto, String> {
    let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
    let model = guard.as_ref().ok_or("no robot loaded")?;
    let arc = Arc::new(model.clone());
    drop(guard);

    if !arc.has_inertia {
        return Err(
            "this robot has no inertial data — load one with <inertial> (showcase6 or dyn_pendulum2)"
                .into(),
        );
    }
    let n = arc.ndof;
    if req.q0.len() != n || !req.q0.iter().all(|x| x.is_finite()) {
        return Err(format!("q0 needs {n} finite values"));
    }
    let engine_name: &'static str = match req.engine.as_deref() {
        Some("mujoco") => "mujoco",
        Some("builtin") => "builtin",
        Some(other) => return Err(format!("unknown engine `{other}` (mujoco|builtin)")),
        None if cfg!(feature = "mujoco") => "mujoco",
        None => "builtin",
    };
    let ground = req.ground.unwrap_or(0.0);
    if !ground.is_finite() {
        return Err("ground must be finite".into());
    }
    let kp = req.kp.unwrap_or(100.0);
    let kd = req.kd.unwrap_or(20.0);
    if !kp.is_finite() || !kd.is_finite() || kp < 0.0 || kd < 0.0 {
        return Err("kp/kd must be finite and non-negative".into());
    }
    let requested_hz = req.emit_hz.unwrap_or(60.0);
    if !requested_hz.is_finite() {
        return Err("emitHz must be finite".into());
    }
    let gains = Gains { kp, kd };

    // The channel is resolved BEFORE the engine is built: a bad override must
    // fail with nothing constructed, and the attach link comes from it.
    let gripper = resolve_gripper(
        &arc,
        req.gripper_joint.as_deref(),
        req.gripper_closed.as_deref(),
    )?;
    // Props weld to the gripper's OWN link; the tip link is the fallback for a
    // channel whose child link somehow cannot be named.
    let attach_link = gripper.as_ref().map(|g| {
        caliper::model::gripper::child_link_name(&arc, g.index)
            .unwrap_or_else(|| arc.frame_name(arc.tip_frame()))
            .to_string()
    });

    // The verdict is resolved BEFORE the engine, for the same reason as the
    // gripper channel: a predicate that cannot be judged in this scene must
    // fail with nothing constructed.
    let success = resolve_success(req.success.as_ref(), engine_name, &req.props)?;

    let (engine, h) = build_engine(
        &arc,
        engine_name,
        &req.q0,
        &req.props,
        ground,
        gains,
        attach_link,
    )?;
    let every = emit_every(clamp_emit_hz(req.emit_hz), h);
    let actual_hz = 1.0 / (every as f64 * h);

    let mut slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    stop_in_slot(&mut slot, true); // an existing session ends as "superseded"

    let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
    let shared = Arc::new(LiveShared::new(req.q0.clone()));
    let thread_shared = shared.clone();
    let thread_model = arc.clone();
    let thread_gripper = gripper.clone();
    let join = std::thread::Builder::new()
        .name(format!("live-sim-{session_id}"))
        .spawn(move || {
            run_session(
                engine,
                thread_model,
                gains,
                h,
                every,
                session_id,
                thread_shared,
                thread_gripper,
                success,
                emitter,
            )
        })
        .map_err(|e| format!("failed to spawn the live sim thread: {e}"))?;

    *slot = Some(LiveSession {
        id: session_id,
        engine: engine_name,
        ndof: n,
        h,
        emit_hz: actual_hz,
        q0: req.q0,
        gripper: gripper.clone(),
        shared,
        policy: Mutex::new(None),
        join: Some(join),
    });

    let props = req
        .props
        .iter()
        .map(|p| PropTrackDto {
            name: p.name.clone(),
            kind: p.kind.clone(),
            half_extents: p.half_extents,
            radius: p.radius,
            length: p.length,
            rgba: p.rgba,
            frames: Vec::new(),
        })
        .collect();
    Ok(LiveStartedDto {
        session_id,
        engine: engine_name.to_string(),
        h,
        emit_hz: actual_hz,
        ndof: n,
        props,
        gripper: gripper.map(|g| GripperDto {
            joint: g.name,
            index: g.index,
            open_target: g.open_target,
            closed_target: g.closed_target,
        }),
    })
}

/// Run `f` on the live session, reaping the slot first if its thread died.
pub(crate) fn with_live<T>(
    state: &AppState,
    f: impl FnOnce(&LiveSession) -> Result<T, String>,
) -> Result<T, String> {
    let mut slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    match slot.as_ref() {
        Some(s) if s.shared.dead.load(Ordering::Relaxed) => {
            // The thread already emitted its `error:` ended event; just reap.
            if let Some(mut dead) = slot.take() {
                if let Some(j) = dead.join.take() {
                    let _ = j.join();
                }
            }
            Err("no live session".into())
        }
        Some(s) => f(s),
        None => Err("no live session".into()),
    }
}

pub(crate) fn live_set_target_impl(state: &AppState, q: &[f64]) -> Result<(), String> {
    with_live(state, |s| {
        if q.len() != s.ndof || !q.iter().all(|x| x.is_finite()) {
            return Err(format!("target needs {} finite values", s.ndof));
        }
        let mut t = s.shared.target.lock().map_err(|_| "state lock poisoned")?;
        *t = q.to_vec();
        Ok(())
    })
}

/// Command the gripper open or closed.
///
/// This is a TARGET MOVE, not a second control path: it writes the gripper
/// joint's slot in the same PD hold target the sliders, the IK gizmo and the
/// keyboard jog write, so there is exactly one thing telling the arm what to
/// do. The grasp-intent flag it also raises is what the weld heuristic (and
/// the `gripper.closed` field of `live://state`) reads.
pub(crate) fn live_gripper_impl(state: &AppState, closed: bool) -> Result<(), String> {
    with_live(state, |s| {
        let g = s.gripper.as_ref().ok_or(
            "this robot has no gripper channel — no joint named like a gripper \
             (gripper/finger/jaw/claw/hand) with both limits",
        )?;
        let mut t = s.shared.target.lock().map_err(|_| "state lock poisoned")?;
        let slot = t
            .get_mut(g.index)
            .ok_or("the gripper joint is out of range for this session's target")?;
        *slot = if closed {
            g.closed_target
        } else {
            g.open_target
        };
        drop(t);
        // The target move and the intent flag are two separate writes, and the
        // session thread can land between them either way round — neither
        // order can produce a wrong grasp. Intent-before-target: the gripper is
        // still measured open, so the engagement test declines and the grab
        // happens a tick later. Target-before-intent: the grasp stays open,
        // which is what "not yet commanded closed" means.
        s.shared.grasp_closed.store(closed, Ordering::Relaxed);
        Ok(())
    })
}

pub(crate) fn live_pause_impl(state: &AppState, paused: bool) -> Result<(), String> {
    with_live(state, |s| {
        s.shared.paused.store(paused, Ordering::Relaxed);
        Ok(())
    })
}

pub(crate) fn live_reset_impl(state: &AppState, q0: Option<Vec<f64>>) -> Result<(), String> {
    with_live(state, |s| {
        let q = match q0 {
            Some(q) => {
                if q.len() != s.ndof || !q.iter().all(|x| x.is_finite()) {
                    return Err(format!("q0 needs {} finite values", s.ndof));
                }
                q
            }
            None => s.q0.clone(),
        };
        let mut r = s
            .shared
            .reset_req
            .lock()
            .map_err(|_| "state lock poisoned")?;
        *r = Some(q);
        Ok(())
    })
}

pub(crate) fn live_stop_impl(state: &AppState) -> Result<(), String> {
    let mut slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    stop_in_slot(&mut slot, false); // idempotent: no session is a no-op
    Ok(())
}

pub(crate) fn live_status_impl(state: &AppState) -> Result<Option<LiveStatusDto>, String> {
    let mut slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    match slot.as_ref() {
        Some(s) if s.shared.dead.load(Ordering::Relaxed) => {
            if let Some(mut dead) = slot.take() {
                if let Some(j) = dead.join.take() {
                    let _ = j.join();
                }
            }
            Ok(None)
        }
        Some(s) => {
            let st = s.shared.status.lock().map(|g| *g).unwrap_or_default();
            Ok(Some(LiveStatusDto {
                session_id: s.id,
                engine: s.engine.to_string(),
                paused: st.paused,
                t: st.t,
                tick: st.tick,
                ndof: s.ndof,
                recording: s.shared.recording.load(Ordering::Relaxed),
            }))
        }
        None => Ok(None),
    }
}

// ===== recording command impls =====

/// What a failed [`rec_request_raw`] hands back: the error to report, plus the
/// request itself when it was still queued and could be taken back. Getting it
/// back is proof the session thread never saw it, so whatever it owns (a boxed
/// `DatasetWriter`) is the caller's to clean up.
struct RecFailed {
    err: String,
    unclaimed: Option<RecRequest>,
}

/// Take a posted request back, but only while it is still ours and still
/// queued. A request the thread already took is gone — the thread owns its
/// reply and, on a shutdown, its writer.
fn reclaim_rec(shared: &LiveShared, id: u64) -> Option<RecRequest> {
    let mut slot = shared.rec_req.lock().ok()?;
    match slot.take() {
        Some((qid, req)) if qid == id => Some(req),
        other => {
            *slot = other;
            None
        }
    }
}

/// Post one recording request to the session thread and wait for its reply.
///
/// The caller holds `rec_gate` (so the slots are ours alone) and NO `AppState`
/// lock: the session thread never takes `AppState` locks, but it can be busy
/// for a whole catch-up batch, and blocking on `state.live` while waiting would
/// stall every other command. A dead/stopping session or a wedged thread ends
/// the wait with an error instead of hanging the webview.
///
/// Giving up on the wait does not cancel the request: a thread stuck in a long
/// service (finalizing a big dataset) still answers afterwards. Every request
/// therefore carries an id and only the reply stamped with OUR id is ours — a
/// stale one is thrown away and the wait continues. Without that, a retried
/// command of the same shape (a second `Finish`, a `Stop{save:false}` after a
/// timed-out `Stop{save:true}`) would consume the abandoned request's answer
/// and report an outcome that never happened to it.
fn rec_request_raw(shared: &LiveShared, req: RecRequest) -> Result<RecReply, RecFailed> {
    let fail = |shared: &LiveShared, id: Option<u64>, err: &str| RecFailed {
        err: err.to_string(),
        unclaimed: id.and_then(|id| reclaim_rec(shared, id)),
    };
    let id = shared.rec_seq.fetch_add(1, Ordering::Relaxed) + 1;
    {
        let Ok(mut slot) = shared.rec_reply.lock() else {
            return Err(fail(shared, None, "state lock poisoned"));
        };
        *slot = None;
    }
    {
        let Ok(mut slot) = shared.rec_req.lock() else {
            return Err(fail(shared, None, "state lock poisoned"));
        };
        *slot = Some((id, req));
    }
    let deadline = Instant::now() + REC_REPLY_TIMEOUT;
    loop {
        if let Ok(mut slot) = shared.rec_reply.lock() {
            match slot.take() {
                Some((rid, reply)) if rid == id => {
                    return reply.map_err(|err| RecFailed {
                        err,
                        unclaimed: None,
                    })
                }
                // A late answer to a request we already gave up on: dropped
                // here (the slot is cleared) so it reaches nobody.
                Some(_) | None => {}
            }
        }
        // Checked AFTER the reply slot: a thread that answered and then exited
        // still hands us its answer.
        if shared.dead.load(Ordering::Relaxed) || shared.stop.load(Ordering::Relaxed) {
            return Err(fail(shared, Some(id), "the live session ended"));
        }
        if Instant::now() >= deadline {
            return Err(fail(
                shared,
                Some(id),
                "the live sim thread did not answer the recording request in 2 s",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// [`rec_request_raw`] for the requests that own nothing — a failure has
/// nothing to clean up, so only the message matters.
fn rec_request(shared: &LiveShared, req: RecRequest) -> Result<RecReply, String> {
    rec_request_raw(shared, req).map_err(|f| f.err)
}

/// Post a record-start request, cleaning up after a start whose dataset THIS
/// call created but the session thread never accepted.
///
/// The window is real: the loop checks `stop` before it services requests, so a
/// `live_stop` (or a superseding `live_start`) landing between the post and the
/// service makes the thread exit without ever taking the request. Getting the
/// request back proves that happened, and it still holds the boxed writer.
/// Dropping the writer auto-finalizes it, which would strand a
/// legitimate-looking — but empty — dataset at the root the user picked; the
/// writer has no consume-without-finalize escape, so it is finalized and the
/// directory removed.
///
/// Removing a directory is safe here only because all three hold: this very
/// call created it milliseconds ago (a start onto an already-open dataset
/// carries no writer), the reclaimed request proves the session thread never
/// touched it, and the path removed is the one the writer itself reports.
/// `rec_gate` is held throughout, so no other recording command can have
/// written there in between.
fn rec_start_request(shared: &LiveShared, req: RecRequest) -> Result<RecReply, String> {
    let ours = matches!(
        &req,
        RecRequest::Start {
            writer: Some(_),
            ..
        }
    );
    rec_request_raw(shared, req).map_err(|f| {
        if ours {
            if let Some(RecRequest::Start {
                writer: Some(w),
                root,
                ..
            }) = f.unclaimed
            {
                let dir = match w.finalize() {
                    Ok(p) => p,
                    Err(e) => {
                        log::warn!(
                            target: "studio::live",
                            "closing the unstarted dataset at {root} failed: {e}"
                        );
                        std::path::PathBuf::from(&root)
                    }
                };
                if let Err(e) = std::fs::remove_dir_all(&dir) {
                    log::warn!(
                        target: "studio::live",
                        "removing the unstarted empty dataset at {} failed: {e}",
                        dir.display()
                    );
                }
            }
            // Not reclaimed: the thread took the request as it was shutting
            // down and now owns the writer — `close_recorder` finalizes it.
        }
        f.err
    })
}

/// Start recording a take, creating the dataset on the first call.
///
/// Everything that can fail cheaply fails HERE, synchronously, before the
/// session thread is involved: the task label, the fps/tick-rate divisibility,
/// the robot match, and the dataset root itself (the writer is created in this
/// command, so a bad root never leaves a half-open session behind).
pub(crate) fn live_record_start_impl(
    state: &AppState,
    req: LiveRecordStartReq,
) -> Result<LiveRecordStartedDto, String> {
    let task = req.task.trim().to_string();
    if task.is_empty() {
        return Err("give the episode a task label (what the demonstration does)".into());
    }
    let fps = req.fps.unwrap_or(DEFAULT_RECORD_FPS);

    // Session facts first, then the AppState locks are dropped: nothing below
    // may hold them while waiting on the session thread.
    let (shared, ndof, h) = with_live(state, |s| Ok((s.shared.clone(), s.ndof, s.h)))?;
    let every = record_every(h, fps)?;
    let (robot_type, joint_names) = {
        let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
        let m = guard.as_ref().ok_or("no robot loaded")?;
        if m.ndof != ndof {
            return Err(
                "the loaded robot changed — restart the live session before recording".into(),
            );
        }
        (m.name.clone(), m.joint_names.clone())
    };

    let _gate = shared.rec_gate.lock().map_err(|_| "state lock poisoned")?;
    let open = shared
        .rec_mirror
        .lock()
        .map_err(|_| "state lock poisoned")?
        .clone();
    let writer = match open {
        // Roots are compared as the caller wrote them — a second dataset must
        // never be created under a running one.
        Some(m) if m.root != req.root || m.fps != fps => {
            return Err(format!(
                "a dataset is already open at {} ({} fps) — finish the open dataset first",
                m.root, m.fps
            ));
        }
        Some(m) if m.task.is_some() => {
            return Err("a take is already recording — stop it first".into());
        }
        Some(_) => None,
        None => {
            let names = (joint_names.len() == ndof).then_some(joint_names);
            let spec = DatasetSpec::new(
                fps,
                robot_type,
                vec![
                    FeatureSpec::vector("observation.state", ndof, names.clone()),
                    FeatureSpec::vector("action", ndof, names),
                ],
            );
            Some(Box::new(
                DatasetWriter::create(&req.root, spec).map_err(|e| e.to_string())?,
            ))
        }
    };

    let reply = rec_start_request(
        &shared,
        RecRequest::Start {
            writer,
            root: req.root.clone(),
            fps,
            record_every: every,
            task,
        },
    )?;
    match reply {
        RecReply::Started { episode_index } => Ok(LiveRecordStartedDto {
            root: req.root,
            fps,
            record_every: every,
            episode_index,
        }),
        _ => Err("unexpected reply to a record start".into()),
    }
}

/// End the take: `save` writes it as an episode, otherwise its frames are
/// thrown away. Either way recording stops — an `Err` reports what became of
/// the frames, it does not leave the take running.
pub(crate) fn live_record_stop_impl(
    state: &AppState,
    save: bool,
) -> Result<LiveRecordStoppedDto, String> {
    let shared = with_live(state, |s| Ok(s.shared.clone()))?;
    let _gate = shared.rec_gate.lock().map_err(|_| "state lock poisoned")?;
    match rec_request(&shared, RecRequest::Stop { save })? {
        RecReply::Stopped {
            saved,
            episode_index,
            frames,
        } => Ok(LiveRecordStoppedDto {
            saved,
            episode_index,
            frames,
        }),
        _ => Err("unexpected reply to a record stop".into()),
    }
}

/// Close the dataset (writing `meta/`). Refused mid-take.
pub(crate) fn live_record_finish_impl(state: &AppState) -> Result<LiveRecordFinishedDto, String> {
    let shared = with_live(state, |s| Ok(s.shared.clone()))?;
    let _gate = shared.rec_gate.lock().map_err(|_| "state lock poisoned")?;
    match rec_request(&shared, RecRequest::Finish)? {
        RecReply::Finished { root, episodes } => Ok(LiveRecordFinishedDto { root, episodes }),
        _ => Err("unexpected reply to a record finish".into()),
    }
}

/// Recording state, or null when no dataset is open (including with no live
/// session). Reads the mirror — it never disturbs the session thread.
pub(crate) fn live_record_status_impl(
    state: &AppState,
) -> Result<Option<LiveRecordStatusDto>, String> {
    let shared = match with_live(state, |s| Ok(s.shared.clone())) {
        Ok(s) => s,
        Err(e) if e == "no live session" => return Ok(None),
        Err(e) => return Err(e),
    };
    let mirror = shared
        .rec_mirror
        .lock()
        .map_err(|_| "state lock poisoned")?
        .clone();
    Ok(mirror.map(|m| LiveRecordStatusDto {
        root: m.root,
        fps: m.fps,
        recording: m.task.is_some(),
        task: m.task,
        buffered_frames: shared.rec_frames.load(Ordering::Relaxed) as usize,
        episodes_saved: m.episodes_saved,
    }))
}

// ===== test seams for the sibling policy bridge (E1) =====

/// Start a plain builtin session, for tests in other modules that need a live
/// session but not the whole `LiveStartReq` surface.
#[cfg(test)]
pub(crate) fn test_start_builtin<E: LiveEmitter>(
    state: &AppState,
    q0: Vec<f64>,
    kp: f64,
    kd: f64,
    emitter: E,
) -> Result<LiveStartedDto, String> {
    live_start_on(
        state,
        LiveStartReq {
            q0,
            engine: Some("builtin".into()),
            props: vec![],
            ground: None,
            kp: Some(kp),
            kd: Some(kd),
            emit_hz: None,
            gripper_joint: None,
            gripper_closed: None,
            success: None,
        },
        emitter,
    )
}

#[cfg(test)]
impl LiveStateEvent {
    /// The PD hold target this state was emitted with.
    pub(crate) fn target(&self) -> &[f64] {
        &self.target
    }
}

#[cfg(test)]
impl LiveStatusDto {
    pub(crate) fn tick(&self) -> u64 {
        self.tick
    }
}

// ===== tauri commands =====

/// Start a live sim session (superseding any existing one).
#[tauri::command]
pub fn live_start(
    req: LiveStartReq,
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<LiveStartedDto, String> {
    logged("live_start", live_start_on(&state, req, TauriEmitter(app)))
}

/// Move the live PD hold target (high-frequency — intentionally not logged).
#[tauri::command]
pub fn live_set_target(q: Vec<f64>, state: tauri::State<'_, AppState>) -> Result<(), String> {
    live_set_target_impl(&state, &q)
}

/// Open/close the gripper: moves the gripper joint's hold target and raises
/// the grasp intent the weld heuristic reads.
#[tauri::command]
pub fn live_gripper(closed: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    logged("live_gripper", live_gripper_impl(&state, closed))
}

/// Freeze/unfreeze stepping (the arm holds mid-physics; never de-energized).
#[tauri::command]
pub fn live_pause(paused: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    logged("live_pause", live_pause_impl(&state, paused))
}

/// Reset the sim to `q0` (or the initial pose) at rest; session survives.
#[tauri::command]
pub fn live_reset(q0: Option<Vec<f64>>, state: tauri::State<'_, AppState>) -> Result<(), String> {
    logged("live_reset", live_reset_impl(&state, q0))
}

/// End the live session (idempotent).
#[tauri::command]
pub fn live_stop(state: tauri::State<'_, AppState>) -> Result<(), String> {
    logged("live_stop", live_stop_impl(&state))
}

/// Current session status, or null when no live session exists (including
/// after an error-ended thread).
#[tauri::command]
pub fn live_status(state: tauri::State<'_, AppState>) -> Result<Option<LiveStatusDto>, String> {
    live_status_impl(&state)
}

/// Start recording a take into a LeRobotDataset v3.0 at `root` (created on the
/// first call), tagged with this episode's task label.
#[tauri::command]
pub fn live_record_start(
    req: LiveRecordStartReq,
    state: tauri::State<'_, AppState>,
) -> Result<LiveRecordStartedDto, String> {
    logged("live_record_start", live_record_start_impl(&state, req))
}

/// Stop the take, saving it as an episode or discarding its frames.
#[tauri::command]
pub fn live_record_stop(
    save: bool,
    state: tauri::State<'_, AppState>,
) -> Result<LiveRecordStoppedDto, String> {
    logged("live_record_stop", live_record_stop_impl(&state, save))
}

/// Close the dataset — refused while a take is running.
#[tauri::command]
pub fn live_record_finish(
    state: tauri::State<'_, AppState>,
) -> Result<LiveRecordFinishedDto, String> {
    logged("live_record_finish", live_record_finish_impl(&state))
}

/// Recording state, or null when no dataset is open.
#[tauri::command]
pub fn live_record_status(
    state: tauri::State<'_, AppState>,
) -> Result<Option<LiveRecordStatusDto>, String> {
    logged("live_record_status", live_record_status_impl(&state))
}

// ===== tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../oracle/fixtures/robots"
        ))
        .join(name)
    }

    fn pendulum_state() -> AppState {
        let state = AppState::default();
        let m = Model::from_urdf(&fixture("dyn_pendulum2.urdf")).expect("fixture loads");
        *state.model.lock().unwrap() = Some(m);
        state
    }

    /// A 3-dof arm whose last joint is literally named `gripper`, hanging over
    /// the origin so it already touches a 5 cm cube resting on the ground.
    fn gripper_state() -> AppState {
        let state = AppState::default();
        let m = Model::from_urdf(&fixture("gripper_arm.urdf")).expect("fixture loads");
        *state.model.lock().unwrap() = Some(m);
        state
    }

    #[derive(Clone, Default)]
    struct Collect {
        states: Arc<Mutex<Vec<LiveStateEvent>>>,
        ended: Arc<Mutex<Vec<LiveEndedEvent>>>,
    }
    impl LiveEmitter for Collect {
        fn state(&self, ev: &LiveStateEvent) {
            self.states.lock().unwrap().push(ev.clone());
        }
        fn ended(&self, ev: &LiveEndedEvent) {
            self.ended.lock().unwrap().push(ev.clone());
        }
    }
    impl Collect {
        fn states(&self) -> Vec<LiveStateEvent> {
            self.states.lock().unwrap().clone()
        }
        fn ended(&self) -> Vec<LiveEndedEvent> {
            self.ended.lock().unwrap().clone()
        }
        fn last_state(&self) -> LiveStateEvent {
            self.states().last().expect("has states").clone()
        }
    }

    fn start_builtin(state: &AppState, q0: Vec<f64>) -> (Collect, LiveStartedDto) {
        let em = Collect::default();
        let dto = live_start_on(
            state,
            LiveStartReq {
                q0,
                engine: Some("builtin".into()),
                props: vec![],
                ground: None,
                kp: None,
                kd: None,
                emit_hz: None,
                gripper_joint: None,
                gripper_closed: None,
                success: None,
            },
            em.clone(),
        )
        .expect("live_start");
        (em, dto)
    }

    fn sleep_ms(ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
    }

    // -- pacing math (pure, no clock) --

    #[test]
    fn emit_every_decimation() {
        // 1/(60·1e-3) = 16.67 → round → 17 (the exact rule: round, floor at 1)
        assert_eq!(emit_every(60.0, 1e-3), 17);
        assert_eq!(emit_every(10.0, 1e-3), 100);
        assert_eq!(emit_every(120.0, 1e-3), 8); // 8.33 rounds DOWN
                                                // faster than one emission per step floors at 1
        assert_eq!(emit_every(120.0, 0.05), 1);
    }

    #[test]
    fn emit_hz_clamps() {
        assert_eq!(clamp_emit_hz(None), 60.0);
        assert_eq!(clamp_emit_hz(Some(5.0)), 10.0);
        assert_eq!(clamp_emit_hz(Some(500.0)), 120.0);
        assert_eq!(clamp_emit_hz(Some(10.0)), 10.0);
        assert_eq!(clamp_emit_hz(Some(120.0)), 120.0);
    }

    #[test]
    fn pacer_accumulates_and_carries_remainder() {
        // h = 1/1024 (exact in binary) so the arithmetic is fp-exact.
        let h = 1.0 / 1024.0;
        let mut p = Pacer::new(h, 0.25);
        assert_eq!(p.plan(0.0), 0);
        assert_eq!(p.plan(10.5 * h), 10); // remainder h/2 carried
        assert_eq!(p.plan(0.5 * h), 1); // carried remainder completes a step
        assert_eq!(p.plan(3.0 * h), 3);
    }

    #[test]
    fn pacer_caps_catchup_debt() {
        let h = 1.0 / 1024.0;
        let mut p = Pacer::new(h, 0.25);
        // A 10 s stall yields only cap/h = 256 steps — excess time is dropped.
        assert_eq!(p.plan(10.0), 256);
        assert_eq!(p.plan(0.0), 0); // and the debt is fully consumed
    }

    // -- builtin session end-to-end, headless --

    #[test]
    fn builtin_session_streams_and_stops() {
        let state = pendulum_state();
        let q0 = vec![0.3, -0.2];
        let (em, dto) = start_builtin(&state, q0.clone());
        assert_eq!(dto.ndof, 2);
        assert_eq!(dto.engine, "builtin");
        assert!((dto.h - 1e-3).abs() < 1e-12);
        assert!((dto.emit_hz - 1.0 / (17.0 * 1e-3)).abs() < 1e-9);
        assert!(dto.props.is_empty());

        sleep_ms(150);
        let states = em.states();
        assert!(states.len() >= 3, "only {} state events", states.len());
        for w in states.windows(2) {
            assert!(w[1].tick > w[0].tick, "tick must increase");
            assert!(w[1].t > w[0].t, "t must increase");
        }
        for s in &states {
            assert_eq!(s.session_id, dto.session_id);
            assert_eq!(s.q.len(), 2);
            assert!(s.q.iter().all(|x| x.is_finite()));
            assert!(s.qd.iter().all(|x| x.is_finite()));
            assert_eq!(s.ncon, 0);
            assert!(s.props.is_empty());
            assert!(!s.frames.is_empty());
        }

        // status reflects the running session
        let st = live_status_impl(&state).unwrap().expect("has status");
        assert_eq!(st.session_id, dto.session_id);
        assert!(!st.paused);
        assert!(st.tick > 0);

        // set a target mid-run: subsequent events carry it
        let tgt = vec![0.1, 0.05];
        live_set_target_impl(&state, &tgt).unwrap();
        sleep_ms(100);
        assert_eq!(em.last_state().target, tgt);

        // stop: ended("stopped"), status null
        live_stop_impl(&state).unwrap();
        let ended = em.ended();
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].reason, "stopped");
        assert_eq!(ended[0].session_id, dto.session_id);
        assert!(live_status_impl(&state).unwrap().is_none());
        // idempotent
        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn builtin_pause_freezes_and_reset_zeroes_clock() {
        let state = pendulum_state();
        let (em, _dto) = start_builtin(&state, vec![0.2, -0.1]);
        sleep_ms(80);

        // pause: one transition event, then ticks stop advancing
        live_pause_impl(&state, true).unwrap();
        sleep_ms(50);
        let a = em.last_state();
        assert!(a.paused);
        sleep_ms(100);
        let b = em.last_state();
        assert_eq!(a.tick, b.tick, "tick advanced while paused");

        // reset(None) emits one state EVEN WHILE PAUSED, with tick/t back to 0
        // (chosen semantics: reset rebuilds the loop → tick and t restart at 0).
        live_reset_impl(&state, None).unwrap();
        sleep_ms(50);
        let r = em.last_state();
        assert_eq!(r.tick, 0);
        assert!(r.t.abs() < 1e-12);
        assert!(r.paused, "reset must keep the paused flag");
        assert_eq!(r.target, vec![0.2, -0.1], "reset restores target to q0");

        // resume: stepping continues from the reset clock
        live_pause_impl(&state, false).unwrap();
        sleep_ms(80);
        let s = em.last_state();
        assert!(!s.paused);
        assert!(s.tick > 0);

        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn builtin_reset_to_distant_pose_holds_there() {
        let state = pendulum_state();
        // Stiffer gains than default so the hold settles within the test window.
        let em = Collect::default();
        let dto = live_start_on(
            &state,
            LiveStartReq {
                q0: vec![0.0, 0.0],
                engine: Some("builtin".into()),
                props: vec![],
                ground: None,
                kp: Some(400.0),
                kd: Some(40.0),
                emit_hz: None,
                gripper_joint: None,
                gripper_closed: None,
                success: None,
            },
            em.clone(),
        )
        .unwrap();
        assert_eq!(dto.ndof, 2);
        sleep_ms(60);

        // Jump far from the old monitor anchor: the rebuilt loop must hold the
        // NEW pose, not rate-limit-fight its way back or snap.
        let far = vec![0.8, -0.5];
        live_reset_impl(&state, Some(far.clone())).unwrap();
        sleep_ms(700); // settle under PD + gravity FF
        let s = em.last_state();
        for (i, (&qi, &fi)) in s.q.iter().zip(&far).enumerate() {
            assert!((qi - fi).abs() < 0.05, "joint {i} at {qi} not holding {fi}");
        }
        live_stop_impl(&state).unwrap();
    }

    // -- error paths --

    #[test]
    fn wrong_target_len_is_refused() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.0, 0.0]);
        let err = live_set_target_impl(&state, &[0.1]).unwrap_err();
        assert!(err.contains("2 finite values"), "got: {err}");
        let err = live_set_target_impl(&state, &[f64::NAN, 0.0]).unwrap_err();
        assert!(err.contains("finite"), "got: {err}");
        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn builtin_with_props_is_refused() {
        let state = pendulum_state();
        let em = Collect::default();
        let err = live_start_on(
            &state,
            LiveStartReq {
                q0: vec![0.0, 0.0],
                engine: Some("builtin".into()),
                props: vec![PropDto {
                    name: "box".into(),
                    kind: "box".into(),
                    half_extents: Some([0.05; 3]),
                    radius: None,
                    length: None,
                    pos: [0.3, 0.0, 0.5],
                    quat: None,
                    mass: None,
                    rgba: None,
                    material: None,
                }],
                ground: None,
                kp: None,
                kd: None,
                emit_hz: None,
                gripper_joint: None,
                gripper_closed: None,
                success: None,
            },
            em,
        )
        .map(|_| ())
        .unwrap_err();
        assert!(err.contains("mujoco"), "got: {err}");
        assert!(live_status_impl(&state).unwrap().is_none());
    }

    #[test]
    fn start_twice_supersedes_first() {
        let state = pendulum_state();
        let (em1, dto1) = start_builtin(&state, vec![0.1, 0.1]);
        sleep_ms(40);
        let (em2, dto2) = start_builtin(&state, vec![0.2, 0.2]);
        assert!(dto2.session_id > dto1.session_id, "ids are monotonic");
        let ended1 = em1.ended();
        assert_eq!(ended1.len(), 1);
        assert_eq!(ended1[0].reason, "superseded");
        assert_eq!(ended1[0].session_id, dto1.session_id);
        assert!(em2.ended().is_empty());
        let st = live_status_impl(&state).unwrap().expect("second alive");
        assert_eq!(st.session_id, dto2.session_id);
        live_stop_impl(&state).unwrap();
        assert_eq!(em2.ended()[0].reason, "stopped");
    }

    #[test]
    fn commands_without_session_fail_gracefully() {
        let state = pendulum_state();
        assert!(live_status_impl(&state).unwrap().is_none());
        assert!(live_set_target_impl(&state, &[0.0, 0.0]).is_err());
        assert!(live_pause_impl(&state, true).is_err());
        assert!(live_reset_impl(&state, None).is_err());
        live_stop_impl(&state).unwrap(); // idempotent no-op
    }

    // -- gripper channel (B1) --

    /// A `LiveStartReq` with everything defaulted but the gripper knobs.
    fn gripper_req(q0: Vec<f64>, joint: Option<&str>, closed: Option<&str>) -> LiveStartReq {
        LiveStartReq {
            q0,
            engine: Some("builtin".into()),
            props: vec![],
            ground: None,
            kp: Some(400.0),
            kd: Some(40.0),
            emit_hz: None,
            gripper_joint: joint.map(str::to_string),
            gripper_closed: closed.map(str::to_string),
            success: None,
        }
    }

    #[test]
    fn engaged_needs_the_gripper_at_or_stalled_against_closed() {
        // open 0.04 → closed 0.0, span 0.04, so "near closed" is q <= 0.01.
        let g = GripperChannel {
            index: 2,
            name: "gripper".into(),
            open_target: 0.04,
            closed_target: 0.0,
        };
        assert!(gripper_engaged(0.0, &g, 0));
        assert!(gripper_engaged(0.01, &g, 0));
        assert!(!gripper_engaged(0.02, &g, 0), "half open is not closed");
        assert!(!gripper_engaged(0.04, &g, 999), "wide open is never closed");
        // Blocked: past halfway toward closed and no longer advancing. This is
        // the case that matters — a gripper holding a prop never reaches its
        // closed target, so a threshold alone would never fire.
        assert!(!gripper_engaged(0.015, &g, GRASP_STALL_TICKS - 1));
        assert!(gripper_engaged(0.015, &g, GRASP_STALL_TICKS));
        assert!(
            !gripper_engaged(0.03, &g, 10_000),
            "stalled on the OPEN side is not a grasp"
        );
        // A degenerate channel can never engage.
        let flat = GripperChannel {
            open_target: 0.0,
            ..g.clone()
        };
        assert!(!gripper_engaged(0.0, &flat, 10_000));
    }

    #[test]
    fn tracker_counts_only_non_advancing_ticks() {
        let g = GripperChannel {
            index: 2,
            name: "gripper".into(),
            open_target: 0.04,
            closed_target: 0.0,
        };
        let mut t = GraspTracker::default();
        assert_eq!(t.observe(0.040, &g), 0); // first tick: no history
        assert_eq!(t.observe(0.030, &g), 0); // closing fast
        assert_eq!(t.observe(0.020, &g), 0);
        assert_eq!(t.observe(0.020, &g), 1); // stopped
        assert_eq!(t.observe(0.020, &g), 2);
        assert_eq!(t.observe(0.010, &g), 0); // moving again resets
        assert_eq!(t.observe(0.011, &g), 1); // moving the WRONG way is not advancing
        t.clear();
        assert_eq!(t.observe(0.011, &g), 0);
    }

    #[test]
    fn gripper_channel_is_detected_and_reported() {
        let state = gripper_state();
        let em = Collect::default();
        let dto = live_start_on(
            &state,
            gripper_req(vec![0.0, 0.0, 0.02], None, None),
            em.clone(),
        )
        .expect("live_start");
        let g = dto.gripper.expect("gripper_arm has a `gripper` joint");
        assert_eq!(g.joint, "gripper");
        assert_eq!(g.index, 2);
        // limits 0..0.04, inset 2% of the range, closed at `lo` by default
        assert!((g.closed_target - 0.0008).abs() < 1e-12, "{g:?}");
        assert!((g.open_target - 0.0392).abs() < 1e-12, "{g:?}");

        sleep_ms(80);
        let s = em.last_state();
        let gs = s.gripper.expect("state events carry the gripper");
        assert!(!gs.closed, "a session starts with the gripper open");
        assert!((gs.q - 0.02).abs() < 0.01, "measured q {}", gs.q);
        assert!(s.held.is_none(), "builtin can never hold anything");

        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn live_gripper_moves_the_hold_target_and_nothing_else() {
        let state = gripper_state();
        let em = Collect::default();
        let dto = live_start_on(
            &state,
            gripper_req(vec![0.1, 0.0, 0.02], None, None),
            em.clone(),
        )
        .unwrap();
        let g = dto.gripper.unwrap();
        sleep_ms(60);
        let before = em.last_state().target;

        live_gripper_impl(&state, true).unwrap();
        sleep_ms(120);
        let s = em.last_state();
        assert_eq!(s.target[g.index], g.closed_target);
        assert_eq!(
            &s.target[..g.index],
            &before[..g.index],
            "other joints moved"
        );
        assert!(s.gripper.unwrap().closed);
        // and the arm actually tracks it
        assert!(
            (s.q[g.index] - g.closed_target).abs() < 0.01,
            "gripper did not close: q = {}",
            s.q[g.index]
        );
        // builtin has no contacts, so a closed gripper still holds nothing
        assert!(s.held.is_none());

        live_gripper_impl(&state, false).unwrap();
        sleep_ms(120);
        let s = em.last_state();
        assert_eq!(s.target[g.index], g.open_target);
        assert!(!s.gripper.unwrap().closed);

        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn a_robot_without_a_gripper_has_no_channel() {
        let state = pendulum_state();
        let em = Collect::default();
        let dto =
            live_start_on(&state, gripper_req(vec![0.0, 0.0], None, None), em.clone()).unwrap();
        assert!(dto.gripper.is_none(), "dyn_pendulum2 has no gripper");
        sleep_ms(60);
        let s = em.last_state();
        assert!(s.gripper.is_none() && s.held.is_none());

        let err = live_gripper_impl(&state, true).unwrap_err();
        assert!(err.contains("no gripper channel"), "got: {err}");
        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn the_gripper_override_wins_and_fails_loudly() {
        let state = pendulum_state();
        let em = Collect::default();
        // An explicit override needs no gripper-ish NAME — the human asked for
        // this joint.
        let dto = live_start_on(
            &state,
            gripper_req(vec![0.0, 0.0], Some("j2"), Some("hi")),
            em.clone(),
        )
        .unwrap();
        let g = dto.gripper.expect("the override installs a channel");
        assert_eq!(g.joint, "j2");
        // limits -3.14..3.14, closed at `hi`
        assert!(g.closed_target > g.open_target, "{g:?}");
        live_stop_impl(&state).unwrap();

        for (joint, closed, want) in [
            (Some("nope"), None, "not a joint of this robot"),
            (Some("j1"), Some("sideways"), "unknown gripperClosed"),
        ] {
            let err = live_start_on(
                &state,
                gripper_req(vec![0.0, 0.0], joint, closed),
                Collect::default(),
            )
            .map(|_| ())
            .unwrap_err();
            assert!(err.contains(want), "got: {err}");
            assert!(
                live_status_impl(&state).unwrap().is_none(),
                "a refused start must leave no session"
            );
        }
    }

    #[test]
    fn reset_reopens_the_gripper() {
        let state = gripper_state();
        let em = Collect::default();
        let dto = live_start_on(
            &state,
            gripper_req(vec![0.0, 0.0, 0.02], None, None),
            em.clone(),
        )
        .unwrap();
        let g = dto.gripper.unwrap();
        live_gripper_impl(&state, true).unwrap();
        sleep_ms(100);
        assert!(em.last_state().gripper.unwrap().closed);

        live_reset_impl(&state, None).unwrap();
        sleep_ms(100);
        let s = em.last_state();
        assert!(
            !s.gripper.unwrap().closed,
            "a reset returns the gripper to OPEN intent"
        );
        assert_eq!(s.target[g.index], 0.02, "and the target back to q0");
        assert!(s.held.is_none());
        live_stop_impl(&state).unwrap();
    }

    // -- success predicates (Wave C) --

    /// The lift predicate every success test below is built around.
    fn lift_cube(height: f64) -> serde_json::Value {
        serde_json::json!({"kind": "lifted", "prop": "cube", "height": height, "ref": "initial"})
    }

    /// A predicate is validated the moment it arrives, and a session that
    /// cannot judge one is never started.
    #[test]
    fn a_bad_success_predicate_is_refused_before_the_session_exists() {
        let state = gripper_state();
        let cases = [
            (
                serde_json::json!({"kind": "lifted", "prop": "cube", "heigth": 0.05}),
                "unknown key",
            ),
            (
                serde_json::json!({"kind": "levitated", "prop": "cube", "height": 0.05}),
                "unknown success predicate kind",
            ),
            (
                serde_json::json!({"kind": "lifted", "prop": "cube", "height": 0.0}),
                "not a lift",
            ),
            (
                serde_json::json!({"kind": "placed_in_zone", "prop": "cube", "zone": "bin"}),
                "by name",
            ),
            (
                serde_json::json!({"kind": "all_of", "terms": []}),
                "at least one term",
            ),
        ];
        for (spec, want) in cases {
            let mut req = gripper_req(vec![0.0, 0.0, 0.02], None, None);
            req.success = Some(spec.clone());
            let err = live_start_on(&state, req, Collect::default())
                .map(|_| ())
                .unwrap_err();
            assert!(err.contains(want), "for {spec}\n  got: {err}");
            assert!(
                live_status_impl(&state).unwrap().is_none(),
                "a refused start must leave no session"
            );
        }
    }

    /// The builtin engine has no props, so it scores nothing — and says so with
    /// `null` rather than a `false` that would read as "not yet".
    #[test]
    fn builtin_reports_no_verdict_at_all() {
        let state = gripper_state();
        let mut req = gripper_req(vec![0.0, 0.0, 0.02], None, None);
        req.success = Some(lift_cube(0.05));
        let em = Collect::default();
        live_start_on(&state, req, em.clone()).expect("builtin start with a predicate");
        sleep_ms(120);
        let states = em.states();
        assert!(!states.is_empty());
        assert!(
            states.iter().all(|s| s.success.is_none()),
            "builtin must report success = null"
        );
        live_stop_impl(&state).unwrap();
    }

    #[test]
    fn no_predicate_means_no_verdict() {
        let state = pendulum_state();
        let (em, _dto) = start_builtin(&state, vec![0.1, 0.0]);
        sleep_ms(80);
        assert!(em.last_state().success.is_none());
        live_stop_impl(&state).unwrap();
    }

    // -- teleop recording (A3) --

    fn rec_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("studio_live_rec_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn start_req(root: &Path, task: &str, fps: Option<u32>) -> LiveRecordStartReq {
        LiveRecordStartReq {
            root: root.display().to_string(),
            task: task.into(),
            fps,
        }
    }

    /// The timestamp the writer stores for frame `k` at `fps`: the exact
    /// `k / fps` seconds, rounded once to the f32 the parquet column holds.
    fn stamp(k: usize, fps: u32) -> f64 {
        ((k as f64 / f64::from(fps)) as f32) as f64
    }

    #[test]
    fn record_every_divides_the_tick_rate_exactly() {
        assert_eq!(record_every(1e-3, 50).unwrap(), 20);
        assert_eq!(record_every(1e-3, 100).unwrap(), 10);
        assert_eq!(record_every(1e-3, 250).unwrap(), 4);
        assert_eq!(record_every(1e-3, 1000).unwrap(), 1);
        // 1000 / 30 is not whole — recording would drift off frame_index/fps
        let err = record_every(1e-3, 30).unwrap_err();
        assert!(err.contains("divide") && err.contains("50"), "got: {err}");
        assert!(record_every(1e-3, 0).is_err());
        // a non-integer tick rate cannot host any exact decimation
        assert!(record_every(1.0 / 999.5, 50).is_err());
    }

    #[test]
    fn record_writes_a_v3_dataset_the_reader_opens() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.2, -0.1]);
        let dir = rec_dir("smoke");

        let started = live_record_start_impl(&state, start_req(&dir, "pick", Some(50))).unwrap();
        assert_eq!(started.fps, 50);
        assert_eq!(started.record_every, 20); // 1 kHz ticks / 50 fps
        assert_eq!(started.episode_index, 0);
        assert_eq!(started.root, dir.display().to_string());

        // drive the arm while the take runs, so the episode is not a constant
        live_set_target_impl(&state, &[0.1, 0.05]).unwrap();
        sleep_ms(250);
        live_set_target_impl(&state, &[-0.15, 0.2]).unwrap();
        sleep_ms(250);

        let st = live_record_status_impl(&state)
            .unwrap()
            .expect("dataset open");
        assert!(st.recording);
        assert_eq!(st.task.as_deref(), Some("pick"));
        assert_eq!(st.fps, 50);
        assert!(st.buffered_frames > 0);
        assert_eq!(st.episodes_saved, 0);
        assert!(live_status_impl(&state).unwrap().unwrap().recording);

        let stopped = live_record_stop_impl(&state, true).unwrap();
        assert!(stopped.saved);
        assert_eq!(stopped.episode_index, Some(0));
        // ~0.5 s at 50 fps ≈ 25 frames; loose bounds keep this robust on a
        // loaded machine while still proving the decimation is not per-tick.
        assert!(
            (5..=45).contains(&stopped.frames),
            "{} frames for ~0.5 s at 50 fps",
            stopped.frames
        );
        let saved = stopped.frames;

        // a second take on the same dataset, thrown away
        let started2 = live_record_start_impl(&state, start_req(&dir, "place", None)).unwrap();
        assert_eq!(started2.episode_index, 1);
        assert_eq!(started2.fps, 50); // the default
        sleep_ms(200);
        let dropped = live_record_stop_impl(&state, false).unwrap();
        assert!(!dropped.saved);
        assert_eq!(dropped.episode_index, None);
        assert!(dropped.frames > 0);

        let st = live_record_status_impl(&state).unwrap().unwrap();
        assert!(!st.recording);
        assert!(st.task.is_none());
        assert_eq!(st.buffered_frames, 0);
        assert_eq!(st.episodes_saved, 1);

        let fin = live_record_finish_impl(&state).unwrap();
        assert_eq!(fin.episodes, 1);
        assert!(live_record_status_impl(&state).unwrap().is_none());

        let r = caliper_dataset::DatasetReader::open(&dir).unwrap();
        assert_eq!(r.total_episodes(), 1);
        assert_eq!(r.fps(), 50);
        let ep = r.read_episode(0).unwrap();
        assert_eq!(ep.len(), saved);
        assert_eq!(ep.tasks, vec!["pick".to_string()]);
        let states = ep.features.get("observation.state").expect("state feature");
        let actions = ep.features.get("action").expect("action feature");
        assert_eq!(states.len(), saved);
        assert_eq!(actions.len(), saved);
        assert!(states.iter().all(|s| s.len() == 2));
        assert!(actions.iter().all(|a| a.len() == 2));
        assert!(
            states.iter().any(|s| (s[0] - states[0][0]).abs() > 1e-4),
            "the recorded arm never moved"
        );
        // exact decimation → exact frame_index/fps timestamps, no resampling
        for (k, ts) in ep.timestamps.iter().enumerate() {
            assert_eq!(*ts, stamp(k, 50), "timestamp {k}");
        }

        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_commands_refuse_bad_requests() {
        let state = pendulum_state();
        let dir = rec_dir("errs");

        // no session at all
        let err = live_record_start_impl(&state, start_req(&dir, "t", None)).unwrap_err();
        assert!(err.contains("no live session"), "got: {err}");
        assert!(live_record_status_impl(&state).unwrap().is_none());

        let (_em, _dto) = start_builtin(&state, vec![0.0, 0.0]);
        let err = live_record_start_impl(&state, start_req(&dir, "   ", None)).unwrap_err();
        assert!(err.contains("task label"), "got: {err}");
        // an fps that does not divide the tick rate — and nothing is created
        let err = live_record_start_impl(&state, start_req(&dir, "t", Some(30))).unwrap_err();
        assert!(err.contains("1000 Hz tick rate"), "got: {err}");
        assert!(!dir.exists(), "a refused start must not create the dataset");

        assert!(live_record_stop_impl(&state, true)
            .unwrap_err()
            .contains("no dataset open"));
        assert!(live_record_finish_impl(&state)
            .unwrap_err()
            .contains("no dataset open"));

        live_record_start_impl(&state, start_req(&dir, "take", Some(50))).unwrap();
        let err = live_record_start_impl(&state, start_req(&dir, "again", Some(50))).unwrap_err();
        assert!(err.contains("already recording"), "got: {err}");
        // a different root, or a different fps, while a dataset is open
        let other = rec_dir("errs_other");
        let err = live_record_start_impl(&state, start_req(&other, "t", Some(50))).unwrap_err();
        assert!(err.contains("finish the open dataset first"), "got: {err}");
        assert!(!other.exists(), "the refused root must not be created");
        let err = live_record_start_impl(&state, start_req(&dir, "t", Some(100))).unwrap_err();
        assert!(err.contains("finish the open dataset first"), "got: {err}");
        // finishing mid-take
        let err = live_record_finish_impl(&state).unwrap_err();
        assert!(err.contains("stop the take first"), "got: {err}");

        sleep_ms(150);
        live_record_stop_impl(&state, true).unwrap();
        let err = live_record_stop_impl(&state, true).unwrap_err();
        assert!(err.contains("not recording"), "got: {err}");
        live_record_finish_impl(&state).unwrap();

        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Poll `f` until it yields, well inside the 2 s recording-reply timeout so
    /// the waiter under test is still waiting when the condition holds.
    fn wait_for<T>(mut f: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(v) = f() {
                return v;
            }
            assert!(Instant::now() < deadline, "the condition never held");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_late_reply_is_never_delivered_to_the_next_command() {
        // Command A gave up waiting while the thread was still servicing it
        // (a long finalize). Command B posts the SAME variant right after, and
        // must not be handed A's answer when it finally lands — that is how the
        // UI came to report "saved episode N" for a take that was discarded.
        let shared = Arc::new(LiveShared::new(vec![0.0, 0.0]));
        let a_id = shared.rec_seq.fetch_add(1, Ordering::Relaxed) + 1;

        let waiter = {
            let shared = shared.clone();
            std::thread::spawn(move || rec_request(&shared, RecRequest::Stop { save: false }))
        };
        // B's request, taken the way the session thread would take it.
        let b_id = wait_for(|| shared.rec_req.lock().unwrap().take().map(|(id, _)| id));
        assert_eq!(b_id, a_id + 1, "ids are monotonic");

        // A's answer arrives late, claiming a saved episode.
        *shared.rec_reply.lock().unwrap() = Some((
            a_id,
            Ok(RecReply::Stopped {
                saved: true,
                episode_index: Some(7),
                frames: 99,
            }),
        ));
        // B throws it away (clearing the slot) instead of returning it.
        wait_for(|| shared.rec_reply.lock().unwrap().is_none().then_some(()));

        *shared.rec_reply.lock().unwrap() = Some((
            b_id,
            Ok(RecReply::Stopped {
                saved: false,
                episode_index: None,
                frames: 3,
            }),
        ));
        match waiter.join().unwrap().expect("B gets its own answer") {
            RecReply::Stopped {
                saved,
                episode_index,
                frames,
            } => {
                assert!(!saved, "B was handed A's reply");
                assert_eq!(episode_index, None);
                assert_eq!(frames, 3);
            }
            _ => panic!("wrong reply variant"),
        }
    }

    #[test]
    fn a_start_that_loses_the_race_to_stop_leaves_no_empty_dataset() {
        let dir = rec_dir("race");
        let spec = DatasetSpec::new(
            50,
            "pendulum".to_string(),
            vec![
                FeatureSpec::vector("observation.state", 2, None),
                FeatureSpec::vector("action", 2, None),
            ],
        );
        let w = Box::new(DatasetWriter::create(&dir, spec).expect("writer"));
        assert!(dir.exists(), "the writer creates its root");

        // The session ends between posting the start and the thread servicing
        // it: the loop checks `stop` first, so the request is never taken and
        // this call is the writer's only owner.
        let shared = LiveShared::new(vec![0.0, 0.0]);
        shared.stop.store(true, Ordering::Relaxed);
        let err = match rec_start_request(
            &shared,
            RecRequest::Start {
                writer: Some(w),
                root: dir.display().to_string(),
                fps: 50,
                record_every: 20,
                task: "t".into(),
            },
        ) {
            Err(e) => e,
            Ok(_) => panic!("a stopped session must not accept a start"),
        };
        assert!(err.contains("live session ended"), "got: {err}");
        assert!(
            shared.rec_req.lock().unwrap().is_none(),
            "the request must be reclaimed, not left queued"
        );
        assert!(
            !dir.exists(),
            "an empty finalized dataset was stranded at {}",
            dir.display()
        );
    }

    #[test]
    fn a_started_dataset_keeps_its_root() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.1, -0.1]);
        let dir = rec_dir("race_ok");

        live_record_start_impl(&state, start_req(&dir, "keep", Some(50))).unwrap();
        assert!(dir.exists());
        sleep_ms(150);
        assert!(live_record_stop_impl(&state, true).unwrap().saved);
        live_record_finish_impl(&state).unwrap();
        assert!(dir.exists(), "an accepted start must leave its dataset");

        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_reset_discards_the_take_but_keeps_the_dataset() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.2, -0.1]);
        let dir = rec_dir("reset");

        live_record_start_impl(&state, start_req(&dir, "spoiled", Some(50))).unwrap();
        sleep_ms(200);
        assert!(
            live_record_status_impl(&state)
                .unwrap()
                .unwrap()
                .buffered_frames
                > 0
        );

        live_reset_impl(&state, None).unwrap();
        sleep_ms(100);
        let st = live_record_status_impl(&state)
            .unwrap()
            .expect("the dataset stays open across a reset");
        assert!(!st.recording, "a reset invalidates the take");
        assert_eq!(st.buffered_frames, 0);
        assert_eq!(st.episodes_saved, 0);
        assert_eq!(st.root, dir.display().to_string());

        // and the dataset still records: the next take saves normally
        live_record_start_impl(&state, start_req(&dir, "clean", Some(50))).unwrap();
        sleep_ms(200);
        assert!(live_record_stop_impl(&state, true).unwrap().saved);
        assert_eq!(live_record_finish_impl(&state).unwrap().episodes, 1);

        let r = caliper_dataset::DatasetReader::open(&dir).unwrap();
        assert_eq!(r.total_episodes(), 1);
        assert_eq!(r.read_episode(0).unwrap().tasks, vec!["clean".to_string()]);
        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pausing_mid_take_freezes_capture_and_resumes_the_same_take() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.2, -0.1]);
        let dir = rec_dir("pause");

        live_record_start_impl(&state, start_req(&dir, "paused", Some(50))).unwrap();
        sleep_ms(200);
        live_pause_impl(&state, true).unwrap();
        sleep_ms(60);
        let a = live_record_status_impl(&state)
            .unwrap()
            .unwrap()
            .buffered_frames;
        sleep_ms(150);
        let b = live_record_status_impl(&state)
            .unwrap()
            .unwrap()
            .buffered_frames;
        assert!(a > 0);
        assert_eq!(a, b, "a paused sim captures no frames");

        live_pause_impl(&state, false).unwrap();
        sleep_ms(200);
        let stopped = live_record_stop_impl(&state, true).unwrap();
        assert!(stopped.saved);
        assert!(stopped.frames > b, "the same take continues after resume");
        live_record_finish_impl(&state).unwrap();

        // Timestamps are frame-count based, so the paused span leaves NO gap.
        let r = caliper_dataset::DatasetReader::open(&dir).unwrap();
        let ep = r.read_episode(0).unwrap();
        assert_eq!(ep.len(), stopped.frames);
        for (k, ts) in ep.timestamps.iter().enumerate() {
            assert_eq!(*ts, stamp(k, 50), "timestamp {k}");
        }
        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stopping_the_session_finalizes_an_open_dataset() {
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.1, 0.0]);
        let dir = rec_dir("sessionend");

        live_record_start_impl(&state, start_req(&dir, "held", Some(50))).unwrap();
        sleep_ms(200);
        assert!(live_record_stop_impl(&state, true).unwrap().saved);
        // leave a take running: ending the session must drop it, not save it
        live_record_start_impl(&state, start_req(&dir, "aborted", Some(50))).unwrap();
        sleep_ms(150);

        live_stop_impl(&state).unwrap(); // joins the thread → finalize is done
        assert!(live_record_status_impl(&state).unwrap().is_none());

        let r = caliper_dataset::DatasetReader::open(&dir).unwrap();
        assert_eq!(r.total_episodes(), 1, "an interrupted take is never saved");
        assert_eq!(r.read_episode(0).unwrap().tasks, vec!["held".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn state_events_carry_the_recording_flag() {
        let state = pendulum_state();
        let (em, _dto) = start_builtin(&state, vec![0.2, -0.1]);
        sleep_ms(60);
        let idle = em.last_state();
        assert!(!idle.recording);
        assert_eq!(idle.rec_frames, 0);

        let dir = rec_dir("events");
        live_record_start_impl(&state, start_req(&dir, "wave", Some(50))).unwrap();
        sleep_ms(150);
        let a = em.last_state();
        assert!(a.recording);
        assert!(a.rec_frames > 0);
        sleep_ms(200);
        let b = em.last_state();
        assert!(
            b.rec_frames > a.rec_frames,
            "recFrames must advance during a take ({} → {})",
            a.rec_frames,
            b.rec_frames
        );

        live_record_stop_impl(&state, false).unwrap();
        sleep_ms(80);
        let c = em.last_state();
        assert!(!c.recording);
        assert_eq!(c.rec_frames, 0);

        live_stop_impl(&state).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Manual acceptance run for the lerobot pairing check: records two ~2 s
    /// episodes at 50 fps into `$CALIPER_LIVE_RECORD_DIR` (a real directory,
    /// not a tempdir) and finalizes it, so the result can be loaded with
    /// lerobot 0.6.0.
    ///
    /// ```text
    /// CALIPER_LIVE_RECORD_DIR=/tmp/live_teleop \
    ///   cargo test -p studio --lib record_dataset_to_env_path -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn record_dataset_to_env_path() {
        let dir = std::env::var("CALIPER_LIVE_RECORD_DIR")
            .expect("set CALIPER_LIVE_RECORD_DIR to the dataset directory to create");
        let state = pendulum_state();
        let (_em, _dto) = start_builtin(&state, vec![0.3, -0.2]);

        for task in ["reach", "return"] {
            live_record_start_impl(
                &state,
                LiveRecordStartReq {
                    root: dir.clone(),
                    task: task.into(),
                    fps: Some(50),
                },
            )
            .unwrap();
            // ~2 s of driving, so the episodes carry real motion
            for tgt in [[0.4, -0.3], [0.0, 0.1], [-0.3, 0.4], [0.1, 0.0]] {
                live_set_target_impl(&state, &tgt).unwrap();
                sleep_ms(500);
            }
            let stopped = live_record_stop_impl(&state, true).unwrap();
            assert!(stopped.saved);
            println!(
                "episode {:?} ({task}): {} frames",
                stopped.episode_index, stopped.frames
            );
        }
        let fin = live_record_finish_impl(&state).unwrap();
        assert_eq!(fin.episodes, 2);
        println!("finalized {} with {} episodes", fin.root, fin.episodes);
        live_stop_impl(&state).unwrap();
    }

    // -- mujoco-gated --

    #[cfg(feature = "mujoco")]
    mod mujoco {
        use super::*;

        fn box_prop() -> PropDto {
            PropDto {
                name: "crate".into(),
                kind: "box".into(),
                half_extents: Some([0.05; 3]),
                radius: None,
                length: None,
                pos: [0.3, 0.0, 0.5],
                quat: None,
                mass: None,
                rgba: None,
                material: None,
            }
        }

        #[test]
        fn mujoco_session_smoke_with_prop() {
            let state = pendulum_state();
            let em = Collect::default();
            let dto = live_start_on(
                &state,
                LiveStartReq {
                    q0: vec![0.2, -0.1],
                    engine: Some("mujoco".into()),
                    props: vec![box_prop()],
                    ground: None,
                    kp: None,
                    kd: None,
                    emit_hz: None,
                    gripper_joint: None,
                    gripper_closed: None,
                    success: None,
                },
                em.clone(),
            )
            .expect("mujoco live_start");
            assert_eq!(dto.engine, "mujoco");
            assert_eq!(dto.ndof, 2);
            assert_eq!(dto.props.len(), 1);
            assert!(dto.props[0].frames.is_empty());

            sleep_ms(150);
            let states = em.states();
            assert!(states.len() >= 3);
            for w in states.windows(2) {
                assert!(w[1].tick > w[0].tick);
            }
            let last = em.last_state();
            assert!(last.q.iter().all(|x| x.is_finite()));
            // the free box is tracked every event, in build order
            assert_eq!(last.props.len(), 1);
            assert!(last.props[0].iter().all(|x| x.is_finite()));
            // the falling box's z decreased from its spawn height
            assert!(last.props[0][2] < 0.5);

            live_stop_impl(&state).unwrap();
            assert_eq!(em.ended()[0].reason, "stopped");
        }

        /// The 5 cm cube the `gripper_arm` fixture is built around: resting on
        /// the ground with the jaw already 1 mm into its top face.
        fn cube_prop() -> PropDto {
            PropDto {
                name: "cube".into(),
                kind: "box".into(),
                half_extents: Some([0.05; 3]),
                radius: None,
                length: None,
                pos: [0.0, 0.0, 0.05],
                quat: None,
                mass: Some(0.05),
                rgba: None,
                material: None,
            }
        }

        fn start_grasp_session(state: &AppState) -> (Collect, LiveStartedDto) {
            start_grasp_session_with(state, None)
        }

        /// The grasp rig, optionally judged by a success predicate.
        fn start_grasp_session_with(
            state: &AppState,
            success: Option<serde_json::Value>,
        ) -> (Collect, LiveStartedDto) {
            let em = Collect::default();
            let dto = live_start_on(
                state,
                LiveStartReq {
                    q0: vec![0.0, 0.0, 0.02],
                    engine: Some("mujoco".into()),
                    props: vec![cube_prop()],
                    ground: Some(0.0),
                    kp: Some(400.0),
                    kd: Some(40.0),
                    emit_hz: None,
                    gripper_joint: None,
                    gripper_closed: None,
                    success,
                },
                em.clone(),
            )
            .expect("mujoco live_start with a gripper");
            (em, dto)
        }

        /// Poll the stream until `f` holds, up to `ms`. Returns whether it did.
        fn wait_for(em: &Collect, ms: u64, f: impl Fn(&LiveStateEvent) -> bool) -> bool {
            let deadline = Instant::now() + Duration::from_millis(ms);
            while Instant::now() < deadline {
                if em.states().last().is_some_and(&f) {
                    return true;
                }
                sleep_ms(10);
            }
            false
        }

        /// The whole grasp loop through the live session: close on a prop in
        /// contact, carry it off the ground, release it, and see it fall.
        #[test]
        fn gripper_grabs_carries_and_releases_a_prop() {
            let state = gripper_state();
            let (em, dto) = start_grasp_session(&state);
            let g = dto.gripper.expect("gripper_arm has a gripper channel");
            assert_eq!(g.joint, "gripper");

            // Idle: touching the cube, but an OPEN gripper holds nothing.
            sleep_ms(150);
            let s = em.last_state();
            assert!(
                s.held.is_none(),
                "nothing is held before the gripper closes"
            );
            assert!(s.ncon > 0, "the jaw should already be touching the cube");
            let rest_z = s.props[0][2];

            // Close: the weld takes the cube it is in contact with.
            live_gripper_impl(&state, true).unwrap();
            assert!(
                wait_for(&em, 2000, |s| s.held.as_deref() == Some("cube")),
                "the gripper never took the cube (last state: {:?})",
                em.last_state().held
            );

            // Carry: swing j1 up; a welded cube must leave the ground with it.
            live_set_target_impl(&state, &[0.7, 0.0, g.closed_target]).unwrap();
            assert!(
                wait_for(&em, 3000, |s| s.props[0][2] > rest_z + 0.05),
                "the cube never left the ground (z {rest_z} → {})",
                em.last_state().props[0][2]
            );
            let s = em.last_state();
            assert_eq!(s.held.as_deref(), Some("cube"), "still held while carried");
            assert!(
                s.props[0][0].abs() > 0.05,
                "the cube should have swung sideways with the arm, x = {}",
                s.props[0][0]
            );
            let carried_z = s.props[0][2];

            // Release: the cube keeps its state and falls.
            live_gripper_impl(&state, false).unwrap();
            assert!(
                wait_for(&em, 1000, |s| s.held.is_none()),
                "releasing did not clear `held`"
            );
            assert!(
                wait_for(&em, 2000, |s| s.props[0][2] < carried_z - 0.03),
                "the released cube did not fall (z {carried_z} → {})",
                em.last_state().props[0][2]
            );

            live_stop_impl(&state).unwrap();
        }

        /// A reset drops whatever the gripper was holding.
        #[test]
        fn reset_releases_a_held_prop() {
            let state = gripper_state();
            let (em, _dto) = start_grasp_session(&state);
            sleep_ms(150);
            live_gripper_impl(&state, true).unwrap();
            assert!(
                wait_for(&em, 2000, |s| s.held.as_deref() == Some("cube")),
                "the gripper never took the cube"
            );

            live_reset_impl(&state, None).unwrap();
            assert!(
                wait_for(&em, 1000, |s| s.held.is_none()
                    && s.gripper.as_ref().is_some_and(|g| !g.closed)),
                "reset must drop the prop and reopen the gripper"
            );
            live_stop_impl(&state).unwrap();
        }

        /// An open gripper never grabs, however long it sits in contact.
        #[test]
        fn contact_alone_is_not_a_grasp() {
            let state = gripper_state();
            let (em, _dto) = start_grasp_session(&state);
            sleep_ms(400);
            let s = em.last_state();
            assert!(
                s.ncon > 0,
                "the rig must be in contact for this to prove anything"
            );
            assert!(s.held.is_none(), "an open gripper grabbed something");
            live_stop_impl(&state).unwrap();
        }

        /// The Wave C payoff: a task's success criterion, judged live. Same
        /// choreography as the grasp test — close on the cube, swing it up —
        /// with the verdict flipping on the tick the lift clears its bar, and
        /// re-anchoring (back to `false`) when the episode resets.
        #[test]
        fn a_lift_predicate_flips_true_when_the_prop_is_carried_up() {
            let state = gripper_state();
            let (em, dto) = start_grasp_session_with(&state, Some(lift_cube(0.05)));
            let g = dto.gripper.expect("gripper_arm has a gripper channel");

            // The baseline is the pose the session STARTED from, so a cube
            // sitting where it spawned is not a lift.
            sleep_ms(150);
            let s = em.last_state();
            assert_eq!(s.success, Some(false), "an untouched cube is not lifted");
            let rest_z = s.props[0][2];

            live_gripper_impl(&state, true).unwrap();
            assert!(
                wait_for(&em, 2000, |s| s.held.as_deref() == Some("cube")),
                "the gripper never took the cube"
            );
            // Still not a lift while it rests on the ground.
            assert_eq!(em.last_state().success, Some(false));

            live_set_target_impl(&state, &[0.7, 0.0, g.closed_target]).unwrap();
            assert!(
                wait_for(&em, 3000, |s| s.success == Some(true)),
                "carrying the cube up never satisfied the predicate (z {rest_z} → {}, \
                 success {:?})",
                em.last_state().props[0][2],
                em.last_state().success
            );
            // The verdict describes the poses in its OWN event.
            let s = em.last_state();
            assert!(
                s.props[0][2] > rest_z + 0.05,
                "success true at z = {} (rest {rest_z})",
                s.props[0][2]
            );

            // A reset re-anchors the baseline: the cube is back on the ground,
            // and the next episode starts from `false` again.
            live_reset_impl(&state, None).unwrap();
            assert!(
                wait_for(&em, 2000, |s| s.success == Some(false) && s.held.is_none()),
                "reset must re-anchor the lift baseline (success {:?})",
                em.last_state().success
            );
            live_stop_impl(&state).unwrap();
        }

        /// A `settled_speed` term needs prop VELOCITIES, and the live session
        /// has them — a resting cube inside the zone is `placed`, the same cube
        /// judged velocity-less would have been an error.
        #[test]
        fn a_settled_zone_predicate_is_answerable_live() {
            let state = gripper_state();
            let spec = serde_json::json!({
                "kind": "placed_in_zone",
                "prop": "cube",
                "zone": {"center": [0.0, 0.0, 0.05], "half": [0.2, 0.2, 0.2]},
                "settled_speed": 0.05,
            });
            let (em, _dto) = start_grasp_session_with(&state, Some(spec));
            assert!(
                wait_for(&em, 2000, |s| s.success == Some(true)),
                "a cube at rest inside the zone should be `placed` (success {:?})",
                em.last_state().success
            );
            live_stop_impl(&state).unwrap();
        }

        /// A predicate about a prop the scene does not contain is refused at
        /// start, not discovered as a dead session on the first state event.
        #[test]
        fn a_predicate_about_a_missing_prop_is_refused() {
            let state = gripper_state();
            let em = Collect::default();
            let err = live_start_on(
                &state,
                LiveStartReq {
                    q0: vec![0.0, 0.0, 0.02],
                    engine: Some("mujoco".into()),
                    props: vec![cube_prop()],
                    ground: Some(0.0),
                    kp: None,
                    kd: None,
                    emit_hz: None,
                    gripper_joint: None,
                    gripper_closed: None,
                    success: Some(lift_cube_named("ball")),
                },
                em,
            )
            .map(|_| ())
            .unwrap_err();
            assert!(err.contains("does not contain"), "got: {err}");
            assert!(live_status_impl(&state).unwrap().is_none());
        }

        fn lift_cube_named(prop: &str) -> serde_json::Value {
            serde_json::json!({"kind": "lifted", "prop": prop, "height": 0.05})
        }

        #[test]
        fn mujoco_reset_is_bitwise_deterministic() {
            // Drives LiveEngine directly (no wall clock): reset + N identical
            // steps twice must be bitwise identical — the MujocoSim::reset
            // (mj_resetData incl. warmstart) guarantee surviving the live path.
            let m = Arc::new(Model::from_urdf(&fixture("dyn_pendulum2.urdf")).unwrap());
            let gains = Gains {
                kp: 100.0,
                kd: 20.0,
            };
            let q0 = vec![0.3, -0.2];
            let (engine, h) =
                build_engine(&m, "mujoco", &q0, &[box_prop()], 0.0, gains, None).unwrap();
            let run = |mut e: LiveEngine| -> (Vec<f64>, LiveEngine) {
                e = e.reset(&q0, &m, gains, h).unwrap();
                let mut sp = TeleopSetpoint::new(vec![0.1, 0.0]).with_tick_budget(u64::MAX);
                for _ in 0..200 {
                    e.step(&mut sp).unwrap();
                }
                let (q, _, _, _) = e.snapshot();
                (q, e)
            };
            let (qa, engine) = run(engine);
            let (qb, _) = run(engine);
            for i in 0..2 {
                assert_eq!(qa[i].to_bits(), qb[i].to_bits(), "joint {i} diverged");
            }
        }
    }
}
