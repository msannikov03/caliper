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

use crate::{bake_frame_row, logged, AppState, PropDto, PropTrackDto};
use caliper::hal::{ControlLoop, Gains, PhysicsSimBackend, TeleopSetpoint};
use caliper::model::Model;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Maximum wall-clock debt (s) the pacer will convert into catch-up steps.
const CATCHUP_CAP_S: f64 = 0.25;

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
        }
    }
}

/// A running live session, held in `AppState.live`. Dropping the slot without
/// `stop` would leak the thread — every taker must raise `stop` and join.
pub(crate) struct LiveSession {
    id: u64,
    engine: &'static str,
    ndof: usize,
    /// Initial pose — the `live_reset(None)` restore point.
    q0: Vec<f64>,
    shared: Arc<LiveShared>,
    join: Option<JoinHandle<()>>,
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

// ===== the engine =====

enum LiveEngine {
    Builtin(ControlLoop<PhysicsSimBackend>),
    #[cfg(feature = "mujoco")]
    Mujoco(ControlLoop<caliper_sim_mujoco::MujocoBackend>),
}

impl LiveEngine {
    fn step(&mut self, sp: &mut TeleopSetpoint) -> Result<(), String> {
        match self {
            LiveEngine::Builtin(l) => l.step(sp, None).map(|_| ()).map_err(|e| e.to_string()),
            #[cfg(feature = "mujoco")]
            LiveEngine::Mujoco(l) => l.step(sp, None).map(|_| ()).map_err(|e| e.to_string()),
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

// ===== the session loop =====

fn set_status(shared: &LiveShared, paused: bool, t: f64, tick: u64) {
    if let Ok(mut s) = shared.status.lock() {
        *s = LiveStatusInner { paused, t, tick };
    }
}

fn state_event(
    engine: &LiveEngine,
    model: &Model,
    shared: &LiveShared,
    session_id: u64,
    paused: bool,
) -> Option<LiveStateEvent> {
    let (q, qd, ncon, props) = engine.snapshot();
    if !(q.iter().all(|x| x.is_finite()) && qd.iter().all(|x| x.is_finite())) {
        return None; // non-finite → the caller ends the session with an error
    }
    let (frames, tip) = bake_frame_row(model, &q);
    let target = shared
        .target
        .lock()
        .map(|t| t.clone())
        .unwrap_or_else(|_| q.clone());
    Some(LiveStateEvent {
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
    })
}

/// The session body, generic over the emitter so tests run it headlessly.
/// Owns the engine until the thread exits; every exit path emits `live://ended`.
#[allow(clippy::too_many_arguments)]
fn run_session<E: LiveEmitter>(
    mut engine: LiveEngine,
    model: Arc<Model>,
    gains: Gains,
    h: f64,
    emit_every: u64,
    session_id: u64,
    shared: Arc<LiveShared>,
    emitter: E,
) {
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
    let mut pacer = Pacer::new(h, CATCHUP_CAP_S);
    let mut last_wall = Instant::now();
    let mut last_paused = shared.paused.load(Ordering::Relaxed);
    let mut since_emit: u64 = 0;

    let fail = |shared: &LiveShared, emitter: &E, detail: String| {
        shared.dead.store(true, Ordering::Relaxed);
        emitter.ended(&LiveEndedEvent {
            session_id,
            reason: format!("error: {detail}"),
        });
    };
    macro_rules! emit_state_or_die {
        ($paused:expr) => {
            match state_event(&engine, &model, &shared, session_id, $paused) {
                Some(ev) => {
                    set_status(&shared, $paused, ev.t, ev.tick);
                    emitter.state(&ev);
                }
                None => {
                    fail(&shared, &emitter, "simulation state went non-finite".into());
                    return;
                }
            }
        };
    }

    emit_state_or_die!(last_paused); // initial snapshot so the UI paints at once

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            let reason = if shared.superseded.load(Ordering::Relaxed) {
                "superseded"
            } else {
                "stopped"
            };
            emitter.ended(&LiveEndedEvent {
                session_id,
                reason: reason.into(),
            });
            return;
        }

        // Reset request (honored even while paused; paused flag survives).
        let req = shared.reset_req.lock().ok().and_then(|mut r| r.take());
        if let Some(qr) = req {
            engine = match engine.reset(&qr, &model, gains, h) {
                Ok(e) => e,
                Err(e) => {
                    fail(&shared, &emitter, format!("reset failed: {e}"));
                    return;
                }
            };
            if let Ok(mut t) = shared.target.lock() {
                *t = qr.clone();
            }
            sp = TeleopSetpoint::new(qr).with_tick_budget(u64::MAX);
            pacer.clear();
            last_wall = Instant::now();
            since_emit = 0;
            emit_state_or_die!(last_paused);
            continue;
        }

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
            if let Err(e) = engine.step(&mut sp) {
                fail(&shared, &emitter, e);
                return;
            }
            since_emit += 1;
            if since_emit >= emit_every {
                since_emit = 0;
                emit_state_or_die!(false);
            }
        }
        set_status(&shared, false, engine.time(), engine.tick());
    }
}

// ===== engine construction (inside the command; errors before any thread) =====

/// Engine + its physics timestep. Validation errors return before any spawn.
fn build_engine(
    model: &Arc<Model>,
    engine_name: &str,
    q0: &[f64],
    props: &[PropDto],
    ground: f64,
    gains: Gains,
) -> Result<(LiveEngine, f64), String> {
    let zeros = vec![0.0; q0.len()];
    match engine_name {
        "builtin" => {
            if !props.is_empty() {
                return Err("props need the mujoco engine — builtin has no contact".into());
            }
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
                let _ = ground;
                Err("contact sim not compiled — build studio with --features mujoco".into())
            }
        }
        other => Err(format!("unknown engine `{other}` (mujoco|builtin)")),
    }
}

// ===== command impls (take &AppState so tests drive them without tauri) =====

/// Raise `stop` on an existing session (reason per `superseded`) and join it.
fn stop_in_slot(slot: &mut Option<LiveSession>, superseded: bool) {
    if let Some(mut s) = slot.take() {
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

    let (engine, h) = build_engine(&arc, engine_name, &req.q0, &req.props, ground, gains)?;
    let every = emit_every(clamp_emit_hz(req.emit_hz), h);
    let actual_hz = 1.0 / (every as f64 * h);

    let mut slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    stop_in_slot(&mut slot, true); // an existing session ends as "superseded"

    let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
    let shared = Arc::new(LiveShared::new(req.q0.clone()));
    let thread_shared = shared.clone();
    let thread_model = arc.clone();
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
                emitter,
            )
        })
        .map_err(|e| format!("failed to spawn the live sim thread: {e}"))?;

    *slot = Some(LiveSession {
        id: session_id,
        engine: engine_name,
        ndof: n,
        q0: req.q0,
        shared,
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
    })
}

/// Run `f` on the live session, reaping the slot first if its thread died.
fn with_live<T>(
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
            }))
        }
        None => Ok(None),
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

// ===== tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
                }],
                ground: None,
                kp: None,
                kd: None,
                emit_hz: None,
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
            let (engine, h) = build_engine(&m, "mujoco", &q0, &[box_prop()], 0.0, gains).unwrap();
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
