//! Policy-in-the-loop bridge (E1): a trained policy DRIVES the running live
//! session, in the same place a human's hands would.
//!
//! Studio spawns a user-pointed python env running `caliper-learn drive` and
//! speaks one line of JSON per message over its stdin/stdout. Every action that
//! comes back is written into [`LiveShared::write_target`] — the SAME single PD
//! hold target the sliders, the IK gizmo, the keyboard jog and `live_gripper`
//! write. There is no second control path: a policy trained with a gripper
//! drives the gripper slot exactly like any other joint, and a human can nudge
//! the arm mid-rollout (the next action simply overwrites the nudge; SPACE
//! suspends the policy outright).
//!
//! # The protocol
//!
//! ```text
//! spawn:  <python> -m caliper_learn.cli drive <ckpt> --urdf <urdf> [--device cpu]
//! py  ->  {"type":"ready","ndof":N,"chunk":C,"policyType":"...","device":"..."}
//!    or   {"type":"error","message":"..."}            (then a nonzero exit)
//! st  ->  {"type":"obs","tick":T,"t":S,"q":[...],"qd":[...]}
//! py  ->  {"type":"action","tick":T,"q":[...]}        (may lag; latest wins)
//! st  ->  {"type":"stop"}                             (then SIGKILL after 2 s)
//! ```
//!
//! Anything else on stdout is a protocol error: the child is killed and the
//! failure surfaces as `live://policy` with `state: "error"`. stderr is pumped
//! into an 8 KB ring so a python traceback is quoted back to the human instead
//! of vanishing into a closed pipe.
//!
//! # Why the sim thread cannot block on the child
//!
//! Three threads, none of them the sim thread:
//!
//! * the SESSION thread publishes `(tick, t, q, qd)` into [`PolicyBus`] on each
//!   state emission — a `Mutex` write into pre-grown vectors (no allocation, no
//!   I/O, and skipped entirely when no policy is attached) followed by a
//!   sequence bump. It never reads the child, never waits for one, and cannot
//!   tell whether a policy is keeping up.
//! * the DRIVER thread polls that sequence at the session's own emit rate,
//!   decimates it down to the requested policy rate, and writes obs lines. It
//!   owns the child process and stdin, so a wedged python blocks only here.
//! * the READER thread blocks on the child's stdout and writes each validated
//!   action into the shared target. Latest-wins falls out of that for free: a
//!   policy that lags simply leaves the last action standing.
//!
//! The bus mutex is the only thing the two sides share, and the driver holds it
//! just long enough to clone a snapshot — never across a write to the child.
//! (The alternative, publishing THROUGH a channel the child drains, would make
//! sim pacing a function of python's latency; this way a 300 ms policy costs
//! the sim exactly nothing.)

use crate::live::{with_live, LiveShared};
use crate::{logged, AppState};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long `live_policy_start` waits for the child's `ready` line. Generous on
/// purpose: importing torch and materializing a checkpoint routinely takes tens
/// of seconds on a cold page cache.
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a stopping child gets to exit on its own before SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(2);

/// Observation rate when the request omits `hz`.
const DEFAULT_OBS_HZ: f64 = 20.0;

/// Driver poll period — fine enough that decimation, pause and stop all land
/// within a frame, cheap enough to be invisible.
const DRIVER_POLL: Duration = Duration::from_millis(1);

/// Bytes of the child's stderr kept for error reporting.
const STDERR_TAIL_BYTES: usize = 8 * 1024;

/// How long a failure path waits for the stderr pump to drain before quoting
/// the tail. Bounded: a child that is still alive never closes stderr.
const STDERR_SETTLE: Duration = Duration::from_millis(200);

/// Outstanding obs whose round-trip is still being timed.
const PENDING_CAP: usize = 64;

/// Longest child line quoted back in an error message.
const QUOTE_CHARS: usize = 200;

// ===== wire types =====

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LivePolicyStartReq {
    /// Path to a python binary, or to a virtualenv directory (in which case
    /// `bin/python` inside it is used).
    python: String,
    /// Checkpoint the policy loads — passed through verbatim.
    ckpt: String,
    /// Torch device (`cpu`, `cuda`, `mps`); `None` lets the policy choose.
    device: Option<String>,
    /// Observation rate in Hz (default 20), decimated from the session's state
    /// emissions. Higher than the session's emit rate simply means "every
    /// emitted state" — the policy never sees a state the UI did not.
    hz: Option<f64>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LivePolicyStartedDto {
    /// Joint count the policy was trained for; must equal the session's.
    ndof: usize,
    /// The policy's RE-PLAN PERIOD (`n_action_steps`): how many ticks it
    /// reuses one forward pass for. DISPLAY METADATA ONLY — the bridge is one
    /// action per obs and the python side buffers chunks itself, so nothing
    /// here may build cadence logic on this number.
    chunk: usize,
    /// The lerobot policy type string, e.g. `"act"`, `"diffusion"`.
    policy_type: String,
    /// Device the policy actually landed on.
    device: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LivePolicyStatusDto {
    /// `"driving"` while actions are flowing, else `"stopped"`. A failure that
    /// ended the drive reads as `"stopped"` here; its detail went out on
    /// `live://policy`.
    state: String,
    /// Actions accepted and written into the hold target since start.
    ticks_driven: u64,
    /// Round-trip of the most recent matched obs→action pair (ms); 0 before the
    /// first one.
    last_latency_ms: f64,
}

/// `live://policy` — the bridge's whole lifecycle in one event.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LivePolicyEvent {
    session_id: u64,
    /// `"driving"` (once, right after `ready`), `"stopped"` or `"error"`.
    state: String,
    /// Policy type for `driving`; the failure (with a stderr tail) for `error`;
    /// why the drive ended, when it was not the human, for `stopped`.
    detail: Option<String>,
}

/// One line FROM the policy process.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FromPolicy {
    Ready {
        ndof: usize,
        chunk: usize,
        #[serde(rename = "policyType")]
        policy_type: String,
        device: String,
    },
    Action {
        tick: u64,
        q: Vec<f64>,
    },
    Error {
        message: String,
    },
}

/// One line TO the policy process. Field order is the protocol's.
#[derive(Serialize)]
struct ObsLine<'a> {
    #[serde(rename = "type")]
    ty: &'static str,
    tick: u64,
    t: f64,
    q: &'a [f64],
    qd: &'a [f64],
}

/// The handshake result, kept for the started DTO.
#[derive(Clone, Debug)]
struct Ready {
    ndof: usize,
    chunk: usize,
    policy_type: String,
    device: String,
}

// ===== emission seam (a second path, not an extension of `LiveEmitter`) =====

/// Where `live://policy` events go. Separate from `live::LiveEmitter` because
/// the session's emitter is MOVED into the session thread at `live_start` and
/// the policy outlives no session but starts long after one: the bridge needs
/// its own handle, and cloning is part of the contract (the command emits
/// `driving`, the driver thread emits the terminal event).
pub(crate) trait PolicyEmitter: Send + Clone + 'static {
    fn policy(&self, ev: &LivePolicyEvent);
}

#[derive(Clone)]
struct TauriPolicyEmitter(tauri::AppHandle);
impl PolicyEmitter for TauriPolicyEmitter {
    fn policy(&self, ev: &LivePolicyEvent) {
        let _ = tauri::Emitter::emit(&self.0, "live://policy", ev);
    }
}

// ===== the obs bus (session thread → driver thread) =====

#[derive(Default)]
struct ObsSnapshot {
    tick: u64,
    t: f64,
    q: Vec<f64>,
    qd: Vec<f64>,
}

/// The one-slot, latest-wins channel the session thread publishes into.
///
/// Lives in `LiveShared` so the session thread reaches it with no extra
/// plumbing, and costs a single relaxed atomic load per emitted state while no
/// policy is attached.
#[derive(Default)]
pub(crate) struct PolicyBus {
    active: AtomicBool,
    seq: AtomicU64,
    obs: Mutex<ObsSnapshot>,
}

impl PolicyBus {
    /// Called by the SESSION thread on every emitted state. Reuses the
    /// snapshot's allocations, so a driving session allocates nothing per tick.
    pub(crate) fn publish(&self, tick: u64, t: f64, q: &[f64], qd: &[f64]) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut s) = self.obs.lock() {
            s.tick = tick;
            s.t = t;
            s.q.clear();
            s.q.extend_from_slice(q);
            s.qd.clear();
            s.qd.extend_from_slice(qd);
        }
        self.seq.fetch_add(1, Ordering::Release);
    }

    fn set_active(&self, on: bool) {
        self.active.store(on, Ordering::Relaxed);
    }

    fn seq(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }

    /// A clone of the latest published state. `None` only on a poisoned lock.
    fn snapshot(&self) -> Option<(u64, f64, Vec<f64>, Vec<f64>)> {
        let s = self.obs.lock().ok()?;
        Some((s.tick, s.t, s.q.clone(), s.qd.clone()))
    }
}

// ===== shared bridge state =====

struct PolicyShared {
    /// Raised by `live_policy_stop`, by the reader on failure, and by session
    /// teardown. The driver's exit condition.
    stop: AtomicBool,
    /// First failure wins; `Some` turns the terminal event into an `error`.
    fail: Mutex<Option<String>>,
    /// True between `ready` and the driver's exit.
    driving: AtomicBool,
    ticks: AtomicU64,
    last_latency_us: AtomicU64,
    /// Obs awaiting their action, for the round-trip measurement.
    pending: Mutex<VecDeque<(u64, Instant)>>,
    stderr_tail: Mutex<VecDeque<u8>>,
    /// The child closed stderr — the tail is complete.
    stderr_done: AtomicBool,
    /// Handshake slot: `None` until the reader has seen the first line.
    ready: Mutex<Option<Result<Ready, String>>>,
    ready_cv: Condvar,
    /// Guarantees exactly one terminal `live://policy` event.
    ended: AtomicBool,
}

impl PolicyShared {
    fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            fail: Mutex::new(None),
            driving: AtomicBool::new(false),
            ticks: AtomicU64::new(0),
            last_latency_us: AtomicU64::new(0),
            pending: Mutex::new(VecDeque::new()),
            stderr_tail: Mutex::new(VecDeque::new()),
            stderr_done: AtomicBool::new(false),
            ready: Mutex::new(None),
            ready_cv: Condvar::new(),
            ended: AtomicBool::new(false),
        }
    }

    /// Record a failure and ask the driver to wind up. The detail is written
    /// BEFORE `stop`, so a driver that observes the stop always observes the
    /// reason with it.
    fn fail(&self, detail: String) {
        if let Ok(mut f) = self.fail.lock() {
            if f.is_none() {
                log::error!(target: "studio::policy", "{detail}");
                *f = Some(detail);
            }
        }
        self.stop.store(true, Ordering::Release);
    }

    fn fail_detail(&self) -> Option<String> {
        self.fail.lock().ok().and_then(|f| f.clone())
    }

    fn publish_ready(&self, r: Result<Ready, String>) {
        if let Ok(mut slot) = self.ready.lock() {
            *slot = Some(r);
        }
        self.ready_cv.notify_all();
    }

    /// Block until the reader has seen the child's first line, or the deadline
    /// passes. Holds no `AppState` lock (the caller drops them first) — a slow
    /// torch import must not freeze the rest of the backend.
    fn await_ready(&self, timeout: Duration) -> Result<Ready, String> {
        let deadline = Instant::now() + timeout;
        let mut slot = self.ready.lock().map_err(|_| "state lock poisoned")?;
        loop {
            if let Some(r) = slot.take() {
                return r;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!(
                    "the policy process did not report ready within {} s — is `caliper-learn` \
                     installed in that env?",
                    timeout.as_secs()
                ));
            }
            let (g, _) = self
                .ready_cv
                .wait_timeout(slot, left)
                .map_err(|_| "state lock poisoned")?;
            slot = g;
        }
    }

    fn push_pending(&self, tick: u64) {
        if let Ok(mut p) = self.pending.lock() {
            if p.len() == PENDING_CAP {
                p.pop_front();
            }
            p.push_back((tick, Instant::now()));
        }
    }

    /// Match an arriving action to the obs that asked for it, discarding the
    /// obs the policy skipped. Unmatched actions leave the latency untouched
    /// rather than inventing a number.
    fn note_latency(&self, tick: u64) {
        let Ok(mut p) = self.pending.lock() else {
            return;
        };
        while let Some((t, at)) = p.pop_front() {
            if t == tick {
                let us = at.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                self.last_latency_us.store(us, Ordering::Relaxed);
                return;
            }
            if t > tick {
                // An action for an obs we never sent: put it back, measure nothing.
                p.push_front((t, at));
                return;
            }
        }
    }

    fn push_stderr(&self, bytes: &[u8]) {
        let Ok(mut t) = self.stderr_tail.lock() else {
            return;
        };
        for b in bytes {
            if t.len() == STDERR_TAIL_BYTES {
                t.pop_front();
            }
            t.push_back(*b);
        }
    }

    /// The child's last words. Called only on failure paths, and only after the
    /// child has been asked to die — so it first gives the stderr pump a bounded
    /// moment to drain, rather than racing it and reporting an empty traceback.
    fn stderr_tail(&self) -> String {
        let deadline = Instant::now() + STDERR_SETTLE;
        while !self.stderr_done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let Ok(t) = self.stderr_tail.lock() else {
            return String::new();
        };
        let bytes: Vec<u8> = t.iter().copied().collect();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

// ===== the running bridge =====

/// A policy driving a live session. Held in `LiveSession.policy`, so EVERY way
/// a session can end — `live_stop`, a superseding `live_start`, a step error
/// reaped by the next command — drops this and takes the child with it. Dropping
/// it without a shutdown is impossible: [`Drop`] performs one.
pub(crate) struct PolicyHandle {
    ps: Arc<PolicyShared>,
    live: Arc<LiveShared>,
    /// OS pid of the policy process, for logs (and for proving in tests that
    /// nothing is orphaned).
    pid: u32,
    driver: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<()>>,
}

impl PolicyHandle {
    #[cfg(test)]
    fn pid(&self) -> u32 {
        self.pid
    }

    /// The drive is over — the child is winding up or already reaped, and this
    /// handle is nothing but a status record. Shutting one of these down joins
    /// threads that have already exited, so it costs nothing.
    fn is_over(&self) -> bool {
        self.ps.stop.load(Ordering::Acquire) || !self.ps.driving.load(Ordering::Relaxed)
    }

    fn status(&self) -> LivePolicyStatusDto {
        let driving = !self.is_over();
        LivePolicyStatusDto {
            state: if driving { "driving" } else { "stopped" }.to_string(),
            ticks_driven: self.ps.ticks.load(Ordering::Relaxed),
            last_latency_ms: self.ps.last_latency_us.load(Ordering::Relaxed) as f64 / 1000.0,
        }
    }

    /// Wind the bridge up: ask the driver to stop (it sends `{"type":"stop"}`,
    /// waits out the grace period and kills the child), then join all three
    /// threads. The reader unblocks when the child's stdout closes, which the
    /// driver guarantees — so joining the driver FIRST is what makes this
    /// terminate. Idempotent.
    pub(crate) fn shutdown(&mut self) {
        let running = self.driver.is_some();
        self.ps.stop.store(true, Ordering::Release);
        for j in [self.driver.take(), self.reader.take(), self.stderr.take()]
            .into_iter()
            .flatten()
        {
            let _ = j.join();
        }
        self.live.policy_bus.set_active(false);
        if running {
            log::info!(
                target: "studio::policy",
                "policy bridge shut down after {} actions (pid {} reaped)",
                self.ps.ticks.load(Ordering::Relaxed),
                self.pid
            );
        }
    }
}

impl Drop for PolicyHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ===== pure helpers =====

/// Emitted states per obs: the session emits at `emit_hz`, the policy wants
/// `obs_hz`, and obs are DECIMATED from emissions so a policy never sees a
/// state the UI did not. Always at least 1.
fn obs_decim(emit_hz: f64, obs_hz: f64) -> u64 {
    if !(emit_hz.is_finite() && obs_hz.is_finite()) || obs_hz <= 0.0 || emit_hz <= 0.0 {
        return 1;
    }
    ((emit_hz / obs_hz).round() as i64).max(1) as u64
}

/// Validate one action and make it safe to hold: exactly `ndof` finite values,
/// clamped into the URDF joint limits.
///
/// A non-finite action is an ERROR, not something to clamp — a NaN target would
/// silently poison the PD loop, and a policy emitting one is broken in a way the
/// human needs told.
fn sanitize_action(
    q: &[f64],
    ndof: usize,
    limits: &[Option<(f64, f64)>],
) -> Result<Vec<f64>, String> {
    if q.len() != ndof {
        return Err(format!(
            "the policy returned {} values but this robot has {ndof} joints",
            q.len()
        ));
    }
    if !q.iter().all(|x| x.is_finite()) {
        return Err("the policy returned a non-finite action".into());
    }
    let mut out = q.to_vec();
    for (i, lim) in limits.iter().enumerate().take(out.len()) {
        if let Some((lo, hi)) = lim {
            out[i] = out[i].clamp(*lo, *hi);
        }
    }
    Ok(out)
}

/// The python binary to spawn: the path as given, or `bin/python` inside a
/// virtualenv directory. Everything that can be checked before a fork is checked
/// here — a typo'd env must not surface as a mystery exit code.
fn resolve_python(spec: &str) -> Result<PathBuf, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(
            "give the path to a python binary, or to the virtualenv directory that has \
             caliper-learn installed"
                .into(),
        );
    }
    let p = Path::new(spec);
    let cand = if p.is_dir() {
        ["bin/python", "bin/python3", "Scripts/python.exe"]
            .iter()
            .map(|rel| p.join(rel))
            .find(|c| c.is_file())
            .ok_or_else(|| {
                format!(
                    "`{}` is a directory with no bin/python inside — point at a virtualenv \
                     (or at the python binary itself)",
                    p.display()
                )
            })?
    } else {
        p.to_path_buf()
    };
    if !cand.is_file() {
        return Err(format!(
            "`{}` does not exist — point at the python of the env that has caliper-learn \
             installed",
            cand.display()
        ));
    }
    if !is_executable(&cand) {
        return Err(format!("`{}` is not executable", cand.display()));
    }
    Ok(cand)
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(_p: &Path) -> bool {
    true
}

/// A child line, safe and short enough to put in an error message.
fn quote(line: &str) -> String {
    let s: String = line.chars().take(QUOTE_CHARS).collect();
    if s.chars().count() < line.chars().count() {
        format!("{s}…")
    } else {
        s
    }
}

/// Append the child's stderr tail to a failure, when it has anything to say.
fn with_stderr(detail: String, tail: &str) -> String {
    if tail.is_empty() {
        detail
    } else {
        format!("{detail}\n--- policy stderr ---\n{tail}")
    }
}

// ===== the three threads =====

/// Pump stderr into the ring buffer until the child closes it.
///
/// Continuously, and from before the handshake: a policy prints its whole
/// loading chatter (torch warnings, lerobot's own stdout redirected here) BEFORE
/// it announces itself, and an undrained stderr pipe would block the child at
/// ~64 KB — deadlocking the handshake against a child that cannot reach its
/// `ready` line. The last thing it writes is a session summary, which is why the
/// ring keeps the TAIL rather than the head.
fn spawn_stderr_pump(err: ChildStderr, ps: Arc<PolicyShared>) -> Result<JoinHandle<()>, String> {
    std::thread::Builder::new()
        .name("live-policy-err".into())
        .spawn(move || {
            let mut r = BufReader::new(err);
            let mut buf = [0u8; 1024];
            loop {
                match r.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => ps.push_stderr(&buf[..n]),
                }
            }
            ps.stderr_done.store(true, Ordering::Release);
        })
        .map_err(|e| format!("failed to spawn the policy stderr thread: {e}"))
}

/// Handshake, then the action loop. Owns the child's stdout for its whole life;
/// every accepted action lands in the session's PD hold target.
///
/// This thread exists so that stdout is drained CONTINUOUSLY and independently
/// of the obs the driver is writing. The policy loop on the other end is
/// synchronous — it answers each obs in order and flushes — so a bridge that
/// wrote several obs before reading would fill both pipe buffers and deadlock:
/// the child blocked writing an answer nobody is reading, us blocked writing an
/// obs it will never get to. Never merge these two loops.
///
/// The one place draining stops early is a protocol failure, where the reader
/// returns instead of consuming what follows. That cannot wedge anything: the
/// same failure raises `stop`, and the driver kills the child within
/// [`STOP_GRACE`] whether or not it is blocked on a write.
fn run_reader(
    stdout: ChildStdout,
    live: Arc<LiveShared>,
    ps: Arc<PolicyShared>,
    ndof: usize,
    limits: Arc<Vec<Option<(f64, f64)>>>,
) {
    let mut lines = BufReader::new(stdout).lines();

    let ready = match lines.next() {
        None => Err("the policy process exited before it reported ready".to_string()),
        Some(Err(e)) => Err(format!("reading from the policy process failed: {e}")),
        Some(Ok(l)) => match serde_json::from_str::<FromPolicy>(&l) {
            Ok(FromPolicy::Ready {
                ndof,
                chunk,
                policy_type,
                device,
            }) => Ok(Ready {
                ndof,
                chunk,
                policy_type,
                device,
            }),
            Ok(FromPolicy::Error { message }) => Err(message),
            Ok(other) => Err(format!(
                "the policy process sent {other:?} before it reported ready"
            )),
            Err(e) => Err(format!(
                "protocol error: `{}` is not a policy message ({e})",
                quote(&l)
            )),
        },
    };
    let ok = ready.is_ok();
    ps.publish_ready(ready);
    if !ok {
        return;
    }

    for line in lines {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                if !ps.stop.load(Ordering::Acquire) {
                    ps.fail(format!("reading from the policy process failed: {e}"));
                }
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<FromPolicy>(&line) {
            Ok(FromPolicy::Action { tick, q }) => match sanitize_action(&q, ndof, &limits) {
                Ok(q) => {
                    live.write_target(q);
                    ps.ticks.fetch_add(1, Ordering::Relaxed);
                    ps.note_latency(tick);
                }
                Err(e) => {
                    ps.fail(e);
                    return;
                }
            },
            Ok(FromPolicy::Error { message }) => {
                ps.fail(format!("the policy reported an error: {message}"));
                return;
            }
            Ok(other) => {
                ps.fail(format!(
                    "protocol error: the policy sent {other:?} mid-drive"
                ));
                return;
            }
            Err(e) => {
                ps.fail(format!(
                    "protocol error: `{}` is not a policy message ({e})",
                    quote(&line)
                ));
                return;
            }
        }
    }
    // EOF. Expected once we asked it to stop; a surprise otherwise.
    if !ps.stop.load(Ordering::Acquire) {
        ps.fail("the policy process exited".to_string());
    }
}

/// Pace obs out to the child, then own the whole teardown: the stop line, the
/// grace period, the kill, the reap, and the one terminal `live://policy` event.
///
/// Teardown lives HERE rather than in [`PolicyHandle::shutdown`] so a child can
/// never be orphaned by a handle nobody drops: whatever ends the drive — the
/// human, the reader, the session — the driver runs this exactly once.
fn run_driver<E: PolicyEmitter>(
    child: Arc<Mutex<Option<Child>>>,
    mut stdin: ChildStdin,
    session_id: u64,
    live: Arc<LiveShared>,
    ps: Arc<PolicyShared>,
    decim: u64,
    emitter: E,
) {
    let mut last_seq = live.policy_bus.seq();
    let stop_detail: Option<String> = loop {
        if ps.stop.load(Ordering::Acquire) {
            break None; // the human, or a failure the reader already recorded
        }
        if live.is_gone() {
            break Some("the live session ended".into());
        }
        // A paused sim shows the policy no passage of time: nothing is sent, and
        // the sequence is re-anchored so resuming does not deliver a burst of
        // stale states.
        if live.is_paused() {
            last_seq = live.policy_bus.seq();
            std::thread::sleep(DRIVER_POLL);
            continue;
        }
        let seq = live.policy_bus.seq();
        if seq.wrapping_sub(last_seq) < decim {
            std::thread::sleep(DRIVER_POLL);
            continue;
        }
        last_seq = seq;
        let Some((tick, t, q, qd)) = live.policy_bus.snapshot() else {
            break Some("the live session's state lock was poisoned".into());
        };
        let line = match serde_json::to_string(&ObsLine {
            ty: "obs",
            tick,
            t,
            q: &q,
            qd: &qd,
        }) {
            Ok(l) => l,
            Err(e) => {
                ps.fail(format!("could not encode an observation: {e}"));
                break None;
            }
        };
        ps.push_pending(tick);
        if let Err(e) = stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
        {
            ps.fail(format!("writing to the policy process failed: {e}"));
            break None;
        }
    };

    ps.driving.store(false, Ordering::Relaxed);
    let failed = ps.fail_detail();
    let ev = match &failed {
        Some(d) => LivePolicyEvent {
            session_id,
            state: "error".into(),
            detail: Some(with_stderr(d.clone(), &ps.stderr_tail())),
        },
        None => LivePolicyEvent {
            session_id,
            state: "stopped".into(),
            detail: stop_detail,
        },
    };
    if !ps.ended.swap(true, Ordering::AcqRel) {
        emitter.policy(&ev);
    }

    // Ask nicely, then insist. Dropping stdin is itself a stop signal (the
    // child's own stdin hits EOF), so a policy that ignores the message still
    // winds up unless it is truly wedged.
    let _ = stdin.write_all(b"{\"type\":\"stop\"}\n");
    let _ = stdin.flush();
    drop(stdin);
    let Some(mut child) = child.lock().ok().and_then(|mut c| c.take()) else {
        log::error!(target: "studio::policy", "the policy child was already taken — cannot reap it");
        return;
    };
    let deadline = Instant::now() + STOP_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            _ => {
                log::warn!(
                    target: "studio::policy",
                    "the policy process did not exit within {} s — killing it",
                    STOP_GRACE.as_secs()
                );
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }
    }
    // The child's last line is its own session summary (queries, clamped ticks).
    // It is not error-surface material on a clean stop, but throwing it away
    // would lose the only account of what the policy actually did.
    if failed.is_none() {
        if let Some(summary) = ps.stderr_tail().lines().last() {
            log::info!(target: "studio::policy", "policy session summary: {summary}");
        }
    }
}

// ===== command impls (take &AppState so tests drive them without tauri) =====

/// Spawn the policy child, drive the handshake, and attach it to the running
/// session.
///
/// Everything cheap fails first and synchronously — no session, a policy already
/// attached, no robot path, an unusable python. The handshake then runs with NO
/// `AppState` lock held: a 30 s torch import must not freeze `live_set_target`.
/// The session is re-checked (by id) before the handle is installed, so a
/// session that ended while the model loaded leaves no stray child.
pub(crate) fn live_policy_start_on<E: PolicyEmitter>(
    state: &AppState,
    req: LivePolicyStartReq,
    emitter: E,
) -> Result<LivePolicyStartedDto, String> {
    let (live, session_id, ndof, emit_hz, attached, spent) = with_live(state, |s| {
        let mut p = s.policy.lock().map_err(|_| "state lock poisoned")?;
        // A bridge that already ended (the human stopped it, or it failed) is
        // a status record, not an obstacle: reap it so a fixed checkpoint can be
        // tried again without a stop round-trip first.
        let spent = match p.as_ref() {
            Some(h) if h.is_over() => p.take(),
            _ => None,
        };
        let attached = p.is_some();
        Ok((s.shared.clone(), s.id, s.ndof, s.emit_hz, attached, spent))
    })?;
    drop(spent);
    if attached {
        return Err("a policy is already driving this session — stop it first".into());
    }
    let obs_hz = match req.hz {
        None => DEFAULT_OBS_HZ,
        Some(hz) if hz.is_finite() && hz > 0.0 => hz,
        Some(_) => return Err("hz must be finite and positive".into()),
    };
    let ckpt = req.ckpt.trim().to_string();
    if ckpt.is_empty() {
        return Err("give the checkpoint the policy should load".into());
    }
    let python = resolve_python(&req.python)?;

    let (urdf, limits) = {
        let guard = state.model.lock().map_err(|_| "state lock poisoned")?;
        let model = guard.as_ref().ok_or("no robot loaded")?;
        if model.ndof != ndof {
            return Err(
                "the loaded robot changed — restart the live session before driving it".into(),
            );
        }
        let urdf = state
            .robot_path
            .lock()
            .map_err(|_| "state lock poisoned")?
            .clone()
            .ok_or(
                "this robot was not loaded from a file — the policy needs its URDF path \
                 (open the robot or the task again)",
            )?;
        (urdf, Arc::new(model.limits.clone()))
    };

    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("caliper_learn.cli")
        .arg("drive")
        .arg(&ckpt)
        .arg("--urdf")
        .arg(&urdf);
    if let Some(dev) = req
        .device
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
    {
        cmd.arg("--device").arg(dev);
    }
    log::info!(
        target: "studio::policy",
        "live_policy_start: {} -m caliper_learn.cli drive {ckpt} --urdf {urdf} (obs {obs_hz} Hz)",
        python.display()
    );
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run `{}`: {e}", python.display()))?;

    // From here on every failure must reap the child before returning.
    let pid = child.id();
    let ps = Arc::new(PolicyShared::new());
    let kill = |child: &mut Child, e: String| -> String {
        let _ = child.kill();
        let _ = child.wait();
        with_stderr(e, &ps.stderr_tail())
    };
    let (stdin, stdout, stderr) =
        match (child.stdin.take(), child.stdout.take(), child.stderr.take()) {
            (Some(i), Some(o), Some(e)) => (i, o, e),
            _ => return Err(kill(&mut child, "the policy process has no pipes".into())),
        };
    let err_thread = match spawn_stderr_pump(stderr, ps.clone()) {
        Ok(t) => t,
        Err(e) => return Err(kill(&mut child, e)),
    };
    let reader = {
        let (live, ps, limits) = (live.clone(), ps.clone(), limits.clone());
        match std::thread::Builder::new()
            .name(format!("live-policy-rx-{session_id}"))
            .spawn(move || run_reader(stdout, live, ps, ndof, limits))
        {
            Ok(t) => t,
            Err(e) => {
                return Err(kill(
                    &mut child,
                    format!("failed to spawn the policy reader thread: {e}"),
                ))
            }
        }
    };

    let ready = match ps.await_ready(READY_TIMEOUT) {
        Ok(r) => r,
        Err(e) => {
            let e = kill(&mut child, e);
            let _ = reader.join();
            let _ = err_thread.join();
            return Err(e);
        }
    };
    if ready.ndof != ndof {
        let e = kill(
            &mut child,
            format!(
                "the policy drives {} joints but this session's robot has {ndof} — load the \
                 robot the policy was trained on",
                ready.ndof
            ),
        );
        let _ = reader.join();
        let _ = err_thread.join();
        return Err(e);
    }

    // The bus goes live BEFORE the driver, so the first emitted state after this
    // point is already published for it. The child moves into a shared slot
    // rather than into the closure directly: a thread that fails to spawn drops
    // its closure, and a dropped `Child` is NOT a killed one.
    let child = Arc::new(Mutex::new(Some(child)));
    live.policy_bus.set_active(true);
    ps.driving.store(true, Ordering::Relaxed);
    let spawned = {
        let (live, ps, emitter, child) = (live.clone(), ps.clone(), emitter.clone(), child.clone());
        let decim = obs_decim(emit_hz, obs_hz);
        std::thread::Builder::new()
            .name(format!("live-policy-tx-{session_id}"))
            .spawn(move || run_driver(child, stdin, session_id, live, ps, decim, emitter))
    };
    let driver = match spawned {
        Ok(t) => t,
        Err(e) => {
            live.policy_bus.set_active(false);
            ps.driving.store(false, Ordering::Relaxed);
            ps.stop.store(true, Ordering::Release);
            if let Some(mut c) = child.lock().ok().and_then(|mut s| s.take()) {
                let _ = c.kill();
                let _ = c.wait();
            }
            let _ = reader.join();
            let _ = err_thread.join();
            return Err(format!("failed to spawn the policy driver thread: {e}"));
        }
    };

    let handle = PolicyHandle {
        ps,
        live: live.clone(),
        pid,
        driver: Some(driver),
        reader: Some(reader),
        stderr: Some(err_thread),
    };

    // Install under the session lock, verifying the session we validated against
    // is still the one running. A failure here DROPS the handle, which shuts the
    // child down — a lost race never leaks a python.
    {
        let slot = state.live.lock().map_err(|_| "state lock poisoned")?;
        match slot.as_ref() {
            Some(s) if s.id == session_id && !s.is_dead() => {
                let mut p = s.policy.lock().map_err(|_| "state lock poisoned")?;
                if p.is_some() {
                    drop(p);
                    drop(slot);
                    return Err("a policy is already driving this session — stop it first".into());
                }
                *p = Some(handle);
            }
            _ => {
                drop(slot);
                return Err("the live session ended while the policy was loading".into());
            }
        }
    }

    emitter.policy(&LivePolicyEvent {
        session_id,
        state: "driving".into(),
        detail: Some(ready.policy_type.clone()),
    });
    log::info!(
        target: "studio::policy",
        "policy `{}` (pid {pid}) is driving session {session_id} on {} ({} dof, chunk {})",
        ready.policy_type, ready.device, ready.ndof, ready.chunk
    );
    Ok(LivePolicyStartedDto {
        ndof: ready.ndof,
        chunk: ready.chunk,
        policy_type: ready.policy_type,
        device: ready.device,
    })
}

/// Stop the policy (idempotent — no session or no policy is a no-op).
///
/// The handle is taken OUT of the session under the lock and shut down after it
/// is dropped: the teardown waits out a grace period, and holding `state.live`
/// through that would stall every other live command.
pub(crate) fn live_policy_stop_impl(state: &AppState) -> Result<(), String> {
    let handle = {
        let slot = state.live.lock().map_err(|_| "state lock poisoned")?;
        match slot.as_ref() {
            Some(s) => s.policy.lock().map_err(|_| "state lock poisoned")?.take(),
            None => None,
        }
    };
    drop(handle); // shuts the child down outside every lock
    Ok(())
}

/// Bridge status, or null when no policy is attached (including with no live
/// session).
pub(crate) fn live_policy_status_impl(
    state: &AppState,
) -> Result<Option<LivePolicyStatusDto>, String> {
    let slot = state.live.lock().map_err(|_| "state lock poisoned")?;
    match slot.as_ref() {
        Some(s) => Ok(s
            .policy
            .lock()
            .map_err(|_| "state lock poisoned")?
            .as_ref()
            .map(PolicyHandle::status)),
        None => Ok(None),
    }
}

// ===== tauri commands =====

/// Spawn a policy in the given python env and let it drive the live session.
#[tauri::command]
pub fn live_policy_start(
    req: LivePolicyStartReq,
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<LivePolicyStartedDto, String> {
    logged(
        "live_policy_start",
        live_policy_start_on(&state, req, TauriPolicyEmitter(app)),
    )
}

/// Stop the policy; the sim keeps running under the last target (idempotent).
#[tauri::command]
pub fn live_policy_stop(state: tauri::State<'_, AppState>) -> Result<(), String> {
    logged("live_policy_stop", live_policy_stop_impl(&state))
}

/// Policy bridge status, or null when nothing is attached.
#[tauri::command]
pub fn live_policy_status(
    state: tauri::State<'_, AppState>,
) -> Result<Option<LivePolicyStatusDto>, String> {
    live_policy_status_impl(&state)
}

// ===== tests =====

#[cfg(test)]
mod tests {
    use super::*;

    // -- pure helpers (no child, no session) --

    #[test]
    fn decimation_maps_the_emit_rate_onto_the_policy_rate() {
        assert_eq!(obs_decim(58.8, 20.0), 3); // the default session/default policy
        assert_eq!(obs_decim(60.0, 60.0), 1);
        assert_eq!(obs_decim(60.0, 10.0), 6);
        // asking for more than the session emits means "every emitted state"
        assert_eq!(obs_decim(60.0, 500.0), 1);
        assert_eq!(obs_decim(f64::NAN, 20.0), 1);
        assert_eq!(obs_decim(60.0, 0.0), 1);
    }

    #[test]
    fn actions_are_clamped_but_never_faked() {
        let lim = vec![Some((-1.0, 1.0)), None];
        assert_eq!(
            sanitize_action(&[2.0, 9.0], 2, &lim).unwrap(),
            vec![1.0, 9.0],
            "limited joints clamp, unlimited ones pass through"
        );
        let err = sanitize_action(&[0.0], 2, &lim).unwrap_err();
        assert!(
            err.contains("1 values") && err.contains("2 joints"),
            "{err}"
        );
        let err = sanitize_action(&[f64::NAN, 0.0], 2, &lim).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
        let err = sanitize_action(&[f64::INFINITY, 0.0], 2, &lim).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
    }

    #[test]
    fn a_bad_python_path_is_refused_before_any_spawn() {
        let err = resolve_python("").unwrap_err();
        assert!(err.contains("virtualenv"), "{err}");
        let err = resolve_python("/nope/nothing/here/python").unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        // a directory that is not a virtualenv
        let err = resolve_python(&std::env::temp_dir().display().to_string()).unwrap_err();
        assert!(err.contains("no bin/python"), "{err}");
        // something real and executable resolves
        assert_eq!(resolve_python("/bin/sh").unwrap(), PathBuf::from("/bin/sh"));
    }

    #[cfg(unix)]
    #[test]
    fn a_non_executable_python_is_refused() {
        let dir = tmp("noexec");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("python");
        std::fs::write(&p, b"#!/bin/sh\n").unwrap();
        let err = resolve_python(&p.display().to_string()).unwrap_err();
        assert!(err.contains("not executable"), "{err}");
        // ...and a venv-shaped directory finds bin/python
        let venv = dir.join("venv");
        std::fs::create_dir_all(venv.join("bin")).unwrap();
        let py = venv.join("bin/python");
        std::fs::write(&py, b"#!/bin/sh\n").unwrap();
        chmod_x(&py);
        assert_eq!(resolve_python(&venv.display().to_string()).unwrap(), py);
    }

    #[test]
    fn quoting_a_child_line_is_bounded() {
        assert_eq!(quote("hi"), "hi");
        let long = "x".repeat(500);
        let q = quote(&long);
        assert_eq!(
            q.chars().count(),
            QUOTE_CHARS + 1,
            "truncated with an ellipsis"
        );
    }

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "studio_policy_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[cfg(unix)]
    fn chmod_x(p: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(p).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(p, perm).unwrap();
    }

    // -- a fake policy child (no python anywhere in this lane) --

    #[cfg(unix)]
    mod fake {
        use super::*;
        use crate::live::{LiveEmitter, LiveEndedEvent, LiveStateEvent};
        use caliper::model::Model;

        /// The fake speaks the protocol from `awk`, wrapped in a `/bin/sh`
        /// script that ignores the `-m caliper_learn.cli drive …` arguments.
        /// awk (not the shell) because `fflush()` is the only portable way to
        /// guarantee a line reaches our pipe the moment it is printed — a
        /// shell's own buffering would deadlock the handshake.
        fn fake_python(tag: &str, awk: &str) -> PathBuf {
            let dir = tmp(tag);
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("python");
            std::fs::write(&p, format!("#!/bin/sh\nexec awk '{awk}'\n")).unwrap();
            chmod_x(&p);
            p
        }

        /// `ready`, then one fixed action per obs, echoing the obs tick back.
        const GOOD: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":8,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"/"type":"stop"/ { exit 0 }"#,
            "\n",
            r#"{ t=$0; sub(/.*"tick":/,"",t); sub(/[^0-9].*/,"",t); printf "{\"type\":\"action\",\"tick\":%s,\"q\":[0.4,-0.3]}\n", t; fflush() }"#,
        );

        /// Actions outside the ±3.14 limits of `dyn_pendulum2`.
        const OUT_OF_LIMITS: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":1,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"/"type":"stop"/ { exit 0 }"#,
            "\n",
            r#"{ printf "{\"type\":\"action\",\"tick\":0,\"q\":[99,-99]}\n"; fflush() }"#,
        );

        /// Ready for a 3-dof robot — refused against a 2-dof session.
        const WRONG_NDOF_READY: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":3,\"chunk\":1,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"{ }"#,
        );

        /// Ready for 2 dof, then actions of the wrong width.
        const WRONG_NDOF_ACTION: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":1,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"{ printf "{\"type\":\"action\",\"tick\":0,\"q\":[0.1,0.2,0.3]}\n"; fflush() }"#,
        );

        /// Ready, then dies on the first obs with something on stderr.
        const DIES: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":1,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"{ print "Traceback: policy blew up" > "/dev/stderr"; exit 3 }"#,
        );

        /// A realistic loader: ~180 KB of chatter on stderr BEFORE the ready
        /// line, then a normal drive. Far past the ~64 KB pipe buffer, so a
        /// bridge that did not drain stderr concurrently would leave this child
        /// blocked mid-chatter, unable to ever announce itself.
        const FLOODS_STDERR: &str = concat!(
            r#"BEGIN { for (i = 0; i < 4000; i++) print "loading shard weights, please wait: " i > "/dev/stderr";"#,
            "\n",
            r#" printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":8,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"/"type":"stop"/ { exit 0 }"#,
            "\n",
            r#"{ t=$0; sub(/.*"tick":/,"",t); sub(/[^0-9].*/,"",t); printf "{\"type\":\"action\",\"tick\":%s,\"q\":[0.4,-0.3]}\n", t; fflush() }"#,
        );

        /// Ready, then a line that is not the protocol at all.
        const GARBAGE: &str = concat!(
            r#"BEGIN { printf "{\"type\":\"ready\",\"ndof\":2,\"chunk\":1,\"policyType\":\"act\",\"device\":\"cpu\"}\n"; fflush() }"#,
            "\n",
            r#"{ printf "loading checkpoint...\n"; fflush() }"#,
        );

        /// Refuses to start, the way a missing checkpoint would.
        const REFUSES: &str = r#"BEGIN { printf "{\"type\":\"error\",\"message\":\"no such checkpoint\"}\n"; fflush(); exit 1 }"#;

        /// Chatters on stdout instead of announcing itself, then hangs — the
        /// handshake must reject the first line rather than wait it out.
        /// (`READY_TIMEOUT` itself is 60 s and not worth a test.)
        const BABBLES: &str = concat!(
            r#"BEGIN { printf "hello from torch\n"; fflush() }"#,
            "\n",
            r#"{ }"#,
        );

        // -- session plumbing --

        #[derive(Clone, Default)]
        struct Collect {
            states: Arc<Mutex<Vec<LiveStateEvent>>>,
            policy: Arc<Mutex<Vec<LivePolicyEvent>>>,
        }
        impl LiveEmitter for Collect {
            fn state(&self, ev: &LiveStateEvent) {
                self.states.lock().unwrap().push(ev.clone());
            }
            fn ended(&self, _ev: &LiveEndedEvent) {}
        }
        impl PolicyEmitter for Collect {
            fn policy(&self, ev: &LivePolicyEvent) {
                self.policy.lock().unwrap().push(ev.clone());
            }
        }
        impl Collect {
            fn last_target(&self) -> Vec<f64> {
                self.states
                    .lock()
                    .unwrap()
                    .last()
                    .expect("the session emitted at least one state")
                    .target()
                    .to_vec()
            }
            fn policy_events(&self) -> Vec<LivePolicyEvent> {
                self.policy.lock().unwrap().clone()
            }
            /// The terminal event, once the drive has wound up.
            fn terminal(&self) -> LivePolicyEvent {
                self.policy_events()
                    .into_iter()
                    .find(|e| e.state != "driving")
                    .expect("a terminal policy event")
            }
        }

        fn pendulum_state() -> AppState {
            let path = PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../oracle/fixtures/robots/dyn_pendulum2.urdf"
            ));
            let state = AppState::default();
            *state.model.lock().unwrap() = Some(Model::from_urdf(&path).expect("fixture loads"));
            *state.robot_path.lock().unwrap() = Some(path.display().to_string());
            state
        }

        fn start_session(state: &AppState) -> Collect {
            let em = Collect::default();
            crate::live::test_start_builtin(state, vec![0.0, 0.0], 400.0, 40.0, em.clone())
                .expect("live_start");
            em
        }

        fn req(python: &Path, hz: Option<f64>) -> LivePolicyStartReq {
            LivePolicyStartReq {
                python: python.display().to_string(),
                ckpt: "/tmp/fake.ckpt".into(),
                device: Some("cpu".into()),
                hz,
            }
        }

        fn sleep_ms(ms: u64) {
            std::thread::sleep(Duration::from_millis(ms));
        }

        /// Wait until `f` holds, or give up after `ms`. Returns whether it held —
        /// the fake child's scheduling is the only thing being waited on.
        fn until(ms: u64, mut f: impl FnMut() -> bool) -> bool {
            let deadline = Instant::now() + Duration::from_millis(ms);
            while Instant::now() < deadline {
                if f() {
                    return true;
                }
                sleep_ms(5);
            }
            f()
        }

        /// Is `pid` still a live (or unreaped) process? `kill -0` succeeds on a
        /// zombie too, so a false here proves the child was WAITED for, not just
        /// signalled.
        fn alive(pid: u32) -> bool {
            std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(format!("kill -0 {pid}"))
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }

        /// The pid of the policy attached to the session, straight from the
        /// handle that owns it.
        fn child_pid(state: &AppState) -> Option<u32> {
            let slot = state.live.lock().unwrap();
            let s = slot.as_ref()?;
            let p = s.policy.lock().unwrap();
            p.as_ref().map(PolicyHandle::pid)
        }

        // -- the drive --

        #[test]
        fn a_policy_drives_the_hold_target() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("good", GOOD);

            let dto = live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            assert_eq!(dto.ndof, 2);
            assert_eq!(dto.chunk, 8);
            assert_eq!(dto.policy_type, "act");
            assert_eq!(dto.device, "cpu");
            assert_eq!(em.policy_events().len(), 1);
            assert_eq!(em.policy_events()[0].state, "driving");
            assert_eq!(em.policy_events()[0].detail.as_deref(), Some("act"));

            // The policy's action becomes THE hold target every input shares.
            assert!(
                until(3_000, || em.last_target() == vec![0.4, -0.3]),
                "the target never reached the policy's action (got {:?})",
                em.last_target()
            );
            let st = live_policy_status_impl(&state).unwrap().expect("attached");
            assert_eq!(st.state, "driving");
            assert!(st.ticks_driven > 0, "no actions were applied");
            assert!(
                st.last_latency_ms >= 0.0 && st.last_latency_ms < 5_000.0,
                "implausible latency {}",
                st.last_latency_ms
            );

            // A human nudge wins for the instant, then the policy flies on.
            crate::live::live_set_target_impl(&state, &[1.0, 1.0]).unwrap();
            assert!(
                until(3_000, || em.last_target() == vec![0.4, -0.3]),
                "the policy did not resume driving after a manual override"
            );

            live_policy_stop_impl(&state).unwrap();
            let ev = em.terminal();
            assert_eq!(ev.state, "stopped");
            assert!(ev.detail.is_none(), "a human stop needs no explanation");
            assert!(live_policy_status_impl(&state).unwrap().is_none());
            // idempotent
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn actions_are_clamped_into_the_joint_limits() {
            let state = pendulum_state();
            let em = start_session(&state);
            // The fixture's own limits, so this test cannot drift from the URDF.
            let (lo, hi) = state.model.lock().unwrap().as_ref().unwrap().limits[0]
                .expect("dyn_pendulum2's joints are limited");
            let py = fake_python("clamp", OUT_OF_LIMITS);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            assert!(
                until(3_000, || {
                    let t = em.last_target();
                    (t[0] - hi).abs() < 1e-12 && (t[1] - lo).abs() < 1e-12
                }),
                "target {:?} was not clamped into [{lo}, {hi}]",
                em.last_target()
            );
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn a_paused_session_shows_the_policy_no_time() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("paused", GOOD);
            live_policy_start_on(&state, req(&py, Some(50.0)), em.clone()).expect("start");
            let ticks = || {
                live_policy_status_impl(&state)
                    .unwrap()
                    .unwrap()
                    .ticks_driven
            };
            assert!(until(3_000, || ticks() > 0), "the policy never got an obs");

            crate::live::live_pause_impl(&state, true).unwrap();
            sleep_ms(120); // let anything in flight land
            let frozen = ticks();
            sleep_ms(250);
            assert_eq!(frozen, ticks(), "obs kept flowing while paused");

            crate::live::live_pause_impl(&state, false).unwrap();
            assert!(
                until(3_000, || ticks() > frozen),
                "the policy did not resume after unpause"
            );
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        // -- failures --

        #[test]
        fn an_action_of_the_wrong_width_ends_the_drive_and_spares_the_sim() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("badlen", WRONG_NDOF_ACTION);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");

            assert!(
                until(3_000, || em.policy_events().len() > 1),
                "no terminal event"
            );
            let ev = em.terminal();
            assert_eq!(ev.state, "error");
            let d = ev.detail.unwrap_or_default();
            assert!(d.contains("3 values") && d.contains("2 joints"), "{d}");
            // the session is untouched and still stepping
            let before = crate::live::live_status_impl(&state)
                .unwrap()
                .expect("alive");
            assert!(until(2_000, || {
                crate::live::live_status_impl(&state)
                    .unwrap()
                    .is_some_and(|s| s.tick() > before.tick())
            }));
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn a_child_that_dies_mid_drive_surfaces_its_stderr() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("dies", DIES);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");

            assert!(
                until(3_000, || em.policy_events().len() > 1),
                "no terminal event"
            );
            let ev = em.terminal();
            assert_eq!(ev.state, "error");
            let d = ev.detail.unwrap_or_default();
            assert!(d.contains("exited"), "{d}");
            assert!(d.contains("policy blew up"), "stderr tail missing: {d}");
            assert!(crate::live::live_status_impl(&state).unwrap().is_some());
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn unparseable_stdout_is_a_protocol_error() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("garbage", GARBAGE);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            assert!(
                until(3_000, || em.policy_events().len() > 1),
                "no terminal event"
            );
            let ev = em.terminal();
            assert_eq!(ev.state, "error");
            let d = ev.detail.unwrap_or_default();
            assert!(d.contains("protocol error"), "{d}");
            assert!(d.contains("loading checkpoint"), "{d}");
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn a_refused_start_leaves_nothing_behind() {
            let state = pendulum_state();
            let em = start_session(&state);

            for (tag, awk, want) in [
                ("refuses", REFUSES, "no such checkpoint"),
                ("babbles", BABBLES, "protocol error"),
                ("ndof", WRONG_NDOF_READY, "drives 3 joints"),
            ] {
                let py = fake_python(tag, awk);
                let err = live_policy_start_on(&state, req(&py, None), em.clone())
                    .map(|_| ())
                    .unwrap_err();
                assert!(err.contains(want), "for {tag}: got {err}");
                assert!(
                    live_policy_status_impl(&state).unwrap().is_none(),
                    "{tag} left a policy attached"
                );
                assert!(em.policy_events().is_empty(), "{tag} emitted an event");
            }
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn starting_without_a_session_or_a_python_fails() {
            let state = pendulum_state();
            let em = Collect::default();
            let py = fake_python("nosession", GOOD);
            let err = live_policy_start_on(&state, req(&py, None), em.clone())
                .map(|_| ())
                .unwrap_err();
            assert!(err.contains("no live session"), "{err}");

            start_session(&state);
            let err = live_policy_start_on(
                &state,
                LivePolicyStartReq {
                    python: "/nope/python".into(),
                    ckpt: "x".into(),
                    device: None,
                    hz: None,
                },
                em.clone(),
            )
            .map(|_| ())
            .unwrap_err();
            assert!(err.contains("does not exist"), "{err}");

            let err = live_policy_start_on(
                &state,
                LivePolicyStartReq {
                    python: py.display().to_string(),
                    ckpt: "  ".into(),
                    device: None,
                    hz: None,
                },
                em.clone(),
            )
            .map(|_| ())
            .unwrap_err();
            assert!(err.contains("checkpoint"), "{err}");

            // status/stop are safe with no policy attached
            assert!(live_policy_status_impl(&state).unwrap().is_none());
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        /// The pipe-deadlock guard. The python loop is synchronous — it answers
        /// each obs and flushes — and it prints its loading chatter to stderr
        /// before it can announce itself. If either pipe were drained only when
        /// convenient (or merged into the obs-writing loop), a child like this
        /// would block on a full buffer and the handshake would hang until the
        /// 60 s timeout instead of taking milliseconds.
        #[test]
        fn a_chatty_loader_cannot_deadlock_the_handshake() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("flood", FLOODS_STDERR);

            let t0 = Instant::now();
            let dto = live_policy_start_on(&state, req(&py, None), em.clone())
                .expect("a chatty loader still starts");
            let took = t0.elapsed();
            assert_eq!(dto.policy_type, "act");
            assert!(
                took < Duration::from_secs(20),
                "the handshake took {took:?} — stderr is not being drained concurrently"
            );
            // ...and the drive itself is unaffected by the backlog.
            assert!(
                until(3_000, || em.last_target() == vec![0.4, -0.3]),
                "the flooded child never drove the target"
            );
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        /// A failure must not wedge the session: the human fixes the checkpoint
        /// and starts again, with no stop round-trip in between.
        #[test]
        fn a_failed_drive_can_be_restarted() {
            let state = pendulum_state();
            let em = start_session(&state);
            live_policy_start_on(
                &state,
                req(&fake_python("broken", GARBAGE), None),
                em.clone(),
            )
            .expect("start");
            assert!(
                until(3_000, || em.policy_events().len() > 1),
                "the broken policy never failed"
            );
            assert_eq!(
                live_policy_status_impl(&state).unwrap().unwrap().state,
                "stopped",
                "a failed bridge reads as stopped"
            );

            live_policy_start_on(&state, req(&fake_python("fixed", GOOD), None), em.clone())
                .expect("the fixed policy starts over the spent one");
            assert!(
                until(3_000, || em.last_target() == vec![0.4, -0.3]),
                "the replacement policy never drove"
            );
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn a_second_policy_is_refused_while_one_drives() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("first", GOOD);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            let err = live_policy_start_on(&state, req(&py, None), em.clone())
                .map(|_| ())
                .unwrap_err();
            assert!(err.contains("already driving"), "{err}");
            live_policy_stop_impl(&state).unwrap();
            crate::live::live_stop_impl(&state).unwrap();
        }

        // -- no orphans --

        #[test]
        fn stopping_reaps_the_child() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("reap", GOOD);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            let pid = child_pid(&state).expect("a running child");
            assert!(alive(pid));

            live_policy_stop_impl(&state).unwrap();
            assert!(!alive(pid), "the policy child outlived live_policy_stop");
            crate::live::live_stop_impl(&state).unwrap();
        }

        #[test]
        fn ending_the_session_takes_the_policy_with_it() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("sessionend", GOOD);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            let pid = child_pid(&state).expect("a running child");

            crate::live::live_stop_impl(&state).unwrap();
            assert!(!alive(pid), "the policy child outlived its session");
            assert_eq!(em.terminal().state, "stopped");
            assert!(live_policy_status_impl(&state).unwrap().is_none());
        }

        #[test]
        fn superseding_the_session_takes_the_policy_with_it() {
            let state = pendulum_state();
            let em = start_session(&state);
            let py = fake_python("supersede", GOOD);
            live_policy_start_on(&state, req(&py, None), em.clone()).expect("start");
            let pid = child_pid(&state).expect("a running child");

            let _em2 = start_session(&state); // a superseding live_start
            assert!(
                !alive(pid),
                "the policy child outlived the superseded session"
            );
            assert!(live_policy_status_impl(&state).unwrap().is_none());
            crate::live::live_stop_impl(&state).unwrap();
        }
    }
}
