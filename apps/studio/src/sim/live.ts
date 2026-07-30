// ============================================================
// sim/live.ts — pure live-sim-session helpers. PURE and headless:
// no React, no Tauri, no store (types only flow the other way),
// so the vitest suite drives every function directly. The store's
// event handlers / rAF coalescer are thin consumers of this module.
//
// Wire discipline: a live session streams "live://state" events at
// ~emitHz from a background physics thread. Per-event work must stay
// off React, so the store stashes ONLY the latest event in a module
// ref (classifyLiveState decides stash/drop) and one rAF flush per
// frame turns it into a single set() (liveFlushFate + liveStatePatch).
// ============================================================

import type { PropTrack, SimProp } from "./props";
import type { SuccessPredicate } from "./task";

/** Camel-case mirror of `LiveStartReq` — what `live_start` is asked with. */
export interface LiveStartReq {
  q0: number[];
  engine: string;
  props: SimProp[];
  /** override the auto-detected gripper joint by caliper joint name (the
   *  backend Errs with the joint list when the name is unknown); omitted =
   *  auto-detect, and a robot with no gripper simply gets no channel */
  gripperJoint?: string | null;
  /** which end of that joint's range CLOSES it; omitted = "lo" */
  gripperClosed?: "lo" | "hi" | null;
  /** ground-plane height of the scene (mujoco); omitted = 0 */
  ground?: number;
  /** success predicate to judge the session against, in the
   *  `caliper-task.json` schema — handed over verbatim from `task_open` and
   *  never inspected here. Omitted = no verdict is computed. */
  success?: SuccessPredicate;
}

/** Camel-case mirror of `GripperDto` — the session's gripper channel: one
 *  joint and the two hold-target values `live_gripper` writes into its slot
 *  (the joint's limits, inset 2% so the PD never slams the stop). */
export interface GripperInfo {
  joint: string;
  /** index of that joint in `q` / `target` */
  index: number;
  openTarget: number;
  closedTarget: number;
}

/** Camel-case mirror of `GripperStateEvent` — what the stream reports about
 *  the jaw. `closed` is the last COMMAND, not a measurement: a gripper
 *  squeezing a prop reads `closed: true` with `q` short of `closedTarget`. */
export interface GripperState {
  closed: boolean;
  q: number;
}

/** Camel-case mirror of `LiveStartedDto` (src-tauri/src/live.rs). */
export interface LiveStartedDto {
  /** monotonic session id — every event carries it for stale filtering */
  sessionId: number;
  engine: string;
  /** physics timestep (s) */
  h: number;
  /** ACTUAL state-event rate after integer-step decimation */
  emitHz: number;
  ndof: number;
  /** static prop shape/color rows in build order; `frames` empty (live
   *  poses arrive per-event, not baked) — reuses the PropTrack shape */
  props: PropTrack[];
  /** the gripper channel this session found; null/absent = no channel, and
   *  then `live_gripper` errors and `held` is always null */
  gripper?: GripperInfo | null;
}

/** Camel-case mirror of `LiveStateEvent` ("live://state"). */
export interface LiveStateEvent {
  sessionId: number;
  tick: number;
  t: number;
  q: number[];
  qd: number[];
  /** column-major world matrices, exactly what get_frames returns */
  frames: number[][];
  tip: [number, number, number];
  ncon: number;
  /** per-prop `[x,y,z,qw,qx,qy,qz]` matching LiveStartedDto.props order */
  props: number[][];
  paused: boolean;
  target: number[];
  /** a take is in progress (A3 recording) */
  recording: boolean;
  /** frames captured in the current take; 0 when not recording */
  recFrames: number;
  /** gripper command + measurement; null/absent with no gripper channel */
  gripper?: GripperState | null;
  /** prop currently WELDED to the gripper (grasp heuristic), else null —
   *  always null on builtin, which has no contacts to grasp with */
  held?: string | null;
  /** the session's success verdict for THIS state, or null when nothing is
   *  scoring it (no predicate, or an engine with no props to judge) */
  success?: boolean | null;
}

/** Camel-case mirror of `LiveRecordStartedDto` (reply to `live_record_start`).
 *  `root` is the path the dataset ACTUALLY lives at — subsequent takes echo it
 *  back, so the UI remembers this and never re-sends its own string. */
export interface LiveRecordStartedDto {
  root: string;
  fps: number;
  /** physics ticks per recorded frame */
  recordEvery: number;
  /** 0-based index this episode gets when saved */
  episodeIndex: number;
}

/** Camel-case mirror of `LiveRecordStoppedDto` (reply to `live_record_stop`). */
export interface LiveRecordStoppedDto {
  saved: boolean;
  /** index of the saved episode; null for a discarded take */
  episodeIndex: number | null;
  /** frames the take held (saved or thrown away) */
  frames: number;
}

/** Camel-case mirror of `LiveRecordFinishedDto` (reply to `live_record_finish`). */
export interface LiveRecordFinishedDto {
  root: string;
  episodes: number;
}

/** Camel-case mirror of `LiveEndedEvent` ("live://ended"). */
export interface LiveEndedEvent {
  sessionId: number;
  /** "stopped" | "superseded" | "error: <detail>" */
  reason: string;
}

/** The store's live slice: session identity + the latest streamed scalars.
 *  `props` is the STATIC shape info from the started DTO; the streaming
 *  poses live beside it in `livePropPoses` (kept separate so the shape
 *  rows stay referentially stable across 60 Hz pose updates). */
export interface LiveInfo {
  sessionId: number;
  engine: string;
  paused: boolean;
  t: number;
  tick: number;
  ncon: number;
  props: PropTrack[];
  /** the STATIC gripper channel, like `props` — null with no gripper joint */
  gripper: GripperInfo | null;
  /** latest streamed jaw command + measurement; null until the first event
   *  (and forever on a session with no channel) */
  gripperState: GripperState | null;
  /** prop welded to the gripper right now, null when nothing is held */
  held: string | null;
  /** latest streamed verdict of the session's success predicate: true/false
   *  per INSTANT (never latched), null when nothing is scoring the session */
  success: boolean | null;
}

/** Fresh live slice for a session the backend just started. */
export function liveInfoFromStarted(dto: LiveStartedDto): LiveInfo {
  return {
    sessionId: dto.sessionId,
    engine: dto.engine,
    paused: false,
    t: 0,
    tick: 0,
    ncon: 0,
    props: dto.props,
    gripper: dto.gripper ?? null,
    gripperState: null,
    held: null,
    // no verdict until the first event: a session that IS being scored reports
    // one on every state, and one that is not reports null forever
    success: null,
  };
}

/** The panel's gripper control, decided in one place: what the button says,
 *  what it explains, and whether there is a channel to press it against.
 *  `closed` is the COMMAND intent the stream reports, never a measurement. */
export interface GripperControl {
  label: string;
  title: string;
  disabled: boolean;
  closed: boolean;
}

/** Why the control is off when a session found no jaw joint (the backend's
 *  auto-detect and its override are the two ways out). */
export const NO_GRIPPER_TITLE =
  "no gripper joint detected — name one 'gripper'/'finger'/'jaw' or pass an override";

export function gripperControl(
  ch: GripperInfo | null,
  st: GripperState | null,
): GripperControl {
  if (!ch) {
    return { label: "gripper: none", title: NO_GRIPPER_TITLE, disabled: true, closed: false };
  }
  // a session starts with the jaw OPEN, which is also what it reads before the
  // first state event lands
  const closed = st?.closed ?? false;
  return {
    label: closed ? "gripper: closed ▸ open" : "gripper: open ▸ close",
    title: `${closed ? "open" : "close"} the gripper (G · pad X) — joint ${ch.joint}`,
    disabled: false,
    closed,
  };
}

/** How far off its commanded target the jaw may sit and still count as having
 *  ARRIVED there, as a fraction of the open→closed span. */
export const GRIP_SEATED_EPS_FRAC = 0.1;

/** Is the jaw where the command asked it to go? False while it is still
 *  travelling AND while it is stopped short on a prop — the panel shows the
 *  measurement then instead of letting the command speak for it. */
export function gripperSeated(ch: GripperInfo, st: GripperState | null): boolean {
  if (!st || !Number.isFinite(st.q)) return false;
  const goal = st.closed ? ch.closedTarget : ch.openTarget;
  const span = Math.abs(ch.closedTarget - ch.openTarget);
  return Math.abs(st.q - goal) <= span * GRIP_SEATED_EPS_FRAC;
}

/** Reconcile a streamed gripper report against a `live_gripper` we sent but
 *  have not seen come back. An event emitted BEFORE the command still carries
 *  the old intent, so until the stream agrees the command wins — otherwise the
 *  button would flip back for a frame, and (worse) the drive loop would adopt
 *  a jaw value the backend has already moved past. `settled` is the caller's
 *  cue to drop the pending intent: the stream has caught up. */
export function reconcileGripper(
  ev: GripperState | null,
  pending: boolean | null,
): { state: GripperState | null; settled: boolean } {
  if (pending === null || !ev) return { state: ev, settled: false };
  if (ev.closed === pending) return { state: ev, settled: true };
  return { state: { ...ev, closed: pending }, settled: false };
}

/** The store's recording slice: the OPEN dataset plus the take running into it,
 *  or null when no dataset is open. It outlives a take (a stopped take leaves
 *  `root`/`fps`/`episodesSaved` behind so the next one continues the same
 *  dataset) but never the session — the backend finalizes on session end. */
export interface LiveRecInfo {
  /** dataset root as the BACKEND reports it, not as we asked for it */
  root: string;
  fps: number;
  /** a take is running right now */
  recording: boolean;
  /** task label of the running take; null between takes */
  task: string | null;
  /** frames captured in the running take */
  frames: number;
  episodesSaved: number;
}

/** Recording slice after a `live_record_start` reply. The dataset the backend
 *  names is authoritative; a reply naming a DIFFERENT root than the one we were
 *  recording into is a new dataset, so its episode count restarts. */
export function liveRecFromStarted(
  prev: LiveRecInfo | null,
  dto: LiveRecordStartedDto,
  task: string,
): LiveRecInfo {
  return {
    root: dto.root,
    fps: dto.fps,
    recording: true,
    task,
    frames: 0,
    episodesSaved: prev && prev.root === dto.root ? prev.episodesSaved : 0,
  };
}

/** Recording slice after a `live_record_stop` reply: the take is over either
 *  way, and only a SAVED one adds an episode to the open dataset. */
export function liveRecAfterStop(prev: LiveRecInfo, dto: LiveRecordStoppedDto): LiveRecInfo {
  return {
    ...prev,
    recording: false,
    task: null,
    frames: 0,
    episodesSaved: prev.episodesSaved + (dto.saved ? 1 : 0),
  };
}

/** Fold one state event's recording fields into the slice.
 *
 *  `armed` is the caller's "the stream has confirmed this take" latch. Until it
 *  is set, a `recording: false` event is an OLDER event crossing the start
 *  reply, not the take ending. Once set, a true→false transition is the only
 *  signal a take died without a command (`live_reset` auto-discards it, and a
 *  rejected frame kills it) — the caller turns `discarded` into the hint.
 *
 *  Returns `prev` itself when nothing moved: the flush set()s what it gets, so
 *  a new object per streamed frame would repaint the panel for nothing. */
export function liveRecPatch(
  prev: LiveRecInfo | null,
  ev: LiveStateEvent,
  armed: boolean,
): { rec: LiveRecInfo | null; discarded: boolean } {
  if (!prev) return { rec: prev, discarded: false }; // no dataset open — not ours
  if (ev.recording) {
    return prev.recording && prev.frames === ev.recFrames
      ? { rec: prev, discarded: false }
      : { rec: { ...prev, recording: true, frames: ev.recFrames }, discarded: false };
  }
  if (!prev.recording || !armed) return { rec: prev, discarded: false };
  return { rec: { ...prev, recording: false, task: null, frames: 0 }, discarded: true };
}

/** What the "live://state" handler does with an incoming event:
 *  - an event of the current (or a NEWER, still-adopting) session → stash
 *    into the latest-event ref (latest-wins within a frame);
 *  - with no current session, stash only while a start is in flight
 *    (its events can beat the live_start reply through the event loop);
 *  - anything older than the current session → drop. */
export type LiveStateFate = "stash" | "drop";
export function classifyLiveState(
  cur: LiveInfo | null,
  evSessionId: number,
  startsInFlight: number,
): LiveStateFate {
  if (cur) return evSessionId >= cur.sessionId ? "stash" : "drop";
  return startsInFlight > 0 ? "stash" : "drop";
}

/** What the rAF flush does with the stashed latest event:
 *  - "apply": it belongs to the current session → one set();
 *  - "keep": it is for a session we have not adopted yet (start reply in
 *    flight) → leave it stashed, the start path re-schedules a flush;
 *  - "drop": nothing stashed, or it describes an older session. */
export type LiveFlushFate = "apply" | "keep" | "drop";
export function liveFlushFate(cur: LiveInfo | null, ev: LiveStateEvent | null): LiveFlushFate {
  if (!ev) return "drop";
  if (!cur) return "keep";
  if (ev.sessionId === cur.sessionId) return "apply";
  return ev.sessionId > cur.sessionId ? "keep" : "drop";
}

/** The single per-frame store patch for one state event: pose + frames go
 *  through the same fields playback uses, the live scalars ride beside them,
 *  and the prop poses replace wholesale (build order matches `live.props`). */
export function liveStatePatch(
  prev: LiveInfo,
  ev: LiveStateEvent,
): { q: number[]; frames: number[][]; live: LiveInfo; livePropPoses: number[][] } {
  return {
    q: ev.q,
    frames: ev.frames,
    live: {
      ...prev,
      paused: ev.paused,
      t: ev.t,
      tick: ev.tick,
      ncon: ev.ncon,
      // the channel itself is static (it rides `prev`); these two are the
      // per-event half of it — a backend without them reads as "no gripper",
      // which is exactly what an older one means
      gripperState: ev.gripper ?? null,
      held: ev.held ?? null,
      // the verdict describes the poses in THIS event, so it replaces
      // wholesale like the rest of them (absent = an older backend = nothing
      // scoring the session, which is what null already means)
      success: ev.success ?? null,
    },
    livePropPoses: ev.props,
  };
}

// ---- policy in the loop (E1) ----
// A trained policy running in the USER'S python environment drives the RUNNING
// session: the child observes at `hz` and writes the SAME PD hold target every
// human input writes, so nothing about the session changes shape while it
// drives — a nudge still lands, a pause still freezes, a take still records.

/** Camel-case mirror of `LivePolicyStartReq` — what `live_policy_start` asks
 *  with. `python` is a binary path OR a venv directory (the backend resolves
 *  it); `ckpt` is the trained-policy directory. */
export interface LivePolicyStartReq {
  python: string;
  ckpt: string;
  /** torch device the child loads onto; omitted = the backend picks */
  device?: string;
  /** observation rate (Hz); omitted = 20 */
  hz?: number;
}

/** Camel-case mirror of the `live_policy_start` reply — what the handshake
 *  learned about the policy that is now driving. */
export interface LivePolicyStartedDto {
  ndof: number;
  /** actions the policy emits per inference — its RE-PLAN period. Display
   *  metadata only: the child paces itself, nothing here schedules on it. */
  chunk: number;
  policyType: string;
  device: string;
}

/** Camel-case mirror of `LivePolicyEvent` ("live://policy"). Exactly one
 *  TERMINAL event ("stopped" or "error") arrives per drive; "driving" fires
 *  once after the handshake, carrying the policy type as its detail. */
export interface LivePolicyEvent {
  sessionId: number;
  state: "driving" | "stopped" | "error";
  /** policy type on "driving", the failure on "error" (a backend failure
   *  detail carries the child's stderr tail verbatim), else null */
  detail: string | null;
}

/** The store's policy slice, or null when no policy is attached to the
 *  session. `loading` covers the whole handshake — a cold torch import plus
 *  weights can take the better part of a minute, and the panel says so. */
export interface LivePolicyInfo {
  state: "loading" | "driving";
  /** what the handshake reported; absent until it lands */
  policyType?: string;
  device?: string;
  chunk?: number;
  /** what the human asked with, kept so the badge can name the checkpoint */
  python: string;
  ckpt: string;
}

/** Why a connect attempt is not worth making, or null when it is. The backend
 *  validates too — this only keeps an obviously empty form off the wire. */
export function policyFormError(python: string, ckpt: string): string | null {
  if (!python.trim()) return "a policy needs a python environment (binary or venv directory)";
  if (!ckpt.trim()) return "a policy needs a trained-checkpoint directory";
  return null;
}

/** The slice after a `live_policy_start` reply lands on the loading one. The
 *  "driving" EVENT may have beaten the reply here (it fires on the same
 *  handshake), so this fills the reply's fields in rather than replacing the
 *  slice wholesale. */
export function livePolicyFromStarted(
  prev: LivePolicyInfo,
  dto: LivePolicyStartedDto,
): LivePolicyInfo {
  return {
    ...prev,
    state: "driving",
    policyType: dto.policyType,
    device: dto.device,
    chunk: dto.chunk,
  };
}

/** What the "live://policy" handler does with an incoming event. A policy
 *  belongs to ONE session, so an event naming any other session describes a
 *  drive that is already over — the same stale filter the state channel uses,
 *  except there is no in-flight window to stash for (a policy can only be
 *  connected to a session that is already adopted). */
export type LivePolicyFate = "apply" | "drop";
export function classifyPolicyEvent(cur: LiveInfo | null, evSessionId: number): LivePolicyFate {
  return cur && cur.sessionId === evSessionId ? "apply" : "drop";
}

/** What the panel says about the policy on the session right now. */
export interface PolicyBadge {
  label: string;
  title: string;
}

export function policyBadge(p: LivePolicyInfo): PolicyBadge {
  if (p.state === "loading") {
    return {
      label: "loading policy…",
      title: `starting ${p.ckpt} in ${p.python} — a cold torch import can take a minute`,
    };
  }
  const type = p.policyType ?? "policy";
  const dev = p.device ?? "?";
  return {
    label: `policy: ${type} @ ${dev}`,
    title:
      `${p.ckpt} — ${p.chunk ? `${p.chunk} actions per inference, ` : ""}` +
      "driving the same hold target your inputs write",
  };
}

/** Store patch for "live://ended": the session is gone either way; only a
 *  reason beyond the two benign ends ("stopped", "superseded") surfaces in
 *  the error banner. */
export function liveEndedPatch(reason: string): {
  live: null;
  livePropPoses: number[][];
  error?: string;
} {
  const benign = reason === "stopped" || reason === "superseded";
  return benign
    ? { live: null, livePropPoses: [] }
    : { live: null, livePropPoses: [], error: `live sim ended — ${reason}` };
}
