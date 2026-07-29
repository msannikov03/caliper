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

import type { PropTrack } from "./props";

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
  };
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
    live: { ...prev, paused: ev.paused, t: ev.t, tick: ev.tick, ncon: ev.ncon },
    livePropPoses: ev.props,
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
