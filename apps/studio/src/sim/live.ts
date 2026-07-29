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
