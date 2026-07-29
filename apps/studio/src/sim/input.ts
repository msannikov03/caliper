// ============================================================
// sim/input.ts — pure human-input helpers for the live session.
// PURE and headless: no React, no Tauri, no store (types only flow
// the other way), so the vitest suite drives every function directly.
// The store's per-frame drive step and App's keymap are thin consumers.
//
// The unit here is ONE FRAME: every integrator takes a dt and returns
// the next value, so the caller (the live rAF coalescer) folds keyboard,
// gamepad and slider edits into a single hold target per frame and sends
// it with at most one live_set_target invoke.
// ============================================================

/** Mirror of the store's `JointKind` (kept local so nothing flows back). */
export type JogKind = "revolute" | "prismatic";

/** Base jog rates. Revolute rad/s, prismatic m/s — slow enough to place a tip
 *  by hand, fast enough to cross the workspace in a couple of seconds. */
export const JOG_RATE_REVOLUTE = 1.5;
export const JOG_RATE_PRISMATIC = 0.25;

/** Limit fallback for a joint the URDF leaves unbounded — the same ±π rad /
 *  ±0.5 m the JointPanel sliders fall back to, so the slider range and the
 *  keyboard jog can never disagree about where a joint ends. */
export const JOG_SPAN_REVOLUTE = Math.PI;
export const JOG_SPAN_PRISMATIC = 0.5;

/** Analog-stick deadband: below this the axis reads exactly 0 (a worn stick
 *  rests around 0.05–0.1 and would otherwise creep the tip forever). */
export const GAMEPAD_DEADBAND = 0.15;

/** Full-deflection cartesian tip speed (m/s). */
export const TIP_RATE = 0.25;

/** How long after the last target change the UI still reads as "driving". */
export const DRIVE_IDLE_MS = 500;

/** Gamepad button indices (standard mapping): A/cross toggles the freeze,
 *  B/circle restarts the session at its starting pose. */
export const PAD_BUTTON_PAUSE = 0;
export const PAD_BUTTON_RESET = 1;

/** Keys held to jog the selected joint (`=`/`+`/↑ raise, `-`/`_`/↓ lower). */
export const JOG_UP_KEYS = ["=", "+", "ArrowUp"];
export const JOG_DOWN_KEYS = ["-", "_", "ArrowDown"];
/** Keys that move the keyboard-jog selection along the chain. */
export const JOINT_PREV_KEYS = ["["];
export const JOINT_NEXT_KEYS = ["]"];

/** Per-second jog rate for one joint: the kind's base rate, capped by the
 *  model's velocity limit when it carries one (0/absent = unbounded). */
export function jogRate(kind: JogKind, velLimit?: number | null): number {
  const base = kind === "prismatic" ? JOG_RATE_PRISMATIC : JOG_RATE_REVOLUTE;
  return velLimit && velLimit > 0 ? Math.min(velLimit, base) : base;
}

/** One frame's jog increment: `dir` (−1/0/+1) × rate × dt. A non-positive dt
 *  (a stalled or rewound clock) contributes nothing. */
export function jogDelta(
  kind: JogKind,
  dt: number,
  dir: number,
  velLimit?: number | null,
): number {
  if (!(dt > 0) || dir === 0 || !Number.isFinite(dir)) return 0;
  return Math.sign(dir) * jogRate(kind, velLimit) * dt;
}

/** Clamp one joint value into its `[lo, hi]`, falling back to the kind's
 *  default span when the URDF leaves the joint unbounded. */
export function clampJoint(
  v: number,
  lim: [number, number] | null | undefined,
  kind: JogKind = "revolute",
): number {
  const span = kind === "prismatic" ? JOG_SPAN_PRISMATIC : JOG_SPAN_REVOLUTE;
  const [lo, hi] = lim ?? [-span, span];
  return Math.min(Math.max(v, lo), hi);
}

/** Pure target integrator: fold per-joint deltas into `target` and clamp each
 *  result into its limit. Entries `deltas` does not cover pass through
 *  untouched (and are NOT re-clamped — only what moved is constrained). */
export function applyJog(
  target: number[],
  deltas: number[],
  limits: ([number, number] | null)[],
  kinds: JogKind[] = [],
): number[] {
  return target.map((v, i) => {
    const d = deltas[i];
    return d ? clampJoint(v + d, limits[i], kinds[i] ?? "revolute") : v;
  });
}

/** Deadband + cubic response for one analog axis: everything inside the
 *  deadband reads 0, the remainder is re-normalized to [0,1] and cubed so
 *  small deflections are gentle while full deflection still reaches ±1.
 *  Sign is preserved (an odd power) and the result is clamped to ±1. */
export function axisResponse(v: number, deadband = GAMEPAD_DEADBAND): number {
  if (!Number.isFinite(v)) return 0;
  const a = Math.abs(v);
  if (a <= deadband) return 0;
  const n = Math.min((a - deadband) / (1 - deadband), 1);
  return Math.sign(v) * n * n * n;
}

/** What the gamepad asks for this frame. `tip` is a cartesian velocity in
 *  URDF world (m/s); the two button fields are EDGES, true only on the frame
 *  the button goes down. */
export interface GamepadIntent {
  tip: [number, number, number];
  /** any non-zero tip component — the tip goal only integrates while true */
  moving: boolean;
  togglePause: boolean;
  reset: boolean;
}

/** Map one polled gamepad frame to an intent. Left stick X/Y (axes 0/1) drive
 *  the tip in world X/Y, the right stick's vertical axis (axis 3) drives world
 *  Z; both vertical axes are inverted because a pad reports "stick pushed
 *  forward" as −1 and forward/up should read positive. `prevButtons` is the
 *  previous frame's pressed state, so a held button fires exactly once. */
export function gamepadIntent(
  axes: readonly number[],
  buttons: readonly boolean[],
  prevButtons: readonly boolean[],
): GamepadIntent {
  const x = axisResponse(axes[0] ?? 0);
  // `0 −` rather than unary minus so a centred stick reports +0, not −0
  const y = 0 - axisResponse(axes[1] ?? 0);
  const z = 0 - axisResponse(axes[3] ?? 0);
  const tip: [number, number, number] = [x * TIP_RATE, y * TIP_RATE, z * TIP_RATE];
  const edge = (i: number) => buttons[i] === true && prevButtons[i] !== true;
  return {
    tip,
    moving: x !== 0 || y !== 0 || z !== 0,
    togglePause: edge(PAD_BUTTON_PAUSE),
    reset: edge(PAD_BUTTON_RESET),
  };
}

/** Integrate a cartesian velocity into a tip goal for one frame. */
export function applyTipVelocity(
  goal: readonly [number, number, number],
  vel: readonly [number, number, number],
  dt: number,
): [number, number, number] {
  const h = dt > 0 ? dt : 0;
  return [goal[0] + vel[0] * h, goal[1] + vel[1] * h, goal[2] + vel[2] * h];
}

/** Net jog direction of the currently-held keys: opposing keys cancel. */
export function jogDirection(pressed: Iterable<string>): -1 | 0 | 1 {
  let up = false;
  let down = false;
  for (const k of pressed) {
    if (JOG_UP_KEYS.includes(k)) up = true;
    else if (JOG_DOWN_KEYS.includes(k)) down = true;
  }
  if (up === down) return 0;
  return up ? 1 : -1;
}

/** Is this key one the live jog holds (so App tracks its keyup)? */
export function isJogKey(key: string): boolean {
  return JOG_UP_KEYS.includes(key) || JOG_DOWN_KEYS.includes(key);
}

/** Move the keyboard-jog selection by `delta`, wrapping at both ends. An
 *  empty model (ndof 0) stays at 0. */
export function stepJoint(cur: number, delta: number, ndof: number): number {
  if (ndof <= 0) return 0;
  const base = Number.isFinite(cur) ? Math.trunc(cur) : 0;
  return (((base + delta) % ndof) + ndof) % ndof;
}

/** What a keydown means to a RUNNING live session. App only dispatches this
 *  when a session is up, the mode is simulate and focus is not in a field —
 *  everything else about the key (repeat, modifiers) stays in App. */
export type LiveKeyAction = "freeze" | "prev-joint" | "next-joint" | "jog" | null;
export function liveKeyAction(key: string): LiveKeyAction {
  if (key === " " || key === "Spacebar") return "freeze";
  if (JOINT_PREV_KEYS.includes(key)) return "prev-joint";
  if (JOINT_NEXT_KEYS.includes(key)) return "next-joint";
  return isJogKey(key) ? "jog" : null;
}
