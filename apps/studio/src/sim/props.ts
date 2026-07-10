// ============================================================
// sim/props.ts — pure contact-sim helpers. PURE and headless:
// no React, no Tauri, no store (types only flow the other way),
// so the vitest suite drives every function directly. The store
// and the 3D PropsLayer are just consumers of this module.
// ============================================================

/** Free-prop primitive kinds the backend accepts (`PropDto.kind`). */
export type PropKind = "box" | "sphere" | "cylinder";

/** One user-added free prop — camelCase mirror of `PropDto` in
 *  src-tauri/src/lib.rs. `kind` selects the populated size fields:
 *  box → halfExtents; sphere → radius; cylinder → radius + length
 *  (FULL length, Z-aligned, MJCF/URDF convention). */
export interface SimProp {
  name: string;
  kind: PropKind;
  halfExtents: [number, number, number] | null;
  radius: number | null;
  length: number | null;
  /** initial world position of the primitive's center (URDF world, Z-up) */
  pos: [number, number, number];
  /** initial world orientation, w-first (MJCF order); null = identity */
  quat: [number, number, number, number] | null;
  /** mass in kg */
  mass: number;
  /** display color in [0,1]; null → the renderer's neutral accent */
  rgba: [number, number, number, number] | null;
}

/** Baked world-pose track of one prop — mirror of `PropTrackDto`.
 *  `frames[k]` (aligned with the parent DTO's `times[k]`) =
 *  `[x, y, z, qw, qx, qy, qz]` (position + w-first quaternion). */
export interface PropTrack {
  name: string;
  kind: string;
  halfExtents: [number, number, number] | null;
  radius: number | null;
  length: number | null;
  rgba: [number, number, number, number] | null;
  frames: [number, number, number, number, number, number, number][];
}

/** Hard cap on user-added props (keeps the MJCF scene + the panel compact). */
export const MAX_PROPS = 5;

/** Does this build expose the MuJoCo contact engine? (`sim_engines` result.)
 *  Everything contact-flavored in the UI gates on this — when it is false the
 *  Simulate surface must render exactly as it did before contact sim existed. */
export function hasContactEngine(engines: string[]): boolean {
  return engines.includes("mujoco");
}

/** Baked-frame index at playback time `t`: nearest sample, clamped into
 *  [0, n-1] — the same rounding `_applyTrajAt` uses for the robot pose, so a
 *  prop and the arm always read the SAME baked instant. Degenerate inputs
 *  (dt ≤ 0, n ≤ 0, t < 0) pin to 0; callers index only when n ≥ 1. */
export function frameIndexAt(t: number, dt: number, n: number): number {
  if (!(dt > 0) || n <= 0) return 0;
  return Math.max(0, Math.min(Math.round(t / dt), n - 1));
}

/** Instrument-adjacent display tones cycled by prop index (rgba in [0,1]). */
const PROP_TONES: [number, number, number, number][] = [
  [0.21, 0.78, 0.83, 1.0], // teal (accent family)
  [0.9, 0.65, 0.25, 1.0], // amber
  [0.55, 0.55, 1.0, 1.0], // violet
  [0.24, 0.84, 0.55, 1.0], // green
  [1.0, 0.36, 0.42, 1.0], // rose
];

/** Build a fresh default prop of `kind`: a small light primitive DROPPED ABOVE
 *  the ground plane (z > 0, URDF world), staggered by index so consecutive
 *  props never spawn inside each other, with a name unique within `existing`
 *  (`box1`, `box2`, `sphere1`, …). Pure — it does not mutate `existing`. */
export function defaultProp(kind: PropKind, existing: SimProp[]): SimProp {
  const names = new Set(existing.map((p) => p.name));
  let ord = 1;
  while (names.has(`${kind}${ord}`)) ord++;
  const i = existing.length;
  // 3-wide lateral fan, then a higher shelf — every slot is airborne and clear
  // of the base plate (x = 0.35 m out) and of every earlier slot.
  const pos: [number, number, number] = [
    0.35,
    ((i % 3) - 1) * 0.12,
    0.3 + Math.floor(i / 3) * 0.15,
  ];
  return {
    name: `${kind}${ord}`,
    kind,
    halfExtents: kind === "box" ? [0.03, 0.03, 0.03] : null,
    radius: kind === "sphere" ? 0.03 : kind === "cylinder" ? 0.025 : null,
    length: kind === "cylinder" ? 0.08 : null,
    pos,
    quat: null,
    mass: 0.1,
    rgba: PROP_TONES[i % PROP_TONES.length],
  };
}

/** Append a default prop of `kind`; at the MAX_PROPS cap the SAME list comes
 *  back unchanged (callers can reference-compare to detect the refusal). */
export function withProp(list: SimProp[], kind: PropKind): SimProp[] {
  if (list.length >= MAX_PROPS) return list;
  return [...list, defaultProp(kind, list)];
}
