// ============================================================
// sim/task.ts — pure task-artifact helpers (a `*.caliper-task.json`).
// PURE and headless: no React, no Tauri, no store (types only flow
// the other way), so the vitest suite drives every function directly.
//
// The backend owns the file: `task_open` parses it, validates it AND
// loads its robot into app state, then hands the whole thing back.
// Everything here is the shape of that reply plus the small decisions
// the UI makes from it — what a live session is asked with, what the
// record panel defaults to, what the verdict badge says. The success
// predicate itself is never inspected: it arrives as JSON and goes
// back to `live_start` unchanged.
// ============================================================

import type { SimProp } from "./props";

/** One named target region — camelCase mirror of `ZoneDto`. Evaluator-side
 *  only: nothing is emitted into MJCF for a zone, so it never collides. */
export interface TaskZone {
  name: string;
  /** center of the axis-aligned box, URDF world (Z-up), metres */
  center: [number, number, number];
  /** half extents, metres */
  half: [number, number, number];
  /** render hint in [0,1]; null = let the renderer choose */
  rgba: [number, number, number, number] | null;
}

/** The gripper override a task carries, in the same spelling `live_start`
 *  takes (`gripperJoint` / `gripperClosed`). */
export interface TaskGripperSpec {
  joint?: string | null;
  closed?: "lo" | "hi" | null;
}

/** A success predicate exactly as the backend serialized it. OPAQUE by
 *  design: it comes from `task_open` and goes straight back to `live_start`,
 *  so nothing in the frontend ever looks inside it. */
export type SuccessPredicate = unknown;

/** Everything `task_open` returns EXCEPT the robot — the task half of the
 *  reply. `TaskDto` (store.ts) adds `robot: RobotInfo` on top, because the
 *  same call also loads the robot into app state. */
export interface TaskSpecDto {
  name: string;
  /** resolved robot path (the task file's `robot`, made absolute) */
  robotPath: string;
  /** start pose, already length-checked against the robot; null = none */
  q0: number[] | null;
  /** ground-plane height of the scene */
  ground: number;
  /** props in the `live_start` wire shape (a task pre-fills the prop editor) */
  props: SimProp[];
  zones: TaskZone[];
  gripper: TaskGripperSpec | null;
  success: SuccessPredicate | null;
  /** the same predicate as one plain-English sentence, or null */
  successDescription: string | null;
  horizonS: number | null;
  fps: number | null;
}

/** The loaded task as the store keeps it: the task half of the reply, minus its
 *  props (those land in `simProps` — the ONE scene both a bake and a live
 *  session are built from), plus the file it came from. */
export interface TaskInfo extends Omit<TaskSpecDto, "props"> {
  /** the `*.caliper-task.json` this was opened from */
  path: string;
}

/** Zone tint when the task names none: translucent green (the "good" family —
 *  a zone is where the verdict wants the prop to end up). */
export const ZONE_RGBA_DEFAULT: [number, number, number, number] = [0.24, 0.84, 0.55, 0.3];

/** Adopt a `task_open` reply as the store's task slice. `path` is the file the
 *  user picked, which the reply itself does not echo back. Field by field on
 *  purpose: the reply also carries the robot and the props, and neither belongs
 *  in a second copy here. */
export function taskInfoFromDto(dto: TaskSpecDto, path: string): TaskInfo {
  return {
    path,
    name: dto.name,
    robotPath: dto.robotPath,
    q0: dto.q0,
    ground: dto.ground,
    zones: dto.zones,
    gripper: dto.gripper,
    success: dto.success,
    successDescription: dto.successDescription,
    horizonS: dto.horizonS,
    fps: dto.fps,
  };
}

/** The task's half of a `live_start` request: the ground it stands on, the
 *  gripper channel it names and the predicate to judge it by. Empty without a
 *  task, so a hand-built session sends exactly what it always did. */
export interface TaskLiveFields {
  ground?: number;
  gripperJoint?: string;
  gripperClosed?: "lo" | "hi";
  success?: SuccessPredicate;
}

export function taskLiveStartFields(task: TaskInfo | null): TaskLiveFields {
  if (!task) return {};
  const f: TaskLiveFields = { ground: task.ground };
  // both halves of the override are optional on the wire: a task may name the
  // jaw joint, its closed end, either or neither (absent = auto-detect)
  if (task.gripper?.joint) f.gripperJoint = task.gripper.joint;
  if (task.gripper?.closed) f.gripperClosed = task.gripper.closed;
  if (task.success !== null && task.success !== undefined) f.success = task.success;
  return f;
}

/** Recording rates the panel offers: the standard choices, plus the task's own
 *  rate when it is not one of them — a dataset recorded at a rate the task does
 *  not declare is not that task's dataset, so the option has to exist rather
 *  than silently snap to 50. Ascending and de-duped. */
export function recFpsChoices(base: readonly number[], taskFps: number | null | undefined): number[] {
  const all = new Set<number>(base);
  if (typeof taskFps === "number" && Number.isFinite(taskFps) && taskFps > 0) all.add(taskFps);
  return [...all].sort((a, b) => a - b);
}

/** The live verdict badge: what it says, how it is styled, what it explains. */
export interface SuccessBadge {
  label: string;
  className: string;
  title: string;
}

/** The badge for `live://state.success`, or null when NOTHING is scoring this
 *  session (`null` = no predicate, or an engine with no props to judge — a
 *  silent "not yet" there would claim a verdict nobody is computing).
 *
 *  The verdict is per-INSTANT and never latched: a session that succeeded and
 *  then knocked the prop out of the zone reads "not yet" again, which is what
 *  the stream says. */
export function successBadge(
  success: boolean | null | undefined,
  description: string | null,
): SuccessBadge | null {
  if (success === null || success === undefined) return null;
  const title =
    description ??
    (success ? "the success predicate holds" : "the success predicate does not hold yet");
  return success
    ? { label: "SUCCESS ✓", className: "badge success", title }
    : { label: "success: not yet", className: "badge", title };
}
