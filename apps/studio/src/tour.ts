// ============================================================
// tour.ts — the first-run tour's step machine. PURE and headless:
// no React, no Tauri, no store, no DOM — the vitest suite drives
// tourReducer/shouldShowTour/markTourDone directly. The overlay UI
// (ui/Tour.tsx) is just a renderer + dispatcher for this.
//
// Contract with the rest of the app:
// - one localStorage flag (TOUR_KEY) decides "seen it" — set on BOTH
//   finish and skip, so the tour never shows uninvited twice;
// - a broken/unavailable storage means the tour NEVER auto-shows
//   (never-nag beats always-nag when we cannot remember the answer);
// - replay is always available explicitly (⌘K → "Show tour");
// - the tour is a pure overlay: it never touches the store, the
//   session-resume path, or the persistent GL <Canvas>.
// ============================================================

/** localStorage flag: present (any value) = the tour was finished or skipped. */
export const TOUR_KEY = "caliper.tourDone";

export interface TourStep {
  id: string;
  /** value of the `[data-tour]` attribute to highlight, or null (centered card) */
  anchor: string | null;
  title: string;
  body: string;
}

/** The guided path, in order. 6 steps: open a robot, the 5 mode tabs
 *  (Graph + Data share one), and the ⌘K palette. Anchors point at
 *  `data-tour` attributes on the toolbar and the mode tabs. */
export const TOUR_STEPS: readonly TourStep[] = [
  {
    id: "open",
    anchor: "open",
    title: "Open a robot",
    body:
      "Load any URDF from disk — Open URDF… (⌘O) — or pick a bundled sample " +
      "from the dropdown. Every load runs the asset doctor in the background; " +
      "broken files get a findings list and, when possible, a Repair & reload.",
  },
  {
    id: "jog",
    anchor: "tab-jog",
    title: "Jog",
    body:
      "Live forward kinematics: drag the joint sliders or grab the tip gizmo " +
      "to drive IK. The HUD tracks manipulability and warns near singularities.",
  },
  {
    id: "motion",
    anchor: "tab-motion",
    title: "Motion",
    body:
      "Plan jerk-limited MOVE_J / MOVE_L trajectories, save named poses, and " +
      "play them back on the transport at the bottom.",
  },
  {
    id: "simulate",
    anchor: "tab-simulate",
    title: "Simulate",
    body:
      "Gravity drops, computed-torque drive-to-goal, RRT planning and collision " +
      "checks — all on the deterministic engine. Builds with the MuJoCo feature " +
      "add a Builtin | Contact toggle for full contact physics.",
  },
  {
    id: "graph-data",
    anchor: "tab-graph",
    title: "Graph & Data",
    body:
      "Graph is a Simulink-style dataflow editor (run, validate, export " +
      ".caliper-graph.json). Data browses and edits LeRobotDataset v3.0 roots — " +
      "episode plots, camera thumbnails, and the dataset doctor — no robot needed.",
  },
  {
    id: "palette",
    anchor: "palette",
    title: "⌘K runs everything",
    body:
      "The command palette knows every action — loading, modes (⌘1…⌘5), " +
      "planning, sim runs, graph and dataset commands. Replay this tour any " +
      "time: ⌘K → “Show tour”.",
  },
];

/** The `n / total` progress label for the card. */
export function stepProgress(i: number): string {
  return `${clampStep(i) + 1} / ${TOUR_STEPS.length}`;
}

/** True on the final step (its Next button reads "Done"). */
export function isLastStep(i: number): boolean {
  return clampStep(i) === TOUR_STEPS.length - 1;
}

/** CSS selector for a step's highlight target, or null for a centered card. */
export function anchorSelector(step: TourStep): string | null {
  return step.anchor === null ? null : `[data-tour="${step.anchor}"]`;
}

/** Clamp an arbitrary number into a valid step index (defensive: a stale or
 *  out-of-range value degrades to the nearest real step, never a crash). */
export function clampStep(i: number): number {
  if (!Number.isFinite(i)) return 0;
  return Math.min(Math.max(Math.trunc(i), 0), TOUR_STEPS.length - 1);
}

export type TourAction = "next" | "back" | "skip";

/** The whole tour as one pure transition: state is the current step index or
 *  null (closed). `skip` closes from anywhere; `next` on the last step closes
 *  (that's "Done"); `back` clamps at the first step; null absorbs everything. */
export function tourReducer(step: number | null, action: TourAction): number | null {
  if (step === null) return null;
  const i = clampStep(step);
  switch (action) {
    case "skip":
      return null;
    case "next":
      return i >= TOUR_STEPS.length - 1 ? null : i + 1;
    case "back":
      return Math.max(0, i - 1);
  }
}

/** The slice of Storage the tour needs — injected so tests pass fakes and a
 *  quota/private-mode failure is exercised without touching real storage. */
export interface TourFlagStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** Should the tour auto-open on this launch? True only when the flag is
 *  readably absent. A storage failure returns false: if we cannot remember
 *  that the user dismissed it, we must not nag them every launch. */
export function shouldShowTour(store: TourFlagStore): boolean {
  try {
    return store.getItem(TOUR_KEY) === null;
  } catch {
    return false;
  }
}

/** Persist "seen it" (finish and skip both land here). Storage failures are
 *  swallowed — the tour still closes, it just may auto-show once more. */
export function markTourDone(store: TourFlagStore): void {
  try {
    store.setItem(TOUR_KEY, "1");
  } catch {
    // storage unavailable (private mode / quota) — degrade gracefully.
  }
}
