// ============================================================
// commands.ts — the ⌘K command-palette model. PURE and headless:
// no React, no Tauri, no store (types only), so the vitest suite
// drives buildCommands/filterCommands directly. The palette UI
// (ui/Palette.tsx) is just a renderer + dispatcher for this.
// ============================================================

import type { StudioMode } from "./store";

export type CommandSection = "Robot" | "Mode" | "Motion" | "Simulate" | "Graph";

/** Section order — build order, header order, and the ranking tiebreak. */
const SECTION_ORDER: Record<CommandSection, number> = {
  Robot: 0,
  Mode: 1,
  Motion: 2,
  Simulate: 3,
  Graph: 4,
};

export interface Command {
  id: string;
  title: string;
  /** small mono annotation: a shortcut, a file path, or why it's disabled */
  hint?: string;
  section: CommandSection;
  enabled: boolean;
  run: () => void;
}

/** Everything buildCommands needs, as plain data + action callbacks — the
 *  palette assembles this from the store; tests pass hand-rolled fixtures.
 *  Every action wraps an EXISTING store/toolbar operation; nothing here may
 *  introduce a new backend call. */
export interface CommandCtx {
  fixtures: [string, string][]; // sample URDFs as [name, path]
  recents: string[]; // most-recent-first file paths
  poses: string[]; // saved named poses (MOVE_J targets)
  mode: StudioMode;
  robotLoaded: boolean;
  hasInertia: boolean;
  /** path of the currently-loaded URDF (the Reload target), null before any load */
  urdfPath: string | null;
  actions: {
    openUrdf: () => void;
    openPath: (path: string, record: boolean) => void;
    setMode: (m: StudioMode) => void;
    planHome: () => void; // planMoveJ(0…0)
    planToPose: (name: string) => void; // planMoveToPose
    driveHome: () => void; // runControl(0…0)
    gravityDrop: () => void; // runGravityDrop
    planRrtHome: () => void; // runPlan(0…0)
    checkCollision: () => void; // checkCollision(null)
    runGraph: () => void;
    validateGraph: () => void;
    duplicateSelection: () => void; // duplicateGraphSelection (⌘D in the editor)
    fitGraphView: () => void; // xyflow fitView via the editor's instance
    exportGraph: () => void; // native save dialog → save_graph_file
    importGraph: () => void; // native open dialog → load_graph_file
  };
}

/** Filename (last path segment) for display; full path goes in the hint/tooltip. */
export function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

/** ModeTabs order — the source of truth ⌘1…⌘4 (and the tabs) index into. */
export const MODE_TABS: { id: StudioMode; label: string }[] = [
  { id: "jog", label: "Jog" },
  { id: "motion", label: "Motion" },
  { id: "simulate", label: "Simulate" },
  { id: "graph", label: "Graph" },
];

/** Assemble the full command list in section order. Gated commands stay
 *  visible but disabled, with the gate spelled out in the hint. */
export function buildCommands(ctx: CommandCtx): Command[] {
  const { fixtures, recents, poses, mode, robotLoaded, hasInertia, urdfPath, actions } = ctx;
  const noRobot = robotLoaded ? null : "no robot loaded";
  const cmds: Command[] = [];

  // ---- Robot ----
  cmds.push({
    id: "robot.open",
    title: "Open URDF…",
    hint: "⌘O",
    section: "Robot",
    enabled: true,
    run: actions.openUrdf,
  });
  cmds.push({
    id: "robot.reload",
    title: "Reload robot",
    hint: urdfPath ? baseName(urdfPath) : "nothing loaded",
    section: "Robot",
    enabled: urdfPath !== null,
    // reselecting a recent re-adds (bumps) it; samples are not recorded
    run: () => {
      if (urdfPath) actions.openPath(urdfPath, recents.includes(urdfPath));
    },
  });
  for (const [name, path] of fixtures) {
    cmds.push({
      id: `robot.sample.${path}`,
      title: `Load sample: ${name}`,
      section: "Robot",
      enabled: true,
      run: () => actions.openPath(path, false), // sample fixtures never enter recents
    });
  }
  for (const path of recents) {
    cmds.push({
      id: `robot.recent.${path}`,
      title: `Open recent: ${baseName(path)}`,
      hint: path,
      section: "Robot",
      enabled: true,
      run: () => actions.openPath(path, true),
    });
  }

  // ---- Mode (mirrors the ModeTabs gating exactly) ----
  MODE_TABS.forEach((t, i) => {
    const active = mode === t.id;
    const needsInertia = t.id === "simulate" && !hasInertia;
    cmds.push({
      id: `mode.${t.id}`,
      title: `Switch to ${t.label}`,
      hint: noRobot ?? (active ? "active" : needsInertia ? "no inertial data" : `⌘${i + 1}`),
      section: "Mode",
      enabled: robotLoaded && !active && !needsInertia,
      run: () => actions.setMode(t.id),
    });
  });

  // ---- Motion (jog/motion own the pose; sim/graph would fight the clip) ----
  const motionGate =
    noRobot ?? (mode === "jog" || mode === "motion" ? null : "switch to Jog or Motion mode");
  cmds.push({
    id: "motion.home",
    title: "Plan move to home",
    hint: motionGate ?? undefined,
    section: "Motion",
    enabled: motionGate === null,
    run: actions.planHome,
  });
  for (const name of poses) {
    cmds.push({
      id: `motion.pose.${name}`,
      title: `Plan to pose: ${name}`,
      hint: motionGate ?? undefined,
      section: "Motion",
      enabled: motionGate === null,
      run: () => actions.planToPose(name),
    });
  }

  // ---- Simulate (mirrors the SimulatePanel gating) ----
  const simGate = noRobot ?? (mode === "simulate" ? null : "switch to Simulate mode");
  const dynGate = simGate ?? (hasInertia ? null : "no inertial data");
  cmds.push({
    id: "sim.drop",
    title: "Run gravity drop",
    hint: dynGate ?? undefined,
    section: "Simulate",
    enabled: dynGate === null,
    run: actions.gravityDrop,
  });
  cmds.push({
    id: "sim.home",
    title: "Drive to home (control)",
    hint: dynGate ?? undefined,
    section: "Simulate",
    enabled: dynGate === null,
    run: actions.driveHome,
  });
  cmds.push({
    id: "sim.plan",
    title: "Plan to home (RRT)",
    hint: simGate ?? undefined,
    section: "Simulate",
    enabled: simGate === null,
    run: actions.planRrtHome,
  });
  cmds.push({
    id: "sim.collision",
    title: "Check collision",
    hint: simGate ?? undefined,
    section: "Simulate",
    enabled: simGate === null,
    run: actions.checkCollision,
  });

  // ---- Graph ----
  const graphGate = noRobot ?? (mode === "graph" ? null : "switch to Graph mode");
  cmds.push({
    id: "graph.run",
    title: "Run graph",
    hint: graphGate ?? undefined,
    section: "Graph",
    enabled: graphGate === null,
    run: actions.runGraph,
  });
  cmds.push({
    id: "graph.validate",
    title: "Validate graph",
    hint: graphGate ?? undefined,
    section: "Graph",
    enabled: graphGate === null,
    run: actions.validateGraph,
  });
  cmds.push({
    id: "graph.duplicate",
    title: "Duplicate selected node",
    hint: graphGate ?? "⌘D",
    section: "Graph",
    enabled: graphGate === null,
    run: actions.duplicateSelection,
  });
  cmds.push({
    id: "graph.fit",
    title: "Fit graph view",
    hint: graphGate ?? undefined,
    section: "Graph",
    enabled: graphGate === null,
    run: actions.fitGraphView,
  });
  cmds.push({
    id: "graph.export",
    title: "Export graph…",
    hint: graphGate ?? undefined,
    section: "Graph",
    enabled: graphGate === null,
    run: actions.exportGraph,
  });
  cmds.push({
    id: "graph.import",
    title: "Import graph…",
    hint: graphGate ?? undefined,
    section: "Graph",
    enabled: graphGate === null,
    run: actions.importGraph,
  });

  return cmds;
}

/** Match rank for a lowercased query against a lowercased title:
 *  0 = title prefix · 1 = contiguous at a word boundary · 2 = scattered
 *  subsequence · null = no match. Lower is better. */
function matchRank(q: string, title: string): number | null {
  if (title.startsWith(q)) return 0;
  for (let i = title.indexOf(q); i > 0; i = title.indexOf(q, i + 1)) {
    if (!/[a-z0-9]/i.test(title[i - 1])) return 1;
  }
  let j = 0;
  for (let i = 0; i < title.length && j < q.length; i++) {
    if (title[i] === q[j]) j++;
  }
  return j === q.length ? 2 : null;
}

/** Case-insensitive subsequence filter with deterministic ranking: prefix >
 *  word-boundary > scattered; ties break by section order, then title. An
 *  empty query returns everything in stable build order. */
export function filterCommands(query: string, commands: Command[]): Command[] {
  const q = query.trim().toLowerCase();
  if (q === "") return commands.slice();
  const hits: { c: Command; rank: number }[] = [];
  for (const c of commands) {
    const rank = matchRank(q, c.title.toLowerCase());
    if (rank !== null) hits.push({ c, rank });
  }
  hits.sort((a, b) => {
    if (a.rank !== b.rank) return a.rank - b.rank;
    const s = SECTION_ORDER[a.c.section] - SECTION_ORDER[b.c.section];
    if (s !== 0) return s;
    return a.c.title < b.c.title ? -1 : a.c.title > b.c.title ? 1 : 0;
  });
  return hits.map((h) => h.c);
}
