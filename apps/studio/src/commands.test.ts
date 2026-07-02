// Headless unit tests for the ⌘K command-palette model (src/commands.ts).
//
// Everything here is pure: no store, no Tauri IPC, no DOM — buildCommands gets
// a hand-rolled ctx and filterCommands gets synthetic command lists, so both
// the content/gating rules and the ranking are machine-verified.

import { describe, it, expect } from "vitest";
import { buildCommands, filterCommands, baseName, MODE_TABS } from "./commands";
import type { Command, CommandCtx } from "./commands";

const noop = () => {};

function mkCtx(over: Partial<CommandCtx> = {}): CommandCtx {
  return {
    fixtures: [
      ["showcase6", "/fx/showcase6.urdf"],
      ["planar2", "/fx/planar2.urdf"],
    ],
    recents: ["/home/mk/arms/left.urdf"],
    poses: ["ready"],
    mode: "jog",
    robotLoaded: true,
    hasInertia: true,
    urdfPath: "/fx/showcase6.urdf",
    actions: {
      openUrdf: noop,
      openPath: noop,
      setMode: noop,
      planHome: noop,
      planToPose: noop,
      driveHome: noop,
      gravityDrop: noop,
      planRrtHome: noop,
      checkCollision: noop,
      runGraph: noop,
      validateGraph: noop,
    },
    ...over,
  };
}

function byId(cmds: Command[], id: string): Command {
  const c = cmds.find((x) => x.id === id);
  if (!c) throw new Error(`missing command: ${id}`);
  return c;
}

describe("buildCommands", () => {
  it("lists every sample fixture and every recent file", () => {
    const cmds = buildCommands(mkCtx());
    const titles = cmds.map((c) => c.title);
    expect(titles).toContain("Load sample: showcase6");
    expect(titles).toContain("Load sample: planar2");
    const rec = byId(cmds, "robot.recent./home/mk/arms/left.urdf");
    expect(rec.title).toBe("Open recent: left.urdf");
    expect(rec.hint).toBe("/home/mk/arms/left.urdf"); // full path in the hint
  });

  it("records recents but never samples when run", () => {
    const calls: [string, boolean][] = [];
    const ctx = mkCtx();
    ctx.actions.openPath = (p, r) => calls.push([p, r]);
    const cmds = buildCommands(ctx);
    byId(cmds, "robot.sample./fx/planar2.urdf").run();
    byId(cmds, "robot.recent./home/mk/arms/left.urdf").run();
    expect(calls).toEqual([
      ["/fx/planar2.urdf", false],
      ["/home/mk/arms/left.urdf", true],
    ]);
  });

  it("disables the active mode and enables the others", () => {
    const cmds = buildCommands(mkCtx({ mode: "motion" }));
    expect(byId(cmds, "mode.motion").enabled).toBe(false);
    expect(byId(cmds, "mode.motion").hint).toBe("active");
    expect(byId(cmds, "mode.jog").enabled).toBe(true);
    expect(byId(cmds, "mode.graph").enabled).toBe(true);
  });

  it("gates Simulate mode on inertial data", () => {
    const cmds = buildCommands(mkCtx({ hasInertia: false }));
    expect(byId(cmds, "mode.simulate").enabled).toBe(false);
    expect(byId(cmds, "mode.simulate").hint).toBe("no inertial data");
  });

  it("disables robot-gated commands (with a hint) when no robot is loaded", () => {
    const cmds = buildCommands(mkCtx({ robotLoaded: false, urdfPath: null }));
    for (const id of ["mode.jog", "motion.home", "sim.drop", "graph.run"]) {
      expect(byId(cmds, id).enabled).toBe(false);
      expect(byId(cmds, id).hint).toBe("no robot loaded");
    }
    expect(byId(cmds, "robot.reload").enabled).toBe(false);
    expect(byId(cmds, "robot.open").enabled).toBe(true); // Open… is always live
  });

  it("gates mode-scoped actions on the current mode", () => {
    const jog = buildCommands(mkCtx()); // jog: motion yes, sim/graph no
    expect(byId(jog, "motion.home").enabled).toBe(true);
    expect(byId(jog, "motion.pose.ready").enabled).toBe(true);
    expect(byId(jog, "sim.drop").enabled).toBe(false);
    expect(byId(jog, "graph.run").enabled).toBe(false);
    const sim = buildCommands(mkCtx({ mode: "simulate" }));
    expect(byId(sim, "sim.drop").enabled).toBe(true);
    expect(byId(sim, "sim.plan").enabled).toBe(true);
    expect(byId(sim, "motion.home").enabled).toBe(false);
    const graph = buildCommands(mkCtx({ mode: "graph" }));
    expect(byId(graph, "graph.run").enabled).toBe(true);
    expect(byId(graph, "graph.validate").enabled).toBe(true);
  });

  it("hints ⌘1…⌘4 in ModeTabs order on enabled mode switches", () => {
    const cmds = buildCommands(mkCtx({ mode: "graph" }));
    MODE_TABS.forEach((t, i) => {
      if (t.id !== "graph") expect(byId(cmds, `mode.${t.id}`).hint).toBe(`⌘${i + 1}`);
    });
  });
});

describe("filterCommands", () => {
  const mk = (title: string, section: Command["section"] = "Robot"): Command => ({
    id: title,
    title,
    section,
    enabled: true,
    run: noop,
  });

  it("empty (or blank) query returns every command in stable build order", () => {
    const cmds = buildCommands(mkCtx());
    expect(filterCommands("", cmds).map((c) => c.id)).toEqual(cmds.map((c) => c.id));
    expect(filterCommands("   ", cmds)).toHaveLength(cmds.length);
  });

  it("ranks prefix > word-boundary > scattered", () => {
    const cmds = [mk("rerun last"), mk("dry run"), mk("run tests")];
    expect(filterCommands("run", cmds).map((c) => c.title)).toEqual([
      "run tests", // prefix
      "dry run", // word boundary
      "rerun last", // scattered (mid-word)
    ]);
  });

  it("matches case-insensitively in both directions", () => {
    const cmds = [mk("Open URDF…")];
    expect(filterCommands("open urdf", cmds)).toHaveLength(1);
    expect(filterCommands("OPEN", cmds)).toHaveLength(1);
  });

  it("matches scattered subsequences but not shuffled letters", () => {
    const cmds = [mk("Load sample: planar2")];
    expect(filterCommands("lsp2", cmds)).toHaveLength(1); // L…s…p…2 in order
    expect(filterCommands("2planar", cmds)).toHaveLength(0); // out of order
  });

  it("breaks rank ties by section order, then title", () => {
    const cmds = [
      mk("Run graph", "Graph"),
      mk("Run gravity drop", "Simulate"),
      mk("Run b", "Simulate"),
    ];
    expect(filterCommands("run", cmds).map((c) => c.title)).toEqual([
      "Run b",
      "Run gravity drop",
      "Run graph",
    ]);
  });

  it("drops non-matching commands entirely", () => {
    expect(filterCommands("zzz", [mk("Open URDF…")])).toHaveLength(0);
  });
});

describe("baseName", () => {
  it("takes the last segment of unix and windows paths", () => {
    expect(baseName("/a/b/arm.urdf")).toBe("arm.urdf");
    expect(baseName("C:\\bots\\arm.urdf")).toBe("arm.urdf");
    expect(baseName("bare.urdf")).toBe("bare.urdf");
  });
});
