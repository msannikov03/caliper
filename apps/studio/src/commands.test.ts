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
    datasetLoaded: false,
    contactEngine: false, // default pins the no-mujoco build baseline
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
      contactDrop: noop,
      contactHold: noop,
      contactDriveHome: noop,
      runGraph: noop,
      validateGraph: noop,
      duplicateSelection: noop,
      fitGraphView: noop,
      exportGraph: noop,
      importGraph: noop,
      openDataset: noop,
      refreshDataset: noop,
      showTour: noop,
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

  it("gates the graph editor commands (duplicate/fit/export/import) on graph mode", () => {
    const ids = ["graph.duplicate", "graph.fit", "graph.export", "graph.import"];
    const jog = buildCommands(mkCtx()); // jog mode → all four gated
    for (const id of ids) {
      expect(byId(jog, id).enabled).toBe(false);
      expect(byId(jog, id).hint).toBe("switch to Graph mode");
    }
    const graph = buildCommands(mkCtx({ mode: "graph" }));
    for (const id of ids) expect(byId(graph, id).enabled).toBe(true);
    expect(byId(graph, "graph.duplicate").hint).toBe("⌘D"); // shortcut surfaces
    const bare = buildCommands(mkCtx({ mode: "graph", robotLoaded: false }));
    for (const id of ids) expect(byId(bare, id).hint).toBe("no robot loaded");
  });

  it("dispatches the graph editor actions when run", () => {
    const calls: string[] = [];
    const ctx = mkCtx({ mode: "graph" });
    ctx.actions.duplicateSelection = () => calls.push("dup");
    ctx.actions.fitGraphView = () => calls.push("fit");
    ctx.actions.exportGraph = () => calls.push("export");
    ctx.actions.importGraph = () => calls.push("import");
    const cmds = buildCommands(ctx);
    byId(cmds, "graph.duplicate").run();
    byId(cmds, "graph.fit").run();
    byId(cmds, "graph.export").run();
    byId(cmds, "graph.import").run();
    expect(calls).toEqual(["dup", "fit", "export", "import"]);
  });

  it("omits the contact-sim commands entirely when the mujoco engine is absent", () => {
    // even in Simulate mode with inertia — the palette must be byte-identical
    // to a pre-contact build (the no-mujoco zero-visual-change pin)
    const cmds = buildCommands(mkCtx({ mode: "simulate" }));
    expect(cmds.some((c) => c.id.startsWith("sim.contact."))).toBe(false);
  });

  it("gates contact-sim commands on simulate mode + inertia when mujoco is present", () => {
    const ids = ["sim.contact.drop", "sim.contact.hold", "sim.contact.home"];
    const sim = buildCommands(mkCtx({ mode: "simulate", contactEngine: true }));
    for (const id of ids) expect(byId(sim, id).enabled).toBe(true);
    const jog = buildCommands(mkCtx({ contactEngine: true })); // jog mode
    for (const id of ids) {
      expect(byId(jog, id).enabled).toBe(false);
      expect(byId(jog, id).hint).toBe("switch to Simulate mode");
    }
    const noDyn = buildCommands(mkCtx({ mode: "simulate", contactEngine: true, hasInertia: false }));
    for (const id of ids) {
      expect(byId(noDyn, id).enabled).toBe(false);
      expect(byId(noDyn, id).hint).toBe("no inertial data");
    }
  });

  it("dispatches the contact-sim actions when run", () => {
    const calls: string[] = [];
    const ctx = mkCtx({ mode: "simulate", contactEngine: true });
    ctx.actions.contactDrop = () => calls.push("drop");
    ctx.actions.contactHold = () => calls.push("hold");
    ctx.actions.contactDriveHome = () => calls.push("home");
    const cmds = buildCommands(ctx);
    byId(cmds, "sim.contact.drop").run();
    byId(cmds, "sim.contact.hold").run();
    byId(cmds, "sim.contact.home").run();
    expect(calls).toEqual(["drop", "hold", "home"]);
  });

  it("hints ⌘1…⌘4 in ModeTabs order on enabled mode switches", () => {
    const cmds = buildCommands(mkCtx({ mode: "graph" }));
    MODE_TABS.forEach((t, i) => {
      if (t.id !== "graph") expect(byId(cmds, `mode.${t.id}`).hint).toBe(`⌘${i + 1}`);
    });
  });

  it("Show tour is always enabled — even with no robot and no dataset", () => {
    const bare = buildCommands(
      mkCtx({ robotLoaded: false, urdfPath: null, datasetLoaded: false, hasInertia: false }),
    );
    const tour = byId(bare, "help.tour");
    expect(tour.enabled).toBe(true);
    expect(tour.section).toBe("Help");
  });

  it("dispatches the tour action when run (and never another)", () => {
    const calls: string[] = [];
    const ctx = mkCtx();
    ctx.actions.showTour = () => calls.push("tour");
    byId(buildCommands(ctx), "help.tour").run();
    expect(calls).toEqual(["tour"]);
  });

  it("filterCommands finds Show tour by prefix and subsequence", () => {
    const cmds = buildCommands(mkCtx());
    expect(filterCommands("show tour", cmds)[0].id).toBe("help.tour");
    expect(filterCommands("tour", cmds).some((c) => c.id === "help.tour")).toBe(true);
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
