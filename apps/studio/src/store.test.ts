// Headless unit tests for Studio store logic.
//
// Tested without rendering, without WebGL, and without a live Tauri process.
// Tauri's invoke() is replaced by vi.fn() so async store actions run headlessly.
// @xyflow/react utility functions are replaced with simple inline implementations
// so the xyflow package itself never runs in jsdom.
//
// Coverage:
//  - _reqId latest-wins: stale FK replies can't clobber newer state
//  - onGraphConnect: type-incompatible wire, cycle guard, one-feeder-per-input
//  - bumpNodeSeq / loadGraph: no duplicate node IDs after graph load
//  - handleGraphError: validation branch, node branch, plain-string fallback
//  - _execGraph (_graphRunId latest-wins): stale run result skipped
//  - runGraph success: stale traj cleared when result has no trajectory
//  - duplicateGraphSelection: fresh ids, +24/+24, deep params, edges untouched
//  - deleteGraphSelection: node removal takes its edges; edge-only removal
//  - exportGraph / importGraph: save_graph_file/load_graph_file seam + banner
//  - live session: start/adopt/flush, ended-with-error, stale-event guard,
//    mode switch tears the session down
//  - teleop recording: dataset picked once and continued, save/discard/empty
//    takes, a take dying on the stream alone, finish → open in Data

// vi.mock calls are hoisted before imports by Vitest.
import { vi, describe, it, expect, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

// The live-session event channels. listen() records the store's handler under
// its channel name so a test can push a payload the way the backend would;
// hoisted because the mock factory below is hoisted above the imports.
const { liveHandlers } = vi.hoisted(() => ({
  liveHandlers: {} as Record<string, (e: { payload: unknown }) => void>,
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (name: string, cb: (e: { payload: unknown }) => void) => {
    liveHandlers[name] = cb;
    return () => delete liveHandlers[name];
  }),
}));

// The two native dialogs the store opens: `save` picks a dataset directory for
// a take, `open` picks a task file for openTask().
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(), open: vi.fn() }));

// Replace xyflow utilities with minimal pure implementations.
// applyNodeChanges / applyEdgeChanges are only used in the pass-through change
// handlers (onGraphNodesChange / onGraphEdgesChange) which we don't test here.
vi.mock("@xyflow/react", () => ({
  applyNodeChanges: (_changes: unknown, nodes: unknown[]) => nodes,
  applyEdgeChanges: (_changes: unknown, edges: unknown[]) => edges,
  addEdge: (edge: unknown, edges: unknown[]) => [...edges, edge],
}));

import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  useStore,
  DEFAULT_REC_FPS,
  handleGraphError,
  bumpNodeSeq,
  mergeRecent,
  validateSession,
  clampQ,
  sessionRestorePlan,
  _resetNodeSeq,
  _flushLive,
  _resetLive,
} from "./store";
import type { RobotInfo, TaskDto, TrajectoryDto, StudioState } from "./store";
import type { GripperInfo, LiveStartedDto, LiveStateEvent } from "./sim/live";
import type { SimProp } from "./sim/props";
import { serializeGraph } from "./graph/serialize";
import { defaultParams } from "./graph/spec";
import type { KindName } from "./graph/spec";
import type { CNode, CEdge, Diagnostics, GraphRunResult } from "./graph/types";

const mockInvoke = vi.mocked(invoke);
const mockSaveDialog = vi.mocked(save);
const mockOpenDialog = vi.mocked(open);

// ---- shared fixtures ----

const MOCK_ROBOT: RobotInfo = {
  name: "panda",
  ndof: 2,
  jointNames: ["j0", "j1"],
  jointKinds: ["revolute", "revolute"],
  limits: [
    [-Math.PI, Math.PI],
    [-Math.PI, Math.PI],
  ],
  frames: [
    {
      name: "root",
      parent: -1,
      anchor: -1,
      jointIndex: -1,
      jointKind: null,
      axis: null,
    },
  ],
  tip: 0,
  hasInertia: true,
};

const EMPTY_DIAG: Diagnostics = {
  nodeErrors: [],
  edgeErrors: [],
  topoOrder: [],
  cycle: [],
};

/** Minimal baked trajectory (2 timesteps, 2 DOF, 0 render frames). */
function mockTraj(): TrajectoryDto {
  return {
    kind: "moveJ",
    duration: 1,
    ndof: 2,
    dt: 1,
    times: [0, 1],
    q: [
      [0, 0],
      [1, 1],
    ],
    qd: [
      [0, 0],
      [0, 0],
    ],
    tipPath: [
      [0, 0, 0],
      [0.1, 0, 0],
    ],
    frames: [[], []], // no render frames in test
    ok: true,
    reached: 1,
    maxJerkRatio: 0,
  };
}

function makeNode(kind: KindName, id: string): CNode {
  return {
    id,
    type: kind,
    position: { x: 0, y: 0 },
    data: { kind, params: defaultParams(kind, 2), status: "idle" },
  };
}

/** Reset store to clean baseline before each test. */
const STORE_RESET = {
  robot: null as RobotInfo | null,
  q: [] as number[],
  frames: [] as number[][],
  report: null,
  loading: false,
  error: null,
  ikOk: null,
  ikResidual: null,
  _reqId: 0,
  _analyzeReqId: 0,
  traj: null as TrajectoryDto | null,
  poses: [],
  playing: false,
  playhead: 0,
  mode: "jog" as const,
  simTraj: null,
  simGravity: true,
  simDamping: 0.2,
  simTorque: [],
  collision: null,
  graphNodes: [] as CNode[],
  graphEdges: [] as CEdge[],
  graphScopes: [],
  graphLive: false,
  graphBanner: null as string | null,
  graphSaved: [] as string[],
  graphName: "",
  _graphRunId: 0,
  recentUrdfs: [] as string[],
  simEngine: "builtin" as const,
  simEngines: ["builtin"],
  simProps: [],
  task: null,
  live: null,
  livePropPoses: [] as number[][],
  liveTarget: [] as number[],
  liveJoint: 0,
  liveDriving: false,
  liveRec: null,
  liveRecTask: "teleop",
  liveRecFps: 50,
  liveRecHint: null,
  liveRecDone: null,
};

beforeEach(() => {
  vi.clearAllMocks();
  useStore.setState(STORE_RESET);
  _resetNodeSeq();
  _resetLive(); // module-level stream state outlives the store otherwise
});

// ---- mergeRecent (recents dedupe / cap / most-recent-first) ----

describe("mergeRecent — recents pure logic", () => {
  it("puts the newest path first and de-dupes an existing entry", () => {
    expect(mergeRecent(["/a.urdf", "/b.urdf", "/c.urdf"], "/b.urdf")).toEqual([
      "/b.urdf",
      "/a.urdf",
      "/c.urdf",
    ]);
  });

  it("re-adding the current head is a no-op in length and order", () => {
    const list = ["/a.urdf", "/b.urdf", "/c.urdf"];
    const out = mergeRecent(list, "/a.urdf");
    expect(out).toEqual(list);
    expect(out).toHaveLength(3);
  });

  it("caps at 8, dropping the oldest entry", () => {
    const list = ["/1", "/2", "/3", "/4", "/5", "/6", "/7", "/8"];
    expect(mergeRecent(list, "/9")).toEqual([
      "/9",
      "/1",
      "/2",
      "/3",
      "/4",
      "/5",
      "/6",
      "/7",
    ]);
  });

  it("a brand-new path prepends without duplicates", () => {
    const out = mergeRecent(["/a"], "/b");
    expect(out).toEqual(["/b", "/a"]);
  });
});

// ---- session persistence (validate / clamp / restore decision) ----

describe("validateSession — stored-session shape check", () => {
  const GOOD = { urdfPath: "/robots/arm.urdf", mode: "motion", q: [0.1, -0.2] };

  it("accepts a well-formed session and round-trips exactly the three fields", () => {
    const s = validateSession({ ...GOOD, extra: "ignored" });
    expect(s).toEqual(GOOD);
  });

  it("rejects non-objects (null, undefined, string, number, array)", () => {
    for (const raw of [null, undefined, "session", 42, [GOOD]]) {
      expect(validateSession(raw)).toBeNull();
    }
  });

  it("rejects a missing / non-string / empty urdfPath", () => {
    expect(validateSession({ ...GOOD, urdfPath: undefined })).toBeNull();
    expect(validateSession({ ...GOOD, urdfPath: 7 })).toBeNull();
    expect(validateSession({ ...GOOD, urdfPath: "" })).toBeNull();
  });

  it("rejects a mode that is not a real StudioMode", () => {
    expect(validateSession({ ...GOOD, mode: "fly" })).toBeNull();
    expect(validateSession({ ...GOOD, mode: 3 })).toBeNull();
    expect(validateSession({ ...GOOD, mode: undefined })).toBeNull();
  });

  it("rejects q that is not an array of finite numbers", () => {
    expect(validateSession({ ...GOOD, q: "0,1" })).toBeNull();
    expect(validateSession({ ...GOOD, q: [0, "1"] })).toBeNull();
    expect(validateSession({ ...GOOD, q: [0, NaN] })).toBeNull();
    expect(validateSession({ ...GOOD, q: [0, Infinity] })).toBeNull();
    expect(validateSession({ ...GOOD, q: undefined })).toBeNull();
  });

  it("accepts every real StudioMode and an empty q", () => {
    for (const mode of ["jog", "motion", "simulate", "graph"]) {
      expect(validateSession({ ...GOOD, mode })).not.toBeNull();
    }
    expect(validateSession({ ...GOOD, q: [] })).toEqual({ ...GOOD, q: [] });
  });
});

describe("clampQ — per-joint limit clamp", () => {
  it("clamps below lo and above hi, passes in-range values through", () => {
    const limits: ([number, number] | null)[] = [
      [-1, 1],
      [-1, 1],
      [-1, 1],
    ];
    expect(clampQ([-5, 0.5, 5], limits)).toEqual([-1, 0.5, 1]);
  });

  it("a null limit leaves the joint unbounded", () => {
    expect(clampQ([99, -99], [null, [-1, 1]])).toEqual([99, -1]);
  });
});

describe("sessionRestorePlan — restore decision", () => {
  const sess = (q: number[], mode: "jog" | "motion" | "simulate" | "graph") => ({
    urdfPath: "/a.urdf",
    mode,
    q,
  });

  it("q-length mismatch → q ignored, mode still restored", () => {
    const plan = sessionRestorePlan(sess([0.1, 0.2, 0.3], "motion"), MOCK_ROBOT); // ndof 2
    expect(plan.q).toBeNull();
    expect(plan.mode).toBe("motion");
  });

  it("simulate without inertia → mode ignored, q still restored (clamped)", () => {
    const noInertia = { ...MOCK_ROBOT, hasInertia: false };
    const plan = sessionRestorePlan(sess([10, -10], "simulate"), noInertia);
    expect(plan.mode).toBeNull();
    expect(plan.q).toEqual([Math.PI, -Math.PI]); // clamped to ±π limits
  });

  it("simulate WITH inertia and matching q → both restored", () => {
    const plan = sessionRestorePlan(sess([0.5, -0.5], "simulate"), MOCK_ROBOT);
    expect(plan.mode).toBe("simulate");
    expect(plan.q).toEqual([0.5, -0.5]);
  });
});

// ---- _reqId latest-wins (FK / IK reply guard) ----

describe("refreshFrames — _reqId latest-wins", () => {
  it("a stale async reply for an older reqId is ignored; only the newer result lands", async () => {
    useStore.setState({ robot: MOCK_ROBOT, q: [0, 0], frames: [] });

    let resolve1!: (v: number[][]) => void;
    let resolve2!: (v: number[][]) => void;
    const p1 = new Promise<number[][]>((r) => {
      resolve1 = r;
    });
    const p2 = new Promise<number[][]>((r) => {
      resolve2 = r;
    });

    // 1st call → p1 (controlled, stale)
    // 2nd call → p2 (controlled, fresh)
    // Any subsequent invoke (for analyze) → null (no-op)
    mockInvoke
      .mockReturnValueOnce(p1)
      .mockReturnValueOnce(p2)
      .mockResolvedValue(null);

    const STALE: number[][] = [Array<number>(16).fill(9)];
    const FRESH: number[][] = [Array<number>(16).fill(1)];

    const run1 = useStore.getState().refreshFrames(); // _reqId → 1
    const run2 = useStore.getState().refreshFrames(); // _reqId → 2

    expect(useStore.getState()._reqId).toBe(2);

    // Resolve the STALE (older) reply first.
    resolve1(STALE);
    await run1;
    // Guard: _reqId (2) !== reqId (1) → bail, frames unchanged.
    expect(useStore.getState().frames).toEqual([]);

    // Resolve the FRESH (newer) reply.
    resolve2(FRESH);
    await run2;
    // Guard passes: _reqId (2) === reqId (2) → frames updated.
    expect(useStore.getState().frames).toEqual(FRESH);
  });
});

// ---- onGraphConnect ----

describe("onGraphConnect — wire validation", () => {
  it("rejects a type-incompatible wire (Config → Pose port)", () => {
    // startConfig emits Config; moveL.goal accepts Pose → incompatible.
    useStore.setState({
      graphNodes: [makeNode("startConfig", "sc_0"), makeNode("moveL", "ml_0")],
      graphEdges: [],
      graphBanner: null,
    });

    useStore.getState().onGraphConnect({
      source: "sc_0",
      target: "ml_0",
      sourceHandle: "config", // Config
      targetHandle: "goal", // expects Pose
    });

    expect(useStore.getState().graphEdges).toHaveLength(0);
    expect(useStore.getState().graphBanner).toContain("Incompatible");
  });

  it("rejects an edge that would create a cycle (BFS guard)", () => {
    // ik_a.config → ik_b.seed already exists.
    // Trying to add ik_b.config → ik_a.seed creates ik_a→ik_b→ik_a.
    const existingEdge: CEdge = {
      id: "e_a_b",
      source: "ik_a",
      target: "ik_b",
      sourceHandle: "config",
      targetHandle: "seed",
    };
    useStore.setState({
      graphNodes: [makeNode("ik", "ik_a"), makeNode("ik", "ik_b")],
      graphEdges: [existingEdge],
      graphBanner: null,
    });

    useStore.getState().onGraphConnect({
      source: "ik_b",
      target: "ik_a",
      sourceHandle: "config", // Config (ik output)
      targetHandle: "seed", // accepts Config (ik optional seed)
    });

    // Edge count unchanged, cycle banner set.
    expect(useStore.getState().graphEdges).toHaveLength(1);
    expect(useStore.getState().graphBanner).toContain("cycle");
  });

  it("replaces an existing feeder into the same input port (one-feeder rule)", () => {
    // sc_0.config → ik_0.seed already exists; connect sc_1.config → ik_0.seed.
    // The old feeder must be removed and only the new one kept.
    const oldFeeder: CEdge = {
      id: "e_old",
      source: "sc_0",
      target: "ik_0",
      sourceHandle: "config",
      targetHandle: "seed",
    };
    useStore.setState({
      graphNodes: [
        makeNode("startConfig", "sc_0"),
        makeNode("startConfig", "sc_1"),
        makeNode("ik", "ik_0"),
      ],
      graphEdges: [oldFeeder],
      graphBanner: null,
    });

    useStore.getState().onGraphConnect({
      source: "sc_1",
      target: "ik_0",
      sourceHandle: "config",
      targetHandle: "seed",
    });

    const edges = useStore.getState().graphEdges;
    const feeders = edges.filter(
      (e) => e.target === "ik_0" && e.targetHandle === "seed",
    );
    expect(feeders).toHaveLength(1);
    expect(feeders[0].source).toBe("sc_1");
    expect(useStore.getState().graphBanner).toBeNull();
  });

  it("a valid compatible wire is accepted and banner is cleared", () => {
    useStore.setState({
      graphNodes: [makeNode("startConfig", "sc_0"), makeNode("moveJ", "mj_0")],
      graphEdges: [],
      graphBanner: "stale error",
    });

    useStore.getState().onGraphConnect({
      source: "sc_0",
      target: "mj_0",
      sourceHandle: "config",
      targetHandle: "start",
    });

    expect(useStore.getState().graphEdges).toHaveLength(1);
    expect(useStore.getState().graphBanner).toBeNull();
  });
});

// ---- bumpNodeSeq via loadGraph ----

describe("bumpNodeSeq — no duplicate IDs after loadGraph", () => {
  it("addGraphNode after loadGraph produces an ID not present in the loaded set", async () => {
    // Simulated stored graph: nodes have high base-36 seq suffixes (5 and 8).
    // Without bumpNodeSeq nodeSeq would be at 0 and generate startConfig_0,
    // startConfig_1, … eventually startConfig_5 → collision!
    const storedJson = JSON.stringify({
      nodes: [
        { id: "startConfig_5", kind: { type: "startConfig", q: [0, 0] } },
        { id: "moveJ_8", kind: { type: "moveJ" } },
      ],
      edges: [],
      metadata: { name: "test-graph" },
    });

    mockInvoke.mockResolvedValueOnce(storedJson); // load_graph response

    useStore.setState({ robot: MOCK_ROBOT });
    await useStore.getState().loadGraph("test-graph");

    const loadedIds = new Set(useStore.getState().graphNodes.map((n) => n.id));
    expect(loadedIds).toContain("startConfig_5");
    expect(loadedIds).toContain("moveJ_8");

    // Adding a node of the SAME kind as a loaded node — this is the collision
    // risk bumpNodeSeq prevents.
    useStore.getState().addGraphNode("startConfig");

    const allNodes = useStore.getState().graphNodes;
    const allIds = allNodes.map((n) => n.id);
    const unique = new Set(allIds);
    expect(unique.size).toBe(allIds.length); // no duplicates
    // The new startConfig node must not collide with the loaded one.
    const newId = allIds.find((id) => id !== "startConfig_5" && id !== "moveJ_8")!;
    expect(newId).toBeDefined();
    expect(newId).not.toBe("startConfig_5");
  });

  it("bumpNodeSeq directly advances nodeSeq past the max loaded suffix", () => {
    _resetNodeSeq(); // nodeSeq = 0
    const nodes: CNode[] = [
      makeNode("startConfig", "startConfig_a"), // base-36 'a' = 10
      makeNode("moveJ", "moveJ_f"), // base-36 'f' = 15
    ];
    bumpNodeSeq(nodes);

    // After bump, next IDs must not reuse 'a' or 'f' suffixes.
    useStore.setState({ robot: MOCK_ROBOT, graphNodes: nodes, graphEdges: [] });
    useStore.getState().addGraphNode("startConfig");
    useStore.getState().addGraphNode("moveJ");

    const newIds = useStore
      .getState()
      .graphNodes.filter(
        (n) => n.id !== "startConfig_a" && n.id !== "moveJ_f",
      )
      .map((n) => n.id);

    for (const id of newIds) {
      const suffix = id.slice(id.lastIndexOf("_") + 1);
      const seq = parseInt(suffix, 36);
      expect(seq).toBeGreaterThan(15); // must be past 'f' (15)
    }
  });
});

// ---- duplicate selection (⌘D) ----

describe("duplicateGraphSelection — clone semantics", () => {
  it("clones only the selected node: new id, +24/+24 offset, deep params, edges untouched", () => {
    const a = { ...makeNode("planRrt", "planRrt_0"), selected: true };
    a.data.params = { ...a.data.params, boxes: [[[0, 0, 0], [0.1, 0.1, 0.1]]] };
    const b = makeNode("moveJ", "moveJ_1");
    const e: CEdge = { id: "e0", source: "planRrt_0", target: "moveJ_1" };
    useStore.setState({ graphNodes: [a, b], graphEdges: [e] });

    useStore.getState().duplicateGraphSelection();

    const s = useStore.getState();
    expect(s.graphNodes).toHaveLength(3);
    const clone = s.graphNodes[2];
    expect(clone.id).not.toBe("planRrt_0");
    expect(clone.data.kind).toBe("planRrt");
    expect(clone.position).toEqual({ x: 24, y: 24 }); // +24/+24 from (0,0)
    expect(clone.data.params).toEqual(a.data.params);
    // params are DEEP-copied: mutating the clone's boxes leaves the original alone
    (clone.data.params.boxes as number[][][])[0][0][0] = 99;
    expect((a.data.params.boxes as number[][][])[0][0][0]).toBe(0);
    // edges are NOT cloned
    expect(s.graphEdges).toHaveLength(1);
    // the selection moves to the clone (chained ⌘D duplicates the copies)
    expect(s.graphNodes[0].selected).toBe(false);
    expect(clone.selected).toBe(true);
  });

  it("mints ids past loaded suffixes — never collides with existing ids", () => {
    const n = { ...makeNode("startConfig", "startConfig_5"), selected: true };
    bumpNodeSeq([n]); // the loadGraph re-seed path
    useStore.setState({ graphNodes: [n], graphEdges: [] });
    useStore.getState().duplicateGraphSelection();
    useStore.getState().duplicateGraphSelection(); // chained ⌘D → clone-of-clone
    const ids = useStore.getState().graphNodes.map((x) => x.id);
    expect(new Set(ids).size).toBe(3); // all unique
    for (const id of ids.filter((i) => i !== "startConfig_5")) {
      expect(parseInt(id.slice(id.lastIndexOf("_") + 1), 36)).toBeGreaterThan(5);
    }
  });

  it("is a no-op when nothing is selected", () => {
    useStore.setState({ graphNodes: [makeNode("moveJ", "moveJ_0")], graphEdges: [] });
    useStore.getState().duplicateGraphSelection();
    expect(useStore.getState().graphNodes).toHaveLength(1);
  });
});

// ---- delete selection (⌫/⌦ toolbar path) ----

describe("deleteGraphSelection — selection removal", () => {
  it("deleting a selected node drops the edges riding on it", () => {
    const a = { ...makeNode("startConfig", "s0"), selected: true };
    const b = makeNode("moveJ", "m0");
    const c = makeNode("view", "v0");
    const edges: CEdge[] = [
      { id: "e0", source: "s0", target: "m0" },
      { id: "e1", source: "m0", target: "v0" },
    ];
    useStore.setState({ graphNodes: [a, b, c], graphEdges: edges });
    useStore.getState().deleteGraphSelection();
    const s = useStore.getState();
    expect(s.graphNodes.map((n) => n.id)).toEqual(["m0", "v0"]);
    expect(s.graphEdges.map((e) => e.id)).toEqual(["e1"]); // e0 rode on s0
  });

  it("deletes a selected edge alone, leaving both endpoint nodes intact", () => {
    const a = makeNode("startConfig", "s0");
    const b = makeNode("moveJ", "m0");
    useStore.setState({
      graphNodes: [a, b],
      graphEdges: [{ id: "e0", source: "s0", target: "m0", selected: true }],
    });
    useStore.getState().deleteGraphSelection();
    expect(useStore.getState().graphNodes).toHaveLength(2);
    expect(useStore.getState().graphEdges).toHaveLength(0);
  });

  it("is a no-op with no selection", () => {
    const nodes = [makeNode("moveJ", "m0")];
    useStore.setState({ graphNodes: nodes, graphEdges: [] });
    useStore.getState().deleteGraphSelection();
    expect(useStore.getState().graphNodes).toBe(nodes); // untouched reference
  });
});

// ---- file export / import (save_graph_file / load_graph_file seam) ----

describe("exportGraph / importGraph — graph file round-trip seam", () => {
  it("exportGraph writes the CURRENT canvas via save_graph_file", async () => {
    const node = makeNode("moveJ", "mj0");
    useStore.setState({
      robot: MOCK_ROBOT,
      graphNodes: [node],
      graphEdges: [],
      graphName: "wave",
    });
    mockInvoke.mockResolvedValueOnce(undefined);
    await useStore.getState().exportGraph("/tmp/wave.caliper-graph.json");
    expect(mockInvoke).toHaveBeenCalledWith("save_graph_file", {
      path: "/tmp/wave.caliper-graph.json",
      graphJson: serializeGraph([node], [], "wave", "panda"),
    });
    expect(useStore.getState().graphBanner).toBeNull();
  });

  it("surfaces a backend write/validation error in the banner", async () => {
    useStore.setState({ robot: MOCK_ROBOT });
    mockInvoke.mockRejectedValueOnce("invalid graph JSON: boom");
    await useStore.getState().exportGraph("/tmp/x.json");
    expect(useStore.getState().graphBanner).toContain("invalid graph JSON");
  });

  it("importGraph adopts the parsed doc and re-seeds node ids", async () => {
    mockInvoke.mockResolvedValueOnce(
      JSON.stringify({
        nodes: [{ id: "startConfig_7", kind: { type: "startConfig", q: [0, 0] } }],
        edges: [],
        metadata: { name: "imported" },
      }),
    );
    useStore.setState({ robot: MOCK_ROBOT });
    await useStore.getState().importGraph("/tmp/imported.caliper-graph.json");
    const s = useStore.getState();
    expect(s.graphNodes.map((n) => n.id)).toEqual(["startConfig_7"]);
    expect(s.graphName).toBe("imported");
    useStore.getState().addGraphNode("startConfig"); // must not collide with _7
    const ids = useStore.getState().graphNodes.map((n) => n.id);
    expect(new Set(ids).size).toBe(2);
  });

  it("falls back to the file stem when the doc carries no name", async () => {
    mockInvoke.mockResolvedValueOnce(JSON.stringify({ nodes: [], edges: [] }));
    useStore.setState({ robot: MOCK_ROBOT });
    await useStore.getState().importGraph("/data/waves/demo.caliper-graph.json");
    expect(useStore.getState().graphName).toBe("demo");
  });

  it("a malformed doc lands in the banner and leaves the canvas untouched", async () => {
    const before = [makeNode("moveJ", "m0")];
    mockInvoke.mockResolvedValueOnce("{not json"); // backend let garbage through
    useStore.setState({ robot: MOCK_ROBOT, graphNodes: before });
    await useStore.getState().importGraph("/tmp/bad.json");
    expect(useStore.getState().graphBanner).not.toBeNull();
    expect(useStore.getState().graphNodes).toBe(before);
  });
});

// ---- handleGraphError ----

describe("handleGraphError — JSON string dispatch", () => {
  it("parses a JSON string and takes the validation branch", () => {
    const nodes = [makeNode("startConfig", "n0"), makeNode("goalPose", "n1")];
    useStore.setState({
      graphNodes: nodes,
      graphEdges: [],
      graphBanner: null,
    });

    const diag: Diagnostics = {
      nodeErrors: [{ nodeId: "n0", message: "missing connection" }],
      edgeErrors: [],
      topoOrder: [],
      cycle: [],
    };
    const errStr = JSON.stringify({ kind: "validation", diagnostics: diag });
    const set: (p: Partial<StudioState>) => void = (p) =>
      useStore.setState(p);

    handleGraphError(errStr, useStore.getState, set);

    const s = useStore.getState();
    expect(s.graphBanner).not.toBeNull();
    // Banner must mention the erroring node.
    expect(s.graphBanner).toContain("n0");
    // Node status updated to error.
    expect(s.graphNodes.find((n) => n.id === "n0")!.data.status).toBe("error");
    // Sibling node untouched.
    expect(s.graphNodes.find((n) => n.id === "n1")!.data.status).toBe("idle");
  });

  it("takes the node branch on kind:node JSON string", () => {
    const nodes = [makeNode("startConfig", "n0"), makeNode("goalPose", "n1")];
    useStore.setState({ graphNodes: nodes, graphEdges: [], graphBanner: null });

    const errStr = JSON.stringify({
      kind: "node",
      nodeId: "n0",
      message: "exec failed",
    });
    const set: (p: Partial<StudioState>) => void = (p) =>
      useStore.setState(p);

    handleGraphError(errStr, useStore.getState, set);

    const s = useStore.getState();
    expect(s.graphBanner).toBe("node n0: exec failed");
    expect(s.graphNodes.find((n) => n.id === "n0")!.data.status).toBe("error");
    expect(s.graphNodes.find((n) => n.id === "n0")!.data.error).toBe("exec failed");
    expect(s.graphNodes.find((n) => n.id === "n1")!.data.status).toBe("idle");
  });

  it("falls back to plain-string banner for a non-JSON string error", () => {
    useStore.setState({ graphNodes: [], graphEdges: [], graphBanner: null });
    const set: (p: Partial<StudioState>) => void = (p) =>
      useStore.setState(p);

    handleGraphError("network timeout", useStore.getState, set);

    expect(useStore.getState().graphBanner).toBe("network timeout");
  });

  it("falls back to plain-string banner for a non-JSON-object string", () => {
    useStore.setState({ graphNodes: [], graphEdges: [], graphBanner: null });
    const set: (p: Partial<StudioState>) => void = (p) =>
      useStore.setState(p);

    // A JSON string that parses to a primitive (number) — no kind field.
    handleGraphError(JSON.stringify(42), useStore.getState, set);

    // Should not crash; banner is set to the string representation.
    expect(useStore.getState().graphBanner).not.toBeNull();
  });
});

// ---- _graphRunId latest-wins ----

describe("_execGraph — _graphRunId latest-wins", () => {
  it("a stale runGraph response for an older run ID is silently dropped", async () => {
    useStore.setState({
      robot: MOCK_ROBOT,
      q: [0, 0],
      graphNodes: [],
      graphEdges: [],
      graphScopes: [],
    });

    // Run 1: slow (we control resolution)
    let resolveRun1!: (v: GraphRunResult) => void;
    const p1 = new Promise<GraphRunResult>((r) => {
      resolveRun1 = r;
    });

    const freshScope = { nodeId: "sc1", signal: "q0", t: [0, 1], y: [0, 1] };
    mockInvoke
      .mockReturnValueOnce(p1) // 1st graph_run → slow
      .mockResolvedValueOnce({
        // 2nd graph_run → fast
        scopes: [freshScope],
        diagnostics: EMPTY_DIAG,
      } satisfies GraphRunResult);

    const run1 = useStore.getState().runGraph(); // _graphRunId → 1
    const run2 = useStore.getState().runGraph(); // _graphRunId → 2

    // run2 resolves immediately via its resolved-value mock.
    await run2;
    expect(useStore.getState().graphScopes).toHaveLength(1);

    // Now settle run1 with a stale (empty) result.
    resolveRun1({ scopes: [], diagnostics: EMPTY_DIAG });
    await run1;

    // run1 guard: _graphRunId (2) !== id (1) → bail → scopes from run2 survive.
    expect(useStore.getState().graphScopes).toHaveLength(1);
    expect(useStore.getState().graphScopes[0].signal).toBe("q0");
  });
});

// ---- runGraph success ----

describe("runGraph success", () => {
  it("clears a stale traj when the graph result has no trajectory", async () => {
    const stale = mockTraj();
    useStore.setState({
      robot: MOCK_ROBOT,
      q: [0, 0],
      graphNodes: [],
      graphEdges: [],
      traj: stale,
      simTraj: null,
    });

    mockInvoke.mockResolvedValueOnce({
      scopes: [],
      diagnostics: EMPTY_DIAG,
      // no trajectory field → undefined → null in _execGraph
    } satisfies GraphRunResult);

    await useStore.getState().runGraph();

    expect(useStore.getState().traj).toBeNull();
    expect(useStore.getState().simTraj).toBeNull();
    expect(useStore.getState().playing).toBe(false);
  });

  it("sets traj and starts playback when the graph result contains a trajectory", async () => {
    useStore.setState({
      robot: MOCK_ROBOT,
      q: [0, 0],
      graphNodes: [],
      graphEdges: [],
      traj: null,
    });

    const traj = mockTraj();
    mockInvoke.mockResolvedValueOnce({
      trajectory: traj,
      scopes: [],
      diagnostics: EMPTY_DIAG,
    } satisfies GraphRunResult);

    await useStore.getState().runGraph();

    const s = useStore.getState();
    // Trajectory installed and playback clock started.
    expect(s.traj).not.toBeNull();
    expect(s.traj?.kind).toBe("moveJ");
    expect(s.playing).toBe(true);
    expect(s.simTraj).toBeNull();
  });
});

// ---- live sim session (store side of src/sim/live.ts) ----

describe("live session — store wiring", () => {
  /** A gripper channel on joint 1 of MOCK_ROBOT: open 0.04 → closed 0.0. */
  const CHANNEL: GripperInfo = {
    joint: "gripper",
    index: 1,
    openTarget: 0.04,
    closedTarget: 0.008,
  };

  function mockStarted(over: Partial<LiveStartedDto> = {}): LiveStartedDto {
    return {
      sessionId: 1,
      engine: "mujoco",
      h: 0.002,
      emitHz: 59.5,
      ndof: 2,
      props: [],
      gripper: null,
      ...over,
    };
  }

  function mockState(over: Partial<LiveStateEvent> = {}): LiveStateEvent {
    return {
      sessionId: 1,
      tick: 300,
      t: 0.6,
      q: [0.1, 0.2],
      qd: [0, 0],
      frames: [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
      tip: [0.3, 0, 0.2],
      ncon: 2,
      props: [[0, 0, 0.05, 1, 0, 0, 0]],
      paused: false,
      target: [0, 0],
      recording: false,
      recFrames: 0,
      ...over,
    };
  }

  /** Only live_start answers with a DTO; every other command resolves empty. */
  function backend(dto: LiveStartedDto) {
    mockInvoke.mockImplementation(async (cmd: string) =>
      cmd === "live_start" ? dto : undefined,
    );
  }

  /** Push one backend event through the recorded listen() handler. */
  function emit(channel: string, payload: unknown) {
    liveHandlers[channel]({ payload });
  }

  beforeEach(() => {
    useStore.setState({
      robot: MOCK_ROBOT,
      q: [0, 0],
      mode: "simulate",
      simEngines: ["builtin", "mujoco"],
      simProps: [],
    });
  });

  it("starts on the engine the panel selected and adopts what the backend returns", async () => {
    useStore.setState({ simEngine: "mujoco" });
    backend(mockStarted({ sessionId: 3 }));

    await useStore.getState().startLive();

    expect(mockInvoke).toHaveBeenCalledWith("live_start", {
      req: { q0: [0, 0], engine: "mujoco", props: [] },
    });
    const s = useStore.getState();
    expect(s.live).toMatchObject({ sessionId: 3, engine: "mujoco", t: 0, paused: false });
    // the session owns the pose: no baked clip survives it
    expect(s.simTraj).toBeNull();
    expect(s.traj).toBeNull();
    expect(s.playing).toBe(false);
  });

  it("falls back to the builtin engine (and sends no props) without mujoco", async () => {
    useStore.setState({ simEngine: "mujoco", simEngines: ["builtin"] });
    backend(mockStarted({ engine: "builtin" }));

    await useStore.getState().startLive();

    expect(mockInvoke).toHaveBeenCalledWith("live_start", {
      req: { q0: [0, 0], engine: "builtin", props: [] },
    });
  });

  it("keeps builtin selectable on a build that HAS the contact engine", async () => {
    // the panel's toggle is the only engine choice — a mujoco build that picks
    // Builtin gets builtin (and its props stay behind, builtin rejects them)
    useStore.setState({ simEngine: "builtin", simProps: [{ name: "box1" } as never] });
    backend(mockStarted({ engine: "builtin" }));

    await useStore.getState().startLive();

    expect(mockInvoke).toHaveBeenCalledWith("live_start", {
      req: { q0: [0, 0], engine: "builtin", props: [] },
    });
  });

  it("refuses a robot with no inertial data and never reaches the backend", async () => {
    useStore.setState({ robot: { ...MOCK_ROBOT, hasInertia: false } });

    await useStore.getState().startLive();

    expect(mockInvoke).not.toHaveBeenCalled();
    expect(useStore.getState().live).toBeNull();
    expect(useStore.getState().error).toBe("this robot has no inertial data");
  });

  it("applies a streamed state event on flush, not on arrival", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    const reqIdBefore = useStore.getState()._reqId;

    emit("live://state", mockState());
    // stashed only — the coalescer owns when it lands
    expect(useStore.getState().q).toEqual([0, 0]);

    _flushLive();

    const s = useStore.getState();
    expect(s.q).toEqual([0.1, 0.2]);
    expect(s.frames).toHaveLength(1);
    expect(s.live).toMatchObject({ t: 0.6, tick: 300, ncon: 2, paused: false });
    expect(s.livePropPoses).toEqual([[0, 0, 0.05, 1, 0, 0, 0]]);
    // fences any in-flight get_frames reply, exactly like _applyTrajAt
    expect(s._reqId).toBeGreaterThan(reqIdBefore);
  });

  it("collapses a burst of events into the last one (latest-wins)", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();

    emit("live://state", mockState({ t: 0.1, q: [1, 1] }));
    emit("live://state", mockState({ t: 0.2, q: [2, 2] }));
    emit("live://state", mockState({ t: 0.3, q: [3, 3] }));
    _flushLive();

    expect(useStore.getState().q).toEqual([3, 3]);
    expect(useStore.getState().live?.t).toBe(0.3);
  });

  it("ignores a state event from a session we already moved past", async () => {
    backend(mockStarted({ sessionId: 5 }));
    await useStore.getState().startLive();

    emit("live://state", mockState({ sessionId: 4, q: [9, 9] }));
    _flushLive();

    const s = useStore.getState();
    expect(s.q).toEqual([0, 0]);
    expect(s.live?.sessionId).toBe(5);
  });

  it("clears the session silently when it ends benignly", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    emit("live://state", mockState());
    _flushLive();

    emit("live://ended", { sessionId: 1, reason: "stopped" });

    const s = useStore.getState();
    expect(s.live).toBeNull();
    expect(s.livePropPoses).toEqual([]);
    expect(s.error).toBeNull();
  });

  it("surfaces the error banner when the session dies on the backend", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();

    emit("live://ended", { sessionId: 1, reason: "error: mujoco step diverged" });

    const s = useStore.getState();
    expect(s.live).toBeNull();
    expect(s.error).toBe("live sim ended — error: mujoco step diverged");
  });

  it("keeps running when an end arrives for an older, superseded session", async () => {
    backend(mockStarted({ sessionId: 6 }));
    await useStore.getState().startLive();

    emit("live://ended", { sessionId: 5, reason: "superseded" });

    expect(useStore.getState().live?.sessionId).toBe(6);
  });

  it("surfaces a failed start and leaves nothing adopted", async () => {
    mockInvoke.mockRejectedValueOnce("no live session available");

    await useStore.getState().startLive();

    const s = useStore.getState();
    expect(s.live).toBeNull();
    expect(s.error).toBe("no live session available");
  });

  it("routes pause/reset to the backend without guessing the new state", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();

    await useStore.getState().pauseLive(true);
    expect(mockInvoke).toHaveBeenCalledWith("live_pause", { paused: true });
    // the flag lands via the state event the transition emits, not optimistically
    expect(useStore.getState().live?.paused).toBe(false);

    await useStore.getState().resetLive();
    expect(mockInvoke).toHaveBeenCalledWith("live_reset", { q0: null });
  });

  it("stops the session when the mode leaves simulate", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();

    useStore.getState().setMode("jog");

    expect(mockInvoke).toHaveBeenCalledWith("live_stop");
  });

  it("tears the session down locally when live_stop itself fails", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockRejectedValueOnce("backend gone");

    await useStore.getState().stopLive();

    const s = useStore.getState();
    expect(s.live).toBeNull();
    expect(s.livePropPoses).toEqual([]);
    expect(s.error).toBe("backend gone");
  });

  // ---- the input layer: every source folds into ONE target per frame ----

  /** Every live_set_target the store has sent, oldest first. */
  function targetCalls(): number[][] {
    return mockInvoke.mock.calls
      .filter((c) => c[0] === "live_set_target")
      .map((c) => (c[1] as { q: number[] }).q);
  }

  /** Install a gamepad for the drive step's navigator.getGamepads() poll. */
  function withPad(axes: number[], buttons: boolean[] = []): void {
    Object.defineProperty(navigator, "getGamepads", {
      configurable: true,
      value: () => [{ axes, buttons: buttons.map((pressed) => ({ pressed })) }],
    });
  }

  /** Drain the microtask queue (the drive step's invokes are fire-and-forget). */
  const settle = () => new Promise((r) => setTimeout(r, 0));

  beforeEach(() => {
    // no pad by default; jsdom has no Gamepad API of its own
    Object.defineProperty(navigator, "getGamepads", { configurable: true, value: () => [] });
  });

  it("sends a slider edit once per frame, carrying the latest value", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockClear();

    useStore.getState().setLiveTargetJoint(1, 0.4);
    useStore.getState().setLiveTargetJoint(1, 0.5);
    expect(targetCalls()).toHaveLength(0); // the frame owns the send, not the event

    _flushLive();

    expect(targetCalls()).toEqual([[0, 0.5]]);
    const s = useStore.getState();
    expect(s.liveTarget).toEqual([0, 0.5]);
    expect(s.liveJoint).toBe(1); // the slider you touched becomes the jog selection
    expect(s.liveDriving).toBe(true);
  });

  it("clamps a slider edit into the joint's limit", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockClear();

    useStore.getState().setLiveTargetJoint(0, 99);
    _flushLive();

    expect(targetCalls()).toEqual([[Math.PI, 0]]);
  });

  it("never re-sends a target nothing moved", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    emit("live://state", mockState());
    mockInvoke.mockClear();

    _flushLive();
    _flushLive();

    expect(targetCalls()).toHaveLength(0);
    expect(useStore.getState().liveDriving).toBe(false);
  });

  it("jogs the selected joint while its key is held, and stops on release", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockClear();
    useStore.getState().selectLiveJoint(1);
    useStore.getState().liveJogKey("=", true);

    _flushLive();

    const [q] = targetCalls();
    expect(q[0]).toBe(0); // only the selected joint moves
    expect(q[1]).toBeGreaterThan(0);
    expect(q[1]).toBeLessThanOrEqual(1.5 / 60); // ≤ one frame at the jog rate

    useStore.getState().liveJogKey("=", false);
    mockInvoke.mockClear();
    _flushLive();
    expect(targetCalls()).toHaveLength(0);
  });

  it("holds a jogged joint at its limit instead of running past it", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockClear();
    useStore.getState().setLiveTargetJoint(0, Math.PI); // pinned at the top stop
    useStore.getState().liveJogKey("=", true);

    _flushLive();

    expect(targetCalls()).toEqual([[Math.PI, 0]]);
    useStore.getState().liveJogKey("=", false);
  });

  it("ignores held keys while the session is frozen", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    emit("live://state", mockState({ paused: true }));
    _flushLive();
    mockInvoke.mockClear();
    useStore.getState().liveJogKey("=", true);

    _flushLive();

    expect(targetCalls()).toHaveLength(0);
    useStore.getState().liveJogKey("=", false);
  });

  it("turns a gamepad button press into the pause it stands for", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockClear();
    withPad([0, 0, 0, 0], [true]);

    _flushLive();
    expect(mockInvoke).toHaveBeenCalledWith("live_pause", { paused: true });

    // held, not re-pressed: the edge already fired
    mockInvoke.mockClear();
    _flushLive();
    expect(mockInvoke).not.toHaveBeenCalledWith("live_pause", expect.anything());
  });

  it("drives the tip through IK on a stick push and adopts the solution", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    emit("live://state", mockState()); // frames + the streamed tip land first
    _flushLive();
    mockInvoke.mockClear();
    mockInvoke.mockImplementation(async (cmd: string) =>
      cmd === "solve_ik_governed" ? { success: true, q: [0.5, -0.5], residual: 1e-7 } : undefined,
    );
    withPad([1, 0, 0, 0]); // left stick hard over: +X in URDF world

    _flushLive();
    await settle();

    const ik = mockInvoke.mock.calls.find((c) => c[0] === "solve_ik_governed");
    expect(ik).toBeDefined();
    const req = (ik as [string, { req: { target: number[]; frame: string; seed: number[] } }])[1]
      .req;
    expect(req.frame).toBe("root"); // MOCK_ROBOT's tip frame
    expect(req.target[12]).toBeGreaterThan(0.3); // pushed out from the streamed tip
    expect(req.target[13]).toBe(0); // untouched axes keep the tip's own pose
    expect(req.target[14]).toBe(0.2);

    _flushLive();
    const sent = targetCalls();
    expect(sent[sent.length - 1]).toEqual([0.5, -0.5]);
    // the IK reply is a TARGET only — the streamed pose is never overwritten
    expect(useStore.getState().q).toEqual([0.1, 0.2]);
  });

  it("never loses the last tip target when a solve is still in flight", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    emit("live://state", mockState());
    _flushLive();
    mockInvoke.mockClear();
    // the first solve hangs; the gizmo's drag-end asks for a second pose while
    // it is still out (the case a plain busy-flag would have dropped)
    let release: () => void = () => {};
    mockInvoke.mockImplementation(async (cmd: string, args?: unknown) => {
      if (cmd !== "solve_ik_governed") return undefined;
      const t = (args as { req: { target: number[] } }).req.target[12];
      if (t === 1) {
        return new Promise((r) => {
          release = () => r({ success: true, q: [1, 1], residual: 0 });
        });
      }
      return { success: true, q: [2, 2], residual: 0 };
    });
    const pose = (x: number) => {
      const m = new Array<number>(16).fill(0);
      m[0] = m[5] = m[10] = m[15] = 1;
      m[12] = x;
      return m;
    };

    void useStore.getState().driveTipLive(pose(1));
    void useStore.getState().driveTipLive(pose(2)); // stashed behind the first
    release();
    await settle();

    _flushLive();
    const sent = targetCalls();
    expect(sent[sent.length - 1]).toEqual([2, 2]);
  });

  it("banners a live_set_target rejection once, not once per frame", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "live_set_target") throw "no live session";
      return undefined;
    });

    useStore.getState().setLiveTargetJoint(0, 0.2);
    _flushLive();
    await settle();
    expect(useStore.getState().error).toBe("no live session");

    useStore.setState({ error: null });
    useStore.getState().setLiveTargetJoint(0, 0.3);
    _flushLive();
    await settle();
    expect(useStore.getState().error).toBeNull(); // same message: already said
  });

  it("re-seeds the hold target from the stream after a reset", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    useStore.getState().setLiveTargetJoint(0, 0.4);
    _flushLive();
    expect(useStore.getState().liveTarget).toEqual([0.4, 0]);

    await useStore.getState().resetLive();
    mockInvoke.mockClear();
    emit("live://state", mockState({ target: [0, 0] }));
    _flushLive();

    // the pre-reset target is forgotten, not re-asserted over the fresh session
    expect(useStore.getState().liveTarget).toEqual([0, 0]);
    expect(targetCalls()).toHaveLength(0);
  });

  it("drops every input scrap when the session ends", async () => {
    backend(mockStarted());
    await useStore.getState().startLive();
    useStore.getState().setLiveTargetJoint(0, 0.4);
    _flushLive();

    emit("live://ended", { sessionId: 1, reason: "stopped" });

    const s = useStore.getState();
    expect(s.liveTarget).toEqual([]);
    expect(s.liveDriving).toBe(false);
    // and a slider that is still on screen for a frame cannot resurrect it
    mockInvoke.mockClear();
    useStore.getState().setLiveTargetJoint(0, 0.9);
    _flushLive();
    expect(targetCalls()).toHaveLength(0);
  });

  // ---- gripper + grasp (B): one channel, one command, one held prop ----

  describe("gripper", () => {
    /** Start a session that FOUND a jaw joint, and forget the start calls. */
    async function startWithGripper(): Promise<void> {
      backend(mockStarted({ gripper: CHANNEL }));
      await useStore.getState().startLive();
      mockInvoke.mockClear(); // the implementation survives; only the log resets
    }

    /** The jaw slot of the last live_set_target the store sent. */
    function lastJaw(): number | undefined {
      const sent = targetCalls();
      return sent[sent.length - 1]?.[CHANNEL.index];
    }

    it("adopts the channel the session reports", async () => {
      await startWithGripper();
      expect(useStore.getState().live?.gripper).toEqual(CHANNEL);
    });

    it("carries no channel when the session found no jaw joint", async () => {
      backend(mockStarted({ gripper: null }));
      await useStore.getState().startLive();
      expect(useStore.getState().live?.gripper).toBeNull();
    });

    it("commands closed, then open, flipping the intent each press", async () => {
      await startWithGripper();

      await useStore.getState().toggleGripper();
      expect(mockInvoke).toHaveBeenCalledWith("live_gripper", { closed: true });
      expect(useStore.getState().live?.gripperState?.closed).toBe(true);

      await useStore.getState().toggleGripper();
      expect(mockInvoke).toHaveBeenLastCalledWith("live_gripper", { closed: false });
      expect(useStore.getState().live?.gripperState?.closed).toBe(false);
    });

    it("does nothing at all on a session with no gripper channel", async () => {
      backend(mockStarted({ gripper: null }));
      await useStore.getState().startLive();
      mockInvoke.mockClear();

      await useStore.getState().toggleGripper();

      expect(mockInvoke).not.toHaveBeenCalledWith("live_gripper", expect.anything());
      expect(useStore.getState().error).toBeNull();
    });

    // THE regression this whole channel hangs on: live_set_target overwrites
    // the WHOLE vector, so a drive mirror that still holds the pre-toggle jaw
    // value drops the grasp on the very next input the human touches.
    it("keeps the closed jaw in the target a later jog ships", async () => {
      await startWithGripper();
      await useStore.getState().toggleGripper();
      expect(useStore.getState().liveTarget[CHANNEL.index]).toBe(CHANNEL.closedTarget);

      useStore.getState().selectLiveJoint(0);
      useStore.getState().liveJogKey("=", true);
      _flushLive();
      useStore.getState().liveJogKey("=", false);

      const sent = targetCalls();
      expect(sent[sent.length - 1][0]).toBeGreaterThan(0); // the jog did move
      expect(lastJaw()).toBe(CHANNEL.closedTarget); // …and the jaw stayed shut
    });

    it("keeps it through a slider edit queued across the toggle", async () => {
      await startWithGripper();
      // the edit is stashed for the next frame; the toggle lands in between
      useStore.getState().setLiveTargetJoint(0, 0.3);
      await useStore.getState().toggleGripper();

      _flushLive();

      expect(targetCalls()).toEqual([[0.3, CHANNEL.closedTarget]]);
    });

    it("keeps it through an IK solution seeded before the toggle", async () => {
      await startWithGripper();
      mockInvoke.mockImplementation(async (cmd: string) =>
        cmd === "solve_ik_governed"
          ? { success: true, q: [0.5, CHANNEL.openTarget], residual: 0 }
          : undefined,
      );
      const pose = new Array<number>(16).fill(0);
      pose[0] = pose[5] = pose[10] = pose[15] = 1;

      // the solve is seeded off the OPEN mirror and lands after the toggle
      await useStore.getState().driveTipLive(pose);
      await useStore.getState().toggleGripper();
      _flushLive();

      expect(lastJaw()).toBe(CHANNEL.closedTarget);
    });

    it("holds the command over a state event that predates it", async () => {
      await startWithGripper();
      await useStore.getState().toggleGripper();
      mockInvoke.mockClear();

      // emitted before live_gripper landed: it still reports the open jaw
      emit("live://state", mockState({ gripper: { closed: false, q: 0.04 }, target: [0, 0.04] }));
      _flushLive();

      // the button must not flip back, and the mirror must not re-open
      expect(useStore.getState().live?.gripperState?.closed).toBe(true);
      useStore.getState().setLiveTargetJoint(0, 0.2);
      _flushLive();
      expect(lastJaw()).toBe(CHANNEL.closedTarget);
    });

    it("adopts the stream's own view of the jaw once it agrees", async () => {
      await startWithGripper();
      await useStore.getState().toggleGripper();
      emit(
        "live://state",
        mockState({ gripper: { closed: true, q: 0.031 }, target: [0, CHANNEL.closedTarget] }),
      );
      _flushLive();
      mockInvoke.mockClear();

      // the jaw is squeezing a prop: commanded closed, measured short of it
      expect(useStore.getState().live?.gripperState).toEqual({ closed: true, q: 0.031 });
      // and a jaw the BACKEND re-opened (not us) lands in the mirror too
      emit("live://state", mockState({ gripper: { closed: false, q: 0.04 }, target: [0, 0.04] }));
      _flushLive();
      expect(useStore.getState().liveTarget[CHANNEL.index]).toBe(CHANNEL.openTarget);
    });

    it("leaves a jaw the human is jogging by hand alone", async () => {
      await startWithGripper();
      useStore.getState().setLiveTargetJoint(CHANNEL.index, 0.02); // hand on the jaw slider
      _flushLive();
      mockInvoke.mockClear();

      // a stale event echoing the pre-jog target must not drag it back
      emit("live://state", mockState({ gripper: { closed: false, q: 0.04 }, target: [0, 0.04] }));
      _flushLive();

      expect(useStore.getState().liveTarget[CHANNEL.index]).toBe(0.02);
    });

    it("commands the jaw while the session is frozen, where no event can confirm it", async () => {
      await startWithGripper();
      emit("live://state", mockState({ paused: true }));
      _flushLive();

      await useStore.getState().toggleGripper();

      // nothing streams while frozen, so the panel would read a stale intent
      // if the command did not paint itself
      expect(useStore.getState().live?.gripperState?.closed).toBe(true);
      // and pressing again flips off the COMMAND, not the last state seen
      await useStore.getState().toggleGripper();
      expect(mockInvoke).toHaveBeenLastCalledWith("live_gripper", { closed: false });
    });

    it("banners a rejected command and moves nothing", async () => {
      await startWithGripper();
      mockInvoke.mockImplementation(async (cmd: string) => {
        if (cmd === "live_gripper") throw "this robot has no gripper channel";
        return undefined;
      });

      await useStore.getState().toggleGripper();

      const s = useStore.getState();
      expect(s.error).toBe("this robot has no gripper channel");
      expect(s.live?.gripperState).toBeNull();
      expect(s.liveTarget).toEqual([0, 0]); // the mirror never learned a jaw value
    });

    it("turns pad X into the same toggle, once per press", async () => {
      await startWithGripper();
      withPad([0, 0, 0, 0], [false, false, true]);

      _flushLive();
      await settle();
      expect(mockInvoke).toHaveBeenCalledWith("live_gripper", { closed: true });

      mockInvoke.mockClear();
      _flushLive(); // still held: the edge already fired
      await settle();
      expect(mockInvoke).not.toHaveBeenCalledWith("live_gripper", expect.anything());
    });

    it("badges the prop the grasp heuristic welded, and un-badges it on release", async () => {
      await startWithGripper();

      emit("live://state", mockState({ gripper: { closed: true, q: 0.031 }, held: "box0" }));
      _flushLive();
      expect(useStore.getState().live?.held).toBe("box0");

      emit("live://state", mockState({ gripper: { closed: false, q: 0.04 }, held: null }));
      _flushLive();
      expect(useStore.getState().live?.held).toBeNull();
    });

    it("holds nothing on a session with no gripper channel (builtin)", async () => {
      backend(mockStarted({ engine: "builtin", gripper: null }));
      await useStore.getState().startLive();

      emit("live://state", mockState());
      _flushLive();

      const s = useStore.getState();
      expect(s.live?.held).toBeNull();
      expect(s.live?.gripperState).toBeNull();
    });
  });

  // ---- teleop recording: takes append episodes to ONE open dataset ----

  describe("recording", () => {
    const ROOT = "/data/teleop_panda";

    /** live_start + the four recording commands; everything else resolves
     *  empty. Per-test overrides ride on mock*Once, which wins over this. */
    function recBackend() {
      mockInvoke.mockImplementation(async (cmd: string) => {
        switch (cmd) {
          case "live_start":
            return mockStarted();
          case "live_record_start":
            return { root: ROOT, fps: 50, recordEvery: 40, episodeIndex: 0 };
          case "live_record_stop":
            return { saved: true, episodeIndex: 0, frames: 120 };
          case "live_record_finish":
            return { root: ROOT, episodes: 1 };
          default:
            return undefined;
        }
      });
    }

    /** The `req` of the last live_record_start, as the backend saw it. */
    function lastStartReq(): { root: string; task: string; fps: number } {
      const calls = mockInvoke.mock.calls.filter((c) => c[0] === "live_record_start");
      return (calls[calls.length - 1] as [string, { req: never }])[1].req;
    }

    beforeEach(() => {
      recBackend();
      mockSaveDialog.mockResolvedValue(ROOT);
    });

    it("picks the dataset directory for the first take and adopts the reply's root", async () => {
      // the dialog's string is only a REQUEST — the reply names where the
      // dataset actually lives, and that is what later takes continue
      mockSaveDialog.mockResolvedValue("/data/as_typed");
      await useStore.getState().startLive();

      await useStore.getState().recordStart("pick the cube", 25);

      expect(mockSaveDialog).toHaveBeenCalledTimes(1);
      expect(lastStartReq()).toEqual({ root: "/data/as_typed", task: "pick the cube", fps: 25 });
      expect(useStore.getState().liveRec).toEqual({
        root: ROOT,
        fps: 50,
        recording: true,
        task: "pick the cube",
        frames: 0,
        episodesSaved: 0,
      });
    });

    it("does nothing when the directory dialog is cancelled", async () => {
      mockSaveDialog.mockResolvedValue(null);
      await useStore.getState().startLive();

      await useStore.getState().recordStart("teleop");

      expect(mockInvoke).not.toHaveBeenCalledWith("live_record_start", expect.anything());
      expect(useStore.getState().liveRec).toBeNull();
    });

    it("never reaches the backend without a session or with a blank label", async () => {
      await useStore.getState().recordStart("teleop"); // no live session
      await useStore.getState().startLive();
      await useStore.getState().recordStart("   ");

      expect(mockInvoke).not.toHaveBeenCalledWith("live_record_start", expect.anything());
      expect(mockSaveDialog).not.toHaveBeenCalled();
    });

    it("continues the open dataset for the next take without asking again", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("first", 50);
      await useStore.getState().recordStop(true);
      mockSaveDialog.mockClear();

      await useStore.getState().recordStart("second", 50);

      expect(mockSaveDialog).not.toHaveBeenCalled();
      expect(lastStartReq()).toEqual({ root: ROOT, task: "second", fps: 50 });
      // the second take's episode count carries the first one
      expect(useStore.getState().liveRec).toMatchObject({
        recording: true,
        task: "second",
        episodesSaved: 1,
      });
    });

    it("counts a saved take as an episode and ends the take", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");

      await useStore.getState().recordStop(true);

      expect(mockInvoke).toHaveBeenCalledWith("live_record_stop", { save: true });
      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({
        recording: false,
        task: null,
        frames: 0,
        episodesSaved: 1,
      });
      expect(s.liveRecHint).toBeNull();
    });

    it("keeps the dataset open but uncounted when a take is discarded", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      mockInvoke.mockResolvedValueOnce({ saved: false, episodeIndex: null, frames: 40 });

      await useStore.getState().recordStop(false);

      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({ root: ROOT, recording: false, episodesSaved: 0 });
      expect(s.liveRecHint).toMatch(/discarded/);
    });

    it("unsticks the panel when saving a take that captured nothing errs", async () => {
      // the backend refuses to save an empty take — but the take IS over, so
      // the panel must leave recording state and say what happened
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      mockInvoke.mockRejectedValueOnce("no frames were captured in this take");

      await useStore.getState().recordStop(true);

      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({ root: ROOT, recording: false, episodesSaved: 0 });
      expect(s.liveRecHint).toBe("no frames were captured in this take");
      expect(s.error).toBeNull(); // nothing is broken; this is not banner-worthy

      // and the next take starts normally, on the same dataset
      await useStore.getState().recordStart("teleop");
      expect(useStore.getState().liveRec).toMatchObject({ recording: true });
    });

    it("counts the take's frames off the state stream", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");

      emit("live://state", mockState({ recording: true, recFrames: 12 }));
      _flushLive();

      expect(useStore.getState().liveRec).toMatchObject({ recording: true, frames: 12 });
    });

    it("reads a take dying with no command behind it as a discard, once", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      emit("live://state", mockState({ recording: true, recFrames: 12 }));
      _flushLive();

      // live_reset auto-discards the take; the flag flipping off IS the notice
      emit("live://state", mockState({ recording: false }));
      _flushLive();

      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({ recording: false, task: null, frames: 0 });
      expect(s.liveRecHint).toMatch(/discarded/);

      // one-shot: every later event says the same thing and must stay quiet
      useStore.setState({ liveRecHint: null });
      emit("live://state", mockState({ recording: false }));
      _flushLive();
      expect(useStore.getState().liveRecHint).toBeNull();
    });

    it("ignores a stale not-recording event that crossed the start reply", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");

      // emitted BEFORE the take began — it says nothing about the take
      emit("live://state", mockState({ recording: false }));
      _flushLive();

      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({ recording: true, task: "teleop" });
      expect(s.liveRecHint).toBeNull();
    });

    it("does not mistake a stop of our own for a discard", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      emit("live://state", mockState({ recording: true, recFrames: 5 }));
      _flushLive();

      let release: (v: unknown) => void = () => {};
      mockInvoke.mockImplementationOnce(() => new Promise((r) => (release = r)));
      const stopping = useStore.getState().recordStop(true);
      // the take stops on the backend before the reply gets back to us
      emit("live://state", mockState({ recording: false }));
      _flushLive();
      expect(useStore.getState().liveRecHint).toBeNull();

      release({ saved: true, episodeIndex: 0, frames: 5 });
      await stopping;

      const s = useStore.getState();
      expect(s.liveRec).toMatchObject({ recording: false, episodesSaved: 1 });
      expect(s.liveRecHint).toBeNull();
    });

    it("closes the dataset and offers it to Data mode", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      await useStore.getState().recordStop(true);

      await useStore.getState().recordFinish();

      expect(mockInvoke).toHaveBeenCalledWith("live_record_finish");
      const s = useStore.getState();
      expect(s.liveRec).toBeNull();
      expect(s.liveRecDone).toEqual({ root: ROOT, episodes: 1 });

      mockInvoke.mockImplementation(async (cmd: string) =>
        cmd === "dataset_open" ? { path: ROOT, episodes: [] } : undefined,
      );
      await useStore.getState().openRecordedDataset();

      expect(mockInvoke).toHaveBeenCalledWith("dataset_open", { path: ROOT });
      expect(useStore.getState().mode).toBe("data");
      expect(useStore.getState().liveRecDone).toBeNull();
    });

    it("refuses to finish mid-take (the backend would too)", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");

      await useStore.getState().recordFinish();

      expect(mockInvoke).not.toHaveBeenCalledWith("live_record_finish");
      expect(useStore.getState().liveRec).toMatchObject({ recording: true });
    });

    it("clears the slice when the session ends, keeping the dataset reachable", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");
      await useStore.getState().recordStop(true);

      // the backend finalizes the open dataset on its way out
      emit("live://ended", { sessionId: 1, reason: "stopped" });

      const s = useStore.getState();
      expect(s.liveRec).toBeNull();
      expect(s.liveRecDone).toEqual({ root: ROOT, episodes: 1 });
      expect(s.liveRecHint).toBeNull();
    });

    it("says so when the session ends mid-take", async () => {
      await useStore.getState().startLive();
      await useStore.getState().recordStart("teleop");

      emit("live://ended", { sessionId: 1, reason: "error: mujoco step diverged" });

      const s = useStore.getState();
      expect(s.liveRec).toBeNull();
      expect(s.liveRecDone).toBeNull(); // nothing was saved into it
      expect(s.liveRecHint).toMatch(/discarded/);
    });
  });
});

// ---- task artifacts (Wave C): open a *.caliper-task.json and run it ----

describe("task artifacts — store wiring", () => {
  const PREDICATE = { kind: "lifted", prop: "cube", height: 0.05, ref: "initial" };
  /** A task prop as the wire delivers it: absent size fields for the kinds that
   *  do not use them, and a contact material the frontend never interprets. */
  const CUBE: SimProp = {
    name: "cube",
    kind: "box",
    halfExtents: [0.05, 0.05, 0.05],
    pos: [0, 0, 0.05],
    mass: 0.05,
    material: "wood",
  };
  const TASK_PATH = "/tasks/lift_cube.caliper-task.json";

  function mockTaskDto(over: Partial<TaskDto> = {}): TaskDto {
    return {
      name: "lift-cube",
      robotPath: "/fx/gripper_arm.urdf",
      robot: MOCK_ROBOT,
      q0: [0.3, -0.2],
      ground: 0.02,
      props: [CUBE],
      zones: [{ name: "bin", center: [0.4, 0.2, 0.02], half: [0.05, 0.05, 0.02], rgba: null }],
      gripper: { joint: "j1", closed: "hi" },
      success: PREDICATE,
      successDescription: "cube is lifted 0.05 m above its initial height",
      horizonS: 20,
      fps: 50,
      ...over,
    };
  }

  function mockStarted(over: Partial<LiveStartedDto> = {}): LiveStartedDto {
    return {
      sessionId: 1,
      engine: "mujoco",
      h: 0.002,
      emitHz: 59.5,
      ndof: 2,
      props: [],
      gripper: null,
      ...over,
    };
  }

  function mockState(over: Partial<LiveStateEvent> = {}): LiveStateEvent {
    return {
      sessionId: 1,
      tick: 10,
      t: 0.2,
      q: [0.3, -0.2],
      qd: [0, 0],
      frames: [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
      tip: [0.3, 0, 0.2],
      ncon: 1,
      props: [[0, 0, 0.05, 1, 0, 0, 0]],
      paused: false,
      target: [0.3, -0.2],
      recording: false,
      recFrames: 0,
      ...over,
    };
  }

  /** `task_open` answers with `dto` (or rejects with `fail`); every follow-up
   *  the adopt path fires answers empty-but-valid. */
  function backend(dto: TaskDto | null, fail?: string) {
    mockInvoke.mockImplementation(async (cmd: string) => {
      switch (cmd) {
        case "task_open":
          if (!dto) throw fail ?? "task_open failed";
          return dto;
        case "robot_info":
          return MOCK_ROBOT;
        case "get_frames":
          return [];
        case "list_poses":
          return [];
        case "sim_engines":
          return ["builtin", "mujoco"];
        case "live_start":
          return mockStarted();
        default:
          return undefined;
      }
    });
  }

  function emit(channel: string, payload: unknown) {
    liveHandlers[channel]({ payload });
  }

  it("adopts the robot the backend already loaded, plus the scene and start pose", async () => {
    backend(mockTaskDto());

    await useStore.getState().openTask(TASK_PATH);

    expect(mockInvoke).toHaveBeenCalledWith("task_open", { path: TASK_PATH });
    // the robot rode along with the reply — asking for it again would be a
    // second load of the same file
    expect(mockInvoke).not.toHaveBeenCalledWith("robot_info", expect.anything());
    const s = useStore.getState();
    expect(s.robot).toBe(MOCK_ROBOT);
    expect(s.urdfPath).toBe("/fx/gripper_arm.urdf");
    expect(s.q).toEqual([0.3, -0.2]); // the task's q0, through the FK refresh
    expect(mockInvoke).toHaveBeenCalledWith("get_frames", { q: [0.3, -0.2] });
    expect(s.mode).toBe("simulate");
    expect(s.error).toBeNull();
  });

  it("pre-fills the prop editor with the task's scene, material and all", async () => {
    backend(mockTaskDto());

    await useStore.getState().openTask(TASK_PATH);

    const s = useStore.getState();
    expect(s.simProps).toEqual([CUBE]);
    // opaque passthrough: the material is carried, never interpreted
    expect(s.simProps[0].material).toBe("wood");
    // props and verdicts are contact-engine features
    expect(s.simEngine).toBe("mujoco");
  });

  it("keeps the task half of the reply as the task slice — and only that", async () => {
    const dto = mockTaskDto();
    backend(dto);

    await useStore.getState().openTask(TASK_PATH);

    // no `robot` and no `props`: the robot is adopted AS the robot, and the
    // props are the prop editor's scene — neither is stored twice
    expect(useStore.getState().task).toEqual({
      path: TASK_PATH,
      name: "lift-cube",
      robotPath: "/fx/gripper_arm.urdf",
      q0: [0.3, -0.2],
      ground: 0.02,
      zones: dto.zones,
      gripper: { joint: "j1", closed: "hi" },
      success: PREDICATE,
      successDescription: "cube is lifted 0.05 m above its initial height",
      horizonS: 20,
      fps: 50,
    });
  });

  it("defaults the record panel to the task's own label and rate", async () => {
    backend(mockTaskDto({ fps: 30 }));

    await useStore.getState().openTask(TASK_PATH);

    expect(useStore.getState().liveRecTask).toBe("lift-cube");
    expect(useStore.getState().liveRecFps).toBe(30);
  });

  it("falls back to the default rate when the task declares none", async () => {
    backend(mockTaskDto({ fps: null }));

    await useStore.getState().openTask(TASK_PATH);

    expect(useStore.getState().liveRecFps).toBe(DEFAULT_REC_FPS);
  });

  it("starts at zero when the task declares no q0", async () => {
    backend(mockTaskDto({ q0: null }));

    await useStore.getState().openTask(TASK_PATH);

    expect(useStore.getState().q).toEqual([0, 0]);
  });

  it("stays out of Simulate mode (loudly) when the task's robot has no dynamics", async () => {
    backend(mockTaskDto({ robot: { ...MOCK_ROBOT, hasInertia: false } }));

    await useStore.getState().openTask(TASK_PATH);

    const s = useStore.getState();
    expect(s.mode).toBe("jog"); // simulate's own gating disables that tab
    expect(s.error).toMatch(/no inertial data/);
    expect(s.task?.name).toBe("lift-cube");
  });

  it("picks the file in the native dialog when called with no path", async () => {
    backend(mockTaskDto());
    mockOpenDialog.mockResolvedValueOnce(TASK_PATH);

    await useStore.getState().openTask();

    expect(mockInvoke).toHaveBeenCalledWith("task_open", { path: TASK_PATH });
  });

  it("does nothing at all when that dialog is cancelled", async () => {
    backend(mockTaskDto());
    mockOpenDialog.mockResolvedValueOnce(null);

    await useStore.getState().openTask();

    expect(mockInvoke).not.toHaveBeenCalled();
    expect(useStore.getState().task).toBeNull();
  });

  it("surfaces a bad task file and re-asserts the robot this UI is showing", async () => {
    useStore.setState({ robot: MOCK_ROBOT, urdfPath: "/fx/showcase6.urdf" });
    backend(null, "task `lift-cube`: q0 has 3 values but `panda` has 2 joints");

    await useStore.getState().openTask(TASK_PATH);

    const s = useStore.getState();
    expect(s.error).toMatch(/q0 has 3 values/);
    expect(s.task).toBeNull();
    // task_open loads the robot BEFORE that check, so app state may now hold
    // the task's robot — re-assert ours so the two ends cannot disagree
    expect(mockInvoke).toHaveBeenCalledWith("robot_info", { path: "/fx/showcase6.urdf" });
  });

  it("a plain robot load drops the task (that robot is not the task's)", async () => {
    backend(mockTaskDto());
    await useStore.getState().openTask(TASK_PATH);
    expect(useStore.getState().task).not.toBeNull();

    await useStore.getState().loadRobot("/fx/showcase6.urdf");

    const s = useStore.getState();
    expect(s.task).toBeNull();
    expect(s.simProps).toEqual([]);
    expect(s.simEngine).toBe("builtin");
    expect(s.mode).toBe("jog");
  });

  it("starts a live session on the task: its ground, gripper, scene and verdict", async () => {
    backend(mockTaskDto());
    await useStore.getState().openTask(TASK_PATH);

    await useStore.getState().startLive();

    expect(mockInvoke).toHaveBeenCalledWith("live_start", {
      req: {
        q0: [0.3, -0.2],
        engine: "mujoco",
        props: [CUBE],
        ground: 0.02,
        gripperJoint: "j1",
        gripperClosed: "hi",
        success: PREDICATE,
      },
    });
  });

  it("sends none of that when no task is loaded", async () => {
    useStore.setState({
      robot: MOCK_ROBOT,
      q: [0, 0],
      mode: "simulate",
      simEngine: "mujoco",
      simEngines: ["builtin", "mujoco"],
    });
    backend(mockTaskDto());

    await useStore.getState().startLive();

    expect(mockInvoke).toHaveBeenCalledWith("live_start", {
      req: { q0: [0, 0], engine: "mujoco", props: [] },
    });
  });

  it("tracks the streamed verdict per instant, without latching it", async () => {
    backend(mockTaskDto());
    await useStore.getState().openTask(TASK_PATH);
    await useStore.getState().startLive();
    expect(useStore.getState().live?.success).toBeNull(); // nothing judged yet

    emit("live://state", mockState({ success: false }));
    _flushLive();
    expect(useStore.getState().live?.success).toBe(false);

    emit("live://state", mockState({ tick: 20, success: true }));
    _flushLive();
    expect(useStore.getState().live?.success).toBe(true);

    // the prop rolls back out of the zone: the badge follows the stream down
    emit("live://state", mockState({ tick: 30, success: false }));
    _flushLive();
    expect(useStore.getState().live?.success).toBe(false);
  });
});
