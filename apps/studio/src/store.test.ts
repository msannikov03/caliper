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

// vi.mock calls are hoisted before imports by Vitest.
import { vi, describe, it, expect, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

// Replace xyflow utilities with minimal pure implementations.
// applyNodeChanges / applyEdgeChanges are only used in the pass-through change
// handlers (onGraphNodesChange / onGraphEdgesChange) which we don't test here.
vi.mock("@xyflow/react", () => ({
  applyNodeChanges: (_changes: unknown, nodes: unknown[]) => nodes,
  applyEdgeChanges: (_changes: unknown, edges: unknown[]) => edges,
  addEdge: (edge: unknown, edges: unknown[]) => [...edges, edge],
}));

import { invoke } from "@tauri-apps/api/core";
import {
  useStore,
  handleGraphError,
  bumpNodeSeq,
  mergeRecent,
  validateSession,
  clampQ,
  sessionRestorePlan,
  _resetNodeSeq,
} from "./store";
import type { RobotInfo, TrajectoryDto, StudioState } from "./store";
import { serializeGraph } from "./graph/serialize";
import { defaultParams } from "./graph/spec";
import type { KindName } from "./graph/spec";
import type { CNode, CEdge, Diagnostics, GraphRunResult } from "./graph/types";

const mockInvoke = vi.mocked(invoke);

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
};

beforeEach(() => {
  vi.clearAllMocks();
  useStore.setState(STORE_RESET);
  _resetNodeSeq();
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
